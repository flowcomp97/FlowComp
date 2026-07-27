from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, T5EncoderModel


def freeze_model(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def mean_pool(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(sequence.dtype)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (sequence * weights).sum(dim=1) / denom


@dataclass
class ModelArguments:
    source_encoder_name_or_path: str = field(default="google-t5/t5-base")
    target_model_name_or_path: str = field(default="meta-llama/Meta-Llama-3-8B-Instruct")
    num_soft_tokens: int = field(default=128)
    latent_dim: int = field(default=0)
    denoiser_layers: int = field(default=6)
    decoder_layers: int = field(default=4)
    num_attention_heads: int = field(default=8)
    dropout: float = field(default=0.1)
    sampling_steps: int = field(default=8)
    source_encoder_trainable: bool = field(default=False)
    freeze_target_model: bool = field(default=True)
    use_flash_attention_2: bool = field(default=True)
    max_decoder_positions: int = field(default=8192)
    soft_token_literal: str = field(default="<flowcomp_soft>")
    reconstruction_decoder_type: str = field(default="transformer")
    target_model_device_map: str = field(default="none")
    latent_generator_type: str = field(default="flow")
    distribution_loss_type: str = field(default="wasserstein")
    soft_token_attention: str = field(default="bidirectional")


@dataclass
class DataArguments:
    train_file: str = field(default="")
    eval_file: str = field(default="")
    streaming: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(default=4096)
    reconstruction_loss_weight: float = field(default=1.0)
    utility_loss_weight: float = field(default=1.0)
    wasserstein_loss_weight: float = field(default=0.1)
    sinkhorn_epsilon: float = field(default=0.1)
    sinkhorn_iters: int = field(default=30)
    early_stopping_patience: int = field(default=0)
    early_stopping_threshold: float = field(default=0.0)
    early_stopping_metric: str = field(default="eval_loss")
    early_stopping_greater_is_better: bool = field(default=False)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        self_attention: str = "bidirectional",
    ) -> None:
        super().__init__()
        if self_attention not in {"bidirectional", "causal"}:
            raise ValueError(f"Unsupported self_attention: {self_attention}")
        self.self_attention = self_attention
        self.self_norm = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, dropout)

    def forward(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self_hidden = self.self_norm(latent)
        attn_mask = None
        if self.self_attention == "causal":
            seq_len = latent.size(1)
            attn_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=latent.device),
                diagonal=1,
            )
        self_out, _ = self.self_attn(
            self_hidden,
            self_hidden,
            self_hidden,
            attn_mask=attn_mask,
            need_weights=False,
        )
        latent = latent + self_out

        key_padding_mask = None
        if context_mask is not None:
            key_padding_mask = ~context_mask.bool()

        cross_hidden = self.cross_norm(latent)
        cross_out, _ = self.cross_attn(
            cross_hidden,
            context,
            context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        latent = latent + cross_out
        latent = latent + self.ffn(self.ffn_norm(latent))
        return latent


class FlowDenoiser(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        self_attention: str = "bidirectional",
    ) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.layers = nn.ModuleList(
            CrossAttentionBlock(hidden_size, num_heads, dropout, self_attention=self_attention)
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        time_embed = self.time_mlp(timestep[:, None]).unsqueeze(1)
        hidden = noisy_latent + time_embed
        for layer in self.layers:
            hidden = layer(hidden, context, context_mask)
        return self.out_proj(self.final_norm(hidden))


class LightweightReconstructionDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        max_positions: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.position_embedding = nn.Embedding(max_positions, hidden_size)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(num_layers))
        self.ffns = nn.ModuleList(FeedForward(hidden_size, dropout) for _ in range(num_layers))
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        memory: torch.Tensor,
        target_length: Optional[int] = None,
        output_length: Optional[int] = None,
    ) -> torch.Tensor:
        seq_len = target_length if target_length is not None else output_length
        if seq_len is None:
            raise ValueError("target_length or output_length must be provided")
        positions = torch.arange(
            seq_len,
            device=memory.device,
            dtype=torch.long,
        ).unsqueeze(0)
        hidden = self.position_embedding(positions).expand(memory.size(0), -1, -1)
        memory_hidden = self.memory_norm(memory)
        cross_out, _ = self.cross_attn(
            self.query_norm(hidden),
            memory_hidden,
            memory_hidden,
            need_weights=False,
        )
        hidden = hidden + cross_out
        for norm, ffn in zip(self.ffn_norms, self.ffns):
            hidden = hidden + ffn(norm(hidden))
        return self.final_norm(hidden)


class TransformerReconstructionDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        max_positions: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.position_embedding = nn.Embedding(max_positions, hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        memory: torch.Tensor,
        target_length: Optional[int] = None,
        output_length: Optional[int] = None,
    ) -> torch.Tensor:
        seq_len = target_length if target_length is not None else output_length
        if seq_len is None:
            raise ValueError("target_length or output_length must be provided")
        positions = torch.arange(
            seq_len,
            device=memory.device,
            dtype=torch.long,
        ).unsqueeze(0)
        target = self.position_embedding(positions).expand(memory.size(0), -1, -1)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=memory.device),
            diagonal=1,
        )
        hidden = self.decoder(
            tgt=target,
            memory=memory,
            tgt_mask=causal_mask,
        )
        return self.final_norm(hidden)


def build_reconstruction_decoder(
    decoder_type: str,
    hidden_size: int,
    num_heads: int,
    num_layers: int,
    max_positions: int,
    dropout: float,
) -> nn.Module:
    if decoder_type == "lightweight":
        return LightweightReconstructionDecoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            max_positions=max_positions,
            dropout=dropout,
        )
    if decoder_type == "transformer":
        return TransformerReconstructionDecoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            max_positions=max_positions,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported reconstruction_decoder_type: {decoder_type}")


class FlowComp(nn.Module):
    def __init__(self, model_args: ModelArguments, training_args: TrainingArguments) -> None:
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16

        self.source_encoder = T5EncoderModel.from_pretrained(
            model_args.source_encoder_name_or_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            cache_dir=training_args.cache_dir,
        )
        if not model_args.source_encoder_trainable:
            freeze_model(self.source_encoder)
            self.source_encoder.eval()

        target_model_load_kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "cache_dir": training_args.cache_dir,
        }
        if model_args.use_flash_attention_2:
            target_model_load_kwargs["attn_implementation"] = "flash_attention_2"
        if model_args.freeze_target_model and model_args.target_model_device_map == "auto":
            target_model_load_kwargs["device_map"] = "auto"

        try:
            self.target_model = AutoModelForCausalLM.from_pretrained(
                model_args.target_model_name_or_path,
                **target_model_load_kwargs,
            )
        except (TypeError, ImportError):
            # Older/backported model classes may reject attn_implementation,
            # or FlashAttention2 may be unavailable in the local environment.
            fallback_load_kwargs = dict(target_model_load_kwargs)
            fallback_load_kwargs.pop("attn_implementation", None)
            self.target_model = AutoModelForCausalLM.from_pretrained(
                model_args.target_model_name_or_path,
                **fallback_load_kwargs,
            )
        if model_args.freeze_target_model:
            freeze_model(self.target_model)
            self.target_model.eval()

        self.source_tokenizer = AutoTokenizer.from_pretrained(
            model_args.source_encoder_name_or_path,
            use_fast=True,
            cache_dir=training_args.cache_dir,
        )
        self.target_tokenizer = AutoTokenizer.from_pretrained(
            model_args.target_model_name_or_path,
            use_fast=False,
            cache_dir=training_args.cache_dir,
        )
        if self.source_tokenizer.pad_token_id is None:
            self.source_tokenizer.pad_token = self.source_tokenizer.eos_token
        if self.target_tokenizer.pad_token_id is None:
            self.target_tokenizer.pad_token = self.target_tokenizer.eos_token

        added_tokens = self.target_tokenizer.add_special_tokens(
            {"additional_special_tokens": [model_args.soft_token_literal]}
        )
        if added_tokens > 0:
            self.target_model.resize_token_embeddings(len(self.target_tokenizer), mean_resizing=False)
            if model_args.freeze_target_model:
                freeze_model(self.target_model)
                self.target_model.eval()
        self.soft_token_id = self.target_tokenizer.convert_tokens_to_ids(model_args.soft_token_literal)
        self.source_pad_token_id = self.source_tokenizer.pad_token_id
        self.target_pad_token_id = self.target_tokenizer.pad_token_id

        self.source_hidden_size = self.source_encoder.config.d_model
        self.target_hidden_size = self.target_model.config.hidden_size
        self.latent_dim = model_args.latent_dim or self.target_hidden_size
        if model_args.latent_generator_type not in {"flow", "one_shot"}:
            raise ValueError(f"Unsupported latent_generator_type: {model_args.latent_generator_type}")
        if model_args.distribution_loss_type not in {"wasserstein", "mmd"}:
            raise ValueError(f"Unsupported distribution_loss_type: {model_args.distribution_loss_type}")
        if model_args.soft_token_attention not in {"bidirectional", "causal"}:
            raise ValueError(f"Unsupported soft_token_attention: {model_args.soft_token_attention}")
        self.source_to_latent = nn.Linear(self.source_hidden_size, self.latent_dim, bias=False)
        self.latent_to_target = nn.Linear(self.latent_dim, self.target_hidden_size, bias=False)
        self.denoiser = FlowDenoiser(
            self.latent_dim,
            model_args.num_attention_heads,
            model_args.denoiser_layers,
            model_args.dropout,
            self_attention=model_args.soft_token_attention,
        )
        if model_args.latent_generator_type == "one_shot":
            self.soft_token_queries = nn.Parameter(
                torch.randn(model_args.num_soft_tokens, self.latent_dim) * 0.02
            )
            self.one_shot_layers = nn.ModuleList(
                CrossAttentionBlock(
                    self.latent_dim,
                    model_args.num_attention_heads,
                    model_args.dropout,
                    self_attention=model_args.soft_token_attention,
                )
                for _ in range(model_args.denoiser_layers)
            )
            self.one_shot_norm = nn.LayerNorm(self.latent_dim)
        self.reconstruction_decoder = build_reconstruction_decoder(
            decoder_type=model_args.reconstruction_decoder_type,
            hidden_size=self.latent_dim,
            num_heads=model_args.num_attention_heads,
            num_layers=model_args.decoder_layers,
            max_positions=model_args.max_decoder_positions,
            dropout=model_args.dropout,
        )
        self.utility_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def encode_source(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.source_encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        return self.source_to_latent(outputs.last_hidden_state.to(self.source_to_latent.weight.dtype))

    def sample_soft_tokens(
        self,
        source_hidden: torch.Tensor,
        source_mask: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn(
                source_hidden.size(0),
                self.model_args.num_soft_tokens,
                self.latent_dim,
                device=source_hidden.device,
                dtype=source_hidden.dtype,
            )
        if self.model_args.latent_generator_type == "one_shot":
            latent = self.soft_token_queries.to(device=source_hidden.device, dtype=source_hidden.dtype)
            latent = latent.unsqueeze(0).expand(source_hidden.size(0), -1, -1)
            for layer in self.one_shot_layers:
                latent = layer(latent, source_hidden, source_mask)
            return self.one_shot_norm(latent)
        latent = noise
        step_size = 1.0 / max(self.model_args.sampling_steps, 1)
        for step in range(self.model_args.sampling_steps, 0, -1):
            timestep = torch.full(
                (source_hidden.size(0),),
                float(step) / self.model_args.sampling_steps,
                device=source_hidden.device,
                dtype=source_hidden.dtype,
            )
            velocity = self.denoiser(latent, timestep, source_hidden, source_mask)
            latent = latent - step_size * velocity
        return latent

    def compute_reconstruction_loss(
        self,
        soft_tokens: torch.Tensor,
        source_hidden: torch.Tensor,
        source_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        reconstructed_hidden = self.reconstruction_decoder(
            soft_tokens,
            output_length=source_hidden.size(1),
        )
        squared_error = (reconstructed_hidden - source_hidden).pow(2).mean(dim=-1)
        weights = source_attention_mask.to(squared_error.dtype)
        return (squared_error * weights).sum() / weights.sum().clamp_min(1.0)

    def compute_utility_loss(
        self,
        soft_tokens: torch.Tensor,
        prompt_answer_ids: torch.Tensor,
        prompt_answer_attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.target_model.get_input_embeddings()
        vocab_size = embedding.num_embeddings
        max_input_id = int(prompt_answer_ids.max().detach().cpu().item())
        if max_input_id >= vocab_size:
            raise ValueError(
                f"prompt_answer_ids contain token id {max_input_id}, "
                f"but target embedding vocab size is {vocab_size}."
            )
        valid_labels = labels[labels.ne(-100)]
        if valid_labels.numel() > 0:
            max_label_id = int(valid_labels.max().detach().cpu().item())
            min_label_id = int(valid_labels.min().detach().cpu().item())
            if min_label_id < 0 or max_label_id >= vocab_size:
                raise ValueError(
                    f"labels contain token ids outside target vocab: "
                    f"min={min_label_id}, max={max_label_id}, vocab_size={vocab_size}."
                )

        target_soft_tokens = self.latent_to_target(soft_tokens)
        inputs_embeds = embedding(prompt_answer_ids)
        target_soft_tokens = target_soft_tokens.to(inputs_embeds.dtype)
        placeholder_mask = prompt_answer_ids.eq(self.soft_token_id)
        if placeholder_mask.any():
            inputs_embeds[placeholder_mask] = target_soft_tokens.reshape(-1, target_soft_tokens.size(-1))
        outputs = self.target_model(
            inputs_embeds=inputs_embeds,
            attention_mask=prompt_answer_attention_mask,
            return_dict=True,
        )
        logits = outputs.logits
        shifted_logits = logits[:, :-1].contiguous().reshape(-1, logits.size(-1))
        shifted_labels = labels[:, 1:].contiguous().reshape(-1)
        return self.utility_loss(shifted_logits, shifted_labels)

    def build_prompt_inputs(
        self,
        soft_tokens: torch.Tensor,
        prompt_ids: torch.Tensor,
    ) -> torch.Tensor:
        target_soft_tokens = self.latent_to_target(soft_tokens)
        soft_prefix = torch.full(
            (prompt_ids.size(0), self.model_args.num_soft_tokens),
            self.soft_token_id,
            dtype=prompt_ids.dtype,
            device=prompt_ids.device,
        )
        prompt_answer_ids = torch.cat([soft_prefix, prompt_ids], dim=1)
        inputs_embeds = self.target_model.get_input_embeddings()(prompt_answer_ids)
        target_soft_tokens = target_soft_tokens.to(inputs_embeds.dtype)
        placeholder_mask = prompt_answer_ids.eq(self.soft_token_id)
        inputs_embeds[placeholder_mask] = target_soft_tokens.reshape(-1, target_soft_tokens.size(-1))
        return inputs_embeds

    @torch.no_grad()
    def compress(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        source_hidden = self.encode_source(encoder_input_ids, encoder_attention_mask)
        return self.sample_soft_tokens(source_hidden, encoder_attention_mask)

    @torch.no_grad()
    def generate(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        stop_strings: Optional[tuple[str, ...]] = None,
    ) -> torch.Tensor:
        soft_tokens = self.compress(encoder_input_ids, encoder_attention_mask)
        generated_ids = []
        inputs_embeds = self.build_prompt_inputs(soft_tokens, prompt_ids)
        attention_mask = torch.cat(
            [
                torch.ones(
                    (prompt_attention_mask.size(0), self.model_args.num_soft_tokens),
                    dtype=prompt_attention_mask.dtype,
                    device=prompt_attention_mask.device,
                ),
                prompt_attention_mask,
            ],
            dim=1,
        )
        past_key_values = None

        stop_strings = stop_strings or tuple()

        for _ in range(max_new_tokens):
            outputs = self.target_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            # Greedy decoding keeps QA answers deterministic and avoids sampling drift.
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids.append(next_token)

            should_stop = torch.all(next_token.eq(self.target_tokenizer.eos_token_id))
            if not should_stop and stop_strings:
                decoded_pieces = [
                    self.target_tokenizer.decode(ids.tolist(), clean_up_tokenization_spaces=False)
                    for ids in next_token
                ]
                should_stop = all(any(stop_text in piece for stop_text in stop_strings) for piece in decoded_pieces)
            if should_stop:
                break

            past_key_values = outputs.past_key_values
            inputs_embeds = self.target_model.get_input_embeddings()(next_token)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.size(0), 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )

        if not generated_ids:
            return torch.empty(
                (prompt_ids.size(0), 0),
                dtype=prompt_ids.dtype,
                device=prompt_ids.device,
            )
        return torch.cat(generated_ids, dim=1)

    def sinkhorn_distance(
        self,
        source_hidden: torch.Tensor,
        source_mask: torch.Tensor,
        soft_tokens: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        epsilon = self.training_args.sinkhorn_epsilon
        iterations = self.training_args.sinkhorn_iters
        for src, mask, tgt in zip(source_hidden, source_mask, soft_tokens):
            src = src[mask.bool()]
            # Normalize squared L2 cost by feature dimension so OT stays on a
            # comparable scale across latent sizes.
            cost = torch.cdist(src.float(), tgt.float(), p=2).pow(2) / src.size(-1)
            a = torch.full((src.size(0),), 1.0 / src.size(0), device=src.device)
            b = torch.full((tgt.size(0),), 1.0 / tgt.size(0), device=tgt.device)
            kernel = torch.exp(-cost / epsilon).clamp_min(1e-8)
            u = torch.ones_like(a)
            v = torch.ones_like(b)
            for _ in range(iterations):
                u = a / (kernel @ v).clamp_min(1e-8)
                v = b / (kernel.transpose(0, 1) @ u).clamp_min(1e-8)
            transport = torch.diag(u) @ kernel @ torch.diag(v)
            losses.append((transport * cost).sum())
        return torch.stack(losses).mean()

    def mmd_distance(
        self,
        source_hidden: torch.Tensor,
        source_mask: torch.Tensor,
        soft_tokens: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        for src, mask, tgt in zip(source_hidden, source_mask, soft_tokens):
            src = src[mask.bool()].float()
            tgt = tgt.float()
            if src.numel() == 0:
                continue
            combined = torch.cat([src, tgt], dim=0)
            pairwise_sq = torch.cdist(combined, combined, p=2).pow(2) / combined.size(-1)
            positive = pairwise_sq[pairwise_sq.gt(0)]
            if positive.numel():
                bandwidth = positive.median().detach().clamp_min(1e-4)
            else:
                bandwidth = torch.tensor(1.0, device=source_hidden.device)

            def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                dist_sq = torch.cdist(x, y, p=2).pow(2) / x.size(-1)
                return torch.exp(-dist_sq / bandwidth)

            losses.append(kernel(src, src).mean() + kernel(tgt, tgt).mean() - 2 * kernel(src, tgt).mean())
        if not losses:
            return soft_tokens.sum() * 0.0
        return torch.stack(losses).mean()

    def compute_distribution_loss(
        self,
        source_hidden: torch.Tensor,
        source_mask: torch.Tensor,
        soft_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.model_args.distribution_loss_type == "mmd":
            return self.mmd_distance(source_hidden, source_mask, soft_tokens)
        return self.sinkhorn_distance(source_hidden, source_mask, soft_tokens)

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        prompt_answer_ids: Optional[torch.Tensor] = None,
        prompt_answer_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        source_hidden = self.encode_source(encoder_input_ids, encoder_attention_mask)
        soft_tokens = self.sample_soft_tokens(source_hidden, encoder_attention_mask)

        reconstruction_loss = soft_tokens.sum() * 0.0
        if self.training_args.reconstruction_loss_weight != 0:
            reconstruction_loss = self.compute_reconstruction_loss(
                soft_tokens,
                source_hidden,
                encoder_attention_mask,
            )
        distribution_loss = soft_tokens.sum() * 0.0
        if self.training_args.wasserstein_loss_weight != 0:
            distribution_loss = self.compute_distribution_loss(source_hidden, encoder_attention_mask, soft_tokens)

        utility_loss = torch.zeros_like(reconstruction_loss)
        if prompt_answer_ids is not None and labels is not None and prompt_answer_attention_mask is not None:
            utility_loss = self.compute_utility_loss(
                soft_tokens,
                prompt_answer_ids,
                prompt_answer_attention_mask,
                labels,
            )

        total_loss = (
            self.training_args.reconstruction_loss_weight * reconstruction_loss
            + self.training_args.utility_loss_weight * utility_loss
            + self.training_args.wasserstein_loss_weight * distribution_loss
        )

        pooled_source = mean_pool(source_hidden, encoder_attention_mask)
        pooled_soft = soft_tokens.mean(dim=1)

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss.detach(),
            "utility_loss": utility_loss.detach(),
            "wasserstein_loss": distribution_loss.detach(),
            "distribution_loss": distribution_loss.detach(),
            "source_norm": pooled_source.norm(dim=-1).mean().detach(),
            "soft_norm": pooled_soft.norm(dim=-1).mean().detach(),
        }
