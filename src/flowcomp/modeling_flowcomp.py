from dataclasses import dataclass, field
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, T5EncoderModel


# Keeps the interpolation endpoints strictly inside (0, 1) so that the
# x-prediction velocity denominator (1 - t) of Eq. (9) stays positive.
TIMESTEP_EPS = 1e-4


def freeze_model(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def mean_pool(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(sequence.dtype)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (sequence * weights).sum(dim=1) / denom


def sinusoidal_timestep_embedding(timestep: torch.Tensor, hidden_size: int) -> torch.Tensor:
    """Create the fixed sinusoidal encoding gamma(t) used by every denoiser layer.

    Implements Appendix D.2 literally:

        gamma_{2i}(t)   = sin(t / 10000^{2i/d})
        gamma_{2i+1}(t) = cos(t / 10000^{2i/d})

    so the sine/cosine pairs are *interleaved* and the exponent denominator is
    the full hidden size d (not d/2 - 1).
    """
    half_size = hidden_size // 2
    if half_size == 0:
        return timestep.new_zeros((timestep.size(0), hidden_size))
    exponents = (
        2.0
        * torch.arange(half_size, device=timestep.device, dtype=torch.float32)
        / float(hidden_size)
    )
    frequencies = torch.exp(-math.log(10_000.0) * exponents)
    angles = timestep.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    # stack -> flatten interleaves as [sin_0, cos_0, sin_1, cos_1, ...].
    embedding = torch.stack([angles.sin(), angles.cos()], dim=-1).flatten(start_dim=-2)
    if hidden_size % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding.to(dtype=timestep.dtype)


@dataclass
class ModelArguments:
    # Table 5 / Table 8: the encoder is T5-Small (d_T5 = 512) and the shared
    # latent dimension is d = 1024 for every reported experiment.
    source_encoder_name_or_path: str = field(default="google-t5/t5-small")
    target_model_name_or_path: str = field(default="meta-llama/Meta-Llama-3-8B-Instruct")
    num_soft_tokens: int = field(default=128)
    latent_dim: int = field(default=1024)
    denoiser_layers: int = field(default=6)
    decoder_layers: int = field(default=4)
    num_attention_heads: int = field(default=8)
    dropout: float = field(default=0.1)
    # Euler steps T used by Algorithm 2.  Inference / validation decoding only:
    # training draws one continuous timestep per step (Algorithm 1, line 2).
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
    noise_std: float = field(default=1.0)
    # Table 7: logit-normal timestep schedule.  The negative logit mean shifts
    # density toward noisier timesteps (t near 0).
    timestep_logit_mean: float = field(default=-1.0)
    timestep_logit_std: float = field(default=1.0)
    # Section 4.4 "Curriculum on Compression Ratio".  Disabled whenever
    # curriculum_max_ratio <= 0, which reproduces the fixed-k setting of
    # Table 8.  When enabled, r(e) = min(r0 + dr * floor(e / e_step), r_max)
    # and k(e) = max(floor(n / r(e)), k_min) with n = model_max_length.
    curriculum_initial_ratio: float = field(default=0.0)
    curriculum_ratio_increment: float = field(default=0.0)
    curriculum_epochs_per_step: int = field(default=1)
    curriculum_max_ratio: float = field(default=0.0)
    curriculum_min_soft_tokens: int = field(default=1)


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
    consistency_loss_weight: float = field(default=0.01)
    sinkhorn_epsilon: float = field(default=0.1)
    # Section 4.3 states that epsilon is annealed during training, while
    # Table 6 reports the constant value 0.1.  Setting a positive final value
    # enables the linear anneal eps(p) = eps0 + p * (eps_final - eps0) over
    # training progress p in [0, 1]; the default (<= 0) keeps eps fixed at
    # sinkhorn_epsilon, which is the configuration reported in Table 6.
    sinkhorn_epsilon_final: float = field(default=0.0)
    sinkhorn_iters: int = field(default=30)
    # Appendix F.4: documents exceeding model_max_length use head+tail
    # truncation, keeping the first and last portions in equal proportion.
    # The ICL (F.2) and RAG (F.3) scenarios keep the leading context only, so
    # "head" stays the default and long-document runs opt in explicitly.
    source_truncation: str = field(default="head")
    source_head_tail_ratio: float = field(default=0.5)
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
        time_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Appendix D.2: gamma(t) is added before self-attention in every
        # denoiser layer.  It conditions attention without repeatedly adding
        # the same vector to the residual stream.
        if time_embedding is not None:
            latent_for_attention = latent + time_embedding.unsqueeze(1)
        else:
            latent_for_attention = latent
        self_hidden = self.self_norm(latent_for_attention)
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
        # This module performs x-prediction: its output is the clean endpoint
        # S_hat_1(t), not a velocity.  The caller derives the velocity as
        # (S_hat_1(t) - S_t) / (1 - t).
        time_embed = sinusoidal_timestep_embedding(timestep, noisy_latent.size(-1)).to(
            device=noisy_latent.device,
            dtype=noisy_latent.dtype,
        )
        hidden = noisy_latent
        for layer in self.layers:
            hidden = layer(hidden, context, context_mask, time_embedding=time_embed)
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
        if model_args.distribution_loss_type not in {"wasserstein", "mmd", "cosine"}:
            raise ValueError(f"Unsupported distribution_loss_type: {model_args.distribution_loss_type}")
        if model_args.soft_token_attention not in {"bidirectional", "causal"}:
            raise ValueError(f"Unsupported soft_token_attention: {model_args.soft_token_attention}")

        # Section 4.4: the curriculum starts at the mildest ratio r0, i.e. the
        # largest soft-token budget, and shrinks to model_args.num_soft_tokens.
        # Every k-shaped parameter is therefore allocated at that upper bound.
        self.curriculum_enabled = (
            model_args.latent_generator_type == "flow"
            and model_args.curriculum_max_ratio > 0.0
            and model_args.curriculum_initial_ratio > 0.0
        )
        self.max_num_soft_tokens = model_args.num_soft_tokens
        if self.curriculum_enabled:
            self.max_num_soft_tokens = max(
                model_args.num_soft_tokens,
                self._raw_curriculum_budget(epoch=0),
            )
        # The active budget is what every forward pass actually uses.  Without
        # a curriculum it is constant and equal to model_args.num_soft_tokens.
        self.active_num_soft_tokens = (
            self.compression_curriculum_budget(epoch=0)
            if self.curriculum_enabled
            else model_args.num_soft_tokens
        )
        # Linear Sinkhorn epsilon anneal progress in [0, 1]; see Section 4.3.
        self.sinkhorn_progress = 0.0

        self.source_to_latent = nn.Linear(self.source_hidden_size, self.latent_dim, bias=False)
        self.latent_to_target = nn.Linear(self.latent_dim, self.target_hidden_size, bias=False)
        # A is one shared k x d parameter, not a context-specific table.  It is
        # overwritten once with pooled source features before training starts.
        self.anchor = nn.Parameter(torch.empty(self.max_num_soft_tokens, self.latent_dim))
        nn.init.normal_(self.anchor, mean=0.0, std=model_args.noise_std)
        self.register_buffer("anchor_initialized", torch.tensor(False), persistent=True)
        self.denoiser = FlowDenoiser(
            self.latent_dim,
            model_args.num_attention_heads,
            model_args.denoiser_layers,
            model_args.dropout,
            self_attention=model_args.soft_token_attention,
        )
        if model_args.latent_generator_type == "one_shot":
            self.soft_token_queries = nn.Parameter(
                torch.randn(self.max_num_soft_tokens, self.latent_dim) * 0.02
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
        self.reconstruction_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def train(self, mode: bool = True):
        """Set train mode while keeping frozen feature extractors deterministic."""
        super().train(mode)
        if not self.model_args.source_encoder_trainable:
            self.source_encoder.eval()
        if self.model_args.freeze_target_model:
            self.target_model.eval()
        return self

    def _raw_curriculum_budget(self, epoch: int) -> int:
        """Unclamped k(e) of Section 4.4, used to size the k-shaped parameters.

            r(e) = min(r0 + dr * floor(e / e_step), r_max)
            k(e) = max(floor(n / r(e)), k_min)

        n is the nominal source length, i.e. training_args.model_max_length.
        """
        args = self.model_args
        if args.curriculum_max_ratio <= 0.0 or args.curriculum_initial_ratio <= 0.0:
            return args.num_soft_tokens
        steps_per_stage = max(args.curriculum_epochs_per_step, 1)
        ratio = args.curriculum_initial_ratio + args.curriculum_ratio_increment * (
            max(epoch, 0) // steps_per_stage
        )
        ratio = max(min(ratio, args.curriculum_max_ratio), 1e-6)
        source_length = int(self.training_args.model_max_length)
        return max(int(source_length // ratio), args.curriculum_min_soft_tokens)

    def compression_curriculum_budget(self, epoch: int) -> int:
        """k(e) clamped to the allocated capacity and the final budget."""
        budget = self._raw_curriculum_budget(epoch)
        capacity = getattr(self, "max_num_soft_tokens", self.model_args.num_soft_tokens)
        return max(min(budget, capacity), self.model_args.num_soft_tokens)

    def set_curriculum_epoch(self, epoch: int) -> int:
        """Advance the compression-ratio curriculum to the given epoch."""
        if not self.curriculum_enabled:
            return self.active_num_soft_tokens
        self.active_num_soft_tokens = self.compression_curriculum_budget(epoch)
        return self.active_num_soft_tokens

    def set_sinkhorn_progress(self, progress: float) -> None:
        """Record training progress in [0, 1] for the Sinkhorn epsilon anneal."""
        self.sinkhorn_progress = float(min(max(progress, 0.0), 1.0))

    def current_sinkhorn_epsilon(self) -> float:
        """Entropic regularization strength, optionally annealed (Section 4.3)."""
        start = self.training_args.sinkhorn_epsilon
        final = getattr(self.training_args, "sinkhorn_epsilon_final", 0.0)
        if final is None or final <= 0.0:
            return start
        return start + (final - start) * self.sinkhorn_progress

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
        """Integrate the x-prediction ODE at inference time with Euler steps."""
        active_k = self.active_num_soft_tokens
        if noise is None:
            noise = torch.randn(
                source_hidden.size(0),
                active_k,
                self.latent_dim,
                device=source_hidden.device,
                dtype=source_hidden.dtype,
            ) * self.model_args.noise_std
        if self.model_args.latent_generator_type == "one_shot":
            latent = self.soft_token_queries[:active_k].to(
                device=source_hidden.device, dtype=source_hidden.dtype
            )
            latent = latent.unsqueeze(0).expand(source_hidden.size(0), -1, -1)
            for layer in self.one_shot_layers:
                latent = layer(latent, source_hidden, source_mask)
            return self.one_shot_norm(latent)

        latent = noise
        step_size = 1.0 / max(self.model_args.sampling_steps, 1)
        # The state begins at t=0 as Gaussian noise.  At every Euler step the
        # denoiser predicts the terminal state and Eq. (9) converts it to a
        # velocity pointing to that endpoint.
        for step in range(max(self.model_args.sampling_steps, 1)):
            timestep = torch.full(
                (source_hidden.size(0),),
                float(step) / max(self.model_args.sampling_steps, 1),
                device=source_hidden.device,
                dtype=source_hidden.dtype,
            )
            predicted_endpoint = self.denoiser(latent, timestep, source_hidden, source_mask)
            velocity = (predicted_endpoint - latent) / (1.0 - timestep).view(-1, 1, 1)
            latent = latent + step_size * velocity
        return latent

    def initialize_anchor_from_batch(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> None:
        """Initialize the shared anchor from contiguous pooled source segments.

        This is called once using a randomly selected training example.  The
        buffer makes the initialization visible in a checkpoint and prevents a
        later evaluation call from changing it.
        """
        if bool(self.anchor_initialized.item()):
            return
        source_device = next(self.source_encoder.parameters()).device
        input_ids = encoder_input_ids.to(source_device)
        attention_mask = encoder_attention_mask.to(source_device)
        source_was_training = self.source_encoder.training
        self.source_encoder.eval()
        with torch.no_grad():
            source_hidden = self.encode_source(input_ids, attention_mask)
            valid_source = source_hidden[0][attention_mask[0].bool()]
            if valid_source.numel() == 0:
                raise ValueError("Cannot initialize the FlowComp anchor from an empty source context.")
            fallback = valid_source.mean(dim=0)
            num_segments = self.max_num_soft_tokens
            boundaries = torch.div(
                torch.arange(
                    num_segments + 1,
                    device=valid_source.device,
                )
                * valid_source.size(0),
                num_segments,
                rounding_mode="floor",
            )
            pooled_segments = []
            for start, end in zip(boundaries[:-1].tolist(), boundaries[1:].tolist()):
                pooled_segments.append(
                    valid_source[start:end].mean(dim=0) if end > start else fallback
                )
            self.anchor.copy_(torch.stack(pooled_segments).to(self.anchor))
            self.anchor_initialized.fill_(True)
        if source_was_training:
            self.source_encoder.train()

    def _initialize_anchor_from_source(
        self,
        source_hidden: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> None:
        """Fallback for custom training loops that do not call the public initializer."""
        if bool(self.anchor_initialized.item()):
            return
        with torch.no_grad():
            valid_source = source_hidden[0][source_mask[0].bool()]
            if valid_source.numel() == 0:
                raise ValueError("Cannot initialize the FlowComp anchor from an empty source context.")
            fallback = valid_source.mean(dim=0)
            num_segments = self.max_num_soft_tokens
            boundaries = torch.div(
                torch.arange(
                    num_segments + 1,
                    device=valid_source.device,
                )
                * valid_source.size(0),
                num_segments,
                rounding_mode="floor",
            )
            segments = [
                valid_source[start:end].mean(dim=0) if end > start else fallback
                for start, end in zip(boundaries[:-1].tolist(), boundaries[1:].tolist())
            ]
            self.anchor.copy_(torch.stack(segments).to(self.anchor))
            self.anchor_initialized.fill_(True)

    def _sample_logit_normal_times(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Draw continuous timesteps from Eq. (13): t = sigmoid(t'), t' ~ N(P_mean, P_std^2).

        Algorithm 1 (line 2) samples one continuous t per gradient step; the
        Euler grid of Algorithm 2 is an inference-time construct and is
        deliberately not imposed here.
        """
        normal = torch.randn(batch_size, device=device, dtype=torch.float32)
        times = torch.sigmoid(
            self.model_args.timestep_logit_mean + self.model_args.timestep_logit_std * normal
        )
        return times.clamp_(min=TIMESTEP_EPS, max=1.0 - 2.0 * TIMESTEP_EPS)

    def sample_training_states(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct two points on the same noise-to-anchor path for Eq. (19)."""
        timestep = self._sample_logit_normal_times(batch_size, device)
        later_timestep = self._sample_logit_normal_times(batch_size, device)
        # Algorithm 1 (line 14) requires t'' > t under the same schedule, so
        # invalid draws are rejected and redrawn from Eq. (13).
        for _ in range(16):
            invalid = later_timestep <= timestep
            if not bool(invalid.any()):
                break
            later_timestep = torch.where(
                invalid,
                self._sample_logit_normal_times(batch_size, device),
                later_timestep,
            )
        # Any unresolved draw is projected to the midpoint of (t, 1), which is
        # strictly greater than t and strictly below 1.
        later_timestep = torch.where(
            later_timestep <= timestep,
            0.5 * (timestep + 1.0),
            later_timestep,
        ).clamp_(max=1.0 - TIMESTEP_EPS)

        timestep = timestep.to(dtype=dtype)
        later_timestep = later_timestep.to(dtype=dtype)
        active_k = self.active_num_soft_tokens
        anchor = self.anchor[:active_k].to(device=device, dtype=dtype).unsqueeze(0)
        noise = torch.randn(
            batch_size,
            active_k,
            self.latent_dim,
            device=device,
            dtype=dtype,
        ) * self.model_args.noise_std
        # Endpoint consistency compares two noise levels on the *same*
        # interpolation path: S_t and S_t'' share epsilon and A, with t'' > t.
        # Reusing `noise` is essential; independent noise would regularize
        # invariance across distinct paths rather than Eq. (19)'s trajectory
        # consistency.
        state = timestep.view(-1, 1, 1) * anchor + (1.0 - timestep).view(-1, 1, 1) * noise
        later_state = (
            later_timestep.view(-1, 1, 1) * anchor
            + (1.0 - later_timestep).view(-1, 1, 1) * noise
        )
        return state, timestep, later_state, later_timestep

    def compute_reconstruction_loss(
        self,
        predicted_endpoint: torch.Tensor,
        encoder_input_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Token-level reconstruction CE from Eq. (17), with tied frozen T5 embeddings."""
        reconstructed_latent = self.reconstruction_decoder(
            predicted_endpoint,
            output_length=encoder_input_ids.size(1),
        )
        # source_to_latent maps source features to the latent space.  Applying
        # its transpose maps decoder states back before scoring with E_T5^T.
        reconstructed_source = F.linear(
            reconstructed_latent,
            self.source_to_latent.weight.transpose(0, 1),
        )
        source_embedding = self.source_encoder.get_input_embeddings().weight
        vocabulary_logits = F.linear(
            reconstructed_source.to(source_embedding.dtype),
            source_embedding,
        )
        labels = encoder_input_ids.masked_fill(~source_attention_mask.bool(), -100)
        return self.reconstruction_loss(
            vocabulary_logits.reshape(-1, vocabulary_logits.size(-1)),
            labels.reshape(-1),
        )

    @staticmethod
    def compute_consistency_loss(
        predicted_endpoint: torch.Tensor,
        later_predicted_endpoint: torch.Tensor,
    ) -> torch.Tensor:
        """Eq. (19): squared Frobenius endpoint gap, averaged over the batch.

        The reduction preserves the paper's per-example squared Frobenius norm;
        it is not a per-element mean squared error.
        """
        difference = (predicted_endpoint - later_predicted_endpoint).flatten(start_dim=1)
        return difference.float().square().sum(dim=-1).mean().to(predicted_endpoint.dtype)

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
            expected_slots = target_soft_tokens.size(0) * target_soft_tokens.size(1)
            actual_slots = int(placeholder_mask.sum().detach().cpu().item())
            if actual_slots != expected_slots:
                raise ValueError(
                    f"The batch carries {actual_slots} soft-token placeholders but the model "
                    f"produced {expected_slots} soft tokens (k = {self.active_num_soft_tokens}). "
                    "When the Section 4.4 curriculum is enabled the collator must observe the "
                    "live model, so run with dataloader_num_workers=0."
                )
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
            (prompt_ids.size(0), soft_tokens.size(1)),
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
                    (prompt_attention_mask.size(0), soft_tokens.size(1)),
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
        """Entropic OT with the cost-rescaling convention of Appendix E.2.

        Appendix E.2: the raw squared-l2 cost can make the Gibbs kernel
        exp(-M / eps) underflow, so the cost matrix is divided by kappa = d
        "so that the per-coordinate average squared distance enters the kernel
        rather than the total squared norm", and eps = 0.1 (Table 6 / 8) is
        calibrated under this convention for d = 1024.

        The rescaling therefore lives *inside the Gibbs kernel only*.  The
        Wasserstein Anchoring loss itself is Eq. 24,

            L_WA = sum_ij Pi*_ij ||c_i - s_j||_2^2,

        i.e. the transport plan is contracted against the *unscaled* squared
        l2 cost; it is not divided by d.
        """
        losses = []
        iterations = self.training_args.sinkhorn_iters
        epsilon = max(self.current_sinkhorn_epsilon(), 1e-6)
        for src, mask, tgt in zip(source_hidden, source_mask, soft_tokens):
            src = src[mask.bool()]
            if src.numel() == 0 or tgt.numel() == 0:
                continue
            # Eq. 24 cost: raw squared l2 between source features and soft tokens.
            cost = torch.cdist(src.float(), tgt.float(), p=2).pow(2)
            kappa = float(max(src.size(-1), 1))  # kappa = d (Appendix E.2)

            a = torch.full((src.size(0),), 1.0 / src.size(0), device=src.device)
            b = torch.full((tgt.size(0),), 1.0 / tgt.size(0), device=tgt.device)
            kernel = torch.exp(-(cost / kappa) / epsilon).clamp_min(1e-8)
            u = torch.ones_like(a)
            v = torch.ones_like(b)
            for _ in range(iterations):
                u = a / (kernel @ v).clamp_min(1e-8)
                v = b / (kernel.transpose(0, 1) @ u).clamp_min(1e-8)
            transport = u.unsqueeze(1) * kernel * v.unsqueeze(0)
            losses.append((transport * cost).sum())
        if not losses:
            return soft_tokens.sum() * 0.0
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

    def cosine_centroid_distance(
        self,
        source_hidden: torch.Tensor,
        source_mask: torch.Tensor,
        soft_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine-centroid alignment ablation of Section 6.3 / Figure 3.

        Each source token is assigned to its most similar soft token, which
        acts as a centroid, and the loss is the mean cosine distance to that
        centroid.  Unlike Wasserstein Anchoring there is no marginal
        constraint, so nothing prevents a subset of soft tokens from
        capturing every source token.
        """
        losses = []
        for src, mask, tgt in zip(source_hidden, source_mask, soft_tokens):
            src = src[mask.bool()].float()
            if src.numel() == 0:
                continue
            normalized_src = F.normalize(src, dim=-1, eps=1e-8)
            normalized_tgt = F.normalize(tgt.float(), dim=-1, eps=1e-8)
            similarity = normalized_src @ normalized_tgt.transpose(0, 1)
            losses.append((1.0 - similarity.max(dim=-1).values).mean())
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
        if self.model_args.distribution_loss_type == "cosine":
            return self.cosine_centroid_distance(source_hidden, source_mask, soft_tokens)
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
        later_predicted_endpoint = None
        if self.model_args.latent_generator_type == "one_shot":
            # Retain the explicit one-shot ablation as a non-flow baseline.
            predicted_endpoint = self.sample_soft_tokens(source_hidden, encoder_attention_mask)
        else:
            # Training follows Algorithm 1: a single randomly-timed noisy
            # state is mapped directly to an endpoint.  It never unrolls the
            # Euler sampler.
            self._initialize_anchor_from_source(source_hidden, encoder_attention_mask)
            noisy_state, timestep, later_noisy_state, later_timestep = self.sample_training_states(
                batch_size=source_hidden.size(0),
                device=source_hidden.device,
                dtype=source_hidden.dtype,
            )
            predicted_endpoint = self.denoiser(
                noisy_state,
                timestep,
                source_hidden,
                encoder_attention_mask,
            )
            if self.training_args.consistency_loss_weight != 0:
                later_predicted_endpoint = self.denoiser(
                    later_noisy_state,
                    later_timestep,
                    source_hidden,
                    encoder_attention_mask,
                )

        reconstruction_loss = predicted_endpoint.sum() * 0.0
        if self.training_args.reconstruction_loss_weight != 0:
            reconstruction_loss = self.compute_reconstruction_loss(
                predicted_endpoint,
                encoder_input_ids,
                encoder_attention_mask,
            )
        distribution_loss = predicted_endpoint.sum() * 0.0
        if self.training_args.wasserstein_loss_weight != 0:
            distribution_loss = self.compute_distribution_loss(
                source_hidden,
                encoder_attention_mask,
                predicted_endpoint,
            )

        utility_loss = torch.zeros_like(reconstruction_loss)
        if prompt_answer_ids is not None and labels is not None and prompt_answer_attention_mask is not None:
            utility_loss = self.compute_utility_loss(
                predicted_endpoint,
                prompt_answer_ids,
                prompt_answer_attention_mask,
                labels,
            )

        consistency_loss = predicted_endpoint.sum() * 0.0
        if later_predicted_endpoint is not None:
            consistency_loss = self.compute_consistency_loss(
                predicted_endpoint,
                later_predicted_endpoint,
            )

        total_loss = (
            self.training_args.reconstruction_loss_weight * reconstruction_loss
            + self.training_args.utility_loss_weight * utility_loss
            + self.training_args.wasserstein_loss_weight * distribution_loss
            + self.training_args.consistency_loss_weight * consistency_loss
        )

        pooled_source = mean_pool(source_hidden, encoder_attention_mask)
        pooled_endpoint = predicted_endpoint.mean(dim=1)

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss.detach(),
            "utility_loss": utility_loss.detach(),
            "wasserstein_loss": distribution_loss.detach(),
            "distribution_loss": distribution_loss.detach(),
            "consistency_loss": consistency_loss.detach(),
            "source_norm": pooled_source.norm(dim=-1).mean().detach(),
            "soft_norm": pooled_endpoint.norm(dim=-1).mean().detach(),
            "anchor_norm": self.anchor.norm(dim=-1).mean().detach(),
        }
