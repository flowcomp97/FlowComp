import json
from pathlib import Path
import random
from typing import Dict, List

import torch
from transformers import Trainer, TrainerCallback
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_utils import get_last_checkpoint

from .truncation import source_token_ids


def tokenize_function(examples, model):
    source_key = "text" if "text" in examples else "input"
    # Appendix F.4: documents exceeding model_max_length are truncated
    # head+tail, keeping the first and last portions in equal proportion.
    # The policy is read from TrainingArguments so that the ICL (F.2) and RAG
    # (F.3) scenarios keep their leading-context behaviour by default.
    truncation_mode = getattr(model.training_args, "source_truncation", "head")
    head_tail_ratio = getattr(model.training_args, "source_head_tail_ratio", 0.5)
    source_tokens = {
        "input_ids": [
            source_token_ids(
                model.source_tokenizer,
                text,
                max_length=model.training_args.model_max_length,
                mode=truncation_mode,
                head_tail_ratio=head_tail_ratio,
                add_special_tokens=True,
                extra={"return_attention_mask": False},
            )["input_ids"]
            for text in examples[source_key]
        ]
    }

    output = {
        "encoder_input_ids": source_tokens["input_ids"],
    }

    has_utility = "prompt" in examples and "answer" in examples
    if has_utility:
        output["prompt_ids"] = []
        output["answer_ids"] = []

    target_eos = model.target_tokenizer.eos_token_id
    if target_eos is None:
        target_eos = model.target_pad_token_id

    # The soft prefix is assembled by the collator, because the compression
    # curriculum of Section 4.4 can change k between epochs.  Truncation is
    # budgeted against the largest prefix the model can ever emit so that a
    # pre-tokenized dataset stays valid for every epoch.
    max_prefix = model.max_num_soft_tokens

    for idx, _ in enumerate(source_tokens["input_ids"]):
        if has_utility:
            prompt_ids = model.target_tokenizer(
                examples["prompt"][idx],
                truncation=False,
                padding=False,
                return_attention_mask=False,
                add_special_tokens=False,
            )["input_ids"]
            answer_ids = model.target_tokenizer(
                examples["answer"][idx],
                truncation=False,
                padding=False,
                return_attention_mask=False,
                add_special_tokens=False,
            )["input_ids"]

            max_target_length = max(
                max_prefix + 2,
                model.training_args.model_max_length,
            )
            available = max_target_length - max_prefix - 1
            if available <= 0:
                raise ValueError(
                    "model_max_length must leave room for at least one target token "
                    "after the FlowComp soft-token prefix."
                )

            if len(prompt_ids) >= available:
                prompt_ids = prompt_ids[: max(available - 1, 0)]
                answer_ids = answer_ids[:1]
            else:
                answer_budget = available - len(prompt_ids)
                answer_ids = answer_ids[:answer_budget]

            output["prompt_ids"].append(prompt_ids)
            output["answer_ids"].append(answer_ids + [target_eos])

    return output



def trainable_state_dict(model) -> Dict[str, torch.Tensor]:
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    # The anchor itself is trainable.  Its initialized flag is the one small
    # piece of non-parameter state needed to resume Algorithm 1 faithfully.
    required_buffers = {"anchor_initialized"}
    state_dict = model.state_dict()
    return {
        name: tensor.detach().cpu()
        for name, tensor in state_dict.items()
        if name in trainable_names or name in required_buffers
    }


def save_trainable_checkpoint(model, output_dir: Path) -> None:
    state_dict = trainable_state_dict(model)
    torch.save(state_dict, output_dir / "pytorch_model.bin")
    metadata = {
        "format": "flowcomp_trainable_state_dict",
        "num_tensors": len(state_dict),
        "num_parameters": int(
            sum(tensor.numel() for name, tensor in state_dict.items() if name != "anchor_initialized")
        ),
        "excluded_prefixes": ["target_model", "source_encoder"],
        "included_buffers": ["anchor_initialized"],
    }
    (output_dir / "checkpoint_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


class FlowCompTrainer(Trainer):
    """Trainer that saves compact model weights plus full resume state.

    Trainer's standard checkpoint path still writes the optimizer, scheduler,
    RNG and TrainerState.  Only the model-saving hook is specialized so the
    frozen backbones are not duplicated in every checkpoint.
    """

    def _save(self, output_dir: str = None, state_dict=None) -> None:
        destination = Path(output_dir or self.args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        save_trainable_checkpoint(model, destination)
        torch.save(self.args, destination / TRAINING_ARGS_NAME)


class FlowCompEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int, threshold: float, metric: str, greater_is_better: bool, output_dir: str):
        self.patience = patience
        self.threshold = threshold
        self.metric = metric
        self.greater_is_better = greater_is_better
        self.best_metric = None
        self.bad_evals = 0
        self.best_checkpoint_dir = Path(output_dir) / "best_checkpoint"

    def _is_better(self, value: float) -> bool:
        if self.best_metric is None:
            return True
        if self.greater_is_better:
            return value > self.best_metric + self.threshold
        return value < self.best_metric - self.threshold

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.patience <= 0 or not metrics or self.metric not in metrics:
            return control
        value = float(metrics[self.metric])
        model = kwargs.get("model")
        if self._is_better(value):
            self.best_metric = value
            self.bad_evals = 0
            if model is not None:
                self.best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                save_trainable_checkpoint(model, self.best_checkpoint_dir)
                metadata = {
                    "metric": self.metric,
                    "best_metric": value,
                    "global_step": state.global_step,
                    "epoch": state.epoch,
                }
                (self.best_checkpoint_dir / "best_metrics.json").write_text(
                    json.dumps(metadata, indent=2),
                    encoding="utf-8",
                )
        else:
            self.bad_evals += 1
            if self.bad_evals >= self.patience:
                control.should_training_stop = True
        return control


class CompressionCurriculumCallback(TrainerCallback):
    """Applies the Section 4.4 curriculum and the Section 4.3 epsilon anneal.

    k(e) is refreshed at the start of every epoch; the Sinkhorn epsilon
    progress is refreshed every step so the anneal is smooth.
    """

    def __init__(self, model):
        self.model = model

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch or 0)
        budget = self.model.set_curriculum_epoch(epoch)
        if self.model.curriculum_enabled:
            print(f"[FlowComp] epoch {epoch}: soft-token budget k = {budget}")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.max_steps:
            self.model.set_sinkhorn_progress(state.global_step / float(state.max_steps))
        return control


class FlowCompDataCollator:
    """Pads a batch and prepends the soft-token placeholder prefix.

    The prefix length is read from the model at call time so that the
    compression-ratio curriculum of Section 4.4 takes effect without
    re-tokenizing the dataset.
    """

    def __init__(
        self,
        source_pad_token_id: int,
        target_pad_token_id: int,
        soft_token_id: int = None,
        model=None,
    ):
        self.source_pad_token_id = source_pad_token_id
        self.target_pad_token_id = target_pad_token_id
        self.model = model
        self.soft_token_id = (
            soft_token_id if soft_token_id is not None else getattr(model, "soft_token_id", None)
        )

    def current_num_soft_tokens(self) -> int:
        if self.model is None:
            raise ValueError(
                "FlowCompDataCollator needs a model reference to size the soft-token prefix."
            )
        return int(self.model.active_num_soft_tokens)

    def pad_tensor(self, values: List[List[int]], fill_value: int) -> torch.Tensor:
        max_len = max(len(v) for v in values)
        result = torch.full((len(values), max_len), fill_value, dtype=torch.long)
        for idx, value in enumerate(values):
            result[idx, : len(value)] = torch.tensor(value, dtype=torch.long)
        return result

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        batch = {
            "encoder_input_ids": self.pad_tensor(
                [example["encoder_input_ids"] for example in examples],
                self.source_pad_token_id,
            ),
        }
        batch["encoder_attention_mask"] = batch["encoder_input_ids"].ne(self.source_pad_token_id).long()

        if "prompt_ids" in examples[0]:
            soft_prefix = [self.soft_token_id] * self.current_num_soft_tokens()
            prompt_answer_ids = []
            labels = []
            for example in examples:
                prompt = list(example["prompt_ids"])
                answer = list(example["answer_ids"])
                prompt_answer_ids.append(soft_prefix + prompt + answer)
                labels.append([-100] * (len(soft_prefix) + len(prompt)) + answer)
            batch["prompt_answer_ids"] = self.pad_tensor(prompt_answer_ids, self.target_pad_token_id)
            batch["labels"] = self.pad_tensor(labels, -100)
            batch["prompt_answer_attention_mask"] = batch["prompt_answer_ids"].ne(self.target_pad_token_id).long()

        return batch


def initialize_shared_anchor(model, train_dataset, data_collator, seed: int) -> None:
    """Initialize A only after Trainer has placed the source encoder on its device."""
    if model.model_args.latent_generator_type != "flow" or bool(model.anchor_initialized.item()):
        return
    try:
        dataset_size = len(train_dataset)
    except TypeError:
        example = next(iter(train_dataset))
    else:
        if dataset_size == 0:
            raise ValueError("Cannot initialize the FlowComp anchor from an empty training dataset.")
        example = train_dataset[random.Random(seed).randrange(dataset_size)]
    batch = data_collator([example])
    model.initialize_anchor_from_batch(
        encoder_input_ids=batch["encoder_input_ids"],
        encoder_attention_mask=batch["encoder_attention_mask"],
    )


def train_model(model, train_dataset, eval_dataset, training_args, data_collator):
    last_checkpoint = None
    overwrite_output_dir = getattr(training_args, "overwrite_output_dir", False)
    if training_args.output_dir and not overwrite_output_dir:
        try:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
        except FileNotFoundError:
            last_checkpoint = None

    callbacks = []
    curriculum_enabled = bool(getattr(model, "curriculum_enabled", False))
    if curriculum_enabled and int(getattr(training_args, "dataloader_num_workers", 0) or 0) > 0:
        # The collator reads the live soft-token budget off the model, which a
        # worker subprocess would only see as a stale copy.
        print("[FlowComp] compression curriculum active: forcing dataloader_num_workers=0")
        training_args.dataloader_num_workers = 0
    if curriculum_enabled or float(getattr(training_args, "sinkhorn_epsilon_final", 0.0) or 0.0) > 0.0:
        callbacks.append(CompressionCurriculumCallback(model))
    early_stopping = None
    if eval_dataset is not None and getattr(training_args, "early_stopping_patience", 0) > 0:
        early_stopping = FlowCompEarlyStoppingCallback(
            patience=training_args.early_stopping_patience,
            threshold=training_args.early_stopping_threshold,
            metric=training_args.early_stopping_metric,
            greater_is_better=training_args.early_stopping_greater_is_better,
            output_dir=training_args.output_dir,
        )
        callbacks.append(early_stopping)

    trainer = FlowCompTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    initialize_shared_anchor(trainer.model, train_dataset, data_collator, training_args.seed)
    resume_from = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if early_stopping is not None and early_stopping.best_checkpoint_dir.exists():
        best_path = early_stopping.best_checkpoint_dir / "pytorch_model.bin"
        if best_path.exists():
            state_dict = torch.load(best_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            train_result.metrics["best_eval_metric"] = early_stopping.best_metric
            train_result.metrics["best_checkpoint"] = str(early_stopping.best_checkpoint_dir)
    save_trainable_checkpoint(model, output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    if eval_dataset is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
