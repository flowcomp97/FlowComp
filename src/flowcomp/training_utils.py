import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint


def tokenize_function(examples, model):
    source_key = "text" if "text" in examples else "input"
    source_tokens = model.source_tokenizer(
        examples[source_key],
        truncation=True,
        max_length=model.training_args.model_max_length,
        padding=False,
        return_attention_mask=False,
        add_special_tokens=False,
    )

    output = {
        "encoder_input_ids": source_tokens["input_ids"],
    }

    has_utility = "prompt" in examples and "answer" in examples
    if has_utility:
        output["prompt_answer_ids"] = []
        output["labels"] = []

    target_eos = model.target_tokenizer.eos_token_id
    if target_eos is None:
        target_eos = model.target_pad_token_id

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
            soft_prefix = [model.soft_token_id] * model.model_args.num_soft_tokens

            max_target_length = max(
                model.model_args.num_soft_tokens + 2,
                model.training_args.model_max_length,
            )
            available = max_target_length - len(soft_prefix) - 1
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

            prompt_answer_ids = soft_prefix + prompt_ids + answer_ids + [target_eos]
            labels = (
                [-100] * (len(soft_prefix) + len(prompt_ids))
                + answer_ids
                + [target_eos]
            )
            output["prompt_answer_ids"].append(prompt_answer_ids)
            output["labels"].append(labels)

    return output



def trainable_state_dict(model) -> Dict[str, torch.Tensor]:
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    state_dict = model.state_dict()
    return {
        name: tensor.detach().cpu()
        for name, tensor in state_dict.items()
        if name in trainable_names
    }


def save_trainable_checkpoint(model, output_dir: Path) -> None:
    state_dict = trainable_state_dict(model)
    torch.save(state_dict, output_dir / "pytorch_model.bin")
    metadata = {
        "format": "flowcomp_trainable_state_dict",
        "num_tensors": len(state_dict),
        "num_parameters": int(sum(tensor.numel() for tensor in state_dict.values())),
        "excluded_prefixes": ["target_model", "source_encoder"],
    }
    (output_dir / "checkpoint_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

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


class FlowCompDataCollator:
    def __init__(self, source_pad_token_id: int, target_pad_token_id: int):
        self.source_pad_token_id = source_pad_token_id
        self.target_pad_token_id = target_pad_token_id

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

        if "prompt_answer_ids" in examples[0]:
            batch["prompt_answer_ids"] = self.pad_tensor(
                [example["prompt_answer_ids"] for example in examples],
                self.target_pad_token_id,
            )
            batch["labels"] = self.pad_tensor([example["labels"] for example in examples], -100)
            batch["prompt_answer_attention_mask"] = batch["prompt_answer_ids"].ne(self.target_pad_token_id).long()

        return batch


def train_model(model, train_dataset, eval_dataset, training_args, data_collator):
    last_checkpoint = None
    overwrite_output_dir = getattr(training_args, "overwrite_output_dir", False)
    if training_args.output_dir and not overwrite_output_dir:
        try:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
        except FileNotFoundError:
            last_checkpoint = None

    if hasattr(training_args, "save_safetensors"):
        training_args.save_safetensors = False
    if hasattr(training_args, "save_strategy"):
        training_args.save_strategy = "no"

    callbacks = []
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
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
