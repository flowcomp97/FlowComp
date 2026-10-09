import argparse
import json
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import _bootstrap  # noqa: F401
import torch

from flowcomp import FlowComp, ModelArguments, TrainingArguments
from flowcomp.paths import output_path


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_label(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"^(?:label|answer|prediction)\s*:\s*", "", text)
    text = text.strip(string.whitespace + string.punctuation)
    text = re.sub(r"\s+", " ", text)
    return text


def first_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text.strip()


def predict_label(raw_prediction: str, labels: Sequence[str]) -> str:
    candidate = normalize_label(first_line(raw_prediction))
    normalized_labels = {normalize_label(label): label for label in labels}
    if candidate in normalized_labels:
        return normalized_labels[candidate]

    candidate_tokens = re.findall(r"[A-Za-z0-9_+-]+", candidate)
    if candidate_tokens:
        first_token = normalize_label(candidate_tokens[0])
        if first_token in normalized_labels:
            return normalized_labels[first_token]

    matches = []
    padded_candidate = f" {candidate} "
    for normalized, label in normalized_labels.items():
        padded_label = f" {normalized} "
        if padded_label in padded_candidate:
            matches.append(label)
    if len(matches) == 1:
        return matches[0]
    return candidate


def macro_f1(golds: Sequence[str], predictions: Sequence[str], labels: Sequence[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(1 for gold, pred in zip(golds, predictions) if gold == label and pred == label)
        fp = sum(1 for gold, pred in zip(golds, predictions) if gold != label and pred == label)
        fn = sum(1 for gold, pred in zip(golds, predictions) if gold == label and pred != label)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / max(len(scores), 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate FlowComp few-shot classification accuracy.")
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_file", default="")
    parser.add_argument("--source_encoder_name_or_path", default="google-t5/t5-small")
    parser.add_argument("--target_model_name_or_path", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--target_model_device_map", default="none")
    parser.add_argument("--latent_generator_type", choices=["flow", "one_shot"], default="flow")
    parser.add_argument(
        "--distribution_loss_type", choices=["wasserstein", "mmd", "cosine"], default="wasserstein"
    )
    parser.add_argument("--soft_token_attention", choices=["bidirectional", "causal"], default="bidirectional")
    # Defaults follow the in-context learning column of Table 8.
    parser.add_argument("--num_soft_tokens", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--sampling_steps", type=int, default=8)
    parser.add_argument("--denoiser_layers", type=int, default=4)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--source_key", default="text")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--answer_key", default="answer")
    parser.add_argument("--labels_key", default="labels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for FlowComp inference noise.")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use_flash_attention_2", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_args = ModelArguments(
        source_encoder_name_or_path=args.source_encoder_name_or_path,
        target_model_name_or_path=args.target_model_name_or_path,
        target_model_device_map=args.target_model_device_map,
        latent_generator_type=args.latent_generator_type,
        distribution_loss_type=args.distribution_loss_type,
        soft_token_attention=args.soft_token_attention,
        num_soft_tokens=args.num_soft_tokens,
        latent_dim=args.latent_dim,
        sampling_steps=args.sampling_steps,
        denoiser_layers=args.denoiser_layers,
        decoder_layers=args.decoder_layers,
        num_attention_heads=args.num_attention_heads,
        freeze_target_model=True,
        source_encoder_trainable=False,
        use_flash_attention_2=args.use_flash_attention_2,
    )
    training_args = TrainingArguments(
        output_dir=str(output_path("scratch", "eval_classification")),
        bf16=args.bf16,
        model_max_length=args.model_max_length,
        per_device_eval_batch_size=1,
        report_to=[],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowComp(model_args, training_args)
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    rows = load_jsonl(args.eval_file)
    outputs = []
    golds = []
    predictions = []
    labels_by_task: Dict[str, set[str]] = defaultdict(set)

    for row in rows:
        labels = row.get(args.labels_key)
        if not labels:
            labels = sorted({item[args.answer_key] for item in rows if item.get("task") == row.get("task")})
        labels = [str(label) for label in labels]

        encoder = model.source_tokenizer(
            row[args.source_key],
            truncation=True,
            max_length=args.model_max_length,
            padding=False,
            return_tensors="pt",
        )
        prompt_tokens = model.target_tokenizer(
            row[args.prompt_key],
            truncation=True,
            max_length=args.model_max_length,
            padding=False,
            return_tensors="pt",
            add_special_tokens=False,
        )

        generated_ids = model.generate(
            encoder_input_ids=encoder["input_ids"].to(device),
            encoder_attention_mask=encoder["attention_mask"].to(device),
            prompt_ids=prompt_tokens["input_ids"].to(device),
            prompt_attention_mask=prompt_tokens["attention_mask"].to(device),
            max_new_tokens=args.max_new_tokens,
        )
        raw_prediction = model.target_tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        predicted_label = predict_label(raw_prediction, labels)
        gold_label = str(row[args.answer_key])
        correct = normalize_label(predicted_label) == normalize_label(gold_label)

        task = row.get("task", row.get("dataset", "classification"))
        labels_by_task[task].update(labels)
        golds.append(gold_label)
        predictions.append(predicted_label)
        outputs.append(
            {
                "id": row.get("id", ""),
                "dataset": row.get("dataset", ""),
                "task": task,
                "query": row.get("query", ""),
                "labels": labels,
                "answer": gold_label,
                "raw_prediction": raw_prediction,
                "prediction": predicted_label,
                "correct": correct,
            }
        )

    correct_count = sum(1 for item in outputs if item["correct"])
    all_labels = sorted({label for labels in labels_by_task.values() for label in labels})
    metrics = {
        "seed": args.seed,
        "num_examples": len(rows),
        "accuracy": correct_count / max(len(rows), 1),
        "macro_f1": macro_f1(golds, predictions, all_labels),
    }

    task_metrics = {}
    for task, task_labels in labels_by_task.items():
        task_outputs = [item for item in outputs if item["task"] == task]
        task_golds = [item["answer"] for item in task_outputs]
        task_predictions = [item["prediction"] for item in task_outputs]
        task_correct = sum(1 for item in task_outputs if item["correct"])
        task_metrics[task] = {
            "num_examples": len(task_outputs),
            "accuracy": task_correct / max(len(task_outputs), 1),
            "macro_f1": macro_f1(task_golds, task_predictions, sorted(task_labels)),
        }
    metrics["by_task"] = task_metrics

    payload = {"metrics": metrics, "predictions": outputs}
    print(json.dumps(metrics, indent=2, ensure_ascii=True))
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_file).write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()
