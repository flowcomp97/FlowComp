import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import _bootstrap  # noqa: F401
import torch
from tqdm.auto import tqdm

from flowcomp import FlowComp, ModelArguments, TrainingArguments
from flowcomp.paths import output_path


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def token_f1(prediction: str, answer: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    answer_tokens = normalize_text(answer).split()
    if not pred_tokens and not answer_tokens:
        return 1.0
    if not pred_tokens or not answer_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(answer_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_n_f1(prediction: str, answer: str, n: int) -> float:
    pred_tokens = normalize_text(prediction).split()
    answer_tokens = normalize_text(answer).split()
    if len(pred_tokens) < n or len(answer_tokens) < n:
        return 0.0
    pred_ngrams = Counter(tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1))
    answer_ngrams = Counter(tuple(answer_tokens[i : i + n]) for i in range(len(answer_tokens) - n + 1))
    overlap = sum((pred_ngrams & answer_ngrams).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(sum(pred_ngrams.values()), 1)
    recall = overlap / max(sum(answer_ngrams.values()), 1)
    return 2 * precision * recall / (precision + recall)


def lcs_length(a: List[str], b: List[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for idx, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr[idx] = prev[idx - 1] + 1
            else:
                curr[idx] = max(prev[idx], curr[idx - 1])
        prev = curr
    return prev[-1]


def rouge_l_f1(prediction: str, answer: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    answer_tokens = normalize_text(answer).split()
    if not pred_tokens or not answer_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, answer_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(answer_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def clean_prediction(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(?:summary|answer|prediction)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip(" \t\"'`")


def predict_choice(prediction: str, labels: List[str]) -> str:
    normalized_prediction = normalize_text(prediction)
    normalized_labels = {normalize_text(label): label for label in labels}
    if normalized_prediction in normalized_labels:
        return normalized_labels[normalized_prediction]

    first_line = prediction.strip().splitlines()[0] if prediction.strip() else ""
    first_token_match = re.match(r"^\s*([A-Da-d1-4])(?:[\).:：\s]|$)", first_line)
    if first_token_match:
        token = first_token_match.group(1).upper()
        index = ord(token) - ord("A") if token.isalpha() else int(token) - 1
        if 0 <= index < len(labels):
            return labels[index]

    matches = []
    for label in labels:
        normalized_label = normalize_text(label)
        if normalized_label and normalized_label in normalized_prediction:
            matches.append(label)
    if len(matches) == 1:
        return matches[0]
    return prediction.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate FlowComp long-document summarization/QA generation.")
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_file", default="")
    parser.add_argument("--source_encoder_name_or_path", default="google-t5/t5-small")
    parser.add_argument("--target_model_name_or_path", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--target_model_device_map", default="none")
    parser.add_argument("--latent_generator_type", choices=["flow", "one_shot"], default="flow")
    parser.add_argument("--distribution_loss_type", choices=["wasserstein", "mmd"], default="wasserstein")
    parser.add_argument("--soft_token_attention", choices=["bidirectional", "causal"], default="bidirectional")
    parser.add_argument("--num_soft_tokens", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--sampling_steps", type=int, default=4)
    parser.add_argument("--denoiser_layers", type=int, default=4)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_examples", type=int, default=0, help="Evaluate only the first N examples. 0 means all.")
    parser.add_argument("--source_key", default="text")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--answer_key", default="answer")
    parser.add_argument("--task_type", choices=["auto", "summarization", "qa", "multiple_choice"], default="auto")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for FlowComp inference noise.")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use_flash_attention_2", action="store_true")
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
        output_dir=str(output_path("scratch", "eval_long_document")),
        bf16=args.bf16,
        model_max_length=args.model_max_length,
        per_device_eval_batch_size=1,
        report_to=[],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading FlowComp model on {device}...", flush=True)
    model = FlowComp(model_args, training_args)
    print(f"Loading checkpoint: {args.checkpoint_path}", flush=True)
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    print("Applying checkpoint weights...", flush=True)
    model.load_state_dict(state_dict, strict=False)
    print(f"Moving model to {device}...", flush=True)
    model.to(device)
    model.eval()

    rows = load_jsonl(args.eval_file)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    print(
        f"Evaluating {len(rows)} examples from {args.eval_file} "
        f"with max_new_tokens={args.max_new_tokens}...",
        flush=True,
    )
    outputs = []
    totals = Counter()

    for row in tqdm(rows, desc="eval", unit="ex"):
        task_type = args.task_type if args.task_type != "auto" else row.get("task_type", "qa")
        encoder = model.source_tokenizer(
            row[args.source_key],
            truncation=True,
            max_length=args.model_max_length,
            padding=False,
            return_tensors="pt",
        )
        prompt_max_length = max(args.model_max_length - args.num_soft_tokens, 1)
        prompt_tokens = model.target_tokenizer(
            row[args.prompt_key],
            truncation=True,
            max_length=prompt_max_length,
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
        prediction = clean_prediction(raw_prediction)
        answer = row[args.answer_key]

        sample = {
            "id": row.get("id", ""),
            "dataset": row.get("dataset", ""),
            "task_type": task_type,
            "prompt": row[args.prompt_key],
            "answer": answer,
            "raw_prediction": raw_prediction,
            "prediction": prediction,
        }
        if task_type == "summarization":
            sample["rouge1"] = rouge_n_f1(prediction, answer, 1)
            sample["rouge2"] = rouge_n_f1(prediction, answer, 2)
            sample["rougeL"] = rouge_l_f1(prediction, answer)
            totals["rouge1"] += sample["rouge1"]
            totals["rouge2"] += sample["rouge2"]
            totals["rougeL"] += sample["rougeL"]
        elif task_type == "multiple_choice" or row.get("labels"):
            labels = [str(label) for label in row.get("labels", [])]
            choice_prediction = predict_choice(prediction, labels) if labels else prediction
            sample["labels"] = labels
            sample["choice_prediction"] = choice_prediction
            sample["correct"] = normalize_text(choice_prediction) == normalize_text(answer)
            totals["accuracy"] += int(sample["correct"])
            if row.get("difficult"):
                sample["difficult"] = int(row.get("difficult", 0))
                totals["hard_count"] += 1
                totals["hard_accuracy"] += int(sample["correct"])
        else:
            norm_pred = normalize_text(prediction)
            norm_answer = normalize_text(answer)
            sample["exact_match"] = norm_pred == norm_answer
            sample["contains_match"] = bool(norm_pred) and bool(norm_answer) and (
                norm_answer in norm_pred or norm_pred in norm_answer
            )
            sample["token_f1"] = token_f1(prediction, answer)
            totals["exact_match"] += int(sample["exact_match"])
            totals["contains_match"] += int(sample["contains_match"])
            totals["token_f1"] += sample["token_f1"]
        outputs.append(sample)

    metrics = {"seed": args.seed, "num_examples": len(rows)}
    for key, value in totals.items():
        if key == "hard_count":
            metrics[key] = int(value)
        elif key == "hard_accuracy":
            metrics[key] = value / max(totals.get("hard_count", 0), 1)
        else:
            metrics[key] = value / max(len(rows), 1)

    payload = {"metrics": metrics, "predictions": outputs}
    print(json.dumps(metrics, indent=2, ensure_ascii=True))
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_file).write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()
