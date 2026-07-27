import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import _bootstrap  # noqa: F401
import torch

from flowcomp import FlowComp, ModelArguments, TrainingArguments
from flowcomp.paths import output_path


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonicalize_prediction(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(?:answer\s*:\s*|the answer is\s+)", "", text, flags=re.IGNORECASE)
    return text.strip(" \t\"'`[](){}")


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


def load_jsonl(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate FlowComp QA generation accuracy")
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_file", default="")
    parser.add_argument("--source_encoder_name_or_path", default="google-t5/t5-small")
    parser.add_argument("--target_model_name_or_path", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--target_model_device_map", default="none")
    parser.add_argument("--latent_generator_type", choices=["flow", "one_shot"], default="flow")
    parser.add_argument("--distribution_loss_type", choices=["wasserstein", "mmd"], default="wasserstein")
    parser.add_argument("--soft_token_attention", choices=["bidirectional", "causal"], default="bidirectional")
    parser.add_argument("--num_soft_tokens", type=int, default=16)
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--sampling_steps", type=int, default=2)
    parser.add_argument("--denoiser_layers", type=int, default=2)
    parser.add_argument("--decoder_layers", type=int, default=1)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--stop_strings", default="")
    parser.add_argument("--source_key", default="text")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--answer_key", default="answer")
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
        output_dir=str(output_path("scratch", "eval_qa")),
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
    exact_match = 0
    contains_match = 0
    f1_sum = 0.0

    for row in rows:
        source_text = row[args.source_key]
        prompt = row[args.prompt_key]
        answer = row[args.answer_key]

        encoder = model.source_tokenizer(
            source_text,
            truncation=True,
            max_length=args.model_max_length,
            padding=False,
            return_tensors="pt",
        )
        prompt_tokens = model.target_tokenizer(
            prompt,
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
            stop_strings=tuple(item for item in args.stop_strings.split("||") if item),
        )
        raw_prediction = model.target_tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        prediction = canonicalize_prediction(raw_prediction)

        norm_pred = normalize_text(prediction)
        norm_answer = normalize_text(answer)
        is_exact = norm_pred == norm_answer
        is_contains = bool(norm_pred) and bool(norm_answer) and (norm_answer in norm_pred or norm_pred in norm_answer)
        sample_f1 = token_f1(prediction, answer)
        exact_match += int(is_exact)
        contains_match += int(is_contains)
        f1_sum += sample_f1

        outputs.append(
            {
                "source": source_text,
                "prompt": prompt,
                "answer": answer,
                "raw_prediction": raw_prediction,
                "prediction": prediction,
                "normalized_prediction": norm_pred,
                "normalized_answer": norm_answer,
                "exact_match": is_exact,
                "contains_match": is_contains,
                "token_f1": sample_f1,
            }
        )

    metrics = {
        "seed": args.seed,
        "num_examples": len(rows),
        "exact_match": exact_match / max(len(rows), 1),
        "contains_match": contains_match / max(len(rows), 1),
        "token_f1": f1_sum / max(len(rows), 1),
    }

    payload = {"metrics": metrics, "predictions": outputs}
    print(json.dumps(metrics, indent=2))
    if args.output_file:
        Path(args.output_file).write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()
