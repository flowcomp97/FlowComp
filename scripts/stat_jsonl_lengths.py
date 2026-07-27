import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

import _bootstrap  # noqa: F401
from transformers import AutoTokenizer


def load_rows(path: Path, limit: int = 0) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def percentile(sorted_values: List[int], q: float) -> int:
    if not sorted_values:
        return 0
    index = min(int(round((len(sorted_values) - 1) * q)), len(sorted_values) - 1)
    return sorted_values[index]


def summarize(values: List[int]) -> Dict[str, float | int]:
    values = sorted(values)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": values[0],
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": values[-1],
        "mean": round(mean(values), 2),
    }


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute character/token length stats for FlowComp JSONL files.")
    parser.add_argument("--data_dir", default="", help="Directory containing train/eval/test JSONL files.")
    parser.add_argument("--files", nargs="*", default=[], help="Explicit JSONL files. Overrides --data_dir.")
    parser.add_argument("--splits", nargs="*", default=["train", "eval", "test"])
    parser.add_argument("--source_tokenizer", default="google-t5/t5-small")
    parser.add_argument("--target_tokenizer", default="")
    parser.add_argument("--source_key", default="text")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--answer_key", default="answer")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows per file for quick sampling. 0 means full file.")
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    if args.files:
        files = [Path(path) for path in args.files]
    elif args.data_dir:
        files = [Path(args.data_dir) / f"{split}.jsonl" for split in args.splits]
    else:
        raise ValueError("Provide --data_dir or --files.")
    files = [path for path in files if path.exists()]
    if not files:
        raise FileNotFoundError("No JSONL files found.")

    target_tokenizer_name = args.target_tokenizer or args.source_tokenizer
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_tokenizer, use_fast=True)
    target_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer_name, use_fast=False)

    report = {
        "source_tokenizer": args.source_tokenizer,
        "target_tokenizer": target_tokenizer_name,
        "limit_per_file": args.limit,
        "files": {},
    }

    for path in files:
        buckets = {
            "context_chars": [],
            "prompt_chars": [],
            "answer_chars": [],
            "context_source_tokens": [],
            "prompt_target_tokens": [],
            "answer_target_tokens": [],
            "target_total_tokens_without_soft": [],
        }
        for row in load_rows(path, args.limit):
            context = str(row.get(args.source_key, ""))
            prompt = str(row.get(args.prompt_key, ""))
            answer = str(row.get(args.answer_key, ""))
            context_tokens = token_len(source_tokenizer, context)
            prompt_tokens = token_len(target_tokenizer, prompt)
            answer_tokens = token_len(target_tokenizer, answer)
            buckets["context_chars"].append(len(context))
            buckets["prompt_chars"].append(len(prompt))
            buckets["answer_chars"].append(len(answer))
            buckets["context_source_tokens"].append(context_tokens)
            buckets["prompt_target_tokens"].append(prompt_tokens)
            buckets["answer_target_tokens"].append(answer_tokens)
            buckets["target_total_tokens_without_soft"].append(prompt_tokens + answer_tokens + 1)

        file_report = {key: summarize(values) for key, values in buckets.items()}
        report["files"][str(path)] = file_report
        print(f"\n{path}")
        for key, stats in file_report.items():
            print(f"  {key}: {stats}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()
