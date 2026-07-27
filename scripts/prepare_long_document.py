import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

import _bootstrap  # noqa: F401
from datasets import Dataset, DatasetDict, load_from_disk


LONG_DOCUMENT_CONFIGS = {
    "longbench_gov_report": {
        "repo": "THUDM/LongBench",
        "config": "gov_report",
        "task_type": "summarization",
    },
    "longbench_qmsum": {
        "repo": "THUDM/LongBench",
        "config": "qmsum",
        "task_type": "summarization",
    },
    "longbench_multi_news": {
        "repo": "THUDM/LongBench",
        "config": "multi_news",
        "task_type": "summarization",
    },
    "longbench_musique": {
        "repo": "THUDM/LongBench",
        "config": "musique",
        "task_type": "qa",
    },
    "longbench_qasper": {
        "repo": "THUDM/LongBench",
        "config": "qasper",
        "task_type": "qa",
    },
    "longbench_quality": {
        "repo": "THUDM/LongBench",
        "config": "quality",
        "task_type": "qa",
        "unavailable_reason": "THUDM/LongBench data.zip does not contain quality.jsonl. Use longbench_qasper for LongBench single-document QA, or add a separate QuALITY raw converter.",
    },
}


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(coerce_text(item) for item in value if coerce_text(item)).strip()
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            text = coerce_text(item)
            if text:
                parts.append(f"{key}: {text}")
        return "\n".join(parts).strip()
    return str(value).strip()


def first_answer(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            text = coerce_text(item)
            if text:
                return text
        return ""
    return coerce_text(value)


def truncate_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_longbench_jsonl(raw_dir: Path, config_name: str) -> Path | None:
    candidates = [
        raw_dir / "data" / f"{config_name}.jsonl",
        raw_dir / f"{config_name}.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(raw_dir.glob(f"**/{config_name}.jsonl"))
    return matches[0] if matches else None


def load_source_dataset(args: argparse.Namespace) -> DatasetDict:
    if args.dataset not in LONG_DOCUMENT_CONFIGS:
        supported = ", ".join(sorted(LONG_DOCUMENT_CONFIGS))
        raise ValueError(f"Unsupported dataset '{args.dataset}'. Supported: {supported}")

    spec = LONG_DOCUMENT_CONFIGS[args.dataset]
    if "unavailable_reason" in spec:
        raise ValueError(spec["unavailable_reason"])

    if args.raw_dir:
        raw_dir = Path(args.raw_dir)
        jsonl_path = find_longbench_jsonl(raw_dir, spec["config"])
        if jsonl_path is not None:
            return DatasetDict({"test": Dataset.from_list(read_jsonl_rows(jsonl_path))})
        available = ", ".join(sorted(path.stem for path in raw_dir.glob("**/*.jsonl")))
        raise FileNotFoundError(
            f"Could not find {spec['config']}.jsonl under {raw_dir}. "
            f"Available LongBench files: {available or 'none'}"
        )

    raise ValueError(
        "--raw_dir is required. Run scripts/download_long_document.sh to download "
        "LongBench data.zip and prepare the JSONL files."
    )


def infer_task_type(args: argparse.Namespace) -> str:
    if args.task_type != "auto":
        return args.task_type
    if args.dataset in LONG_DOCUMENT_CONFIGS:
        return LONG_DOCUMENT_CONFIGS[args.dataset]["task_type"]
    return "qa"


def build_prompt(row: Dict[str, Any], task_type: str) -> str:
    instruction = coerce_text(row.get("input"))
    if task_type == "summarization":
        if instruction:
            return f"{instruction}\n\nSummary:"
        return "Summarize the long document concisely.\nSummary:"
    if instruction:
        return f"{instruction}\n\nAnswer:"
    question = coerce_text(row.get("question"))
    if question:
        return f"Answer the question using the long document.\nQuestion: {question}\nAnswer:"
    return "Answer the question using the long document.\nAnswer:"


def convert_row(row: Dict[str, Any], args: argparse.Namespace, task_type: str) -> Dict[str, Any] | None:
    source_text = coerce_text(row.get(args.source_field))
    if not source_text:
        for key in ("context", "document", "article", "text", "passage", "input"):
            source_text = coerce_text(row.get(key))
            if source_text:
                break
    source_text = truncate_words(source_text, args.max_source_words)

    answer = first_answer(row.get(args.answer_field))
    if not answer:
        for key in ("answers", "answer", "summary", "output", "target"):
            answer = first_answer(row.get(key))
            if answer:
                break

    prompt = build_prompt(row, task_type)
    if not source_text or not prompt or not answer:
        return None

    example_id = row.get("_id", row.get("id", row.get("qid", "")))
    converted = {
        "id": coerce_text(example_id),
        "dataset": args.dataset,
        "task_type": task_type,
        "text": source_text,
        "prompt": prompt,
        "answer": answer,
    }
    if "all_classes" in row:
        converted["labels"] = row["all_classes"]
    if "length" in row:
        converted["source_length"] = row["length"]
    return converted


def sample_rows(rows: List[Dict[str, Any]], target_size: int, seed: int) -> List[Dict[str, Any]]:
    if target_size <= 0 or len(rows) <= target_size:
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), target_size))
    return [rows[index] for index in indices]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def select_splits(dataset: DatasetDict, args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    if args.train_split in dataset and args.eval_split in dataset:
        return dataset[args.train_split], dataset[args.eval_split]

    if args.train_split in dataset:
        source = dataset[args.train_split]
    elif args.eval_split in dataset:
        source = dataset[args.eval_split]
    else:
        source = dataset[next(iter(dataset.keys()))]

    source_len = len(source)
    if source_len < 2:
        raise ValueError("Need at least two examples to create train/eval splits from a single LongBench file.")
    if args.eval_size <= 0:
        test_size = max(1, int(round(source_len * args.eval_fraction)))
    elif args.eval_size >= source_len:
        test_size = max(1, int(round(source_len * args.eval_fraction)))
    else:
        test_size = args.eval_size
    test_size = min(test_size, source_len - 1)
    split = source.train_test_split(test_size=test_size, seed=args.seed)
    return split["train"], split["test"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare long-document summarization/QA JSONL for FlowComp.")
    parser.add_argument("--dataset", choices=sorted(LONG_DOCUMENT_CONFIGS), required=True)
    parser.add_argument("--raw_dir", default="")
    parser.add_argument("--save_raw_dir", default="")
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task_type", choices=["auto", "summarization", "qa"], default="auto")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="validation")
    parser.add_argument("--source_field", default="context")
    parser.add_argument("--answer_field", default="answers")
    parser.add_argument("--train_size", type=int, default=2000)
    parser.add_argument("--eval_size", type=int, default=500)
    parser.add_argument("--eval_fraction", type=float, default=0.2)
    parser.add_argument("--max_source_words", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_source_dataset(args)
    task_type = infer_task_type(args)
    train_split, eval_split = select_splits(dataset, args)

    train_rows = [row for row in (convert_row(dict(item), args, task_type) for item in train_split) if row]
    eval_rows = [row for row in (convert_row(dict(item), args, task_type) for item in eval_split) if row]
    train_rows = sample_rows(train_rows, args.train_size, args.seed + 1)
    eval_rows = sample_rows(eval_rows, args.eval_size, args.seed + 2)

    if not train_rows or not eval_rows:
        raise ValueError("Long-document preprocessing produced an empty split.")

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "eval.jsonl", eval_rows)
    summary = {
        "dataset": args.dataset,
        "task_type": task_type,
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "eval_fraction": args.eval_fraction,
        "output_dir": str(output_dir),
        "raw_dir": args.raw_dir or args.save_raw_dir,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
