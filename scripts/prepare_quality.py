import argparse
import html
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import _bootstrap  # noqa: F401


def clean_article(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<\s*(?:br|/p|p)\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def truncate_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_split_file(raw_dir: Path, split: str, prefer_htmlstripped: bool) -> Path:
    patterns = []
    if prefer_htmlstripped:
        patterns.extend([
            f"**/*htmlstripped.{split}",
            f"**/*htmlstripped.{split}.jsonl",
        ])
    patterns.extend([
        f"**/*.{split}",
        f"**/*.{split}.jsonl",
        f"**/*{split}*.jsonl",
    ])
    for pattern in patterns:
        matches = sorted(path for path in raw_dir.glob(pattern) if path.is_file())
        if matches:
            return matches[0]
    available = ", ".join(sorted(str(path.relative_to(raw_dir)) for path in raw_dir.glob("**/*") if path.is_file()))
    raise FileNotFoundError(f"Could not find QuALITY {split} file under {raw_dir}. Available files: {available}")


def option_label(index: int) -> str:
    return chr(ord("A") + index)


def build_prompt(question: str, options: List[str]) -> str:
    option_lines = [f"{option_label(idx)}. {option}" for idx, option in enumerate(options)]
    return (
        "Answer the multiple-choice question using the long document.\n"
        f"Question: {question.strip()}\n"
        "Options:\n"
        + "\n".join(option_lines)
        + "\nAnswer with the correct option text.\nAnswer:"
    )


def convert_split(path: Path, split: str, max_source_words: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for article_row in read_jsonl(path):
        article = clean_article(article_row.get("article", ""))
        article = truncate_words(article, max_source_words)
        if not article:
            continue
        title = str(article_row.get("title", "")).strip()
        source_text = f"Title: {title}\n\n{article}" if title else article
        questions = article_row.get("questions") or []
        for q_idx, question_row in enumerate(questions):
            options = [str(option).strip() for option in question_row.get("options", [])]
            if len(options) != 4:
                continue
            gold_label = question_row.get("gold_label")
            if gold_label is None:
                continue
            answer_index = int(gold_label) - 1
            if answer_index < 0 or answer_index >= len(options):
                continue
            question = str(question_row.get("question", "")).strip()
            if not question:
                continue
            rows.append(
                {
                    "id": f"{article_row.get('article_id', '')}:{article_row.get('set_unique_id', '')}:{q_idx}",
                    "dataset": "quality",
                    "split": split,
                    "task_type": "multiple_choice",
                    "text": source_text,
                    "prompt": build_prompt(question, options),
                    "answer": options[answer_index],
                    "answer_index": answer_index,
                    "answer_label": option_label(answer_index),
                    "labels": options,
                    "question": question,
                    "difficult": int(question_row.get("difficult", 0) or 0),
                    "article_id": article_row.get("article_id", ""),
                    "title": title,
                }
            )
    return rows


def sample_rows(rows: List[Dict[str, Any]], target_size: int, seed: int) -> List[Dict[str, Any]]:
    if target_size <= 0 or len(rows) <= target_size:
        return rows
    rng = random.Random(seed)
    return [rows[idx] for idx in sorted(rng.sample(range(len(rows)), target_size))]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare QuALITY multiple-choice long-document QA for FlowComp.")
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_size", type=int, default=0)
    parser.add_argument("--eval_size", type=int, default=0)
    parser.add_argument("--max_source_words", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefer_htmlstripped", action="store_true", default=True)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    train_path = find_split_file(raw_dir, "train", args.prefer_htmlstripped)
    dev_path = find_split_file(raw_dir, "dev", args.prefer_htmlstripped)
    train_rows = convert_split(train_path, "train", args.max_source_words)
    eval_rows = convert_split(dev_path, "dev", args.max_source_words)
    train_rows = sample_rows(train_rows, args.train_size, args.seed + 1)
    eval_rows = sample_rows(eval_rows, args.eval_size, args.seed + 2)
    if not train_rows or not eval_rows:
        raise ValueError("QuALITY preprocessing produced an empty split.")

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "eval.jsonl", eval_rows)
    summary = {
        "dataset": "quality",
        "task_type": "multiple_choice",
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "train_source_file": str(train_path),
        "eval_source_file": str(dev_path),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
