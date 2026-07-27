import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pyarrow.parquet as pq


def build_prompt(question: str) -> str:
    return (
        "Answer the open-domain question concisely.\n"
        f"Question: {question}\n"
        "Answer:"
    )


def clean_detokenized_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([([])\s+", r"\1", text)
    return text


def iter_parquet_rows(paths: Iterable[Path]) -> Iterable[Dict]:
    for path in sorted(paths):
        table = pq.read_table(path)
        for row in table.to_pylist():
            yield row


def is_yes_no_question(question: str) -> bool:
    prefixes = (
        "is ", "are ", "was ", "were ", "do ", "does ", "did ",
        "can ", "could ", "will ", "would ", "should ", "has ",
        "have ", "had ", "am ", "may ", "might ", "must ",
    )
    normalized = question.strip().lower()
    return normalized.startswith(prefixes)


def extract_nq_answer(row: Dict) -> str:
    annotations = row["annotations"]
    short_answers = annotations["short_answers"]
    for candidate in short_answers:
        texts = [text.strip() for text in candidate["text"] if text.strip()]
        if texts:
            return texts[0]

    question = row["question"]["text"]
    if not is_yes_no_question(question):
        return ""

    values = {value for value in annotations["yes_no_answer"] if value != -1}
    if len(values) != 1:
        return ""

    value = next(iter(values))
    if value == 0:
        return "yes"
    if value == 1:
        return "no"
    return ""


def build_nq_context(row: Dict, max_context_tokens: int) -> str:
    tokens = row["document"]["tokens"]["token"]
    is_html = row["document"]["tokens"]["is_html"]
    visible_tokens = [token for token, html in zip(tokens, is_html) if not html]
    visible_tokens = visible_tokens[:max_context_tokens]
    title = row["document"]["title"].strip()
    body = clean_detokenized_text(" ".join(visible_tokens))
    if title:
        return f"Title: {title}\n\n{body}".strip()
    return body


def convert_nq_row(row: Dict, max_context_tokens: int) -> Dict | None:
    answer = extract_nq_answer(row)
    if not answer:
        return None

    question = row["question"]["text"].strip()
    context = build_nq_context(row, max_context_tokens)
    if not question or not context:
        return None

    return {
        "id": row["id"],
        "title": row["document"]["title"].strip(),
        "text": context,
        "prompt": build_prompt(question),
        "answer": answer,
        "question": question,
        "dataset": "natural_questions",
    }


def build_trivia_context(row: Dict) -> str:
    question = row["question"].strip()
    search_titles = row["search_results"]["title"]
    search_descriptions = row["search_results"]["description"]
    entity_titles = row["entity_pages"]["title"]

    snippets: List[str] = []
    seen = set()
    for title in entity_titles:
        title = title.strip()
        if title and title not in seen:
            snippets.append(f"Entity title: {title}")
            seen.add(title)
    for title, description in zip(search_titles, search_descriptions):
        title = title.strip()
        description = description.strip()
        parts = [part for part in [title, description] if part]
        if parts:
            key = " | ".join(parts)
            if key not in seen:
                snippets.append(f"Search result: {key}")
                seen.add(key)

    if snippets:
        return "\n".join(snippets[:8])
    return f"Question-only open-domain QA example.\nQuestion: {question}"


def convert_trivia_row(row: Dict, dataset_name: str) -> Dict | None:
    question = row["question"].strip()
    answer = row["answer"]["value"].strip()
    if not question or not answer:
        return None

    return {
        "id": row["question_id"],
        "title": "",
        "text": build_trivia_context(row),
        "prompt": build_prompt(question),
        "answer": answer,
        "question": question,
        "dataset": dataset_name,
    }


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def is_eval_example(example_id: object, seed: int) -> bool:
    digest = hashlib.md5(f"{seed}:{example_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10 == 0


def maybe_add_sample(rows: List[Dict], row: Dict, target_size: int, seen_count: int, rng: random.Random) -> None:
    if target_size <= 0:
        rows.append(row)
        return
    if len(rows) < target_size:
        rows.append(row)
        return
    replace_index = rng.randrange(seen_count)
    if replace_index < target_size:
        rows[replace_index] = row


def resolve_trivia_raw_dir(raw_root: Path, variant: str) -> Tuple[Path, str]:
    if variant == "rc":
        raw_dir = raw_root / "rc"
        dataset_name = "trivia_qa_rc"
    elif variant == "rc.nocontext":
        raw_dir = raw_root / "rc.nocontext"
        dataset_name = "trivia_qa_rc_nocontext"
    else:
        if (raw_root / "rc").exists():
            raw_dir = raw_root / "rc"
            dataset_name = "trivia_qa_rc"
        elif (raw_root / "rc.nocontext").exists():
            raw_dir = raw_root / "rc.nocontext"
            dataset_name = "trivia_qa_rc_nocontext"
        else:
            raise ValueError(f"Could not find rc or rc.nocontext under {raw_root}")

    if not raw_dir.exists():
        raise ValueError(f"Missing TriviaQA directory: {raw_dir}")
    return raw_dir, dataset_name


def prepare_nq(args: argparse.Namespace) -> Dict[str, int]:
    raw_dir = Path(args.raw_dir)
    train_paths = list((raw_dir / "default").glob("train-*.parquet"))
    validation_paths = list((raw_dir / "default").glob("validation-*.parquet"))

    train_rows: List[Dict] = []
    eval_rows: List[Dict] = []
    train_seen = 0
    eval_seen = 0
    train_rng = random.Random(args.seed + 1)
    eval_rng = random.Random(args.seed + 2)

    for row in iter_parquet_rows(train_paths):
        converted = convert_nq_row(row, args.max_context_tokens)
        if converted is None:
            continue
        train_seen += 1
        maybe_add_sample(train_rows, converted, args.train_size, train_seen, train_rng)

    for row in iter_parquet_rows(validation_paths):
        converted = convert_nq_row(row, args.max_context_tokens)
        if converted is None:
            continue
        eval_seen += 1
        maybe_add_sample(eval_rows, converted, args.eval_size, eval_seen, eval_rng)

    if not train_rows or not eval_rows:
        raise ValueError("Natural Questions preprocessing produced an empty split.")

    write_jsonl(Path(args.output_dir) / "train.jsonl", train_rows)
    write_jsonl(Path(args.output_dir) / "eval.jsonl", eval_rows)
    return {
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "source_rows": train_seen + eval_seen,
        "train_source_rows": train_seen,
        "eval_source_rows": eval_seen,
    }


def prepare_trivia(args: argparse.Namespace) -> Dict[str, int | str]:
    raw_dir, dataset_name = resolve_trivia_raw_dir(Path(args.raw_dir), args.trivia_variant)

    train_rows: List[Dict] = []
    eval_rows: List[Dict] = []
    train_seen = 0
    eval_seen = 0
    train_rng = random.Random(args.seed + 1)
    eval_rng = random.Random(args.seed + 2)

    for row in iter_parquet_rows(raw_dir.glob("train-*.parquet")):
        converted = convert_trivia_row(row, dataset_name)
        if converted is None:
            continue
        train_seen += 1
        maybe_add_sample(train_rows, converted, args.train_size, train_seen, train_rng)

    for row in iter_parquet_rows(raw_dir.glob("validation-*.parquet")):
        converted = convert_trivia_row(row, dataset_name)
        if converted is None:
            continue
        eval_seen += 1
        maybe_add_sample(eval_rows, converted, args.eval_size, eval_seen, eval_rng)

    if not train_rows or not eval_rows:
        raise ValueError("TriviaQA preprocessing produced an empty split.")

    write_jsonl(Path(args.output_dir) / "train.jsonl", train_rows)
    write_jsonl(Path(args.output_dir) / "eval.jsonl", eval_rows)
    return {
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "source_rows": train_seen + eval_seen,
        "train_source_rows": train_seen,
        "eval_source_rows": eval_seen,
        "dataset_name": dataset_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["natural_questions", "trivia_qa"], required=True)
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_size", type=int, default=256)
    parser.add_argument("--eval_size", type=int, default=64)
    parser.add_argument("--max_context_tokens", type=int, default=768)
    parser.add_argument("--trivia_variant", choices=["auto", "rc", "rc.nocontext"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset == "natural_questions":
        summary = prepare_nq(args)
    else:
        summary = prepare_trivia(args)

    summary["dataset"] = args.dataset
    summary["output_dir"] = args.output_dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
