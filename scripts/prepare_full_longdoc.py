import argparse
import html
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import _bootstrap  # noqa: F401
from datasets import load_dataset


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, list):
        return "\n".join(clean_text(item) for item in text if clean_text(item)).strip()
    if isinstance(text, dict):
        return "\n".join(f"{k}: {clean_text(v)}" for k, v in text.items() if clean_text(v)).strip()
    text = html.unescape(str(text))
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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def sample_rows(rows: List[Dict[str, Any]], size: int, seed: int) -> List[Dict[str, Any]]:
    if size <= 0 or len(rows) <= size:
        return rows
    rng = random.Random(seed)
    return [rows[idx] for idx in sorted(rng.sample(range(len(rows)), size))]


def prepare_govreport(args: argparse.Namespace) -> Dict[str, int]:
    dataset = load_dataset(
        args.govreport_repo,
        args.govreport_config,
        cache_dir=args.cache_dir or None,
    )
    split_map = {"train": "train", "eval": "validation", "test": "test"}
    counts = {}
    for out_split, hf_split in split_map.items():
        rows = []
        for idx, item in enumerate(dataset[hf_split]):
            source = clean_text(item.get("report") or item.get("document") or item.get("text"))
            answer = clean_text(item.get("summary"))
            if not source or not answer:
                continue
            rows.append(
                {
                    "id": clean_text(item.get("id", f"{hf_split}-{idx}")),
                    "dataset": "govreport",
                    "split": out_split,
                    "task_type": "summarization",
                    "text": truncate_words(source, args.max_source_words),
                    "prompt": "Summarize the government report.\nSummary:",
                    "answer": answer,
                }
            )
        rows = sample_rows(rows, getattr(args, f"{out_split}_size"), args.seed + len(counts))
        write_jsonl(Path(args.output_dir) / f"{out_split}.jsonl", rows)
        counts[out_split] = len(rows)
    return counts


def qmsum_transcript_text(row: Dict[str, Any]) -> str:
    transcript = row.get("meeting_transcripts") or row.get("transcript") or []
    if isinstance(transcript, list):
        parts = []
        for turn in transcript:
            if isinstance(turn, dict):
                speaker = clean_text(turn.get("speaker") or turn.get("speaker_name"))
                content = clean_text(turn.get("content") or turn.get("text") or turn.get("utterance"))
                if speaker and content:
                    parts.append(f"{speaker}: {content}")
                elif content:
                    parts.append(content)
            else:
                parts.append(clean_text(turn))
        return "\n".join(part for part in parts if part)
    return clean_text(transcript)


def qmsum_query_rows(row: Dict[str, Any]) -> Iterable[tuple[str, str, str]]:
    for group_name in ("general_query_list", "specific_query_list", "query_list"):
        for idx, query in enumerate(row.get(group_name) or []):
            if not isinstance(query, dict):
                continue
            question = clean_text(query.get("query") or query.get("question"))
            answer = clean_text(query.get("answer") or query.get("summary"))
            if question and answer:
                yield f"{group_name}-{idx}", question, answer


def find_qmsum_split_files(raw_dir: Path, split: str) -> List[Path]:
    candidates = []
    for rel in [f"data/ALL/jsonl/{split}", f"data/ALL/{split}", f"QMSum-master/data/ALL/jsonl/{split}", f"QMSum-main/data/ALL/jsonl/{split}"]:
        path = raw_dir / rel
        if path.is_dir():
            candidates.extend(sorted(path.glob("*.json*")))
        elif path.is_file():
            candidates.append(path)
    if candidates:
        return candidates
    patterns = [f"**/data/ALL/jsonl/{split}/**/*.json*", f"**/data/ALL/{split}/**/*.json*", f"**/*{split}*.jsonl"]
    for pattern in patterns:
        candidates.extend(sorted(raw_dir.glob(pattern)))
    return sorted(set(candidates))


def load_qmsum_records(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        yield from read_jsonl(path)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        yield from payload
    else:
        yield payload


def prepare_qmsum(args: argparse.Namespace) -> Dict[str, int]:
    split_map = {"train": "train", "eval": "val", "test": "test"}
    counts = {}
    raw_dir = Path(args.raw_dir)
    for out_split, raw_split in split_map.items():
        rows = []
        for path in find_qmsum_split_files(raw_dir, raw_split):
            for row in load_qmsum_records(path):
                source = qmsum_transcript_text(row)
                if not source:
                    continue
                meeting_id = clean_text(row.get("meeting_id") or row.get("id") or path.stem)
                for qid, question, answer in qmsum_query_rows(row):
                    rows.append(
                        {
                            "id": f"{meeting_id}:{qid}",
                            "dataset": "qmsum",
                            "split": out_split,
                            "task_type": "summarization",
                            "text": truncate_words(source, args.max_source_words),
                            "prompt": f"Summarize the meeting according to the query.\nQuery: {question}\nSummary:",
                            "answer": answer,
                            "query": question,
                        }
                    )
        if not rows:
            raise FileNotFoundError(f"No QMSum rows found for split {raw_split} under {raw_dir}")
        rows = sample_rows(rows, getattr(args, f"{out_split}_size"), args.seed + len(counts))
        write_jsonl(Path(args.output_dir) / f"{out_split}.jsonl", rows)
        counts[out_split] = len(rows)
    return counts


def musique_context(row: Dict[str, Any]) -> str:
    paragraphs = row.get("paragraphs") or row.get("contexts") or []
    parts = []
    for idx, paragraph in enumerate(paragraphs, start=1):
        if isinstance(paragraph, dict):
            title = clean_text(paragraph.get("title"))
            text = clean_text(paragraph.get("paragraph_text") or paragraph.get("text") or paragraph.get("context"))
            prefix = f"Passage {idx}"
            if title:
                prefix += f": {title}"
            if text:
                parts.append(f"{prefix}\n{text}")
        else:
            text = clean_text(paragraph)
            if text:
                parts.append(f"Passage {idx}\n{text}")
    return "\n\n".join(parts)


def find_musique_file(raw_dir: Path, variant: str, split: str) -> Path:
    names = [
        f"musique_{variant}_v1.0_{split}.jsonl",
        f"musique_{variant}_{split}.jsonl",
        f"musique_{variant}_v1.0_{split}.json",
    ]
    for name in names:
        matches = sorted(raw_dir.glob(f"**/{name}"))
        if matches:
            return matches[0]
    available = ", ".join(sorted(path.name for path in raw_dir.glob("**/*.jsonl")))
    raise FileNotFoundError(f"Could not find MuSiQue {variant} {split} under {raw_dir}. Available jsonl: {available}")


def prepare_musique(args: argparse.Namespace) -> Dict[str, int]:
    split_map = {"train": "train", "eval": "dev", "test": "test"}
    counts = {}
    raw_dir = Path(args.raw_dir)
    for out_split, raw_split in split_map.items():
        try:
            path = find_musique_file(raw_dir, args.musique_variant, raw_split)
        except FileNotFoundError:
            if out_split == "test":
                counts[out_split] = 0
                continue
            raise
        rows = []
        for row in read_jsonl(path):
            source = musique_context(row)
            question = clean_text(row.get("question"))
            answer = clean_text(row.get("answer"))
            if not source or not question or not answer:
                if out_split == "test" and source and question:
                    answer = ""
                else:
                    continue
            rows.append(
                {
                    "id": clean_text(row.get("id")),
                    "dataset": f"musique_{args.musique_variant}",
                    "split": out_split,
                    "task_type": "qa",
                    "text": truncate_words(source, args.max_source_words),
                    "prompt": f"Answer the multi-hop question using the passages.\nQuestion: {question}\nAnswer:",
                    "answer": answer,
                    "question": question,
                    "answerable": row.get("answerable", True),
                }
            )
        rows = sample_rows(rows, getattr(args, f"{out_split}_size"), args.seed + len(counts))
        write_jsonl(Path(args.output_dir) / f"{out_split}.jsonl", rows)
        counts[out_split] = len(rows)
    return counts


def option_label(index: int) -> str:
    return chr(ord("A") + index)


def quality_prompt(question: str, options: List[str]) -> str:
    lines = [f"{option_label(idx)}. {option}" for idx, option in enumerate(options)]
    return "Answer the multiple-choice question using the long document.\nQuestion: " + question + "\nOptions:\n" + "\n".join(lines) + "\nAnswer with the correct option text.\nAnswer:"


def find_quality_file(raw_dir: Path, split: str) -> Path:
    patterns = [f"**/*htmlstripped.{split}", f"**/*htmlstripped.{split}.jsonl", f"**/*.{split}", f"**/*.{split}.jsonl"]
    for pattern in patterns:
        matches = sorted(path for path in raw_dir.glob(pattern) if path.is_file())
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find QuALITY {split} under {raw_dir}")


def prepare_quality(args: argparse.Namespace) -> Dict[str, int]:
    split_map = {"train": "train", "eval": "dev", "test": "test"}
    counts = {}
    raw_dir = Path(args.raw_dir)
    for out_split, raw_split in split_map.items():
        path = find_quality_file(raw_dir, raw_split)
        rows = []
        for article_row in read_jsonl(path):
            article = truncate_words(clean_text(article_row.get("article")), args.max_source_words)
            title = clean_text(article_row.get("title"))
            source = f"Title: {title}\n\n{article}" if title else article
            for q_idx, question_row in enumerate(article_row.get("questions") or []):
                options = [clean_text(option) for option in question_row.get("options", [])]
                gold_label = question_row.get("gold_label")
                if not source or len(options) != 4 or gold_label is None:
                    continue
                answer_idx = int(gold_label) - 1
                if answer_idx < 0 or answer_idx >= len(options):
                    continue
                question = clean_text(question_row.get("question"))
                rows.append(
                    {
                        "id": f"{article_row.get('article_id', '')}:{article_row.get('set_unique_id', '')}:{q_idx}",
                        "dataset": "quality",
                        "split": out_split,
                        "task_type": "multiple_choice",
                        "text": source,
                        "prompt": quality_prompt(question, options),
                        "answer": options[answer_idx],
                        "answer_index": answer_idx,
                        "answer_label": option_label(answer_idx),
                        "labels": options,
                        "question": question,
                        "difficult": int(question_row.get("difficult", 0) or 0),
                    }
                )
        rows = sample_rows(rows, getattr(args, f"{out_split}_size"), args.seed + len(counts))
        write_jsonl(Path(args.output_dir) / f"{out_split}.jsonl", rows)
        counts[out_split] = len(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare full long-document datasets for FlowComp.")
    parser.add_argument("--dataset", choices=["govreport", "qmsum", "musique", "quality"], required=True)
    parser.add_argument("--raw_dir", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--govreport_repo", default="ccdv/govreport-summarization")
    parser.add_argument("--govreport_config", default="document")
    parser.add_argument("--musique_variant", choices=["ans", "full"], default="ans")
    parser.add_argument("--train_size", type=int, default=0)
    parser.add_argument("--eval_size", type=int, default=0)
    parser.add_argument("--test_size", type=int, default=0)
    parser.add_argument("--max_source_words", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset == "govreport":
        counts = prepare_govreport(args)
    elif args.dataset == "qmsum":
        counts = prepare_qmsum(args)
    elif args.dataset == "musique":
        counts = prepare_musique(args)
    elif args.dataset == "quality":
        counts = prepare_quality(args)
    else:
        raise ValueError(args.dataset)

    summary = {
        "dataset": args.dataset,
        "raw_dir": args.raw_dir,
        "output_dir": args.output_dir,
        "splits": counts,
        "max_source_words": args.max_source_words,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
