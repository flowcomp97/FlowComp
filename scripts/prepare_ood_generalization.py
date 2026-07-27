import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import _bootstrap  # noqa: F401
from datasets import Dataset, DatasetDict, load_dataset


DATASET_SPECS = {
    "pubmedqa": {
        "domain": "biomedical",
        "flowcomp_subdir": "pubmedqa",
        "sac_subdir": "pubmedqa_mrqa_like",
    },
    "cuad": {
        "domain": "legal",
        "flowcomp_subdir": "cuad",
        "sac_subdir": "cuad_mrqa_like",
    },
    "legal_rag_qa": {
        "domain": "legal",
        "flowcomp_subdir": "legal_rag_qa",
        "sac_subdir": "legal_rag_qa_mrqa_like",
    },
    "casehold": {
        "domain": "legal",
        "flowcomp_subdir": "casehold",
        "sac_subdir": "casehold_mrqa_like",
    },
}


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split()).strip()
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


def truncate_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def first_answer(value: Any) -> str:
    if isinstance(value, dict):
        texts = value.get("text") or value.get("texts") or value.get("answer") or value.get("answers")
        return first_answer(texts)
    if isinstance(value, list):
        for item in value:
            text = first_answer(item)
            if text:
                return text
        return ""
    return coerce_text(value)


def load_pubmedqa() -> DatasetDict:
    attempts = [
        ("qiaojin/PubMedQA", "pqa_labeled"),
        ("pubmed_qa", "pqa_labeled"),
        ("bigbio/pubmed_qa", "pubmed_qa_source"),
        ("bigbio/pubmed_qa", "pubmed_qa_bigbio_qa"),
    ]
    errors = []
    for repo, config in attempts:
        try:
            data = load_dataset(repo, config)
            if isinstance(data, DatasetDict):
                return data
            return DatasetDict({"train": data})
        except Exception as exc:  # noqa: BLE001 - keep trying known mirrors/configs.
            errors.append(f"{repo}/{config}: {exc}")
    raise RuntimeError("Could not load PubMedQA from known Hugging Face sources:\n" + "\n".join(errors))



def load_legal_rag_qa() -> DatasetDict:
    corpus = load_dataset("isaacus/legal-rag-qa", "corpus", split="test")
    qa = load_dataset("isaacus/legal-rag-qa", "qa", split="test")
    corpus_by_id = {str(row["id"]): dict(row) for row in corpus}
    rows = []
    for row in qa:
        item = dict(row)
        passages = []
        for passage_id in item.get("relevant_passages") or []:
            passage = corpus_by_id.get(str(passage_id))
            if passage:
                title = coerce_text(passage.get("title"))
                text = coerce_text(passage.get("text"))
                passages.append(f"Title: {title}\n{text}" if title else text)
        item["context"] = "\n\n".join(part for part in passages if part)
        rows.append(item)
    return DatasetDict({"test": Dataset.from_list(rows)})


def load_casehold() -> DatasetDict:
    attempts = [("Gshgrapes/lex_glue", "case_hold"), ("lex_glue", "case_hold")]
    errors = []
    for repo, config in attempts:
        try:
            data = load_dataset(repo, config)
            if isinstance(data, DatasetDict):
                return data
            return DatasetDict({"train": data})
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{repo}/{config}: {exc}")
    raise RuntimeError("Could not load CaseHOLD from known Hugging Face sources:\n" + "\n".join(errors))


def load_cuad() -> DatasetDict:
    errors = []
    attempts = [
        ("theatticusproject/cuad-qa", {}),
        ("theatticusproject/cuad", {"verification_mode": "no_checks"}),
    ]
    for repo, kwargs in attempts:
        try:
            data = load_dataset(repo, **kwargs)
            if isinstance(data, DatasetDict):
                return data
            return DatasetDict({"train": data})
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{repo}: {exc}")
    raise RuntimeError("Could not load CUAD from known Hugging Face sources:\n" + "\n".join(errors))


def pubmed_context(row: Dict[str, Any]) -> str:
    context = row.get("context", "")
    if isinstance(context, dict):
        labels = context.get("labels") or []
        contexts = context.get("contexts") or context.get("text") or []
        if isinstance(contexts, str):
            contexts = [contexts]
        parts = []
        for idx, item in enumerate(contexts):
            item_text = coerce_text(item)
            if not item_text:
                continue
            label = coerce_text(labels[idx]) if idx < len(labels) else "Abstract"
            parts.append(f"{label}: {item_text}")
        if parts:
            return "\n".join(parts)
    text = coerce_text(context)
    if text:
        return text
    for key in ("abstract", "contexts", "passage", "text"):
        text = coerce_text(row.get(key))
        if text:
            return text
    return ""


def convert_pubmed_row(row: Dict[str, Any], idx: int, max_source_words: int) -> Tuple[Dict[str, Any], Dict[str, Any]] | None:
    question = coerce_text(row.get("question"))
    answer = coerce_text(row.get("final_decision") or row.get("answer") or row.get("label"))
    if answer.lower() not in {"yes", "no", "maybe"}:
        answer = coerce_text(row.get("final_decision"))
    context = truncate_words(pubmed_context(row), max_source_words)
    if not question or not answer or not context:
        return None
    example_id = coerce_text(row.get("pubid") or row.get("id") or idx)
    prompt = (
        "Answer the biomedical question using the PubMed abstract. "
        "Respond with exactly one of: yes, no, maybe.\n"
        f"Question: {question}\nAnswer:"
    )
    flowcomp = {
        "id": example_id,
        "dataset": "pubmedqa",
        "domain": "biomedical",
        "text": context,
        "prompt": prompt,
        "answer": answer.lower(),
        "question": question,
    }
    sac = {
        "id": example_id,
        "subset": "PubMedQA",
        "context": context,
        "question": prompt,
        "answers": [answer.lower()],
    }
    return flowcomp, sac


def cuad_answer(row: Dict[str, Any]) -> str:
    answer = first_answer(row.get("answers"))
    if answer:
        return answer
    for key in ("answer", "clause", "text"):
        answer = coerce_text(row.get(key))
        if answer:
            return answer
    return "not mentioned"


def convert_cuad_row(row: Dict[str, Any], idx: int, max_source_words: int) -> Tuple[Dict[str, Any], Dict[str, Any]] | None:
    context = coerce_text(row.get("context") or row.get("contract") or row.get("document") or row.get("clause_with_context"))
    question = coerce_text(row.get("question"))
    if not question:
        label = coerce_text(row.get("label"))
        question = f"Which clause in the contract corresponds to {label}?" if label else "Which contract clause answers the legal review question?"
    context = truncate_words(context, max_source_words)
    answer = cuad_answer(row)
    if not context or not question:
        return None
    example_id = coerce_text(row.get("id") or row.get("title") or row.get("contract_name") or idx)
    prompt = (
        "Answer the legal contract review question using the contract text. "
        "Extract the relevant clause span. If no clause is present, answer 'not mentioned'.\n"
        f"Question: {question}\nAnswer:"
    )
    flowcomp = {
        "id": example_id,
        "dataset": "cuad",
        "domain": "legal",
        "text": context,
        "prompt": prompt,
        "answer": answer,
        "question": question,
    }
    sac = {
        "id": example_id,
        "subset": "CUAD",
        "context": context,
        "question": prompt,
        "answers": [answer],
    }
    return flowcomp, sac


def convert_legal_rag_qa_row(row: Dict[str, Any], idx: int, max_source_words: int) -> Tuple[Dict[str, Any], Dict[str, Any]] | None:
    context = truncate_words(coerce_text(row.get("context")), max_source_words)
    question = coerce_text(row.get("question"))
    answer = coerce_text(row.get("answer"))
    if not context or not question or not answer:
        return None
    example_id = coerce_text(row.get("id") or idx)
    prompt = (
        "Answer the legal question using the provided legal passage. "
        "Be concise and rely only on the passage.\n"
        f"Question: {question}\nAnswer:"
    )
    flowcomp = {
        "id": example_id,
        "dataset": "legal_rag_qa",
        "domain": "legal",
        "text": context,
        "prompt": prompt,
        "answer": answer,
        "question": question,
    }
    sac = {
        "id": example_id,
        "subset": "LegalRAGQA",
        "context": context,
        "question": prompt,
        "answers": [answer],
    }
    return flowcomp, sac


def convert_casehold_row(row: Dict[str, Any], idx: int, max_source_words: int) -> Tuple[Dict[str, Any], Dict[str, Any]] | None:
    context = truncate_words(coerce_text(row.get("context")), max_source_words)
    endings = row.get("endings") or row.get("holdings") or []
    if not context or not endings:
        return None
    try:
        label_idx = int(row.get("label"))
    except Exception:
        return None
    if label_idx < 0 or label_idx >= len(endings):
        return None
    letters = ["A", "B", "C", "D", "E", "F", "G"]
    options = []
    for option_idx, ending in enumerate(endings):
        options.append(f"{letters[option_idx]}. {coerce_text(ending)}")
    answer = letters[label_idx]
    question = "Which option best fills the <HOLDING> placeholder in the legal case excerpt?"
    prompt = f"{question}\n" + "\n".join(options[: len(endings)]) + "\nAnswer with the option letter only.\nAnswer:"
    example_id = coerce_text(row.get("id") or idx)
    flowcomp = {
        "id": example_id,
        "dataset": "casehold",
        "domain": "legal",
        "text": context,
        "prompt": prompt,
        "answer": answer,
        "question": question,
        "labels": letters[: len(endings)],
    }
    sac = {
        "id": example_id,
        "subset": "CaseHOLD",
        "context": context,
        "question": prompt,
        "answers": [answer],
    }
    return flowcomp, sac


def split_dataset(data: DatasetDict, train_split: str, eval_split: str, eval_fraction: float, seed: int) -> Tuple[Dataset, Dataset]:
    if train_split in data and eval_split in data:
        return data[train_split], data[eval_split]
    if "train" in data and "test" in data:
        return data["train"], data["test"]
    source = data[train_split] if train_split in data else data[next(iter(data.keys()))]
    if len(source) < 2:
        raise ValueError("Need at least two rows to create train/eval splits.")
    test_size = max(1, int(round(len(source) * eval_fraction)))
    test_size = min(test_size, len(source) - 1)
    split = source.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]


def sample_rows(rows: List[Tuple[Dict[str, Any], Dict[str, Any]]], size: int, seed: int) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    if size <= 0 or len(rows) <= size:
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), size))
    return [rows[index] for index in indices]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def prepare_one(dataset_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    if dataset_name == "pubmedqa":
        raw = load_pubmedqa()
        converter = convert_pubmed_row
        default_max_words = args.pubmedqa_max_source_words
    elif dataset_name == "cuad":
        raw = load_cuad()
        converter = convert_cuad_row
        default_max_words = args.cuad_max_source_words
    elif dataset_name == "legal_rag_qa":
        raw = load_legal_rag_qa()
        converter = convert_legal_rag_qa_row
        default_max_words = args.legal_rag_qa_max_source_words
    elif dataset_name == "casehold":
        raw = load_casehold()
        converter = convert_casehold_row
        default_max_words = args.casehold_max_source_words
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_split, eval_split = split_dataset(raw, args.train_split, args.eval_split, args.eval_fraction, args.seed)
    train_pairs = [item for idx, row in enumerate(train_split) if (item := converter(dict(row), idx, default_max_words))]
    eval_pairs = [item for idx, row in enumerate(eval_split) if (item := converter(dict(row), idx, default_max_words))]
    train_pairs = sample_rows(train_pairs, args.train_size, args.seed + 1)
    eval_pairs = sample_rows(eval_pairs, args.eval_size, args.seed + 2)
    if not train_pairs or not eval_pairs:
        raise ValueError(f"{dataset_name} preprocessing produced an empty split.")

    output_root = Path(args.output_dir)
    flowcomp_dir = output_root / DATASET_SPECS[dataset_name]["flowcomp_subdir"]
    sac_dir = output_root / DATASET_SPECS[dataset_name]["sac_subdir"]
    write_jsonl(flowcomp_dir / "train.jsonl", [pair[0] for pair in train_pairs])
    write_jsonl(flowcomp_dir / "eval.jsonl", [pair[0] for pair in eval_pairs])
    write_jsonl(sac_dir / "train.jsonl", [pair[1] for pair in train_pairs])
    write_jsonl(sac_dir / "test.jsonl", [pair[1] for pair in eval_pairs])
    write_jsonl(sac_dir / "validation.jsonl", [pair[1] for pair in eval_pairs])
    return {
        "dataset": dataset_name,
        "domain": DATASET_SPECS[dataset_name]["domain"],
        "flowcomp_dir": str(flowcomp_dir),
        "sac_dir": str(sac_dir),
        "train_size": len(train_pairs),
        "eval_size": len(eval_pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OOD-domain QA data for FlowComp and SAC.")
    parser.add_argument("--dataset", choices=["all", "legal_new", *sorted(DATASET_SPECS)], default="all")
    default_output_dir = Path(__file__).resolve().parents[1] / "data/ood_generalization"
    parser.add_argument("--output_dir", default=str(default_output_dir))
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--eval_size", type=int, default=500)
    parser.add_argument("--eval_fraction", type=float, default=0.2)
    parser.add_argument("--pubmedqa_max_source_words", type=int, default=512)
    parser.add_argument("--cuad_max_source_words", type=int, default=1536)
    parser.add_argument("--legal_rag_qa_max_source_words", type=int, default=1536)
    parser.add_argument("--casehold_max_source_words", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset == "all":
        names = sorted(DATASET_SPECS)
    elif args.dataset == "legal_new":
        names = ["legal_rag_qa", "casehold"]
    else:
        names = [args.dataset]
    summaries = [prepare_one(name, args) for name in names]
    summary_path = Path(args.output_dir) / "prepare_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
