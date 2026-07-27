import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from datasets import ClassLabel, load_dataset


SST2_LABELS = ["negative", "positive"]
DEFAULT_BBH_TASKS = [
    "date_understanding",
    "disambiguation_qa",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_three_objects",
]


def normalize_space(text: object) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def class_names_from_feature(feature, fallback: Sequence[str]) -> List[str]:
    if isinstance(feature, ClassLabel):
        return list(feature.names)
    return list(fallback)


def load_sst2() -> tuple[List[Dict], List[Dict], List[str]]:
    dataset = load_dataset("nyu-mll/glue", "sst2")
    labels = class_names_from_feature(dataset["train"].features["label"], SST2_LABELS)
    train = [
        {
            "id": f"sst2-train-{idx}",
            "input": normalize_space(row["sentence"]),
            "label": labels[int(row["label"])],
            "dataset": "sst2",
            "task": "sst2",
        }
        for idx, row in enumerate(dataset["train"])
        if int(row["label"]) >= 0
    ]
    eval_rows = [
        {
            "id": f"sst2-validation-{idx}",
            "input": normalize_space(row["sentence"]),
            "label": labels[int(row["label"])],
            "dataset": "sst2",
            "task": "sst2",
        }
        for idx, row in enumerate(dataset["validation"])
        if int(row["label"]) >= 0
    ]
    return train, eval_rows, labels


def load_trec(label_level: str) -> tuple[List[Dict], List[Dict], List[str]]:
    dataset = load_dataset("SetFit/TREC-QC")
    label_key = "label_text" if label_level == "fine" else "label_coarse_text"
    labels = sorted({normalize_space(row[label_key]) for row in dataset["train"]})

    def convert(split_name: str) -> List[Dict]:
        rows = []
        for idx, row in enumerate(dataset[split_name]):
            rows.append(
                {
                    "id": f"trec-{split_name}-{idx}",
                    "input": normalize_space(row["text"]),
                    "label": normalize_space(row[label_key]),
                    "dataset": "trec",
                    "task": f"trec_{label_level}",
                }
            )
        return rows

    return convert("train"), convert("test"), labels


def extract_bbh_answer(target: object) -> str:
    text = normalize_space(target)
    match = re.search(r"(?:the answer is|answer:)\s*([A-Za-z0-9_\- ,.'()]+)", text, flags=re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    return text.strip(" .")


def load_one_bbh_task(task: str) -> tuple[List[Dict], List[Dict], List[str]]:
    dataset = load_dataset("lukaemon/bbh", task)
    split_name = "train" if "train" in dataset else "test"
    rows = []
    for idx, row in enumerate(dataset[split_name]):
        label = extract_bbh_answer(row.get("target", row.get("answer", "")))
        query = row.get("input", row.get("question", ""))
        if not label or not query:
            continue
        rows.append(
            {
                "id": f"bbh-{task}-{idx}",
                "input": normalize_space(query),
                "label": label,
                "dataset": "bbh",
                "task": task,
            }
        )
    labels = sorted({row["label"] for row in rows})
    return rows, rows, labels


def sample_rows(rows: List[Dict], size: int, rng: random.Random) -> List[Dict]:
    rows = list(rows)
    rng.shuffle(rows)
    if size <= 0 or size >= len(rows):
        return rows
    return rows[:size]


def split_bbh_train_eval(rows: List[Dict], train_size: int, eval_size: int, rng: random.Random) -> tuple[List[Dict], List[Dict]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    if train_size <= 0:
        train_size = max(1, int(0.7 * len(shuffled)))
    train_rows = shuffled[: min(train_size, len(shuffled))]
    remaining = shuffled[len(train_rows) :]
    if not remaining:
        remaining = shuffled
    eval_rows = sample_rows(remaining, eval_size, rng)
    return train_rows, eval_rows


def rows_by_label(rows: Iterable[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    return grouped


def sample_support(
    pool: List[Dict],
    query_id: str,
    labels: Sequence[str],
    shots: int,
    rng: random.Random,
) -> List[Dict]:
    if shots <= 0:
        return []

    grouped = rows_by_label(row for row in pool if row["id"] != query_id)
    support: List[Dict] = []
    seen_ids = set()
    label_cycle = list(labels)
    rng.shuffle(label_cycle)

    shuffled_buckets = {}
    for label in label_cycle:
        candidates = list(grouped.get(label, []))
        rng.shuffle(candidates)
        shuffled_buckets[label] = candidates

    while len(support) < shots and label_cycle:
        added_this_round = False
        for label in label_cycle:
            candidates = shuffled_buckets.get(label, [])
            while candidates and candidates[-1]["id"] in seen_ids:
                candidates.pop()
            if not candidates:
                continue
            sampled = candidates.pop()
            support.append(sampled)
            seen_ids.add(sampled["id"])
            added_this_round = True
            if len(support) >= shots:
                break
        if not added_this_round:
            break

    if len(support) < shots:
        candidates = [row for row in pool if row["id"] != query_id]
        rng.shuffle(candidates)
        for row in candidates:
            if row["id"] not in seen_ids:
                support.append(row)
                seen_ids.add(row["id"])
            if len(support) >= shots:
                break

    return support[:shots]


def build_source_text(support: Sequence[Dict], labels: Sequence[str], task_name: str) -> str:
    lines = [
        "Few-shot classification support set.",
        f"Task: {task_name}",
        "Allowed labels: " + ", ".join(labels),
    ]
    for idx, row in enumerate(support, start=1):
        lines.extend(
            [
                "",
                f"Example {idx}",
                f"Input: {row['input']}",
                f"Label: {row['label']}",
            ]
        )
    return "\n".join(lines)


def build_prompt(query: str, labels: Sequence[str]) -> str:
    return (
        "Classify the input using the compressed support examples.\n"
        "Choose exactly one label from: "
        + ", ".join(labels)
        + "\n"
        f"Input: {query}\n"
        "Label:"
    )


def convert_to_flowcomp_rows(
    rows: Sequence[Dict],
    support_pool: List[Dict],
    labels: Sequence[str],
    shots: int,
    seed: int,
) -> List[Dict]:
    converted = []
    for index, row in enumerate(rows):
        rng = random.Random(seed + index)
        support = sample_support(support_pool, row["id"], labels, shots, rng)
        converted.append(
            {
                "id": row["id"],
                "dataset": row["dataset"],
                "task": row["task"],
                "text": build_source_text(support, labels, row["task"]),
                "prompt": build_prompt(row["input"], labels),
                "answer": row["label"],
                "label": row["label"],
                "labels": list(labels),
                "query": row["input"],
                "shots": len(support),
            }
        )
    return converted


def prepare_standard_dataset(args: argparse.Namespace, dataset_name: str) -> Dict[str, int | str]:
    if dataset_name == "sst2":
        train_raw, eval_raw, labels = load_sst2()
    elif dataset_name == "trec":
        train_raw, eval_raw, labels = load_trec(args.trec_label_level)
    else:
        raise ValueError(f"Unsupported standard dataset: {dataset_name}")

    rng = random.Random(args.seed)
    train_raw = sample_rows(train_raw, args.train_size, rng)
    eval_raw = sample_rows(eval_raw, args.eval_size, rng)

    train_rows = convert_to_flowcomp_rows(train_raw, train_raw, labels, args.shots, args.seed + 1000)
    eval_rows = convert_to_flowcomp_rows(eval_raw, train_raw, labels, args.shots, args.seed + 2000)

    out_dir = Path(args.output_dir) / dataset_name
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "eval.jsonl", eval_rows)
    return {
        "dataset": dataset_name,
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "num_labels": len(labels),
        "output_dir": str(out_dir),
    }


def prepare_bbh(args: argparse.Namespace) -> Dict[str, int | str]:
    task_names = DEFAULT_BBH_TASKS if args.bbh_tasks == "default" else [task.strip() for task in args.bbh_tasks.split(",") if task.strip()]
    all_train_rows: List[Dict] = []
    all_eval_rows: List[Dict] = []
    rng = random.Random(args.seed)

    for task in task_names:
        rows, _, labels = load_one_bbh_task(task)
        if len(labels) < 2:
            continue
        train_raw, eval_raw = split_bbh_train_eval(rows, args.train_size_per_bbh_task, args.eval_size_per_bbh_task, rng)
        all_train_rows.extend(convert_to_flowcomp_rows(train_raw, train_raw, labels, args.shots, args.seed + len(all_train_rows)))
        all_eval_rows.extend(convert_to_flowcomp_rows(eval_raw, train_raw, labels, args.shots, args.seed + 10000 + len(all_eval_rows)))

    if not all_train_rows or not all_eval_rows:
        raise ValueError("BBH preprocessing produced an empty split. Try different --bbh_tasks.")

    out_dir = Path(args.output_dir) / "bbh"
    write_jsonl(out_dir / "train.jsonl", all_train_rows)
    write_jsonl(out_dir / "eval.jsonl", all_eval_rows)
    return {
        "dataset": "bbh",
        "tasks": len(task_names),
        "train_size": len(all_train_rows),
        "eval_size": len(all_eval_rows),
        "output_dir": str(out_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare few-shot classification data for FlowComp ICL experiments.")
    parser.add_argument("--dataset", choices=["sst2", "trec", "bbh", "all"], default="all")
    default_output_dir = Path(__file__).resolve().parents[1] / "data/icl/processed"
    parser.add_argument("--output_dir", default=str(default_output_dir))
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--train_size", type=int, default=2000)
    parser.add_argument("--eval_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--trec_label_level", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--bbh_tasks", default="default")
    parser.add_argument("--train_size_per_bbh_task", type=int, default=120)
    parser.add_argument("--eval_size_per_bbh_task", type=int, default=80)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    datasets = ["sst2", "trec", "bbh"] if args.dataset == "all" else [args.dataset]
    summaries = []
    for dataset_name in datasets:
        if dataset_name == "bbh":
            summaries.append(prepare_bbh(args))
        else:
            summaries.append(prepare_standard_dataset(args, dataset_name))

    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summaries, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
