#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

import _bootstrap  # noqa: F401
import torch

from flowcomp import FlowComp, ModelArguments, TrainingArguments
from flowcomp.paths import output_path


def load_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def latest_run(base: Path) -> Path | None:
    marker = base / "LATEST_RUN"
    if marker.exists():
        p = Path(marker.read_text(encoding="utf-8").strip())
        if p.exists():
            return p
    runs = [p for p in base.iterdir() if p.is_dir()] if base.exists() else []
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def default_checkpoint(root: Path, dataset: str, soft: int) -> Path:
    ratio = 512 // soft
    base = root / "outputs" / "compression_sweep" / "flowcomp" / dataset / f"{soft}tok_{ratio}x"
    run = latest_run(base)
    if not run or not (run / "pytorch_model.bin").exists():
        raise FileNotFoundError(f"Missing FlowComp checkpoint under {base}")
    return run / "pytorch_model.bin"


def get_text(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("context") or row.get("input") or "")


def get_prompt(row: dict[str, Any]) -> str:
    return str(row.get("prompt") or row.get("question") or "Answer:")


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def elapsed(fn: Callable[[], Any]) -> tuple[Any, float]:
    sync()
    start = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - start


def stats_ms(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "std_ms": 0.0, "p50_ms": 0.0}
    vals = sorted(v * 1000.0 for v in values)
    return {
        "mean_ms": mean(vals),
        "std_ms": stdev(vals) if len(vals) > 1 else 0.0,
        "p50_ms": vals[len(vals) // 2],
    }


def timed_prefill(target_model, inputs_embeds, attention_mask):
    def call():
        with torch.no_grad():
            return target_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
    return elapsed(call)


def benchmark_step(args, rows: list[dict[str, Any]], step: int, device: torch.device) -> dict[str, Any]:
    soft = 512 // args.compression_ratio
    ckpt = Path(args.checkpoint_path) if args.checkpoint_path else default_checkpoint(Path(args.root_dir), args.dataset, soft)
    model_args = ModelArguments(
        source_encoder_name_or_path=args.source_encoder_name_or_path,
        target_model_name_or_path=args.target_model_name_or_path,
        target_model_device_map="none",
        num_soft_tokens=soft,
        latent_dim=args.latent_dim,
        sampling_steps=step,
        denoiser_layers=args.denoiser_layers,
        decoder_layers=args.decoder_layers,
        num_attention_heads=args.num_attention_heads,
        freeze_target_model=True,
        source_encoder_trainable=False,
    )
    training_args = TrainingArguments(
        output_dir=str(output_path("scratch", "denoising_latency")),
        bf16=args.bf16,
        model_max_length=args.model_max_length,
        report_to=[],
    )
    model = FlowComp(model_args, training_args)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    compress_times, prefill_times, source_lens, compressed_lens = [], [], [], []
    for row in rows:
        encoder = model.source_tokenizer(
            get_text(row),
            truncation=True,
            max_length=args.model_max_length,
            return_tensors="pt",
        )
        prompt = model.target_tokenizer(
            get_prompt(row),
            truncation=True,
            max_length=args.model_max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        enc_ids = encoder["input_ids"].to(device)
        enc_mask = encoder["attention_mask"].to(device)
        prompt_ids = prompt["input_ids"].to(device)
        prompt_mask = prompt["attention_mask"].to(device)
        source_lens.append(int(enc_ids.shape[1]))
        compressed_lens.append(int(soft + prompt_ids.shape[1]))
        for r in range(args.warmup + args.repeats):
            soft_tokens, ct = elapsed(lambda: model.compress(enc_ids, enc_mask))
            inputs_embeds = model.build_prompt_inputs(soft_tokens, prompt_ids)
            attention_mask = torch.cat(
                [torch.ones((1, soft), dtype=prompt_mask.dtype, device=device), prompt_mask],
                dim=1,
            )
            _, pt = timed_prefill(model.target_model, inputs_embeds, attention_mask)
            if r >= args.warmup:
                compress_times.append(ct)
                prefill_times.append(pt)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    cs = stats_ms(compress_times)
    ps = stats_ms(prefill_times)
    return {
        "sampling_steps": step,
        "status": "done",
        "checkpoint_path": str(ckpt),
        "compression_ratio": args.compression_ratio,
        "num_soft_tokens": soft,
        "avg_source_tokens": mean(source_lens) if source_lens else 0,
        "avg_prefill_tokens": mean(compressed_lens) if compressed_lens else 0,
        "compress_time_ms": cs["mean_ms"],
        "compress_std_ms": cs["std_ms"],
        "prefill_time_ms": ps["mean_ms"],
        "prefill_std_ms": ps["std_ms"],
        "total_latency_ms": cs["mean_ms"] + ps["mean_ms"],
        "num_latency_examples": len(rows),
    }


def write_table(rows: list[dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    lines = [
        "| T | Status | EM | F1 | Compress ms | Prefill ms | Total ms | Speedup vs T=8 | N eval | N latency |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    t8 = next((r for r in rows if r.get("sampling_steps") == 8 and isinstance(r.get("total_latency_ms"), (int, float))), None)
    t8_total = t8.get("total_latency_ms") if t8 else None
    for r in sorted(rows, key=lambda x: int(x.get("sampling_steps", 0))):
        def f(x):
            return f"{x:.4f}" if isinstance(x, (int, float)) and abs(x) <= 1 else f"{x:.2f}" if isinstance(x, (int, float)) else ""
        speed = ""
        if isinstance(t8_total, (int, float)) and isinstance(r.get("total_latency_ms"), (int, float)) and r["total_latency_ms"] > 0:
            speed = f"{t8_total / r['total_latency_ms']:.2f}"
        lines.append(
            f"| {r.get('sampling_steps')} | {r.get('status', '')} | {f(r.get('exact_match'))} | {f(r.get('token_f1'))} | "
            f"{f(r.get('compress_time_ms'))} | {f(r.get('prefill_time_ms'))} | {f(r.get('total_latency_ms'))} | "
            f"{speed} | {r.get('num_eval_examples') or ''} | {r.get('num_latency_examples') or ''} |"
        )
    md = "\n".join(lines) + "\n"
    output_file.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Saved {output_file}")
    print(f"Saved {output_file.with_suffix('.md')}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark FlowComp denoising-step latency.")
    ap.add_argument("--root_dir", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--dataset", default="triviaqa_rc")
    ap.add_argument("--eval_file", default="")
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--steps", default="1,2,4,8,16,32")
    ap.add_argument("--checkpoint_path", default="")
    ap.add_argument("--compression_ratio", type=int, default=16)
    ap.add_argument("--max_examples", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--model_max_length", type=int, default=512)
    ap.add_argument("--target_model_name_or_path", default=os.environ.get("FLOWCOMP_TARGET_MODEL", "models/Meta-Llama-3-8B-Instruct"))
    ap.add_argument("--source_encoder_name_or_path", default="google-t5/t5-small")
    ap.add_argument("--latent_dim", type=int, default=1024)
    ap.add_argument("--denoiser_layers", type=int, default=4)
    ap.add_argument("--decoder_layers", type=int, default=2)
    ap.add_argument("--num_attention_heads", type=int, default=8)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--rank", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root_dir).resolve()
    if not args.eval_file:
        args.eval_file = str(root / "data" / "odqa" / "processed" / "triviaqa_rc_openqa" / "eval.jsonl")
    rows = load_jsonl(Path(args.eval_file), args.max_examples)
    device = torch.device(f"cuda:{args.rank}" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_file)
    existing = json.loads(out.read_text(encoding="utf-8"))["rows"] if out.exists() else []
    by_step = {int(r["sampling_steps"]): r for r in existing if "sampling_steps" in r}

    for step in [int(x) for x in args.steps.split(",") if x.strip()]:
        print(f"=== denoising latency T={step} ===", flush=True)
        try:
            row = benchmark_step(args, rows, step, device)
        except Exception as exc:
            row = {"sampling_steps": step, "status": "failed", "note": repr(exc)}
        if step in by_step:
            merged = dict(by_step[step])
            merged.update(row)
            by_step[step] = merged
        else:
            by_step[step] = row
        write_table(list(by_step.values()), out)


if __name__ == "__main__":
    main()
