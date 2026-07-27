#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

import _bootstrap  # noqa: F401
import torch

METHOD_DIRS = {
    "icae": "baselines/icae_sac_branch",
    "sac": "baselines/sac",
}

METHOD_ORDER = ["full_context", "llmlingua2", "icae", "sac", "flowcomp_t8", "flowcomp_t4"]


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


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def elapsed(fn: Callable[[], Any]) -> tuple[Any, float]:
    cuda_sync()
    start = time.perf_counter()
    out = fn()
    cuda_sync()
    return out, time.perf_counter() - start


def cleanup(obj: Any = None) -> None:
    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def stats_ms(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "std_ms": 0.0, "p50_ms": 0.0}
    vals = sorted(v * 1000.0 for v in values)
    return {
        "mean_ms": mean(vals),
        "std_ms": stdev(vals) if len(vals) > 1 else 0.0,
        "p50_ms": vals[len(vals) // 2],
    }


def get_text(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("context") or row.get("input") or "")


def get_prompt(row: dict[str, Any]) -> str:
    return str(row.get("prompt") or row.get("question") or "Answer:")


def timed_target_prefill(model, attention_mask=None, input_ids=None, inputs_embeds=None, position_ids=None, past_key_values=None):
    def call():
        with torch.no_grad():
            return model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
    return elapsed(call)


def benchmark_full_context(args, rows, device) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.target_model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map=f"cuda:{args.rank}" if str(device).startswith("cuda") else None,
    )
    model.eval()
    prefill_times = []
    token_counts = []
    for idx, row in enumerate(rows):
        text = get_text(row)
        prompt = get_prompt(row)
        full_prompt = f"### Context:\n{text}\n\n{prompt}"
        enc = tokenizer(full_prompt, truncation=True, max_length=args.model_max_length, return_tensors="pt", add_special_tokens=True).to(device)
        token_counts.append(int(enc["input_ids"].shape[1]))
        for r in range(args.warmup + args.repeats):
            _, t = timed_target_prefill(model, input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))
            if r >= args.warmup:
                prefill_times.append(t)
    cleanup(model)
    s = stats_ms(prefill_times)
    return {
        "method": "Full Context",
        "status": "done",
        "compression_ratio": 1,
        "avg_input_tokens": mean(token_counts) if token_counts else 0,
        "avg_compressed_tokens": mean(token_counts) if token_counts else 0,
        "compress_time_ms": 0.0,
        "prefill_time_ms": s["mean_ms"],
        "total_latency_ms": s["mean_ms"],
        "prefill_std_ms": s["std_ms"],
        "num_examples": len(rows),
    }


def benchmark_llmlingua2(args, rows, device) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.target_model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map=f"cuda:{args.rank}" if str(device).startswith("cuda") else None,
    )
    model.eval()
    compressor = None
    note = ""
    try:
        from llmlingua import PromptCompressor  # type: ignore
        compressor = PromptCompressor(model_name=args.llmlingua2_model_name, use_llmlingua2=True, device_map=str(device))
        note = "llmlingua PromptCompressor(use_llmlingua2=True)"
    except Exception as exc:
        if not args.llmlingua2_truncation_fallback:
            cleanup(model)
            return {"method": "LLMLingua-2", "status": "unavailable", "note": f"llmlingua unavailable: {exc}"}
        note = f"TRUNCATION FALLBACK, llmlingua unavailable: {exc}"

    compress_times, prefill_times, src_counts, cmp_counts = [], [], [], []
    for row in rows:
        text = get_text(row)
        prompt = get_prompt(row)
        full_ids = tokenizer(text, add_special_tokens=False)["input_ids"][: args.model_max_length]
        target_context_tokens = max(1, len(full_ids) // args.compression_ratio)
        src_counts.append(len(full_ids))
        for r in range(args.warmup + args.repeats):
            def compress_call():
                if compressor is not None:
                    result = compressor.compress_prompt(text, rate=1.0 / args.compression_ratio, force_tokens=["\n", "."])
                    if isinstance(result, dict):
                        return result.get("compressed_prompt") or result.get("compressed_text") or str(result)
                    return str(result)
                ids = full_ids[:target_context_tokens]
                return tokenizer.decode(ids, skip_special_tokens=True)
            compressed_text, ct = elapsed(compress_call)
            enc = tokenizer(f"### Context:\n{compressed_text}\n\n{prompt}", truncation=True, max_length=args.model_max_length, return_tensors="pt", add_special_tokens=True).to(device)
            _, pt = timed_target_prefill(model, input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))
            if r >= args.warmup:
                compress_times.append(ct)
                prefill_times.append(pt)
                cmp_counts.append(int(enc["input_ids"].shape[1]))
    cleanup(model)
    cs, ps = stats_ms(compress_times), stats_ms(prefill_times)
    return {
        "method": "LLMLingua-2",
        "status": "done",
        "note": note,
        "compression_ratio": args.compression_ratio,
        "avg_input_tokens": mean(src_counts) if src_counts else 0,
        "avg_compressed_tokens": mean(cmp_counts) if cmp_counts else 0,
        "compress_time_ms": cs["mean_ms"],
        "prefill_time_ms": ps["mean_ms"],
        "total_latency_ms": cs["mean_ms"] + ps["mean_ms"],
        "compress_std_ms": cs["std_ms"],
        "prefill_std_ms": ps["std_ms"],
        "num_examples": len(rows),
    }


def latest_run(base: Path) -> Path | None:
    marker = base / "LATEST_RUN"
    if marker.exists():
        p = Path(marker.read_text(encoding="utf-8").strip())
        if p.exists():
            return p
    runs = [p for p in base.iterdir() if p.is_dir()] if base.exists() else []
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def flowcomp_ckpt(root: Path, method: str, dataset: str, soft: int) -> Path:
    ratio = 512 // soft
    base = root / "outputs" / "compression_sweep" / method / dataset / f"{soft}tok_{ratio}x"
    run = latest_run(base)
    if not run or not (run / "pytorch_model.bin").exists():
        raise FileNotFoundError(f"Missing FlowComp 32x checkpoint under {base}")
    return run / "pytorch_model.bin"


def benchmark_flowcomp(args, rows, device, sampling_steps: int) -> dict[str, Any]:
    from flowcomp import FlowComp, ModelArguments, TrainingArguments
    from flowcomp.paths import output_path
    soft = 512 // args.compression_ratio
    ckpt = Path(args.flowcomp_checkpoint) if args.flowcomp_checkpoint else flowcomp_ckpt(Path(args.root_dir), "flowcomp", args.dataset, soft)
    model_args = ModelArguments(
        source_encoder_name_or_path=args.source_encoder_name_or_path,
        target_model_name_or_path=args.target_model_name_or_path,
        target_model_device_map="none",
        num_soft_tokens=soft,
        latent_dim=args.latent_dim,
        sampling_steps=sampling_steps,
        denoiser_layers=args.denoiser_layers,
        decoder_layers=args.decoder_layers,
        num_attention_heads=args.num_attention_heads,
        freeze_target_model=True,
        source_encoder_trainable=False,
    )
    training_args = TrainingArguments(
        output_dir=str(output_path("scratch", "latency")),
        bf16=args.bf16,
        model_max_length=args.model_max_length,
        report_to=[],
    )
    model = FlowComp(model_args, training_args)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    compress_times, prefill_times, src_counts, cmp_counts = [], [], [], []
    for row in rows:
        encoder = model.source_tokenizer(get_text(row), truncation=True, max_length=args.model_max_length, return_tensors="pt")
        prompt = model.target_tokenizer(get_prompt(row), truncation=True, max_length=args.model_max_length, return_tensors="pt", add_special_tokens=False)
        enc_ids = encoder["input_ids"].to(device)
        enc_mask = encoder["attention_mask"].to(device)
        prompt_ids = prompt["input_ids"].to(device)
        prompt_mask = prompt["attention_mask"].to(device)
        src_counts.append(int(enc_ids.shape[1]))
        cmp_counts.append(soft + int(prompt_ids.shape[1]))
        for r in range(args.warmup + args.repeats):
            soft_tokens, ct = elapsed(lambda: model.compress(enc_ids, enc_mask))
            inputs_embeds = model.build_prompt_inputs(soft_tokens, prompt_ids)
            attn = torch.cat([torch.ones((1, soft), dtype=prompt_mask.dtype, device=device), prompt_mask], dim=1)
            _, pt = timed_target_prefill(model.target_model, inputs_embeds=inputs_embeds, attention_mask=attn)
            if r >= args.warmup:
                compress_times.append(ct)
                prefill_times.append(pt)
    cleanup(model)
    cs, ps = stats_ms(compress_times), stats_ms(prefill_times)
    return {
        "method": f"FlowComp T={sampling_steps}",
        "status": "done",
        "checkpoint": str(ckpt),
        "compression_ratio": args.compression_ratio,
        "avg_input_tokens": mean(src_counts) if src_counts else 0,
        "avg_compressed_tokens": mean(cmp_counts) if cmp_counts else 0,
        "compress_time_ms": cs["mean_ms"],
        "prefill_time_ms": ps["mean_ms"],
        "total_latency_ms": cs["mean_ms"] + ps["mean_ms"],
        "compress_std_ms": cs["std_ms"],
        "prefill_std_ms": ps["std_ms"],
        "num_examples": len(rows),
    }


def import_baseline_model(method_dir: Path):
    for name in list(sys.modules):
        if name == "model" or name.startswith("model.") or name == "path_config":
            del sys.modules[name]
    sys.path.insert(0, str(method_dir))
    sys.path.insert(0, str(method_dir.parent / "sac" / ".deps" / "transformers443"))
    module = importlib.import_module("model.modeling")
    return module.get_model, module.load_adapter


def resolve_baseline_work_dir(args, method: str) -> Path:
    root = Path(args.root_dir)
    if method == "sac":
        override = args.sac_work_dir
        default = root / "baselines" / "sac" / "experiment" / f"compression_sweep_{args.dataset}_32tok_16x"
    else:
        override = args.icae_work_dir
        default = root / "baselines" / "icae_sac_branch" / "experiment" / f"main_table_{args.dataset}"
    return Path(override) if override else default


def benchmark_soft_baseline(args, rows, device, method: str) -> dict[str, Any]:
    root = Path(args.root_dir).resolve()
    method_dir = root / METHOD_DIRS[method]
    work_dir = resolve_baseline_work_dir(args, method)
    output_dir = work_dir / "output"
    config_path = output_dir / "config.json"
    if not config_path.exists():
        config_path = work_dir / "config.json"
    if not config_path.exists():
        return {"method": method.upper(), "status": "unavailable", "note": f"missing config: {config_path}"}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training_config = config["sft_training_config"]
    task_config = dict(config["sft_task_config"])
    get_model, load_adapter = import_baseline_model(method_dir)
    rank = 0 if str(device).startswith("cuda") else -1
    if rank < 0:
        return {"method": method.upper(), "status": "unavailable", "note": "baseline loader expects CUDA"}
    model = get_model(training_config["model_id"], task_config, rank)
    adapter_path = output_dir / "instruction_adapter.pt"
    if not adapter_path.exists():
        adapter_path = output_dir / "adapter.pt"
    note = ""
    if adapter_path.exists():
        try:
            model = load_adapter(model, save_path_and_name=str(adapter_path), log=True)
        except Exception as exc:
            note = f"adapter load failed, using initialized adapter: {exc}"
    else:
        note = f"adapter missing under {output_dir}; using initialized adapter"
    model.eval()
    compress_times, prefill_times, src_counts, cmp_counts = [], [], [], []
    for row in rows:
        context_ids = model.tokenizer(get_text(row), add_special_tokens=False)["input_ids"][: args.model_max_length]
        prefix_ids = model.tokenizer("### Context:\n", add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([[model.tokenizer.bos_token_id] + prefix_ids + context_ids], dtype=torch.long, device=device)
        prompt_ids = model.tokenizer(get_prompt(row), add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        src_counts.append(int(input_ids.shape[1]))
        for r in range(args.warmup + args.repeats):
            compressed, ct = elapsed(lambda: model.compress({"input_ids": input_ids}))
            lm_emb = model.decoder.model.embed_tokens(prompt_ids)
            expand_lm_token = model.special_tokens[1:2].unsqueeze(0).expand(1, 1, lm_emb.size(-1))
            position_ids = None
            past = None
            if method == "sac":
                compress_token_ids, compress_token, end_idx, _encoder_hidden_states, encoder_past_key_values, _encoder_mem_size = compressed
                inputs_embeds = torch.cat([expand_lm_token, lm_emb], dim=1)
                past = encoder_past_key_values
                if task_config.get("use_pe"):
                    position_ids = torch.arange(end_idx, end_idx + inputs_embeds.size(1), device=device).unsqueeze(0)
                compressed_length = int(past[0][0].shape[2]) + int(inputs_embeds.shape[1])
            elif method == "icae":
                compress_token_ids, compress_token, end_idx = compressed
                inputs_embeds = torch.cat([compress_token, expand_lm_token, lm_emb], dim=1)
                if task_config.get("use_pe"):
                    latter_position_ids = torch.arange(end_idx, end_idx + lm_emb.size(1) + 1, device=device).unsqueeze(0)
                    position_ids = torch.cat([compress_token_ids, latter_position_ids], dim=1)
                compressed_length = int(inputs_embeds.shape[1])
            else:
                raise ValueError(f"Unsupported soft baseline method: {method}")
            _, pt = timed_target_prefill(model.decoder, inputs_embeds=inputs_embeds, position_ids=position_ids, past_key_values=past)
            if r >= args.warmup:
                compress_times.append(ct)
                prefill_times.append(pt)
                cmp_counts.append(compressed_length)
    cleanup(model)
    cs, ps = stats_ms(compress_times), stats_ms(prefill_times)
    label = "ICAE" if method == "icae" else "SAC"
    return {
        "method": label,
        "status": "done",
        "note": note,
        "work_dir": str(work_dir),
        "config_mem_size": task_config.get("mem_size"),
        "config_compress_ratio": task_config.get("compress_ratio"),
        "compression_ratio": args.compression_ratio,
        "avg_input_tokens": mean(src_counts) if src_counts else 0,
        "avg_compressed_tokens": mean(cmp_counts) if cmp_counts else 0,
        "compress_time_ms": cs["mean_ms"],
        "prefill_time_ms": ps["mean_ms"],
        "total_latency_ms": cs["mean_ms"] + ps["mean_ms"],
        "compress_std_ms": cs["std_ms"],
        "prefill_std_ms": ps["std_ms"],
        "num_examples": len(rows),
    }


def add_speedups(rows: list[dict[str, Any]]) -> None:
    full = next((r for r in rows if r.get("method") == "Full Context" and r.get("status") == "done"), None)
    base = full.get("total_latency_ms") if full else None
    if not isinstance(base, (int, float)) or base <= 0:
        return
    for r in rows:
        total = r.get("total_latency_ms")
        if isinstance(total, (int, float)) and total > 0:
            r["speedup_vs_full"] = base / total


def write_outputs(rows: list[dict[str, Any]], args) -> None:
    add_speedups(rows)
    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "compression_ratio": args.compression_ratio,
        "max_examples": args.max_examples,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "| Method | Status | Compress ms | Prefill ms | Total ms | Speedup | Input tok | Compressed tok | Note |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        def f(x): return f"{x:.2f}" if isinstance(x, (int, float)) else ""
        lines.append(
            f"| {r.get('method')} | {r.get('status')} | {f(r.get('compress_time_ms'))} | "
            f"{f(r.get('prefill_time_ms'))} | {f(r.get('total_latency_ms'))} | {f(r.get('speedup_vs_full'))} | "
            f"{f(r.get('avg_input_tokens'))} | {f(r.get('avg_compressed_tokens'))} | {str(r.get('note','')).replace('|','/')} |"
        )
    md = "\n".join(lines) + "\n"
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Saved {out}")
    print(f"Saved {out.with_suffix('.md')}")


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end latency benchmark at 16x compression.")
    ap.add_argument("--root_dir", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--dataset", default="triviaqa_rc")
    ap.add_argument("--eval_file", default="")
    ap.add_argument("--output_file", default="outputs/latency_16x/triviaqa_rc_latency.json")
    ap.add_argument("--methods", default=",".join(METHOD_ORDER))
    ap.add_argument("--max_examples", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--compression_ratio", type=int, default=16)
    ap.add_argument("--model_max_length", type=int, default=512)
    ap.add_argument("--target_model_name_or_path", default=os.environ.get("FLOWCOMP_TARGET_MODEL", "models/Meta-Llama-3-8B-Instruct"))
    ap.add_argument("--source_encoder_name_or_path", default="google-t5/t5-small")
    ap.add_argument("--latent_dim", type=int, default=1024)
    ap.add_argument("--denoiser_layers", type=int, default=4)
    ap.add_argument("--decoder_layers", type=int, default=2)
    ap.add_argument("--num_attention_heads", type=int, default=8)
    ap.add_argument("--flowcomp_checkpoint", default="")
    ap.add_argument("--sac_work_dir", default="")
    ap.add_argument("--icae_work_dir", default="")
    ap.add_argument("--llmlingua2_model_name", default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
    ap.add_argument("--llmlingua2_truncation_fallback", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--rank", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root_dir).resolve()
    if not args.eval_file:
        candidate = root / "data" / "odqa" / "processed" / "triviaqa_rc_openqa" / "eval.jsonl"
        if not candidate.exists():
            candidate = root / "data" / "main_table" / "baseline_mrqa_like" / args.dataset / "eval.jsonl"
        args.eval_file = str(candidate)
    rows = load_jsonl(Path(args.eval_file), args.max_examples)
    device = torch.device(f"cuda:{args.rank}" if torch.cuda.is_available() else "cpu")

    out_path = Path(args.output_file)
    results = []
    if out_path.exists():
        try:
            existing_payload = json.loads(out_path.read_text(encoding="utf-8"))
            results = [r for r in existing_payload.get("rows", []) if r.get("method")]
        except Exception:
            results = []
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        print(f"=== latency benchmark: {method} ===", flush=True)
        try:
            if method == "full_context":
                row = benchmark_full_context(args, rows, device)
            elif method == "llmlingua2":
                row = benchmark_llmlingua2(args, rows, device)
            elif method == "icae":
                row = benchmark_soft_baseline(args, rows, device, "icae")
            elif method == "sac":
                row = benchmark_soft_baseline(args, rows, device, "sac")
            elif method == "flowcomp_t8":
                row = benchmark_flowcomp(args, rows, device, 8)
            elif method == "flowcomp_t4":
                row = benchmark_flowcomp(args, rows, device, 4)
            else:
                row = {"method": method, "status": "unknown"}
        except Exception as exc:
            labels = {
                "full_context": "Full Context",
                "llmlingua2": "LLMLingua-2",
                "icae": "ICAE",
                "sac": "SAC",
                "flowcomp_t8": "FlowComp T=8",
                "flowcomp_t4": "FlowComp T=4",
            }
            row = {"method": labels.get(method, method), "status": "failed", "note": repr(exc)}
            cleanup()
        labels = {
            "full_context": "Full Context",
            "llmlingua2": "LLMLingua-2",
            "icae": "ICAE",
            "sac": "SAC",
            "flowcomp_t8": "FlowComp T=8",
            "flowcomp_t4": "FlowComp T=4",
        }
        label = labels.get(method, row.get("method", method))
        results = [r for r in results if r.get("method") != label and r.get("method") != method]
        results.append(row)
        order = {"Full Context": 0, "LLMLingua-2": 1, "ICAE": 2, "SAC": 3, "FlowComp T=8": 4, "FlowComp T=4": 5}
        results.sort(key=lambda r: order.get(r.get("method"), 99))
        write_outputs(results, args)
    write_outputs(results, args)


if __name__ == "__main__":
    main()
