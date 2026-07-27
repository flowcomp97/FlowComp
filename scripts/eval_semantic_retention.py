#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import torch


BASELINE_DIRS = {
    'icae': 'baselines/icae_sac_branch',
    '500xcompress': 'baselines/500xcompress',
    'sac': 'baselines/sac',
}


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


def score_prediction(prediction: str, answer: str, f1_threshold: float) -> dict[str, Any]:
    pred = canonicalize_prediction(prediction)
    npred = normalize_text(pred)
    nans = normalize_text(answer)
    exact = npred == nans
    contains = bool(npred) and bool(nans) and (nans in npred or npred in nans)
    f1 = token_f1(pred, answer)
    retained = bool(exact or contains or f1 >= f1_threshold)
    return {
        'prediction': pred,
        'normalized_prediction': npred,
        'normalized_answer': nans,
        'exact_match': exact,
        'contains_match': contains,
        'token_f1': f1,
        'retained': retained,
    }


def load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def latest_checkpoint(output_dir: Path) -> Path:
    marker = output_dir / 'LATEST_RUN'
    if marker.exists():
        run = Path(marker.read_text(encoding='utf-8').strip())
        ckpt = run / 'pytorch_model.bin'
        if ckpt.exists():
            return ckpt
    ckpt = output_dir / 'pytorch_model.bin'
    if ckpt.exists():
        return ckpt
    candidates = sorted(output_dir.glob('*/pytorch_model.bin'), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f'No pytorch_model.bin found under {output_dir}')


def load_flowcomp(args: argparse.Namespace):
    from flowcomp import FlowComp, ModelArguments, TrainingArguments
    from flowcomp.paths import output_path

    ckpt = Path(args.checkpoint_path) if args.checkpoint_path else latest_checkpoint(Path(args.flowcomp_output_dir))
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
        output_dir=str(output_path('scratch', 'semantic_retention_eval')),
        bf16=args.bf16,
        model_max_length=args.model_max_length,
        per_device_eval_batch_size=1,
        report_to=[],
    )
    model = FlowComp(model_args, training_args)
    state_dict = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print(f'Loaded FlowComp checkpoint: {ckpt}')
    return model


def flowcomp_predict(model, row: dict[str, Any], args: argparse.Namespace) -> str:
    device = next(model.parameters()).device
    source_text = row.get(args.source_key) or row.get('context') or row.get('text')
    prompt = row.get(args.prompt_key) or row.get('question') or row['prompt']
    encoder = model.source_tokenizer(
        source_text,
        truncation=True,
        max_length=args.model_max_length,
        padding=False,
        return_tensors='pt',
    )
    prompt_tokens = model.target_tokenizer(
        prompt,
        truncation=True,
        max_length=args.model_max_length,
        padding=False,
        return_tensors='pt',
        add_special_tokens=False,
    )
    generated_ids = model.generate(
        encoder_input_ids=encoder['input_ids'].to(device),
        encoder_attention_mask=encoder['attention_mask'].to(device),
        prompt_ids=prompt_tokens['input_ids'].to(device),
        prompt_attention_mask=prompt_tokens['attention_mask'].to(device),
        max_new_tokens=args.max_new_tokens,
    )
    return model.target_tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


def import_baseline(method_dir: Path):
    for name in list(sys.modules):
        if name == 'model' or name.startswith('model.') or name == 'path_config':
            del sys.modules[name]
    sys.path.insert(0, str(method_dir))
    sys.path.insert(0, str(method_dir.parent / 'sac' / '.deps' / 'transformers443'))
    module = importlib.import_module('model.modeling')
    return module.get_model, module.load_adapter


def load_baseline(method: str, args: argparse.Namespace):
    root = Path(args.root_dir).resolve()
    method_dir = root / BASELINE_DIRS[method]
    work_dir = Path(args.baseline_work_dir) if args.baseline_work_dir else method_dir / 'experiment' / f'main_table_{args.baseline_dataset}'
    config_path = work_dir / 'output' / 'config.json'
    if not config_path.exists():
        config_path = work_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'Missing baseline config: {config_path}')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    task_config = config['sft_task_config']
    model_id = config['sft_training_config']['model_id']
    get_model, load_adapter = import_baseline(method_dir)
    rank = int(args.rank)
    model = get_model(model_id, task_config, rank)
    adapter = Path(args.adapter_path) if args.adapter_path else work_dir / 'output' / 'instruction_adapter.pt'
    if not adapter.exists():
        adapter = work_dir / 'output' / 'adapter.pt'
    if not adapter.exists():
        raise FileNotFoundError(f'Missing adapter for {method}: {adapter}')
    model = load_adapter(model, save_path_and_name=str(adapter), log=True)
    model.eval()
    print(f'Loaded {method} adapter: {adapter}')
    return model


def build_baseline_context_ids(tokenizer, context: str, max_context_tokens: int) -> torch.Tensor:
    prefix = tokenizer('### Context:\n', add_special_tokens=False)['input_ids']
    context_ids = tokenizer(context, add_special_tokens=False)['input_ids'][:max_context_tokens]
    ids = [tokenizer.bos_token_id] + prefix + context_ids
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def baseline_predict(model, row: dict[str, Any], args: argparse.Namespace) -> str:
    context = row.get(args.source_key) or row.get('context') or row.get('text')
    prompt = row.get(args.prompt_key) or row.get('question') or row['prompt']
    device = torch.device(f'cuda:{args.rank}' if torch.cuda.is_available() else 'cpu')
    input_ids = build_baseline_context_ids(model.tokenizer, str(context), args.model_max_length).to(device)
    prompt_ids = model.tokenizer(str(prompt), add_special_tokens=False, return_tensors='pt')['input_ids'].to(device)
    inputs = {'input_ids': input_ids, 'lm_targets': prompt_ids}
    with torch.no_grad():
        ids = model.lm_inference(inputs, generate_num=args.max_new_tokens)
    return model.tokenizer.decode(ids, skip_special_tokens=True)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {'overall': {}, 'by_group': {}}
    def agg(items: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(items)
        if n == 0:
            return {'num_examples': 0}
        return {
            'num_examples': n,
            'retention': sum(1 for x in items if x['retained']) / n,
            'exact_match': sum(1 for x in items if x['exact_match']) / n,
            'contains_match': sum(1 for x in items if x['contains_match']) / n,
            'token_f1': sum(float(x['token_f1']) for x in items) / n,
        }
    summary['overall'] = agg(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[r.get('salience_group', 'unknown')].append(r)
    for group, items in sorted(groups.items()):
        summary['by_group'][group] = agg(items)
    head = summary['by_group'].get('head', {}).get('retention')
    tail = summary['by_group'].get('tail', {}).get('retention')
    if isinstance(head, float) and isinstance(tail, float):
        summary['tail_gap_vs_head'] = tail - head
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description='Evaluate head/tail fact retention from compressed representations.')
    ap.add_argument('--method', required=True, choices=['flowcomp', 'flowcomp_no_wa', 'icae', '500xcompress', 'sac'])
    ap.add_argument('--eval_file', required=True)
    ap.add_argument('--output_file', required=True)
    ap.add_argument('--root_dir', default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--max_examples', type=int, default=0)
    ap.add_argument('--source_key', default='text')
    ap.add_argument('--prompt_key', default='prompt')
    ap.add_argument('--answer_key', default='answer')
    ap.add_argument('--max_new_tokens', type=int, default=16)
    ap.add_argument('--model_max_length', type=int, default=2048)
    ap.add_argument('--f1_threshold', type=float, default=0.50)
    ap.add_argument('--rank', type=int, default=0)

    # FlowComp args.
    ap.add_argument('--checkpoint_path', default='')
    ap.add_argument('--flowcomp_output_dir', default='outputs/triviaqa_rc_llama3')
    ap.add_argument('--source_encoder_name_or_path', default='google-t5/t5-small')
    ap.add_argument('--target_model_name_or_path', default='meta-llama/Meta-Llama-3-8B-Instruct')
    ap.add_argument('--target_model_device_map', default='none')
    ap.add_argument('--latent_generator_type', choices=['flow', 'one_shot'], default='flow')
    ap.add_argument('--distribution_loss_type', choices=['wasserstein', 'mmd'], default='wasserstein')
    ap.add_argument('--soft_token_attention', choices=['bidirectional', 'causal'], default='bidirectional')
    ap.add_argument('--num_soft_tokens', type=int, default=64)
    ap.add_argument('--latent_dim', type=int, default=1024)
    ap.add_argument('--sampling_steps', type=int, default=8)
    ap.add_argument('--denoiser_layers', type=int, default=4)
    ap.add_argument('--decoder_layers', type=int, default=2)
    ap.add_argument('--num_attention_heads', type=int, default=8)
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--use_flash_attention_2', action='store_true')

    # Baseline args.
    ap.add_argument('--baseline_dataset', default='triviaqa_rc')
    ap.add_argument('--baseline_work_dir', default='')
    ap.add_argument('--adapter_path', default='')
    args = ap.parse_args()

    root = Path(args.root_dir).resolve()
    if args.method in {'flowcomp', 'flowcomp_no_wa'}:
        if args.method == 'flowcomp_no_wa' and not args.checkpoint_path and args.flowcomp_output_dir == 'outputs/triviaqa_rc_llama3':
            args.flowcomp_output_dir = str(root / 'outputs' / 'ablations' / 'no_wa' / 'triviaqa_rc')
        if not Path(args.flowcomp_output_dir).is_absolute():
            args.flowcomp_output_dir = str(root / args.flowcomp_output_dir)
        model = load_flowcomp(args)
        predict = lambda row: flowcomp_predict(model, row, args)
    else:
        model = load_baseline(args.method, args)
        predict = lambda row: baseline_predict(model, row, args)

    rows = load_jsonl(Path(args.eval_file), args.max_examples)
    outputs = []
    for idx, row in enumerate(rows):
        answer = str(row[args.answer_key])
        try:
            raw = predict(row)
        except Exception as exc:
            raw = ''
            row = dict(row)
            row['error'] = repr(exc)
        scored = score_prediction(raw, answer, args.f1_threshold)
        outputs.append({
            'id': row.get('id', idx),
            'doc_id': row.get('doc_id'),
            'salience_group': row.get('salience_group', 'unknown'),
            'salience': row.get('salience'),
            'fact': row.get('fact'),
            'masked_fact': row.get('masked_fact'),
            'prompt': row.get(args.prompt_key),
            'answer': answer,
            'raw_prediction': raw,
            **scored,
            **({'error': row['error']} if 'error' in row else {}),
        })
        if (idx + 1) % 20 == 0:
            print(f'Evaluated {idx + 1}/{len(rows)}', flush=True)

    summary = summarize(outputs)
    payload = {'method': args.method, 'settings': vars(args), 'summary': summary, 'predictions': outputs}
    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')
    print(json.dumps({'method': args.method, 'summary': summary}, indent=2))
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
