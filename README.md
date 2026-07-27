# FlowComp

This repository contains the core implementation used for FlowComp experiments. It is packaged for conference artifact review: model code, training and evaluation entry points, and data-preparation utilities are included; large generated artifacts, machine-specific configs, third-party baseline source trees, checkpoints, raw datasets, and non-essential demos are intentionally excluded.

## Contents

```text
flowcomp/
  src/flowcomp/                 # FlowComp model and training utilities
  scripts/train.py              # Main training entry point
  scripts/eval_qa.py            # Open-domain QA evaluation
  scripts/eval_classification.py # In-context classification evaluation
  scripts/eval_long_document.py # Long-document QA/summarization evaluation
  scripts/prepare_*.py          # Dataset conversion utilities
  scripts/benchmark_*.py        # Latency measurement utilities
  experiments/*.sh              # Minimal shell launchers
```

## Environment

Python 3.10 or newer is recommended.

```bash
cd flowcomp
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

For Llama-family target models, make sure the Hugging Face account used by the environment has access to the model weights. FlashAttention is optional; evaluation scripts use standard attention unless `--use_flash_attention_2` is passed.

## Data Format

Training and evaluation files are JSONL. Each line is one example.

For reconstruction-only pretraining:

```json
{"text": "source document or context"}
```

For utility training and downstream evaluation:

```json
{"text": "source document or context", "prompt": "task prompt", "answer": "gold answer"}
```

Classification examples may additionally include a label set:

```json
{"text": "few-shot demonstrations or context", "prompt": "query prompt", "answer": "positive", "labels": ["negative", "positive"]}
```

## Training

The main trainer loads a source encoder, a frozen target causal LM, and the trainable FlowComp compression modules. The checkpoint written to `output_dir/pytorch_model.bin` contains only trainable FlowComp parameters, not the frozen backbone weights.

```bash
FLOWCOMP_TRAIN_FILE=data/train.jsonl \
FLOWCOMP_EVAL_FILE=data/dev.jsonl \
FLOWCOMP_OUTPUT_DIR=outputs/flowcomp_run \
bash experiments/run_train.sh
```

Important configuration knobs can be set through environment variables:

```bash
FLOWCOMP_SOURCE_MODEL=google-t5/t5-small
FLOWCOMP_TARGET_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
FLOWCOMP_NUM_SOFT_TOKENS=64
FLOWCOMP_LATENT_DIM=1024
FLOWCOMP_SAMPLING_STEPS=4
FLOWCOMP_DENOISER_LAYERS=4
FLOWCOMP_DECODER_LAYERS=2
FLOWCOMP_MODEL_MAX_LENGTH=1024
```

Equivalent direct invocation:

```bash
python scripts/train.py \
  --train_file data/train.jsonl \
  --eval_file data/dev.jsonl \
  --output_dir outputs/flowcomp_run \
  --source_encoder_name_or_path google-t5/t5-small \
  --target_model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
  --num_soft_tokens 64 \
  --latent_dim 1024 \
  --sampling_steps 4 \
  --denoiser_layers 4 \
  --decoder_layers 2 \
  --num_attention_heads 8 \
  --model_max_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 8e-5 \
  --num_train_epochs 2 \
  --bf16 true \
  --freeze_target_model true \
  --source_encoder_trainable false \
  --remove_unused_columns false \
  --report_to none
```

## Evaluation

Set `FLOWCOMP_CHECKPOINT` to the trained `pytorch_model.bin` file. The evaluation scripts reconstruct the FlowComp architecture from command-line arguments, load the trainable parameters, and report metrics plus per-example predictions.

Open-domain QA:

```bash
FLOWCOMP_EVAL_FILE=data/qa_test.jsonl \
FLOWCOMP_CHECKPOINT=outputs/flowcomp_run/pytorch_model.bin \
FLOWCOMP_OUTPUT_FILE=outputs/eval_qa.json \
bash experiments/run_eval_qa.sh
```

Few-shot classification:

```bash
FLOWCOMP_EVAL_FILE=data/classification_test.jsonl \
FLOWCOMP_CHECKPOINT=outputs/flowcomp_run/pytorch_model.bin \
FLOWCOMP_OUTPUT_FILE=outputs/eval_classification.json \
bash experiments/run_eval_classification.sh
```

Long-document tasks:

```bash
FLOWCOMP_EVAL_FILE=data/longdoc_test.jsonl \
FLOWCOMP_CHECKPOINT=outputs/flowcomp_run/pytorch_model.bin \
FLOWCOMP_OUTPUT_FILE=outputs/eval_longdoc.json \
bash experiments/run_eval_longdoc.sh
```

When the checkpoint was trained with non-default architecture settings, pass the same values to the Python evaluator, for example `--num_soft_tokens`, `--latent_dim`, `--sampling_steps`, `--denoiser_layers`, and `--decoder_layers`.

## Dataset Preparation

The `scripts/prepare_*.py` utilities convert common task formats into the JSONL schema above. They are intentionally lightweight and write local files under user-specified paths. Raw datasets are not included in this artifact.

Useful helpers:

```bash
python scripts/stat_jsonl_lengths.py --files data/train.jsonl --source_tokenizer google-t5/t5-small
python scripts/prepare_icl_classification.py --help
python scripts/prepare_open_domain_qa.py --help
python scripts/prepare_long_document.py --help
```

## Reproducibility Notes

Set random seeds through the relevant script arguments where exposed. For GPU runs, record CUDA, PyTorch, Transformers, and model checkpoint versions alongside each run. The repository ignores generated outputs and local configuration files by default so that the submitted artifact remains small and reviewable.

## Citation

If this code is used in follow-up work, please cite the accompanying FlowComp paper.
