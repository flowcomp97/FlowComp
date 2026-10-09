# FlowComp

Reference implementation of FlowComp, with one example training entry point and one example inference/evaluation entry point.

## Layout

```text
src/flowcomp/      # model and training utilities
scripts/train.py   # training
scripts/eval_*.py  # evaluation
scripts/prepare_*.py  # dataset conversion helpers
experiments/*.sh   # example launchers
```

## Install

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e .
```

Gated target models require a Hugging Face account with access to the weights.

## Data

JSONL, one example per line.

```json
{"text": "source context"}
{"text": "source context", "prompt": "task prompt", "answer": "gold answer"}
{"text": "demonstrations", "prompt": "query", "answer": "positive", "labels": ["negative", "positive"]}
```

The first form is reconstruction-only; the second adds the downstream task; `labels` is used by classification evaluation.

## Train

```bash
FLOWCOMP_TRAIN_FILE=data/train.jsonl \
FLOWCOMP_EVAL_FILE=data/dev.jsonl \
FLOWCOMP_OUTPUT_DIR=outputs/run \
bash experiments/run_train.sh
```

Or directly:

```bash
python scripts/train.py \
  --train_file data/train.jsonl \
  --eval_file data/dev.jsonl \
  --output_dir outputs/run \
  --source_encoder_name_or_path google-t5/t5-small \
  --target_model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
  --num_soft_tokens 64 \
  --latent_dim 1024 \
  --sampling_steps 8 \
  --denoiser_layers 4 \
  --decoder_layers 2 \
  --model_max_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 8e-5 \
  --num_train_epochs 2 \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --bf16 true \
  --remove_unused_columns false \
  --report_to none
```

`pytorch_model.bin` holds only the trainable FlowComp parameters; the frozen backbones are not saved. `--save_strategy steps` additionally writes resumable `checkpoint-<step>/` directories containing optimizer, scheduler, trainer, and RNG state, and honours `--save_total_limit`.

Common launcher variables: `FLOWCOMP_SOURCE_MODEL`, `FLOWCOMP_TARGET_MODEL`, `FLOWCOMP_NUM_SOFT_TOKENS`, `FLOWCOMP_LATENT_DIM`, `FLOWCOMP_SAMPLING_STEPS`, `FLOWCOMP_DENOISER_LAYERS`, `FLOWCOMP_DECODER_LAYERS`, `FLOWCOMP_MODEL_MAX_LENGTH`, `FLOWCOMP_SEED`.

For multi-GPU, launch `scripts/train.py` with `torchrun --nproc_per_node <N>`.

## Evaluate

```bash
FLOWCOMP_EVAL_FILE=data/test.jsonl \
FLOWCOMP_CHECKPOINT=outputs/run/pytorch_model.bin \
FLOWCOMP_OUTPUT_FILE=outputs/eval.json \
bash experiments/run_eval_qa.sh          # or run_eval_classification.sh / run_eval_longdoc.sh
```

The evaluators rebuild the architecture from command-line arguments, so pass the same architecture flags used for training (`--num_soft_tokens`, `--latent_dim`, `--sampling_steps`, `--denoiser_layers`, `--decoder_layers`, `--model_max_length`).

## Selected flags

| Flag | Meaning |
| --- | --- |
| `--sampling_steps` | Euler steps at inference |
| `--timestep_logit_mean` / `--timestep_logit_std` | logit-normal timestep schedule |
| `--latent_generator_type` | `flow` or `one_shot` |
| `--distribution_loss_type` | `wasserstein`, `mmd`, or `cosine` |
| `--soft_token_attention` | `bidirectional` or `causal` |
| `--reconstruction_loss_weight` / `--utility_loss_weight` / `--wasserstein_loss_weight` / `--consistency_loss_weight` | loss weights (`0` disables) |
| `--curriculum_*` | optional compression-ratio curriculum, off by default |

## Citation

Please cite the accompanying FlowComp paper.
