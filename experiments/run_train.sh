#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${FLOWCOMP_TRAIN_FILE:?Set FLOWCOMP_TRAIN_FILE to a JSONL training file.}"
: "${FLOWCOMP_EVAL_FILE:=}"
: "${FLOWCOMP_OUTPUT_DIR:=outputs/flowcomp}"
: "${FLOWCOMP_SOURCE_MODEL:=google-t5/t5-small}"
: "${FLOWCOMP_TARGET_MODEL:=meta-llama/Meta-Llama-3-8B-Instruct}"
: "${FLOWCOMP_NUM_SOFT_TOKENS:=64}"
: "${FLOWCOMP_LATENT_DIM:=1024}"
: "${FLOWCOMP_SAMPLING_STEPS:=4}"
: "${FLOWCOMP_DENOISER_LAYERS:=4}"
: "${FLOWCOMP_DECODER_LAYERS:=2}"
: "${FLOWCOMP_ATTN_HEADS:=8}"
: "${FLOWCOMP_MODEL_MAX_LENGTH:=1024}"
# Appendix F.4: long documents use head+tail truncation.  The ICL (F.2) and
# RAG (F.3) scenarios keep the leading context only, so "head" is the default
# and long-document runs set FLOWCOMP_SOURCE_TRUNCATION=head_tail.
: "${FLOWCOMP_SOURCE_TRUNCATION:=head}"
: "${FLOWCOMP_SOURCE_HEAD_TAIL_RATIO:=0.5}"
: "${FLOWCOMP_TRAIN_BATCH_SIZE:=1}"
: "${FLOWCOMP_EVAL_BATCH_SIZE:=1}"
: "${FLOWCOMP_GRAD_ACCUM_STEPS:=8}"
: "${FLOWCOMP_LEARNING_RATE:=8e-5}"
: "${FLOWCOMP_NUM_EPOCHS:=2}"
: "${FLOWCOMP_CONSISTENCY_LOSS_WEIGHT:=0.01}"
: "${FLOWCOMP_BF16:=true}"
# Appendix E.1 / Table 8: cosine decay without warmup, weight decay 0.01,
# AdamW betas (0.9, 0.999), seed 42, checkpoints every 500 steps keeping 2.
: "${FLOWCOMP_LR_SCHEDULER:=cosine}"
: "${FLOWCOMP_WEIGHT_DECAY:=0.01}"
: "${FLOWCOMP_WARMUP_RATIO:=0.0}"
: "${FLOWCOMP_ADAM_BETA1:=0.9}"
: "${FLOWCOMP_ADAM_BETA2:=0.999}"
: "${FLOWCOMP_SAVE_STEPS:=500}"
: "${FLOWCOMP_SAVE_TOTAL_LIMIT:=2}"
: "${FLOWCOMP_SEED:=42}"

mkdir -p "${FLOWCOMP_OUTPUT_DIR}" logs
export TOKENIZERS_PARALLELISM=false

python scripts/train.py \
  --train_file "${FLOWCOMP_TRAIN_FILE}" \
  --eval_file "${FLOWCOMP_EVAL_FILE}" \
  --output_dir "${FLOWCOMP_OUTPUT_DIR}" \
  --source_encoder_name_or_path "${FLOWCOMP_SOURCE_MODEL}" \
  --target_model_name_or_path "${FLOWCOMP_TARGET_MODEL}" \
  --num_soft_tokens "${FLOWCOMP_NUM_SOFT_TOKENS}" \
  --latent_dim "${FLOWCOMP_LATENT_DIM}" \
  --sampling_steps "${FLOWCOMP_SAMPLING_STEPS}" \
  --denoiser_layers "${FLOWCOMP_DENOISER_LAYERS}" \
  --decoder_layers "${FLOWCOMP_DECODER_LAYERS}" \
  --num_attention_heads "${FLOWCOMP_ATTN_HEADS}" \
  --model_max_length "${FLOWCOMP_MODEL_MAX_LENGTH}" \
  --source_truncation "${FLOWCOMP_SOURCE_TRUNCATION}" \
  --source_head_tail_ratio "${FLOWCOMP_SOURCE_HEAD_TAIL_RATIO}" \
  --per_device_train_batch_size "${FLOWCOMP_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${FLOWCOMP_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${FLOWCOMP_GRAD_ACCUM_STEPS}" \
  --learning_rate "${FLOWCOMP_LEARNING_RATE}" \
  --num_train_epochs "${FLOWCOMP_NUM_EPOCHS}" \
  --consistency_loss_weight "${FLOWCOMP_CONSISTENCY_LOSS_WEIGHT}" \
  --lr_scheduler_type "${FLOWCOMP_LR_SCHEDULER}" \
  --weight_decay "${FLOWCOMP_WEIGHT_DECAY}" \
  --warmup_ratio "${FLOWCOMP_WARMUP_RATIO}" \
  --adam_beta1 "${FLOWCOMP_ADAM_BETA1}" \
  --adam_beta2 "${FLOWCOMP_ADAM_BETA2}" \
  --save_strategy steps \
  --save_steps "${FLOWCOMP_SAVE_STEPS}" \
  --save_total_limit "${FLOWCOMP_SAVE_TOTAL_LIMIT}" \
  --seed "${FLOWCOMP_SEED}" \
  --bf16 "${FLOWCOMP_BF16}" \
  --freeze_target_model true \
  --source_encoder_trainable false \
  --remove_unused_columns false \
  --report_to none
