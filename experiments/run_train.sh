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
: "${FLOWCOMP_TRAIN_BATCH_SIZE:=1}"
: "${FLOWCOMP_EVAL_BATCH_SIZE:=1}"
: "${FLOWCOMP_GRAD_ACCUM_STEPS:=8}"
: "${FLOWCOMP_LEARNING_RATE:=8e-5}"
: "${FLOWCOMP_NUM_EPOCHS:=2}"
: "${FLOWCOMP_BF16:=true}"

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
  --per_device_train_batch_size "${FLOWCOMP_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${FLOWCOMP_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${FLOWCOMP_GRAD_ACCUM_STEPS}" \
  --learning_rate "${FLOWCOMP_LEARNING_RATE}" \
  --num_train_epochs "${FLOWCOMP_NUM_EPOCHS}" \
  --bf16 "${FLOWCOMP_BF16}" \
  --freeze_target_model true \
  --source_encoder_trainable false \
  --remove_unused_columns false \
  --report_to none
