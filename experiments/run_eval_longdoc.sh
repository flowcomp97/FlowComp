#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${FLOWCOMP_EVAL_FILE:?Set FLOWCOMP_EVAL_FILE to a JSONL evaluation file.}"
: "${FLOWCOMP_CHECKPOINT:?Set FLOWCOMP_CHECKPOINT to a FlowComp pytorch_model.bin checkpoint.}"
: "${FLOWCOMP_OUTPUT_FILE:=outputs/eval_longdoc.json}"
# Table 8 / Appendix F.4: the long-document column uses a 2048-token T5 budget
# with head+tail truncation, 128 soft tokens, 8 Euler steps and a 4-layer
# denoiser with a 2-layer decoder.
: "${FLOWCOMP_MODEL_MAX_LENGTH:=2048}"
: "${FLOWCOMP_NUM_SOFT_TOKENS:=128}"
: "${FLOWCOMP_SAMPLING_STEPS:=8}"
: "${FLOWCOMP_DENOISER_LAYERS:=4}"
: "${FLOWCOMP_DECODER_LAYERS:=2}"
: "${FLOWCOMP_SOURCE_TRUNCATION:=head_tail}"
: "${FLOWCOMP_SOURCE_HEAD_TAIL_RATIO:=0.5}"

python scripts/eval_long_document.py \
  --eval_file "${FLOWCOMP_EVAL_FILE}" \
  --checkpoint_path "${FLOWCOMP_CHECKPOINT}" \
  --output_file "${FLOWCOMP_OUTPUT_FILE}" \
  --model_max_length "${FLOWCOMP_MODEL_MAX_LENGTH}" \
  --num_soft_tokens "${FLOWCOMP_NUM_SOFT_TOKENS}" \
  --sampling_steps "${FLOWCOMP_SAMPLING_STEPS}" \
  --denoiser_layers "${FLOWCOMP_DENOISER_LAYERS}" \
  --decoder_layers "${FLOWCOMP_DECODER_LAYERS}" \
  --source_truncation "${FLOWCOMP_SOURCE_TRUNCATION}" \
  --source_head_tail_ratio "${FLOWCOMP_SOURCE_HEAD_TAIL_RATIO}"
