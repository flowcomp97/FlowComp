#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${FLOWCOMP_EVAL_FILE:?Set FLOWCOMP_EVAL_FILE to a JSONL evaluation file.}"
: "${FLOWCOMP_CHECKPOINT:?Set FLOWCOMP_CHECKPOINT to a FlowComp pytorch_model.bin checkpoint.}"
: "${FLOWCOMP_OUTPUT_FILE:=outputs/eval_classification.json}"

python scripts/eval_classification.py \
  --eval_file "${FLOWCOMP_EVAL_FILE}" \
  --checkpoint_path "${FLOWCOMP_CHECKPOINT}" \
  --output_file "${FLOWCOMP_OUTPUT_FILE}"
