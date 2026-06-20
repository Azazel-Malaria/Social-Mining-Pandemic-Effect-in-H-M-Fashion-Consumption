#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
source "$SCRIPT_DIR/_argparse.sh" "$@"
DATA_ROOT="${data_root:-./data}"
CUDA_ID="${cuda_id:-0}"

bash scripts/prepare_hm.sh --data_root "$DATA_ROOT" --output_root "${output_root:-./output}"

CMD=(bash scripts/prepare_amazon.sh
  --data_root "$DATA_ROOT"
  --review_sample_ratio "${review_sample_ratio:-1.0}"
  --sample_seed "${sample_seed:-42}"
  --time_filter_from_hm "${time_filter_from_hm:-1}"
)
[[ -n "${max_items:-}" ]] && CMD+=(--max_items "$max_items")
[[ -n "${max_reviews:-}" ]] && CMD+=(--max_reviews "$max_reviews")
"${CMD[@]}"

bash scripts/run_process_data.sh \
  --data_root "$DATA_ROOT" \
  --cuda_id "$CUDA_ID" \
  --prompt_subject "${prompt_subject:-product_group_name}" \
  --retrieve_mode "${retrieve_mode:-local}" \
  --qwen_model "${qwen_model:-Qwen/Qwen3-4B-Instruct-2507}" \
  --text_encoder "${text_encoder:-Qwen/Qwen3-Embedding-0.6B}" \
  --mock "${mock:-0}"
