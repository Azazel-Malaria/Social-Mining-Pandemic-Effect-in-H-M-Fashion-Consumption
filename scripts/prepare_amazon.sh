#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
source "$SCRIPT_DIR/_argparse.sh" "$@"
read -r -a CATS <<< "${amazon_categories:-Amazon_Fashion Clothing_Shoes_and_Jewelry}"
CMD=(python -m util.data_preprocess_amazon
  --data_root "${data_root:-./data}"
  --amazon_categories "${CATS[@]}"
  --time_filter_from_hm "${time_filter_from_hm:-1}"
  --review_sample_ratio "${review_sample_ratio:-1.0}"
  --sample_seed "${sample_seed:-42}"
)
[[ -n "${amazon_raw_dir:-}" ]] && CMD+=(--amazon_raw_dir "$amazon_raw_dir")
[[ -n "${amazon_processed_dir:-}" ]] && CMD+=(--amazon_processed_dir "$amazon_processed_dir")
[[ -n "${max_items:-}" ]] && CMD+=(--max_items "$max_items")
[[ -n "${max_reviews:-}" ]] && CMD+=(--max_reviews "$max_reviews")
[[ -n "${time_start:-}" ]] && CMD+=(--time_start "$time_start")
[[ -n "${time_end:-}" ]] && CMD+=(--time_end "$time_end")
"${CMD[@]}"
