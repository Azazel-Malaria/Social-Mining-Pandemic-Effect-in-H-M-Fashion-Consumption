#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
source "$SCRIPT_DIR/_argparse.sh" "$@"
python -m util.make_social_figures \
  --social_output_root "${social_output_root:-./social_output/k}" \
  --data_root "${data_root:-./data}" \
  --event_month "${event_month:-2020-03}" \
  --embedding_method "${embedding_method:-tsne}" \
  --fig_max_items "${fig_max_items:-4000}" \
  --style_dims "${style_dims:-formal,comfort,homewear,value,casual,office}" \
  --run_amazon_hm_compare "${run_amazon_hm_compare:-0}" \
  --max_amazon_reviews "${max_amazon_reviews:-300000}" \
  --seed "${seed:-42}"
