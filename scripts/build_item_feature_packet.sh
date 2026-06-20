#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
source "$SCRIPT_DIR/_argparse.sh" "$@"
python -m util.build_item_feature_packet \
  --data_root "${data_root:-./data}" \
  --mm_prefix "${mm_prefix:-siglip}" \
  --output_prefix "${output_prefix:-clip}" \
  --base_dim "${base_dim:-0}" \
  --image_weight "${image_weight:-1.0}" \
  --text_weight "${text_weight:-1.0}" \
  --seed "${seed:-42}" \
  --force_reproject "${force_reproject:-0}"
