#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
source "$SCRIPT_DIR/_argparse.sh" "$@"
CMD=(python -m util.data_preprocess_hm
  --data_root "${data_root:-./data}"
  --max_history_items "${max_history_items:-80}"
  --min_history_items "${min_history_items:-1}"
  --anchor_stride_days "${anchor_stride_days:-7}"
  --seed "${seed:-42}"
  --skip_existing "${skip_existing:-1}"
)
[[ -n "${max_users:-}" ]] && CMD+=(--max_users "$max_users")
[[ -n "${max_windows_per_user:-}" ]] && CMD+=(--max_windows_per_user "$max_windows_per_user")
"${CMD[@]}"
