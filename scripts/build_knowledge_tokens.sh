#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
source "$SCRIPT_DIR/_argparse.sh" "$@"
python -m util.knowledge_token_builder \
  --data_root "${data_root:-./data}" \
  --prompt_subject "${prompt_subject:-product_group_name}" \
  --max_prompts_per_item "${max_prompts_per_item:-64}" \
  --embed_dim "${embed_dim:-256}" \
  --include_temporal "${include_temporal:-0}" \
  --seed "${seed:-42}"
