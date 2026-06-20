#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
source "$SCRIPT_DIR/_argparse.sh" "$@"
python -m util.encode_knowledge_prompts \
  --data_root "${data_root:-./data}" \
  --text_encoder "${text_encoder:-Qwen/Qwen3-Embedding-0.6B}" \
  --cuda_id "${cuda_id:-0}" \
  --batch_size "${batch_size:-32}" \
  --max_length "${max_length:-256}" \
  --include_temporal "${include_temporal:-0}"
