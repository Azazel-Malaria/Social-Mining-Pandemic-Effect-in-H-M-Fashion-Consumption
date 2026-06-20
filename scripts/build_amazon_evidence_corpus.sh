#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
source "$SCRIPT_DIR/_argparse.sh" "$@"
CMD=(python -m util.amazon_evidence_corpus
  --data_root "${data_root:-./data}"
  --max_reviews_per_asin "${max_reviews_per_asin:-5}"
  --min_review_chars "${min_review_chars:-10}"
  --seed "${seed:-42}"
)
[[ -n "${max_passages:-}" ]] && CMD+=(--max_passages "$max_passages")
"${CMD[@]}"
