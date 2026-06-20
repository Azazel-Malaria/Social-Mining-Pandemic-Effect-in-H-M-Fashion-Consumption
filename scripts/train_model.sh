#!/usr/bin/env bash
# Final simplified training entry.  The project now keeps only two model groups:
#   1) CLIP-like frozen item features + Transformer behavior model
#   2) The same model with routed Amazon-Qwen knowledge injection
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/train_clip_transformer.sh" "$@"
