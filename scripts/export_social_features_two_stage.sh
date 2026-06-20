#!/usr/bin/env bash
# Backward-compatible alias for the cleaned pipeline.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_social_inference.sh" "$@"
