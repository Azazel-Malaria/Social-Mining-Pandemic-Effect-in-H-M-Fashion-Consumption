#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
source "$SCRIPT_DIR/_argparse.sh" "$@"
export CUDA_VISIBLE_DEVICES="${cuda_id:-0}"
MM_BACKBONE="${mm_backbone:-siglip}"
if [[ "$MM_BACKBONE" == "siglip" ]]; then
  DEFAULT_MODEL="google/siglip-base-patch16-224"
elif [[ "$MM_BACKBONE" == "openclip" ]]; then
  DEFAULT_MODEL="ViT-B-32"
else
  echo "Unknown --mm_backbone $MM_BACKBONE. Use siglip or openclip." >&2
  exit 1
fi
CMD=(python -m util.multimodal_feature_extractor
  --data_root "${data_root:-./data}"
  --mm_backbone "$MM_BACKBONE"
  --model_name "${model_name:-$DEFAULT_MODEL}"
  --openclip_pretrained "${openclip_pretrained:-laion2b_s34b_b79k}"
  --output_prefix "${output_prefix:-$MM_BACKBONE}"
  --batch_size "${batch_size:-64}"
  --text_batch_size "${text_batch_size:-256}"
  --text_max_length "${text_max_length:-64}"
  --cuda_id 0
  --dtype "${dtype:-fp16}"
)
[[ -n "${limit_items:-}" ]] && CMD+=(--limit_items "$limit_items")
"${CMD[@]}"
