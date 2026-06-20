#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
source "$SCRIPT_DIR/_argparse.sh" "$@"
MM_BACKBONE="${mm_backbone:-siglip}"
ITEM_PREFIX="${item_prefix:-clip}"
if [[ "$MM_BACKBONE" == "siglip" ]]; then
  DEFAULT_MODEL="google/siglip-base-patch16-224"
elif [[ "$MM_BACKBONE" == "openclip" ]]; then
  DEFAULT_MODEL="ViT-B-32"
else
  echo "Unknown --mm_backbone $MM_BACKBONE. Use siglip or openclip." >&2
  exit 1
fi
CUDA_ID="${cuda_id:-0}"

echo "[Stage 1.1] Precompute frozen CLIP-like image/detail_desc features"
bash scripts/precompute_multimodal_features.sh \
  --data_root "${data_root:-./data}" \
  --cuda_id "$CUDA_ID" \
  --mm_backbone "$MM_BACKBONE" \
  --model_name "${model_name:-$DEFAULT_MODEL}" \
  --openclip_pretrained "${openclip_pretrained:-laion2b_s34b_b79k}" \
  --output_prefix "${mm_prefix:-$MM_BACKBONE}" \
  --batch_size "${batch_size:-64}" \
  --text_batch_size "${text_batch_size:-256}" \
  --text_max_length "${text_max_length:-64}" \
  --dtype "${dtype:-fp16}"

echo "[Stage 1.2] Build item feature packet"
bash scripts/build_item_feature_packet.sh \
  --data_root "${data_root:-./data}" \
  --mm_prefix "${mm_prefix:-$MM_BACKBONE}" \
  --output_prefix "$ITEM_PREFIX" \
  --base_dim "${base_dim:-0}" \
  --image_weight "${image_weight:-1.0}" \
  --text_weight "${text_weight:-1.0}" \
  --seed "${seed:-42}" \
  --force_reproject "${force_reproject:-0}"

echo "[Stage 1.3] Route knowledge prompts to each item by visual/text similarity"
bash scripts/route_item_knowledge.sh \
  --data_root "${data_root:-./data}" \
  --item_prefix "$ITEM_PREFIX" \
  --mm_backbone "$MM_BACKBONE" \
  --model_name "${model_name:-$DEFAULT_MODEL}" \
  --openclip_pretrained "${openclip_pretrained:-laion2b_s34b_b79k}" \
  --prompt_subject "${prompt_subject:-product_group_name}" \
  --top_per_dim "${top_per_dim:-1}" \
  --bottom_per_dim "${bottom_per_dim:-2}" \
  --merge_mode "${merge_mode:-topk}" \
  --visual_weight "${visual_weight:-1.0}" \
  --text_weight "${text_weight:-1.0}" \
  --include_temporal "${include_temporal:-0}" \
  --cuda_id "$CUDA_ID" \
  --dtype "${dtype:-fp16}" \
  --prompt_batch_size "${prompt_batch_size:-256}" \
  --prompt_max_length "${prompt_max_length:-64}" \
  --reencode_prompts "${reencode_prompts:-0}"
