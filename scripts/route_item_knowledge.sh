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
CMD=(python -m util.route_item_knowledge
  --data_root "${data_root:-./data}"
  --item_prefix "${item_prefix:-clip}"
  --mm_backbone "$MM_BACKBONE"
  --model_name "${model_name:-$DEFAULT_MODEL}"
  --openclip_pretrained "${openclip_pretrained:-laion2b_s34b_b79k}"
  --prompt_subject "${prompt_subject:-product_group_name}"
  --top_per_dim "${top_per_dim:-1}"
  --bottom_per_dim "${bottom_per_dim:-2}"
  --merge_mode "${merge_mode:-topk}"
  --softmax_tau "${softmax_tau:-0.07}"
  --visual_weight "${visual_weight:-1.0}"
  --text_weight "${text_weight:-1.0}"
  --include_temporal "${include_temporal:-0}"
  --cuda_id 0
  --dtype "${dtype:-fp16}"
  --prompt_batch_size "${prompt_batch_size:-256}"
  --prompt_max_length "${prompt_max_length:-64}"
  --reencode_prompts "${reencode_prompts:-0}"
)
[[ -n "${dimensions:-}" ]] && CMD+=(--dimensions "$dimensions")
"${CMD[@]}"
