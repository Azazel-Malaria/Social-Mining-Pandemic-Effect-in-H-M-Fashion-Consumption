#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
source "$SCRIPT_DIR/_argparse.sh" "$@"
CUDA_ID="${cuda_id:-0}"
export CUDA_VISIBLE_DEVICES="$CUDA_ID"
# After masking CUDA_VISIBLE_DEVICES, the selected physical GPU becomes cuda:0 inside Python.
# Example: --cuda_id 3 -> CUDA_VISIBLE_DEVICES=3 and train_behavior.py uses --cuda_id 0.
INTERNAL_CUDA_ID="${internal_cuda_id:-0}"
echo "[train_clip_transformer] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; python device cuda:${INTERNAL_CUDA_ID}"

ARGS=(
  --data_root "${data_root:-./data}"
  --output_root "${output_root:-./output}"
  --item_prefix "${item_prefix:-clip}"
  --use_knowledge "${use_knowledge:-0}"
  --include_temporal "${include_temporal:-0}"
  --cuda_id "$INTERNAL_CUDA_ID"
  --epochs "${epochs:-10}"
  --batch_size "${batch_size:-64}"
  --num_workers "${num_workers:-8}"
  --amp "${amp:-1}"
  --sample_strategy "${sample_strategy:-stratified_month_group}"
  --hidden_dim "${hidden_dim:-256}"
  --item_adapter_layers "${item_adapter_layers:-2}"
  --item_adapter_heads "${item_adapter_heads:-4}"
  --user_layers "${user_layers:-2}"
  --user_heads "${user_heads:-4}"
  --ffn_dim "${ffn_dim:-512}"
  --dropout "${dropout:-0.1}"
  --lr "${lr:-5e-4}"
  --weight_decay "${weight_decay:-0.05}"
  --lambda_style "${lambda_style:-0.05}"
  --lambda_prompt_infonce "${lambda_prompt_infonce:-0.05}"
  --prompt_temperature "${prompt_temperature:-0.07}"
  --prompt_negatives_per_positive "${prompt_negatives_per_positive:-2}"
  --prompt_negative_mode "${prompt_negative_mode:-bottomk}"
  --prompt_infonce_on "${prompt_infonce_on:-positives}"
  --lambda_anchor "${lambda_anchor:-0.01}"
  --use_two_tower "${use_two_tower:-0}"
  --item_encode_chunk_size "${item_encode_chunk_size:-2048}"
  --candidate_chunk_size "${candidate_chunk_size:-4}"
  --transformer_injection "${transformer_injection:-False}"
  --transformer_injection_layers "${transformer_injection_layers:-0}"
  --transformer_injection_strength "${transformer_injection_strength:-1.0}"
  --topk_metric "${topk_metric:-12}"
  --grad_clip "${grad_clip:-1.0}"
  --seed "${seed:-42}"
  --use_wandb "${use_wandb:-0}"
  --wandb_project "${wandb_project:-HM_social_mining}"
  --wandb_entity "${wandb_entity:-}"
  --wandb_run_name "${wandb_run_name:-}"
  --wandb_group "${wandb_group:-}"
  --wandb_tags "${wandb_tags:-two_stage,clip_transformer}"
  --wandb_mode "${wandb_mode:-online}"
  --wandb_log_train_steps "${wandb_log_train_steps:-0}"
  --wandb_log_interval "${wandb_log_interval:-50}"
)

# These four are intentionally optional. If omitted, train_behavior.py uses the full setting:
#   --num_negatives omitted      -> all non-positive/non-history items as negatives
#   --max_history_items omitted  -> full user history
#   --max_train_samples omitted  -> all train windows
#   --max_eval_samples omitted   -> all val/test windows
[[ -n "${num_negatives:-}" ]] && ARGS+=(--num_negatives "$num_negatives")
[[ -n "${max_history_items:-}" ]] && ARGS+=(--max_history_items "$max_history_items")
[[ -n "${max_train_samples:-}" ]] && ARGS+=(--max_train_samples "$max_train_samples")
[[ -n "${max_eval_samples:-}" ]] && ARGS+=(--max_eval_samples "$max_eval_samples")
[[ -n "${prompt_infonce_max_items_per_batch:-}" ]] && ARGS+=(--prompt_infonce_max_items_per_batch "$prompt_infonce_max_items_per_batch")
[[ -n "${interaction_layers:-}" ]] && ARGS+=(--interaction_layers "$interaction_layers")
[[ -n "${interaction_heads:-}" ]] && ARGS+=(--interaction_heads "$interaction_heads")

python -m train_behavior "${ARGS[@]}"
