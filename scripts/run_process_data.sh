#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
source "$SCRIPT_DIR/_argparse.sh" "$@"
DATA_ROOT="${data_root:-./data}"
CUDA_ID="${cuda_id:-0}"
PROMPT_SUBJECT="${prompt_subject:-product_group_name}"
RETRIEVE_MODE="${retrieve_mode:-local}"
QWEN_MODEL="${qwen_model:-Qwen/Qwen3-4B-Instruct-2507}"
TEXT_ENCODER="${text_encoder:-Qwen/Qwen3-Embedding-0.6B}"
MOCK="${mock:-0}"
INCLUDE_TEMPORAL="${include_temporal:-0}"

CMD=(bash scripts/build_amazon_evidence_corpus.sh
  --data_root "$DATA_ROOT"
  --max_reviews_per_asin "${max_reviews_per_asin:-5}"
)
[[ -n "${max_passages:-}" ]] && CMD+=(--max_passages "$max_passages")
"${CMD[@]}"

CMD=(bash scripts/retrieve_generate_knowledge_prompts.sh
  --data_root "$DATA_ROOT"
  --prompt_subject "$PROMPT_SUBJECT"
  --retrieve_mode "$RETRIEVE_MODE"
  --retriever_type "${retriever_type:-tfidf}"
  --qwen_model "$QWEN_MODEL"
  --cuda_id "$CUDA_ID"
  --query_variants_per_dim "${query_variants_per_dim:-6}"
  --retrieval_topn_per_query "${retrieval_topn_per_query:-20}"
  --evidence_max_per_dimension "${evidence_max_per_dimension:-60}"
  --max_new_tokens "${max_new_tokens:-256}"
  --mock "$MOCK"
)
[[ -n "${max_prompt_subjects:-}" ]] && CMD+=(--max_prompt_subjects "$max_prompt_subjects")
[[ -n "${prompts_per_dim:-}" ]] && CMD+=(--prompts_per_dim "$prompts_per_dim")
"${CMD[@]}"

bash scripts/encode_knowledge_prompts.sh \
  --data_root "$DATA_ROOT" \
  --text_encoder "$TEXT_ENCODER" \
  --cuda_id "$CUDA_ID" \
  --include_temporal "$INCLUDE_TEMPORAL"

bash scripts/build_knowledge_tokens.sh \
  --data_root "$DATA_ROOT" \
  --prompt_subject "$PROMPT_SUBJECT" \
  --max_prompts_per_item "${max_prompts_per_item:-64}" \
  --embed_dim "${embed_dim:-256}" \
  --include_temporal "$INCLUDE_TEMPORAL"


if [[ -n "${mm_backbone:-}" && "${skip_mm_precompute:-0}" != "1" ]]; then
  CMD=(bash scripts/precompute_multimodal_features.sh
    --data_root "$DATA_ROOT"
    --cuda_id "$CUDA_ID"
    --mm_backbone "${mm_backbone}"
    --output_prefix "${mm_feature_prefix:-${mm_backbone}}"
    --batch_size "${mm_batch_size:-64}"
    --text_batch_size "${mm_text_batch_size:-256}"
    --text_max_length "${mm_text_max_length:-64}"
  )
  [[ -n "${mm_model_name:-}" ]] && CMD+=(--model_name "$mm_model_name")
  [[ -n "${openclip_pretrained:-}" ]] && CMD+=(--openclip_pretrained "$openclip_pretrained")
  "${CMD[@]}"
fi
