#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
source "$SCRIPT_DIR/_argparse.sh" "$@"
CMD=(python -m util.retrieve_generate_knowledge_prompts
  --data_root "${data_root:-./data}"
  --prompt_subject "${prompt_subject:-product_group_name}"
  --retrieve_mode "${retrieve_mode:-local}"
  --retriever_type "${retriever_type:-tfidf}"
  --qwen_model "${qwen_model:-Qwen/Qwen3-4B-Instruct-2507}"
  --cuda_id "${cuda_id:-0}"
  --query_variants_per_dim "${query_variants_per_dim:-6}"
  --retrieval_topn_per_query "${retrieval_topn_per_query:-20}"
  --evidence_max_per_dimension "${evidence_max_per_dimension:-60}"
  --mock "${mock:-0}"
  --max_input_tokens "${max_input_tokens:-4096}"
  --max_new_tokens "${max_new_tokens:-256}"
  --temperature "${temperature:-0.2}"
)
[[ -n "${max_prompt_subjects:-}" ]] && CMD+=(--max_prompt_subjects "$max_prompt_subjects")
[[ -n "${min_subject_items:-}" ]] && CMD+=(--min_subject_items "$min_subject_items")
[[ -n "${prompts_per_dim:-}" ]] && CMD+=(--prompts_per_dim "$prompts_per_dim")
"${CMD[@]}"
