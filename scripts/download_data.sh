#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
source "$SCRIPT_DIR/_argparse.sh" "$@"
DATA_ROOT="${data_root:-./data}"
DOWNLOAD_HM="${download_hm:-1}"
DOWNLOAD_AMAZON="${download_amazon:-1}"
mkdir -p "$DATA_ROOT/hm/raw" "$DATA_ROOT/amazon/raw"

if [[ "$DOWNLOAD_HM" == "1" ]]; then
  if [[ -f "$DATA_ROOT/hm/raw/articles.csv" && -f "$DATA_ROOT/hm/raw/transactions_train.csv" ]]; then
    echo "[skip] H&M raw files already exist under $DATA_ROOT/hm/raw"
  else
    echo "[download] H&M Kaggle competition data"
    kaggle competitions download -c h-and-m-personalized-fashion-recommendations -p "$DATA_ROOT/hm/raw"
    unzip -n "$DATA_ROOT/hm/raw/h-and-m-personalized-fashion-recommendations.zip" -d "$DATA_ROOT/hm/raw"
  fi
fi

if [[ "$DOWNLOAD_AMAZON" == "1" ]]; then
  if [[ -d "$DATA_ROOT/amazon/raw/review_categories" || -d "$DATA_ROOT/amazon/raw/meta_categories" ]]; then
    echo "[skip] Amazon Reviews 2023 files seem to exist under $DATA_ROOT/amazon/raw"
  else
    echo "[download] Amazon Reviews 2023 from Hugging Face"
    hf download McAuley-Lab/Amazon-Reviews-2023 \
      --repo-type=dataset \
      --local-dir "$DATA_ROOT/amazon/raw"
  fi
fi
