from __future__ import annotations

import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from util.io_utils import ensure_dir, save_json


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def build_product_text(row: pd.Series) -> str:
    fields = ["title", "clean_text", "features", "description", "categories", "details"]
    parts = []
    for f in fields:
        if f in row.index:
            v = safe_str(row.get(f))
            if v and v.lower() not in {"nan", "none", "[]", "{}"}:
                parts.append(f"{f}: {v}")
    return " | ".join(parts)[:2000]


def build_review_text(row: pd.Series) -> str:
    fields = ["title", "text", "review_text", "review_title"]
    parts = []
    for f in fields:
        if f in row.index:
            v = safe_str(row.get(f))
            if v and v.lower() not in {"nan", "none"}:
                parts.append(v)
    return " ".join(parts)


def review_weight(row: pd.Series) -> float:
    def num(col, default=0.0):
        try:
            v = row.get(col, default)
            if pd.isna(v):
                return default
            return float(v)
        except Exception:
            return default
    rating = max(num("rating", 3.0), 1.0)
    helpful = max(num("helpful_vote", 0.0), 0.0)
    verified = 1.0 if str(row.get("verified_purchase", row.get("verified", ""))).lower() in {"true", "1", "yes"} else 0.0
    return (1.0 + verified) * rating * np.log2(2.0 + helpful)


def build_corpus(data_root: Path, max_reviews_per_asin: int = 5, max_passages: int | None = None,
                 seed: int = 42, min_review_chars: int = 10) -> None:
    amazon_proc = data_root / "amazon" / "processed"
    cross_dir = data_root / "cross_domain"
    ensure_dir(cross_dir)
    items_path = amazon_proc / "amazon_items_filtered.parquet"
    reviews_path = amazon_proc / "amazon_reviews_filtered.parquet"
    if not items_path.exists():
        raise FileNotFoundError(f"Missing {items_path}. Run prepare_amazon.sh first.")
    if not reviews_path.exists():
        raise FileNotFoundError(f"Missing {reviews_path}. Run prepare_amazon.sh first.")
    items = pd.read_parquet(items_path)
    reviews = pd.read_parquet(reviews_path)
    items["parent_asin"] = items["parent_asin"].astype(str)
    reviews["parent_asin"] = reviews["parent_asin"].astype(str)
    items_map = {a: row for a, row in items.set_index("parent_asin").iterrows()}
    reviews["_review_weight"] = reviews.apply(review_weight, axis=1)
    reviews["_review_text"] = reviews.apply(build_review_text, axis=1)
    before = len(reviews)
    reviews = reviews[reviews["_review_text"].str.len() >= min_review_chars].copy()
    reviews = reviews.sort_values(["parent_asin", "_review_weight"], ascending=[True, False])
    if max_reviews_per_asin and max_reviews_per_asin > 0:
        reviews = reviews.groupby("parent_asin", group_keys=False).head(max_reviews_per_asin).copy()
    rows = []
    for ridx, r in tqdm(reviews.iterrows(), total=len(reviews), desc="Build Amazon evidence passages"):
        asin = str(r["parent_asin"])
        item = items_map.get(asin)
        prod = build_product_text(item) if item is not None else ""
        rev = safe_str(r.get("_review_text"))[:1800]
        text = f"Amazon fashion product evidence. Product: {prod}. Customer review: {rev}"
        rows.append({
            "passage_id": f"{asin}_{ridx}",
            "parent_asin": asin,
            "text": text,
            "product_text": prod,
            "review_text": rev,
            "rating": r.get("rating", np.nan),
            "timestamp": r.get("timestamp", r.get("date", "")),
            "verified_purchase": r.get("verified_purchase", r.get("verified", "")),
            "helpful_vote": r.get("helpful_vote", 0),
            "review_weight": float(r.get("_review_weight", 0.0)),
        })
    df = pd.DataFrame(rows)
    if max_passages is not None and max_passages > 0 and len(df) > max_passages:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(df), size=max_passages, replace=False)
        df = df.iloc[np.sort(keep)].reset_index(drop=True)
    out_parquet = cross_dir / "amazon_evidence_passages.parquet"
    out_jsonl = cross_dir / "amazon_evidence_passages.jsonl"
    df.to_parquet(out_parquet, index=False)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    save_json({
        "input_reviews": int(before),
        "after_text_filter_and_per_asin_topk": int(len(reviews)),
        "final_passages": int(len(df)),
        "max_reviews_per_asin": int(max_reviews_per_asin),
        "max_passages": max_passages,
        "outputs": [str(out_parquet), str(out_jsonl)],
    }, cross_dir / "amazon_evidence_corpus_report.json")
    print(f"Saved evidence corpus: {len(df)} passages -> {out_parquet}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data")
    p.add_argument("--max_reviews_per_asin", type=int, default=5)
    p.add_argument("--max_passages", type=int, default=None)
    p.add_argument("--min_review_chars", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build_corpus(Path(args.data_root), args.max_reviews_per_asin, args.max_passages, args.seed, args.min_review_chars)


if __name__ == "__main__":
    main()
