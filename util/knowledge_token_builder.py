from __future__ import annotations

import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from util.io_utils import ensure_dir, parse_bool


def normalize(x, axis=-1):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def random_projection_matrix(in_dim: int, out_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0 / np.sqrt(in_dim), size=(in_dim, out_dim)).astype(np.float32)


def build_category_tokens(data_root: Path, prompt_subject: str = "product_group_name",
                          max_prompts_per_item: int = 64, embed_dim: int = 256,
                          include_temporal: bool = False, seed: int = 42) -> None:
    hm_proc = data_root / "hm" / "processed"
    cross_dir = ensure_dir(data_root / "cross_domain")
    hm_items = pd.read_parquet(hm_proc / "hm_item_text.parquet")
    hm_items["article_id"] = hm_items["article_id"].astype(str)
    if prompt_subject not in hm_items.columns:
        raise ValueError(f"prompt_subject={prompt_subject} not found in hm_item_text.parquet")
    suffix = "with_temporal" if include_temporal else "static"
    emb_path = cross_dir / f"knowledge_prompt_embeddings_{suffix}.npy"
    idx_path = cross_dir / f"knowledge_prompt_embedding_index_{suffix}.parquet"
    prm_path = cross_dir / f"knowledge_prompts_encoded_{suffix}.parquet"
    if not emb_path.exists() or not idx_path.exists() or not prm_path.exists():
        raise FileNotFoundError(f"Missing encoded prompt files for suffix={suffix}. Run encode_knowledge_prompts.sh first.")
    raw_emb = np.load(emb_path).astype(np.float32)
    idx = pd.read_parquet(idx_path)
    prompts = pd.read_parquet(prm_path)
    prompts["prompt_id"] = prompts["prompt_id"].astype(str)
    idx_map = {str(pid): int(row) for pid, row in zip(idx["prompt_id"], idx["row"])}
    if raw_emb.shape[1] != embed_dim:
        proj = random_projection_matrix(raw_emb.shape[1], embed_dim, seed)
        emb = normalize(raw_emb @ proj)
        np.save(cross_dir / "knowledge_prompt_projection.npy", proj)
    else:
        emb = normalize(raw_emb)
    # Keep deterministic ordering by dimension then prompt_id.
    dim_order = {
        "material": 0, "color": 1, "design": 2, "occasion": 3, "target": 4,
        "value": 5, "positive": 6, "negative": 7, "social_indicating": 8,
        "social_reflecting": 9, "temporal": 10, "seasonal": 11,
    }
    prompts["_dim_order"] = prompts["dimension"].map(dim_order).fillna(99).astype(int)
    grouped = {str(k): v.sort_values(["_dim_order", "prompt_id"]) for k, v in prompts.groupby("subject")}
    n_items = len(hm_items)
    tokens = np.zeros((n_items, max_prompts_per_item, embed_dim), dtype=np.float32)
    meta_rows = []
    reliability_rows = []
    for row_idx, row in tqdm(hm_items.iterrows(), total=n_items, desc="Assign category-level knowledge tokens"):
        aid = str(row["article_id"])
        subj = str(row.get(prompt_subject, "Unknown"))
        g = grouped.get(subj)
        if g is None or g.empty:
            # Try stripped/case-insensitive fallback.
            g = None
            subj_norm = subj.strip().lower()
            for key, val in grouped.items():
                if key.strip().lower() == subj_norm:
                    g = val
                    break
        rel = 0.0
        if g is not None and not g.empty:
            rel = 1.0
            for k, (_, pr) in enumerate(g.head(max_prompts_per_item).iterrows()):
                pid = str(pr["prompt_id"])
                eidx = idx_map.get(pid)
                if eidx is not None:
                    tokens[row_idx, k] = emb[eidx]
                    meta_rows.append({
                        "article_id": aid,
                        "prompt_id": pid,
                        "knowledge_prompt": pr.get("knowledge_prompt", ""),
                        "dimension": pr.get("dimension", ""),
                        "prompt_subject": prompt_subject,
                        "subject": subj,
                        "token_index": k,
                        "include_temporal": bool(include_temporal),
                    })
        reliability_rows.append({"article_id": aid, "reliability": rel, "subject": subj, "prompt_subject": prompt_subject})
    torch.save(torch.from_numpy(tokens), cross_dir / "hm_amazon_knowledge_tokens.pt")
    pd.DataFrame(meta_rows).to_parquet(cross_dir / "hm_amazon_knowledge_meta.parquet", index=False)
    pd.DataFrame(reliability_rows).to_parquet(cross_dir / "hm_amazon_reliability.parquet", index=False)
    print(f"Saved category-level knowledge tokens: {tokens.shape} to {cross_dir / 'hm_amazon_knowledge_tokens.pt'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data")
    p.add_argument("--prompt_subject", choices=["product_group_name", "product_type_name"], default="product_group_name")
    p.add_argument("--max_prompts_per_item", type=int, default=64)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--include_temporal", type=parse_bool, default=False,
                   help="Default false for 7-day prediction. Set true only for social-task fine-tuning/analysis.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build_category_tokens(Path(args.data_root), args.prompt_subject, args.max_prompts_per_item,
                          args.embed_dim, args.include_temporal, args.seed)


if __name__ == "__main__":
    main()
