from __future__ import annotations

import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from util.io_utils import ensure_dir, parse_bool


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def encode_texts(texts: List[str], model_name: str, batch_size: int = 32,
                 max_length: int = 8192, normalize: bool = True, device: str | None = None) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()
    outs = []
    for start in tqdm(range(0, len(texts), batch_size), desc=f"Encode {model_name}"):
        batch = ["" if x is None else str(x) for x in texts[start:start + batch_size]]
        tok = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**tok)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                emb = out.pooler_output
            else:
                emb = mean_pool(out.last_hidden_state, tok["attention_mask"])
            if normalize:
                emb = F.normalize(emb, dim=-1)
        outs.append(emb.cpu().float().numpy())
    if not outs:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(outs, axis=0)


def save_embeddings(ids: Iterable[str], embeddings: np.ndarray, out_npy: Path, out_index: Path, id_col: str):
    ensure_dir(out_npy.parent)
    ids = list(ids)
    np.save(out_npy, embeddings.astype(np.float32))
    pd.DataFrame({id_col: ids, "row": np.arange(len(ids), dtype=np.int64)}).to_parquet(out_index, index=False)


def load_neighbor_asins(data_root: Path) -> Optional[Set[str]]:
    path = data_root / "cross_domain" / "hm_amazon_neighbors.parquet"
    if not path.exists():
        return None
    neigh = pd.read_parquet(path, columns=["parent_asin"])
    return set(neigh["parent_asin"].astype(str).dropna().tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--text_encoder", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--encode_amazon", type=parse_bool, default=True)
    parser.add_argument("--encode_hm", type=parse_bool, default=False)
    parser.add_argument("--only_neighbor_asins", type=parse_bool, default=True,
                        help="Encode only ASINs referenced by hm_amazon_neighbors.parquet. Recommended for prototype process_data.")
    args = parser.parse_args()

    device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"
    data_root = Path(args.data_root)
    amazon_proc = data_root / "amazon" / "processed"
    hm_proc = data_root / "hm" / "processed"

    neighbor_asins = load_neighbor_asins(data_root) if args.only_neighbor_asins else None
    if neighbor_asins is not None:
        print(f"[Embedding] Restricting Amazon product encoding to {len(neighbor_asins)} neighbor/prototype representative ASINs.", flush=True)

    if args.encode_amazon:
        items = pd.read_parquet(amazon_proc / "amazon_items_filtered.parquet")
        items["parent_asin"] = items["parent_asin"].astype(str)
        if neighbor_asins is not None:
            before = len(items)
            items = items[items["parent_asin"].isin(neighbor_asins)].copy()
            print(f"[Embedding] Amazon product text rows: {before} -> {len(items)} after neighbor-ASIN filter.", flush=True)
        prod_texts = items.get("clean_text", pd.Series("", index=items.index)).fillna("").astype(str).tolist()
        prod_emb = encode_texts(prod_texts, args.text_encoder, args.batch_size, args.max_length, True, device)
        save_embeddings(items["parent_asin"].astype(str), prod_emb,
                        amazon_proc / "amazon_product_text_embeddings.npy",
                        amazon_proc / "amazon_product_text_embedding_index.parquet", "parent_asin")

        qwen_path = amazon_proc / "amazon_qwen_summaries.parquet"
        if qwen_path.exists():
            qwen = pd.read_parquet(qwen_path)
            qwen["parent_asin"] = qwen["parent_asin"].astype(str)
            if neighbor_asins is not None:
                before = len(qwen)
                qwen = qwen[qwen["parent_asin"].isin(neighbor_asins)].copy()
                print(f"[Embedding] Qwen/prototype summary rows: {before} -> {len(qwen)} after neighbor-ASIN filter.", flush=True)
            text_col = "qwen_full_summary" if "qwen_full_summary" in qwen.columns else None
            if text_col is not None and len(qwen):
                qwen_texts = qwen[text_col].fillna("").astype(str).tolist()
                qwen_emb = encode_texts(qwen_texts, args.text_encoder, args.batch_size, args.max_length, True, device)
                save_embeddings(qwen["parent_asin"].astype(str), qwen_emb,
                                amazon_proc / "amazon_qwen_summary_embeddings.npy",
                                amazon_proc / "amazon_qwen_summary_embedding_index.parquet", "parent_asin")
            else:
                print("[Embedding] No qwen_full_summary rows to encode; qwen token will fall back to product embedding.", flush=True)

    if args.encode_hm:
        hm = pd.read_parquet(hm_proc / "hm_item_text.parquet")
        hm_texts = hm.get("hm_clean_text", pd.Series("", index=hm.index)).fillna("").astype(str).tolist()
        hm_emb = encode_texts(hm_texts, args.text_encoder, args.batch_size, args.max_length, True, device)
        save_embeddings(hm["article_id"].astype(str), hm_emb,
                        hm_proc / "hm_item_text_embeddings.npy",
                        hm_proc / "hm_item_text_embedding_index.parquet", "article_id")

    print("Text embedding finished.")


if __name__ == "__main__":
    main()
