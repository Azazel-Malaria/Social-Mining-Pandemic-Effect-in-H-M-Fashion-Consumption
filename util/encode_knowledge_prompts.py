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

from util.io_utils import parse_bool
from util.text_embedding_utils import encode_texts, save_embeddings


def encode_prompts(data_root: Path, text_encoder: str, cuda_id: int = 0, batch_size: int = 32,
                   max_length: int = 256, include_temporal: bool = False) -> None:
    cross_dir = data_root / "cross_domain"
    static_path = cross_dir / "knowledge_prompts_static.parquet"
    if not static_path.exists():
        raise FileNotFoundError(f"Missing {static_path}. Run retrieve_generate_knowledge_prompts.sh first.")
    static = pd.read_parquet(static_path)
    frames = [static]
    if include_temporal:
        temporal_path = cross_dir / "knowledge_prompts_temporal.parquet"
        if temporal_path.exists():
            frames.append(pd.read_parquet(temporal_path))
    prompts = pd.concat(frames, ignore_index=True)
    prompts["prompt_id"] = prompts["prompt_id"].astype(str)
    texts = prompts["knowledge_prompt"].fillna("").astype(str).tolist()
    device = f"cuda:{cuda_id}"
    emb = encode_texts(texts, text_encoder, batch_size=batch_size, max_length=max_length, normalize=True, device=device)
    suffix = "with_temporal" if include_temporal else "static"
    save_embeddings(prompts["prompt_id"].astype(str), emb,
                    cross_dir / f"knowledge_prompt_embeddings_{suffix}.npy",
                    cross_dir / f"knowledge_prompt_embedding_index_{suffix}.parquet",
                    "prompt_id")
    prompts.to_parquet(cross_dir / f"knowledge_prompts_encoded_{suffix}.parquet", index=False)
    print(f"Encoded {len(prompts)} {suffix} prompts using {text_encoder}.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data")
    p.add_argument("--text_encoder", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--cuda_id", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--include_temporal", type=parse_bool, default=False,
                   help="Default false for 7-day prediction. Set true only for social-task analysis/fine-tuning.")
    args = p.parse_args()
    encode_prompts(Path(args.data_root), args.text_encoder, args.cuda_id, args.batch_size, args.max_length, args.include_temporal)


if __name__ == "__main__":
    main()
