from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
import numpy as np
import pandas as pd

from util.io_utils import ensure_dir, parse_bool


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def stable_random_projection(in_dim: int, out_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0 / np.sqrt(max(in_dim, 1)), size=(in_dim, out_dim)).astype(np.float32)


def build_packet(
    data_root: Path,
    mm_prefix: str = "siglip",
    output_prefix: str = "clip",
    base_dim: int = 0,
    image_weight: float = 1.0,
    text_weight: float = 1.0,
    seed: int = 42,
    force_reproject: bool = False,
) -> None:
    """Build frozen item feature packet used by the lightweight behavior model.

    Stage-1 input:
      data/hm/processed/{mm_prefix}_image_features.npy
      data/hm/processed/{mm_prefix}_text_features.npy
      data/hm/processed/{mm_prefix}_feature_index.parquet

    Stage-1 output:
      data/hm/processed/{output_prefix}_item_base_features.npy
      data/hm/processed/{output_prefix}_item_image_features.npy
      data/hm/processed/{output_prefix}_item_text_features.npy
      data/hm/processed/{output_prefix}_item_feature_packet.parquet

    The image/text towers are frozen.  We only normalize, align by article_id and
    form a compact base embedding.  If base_dim <= 0 and image/text dimensions
    match, the base feature is their normalized weighted average in the original
    CLIP-like common space.
    """
    hm_proc = data_root / "hm" / "processed"
    item_path = hm_proc / "hm_item_text.parquet"
    if not item_path.exists():
        raise FileNotFoundError(f"Missing {item_path}. Run prepare_hm.sh first.")
    img_path = hm_proc / f"{mm_prefix}_image_features.npy"
    txt_path = hm_proc / f"{mm_prefix}_text_features.npy"
    idx_path = hm_proc / f"{mm_prefix}_feature_index.parquet"
    if not img_path.exists() or not txt_path.exists() or not idx_path.exists():
        raise FileNotFoundError(
            f"Missing {mm_prefix} precomputed features. Run scripts/precompute_multimodal_features.sh first."
        )

    items = pd.read_parquet(item_path)
    items["article_id"] = items["article_id"].astype(str)
    article_ids = items["article_id"].tolist()

    raw_img = np.load(img_path).astype(np.float32)
    raw_txt = np.load(txt_path).astype(np.float32)
    feat_idx = pd.read_parquet(idx_path)
    feat_idx["article_id"] = feat_idx["article_id"].astype(str)
    fmap = {a: int(r) for a, r in zip(feat_idx["article_id"], feat_idx["feature_row"])}

    img = np.zeros((len(article_ids), raw_img.shape[1]), dtype=np.float32)
    txt = np.zeros((len(article_ids), raw_txt.shape[1]), dtype=np.float32)
    found = np.zeros(len(article_ids), dtype=np.int8)
    for i, aid in enumerate(article_ids):
        r = fmap.get(aid)
        if r is not None:
            img[i] = raw_img[r]
            txt[i] = raw_txt[r]
            found[i] = 1
    img = l2_normalize(img)
    txt = l2_normalize(txt)

    same_dim = img.shape[1] == txt.shape[1]
    if base_dim and base_dim > 0:
        out_dim = int(base_dim)
    elif same_dim and not force_reproject:
        out_dim = img.shape[1]
    else:
        out_dim = min(256, max(img.shape[1], txt.shape[1]))

    if same_dim and out_dim == img.shape[1] and not force_reproject:
        base = l2_normalize(image_weight * img + text_weight * txt)
        img_aligned = img
        txt_aligned = txt
        proj_info = {"mode": "native_weighted_average", "base_dim": int(base.shape[1])}
    else:
        p_img = stable_random_projection(img.shape[1], out_dim, seed)
        p_txt = stable_random_projection(txt.shape[1], out_dim, seed + 17)
        img_aligned = l2_normalize(img @ p_img)
        txt_aligned = l2_normalize(txt @ p_txt)
        base = l2_normalize(image_weight * img_aligned + text_weight * txt_aligned)
        np.save(hm_proc / f"{output_prefix}_image_projection.npy", p_img)
        np.save(hm_proc / f"{output_prefix}_text_projection.npy", p_txt)
        proj_info = {"mode": "random_projection_weighted_average", "base_dim": int(base.shape[1])}

    np.save(hm_proc / f"{output_prefix}_item_image_features.npy", img_aligned.astype(np.float32))
    np.save(hm_proc / f"{output_prefix}_item_text_features.npy", txt_aligned.astype(np.float32))
    np.save(hm_proc / f"{output_prefix}_item_base_features.npy", base.astype(np.float32))

    packet = items.copy()
    packet["feature_row"] = np.arange(len(packet), dtype=np.int64)
    packet["mm_prefix"] = mm_prefix
    packet["output_prefix"] = output_prefix
    packet["has_mm_feature"] = found.astype(bool)
    packet["image_feature_dim"] = int(img_aligned.shape[1])
    packet["text_feature_dim"] = int(txt_aligned.shape[1])
    packet["base_feature_dim"] = int(base.shape[1])
    packet["fusion_mode"] = proj_info["mode"]
    packet["image_weight"] = float(image_weight)
    packet["text_weight"] = float(text_weight)
    packet.to_parquet(hm_proc / f"{output_prefix}_item_feature_packet.parquet", index=False)

    print("========== Item feature packet built ==========")
    print(f"mm_prefix:     {mm_prefix}")
    print(f"output_prefix: {output_prefix}")
    print(f"items:         {len(packet):,}")
    print(f"base shape:    {base.shape}")
    print(f"mode:          {proj_info['mode']}")
    print(f"saved to:      {hm_proc}")
    print("==============================================")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data")
    p.add_argument("--mm_prefix", default="siglip")
    p.add_argument("--output_prefix", default="clip")
    p.add_argument("--base_dim", type=int, default=0, help="0 keeps native common embedding dim when possible.")
    p.add_argument("--image_weight", type=float, default=1.0)
    p.add_argument("--text_weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_reproject", type=parse_bool, default=False)
    args = p.parse_args()
    build_packet(Path(args.data_root), args.mm_prefix, args.output_prefix, args.base_dim,
                 args.image_weight, args.text_weight, args.seed, args.force_reproject)


if __name__ == "__main__":
    main()
