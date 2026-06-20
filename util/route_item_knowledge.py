from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from util.fashion_knowledge_schema import STATIC_DIMS, TEMPORAL_DIMS
from util.io_utils import ensure_dir, parse_bool, resolve_device
from util.multimodal_feature_extractor import SiglipBackend, OpenCLIPBackend, batched


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


@torch.no_grad()
def encode_prompt_texts(
    texts: list[str],
    mm_backbone: str,
    model_name: str,
    openclip_pretrained: str,
    cuda_id: int,
    dtype: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    device = resolve_device(cuda_id)
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    if mm_backbone == "siglip":
        backend = SiglipBackend(model_name, device, torch_dtype)
    elif mm_backbone == "openclip":
        backend = OpenCLIPBackend(model_name, openclip_pretrained, device, torch_dtype)
    else:
        raise ValueError(f"Unsupported mm_backbone={mm_backbone}")
    chunks = []
    for _, batch_texts in tqdm(list(batched(texts, batch_size)), desc=f"Encode knowledge prompts with {mm_backbone} text tower"):
        chunks.append(backend.encode_texts(batch_texts, max_length))
    return torch.cat(chunks, dim=0).numpy().astype(np.float32)


def load_or_encode_prompt_features(args, prompts: pd.DataFrame, data_root: Path) -> tuple[np.ndarray, pd.DataFrame]:
    cross_dir = ensure_dir(data_root / "cross_domain")
    suffix = "with_temporal" if args.include_temporal else "static"
    feat_path = cross_dir / f"{args.item_prefix}_knowledge_prompt_{args.mm_backbone}_{suffix}_features.npy"
    idx_path = cross_dir / f"{args.item_prefix}_knowledge_prompt_{args.mm_backbone}_{suffix}_index.parquet"
    if feat_path.exists() and idx_path.exists() and not args.reencode_prompts:
        emb = np.load(feat_path).astype(np.float32)
        idx = pd.read_parquet(idx_path)
        return emb, idx

    texts = prompts["knowledge_prompt"].fillna("").astype(str).tolist()
    emb = encode_prompt_texts(
        texts=texts,
        mm_backbone=args.mm_backbone,
        model_name=args.model_name,
        openclip_pretrained=args.openclip_pretrained,
        cuda_id=args.cuda_id,
        dtype=args.dtype,
        batch_size=args.prompt_batch_size,
        max_length=args.prompt_max_length,
    )
    # If item feature packet was projected to a smaller shared space, apply the
    # same text projection to prompt text embeddings so visual/text-prompt
    # similarity remains dimensionally consistent.
    proj_path = data_root / "hm" / "processed" / f"{args.item_prefix}_text_projection.npy"
    if proj_path.exists():
        proj = np.load(proj_path).astype(np.float32)
        if emb.shape[1] == proj.shape[0]:
            emb = emb @ proj
    emb = l2_normalize(emb)
    idx = pd.DataFrame({
        "prompt_id": prompts["prompt_id"].astype(str).tolist(),
        "row": np.arange(len(prompts), dtype=np.int64),
        "mm_backbone": args.mm_backbone,
        "model_name": args.model_name,
        "openclip_pretrained": args.openclip_pretrained if args.mm_backbone == "openclip" else "",
        "feature_dim": int(emb.shape[1]),
    })
    np.save(feat_path, emb.astype(np.float32))
    idx.to_parquet(idx_path, index=False)
    return emb.astype(np.float32), idx


def softmax_np(x: np.ndarray, tau: float = 0.07) -> np.ndarray:
    z = x / max(tau, 1e-6)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.maximum(e.sum(), 1e-12)


def route_item_knowledge(args) -> None:
    data_root = Path(args.data_root)
    hm_proc = data_root / "hm" / "processed"
    cross_dir = ensure_dir(data_root / "cross_domain")

    packet_path = hm_proc / f"{args.item_prefix}_item_feature_packet.parquet"
    img_path = hm_proc / f"{args.item_prefix}_item_image_features.npy"
    txt_path = hm_proc / f"{args.item_prefix}_item_text_features.npy"
    if not packet_path.exists() or not img_path.exists() or not txt_path.exists():
        raise FileNotFoundError(
            f"Missing item feature packet for prefix={args.item_prefix}. Run build_item_feature_packet.sh first."
        )
    items = pd.read_parquet(packet_path)
    items["article_id"] = items["article_id"].astype(str)
    image_feat = l2_normalize(np.load(img_path).astype(np.float32))
    text_feat = l2_normalize(np.load(txt_path).astype(np.float32))

    static_path = cross_dir / "knowledge_prompts_static.parquet"
    if not static_path.exists():
        raise FileNotFoundError(f"Missing {static_path}. Run retrieve_generate_knowledge_prompts.sh first.")
    frames = [pd.read_parquet(static_path)]
    if args.include_temporal:
        temporal_path = cross_dir / "knowledge_prompts_temporal.parquet"
        if temporal_path.exists():
            frames.append(pd.read_parquet(temporal_path))
    prompts = pd.concat(frames, ignore_index=True)
    prompts["prompt_id"] = prompts["prompt_id"].astype(str)
    prompts["subject"] = prompts["subject"].astype(str)
    prompts["dimension"] = prompts["dimension"].astype(str)

    dims = STATIC_DIMS + (TEMPORAL_DIMS if args.include_temporal else [])
    if args.dimensions:
        dims = [d.strip() for d in args.dimensions.split(",") if d.strip()]

    prompt_feat, prompt_idx = load_or_encode_prompt_features(args, prompts, data_root)
    if prompt_feat.shape[1] != image_feat.shape[1] or prompt_feat.shape[1] != text_feat.shape[1]:
        raise ValueError(
            f"Dimension mismatch: prompt={prompt_feat.shape[1]}, image={image_feat.shape[1]}, text={text_feat.shape[1]}. "
            "Use matching item packet/projection and prompt encoder."
        )

    pid_to_row = {str(pid): int(r) for pid, r in zip(prompt_idx["prompt_id"], prompt_idx["row"])}
    prompts["prompt_row"] = prompts["prompt_id"].map(pid_to_row)
    prompts = prompts[prompts["prompt_row"].notna()].copy()
    prompts["prompt_row"] = prompts["prompt_row"].astype(int)

    subj_lookup = {str(s).strip().lower(): str(s) for s in prompts["subject"].dropna().unique()}
    grouped = {(str(s), str(d)): g.reset_index(drop=True) for (s, d), g in prompts.groupby(["subject", "dimension"])}
    global_by_dim = {str(d): g.reset_index(drop=True) for d, g in prompts.groupby("dimension")}

    if args.merge_mode == "softmax":
        max_tokens = len(dims)
    else:
        max_tokens = len(dims) * int(args.top_per_dim)
    n = len(items)
    d = int(prompt_feat.shape[1])
    routed = np.zeros((n, max_tokens, d), dtype=np.float32)
    mask = np.zeros((n, max_tokens), dtype=np.bool_)
    meta_rows = []
    score_rows = []

    for i, row in tqdm(items.iterrows(), total=n, desc="Route item-level knowledge prompts"):
        aid = str(row["article_id"])
        subj_raw = str(row.get(args.prompt_subject, ""))
        subj = subj_lookup.get(subj_raw.strip().lower(), subj_raw)
        token_idx = 0
        score_row = {"article_id": aid, "subject": subj_raw, "prompt_subject": args.prompt_subject}
        for dim in dims:
            g = grouped.get((subj, dim))
            if g is None or g.empty:
                g = global_by_dim.get(dim)
            if g is None or g.empty:
                score_row[f"{dim}_score"] = 0.0
                score_row[f"{dim}_prompt"] = ""
                continue
            rows = g["prompt_row"].to_numpy(dtype=np.int64)
            pf = prompt_feat[rows]
            sim_img = pf @ image_feat[i]
            sim_txt = pf @ text_feat[i]
            scores = args.visual_weight * sim_img + args.text_weight * sim_txt
            k = min(int(args.top_per_dim), len(scores))
            top_local = np.argsort(-scores)[:k]
            top_scores = scores[top_local]
            # Bottom-k prompts are the explicit prompt-level negatives: they are
            # the least similar prompts under the same subject/dimension pool.
            # Do not treat all unselected prompts as negatives, because many of
            # them may still be partially relevant.
            neg_k = min(int(args.bottom_per_dim), max(len(scores) - len(set(top_local.tolist())), 0))
            top_set = set(int(x) for x in top_local.tolist())
            bottom_all = [int(x) for x in np.argsort(scores).tolist() if int(x) not in top_set]
            bottom_local = np.asarray(bottom_all[:neg_k], dtype=np.int64)
            bottom_ids = []
            bottom_scores = []
            bottom_prompts = []
            for blocal in bottom_local:
                bpr = g.iloc[int(blocal)]
                bottom_ids.append(str(bpr.get("prompt_id", "")))
                bottom_scores.append(float(scores[int(blocal)]))
                bottom_prompts.append(str(bpr.get("knowledge_prompt", "")))
            score_row[f"{dim}_score"] = float(top_scores[0]) if len(top_scores) else 0.0
            score_row[f"{dim}_prompt"] = str(g.iloc[int(top_local[0])].get("knowledge_prompt", "")) if len(top_local) else ""
            if args.merge_mode == "softmax":
                w = softmax_np(top_scores, args.softmax_tau)
                merged = (pf[top_local] * w[:, None]).sum(axis=0)
                routed[i, token_idx] = l2_normalize(merged[None, :])[0]
                mask[i, token_idx] = True
                selected_prompts = []
                selected_scores = []
                selected_ids = []
                for local, sc in zip(top_local, top_scores):
                    pr = g.iloc[int(local)]
                    selected_prompts.append(str(pr.get("knowledge_prompt", "")))
                    selected_scores.append(float(sc))
                    selected_ids.append(str(pr.get("prompt_id", "")))
                meta_rows.append({
                    "article_id": aid,
                    "token_index": token_idx,
                    "dimension": dim,
                    "merge_mode": "softmax",
                    "prompt_ids": "||".join(selected_ids),
                    "knowledge_prompt": " || ".join(selected_prompts),
                    "score": float(top_scores[0]) if len(top_scores) else 0.0,
                    "negative_prompt_ids": "||".join(bottom_ids),
                    "negative_scores": "||".join([f"{x:.8f}" for x in bottom_scores]),
                    "negative_prompts": " || ".join(bottom_prompts),
                    "negative_mode": "bottomk",
                    "subject": subj_raw,
                    "prompt_subject": args.prompt_subject,
                })
                token_idx += 1
            else:
                for local, sc in zip(top_local, top_scores):
                    if token_idx >= max_tokens:
                        break
                    pr = g.iloc[int(local)]
                    routed[i, token_idx] = pf[int(local)]
                    mask[i, token_idx] = True
                    meta_rows.append({
                        "article_id": aid,
                        "token_index": token_idx,
                        "dimension": dim,
                        "merge_mode": "topk",
                        "prompt_ids": str(pr.get("prompt_id", "")),
                        "knowledge_prompt": str(pr.get("knowledge_prompt", "")),
                        "score": float(sc),
                        "negative_prompt_ids": "||".join(bottom_ids),
                        "negative_scores": "||".join([f"{x:.8f}" for x in bottom_scores]),
                        "negative_prompts": " || ".join(bottom_prompts),
                        "negative_mode": "bottomk",
                        "subject": subj_raw,
                        "prompt_subject": args.prompt_subject,
                    })
                    token_idx += 1
        score_rows.append(score_row)

    suffix = "with_temporal" if args.include_temporal else "static"
    out_prefix = f"{args.item_prefix}_routed_knowledge_{suffix}"
    np.save(cross_dir / f"{out_prefix}_tokens.npy", routed.astype(np.float32))
    np.save(cross_dir / f"{out_prefix}_mask.npy", mask.astype(np.bool_))
    pd.DataFrame(meta_rows).to_parquet(cross_dir / f"{out_prefix}_meta.parquet", index=False)
    pd.DataFrame(score_rows).to_parquet(cross_dir / f"{out_prefix}_scores.parquet", index=False)
    # Stable aliases for the default static training path.
    if not args.include_temporal:
        np.save(cross_dir / "item_routed_knowledge_tokens.npy", routed.astype(np.float32))
        np.save(cross_dir / "item_routed_knowledge_mask.npy", mask.astype(np.bool_))
        pd.DataFrame(meta_rows).to_parquet(cross_dir / "item_routed_knowledge_meta.parquet", index=False)
        pd.DataFrame(score_rows).to_parquet(cross_dir / "item_knowledge_scores.parquet", index=False)

    print("========== Item-level knowledge routing finished ==========")
    print(f"items:       {n:,}")
    print(f"tokens:      {routed.shape}")
    print(f"merge_mode:  {args.merge_mode}")
    print(f"top_per_dim: {args.top_per_dim}")
    print(f"output:      {cross_dir}")
    print("==========================================================")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data")
    p.add_argument("--item_prefix", default="clip")
    p.add_argument("--mm_backbone", choices=["siglip", "openclip"], default="siglip")
    p.add_argument("--model_name", default="google/siglip-base-patch16-224")
    p.add_argument("--openclip_pretrained", default="laion2b_s34b_b79k")
    p.add_argument("--prompt_subject", choices=["product_group_name", "product_type_name"], default="product_group_name")
    p.add_argument("--dimensions", default="")
    p.add_argument("--top_per_dim", type=int, default=1)
    p.add_argument("--bottom_per_dim", type=int, default=2, help="Number of least-similar prompts per dimension stored as prompt-level negatives.")
    p.add_argument("--merge_mode", choices=["topk", "softmax"], default="topk")
    p.add_argument("--softmax_tau", type=float, default=0.07)
    p.add_argument("--visual_weight", type=float, default=1.0)
    p.add_argument("--text_weight", type=float, default=1.0)
    p.add_argument("--include_temporal", type=parse_bool, default=False)
    p.add_argument("--cuda_id", type=int, default=0)
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    p.add_argument("--prompt_batch_size", type=int, default=256)
    p.add_argument("--prompt_max_length", type=int, default=64)
    p.add_argument("--reencode_prompts", type=parse_bool, default=False)
    args = p.parse_args()
    route_item_knowledge(args)


if __name__ == "__main__":
    main()
