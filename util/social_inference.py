from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model.social_behavior_model import SocialBehaviorModel
from util.axis_prompt_utils import (
    AXIS_IDS,
    COMPOSITE_IDS,
    axis_prompt_meta_from_schema,
    load_or_generate_axis_prompt_schema,
    robust_sigmoid_scores,
)
from util.io_utils import ensure_dir, parse_bool, resolve_device, save_json
from util.text_embedding_utils import encode_texts
from util.style_taxonomy import STYLES

NATIVE_PROTO_DIMS = [
    "material",
    "color",
    "design",
    "occasion",
    "target",
    "value",
    "social_indicating",
    "social_reflecting",
    "seasonal",
    "temporal",
]


def _article_key(x: Any) -> str:
    s = str(x)
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(10) if s.isdigit() else s


def _parse_list(x: Any) -> list:
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s or s.lower() == "nan":
            return []
        try:
            y = ast.literal_eval(s)
            if isinstance(y, (list, tuple, set, np.ndarray)):
                return list(y)
        except Exception:
            pass
        if "||" in s:
            return [v.strip() for v in s.split("||") if v.strip()]
        # Do not split natural-language prompts by whitespace unless the string
        # looks like a compact id list.  A single prompt should remain one prompt.
        return [s]
    try:
        return list(x)
    except Exception:
        return []


def _safe_dim(x: Any) -> str:
    s = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    s = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in s).strip("_")
    return s or "unknown"


def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return s * 0.0
    return (s - s.mean()) / sd


def _schema_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


def _load_checkpoint(path: Path) -> tuple[dict | None, dict]:
    if not path.exists():
        return None, {}
    ckpt = torch.load(path, map_location="cpu")
    return ckpt, dict(ckpt.get("args", {}))


def _default_checkpoint(args) -> Path:
    name = "clip_transformer_k" if bool(args.use_knowledge) else "clip_transformer_nok"
    return Path(args.output_root) / name / "best_model.pt"


def _load_items_and_features(data_root: Path, item_prefix: str):
    hm_proc = data_root / "hm" / "processed"
    packet_path = hm_proc / f"{item_prefix}_item_feature_packet.parquet"
    base_path = hm_proc / f"{item_prefix}_item_base_features.npy"
    if not packet_path.exists() or not base_path.exists():
        raise FileNotFoundError(f"Missing {packet_path} or {base_path}. Run stage1_prepare_clip_knowledge.sh first.")
    items = pd.read_parquet(packet_path)
    items["article_id"] = items["article_id"].map(_article_key)
    base = np.load(base_path).astype(np.float32)
    if len(items) != len(base):
        raise RuntimeError(f"Item packet rows ({len(items)}) != base feature rows ({len(base)})")
    return items, base


def _load_knowledge(data_root: Path, item_prefix: str, use_knowledge: bool, include_temporal: bool):
    if not use_knowledge:
        return None, None
    cross = data_root / "cross_domain"
    if include_temporal:
        tok = cross / f"{item_prefix}_routed_knowledge_with_temporal_tokens.npy"
        mask = cross / f"{item_prefix}_routed_knowledge_with_temporal_mask.npy"
    else:
        tok = cross / "item_routed_knowledge_tokens.npy"
        mask = cross / "item_routed_knowledge_mask.npy"
    if not tok.exists() or not mask.exists():
        print(f"[WARN] Routed knowledge not found: {tok}. Inference will use base item features only.")
        return None, None
    return np.load(tok).astype(np.float32), np.load(mask).astype(np.bool_)


def _make_metadata(items: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "article_id", "prod_name", "product_type_name", "product_group_name", "graphical_appearance_name",
        "colour_group_name", "perceived_colour_value_name", "perceived_colour_master_name", "department_name",
        "index_name", "index_group_name", "section_name", "garment_group_name", "detail_desc",
    ]
    cols = [c for c in cols if c in items.columns]
    out = items[cols].copy()
    out["article_id"] = out["article_id"].map(_article_key)
    return out


def _make_model(args, base_dim: int, knowledge_dim: int, use_knowledge: bool, cfg: dict) -> SocialBehaviorModel:
    return SocialBehaviorModel(
        base_dim=base_dim,
        hidden_dim=int(cfg.get("hidden_dim", args.hidden_dim)),
        num_styles=len(STYLES),
        use_knowledge=use_knowledge,
        knowledge_dim=int(knowledge_dim) if use_knowledge else 0,
        use_two_tower=bool(int(cfg.get("use_two_tower", args.use_two_tower))),
        item_adapter_layers=int(cfg.get("item_adapter_layers", args.item_adapter_layers)),
        item_adapter_heads=int(cfg.get("item_adapter_heads", args.item_adapter_heads)),
        user_layers=int(cfg.get("user_layers", args.user_layers)),
        user_heads=int(cfg.get("user_heads", args.user_heads)),
        interaction_layers=cfg.get("interaction_layers", None),
        interaction_heads=cfg.get("interaction_heads", None),
        ffn_dim=int(cfg.get("ffn_dim", args.ffn_dim)),
        dropout=float(cfg.get("dropout", args.dropout)),
        max_history=int(cfg.get("max_history_items") or args.max_history_items),
        item_encode_chunk_size=int(cfg.get("item_encode_chunk_size", args.item_encode_chunk_size)),
        candidate_chunk_size=int(cfg.get("candidate_chunk_size", args.candidate_chunk_size)),
        transformer_injection=bool(cfg.get("transformer_injection", False)),
        knowledge_factor_dim=int(cfg.get("knowledge_factor_dim", 0)),
        transformer_injection_layers=int(cfg.get("transformer_injection_layers", 0) or 0),
        transformer_injection_strength=float(cfg.get("transformer_injection_strength", 1.0)),
    )


def compute_item_embeddings_and_model(args, items: pd.DataFrame, base: np.ndarray, knowledge: np.ndarray | None, kmask: np.ndarray | None):
    ckpt_path = Path(args.checkpoint) if args.checkpoint else _default_checkpoint(args)
    ckpt, cfg = _load_checkpoint(ckpt_path)
    if ckpt is None:
        msg = (
            f"Checkpoint not found: {ckpt_path}. Social inference should load the trained Stage-2 backbone. "
            f"Pass --checkpoint /path/to/best_model.pt, or set --allow_base_fallback 1 only for debugging."
        )
        if bool(args.allow_base_fallback):
            print(f"[WARN] {msg}")
            return base.astype(np.float32), None, {}
        raise FileNotFoundError(msg)
    print(f"[social_inference] Loading trained checkpoint: {ckpt_path}")
    if "epoch" in ckpt or "best_metric" in ckpt:
        print(f"[social_inference] checkpoint epoch={ckpt.get('epoch', 'NA')}, best_metric={ckpt.get('best_metric', 'NA')}")

    use_knowledge = bool(args.use_knowledge) and knowledge is not None and kmask is not None
    device = resolve_device(args.cuda_id)
    model = _make_model(args, base.shape[1], int(knowledge.shape[-1]) if use_knowledge else 0, use_knowledge, cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    critical_missing = [k for k in missing if not k.startswith("interaction_encoder.")]
    if critical_missing or unexpected:
        raise RuntimeError(f"Checkpoint load mismatch. missing={critical_missing}, unexpected={unexpected}")
    model.eval()

    chunks = []
    with torch.no_grad():
        for st in tqdm(range(0, len(items), args.batch_size), desc="Social inference: item embeddings"):
            ed = min(st + args.batch_size, len(items))
            bf = torch.from_numpy(base[st:ed]).to(device)
            if use_knowledge:
                kt = torch.from_numpy(knowledge[st:ed]).to(device)
                mk = torch.from_numpy(kmask[st:ed]).to(device)
                out = model.encode_items(bf, kt, mk)
            else:
                out = model.encode_items(bf)
            chunks.append(out["item_emb"].detach().cpu().float().numpy())
    emb = np.concatenate(chunks, axis=0).astype(np.float32)
    # Keep only the small trained projector on CPU for axis prompt projection.
    model = model.cpu().eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return emb, model, cfg


def _read_routed_knowledge_scores(data_root: Path, items: pd.DataFrame) -> pd.DataFrame:
    path = data_root / "cross_domain" / "item_knowledge_scores.parquet"
    if path.exists():
        out = pd.read_parquet(path)
        if "article_id" not in out.columns:
            out.insert(0, "article_id", items["article_id"].values)
        out["article_id"] = out["article_id"].map(_article_key)
        return out
    return pd.DataFrame({"article_id": items["article_id"].values})


def _knowledge_prototypes(data_root: Path, items: pd.DataFrame, tau: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    cross = data_root / "cross_domain"
    meta_path = cross / "item_routed_knowledge_meta.parquet"
    base = pd.DataFrame({"article_id": items["article_id"].values})
    if not meta_path.exists():
        return base.copy(), _schema_df(["article_id", "dimension", "prototype", "prototype_score", "prototype_confidence", "prototype_entropy", "prototype_margin"])
    meta = pd.read_parquet(meta_path)
    if "article_id" not in meta.columns:
        return base.copy(), _schema_df(["article_id", "dimension", "prototype", "prototype_score", "prototype_confidence", "prototype_entropy", "prototype_margin"])
    meta["article_id"] = meta["article_id"].map(_article_key)
    dim_col = next((c for c in ["dimension", "relation", "knowledge_dim", "prompt_dimension"] if c in meta.columns), None)
    prompt_col = next((c for c in ["knowledge_prompt", "prompt", "selected_prompt", "positive_prompts", "routed_prompt", "text"] if c in meta.columns), None)
    score_col = next((c for c in ["score", "selected_score", "positive_scores", "similarity", "prompt_scores"] if c in meta.columns), None)
    pid_col = next((c for c in ["prompt_ids", "prompt_id", "positive_prompt_ids", "selected_prompt_ids"] if c in meta.columns), None)
    if dim_col is None:
        return base.copy(), _schema_df(["article_id", "dimension", "prototype", "prototype_score", "prototype_confidence", "prototype_entropy", "prototype_margin"])

    rows = []
    allowed = set(NATIVE_PROTO_DIMS)
    for _, r in meta.iterrows():
        dim = _safe_dim(r.get(dim_col, "unknown"))
        if dim not in allowed:
            continue
        prompts = _parse_list(r.get(prompt_col, "")) if prompt_col else []
        pids = _parse_list(r.get(pid_col, "")) if pid_col else []
        scores = _parse_list(r.get(score_col, "")) if score_col else []
        if not prompts and pids:
            prompts = pids
        if not prompts:
            continue
        vals = []
        for j, prompt in enumerate(prompts):
            try:
                sc = float(scores[j]) if j < len(scores) else np.nan
            except Exception:
                sc = np.nan
            vals.append((j, str(prompt), str(pids[j]) if j < len(pids) else "", sc))
        # If no numeric scores are available, preserve order and use first prompt.
        num = np.array([v[3] for v in vals], dtype=float)
        if np.isfinite(num).any():
            fill = np.nanmin(num[np.isfinite(num)]) - 1.0
            num2 = np.nan_to_num(num, nan=fill)
            best_idx = int(np.argmax(num2))
            ex = np.exp((num2 - np.max(num2)) / max(float(tau), 1e-6))
            prob = ex / max(float(ex.sum()), 1e-12)
            conf = float(prob[best_idx])
            entropy = float(-(prob * np.log(prob + 1e-12)).sum())
            ss = np.sort(num2)[::-1]
            margin = float(ss[0] - ss[1]) if len(ss) > 1 else np.nan
        else:
            best_idx, conf, entropy, margin = 0, np.nan, np.nan, np.nan
        _, prompt, pid, sc = vals[best_idx]
        rows.append({
            "article_id": r["article_id"],
            "dimension": dim,
            "prototype": prompt,
            "prompt_id": pid,
            "prototype_score": float(sc) if np.isfinite(sc) else np.nan,
            "prototype_confidence": conf,
            "prototype_entropy": entropy,
            "prototype_margin": margin,
        })
    if not rows:
        return base.copy(), _schema_df(["article_id", "dimension", "prototype", "prototype_score", "prototype_confidence", "prototype_entropy", "prototype_margin"])
    long = pd.DataFrame(rows)
    # one row per article-dimension
    long = long.sort_values(["article_id", "dimension", "prototype_score"], ascending=[True, True, False]).drop_duplicates(["article_id", "dimension"])
    wide = base.copy()
    for dim, g in long.groupby("dimension"):
        g = g.drop_duplicates("article_id")
        ren = {
            "prototype": f"{dim}_proto",
            "prototype_score": f"{dim}_proto_score",
            "prototype_confidence": f"{dim}_proto_confidence",
            "prototype_entropy": f"{dim}_proto_entropy",
            "prototype_margin": f"{dim}_proto_margin",
        }
        wide = wide.merge(g[["article_id", "prototype", "prototype_score", "prototype_confidence", "prototype_entropy", "prototype_margin"]].rename(columns=ren), on="article_id", how="left")
    return wide, long


def _safe_encoder_tag(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_name).strip())[:120] or "unknown"


def _first_linear_in_features(module: torch.nn.Module | None) -> int | None:
    if module is None:
        return None
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            return int(m.in_features)
    return None


def _knowledge_projector_input_dim(model: SocialBehaviorModel | None) -> int | None:
    if model is None or not getattr(model, "use_knowledge", False):
        return None
    item_encoder = getattr(model, "item_encoder", None)
    projector = getattr(item_encoder, "knowledge_projector", None)
    return _first_linear_in_features(projector)


def _resolve_axis_text_encoder(text_encoder: str, expected_dim: int | None = None, score_space: str = "item_text") -> str:
    name = str(text_encoder or "auto").strip()
    if name.lower() != "auto":
        return name
    score_space = str(score_space or "item_text").strip().lower()

    # Direct post-hoc axis measurement: encode item metadata text and axis prompts
    # with the same text encoder, then compare them in that encoder space.  This
    # deliberately bypasses the trained knowledge_projector, so no checkpoint
    # knowledge_dim is required.
    if score_space in {"item_text", "text", "metadata_text"}:
        return "Qwen/Qwen3-Embedding-4B"

    # Legacy route: encode prompts in the same input space as the training-time
    # knowledge tokens, then pass them through the trained knowledge_projector.
    if expected_dim == 768:
        return "openai/clip-vit-large-patch14"
    if expected_dim == 512:
        return "openai/clip-vit-base-patch32"
    raise RuntimeError(
        f"--text_encoder auto cannot infer a compatible encoder for expected knowledge_dim={expected_dim}. "
        "For the recommended direct route, use --axis_score_space item_text; "
        "for the legacy projector route, pass the exact text encoder used when knowledge tokens were generated."
    )


def _encode_clip_texts(texts: list[str], model_name: str, batch_size: int, max_length: int, device: str) -> np.ndarray:
    # Local import keeps the non-CLIP path unchanged.
    from transformers import AutoTokenizer, CLIPModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    clip_model = CLIPModel.from_pretrained(model_name).to(device).eval()
    outs = []
    use_max_len = min(int(max_length), 77)  # CLIP text context length.
    with torch.no_grad():
        for st in range(0, len(texts), int(batch_size)):
            batch = texts[st: st + int(batch_size)]
            tok = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=use_max_len,
                return_tensors="pt",
            )
            tok = {k: v.to(device) for k, v in tok.items()}
            feat = clip_model.get_text_features(**tok)
            feat = F.normalize(feat.float(), dim=-1)
            outs.append(feat.detach().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def _encode_axis_texts(texts: list[str], model_name: str, batch_size: int, max_length: int, device: str) -> np.ndarray:
    lower = model_name.lower()
    if lower.startswith("openai/clip-") or "clip-vit" in lower:
        return _encode_clip_texts(texts, model_name, batch_size, max_length, device)
    return encode_texts(
        texts,
        model_name=model_name,
        batch_size=int(batch_size),
        max_length=int(max_length),
        normalize=True,
        device=device,
    ).astype(np.float32)


def _load_or_encode_axis_prompt_embeddings(
    args,
    data_root: Path,
    out_dir: Path,
    expected_prompt_dim: int | None = None,
    score_space: str = "item_text",
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    cross = ensure_dir(data_root / "cross_domain")
    prompt_json = Path(args.axis_prompt_json) if args.axis_prompt_json else cross / "style_axis_prompts.json"
    schema = load_or_generate_axis_prompt_schema(
        out_json=prompt_json,
        qwen_model=args.qwen_model,
        cuda_id=int(args.cuda_id),
        force_regenerate=bool(args.axis_prompt_force_regenerate),
        auto_generate=bool(args.auto_generate_axis_prompts),
        mock=bool(args.mock_qwen_axis_prompts),
        prompts_per_polarity=int(args.axis_prompts_per_polarity),
        max_input_tokens=int(args.axis_qwen_max_input_tokens),
        max_new_tokens=int(args.axis_qwen_max_new_tokens),
        temperature=float(args.axis_qwen_temperature),
    )
    meta = axis_prompt_meta_from_schema(schema)
    if meta.empty:
        raise RuntimeError("Axis prompt schema contains no prompts.")

    text_encoder = _resolve_axis_text_encoder(args.text_encoder, expected_prompt_dim, score_space=score_space)
    enc_tag = _safe_encoder_tag(text_encoder)
    dim_tag = f"d{expected_prompt_dim}" if expected_prompt_dim is not None else "dauto"
    meta_path = cross / f"style_axis_prompt_meta.{enc_tag}.{dim_tag}.parquet"
    emb_path = cross / f"style_axis_prompt_embeddings.{enc_tag}.{dim_tag}.npy"

    # Mirror schema/meta in social output for reproducibility.
    save_json(schema, out_dir / "style_axis_prompts.used.json")
    meta.to_parquet(out_dir / "style_axis_prompt_meta.parquet", index=False)
    need_encode = bool(args.axis_prompt_force_regenerate) or (not emb_path.exists()) or (not meta_path.exists())
    if not need_encode:
        try:
            old_meta = pd.read_parquet(meta_path)
            old_emb = np.load(emb_path).astype(np.float32)
            same_texts = (
                len(old_meta) == len(meta)
                and old_meta.get("prompt_text", pd.Series(dtype=str)).astype(str).tolist() == meta["prompt_text"].astype(str).tolist()
                and len(old_emb) == len(meta)
            )
            same_dim = expected_prompt_dim is None or int(old_emb.shape[1]) == int(expected_prompt_dim)
            if same_texts and same_dim:
                return meta, old_emb, schema
            need_encode = True
        except Exception:
            need_encode = True
    if need_encode:
        print(f"[social_inference] Encoding {len(meta)} axis prompts with compatible text encoder: {text_encoder}")
        device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"
        emb = _encode_axis_texts(
            meta["prompt_text"].astype(str).tolist(),
            model_name=text_encoder,
            batch_size=int(args.axis_encode_batch_size),
            max_length=int(args.axis_text_max_length),
            device=device,
        ).astype(np.float32)
        if expected_prompt_dim is not None and int(emb.shape[1]) != int(expected_prompt_dim):
            raise RuntimeError(
                f"Axis prompt encoder '{text_encoder}' produced dim={emb.shape[1]}, but the legacy projector route expects "
                f"dim={expected_prompt_dim}. Use --axis_score_space item_text to bypass knowledge_projector, "
                "or pass the exact text encoder used to build item_routed_knowledge_tokens.npy."
            )
        np.save(emb_path, emb)
        meta.to_parquet(meta_path, index=False)
        # Do not write the old untagged style_axis_prompt_embeddings.npy in the
        # direct item_text route.  That file was previously interpreted as a
        # projector-input cache and can silently pollute legacy runs.
        if str(score_space or "").strip().lower() not in {"item_text", "text", "metadata_text"}:
            np.save(cross / "style_axis_prompt_embeddings.npy", emb)
            meta.to_parquet(cross / "style_axis_prompt_meta.parquet", index=False)
    else:
        emb = np.load(emb_path).astype(np.float32)
    return meta, emb, schema


def _project_axis_prompt_embeddings(model: SocialBehaviorModel | None, prompt_emb: np.ndarray, item_emb_dim: int) -> np.ndarray:
    if prompt_emb.shape[1] == item_emb_dim:
        return prompt_emb.astype(np.float32)
    expected = _knowledge_projector_input_dim(model)
    if model is None or not getattr(model, "use_knowledge", False) or getattr(model.item_encoder, "knowledge_projector", None) is None:
        raise RuntimeError(
            f"Axis prompt embedding dim={prompt_emb.shape[1]} does not match item embedding dim={item_emb_dim}, "
            "and no trained knowledge_projector is available. Run inference with --use_knowledge 1 and a knowledge-injection checkpoint."
        )
    if expected is not None and int(prompt_emb.shape[1]) != int(expected):
        raise RuntimeError(
            f"Axis prompt embedding dim={prompt_emb.shape[1]} cannot be fed into the trained knowledge_projector, "
            f"which expects dim={expected}. Re-encode axis prompts with a compatible text encoder; for the CLIP route use --text_encoder auto "
            "or --text_encoder openai/clip-vit-large-patch14 when expected dim is 768."
        )
    with torch.no_grad():
        x = torch.from_numpy(prompt_emb.astype(np.float32))
        y = model.item_encoder.project_prompts(x).detach().cpu().float().numpy()
    return y.astype(np.float32)


def _build_item_axis_texts(items: pd.DataFrame) -> list[str]:
    """Build stable item-level text descriptions for direct axis probing.

    This uses only H&M item metadata / article text.  The text is encoded with
    the same encoder as axis prompts, so cosine similarity is well-defined even
    when the trained model embedding has a different dimension.
    """
    preferred_cols = [
        "prod_name",
        "product_type_name",
        "product_group_name",
        "graphical_appearance_name",
        "colour_group_name",
        "perceived_colour_value_name",
        "perceived_colour_master_name",
        "department_name",
        "index_name",
        "index_group_name",
        "section_name",
        "garment_group_name",
        "detail_desc",
    ]
    cols = [c for c in preferred_cols if c in items.columns]
    label = {
        "prod_name": "product",
        "product_type_name": "type",
        "product_group_name": "group",
        "graphical_appearance_name": "appearance",
        "colour_group_name": "color",
        "perceived_colour_value_name": "color value",
        "perceived_colour_master_name": "master color",
        "department_name": "department",
        "index_name": "index",
        "index_group_name": "index group",
        "section_name": "section",
        "garment_group_name": "garment group",
        "detail_desc": "description",
    }
    texts: list[str] = []
    for _, row in items.iterrows():
        parts = []
        for c in cols:
            v = row.get(c, None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            v = str(v).strip()
            if not v or v.lower() in {"nan", "none", "null"}:
                continue
            parts.append(f"{label.get(c, c)}: {v}")
        if not parts:
            parts.append("generic clothing item")
        texts.append("; ".join(parts))
    return texts


def _load_or_encode_item_axis_embeddings(
    args,
    data_root: Path,
    out_dir: Path,
    items: pd.DataFrame,
    text_encoder: str,
    expected_dim: int | None = None,
) -> np.ndarray:
    cross = ensure_dir(data_root / "cross_domain")
    enc_tag = _safe_encoder_tag(text_encoder)
    meta_path = cross / f"item_axis_text_meta.{enc_tag}.parquet"
    emb_path = cross / f"item_axis_text_embeddings.{enc_tag}.npy"
    item_texts = _build_item_axis_texts(items)
    item_meta = pd.DataFrame({
        "article_id": items["article_id"].map(_article_key).values,
        "item_axis_text": item_texts,
    })
    item_meta.to_parquet(out_dir / "item_axis_texts.parquet", index=False)

    need_encode = bool(args.axis_prompt_force_regenerate) or (not emb_path.exists()) or (not meta_path.exists())
    if not need_encode:
        try:
            old_meta = pd.read_parquet(meta_path)
            old_emb = np.load(emb_path).astype(np.float32)
            same_items = (
                len(old_meta) == len(item_meta)
                and old_meta.get("article_id", pd.Series(dtype=str)).astype(str).tolist() == item_meta["article_id"].astype(str).tolist()
                and old_meta.get("item_axis_text", pd.Series(dtype=str)).astype(str).tolist() == item_meta["item_axis_text"].astype(str).tolist()
                and len(old_emb) == len(item_meta)
            )
            same_dim = expected_dim is None or int(old_emb.shape[1]) == int(expected_dim)
            if same_items and same_dim:
                return old_emb
            need_encode = True
        except Exception:
            need_encode = True

    if need_encode:
        print(f"[social_inference] Encoding {len(item_meta)} item metadata texts with axis text encoder: {text_encoder}")
        device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"
        emb = _encode_axis_texts(
            item_meta["item_axis_text"].astype(str).tolist(),
            model_name=text_encoder,
            batch_size=int(args.axis_encode_batch_size),
            max_length=int(args.axis_text_max_length),
            device=device,
        ).astype(np.float32)
        if expected_dim is not None and int(emb.shape[1]) != int(expected_dim):
            raise RuntimeError(
                f"Item text encoder '{text_encoder}' produced dim={emb.shape[1]}, but axis prompt embedding dim={expected_dim}. "
                "Item texts and axis prompts must be encoded by the same encoder in the direct item_text route."
            )
        np.save(emb_path, emb)
        item_meta.to_parquet(meta_path, index=False)
    else:
        emb = np.load(emb_path).astype(np.float32)
    return emb.astype(np.float32)


def _axis_scores_from_embeddings(items: pd.DataFrame, item_emb: np.ndarray, prompt_meta: pd.DataFrame, prompt_proj: np.ndarray, tau: float) -> pd.DataFrame:
    item_norm = item_emb / np.maximum(np.linalg.norm(item_emb, axis=1, keepdims=True), 1e-12)
    prompt_norm = prompt_proj / np.maximum(np.linalg.norm(prompt_proj, axis=1, keepdims=True), 1e-12)
    out = pd.DataFrame({"article_id": items["article_id"].values})
    for axis_id in prompt_meta["axis_id"].dropna().astype(str).unique():
        pos_idx = prompt_meta.index[(prompt_meta["axis_id"].astype(str) == axis_id) & (prompt_meta["polarity"].astype(str) == "positive")].to_numpy()
        neg_idx = prompt_meta.index[(prompt_meta["axis_id"].astype(str) == axis_id) & (prompt_meta["polarity"].astype(str) == "negative")].to_numpy()
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        pos = item_norm @ prompt_norm[pos_idx].T
        neg = item_norm @ prompt_norm[neg_idx].T
        pos_score = pos.mean(axis=1)
        neg_score = neg.mean(axis=1)
        raw_margin = pos_score - neg_score
        score, margin = robust_sigmoid_scores(raw_margin, tau=tau)
        out[f"{axis_id}_pos_score"] = pos_score.astype(np.float32)
        out[f"{axis_id}_neg_score"] = neg_score.astype(np.float32)
        out[f"{axis_id}_raw_margin"] = raw_margin.astype(np.float32)
        out[f"{axis_id}_margin"] = margin.astype(np.float32)
        out[f"{axis_id}_score"] = score.astype(np.float32)
        out[f"{axis_id}_confidence"] = (2.0 * np.abs(score - 0.5)).astype(np.float32)
    return out


def _composite_scores(axis_scores: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"article_id": axis_scores["article_id"].values})
    def s(name: str) -> np.ndarray:
        col = f"{name}_score"
        if col not in axis_scores.columns:
            return np.full(len(axis_scores), np.nan, dtype=np.float32)
        return pd.to_numeric(axis_scores[col], errors="coerce").to_numpy(dtype=np.float32)
    formality = s("formality_structure")
    public = s("public_private_occasion")
    comfort = s("comfort_ease")
    fit = s("fit_looseness")
    trend = s("trend_expressiveness")
    practical = s("practical_value")
    design = s("design_complexity")
    out["office_like_score"] = (formality * public).astype(np.float32)
    out["homewear_like_score"] = ((1 - public) * comfort * fit).astype(np.float32)
    out["casual_like_score"] = (1 - formality).astype(np.float32)
    out["social_outing_like_score"] = (public * trend).astype(np.float32)
    out["value_basic_score"] = (practical * (1 - trend) * (1 - design)).astype(np.float32)
    return out


def _apply_copula_calibration(scores: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    if len(score_cols) < 2:
        return scores
    try:
        from sklearn.preprocessing import QuantileTransformer
    except Exception as exc:
        print(f"[WARN] Copula calibration skipped: sklearn unavailable ({exc})")
        return scores
    out = scores.copy()
    X = out[score_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n = X.shape[0]
    try:
        qt = QuantileTransformer(n_quantiles=max(10, min(1000, n)), output_distribution="normal", random_state=0)
        Z = np.clip(qt.fit_transform(X), -6, 6)
        Sigma = np.corrcoef(Z, rowvar=False)
        Sigma = np.nan_to_num(Sigma)
        Sigma = 0.98 * Sigma + 0.02 * np.eye(Sigma.shape[0])
        invS = np.linalg.pinv(Sigma)
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            logdet = 0.0
        A = invS - np.eye(Sigma.shape[0])
        log_c = -0.5 * logdet - 0.5 * np.einsum("ij,jk,ik->i", Z, A, Z)
        conf = 1.0 / (1.0 + np.exp(-_zscore(pd.Series(log_c)).to_numpy()))
        out["copula_confidence"] = conf.astype(np.float32)
        for c in score_cols:
            out[f"copula_calibrated_{c}"] = (pd.to_numeric(out[c], errors="coerce").fillna(0).to_numpy() * conf).astype(np.float32)
        print(f"[social_inference] Added optional copula calibration for {len(score_cols)} semantic score columns.")
    except Exception as exc:
        print(f"[WARN] Copula calibration skipped/failed: {type(exc).__name__}: {exc}")
    return out


def _load_transactions(data_root: Path) -> pd.DataFrame:
    path = data_root / "hm" / "processed" / "hm_transactions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing transactions: {path}")
    tx = pd.read_parquet(path)
    tx["article_id"] = tx["article_id"].map(_article_key)
    tx["customer_id"] = tx["customer_id"].astype(str) if "customer_id" in tx.columns else ""
    tx["t_dat"] = pd.to_datetime(tx["t_dat"])
    tx["month"] = tx["t_dat"].dt.to_period("M").astype(str)
    if "price" not in tx.columns:
        tx["price"] = np.nan
    return tx


def _dense_item_month(items: pd.DataFrame, tx: pd.DataFrame, dense: bool = True) -> pd.DataFrame:
    monthly = tx.groupby(["article_id", "month"], as_index=False).agg(
        sales_count=("article_id", "size"),
        avg_price=("price", "mean"),
        unique_customers=("customer_id", "nunique") if "customer_id" in tx.columns else ("article_id", "size"),
    )
    if dense:
        months = sorted(tx["month"].unique())
        grid = pd.MultiIndex.from_product([items["article_id"].values, months], names=["article_id", "month"]).to_frame(index=False)
        monthly = grid.merge(monthly, on=["article_id", "month"], how="left")
        monthly["sales_count"] = monthly["sales_count"].fillna(0).astype(int)
        monthly["unique_customers"] = monthly["unique_customers"].fillna(0).astype(int)
    monthly["avg_price"] = pd.to_numeric(monthly["avg_price"], errors="coerce")
    return monthly


def _attach_covid(panel: pd.DataFrame, covid_csv: str | None, covid_location: str) -> pd.DataFrame:
    if not covid_csv:
        return panel
    cpath = Path(covid_csv)
    if not cpath.exists():
        print(f"[WARN] COVID csv not found: {cpath}")
        return panel
    covid = pd.read_csv(cpath)
    if "location" in covid.columns:
        covid = covid[covid["location"].astype(str) == str(covid_location)].copy()
    if covid.empty or "date" not in covid.columns:
        print(f"[WARN] No OWID rows for location={covid_location}; COVID columns not merged.")
        return panel
    covid["date"] = pd.to_datetime(covid["date"])
    covid["month"] = covid["date"].dt.to_period("M").astype(str)
    val_cols = [c for c in [
        "new_cases_smoothed_per_million", "new_deaths_smoothed_per_million",
        "new_cases_per_million", "new_deaths_per_million", "new_cases_smoothed", "new_deaths_smoothed",
        "stringency_index", "reproduction_rate", "positive_rate", "hosp_patients_per_million", "icu_patients_per_million",
    ] if c in covid.columns]
    cm = covid.groupby("month", as_index=False)[val_cols].mean() if val_cols else covid[["month"]].drop_duplicates()
    case_col = "new_cases_smoothed_per_million" if "new_cases_smoothed_per_million" in cm.columns else ("new_cases_per_million" if "new_cases_per_million" in cm.columns else None)
    death_col = "new_deaths_smoothed_per_million" if "new_deaths_smoothed_per_million" in cm.columns else ("new_deaths_per_million" if "new_deaths_per_million" in cm.columns else None)
    if case_col:
        cm["covid_cases_index"] = np.log1p(cm[case_col].fillna(0.0).clip(lower=0))
        cm["covid_cases_z"] = _zscore(cm["covid_cases_index"])
    if death_col:
        cm["covid_deaths_index"] = np.log1p(cm[death_col].fillna(0.0).clip(lower=0))
        cm["covid_deaths_z"] = _zscore(cm["covid_deaths_index"])
    if "stringency_index" in cm.columns:
        cm["covid_stringency_index"] = pd.to_numeric(cm["stringency_index"], errors="coerce")
        cm["covid_stringency_z"] = _zscore(cm["covid_stringency_index"])
    if "reproduction_rate" in cm.columns:
        cm["covid_reproduction_rate"] = pd.to_numeric(cm["reproduction_rate"], errors="coerce")
        cm["covid_reproduction_z"] = _zscore(cm["covid_reproduction_rate"])
    for raw, outname in [("positive_rate", "covid_positive_rate"), ("hosp_patients_per_million", "covid_hosp_patients_pm"), ("icu_patients_per_million", "covid_icu_patients_pm")]:
        if raw in cm.columns:
            cm[outname] = pd.to_numeric(cm[raw], errors="coerce")
            cm[f"{outname}_z"] = _zscore(cm[outname])
    return panel.merge(cm, on="month", how="left")


def _weekly_covid(covid_csv: str | None, covid_location: str) -> pd.DataFrame:
    if not covid_csv or not Path(covid_csv).exists():
        return pd.DataFrame({"week": []})
    covid = pd.read_csv(covid_csv)
    if "location" in covid.columns:
        covid = covid[covid["location"].astype(str) == str(covid_location)].copy()
    if covid.empty or "date" not in covid.columns:
        return pd.DataFrame({"week": []})
    covid["date"] = pd.to_datetime(covid["date"])
    covid["week"] = covid["date"].dt.to_period("W-SUN").astype(str)
    val_cols = [c for c in ["new_cases_smoothed_per_million", "new_deaths_smoothed_per_million", "new_cases_per_million", "new_deaths_per_million", "stringency_index", "reproduction_rate"] if c in covid.columns]
    wk = covid.groupby("week", as_index=False)[val_cols].mean() if val_cols else covid[["week"]].drop_duplicates()
    if "new_cases_smoothed_per_million" in wk.columns:
        wk["covid_cases_index"] = np.log1p(wk["new_cases_smoothed_per_million"].fillna(0).clip(lower=0))
    elif "new_cases_per_million" in wk.columns:
        wk["covid_cases_index"] = np.log1p(wk["new_cases_per_million"].fillna(0).clip(lower=0))
    if "new_deaths_smoothed_per_million" in wk.columns:
        wk["covid_deaths_index"] = np.log1p(wk["new_deaths_smoothed_per_million"].fillna(0).clip(lower=0))
    elif "new_deaths_per_million" in wk.columns:
        wk["covid_deaths_index"] = np.log1p(wk["new_deaths_per_million"].fillna(0).clip(lower=0))
    if "stringency_index" in wk.columns:
        wk["covid_stringency_index"] = pd.to_numeric(wk["stringency_index"], errors="coerce")
    if "reproduction_rate" in wk.columns:
        wk["covid_reproduction_rate"] = pd.to_numeric(wk["reproduction_rate"], errors="coerce")
    for c in [x for x in wk.columns if x.startswith("covid_") and not x.endswith("_z")]:
        wk[f"{c}_z"] = _zscore(wk[c])
    return wk


def _load_google_mobility_global(mobility_csv: str | None) -> pd.DataFrame:
    if not mobility_csv or not Path(mobility_csv).exists():
        return pd.DataFrame({"month": []})
    mob = pd.read_csv(mobility_csv)
    if "date" not in mob.columns:
        return pd.DataFrame({"month": []})
    mob["date"] = pd.to_datetime(mob["date"])
    mob["month"] = mob["date"].dt.to_period("M").astype(str)
    cols = [c for c in mob.columns if c.endswith("_percent_change_from_baseline")]
    if not cols:
        return pd.DataFrame({"month": []})
    gm = mob.groupby("month", as_index=False)[cols].mean()
    rename = {c: "global_mobility_" + c.replace("_percent_change_from_baseline", "") for c in cols}
    return gm.rename(columns=rename)


def _attach_google_mobility_global(panel: pd.DataFrame, mobility_csv: str | None, enabled: bool) -> pd.DataFrame:
    if not enabled:
        return panel
    gm = _load_google_mobility_global(mobility_csv)
    if gm.empty or "month" not in gm.columns:
        print("[WARN] Google Mobility global-average proxy not merged: missing/empty file.")
        return panel
    return panel.merge(gm, on="month", how="left")


def _load_customers(data_root: Path) -> pd.DataFrame:
    path = data_root / "hm" / "processed" / "hm_customers.parquet"
    if not path.exists():
        return pd.DataFrame()
    cust = pd.read_parquet(path)
    if "customer_id" in cust.columns:
        cust["customer_id"] = cust["customer_id"].astype(str)
    return cust


def _score_columns(df: pd.DataFrame) -> list[str]:
    # Raw semantic score columns only.  Exclude intermediate pos/neg/margin/confidence and copula copies.
    return [
        c for c in df.columns
        if c.endswith("_score")
        and not c.endswith("_pos_score")
        and not c.endswith("_neg_score")
        and not c.startswith("copula_calibrated_")
    ]


def _weighted_axis_panel(item_month: pd.DataFrame, score_cols: list[str], dim_name: str = "axis_dim") -> pd.DataFrame:
    rows = []
    for sc in score_cols:
        tmp = item_month[["month", "sales_count", sc]].copy()
        tmp[sc] = pd.to_numeric(tmp[sc], errors="coerce").fillna(0.0)
        tmp["weighted"] = tmp[sc] * pd.to_numeric(tmp["sales_count"], errors="coerce").fillna(0.0)
        g = tmp.groupby("month", as_index=False).agg(weighted_sum=("weighted", "sum"), sales_count=("sales_count", "sum"))
        g["style_share"] = g["weighted_sum"] / g["sales_count"].replace(0, np.nan)
        g[dim_name] = sc.replace("_score", "")
        rows.append(g[[dim_name, "month", "style_share", "sales_count"]])
    return pd.concat(rows, ignore_index=True) if rows else _schema_df([dim_name, "month", "style_share", "sales_count"])


def build_panels(args, items: pd.DataFrame, semantic_scores: pd.DataFrame, protos: pd.DataFrame, out_dir: Path) -> dict:
    data_root = Path(args.data_root)
    tx = _load_transactions(data_root)
    meta = _make_metadata(items)
    score_cols = _score_columns(semantic_scores)
    proto_cols = [c for c in protos.columns if c.endswith("_proto")]

    item_month = _dense_item_month(meta[["article_id"]], tx, dense=bool(args.dense_item_month_panel))
    item_month = item_month.merge(meta, on="article_id", how="left")
    item_month = item_month.merge(semantic_scores[["article_id"] + score_cols], on="article_id", how="left")
    item_month = item_month.merge(protos[["article_id"] + proto_cols], on="article_id", how="left")
    for c in score_cols:
        item_month[c] = pd.to_numeric(item_month[c], errors="coerce").fillna(0.0)
    item_month = _attach_covid(item_month, args.covid_csv, args.covid_location)
    item_month = _attach_google_mobility_global(item_month, args.google_mobility_csv, bool(args.use_google_mobility))
    item_month.to_parquet(out_dir / "item_monthly_panel.parquet", index=False)
    item_month.to_parquet(out_dir / "item_monthly_sales.parquet", index=False)

    # Month calendar for diagnostics.
    month_calendar = tx.groupby("month", as_index=False).size().rename(columns={"size": "transactions"})
    month_calendar = _attach_covid(month_calendar, args.covid_csv, args.covid_location)
    month_calendar = _attach_google_mobility_global(month_calendar, args.google_mobility_csv, bool(args.use_google_mobility))
    month_calendar.to_parquet(out_dir / "month_calendar.parquet", index=False)

    # Native H&M category panel.
    cat_fields = [c for c in ["product_group_name", "product_type_name", "garment_group_name", "department_name"] if c in item_month.columns]
    category_panel = _schema_df(["category_field", "category_value", "month", "sales_count", "total_sales", "sales_share"])
    cat_rows = []
    total_by_month = item_month.groupby("month", as_index=False)["sales_count"].sum().rename(columns={"sales_count": "total_sales"})
    for cf in cat_fields:
        g = item_month.groupby([cf, "month"], dropna=False, as_index=False)["sales_count"].sum().rename(columns={cf: "category_value"})
        g["category_field"] = cf
        g = g.merge(total_by_month, on="month", how="left")
        g["sales_share"] = g["sales_count"] / g["total_sales"].replace(0, np.nan)
        cat_rows.append(g[["category_field", "category_value", "month", "sales_count", "total_sales", "sales_share"]])
    if cat_rows:
        category_panel = pd.concat(cat_rows, ignore_index=True)
        category_panel = _attach_covid(category_panel, args.covid_csv, args.covid_location)
        category_panel = _attach_google_mobility_global(category_panel, args.google_mobility_csv, bool(args.use_google_mobility))
    category_panel.to_parquet(out_dir / "category_monthly_panel.parquet", index=False)

    # Prototype-month panel, strictly from native routed prompt prototypes.
    proto_panel = _schema_df(["dimension", "prototype", "month", "sales_count", "total_sales", "sales_share"])
    proto_rows = []
    for pc in proto_cols:
        dim = pc[:-6]
        tmp = item_month.dropna(subset=[pc]).copy()
        if tmp.empty:
            continue
        g = tmp.groupby([pc, "month"], as_index=False)["sales_count"].sum().rename(columns={pc: "prototype"})
        g["dimension"] = dim
        g = g.merge(total_by_month, on="month", how="left")
        g["sales_share"] = g["sales_count"] / g["total_sales"].replace(0, np.nan)
        proto_rows.append(g[["dimension", "prototype", "month", "sales_count", "total_sales", "sales_share"]])
    if proto_rows:
        proto_panel = pd.concat(proto_rows, ignore_index=True)
        proto_panel = _attach_covid(proto_panel, args.covid_csv, args.covid_location)
        proto_panel = _attach_google_mobility_global(proto_panel, args.google_mobility_csv, bool(args.use_google_mobility))
    proto_panel.to_parquet(out_dir / "prototype_monthly_panel.parquet", index=False)

    # Axis/composite month panel.
    axis_panel = _weighted_axis_panel(item_month, score_cols, dim_name="knowledge_dim")
    if len(axis_panel):
        axis_panel = _attach_covid(axis_panel, args.covid_csv, args.covid_location)
        axis_panel = _attach_google_mobility_global(axis_panel, args.google_mobility_csv, bool(args.use_google_mobility))
    axis_panel.to_parquet(out_dir / "axis_monthly_panel.parquet", index=False)
    # Backward-compatible alias for old analysis/figure code.
    axis_panel.to_parquet(out_dir / "knowledge_monthly_panel.parquet", index=False)

    # User monthly preference panel.
    user_panel = _schema_df(["customer_id", "month", "transactions"])
    if "customer_id" in tx.columns and score_cols:
        utx = tx[["customer_id", "article_id", "month"]].merge(semantic_scores[["article_id"] + score_cols], on="article_id", how="left")
        for c in score_cols:
            utx[c] = pd.to_numeric(utx[c], errors="coerce").fillna(0.0)
        agg = {"article_id": "size"}
        prefs = utx.groupby(["customer_id", "month"], as_index=False).agg(transactions=("article_id", "size"))
        for c in score_cols:
            pref = utx.groupby(["customer_id", "month"], as_index=False)[c].mean().rename(columns={c: c.replace("_score", "_pref")})
            prefs = prefs.merge(pref, on=["customer_id", "month"], how="left")
        user_panel = prefs
        cap = int(args.max_user_month_rows or 0)
        if cap > 0 and len(user_panel) > cap:
            user_panel = user_panel.sample(n=cap, random_state=args.seed).reset_index(drop=True)
        user_panel = _attach_covid(user_panel, args.covid_csv, args.covid_location)
        user_panel = _attach_google_mobility_global(user_panel, args.google_mobility_csv, bool(args.use_google_mobility))
    customers = _load_customers(data_root)
    customer_cols = [c for c in ["customer_id", "age", "club_member_status", "fashion_news_frequency", "FN", "Active"] if c in customers.columns]
    if customer_cols:
        customers[customer_cols].to_parquet(out_dir / "customer_social_attributes.parquet", index=False)
        if len(user_panel) and "customer_id" in user_panel.columns:
            user_panel = user_panel.merge(customers[customer_cols], on="customer_id", how="left")
    user_panel.to_parquet(out_dir / "user_monthly_panel.parquet", index=False)

    # Channel and channel-axis panels.
    channel_panel = _schema_df(["sales_channel_id", "month", "transactions", "total_transactions", "channel_share"])
    channel_axis_panel = _schema_df(["sales_channel_id", "month", "style_share", "transactions", "knowledge_dim"])
    if "sales_channel_id" in tx.columns:
        ch = tx.copy()
        ch["sales_channel_id"] = ch["sales_channel_id"].astype(str)
        month_tot = ch.groupby("month", as_index=False).size().rename(columns={"size": "total_transactions"})
        channel_panel = ch.groupby(["sales_channel_id", "month"], as_index=False).size().rename(columns={"size": "transactions"})
        channel_panel = channel_panel.merge(month_tot, on="month", how="left")
        channel_panel["channel_share"] = channel_panel["transactions"] / channel_panel["total_transactions"].replace(0, np.nan)
        channel_panel = _attach_covid(channel_panel, args.covid_csv, args.covid_location)
        channel_panel = _attach_google_mobility_global(channel_panel, args.google_mobility_csv, bool(args.use_google_mobility))
        if score_cols:
            chs = ch[["article_id", "month", "sales_channel_id"]].merge(semantic_scores[["article_id"] + score_cols], on="article_id", how="left")
            for c in score_cols:
                chs[c] = pd.to_numeric(chs[c], errors="coerce").fillna(0.0)
            rows = []
            for sc in score_cols:
                tmp = chs.groupby(["sales_channel_id", "month"], as_index=False).agg(style_share=(sc, "mean"), transactions=("article_id", "size"))
                tmp["knowledge_dim"] = sc.replace("_score", "")
                rows.append(tmp)
            channel_axis_panel = pd.concat(rows, ignore_index=True) if rows else channel_axis_panel
            if len(channel_axis_panel):
                channel_axis_panel = _attach_covid(channel_axis_panel, args.covid_csv, args.covid_location)
                channel_axis_panel = _attach_google_mobility_global(channel_axis_panel, args.google_mobility_csv, bool(args.use_google_mobility))
    channel_panel.to_parquet(out_dir / "channel_monthly_panel.parquet", index=False)
    channel_axis_panel.to_parquet(out_dir / "channel_axis_monthly_panel.parquet", index=False)
    channel_axis_panel.to_parquet(out_dir / "channel_style_monthly_panel.parquet", index=False)

    # Weekly axis panel for lead-lag analysis.
    weekly_axis = _schema_df(["week", "style_share", "transactions", "knowledge_dim"])
    if score_cols:
        wtx = tx[["article_id", "t_dat"]].copy()
        wtx["week"] = wtx["t_dat"].dt.to_period("W-SUN").astype(str)
        wtx = wtx.merge(semantic_scores[["article_id"] + score_cols], on="article_id", how="left")
        for c in score_cols:
            wtx[c] = pd.to_numeric(wtx[c], errors="coerce").fillna(0.0)
        rows = []
        for sc in score_cols:
            tmp = wtx.groupby("week", as_index=False).agg(style_share=(sc, "mean"), transactions=("article_id", "size"))
            tmp["knowledge_dim"] = sc.replace("_score", "")
            rows.append(tmp)
        weekly_axis = pd.concat(rows, ignore_index=True) if rows else weekly_axis
        wk_covid = _weekly_covid(args.covid_csv, args.covid_location)
        if len(weekly_axis) and len(wk_covid):
            weekly_axis = weekly_axis.merge(wk_covid, on="week", how="left")
    weekly_axis.to_parquet(out_dir / "axis_weekly_panel.parquet", index=False)
    weekly_axis.to_parquet(out_dir / "knowledge_weekly_panel.parquet", index=False)

    return {
        "item_monthly_panel": len(item_month),
        "category_monthly_panel": len(category_panel),
        "prototype_monthly_panel": len(proto_panel),
        "axis_monthly_panel": len(axis_panel),
        "user_monthly_panel": len(user_panel),
        "channel_monthly_panel": len(channel_panel),
        "channel_axis_monthly_panel": len(channel_axis_panel),
        "axis_weekly_panel": len(weekly_axis),
        "month_calendar": len(month_calendar),
    }


def run_social_inference(args) -> None:
    data_root = Path(args.data_root)
    out_dir = ensure_dir(Path(args.social_output_root))
    items, base = _load_items_and_features(data_root, args.item_prefix)
    knowledge, kmask = _load_knowledge(data_root, args.item_prefix, bool(args.use_knowledge), bool(args.include_temporal))
    emb, model, cfg = compute_item_embeddings_and_model(args, items, base, knowledge, kmask)

    emb_cols = [f"embedding_{i}" for i in range(emb.shape[1])]
    emb_df = pd.DataFrame(emb, columns=emb_cols)
    emb_df.insert(0, "article_id", items["article_id"].values)
    emb_df.to_parquet(out_dir / "item_embeddings.parquet", index=False)

    meta = _make_metadata(items)
    meta.to_parquet(out_dir / "item_metadata.parquet", index=False)

    # Preserve original routed knowledge scores, but do not treat them as continuous social axes.
    routed_scores = _read_routed_knowledge_scores(data_root, items)
    routed_scores.to_parquet(out_dir / "item_knowledge_scores.parquet", index=False)

    protos, proto_long = _knowledge_prototypes(data_root, items, tau=float(args.prototype_softmax_tau))
    protos.to_parquet(out_dir / "item_knowledge_prototypes.parquet", index=False)
    proto_long.to_parquet(out_dir / "item_knowledge_prototypes_long.parquet", index=False)

    # Generate fixed generic clothing axis prompts with local Qwen LLM, cache them, encode, and score items.
    # Recommended route: item_text.  We encode both H&M item metadata text and axis prompts with the same
    # text encoder, then directly compute cosine similarities.  This bypasses knowledge_projector and avoids
    # the 2560-vs-768 mismatch caused by feeding Qwen embeddings into a CLIP-trained projector.
    axis_score_space = str(args.axis_score_space).strip().lower()
    if axis_score_space in {"item_text", "text", "metadata_text"}:
        text_encoder = _resolve_axis_text_encoder(args.text_encoder, expected_dim=None, score_space="item_text")
        prompt_meta, prompt_emb, prompt_schema = _load_or_encode_axis_prompt_embeddings(
            args, data_root, out_dir, expected_prompt_dim=None, score_space="item_text"
        )
        item_axis_emb = _load_or_encode_item_axis_embeddings(
            args, data_root, out_dir, items, text_encoder=text_encoder, expected_dim=int(prompt_emb.shape[1])
        )
        np.save(out_dir / "item_axis_text_embeddings.npy", item_axis_emb.astype(np.float32))
        np.save(out_dir / "style_axis_prompt_embeddings.used.npy", prompt_emb.astype(np.float32))
        axis_scores = _axis_scores_from_embeddings(items, item_axis_emb, prompt_meta, prompt_emb, tau=float(args.axis_score_tau))
    elif axis_score_space in {"projector", "model", "model_embedding"}:
        expected_prompt_dim = _knowledge_projector_input_dim(model)
        if expected_prompt_dim is None:
            expected_prompt_dim = int(emb.shape[1])
        prompt_meta, prompt_emb, prompt_schema = _load_or_encode_axis_prompt_embeddings(
            args, data_root, out_dir, expected_prompt_dim=expected_prompt_dim, score_space="projector"
        )
        prompt_proj = _project_axis_prompt_embeddings(model, prompt_emb, item_emb_dim=emb.shape[1])
        np.save(out_dir / "style_axis_prompt_projected_embeddings.npy", prompt_proj.astype(np.float32))
        axis_scores = _axis_scores_from_embeddings(items, emb, prompt_meta, prompt_proj, tau=float(args.axis_score_tau))
    else:
        raise ValueError(f"Unknown --axis_score_space={args.axis_score_space}. Use item_text or projector.")
    axis_scores.to_parquet(out_dir / "item_axis_scores.parquet", index=False)

    composite_scores = _composite_scores(axis_scores)
    composite_scores.to_parquet(out_dir / "item_composite_scores.parquet", index=False)

    semantic_scores = axis_scores[["article_id"] + [c for c in axis_scores.columns if c.endswith("_score") and not c.endswith("_pos_score") and not c.endswith("_neg_score")]].merge(
        composite_scores, on="article_id", how="left"
    )
    if bool(args.use_copula_calibration):
        semantic_scores = _apply_copula_calibration(semantic_scores, [c for c in semantic_scores.columns if c.endswith("_score")])
    semantic_scores.to_parquet(out_dir / "item_semantic_scores.parquet", index=False)

    panel_sizes = build_panels(args, items, semantic_scores, protos, out_dir)
    manifest = {
        "stage": "social_inference_axis_v2",
        "data_root": str(data_root),
        "social_output_root": str(out_dir),
        "item_prefix": args.item_prefix,
        "use_knowledge": bool(args.use_knowledge),
        "checkpoint": str(Path(args.checkpoint) if args.checkpoint else _default_checkpoint(args)),
        "num_items": int(len(items)),
        "embedding_dim": int(emb.shape[1]),
        "native_prototype_dimensions": NATIVE_PROTO_DIMS,
        "prototype_columns": [c for c in protos.columns if c.endswith("_proto")],
        "axis_ids": AXIS_IDS,
        "composite_ids": COMPOSITE_IDS,
        "semantic_score_columns": [c for c in semantic_scores.columns if c.endswith("_score")],
        "axis_prompt_source": prompt_schema.get("source", "unknown"),
        "axis_prompt_json": str(Path(args.axis_prompt_json) if args.axis_prompt_json else data_root / "cross_domain" / "style_axis_prompts.json"),
        "text_encoder": args.text_encoder,
        "qwen_model": args.qwen_model,
        "axis_scores_note": "Axis prompts are post-hoc semantic probes on behavior-calibrated item embeddings; they are not native routed prompt prototypes.",
        "panel_sizes": panel_sizes,
        "covid_csv": args.covid_csv,
        "covid_location": args.covid_location,
        "use_copula_calibration": bool(args.use_copula_calibration),
        "use_google_mobility": bool(args.use_google_mobility),
        "google_mobility_csv": args.google_mobility_csv,
        "external_exposure_note": "H&M has no interpretable customer geography; COVID and optional mobility are common time-series proxies.",
    }
    save_json(manifest, out_dir / "social_inference_manifest.json")
    print("========== Social inference finished ==========")
    for k, v in panel_sizes.items():
        print(f"{k}: {v:,} rows")
    print(f"Axis prompts: {manifest['axis_prompt_json']} ({manifest['axis_prompt_source']})")
    print(f"Output: {out_dir}")
    print("==============================================")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data")
    p.add_argument("--output_root", default="./output")
    p.add_argument("--social_output_root", default="./social_output/k")
    p.add_argument("--item_prefix", default="clip")
    p.add_argument("--use_knowledge", type=parse_bool, default=True)
    p.add_argument("--include_temporal", type=parse_bool, default=False)
    p.add_argument("--checkpoint", default="", help="Path to trained Stage-2 checkpoint. Defaults to output_root/clip_transformer_{k,nok}/best_model.pt.")
    p.add_argument("--allow_base_fallback", type=parse_bool, default=False)
    p.add_argument("--cuda_id", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--item_adapter_layers", type=int, default=2)
    p.add_argument("--item_adapter_heads", type=int, default=4)
    p.add_argument("--user_layers", type=int, default=2)
    p.add_argument("--user_heads", type=int, default=4)
    p.add_argument("--ffn_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_history_items", type=int, default=20)
    p.add_argument("--use_two_tower", type=int, default=0)
    p.add_argument("--item_encode_chunk_size", type=int, default=2048)
    p.add_argument("--candidate_chunk_size", type=int, default=4)
    p.add_argument("--transformer_injection", type=parse_bool, default=False)
    p.add_argument("--transformer_injection_layers", type=int, default=0)
    p.add_argument("--transformer_injection_strength", type=float, default=1.0)

    # Axis prompt generation/scoring.
    p.add_argument("--auto_generate_axis_prompts", type=parse_bool, default=True, help="Generate fixed generic-clothing axis prompts with local Qwen if style_axis_prompts.json is absent.")
    p.add_argument("--axis_prompt_json", default="", help="Optional path to style_axis_prompts.json. Defaults to data/cross_domain/style_axis_prompts.json.")
    p.add_argument("--axis_prompt_force_regenerate", type=parse_bool, default=False)
    p.add_argument("--mock_qwen_axis_prompts", type=parse_bool, default=False, help="Use built-in prompt bank instead of calling Qwen; useful for debugging.")
    p.add_argument("--qwen_model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--axis_qwen_max_input_tokens", type=int, default=4096)
    p.add_argument("--axis_qwen_max_new_tokens", type=int, default=512)
    p.add_argument("--axis_qwen_temperature", type=float, default=0.2)
    p.add_argument("--axis_prompts_per_polarity", type=int, default=4)
    p.add_argument("--axis_score_space", default="item_text", choices=["item_text", "projector", "model", "model_embedding"], help="Where to compute axis similarities. item_text encodes H&M item metadata and prompts with the same text encoder; projector is the legacy knowledge_projector route.")
    p.add_argument("--text_encoder", default="auto", help="Text encoder for direct item_text axis probing. With --axis_score_space item_text, auto selects Qwen/Qwen3-Embedding-4B. With --axis_score_space projector, auto selects a CLIP encoder compatible with the checkpoint knowledge_dim.")
    p.add_argument("--axis_encode_batch_size", type=int, default=32)
    p.add_argument("--axis_text_max_length", type=int, default=256)
    p.add_argument("--axis_score_tau", type=float, default=1.0)
    p.add_argument("--prototype_softmax_tau", type=float, default=0.2)

    p.add_argument("--use_copula_calibration", type=parse_bool, default=False)
    p.add_argument("--dense_item_month_panel", type=parse_bool, default=True)
    p.add_argument("--max_user_month_rows", type=int, default=2000000)
    p.add_argument("--covid_csv", default="")
    p.add_argument("--covid_location", default="World")
    p.add_argument("--use_google_mobility", type=parse_bool, default=False, help="Optional mechanism proxy. If enabled, use global average Google Mobility only; no region matching is attempted.")
    p.add_argument("--google_mobility_csv", default="")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_social_inference(args)


if __name__ == "__main__":
    main()
