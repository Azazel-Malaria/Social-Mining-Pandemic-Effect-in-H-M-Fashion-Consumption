from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from util.io_utils import ensure_dir, resolve_device


def build_detail_text(row: pd.Series) -> str:
    """Use H&M article.csv detail_desc as the main text source.

    The user explicitly wanted SigLIP/OpenCLIP text to be based on article.csv
    detail_desc. We only fall back to short metadata when detail_desc is empty.
    """
    detail = row.get("detail_desc", "")
    if pd.notna(detail) and str(detail).strip():
        return str(detail).strip()
    fallback_fields = ["prod_name", "product_type_name", "product_group_name", "colour_group_name"]
    parts = []
    for f in fallback_fields:
        v = row.get(f, "")
        if pd.notna(v) and str(v).strip():
            parts.append(str(v).strip())
    return "; ".join(parts) if parts else "H&M fashion product"


def load_image(path: str | Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), color=(0, 0, 0))


def batched(xs: list, batch_size: int) -> Iterable[tuple[int, list]]:
    for start in range(0, len(xs), batch_size):
        yield start, xs[start:start + batch_size]


def extract_feature_tensor(output, name: str) -> torch.Tensor:
    """Convert Hugging Face/OpenCLIP-style outputs into a 2D feature tensor.

    Some transformers versions return a tensor from get_*_features(), while some
    SigLIP variants return BaseModelOutputWithPooling.  This helper keeps the
    precompute script robust across versions.
    """
    if torch.is_tensor(output):
        feat = output
    elif hasattr(output, "pooler_output") and output.pooler_output is not None:
        feat = output.pooler_output
    elif hasattr(output, "image_embeds") and output.image_embeds is not None:
        feat = output.image_embeds
    elif hasattr(output, "text_embeds") and output.text_embeds is not None:
        feat = output.text_embeds
    elif hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        hidden = output.last_hidden_state
        if hidden.ndim == 3:
            feat = hidden[:, 0]
        elif hidden.ndim == 2:
            feat = hidden
        else:
            raise RuntimeError(f"{name} last_hidden_state has unsupported shape: {tuple(hidden.shape)}")
    elif isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        feat = output[0]
        if feat.ndim == 3:
            feat = feat[:, 0]
    else:
        raise RuntimeError(f"Could not extract tensor features from {name} output of type {type(output)}.")

    if feat.ndim == 3:
        feat = feat[:, 0]
    if feat.ndim != 2:
        raise RuntimeError(f"{name} feature tensor must be 2D [batch, dim], got {tuple(feat.shape)}.")
    return feat


class SiglipBackend:
    def __init__(self, model_name: str, device: torch.device, dtype: torch.dtype):
        from transformers import AutoProcessor, AutoModel
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
        self.device = device

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if hasattr(self.model, "get_image_features"):
            out = self.model.get_image_features(**inputs)
        else:
            out = self.model(**inputs)
        feat = extract_feature_tensor(out, "SigLIP image")
        return F.normalize(feat.float(), dim=-1).cpu()

    @torch.no_grad()
    def encode_texts(self, texts: list[str], max_length: int) -> torch.Tensor:
        inputs = self.processor(text=texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        # Some processors return pixel_values only for images; keep tensor values only.
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        if hasattr(self.model, "get_text_features"):
            out = self.model.get_text_features(**inputs)
        else:
            out = self.model(**inputs)
        feat = extract_feature_tensor(out, "SigLIP text")
        return F.normalize(feat.float(), dim=-1).cpu()


class OpenCLIPBackend:
    def __init__(self, model_name: str, pretrained: str, device: torch.device, dtype: torch.dtype):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(device=device, dtype=dtype).eval()
        self.device = device

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        x = torch.stack([self.preprocess(img) for img in images], dim=0).to(self.device)
        feat = self.model.encode_image(x)
        return F.normalize(feat.float(), dim=-1).cpu()

    @torch.no_grad()
    def encode_texts(self, texts: list[str], max_length: int) -> torch.Tensor:
        # open_clip tokenizer has its own context length. Truncate input strings roughly to avoid very long detail_desc.
        texts = [t[: max_length * 8] for t in texts]
        tokens = self.tokenizer(texts).to(self.device)
        feat = self.model.encode_text(tokens)
        return F.normalize(feat.float(), dim=-1).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--mm_backbone", choices=["siglip", "openclip"], default="siglip")
    parser.add_argument("--model_name", default="google/siglip-base-patch16-224")
    parser.add_argument("--openclip_pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--output_prefix", default="", help="Default is mm_backbone, e.g. siglip or openclip.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--text_batch_size", type=int, default=256)
    parser.add_argument("--text_max_length", type=int, default=64)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--limit_items", type=int, default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    hm_proc = data_root / "hm" / "processed"
    item_path = hm_proc / "hm_item_text.parquet"
    if not item_path.exists():
        raise FileNotFoundError(f"Missing {item_path}. Run prepare_hm.sh first.")
    items = pd.read_parquet(item_path)
    items["article_id"] = items["article_id"].astype(str)
    if args.limit_items is not None:
        items = items.head(args.limit_items).reset_index(drop=True)
    article_ids = items["article_id"].tolist()
    image_paths = items["image_path"].astype(str).tolist()
    texts = [build_detail_text(row) for _, row in items.iterrows()]

    device = resolve_device(args.cuda_id)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    prefix = args.output_prefix.strip() or args.mm_backbone

    if args.mm_backbone == "siglip":
        backend = SiglipBackend(args.model_name, device, dtype)
    else:
        backend = OpenCLIPBackend(args.model_name, args.openclip_pretrained, device, dtype)

    image_chunks = []
    for _, paths in tqdm(list(batched(image_paths, args.batch_size)), desc=f"Encode {args.mm_backbone} images"):
        imgs = [load_image(p) for p in paths]
        image_chunks.append(backend.encode_images(imgs))
    image_features = torch.cat(image_chunks, dim=0).numpy().astype(np.float32)

    text_chunks = []
    for _, batch_texts in tqdm(list(batched(texts, args.text_batch_size)), desc=f"Encode {args.mm_backbone} detail_desc"):
        text_chunks.append(backend.encode_texts(batch_texts, args.text_max_length))
    text_features = torch.cat(text_chunks, dim=0).numpy().astype(np.float32)

    ensure_dir(hm_proc)
    np.save(hm_proc / f"{prefix}_image_features.npy", image_features)
    np.save(hm_proc / f"{prefix}_text_features.npy", text_features)
    idx = pd.DataFrame({
        "article_id": article_ids,
        "feature_row": np.arange(len(article_ids), dtype=np.int64),
        "mm_backbone": args.mm_backbone,
        "model_name": args.model_name,
        "openclip_pretrained": args.openclip_pretrained if args.mm_backbone == "openclip" else "",
        "image_dim": int(image_features.shape[1]),
        "text_dim": int(text_features.shape[1]),
        "text_source": "detail_desc_fallback_metadata",
    })
    idx.to_parquet(hm_proc / f"{prefix}_feature_index.parquet", index=False)
    print("========== Multimodal feature precompute finished ==========")
    print(f"backbone: {args.mm_backbone}")
    print(f"model: {args.model_name}")
    print(f"items: {len(article_ids)}")
    print(f"image features: {hm_proc / f'{prefix}_image_features.npy'} {image_features.shape}")
    print(f"text features:  {hm_proc / f'{prefix}_text_features.npy'} {text_features.shape}")
    print(f"index:          {hm_proc / f'{prefix}_feature_index.parquet'}")
    print("===========================================================")


if __name__ == "__main__":
    main()
