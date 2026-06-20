from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import pandas as pd

# Social-science style dimensions used for downstream analysis and figures.
# They are deliberately separate from the model auxiliary style head so changing
# this list will not change checkpoint shape.
SOCIAL_STYLES: List[str] = [
    "formal",
    "office",
    "comfort",
    "homewear",
    "casual",
    "value",
    "sportswear",
    "outdoor",
    "minimal",
    "romantic",
]

SOCIAL_STYLE_KEYWORDS: Dict[str, List[str]] = {
    "formal": ["formal", "tailored", "suit", "blazer", "dress shirt", "occasion", "party", "smart"],
    "office": ["office", "business", "work", "workwear", "commuting", "trouser", "chinos", "blazer", "shirt"],
    "comfort": ["comfort", "comfortable", "soft", "stretch", "relaxed", "loose", "breathable", "easy", "lounge"],
    "homewear": ["home", "lounge", "loungewear", "night", "pajama", "pyjama", "sleep", "robe", "indoor"],
    "casual": ["casual", "daily", "everyday", "relaxed", "denim", "tee", "t-shirt", "sweatshirt", "hoodie"],
    "value": ["value", "affordable", "cheap", "price", "budget", "durable", "basic", "essential", "multipack"],
    "sportswear": ["sport", "sportswear", "active", "training", "running", "yoga", "leggings", "performance"],
    "outdoor": ["outdoor", "hiking", "rain", "wind", "waterproof", "coat", "jacket", "warm", "winter"],
    "minimal": ["minimal", "clean", "simple", "neutral", "plain", "solid", "monochrome", "classic"],
    "romantic": ["romantic", "lace", "floral", "ruffle", "frill", "embroidered", "pink", "dress"],
}

TEXT_COLS = ["prod_name", "product_type_name", "product_group_name", "colour_group_name", "detail_desc"]


def clean_text(text: object) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\-\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def join_item_text(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in TEXT_COLS if c in df.columns]
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1)


def infer_social_style_scores(text: object) -> np.ndarray:
    text = clean_text(text)
    scores = np.ones(len(SOCIAL_STYLES), dtype=np.float32) * 0.05
    for idx, style in enumerate(SOCIAL_STYLES):
        for kw in SOCIAL_STYLE_KEYWORDS[style]:
            if kw in text:
                scores[idx] += 1.0
    scores = scores / max(float(scores.sum()), 1e-8)
    return scores.astype(np.float32)


def add_social_style_scores(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Append formal_score/comfort_score/... columns inferred from item text.

    The heuristic is intentionally simple and transparent; it is used only for
    social analysis variables/figures, not for training the backbone.
    """
    out = df.copy()
    texts = join_item_text(out)
    arr = np.vstack([infer_social_style_scores(x) for x in texts]) if len(out) else np.zeros((0, len(SOCIAL_STYLES)), dtype=np.float32)
    for j, style in enumerate(SOCIAL_STYLES):
        out[f"{prefix}{style}_score"] = arr[:, j].astype(np.float32)
    return out
