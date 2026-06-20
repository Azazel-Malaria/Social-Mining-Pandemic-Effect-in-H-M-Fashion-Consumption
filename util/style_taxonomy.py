from __future__ import annotations

import re
from typing import Dict, Iterable, List

import numpy as np

STYLES: List[str] = [
    "basic",
    "casual",
    "formal",
    "sportswear",
    "streetwear",
    "minimal",
    "vintage",
    "romantic",
    "workwear",
    "homewear",
]

STYLE_KEYWORDS: Dict[str, List[str]] = {
    "basic": ["basic", "plain", "essential", "solid", "everyday", "regular", "jersey"],
    "casual": ["casual", "relaxed", "daily", "denim", "tee", "t-shirt", "sweatshirt", "hoodie"],
    "formal": ["formal", "suit", "blazer", "shirt", "tailored", "dress", "party", "occasion"],
    "sportswear": ["sport", "sportswear", "active", "training", "running", "yoga", "leggings"],
    "streetwear": ["street", "oversized", "graphic", "cargo", "hoodie", "sneaker", "urban"],
    "minimal": ["minimal", "clean", "simple", "neutral", "monochrome", "fine-knit"],
    "vintage": ["vintage", "retro", "washed", "classic", "corduroy", "flared"],
    "romantic": ["romantic", "lace", "floral", "ruffle", "frill", "embroidered", "pink"],
    "workwear": ["workwear", "office", "business", "smart", "trouser", "chinos", "blazer"],
    "homewear": ["home", "lounge", "night", "pajama", "pyjama", "sleep", "robe", "soft"],
}


def clean_text(text: object) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\-\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def infer_style_scores(text: object) -> np.ndarray:
    text = clean_text(text)
    scores = np.ones(len(STYLES), dtype=np.float32) * 0.05
    for idx, style in enumerate(STYLES):
        for kw in STYLE_KEYWORDS[style]:
            if kw in text:
                scores[idx] += 1.0
    return scores


def infer_style_probs(text: object) -> np.ndarray:
    scores = infer_style_scores(text)
    probs = scores / max(scores.sum(), 1e-8)
    return probs.astype(np.float32)


def style_columns(prefix: str = "style_") -> List[str]:
    return [f"{prefix}{s}" for s in STYLES]
