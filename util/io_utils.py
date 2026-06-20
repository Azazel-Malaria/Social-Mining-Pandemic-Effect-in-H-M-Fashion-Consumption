import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(cuda_id: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_id}")
    return torch.device("cpu")


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def article_to_str(article_id: Any) -> str:
    s = str(article_id)
    if s.endswith('.0'):
        s = s[:-2]
    return s.zfill(10)


def find_hm_image_path(image_root: str | Path, article_id: Any) -> str:
    """Return the canonical Kaggle H&M image path if it exists.

    Kaggle stores images as images/010/0108775015.jpg, where the first
    three digits of article_id form the sub-directory name.
    """
    image_root = Path(image_root)
    aid = article_to_str(article_id)
    candidates = [
        image_root / aid[:3] / f"{aid}.jpg",
        image_root / f"{aid}.jpg",
        image_root / aid[:3] / f"{aid}.png",
        image_root / f"{aid}.png",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(candidates[0])


def parse_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).lower() in {"1", "true", "yes", "y"}


def format_float_dict(metrics: Dict[str, float], ndigits: int = 3) -> Dict[str, float]:
    return {k: round(float(v), ndigits) for k, v in metrics.items()}


def move_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    return batch
