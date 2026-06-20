from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import torch


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def topk_metrics(scores, labels, mask=None, k: int = 12) -> Dict[str, float]:
    """Compute ranking metrics for multi-positive candidate lists.

    labels can be soft probabilities; labels > 0 are treated as positives.
    accuracy@k is defined as whether the top-1 item is positive. hit_rate@k is
    whether any positive appears in top-k.
    """
    scores = _to_numpy(scores).astype(np.float64)
    labels = _to_numpy(labels).astype(np.float64)
    if mask is None:
        mask = np.ones_like(labels, dtype=bool)
    else:
        mask = _to_numpy(mask).astype(bool)

    out = {f"accuracy@{k}": [], f"hit_rate@{k}": [], f"precision@{k}": [],
           f"recall@{k}": [], f"map@{k}": [], f"ndcg@{k}": []}

    for s, y, m in zip(scores, labels, mask):
        valid_idx = np.where(m)[0]
        if len(valid_idx) == 0:
            continue
        pos = y[valid_idx] > 0
        num_pos = int(pos.sum())
        if num_pos == 0:
            continue
        order_valid = valid_idx[np.argsort(-s[valid_idx])]
        kk = min(k, len(order_valid))
        top = order_valid[:kk]
        top_pos = y[top] > 0
        out[f"accuracy@{k}"].append(float(y[order_valid[0]] > 0))
        out[f"hit_rate@{k}"].append(float(top_pos.any()))
        out[f"precision@{k}"].append(float(top_pos.sum() / kk))
        out[f"recall@{k}"].append(float(top_pos.sum() / num_pos))

        # Average precision@k.
        hits = 0
        ap = 0.0
        for rank, idx in enumerate(top, start=1):
            if y[idx] > 0:
                hits += 1
                ap += hits / rank
        out[f"map@{k}"].append(float(ap / min(num_pos, kk)))

        # NDCG@k with binary relevance.
        rel = top_pos.astype(np.float64)
        discounts = 1.0 / np.log2(np.arange(2, kk + 2))
        dcg = float((rel * discounts).sum())
        ideal_len = min(num_pos, kk)
        idcg = float(discounts[:ideal_len].sum())
        out[f"ndcg@{k}"].append(dcg / idcg if idcg > 0 else 0.0)

    return {name: (float(np.mean(vals)) if vals else 0.0) for name, vals in out.items()}


class MetricAverager:
    def __init__(self):
        self.storage = {}
        self.counts = {}

    def update(self, metrics: Dict[str, float], n: int = 1):
        for k, v in metrics.items():
            self.storage[k] = self.storage.get(k, 0.0) + float(v) * n
            self.counts[k] = self.counts.get(k, 0) + n

    def compute(self) -> Dict[str, float]:
        return {k: self.storage[k] / max(self.counts[k], 1) for k in self.storage}
