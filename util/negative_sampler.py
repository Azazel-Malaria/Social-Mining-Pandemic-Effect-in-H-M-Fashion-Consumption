from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass
class PopularityNegativeSampler:
    article_indices: np.ndarray
    probs: np.ndarray
    seed: int = 42

    @classmethod
    def from_transactions(cls, transactions: pd.DataFrame, article_to_idx: dict, seed: int = 42,
                          power: float = 0.75):
        counts = transactions["article_id"].astype(str).map(article_to_idx).dropna().astype(int).value_counts()
        article_indices = counts.index.to_numpy(dtype=np.int64)
        weights = counts.to_numpy(dtype=np.float64) ** power
        probs = weights / max(weights.sum(), 1e-12)
        return cls(article_indices=article_indices, probs=probs, seed=seed)

    @classmethod
    def uniform(cls, n_items: int, seed: int = 42):
        article_indices = np.arange(n_items, dtype=np.int64)
        probs = np.ones(n_items, dtype=np.float64) / max(n_items, 1)
        return cls(article_indices=article_indices, probs=probs, seed=seed)

    def sample(self, n: int, forbidden: Iterable[int] = ()) -> list[int]:
        rng = np.random.default_rng(self.seed + np.random.randint(0, 10_000_000))
        forbidden = set(int(x) for x in forbidden)
        result = []
        tries = 0
        while len(result) < n and tries < n * 50 + 100:
            batch_size = max(n * 2, 32)
            draw = rng.choice(self.article_indices, size=batch_size, replace=True, p=self.probs)
            for x in draw:
                x = int(x)
                if x not in forbidden and x not in result:
                    result.append(x)
                    if len(result) >= n:
                        break
            tries += 1
        if len(result) < n:
            all_items = self.article_indices.tolist()
            rng.shuffle(all_items)
            for x in all_items:
                x = int(x)
                if x not in forbidden and x not in result:
                    result.append(x)
                    if len(result) >= n:
                        break
        return result
