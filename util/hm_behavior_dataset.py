from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from util.negative_sampler import PopularityNegativeSampler
from util.style_taxonomy import STYLES, style_columns


def parse_list_cell(x: Any) -> list:
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            y = ast.literal_eval(x)
            return list(y) if isinstance(y, (list, tuple)) else []
        except Exception:
            return [v for v in x.split() if v]
    return list(x) if hasattr(x, "__iter__") else []


def _article_key(x: Any) -> str:
    s = str(x)
    return s.zfill(10) if s.isdigit() else s


def _maybe_int(x: Any) -> int | None:
    if x is None:
        return None
    try:
        v = int(x)
    except Exception:
        return None
    return None if v < 0 else v


def _normalize_score_matrix(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype(np.float32, copy=True)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        good = np.isfinite(col)
        if not good.any():
            mat[:, j] = 0.0
            continue
        lo = np.nanmin(col[good])
        hi = np.nanmax(col[good])
        col = np.where(good, col, 0.0)
        if hi > lo and (lo < 0.0 or hi > 1.0):
            col = (col - lo) / max(hi - lo, 1e-8)
        mat[:, j] = np.clip(col, 0.0, 1.0)
    return mat.astype(np.float32)


class HMBehaviorDataset(Dataset):
    """Two-stage behavior dataset.

    It never loads images or tokenizes texts.  Each item is represented by frozen
    stage-1 CLIP/SigLIP/OpenCLIP image-text features and optional routed compact
    knowledge prompts selected offline per item.

    Important separation:
      - 7-day training negatives are unpurchased candidate items.
      - prompt InfoNCE negatives are unselected prompts under the same
        subject/dimension, prepared in collate from the prompt bank.
    """

    def __init__(
        self,
        data_root: str | Path = "./data",
        split: str = "train",
        item_prefix: str = "clip",
        use_knowledge: bool = False,
        include_temporal: bool = False,
        num_negatives: int | None = None,
        max_history_items: int | None = None,
        max_samples: int | None = None,
        sample_strategy: str = "stratified_month_group",
        seed: int = 42,
        prompt_negatives_per_positive: int = 2,
        prompt_negative_mode: str = "bottomk",
        prompt_infonce_on: str = "positives",
        prompt_infonce_max_items_per_batch: int | None = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.item_prefix = item_prefix
        self.use_knowledge = bool(use_knowledge)
        self.include_temporal = bool(include_temporal)
        self.num_negatives = _maybe_int(num_negatives)
        self.max_history_items = _maybe_int(max_history_items)
        self.prompt_negatives_per_positive = max(0, int(prompt_negatives_per_positive))
        self.prompt_negative_mode = str(prompt_negative_mode or "bottomk")
        self.prompt_infonce_on = str(prompt_infonce_on or "positives")
        self.prompt_infonce_max_items_per_batch = _maybe_int(prompt_infonce_max_items_per_batch)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

        hm_proc = self.data_root / "hm" / "processed"
        packet_path = hm_proc / f"{item_prefix}_item_feature_packet.parquet"
        base_path = hm_proc / f"{item_prefix}_item_base_features.npy"
        if not packet_path.exists() or not base_path.exists():
            raise FileNotFoundError(
                f"Missing {item_prefix} item packet/base features. Run stage1_prepare_clip_knowledge.sh first."
            )
        self.items = pd.read_parquet(packet_path)
        self.items["article_id"] = self.items["article_id"].astype(str)
        self.article_ids = self.items["article_id"].tolist()
        self.article_to_idx = {a: i for i, a in enumerate(self.article_ids)}
        self.base_features = np.load(base_path).astype(np.float32)
        self.base_dim = int(self.base_features.shape[1])
        self.all_indices = np.arange(len(self.article_ids), dtype=np.int64)

        windows = pd.read_parquet(hm_proc / "hm_7day_windows.parquet")
        windows["split"] = windows["split"].astype(str)
        windows["anchor_date"] = pd.to_datetime(windows["anchor_date"])
        windows = windows[windows["split"] == split].reset_index(drop=True)
        if windows.empty:
            raise RuntimeError(f"No {split} samples found in hm_7day_windows.parquet")
        self.windows = self._sample_windows(windows, max_samples, sample_strategy)
        self.max_observed_history_len = int(max((len(parse_list_cell(x)) for x in self.windows["history_article_ids"]), default=1))

        tx_path = hm_proc / "hm_transactions.parquet"
        if tx_path.exists():
            tx = pd.read_parquet(tx_path)
            tx["article_id"] = tx["article_id"].astype(str)
            self.neg_sampler = PopularityNegativeSampler.from_transactions(tx, self.article_to_idx, seed=seed)
        else:
            self.neg_sampler = PopularityNegativeSampler.uniform(len(self.article_ids), seed=seed)

        style_path = hm_proc / "hm_style_weak_labels.parquet"
        self.style_probs = np.ones((len(self.article_ids), len(STYLES)), dtype=np.float32) / len(STYLES)
        if style_path.exists():
            styles = pd.read_parquet(style_path)
            styles["article_id"] = styles["article_id"].astype(str)
            cols = style_columns()
            for _, row in styles.iterrows():
                idx = self.article_to_idx.get(str(row["article_id"]))
                if idx is not None:
                    self.style_probs[idx] = row[cols].astype(float).to_numpy(dtype=np.float32)

        self.knowledge_dim = 0
        self.knowledge_tokens = None
        self.knowledge_mask = None
        self.prompt_features = None
        self.item_prompt_pos_rows = None
        self.item_prompt_mask = None
        self.item_prompt_neg_pools: list[list[np.ndarray]] | None = None
        self.knowledge_factors = None
        self.knowledge_factor_columns: list[str] = []
        self.knowledge_factor_dim = 0
        if self.use_knowledge:
            self._load_routed_knowledge(include_temporal)
            self._load_prompt_infonce_bank(include_temporal)
            self._load_knowledge_factors()

        print(f"[HMBehaviorDataset] split={split} rows={len(self.windows):,} num_negatives={self.num_negatives if self.num_negatives is not None else 'ALL'} max_history_items={self.max_history_items if self.max_history_items is not None else 'ALL'} max_samples={max_samples if max_samples is not None else 'ALL'} knowledge_factor_dim={self.knowledge_factor_dim}")

    def _sample_windows(self, windows: pd.DataFrame, max_samples: int | None, strategy: str) -> pd.DataFrame:
        max_samples = _maybe_int(max_samples)
        if max_samples is None or len(windows) <= max_samples:
            return windows.reset_index(drop=True)
        rng = np.random.default_rng(self.seed)
        w = windows.copy()
        if strategy == "random":
            idx = rng.choice(len(w), size=max_samples, replace=False)
            return w.iloc[np.sort(idx)].reset_index(drop=True)
        w["_month"] = w["anchor_date"].dt.to_period("M").astype(str)
        if strategy == "stratified_month_group":
            item_group = dict(zip(self.items["article_id"], self.items.get("product_group_name", pd.Series(["Unknown"] * len(self.items))).astype(str)))
            def first_group(cell):
                ids = parse_list_cell(cell)
                return item_group.get(_article_key(ids[0]), "Unknown") if ids else "Unknown"
            w["_group"] = w["target_article_ids"].map(first_group)
            keys = ["_month", "_group"]
        else:
            keys = ["_month"]
        groups = list(w.groupby(keys, sort=True))
        quota = max(1, int(np.ceil(max_samples / max(len(groups), 1))))
        parts = []
        for _, g in groups:
            take = min(len(g), quota)
            parts.append(g.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
        out = pd.concat(parts, ignore_index=True)
        if len(out) > max_samples:
            out = out.sample(n=max_samples, random_state=self.seed)
        return out.drop(columns=[c for c in out.columns if c.startswith("_")], errors="ignore").reset_index(drop=True)


    def _load_knowledge_factors(self) -> None:
        """Load stable per-item knowledge factors for optional Transformer injection.

        These factors are low-dimensional, interpretable variables used to modulate
        the candidate-history Transformer layers when --transformer_injection is
        enabled.  They are deliberately separate from raw routed prompt tokens:
        raw prompts still serve the item adapter / prompt InfoNCE path, while the
        factors keep social-analysis variables stable and cheap to inject.
        """
        cross_dir = self.data_root / "cross_domain"
        score_path = cross_dir / "item_knowledge_scores.parquet"
        mats = []
        cols: list[str] = []
        if score_path.exists():
            scores = pd.read_parquet(score_path)
            if "article_id" in scores.columns:
                scores["article_id"] = scores["article_id"].astype(str).map(_article_key)
                scores = scores.drop_duplicates("article_id", keep="last").set_index("article_id")
                score_cols = [c for c in scores.columns if str(c).endswith("_score")]
                if score_cols:
                    mat = np.zeros((len(self.article_ids), len(score_cols)), dtype=np.float32)
                    for j, c in enumerate(score_cols):
                        ser = pd.to_numeric(scores[c], errors="coerce")
                        vals = ser.reindex(self.article_ids).to_numpy(dtype=np.float32, na_value=np.nan)
                        mat[:, j] = vals
                    mats.append(_normalize_score_matrix(mat))
                    cols.extend(score_cols)
        # Always append weak H&M style factors.  This guarantees a usable factor
        # vector even if routed score columns are sparse, and it makes the
        # injected variables align with later social analysis dimensions.
        mats.append(self.style_probs.astype(np.float32))
        cols.extend([f"style_{s}" for s in STYLES])
        self.knowledge_factors = np.concatenate(mats, axis=1).astype(np.float32)
        self.knowledge_factor_columns = cols
        self.knowledge_factor_dim = int(self.knowledge_factors.shape[1])

    def _load_routed_knowledge(self, include_temporal: bool) -> None:
        cross_dir = self.data_root / "cross_domain"
        if include_temporal:
            tok_path = cross_dir / f"{self.item_prefix}_routed_knowledge_with_temporal_tokens.npy"
            mask_path = cross_dir / f"{self.item_prefix}_routed_knowledge_with_temporal_mask.npy"
        else:
            tok_path = cross_dir / "item_routed_knowledge_tokens.npy"
            mask_path = cross_dir / "item_routed_knowledge_mask.npy"
        if not tok_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Missing routed item knowledge tokens. Run scripts/route_item_knowledge.sh first. Missing {tok_path}")
        self.knowledge_tokens = np.load(tok_path).astype(np.float32)
        self.knowledge_mask = np.load(mask_path).astype(np.bool_)
        self.knowledge_dim = int(self.knowledge_tokens.shape[-1])

    def _find_prompt_feature_files(self, include_temporal: bool) -> tuple[Path, Path]:
        cross_dir = self.data_root / "cross_domain"
        suffix = "with_temporal" if include_temporal else "static"
        candidates = sorted(cross_dir.glob(f"{self.item_prefix}_knowledge_prompt_*_{suffix}_features.npy"))
        if not candidates:
            raise FileNotFoundError(
                f"No prompt feature bank found for item_prefix={self.item_prefix}, suffix={suffix}. "
                "Run scripts/route_item_knowledge.sh; it encodes prompt features for prompt-level InfoNCE."
            )
        feat_path = candidates[0]
        idx_path = Path(str(feat_path).replace("_features.npy", "_index.parquet"))
        if not idx_path.exists():
            raise FileNotFoundError(f"Missing prompt feature index: {idx_path}")
        return feat_path, idx_path

    def _load_prompt_infonce_bank(self, include_temporal: bool) -> None:
        cross_dir = self.data_root / "cross_domain"
        feat_path, idx_path = self._find_prompt_feature_files(include_temporal)
        self.prompt_features = np.load(feat_path).astype(np.float32)
        prompt_idx = pd.read_parquet(idx_path)
        prompt_idx["prompt_id"] = prompt_idx["prompt_id"].astype(str)
        pid_to_row = {p: int(r) for p, r in zip(prompt_idx["prompt_id"], prompt_idx["row"])}

        static_path = cross_dir / "knowledge_prompts_static.parquet"
        frames = [pd.read_parquet(static_path)] if static_path.exists() else []
        if include_temporal:
            temporal_path = cross_dir / "knowledge_prompts_temporal.parquet"
            if temporal_path.exists():
                frames.append(pd.read_parquet(temporal_path))
        if not frames:
            raise FileNotFoundError("Missing knowledge_prompts_static.parquet for prompt-level InfoNCE.")
        prompts = pd.concat(frames, ignore_index=True)
        prompts["prompt_id"] = prompts["prompt_id"].astype(str)
        prompts["subject"] = prompts.get("subject", "").astype(str)
        prompts["dimension"] = prompts.get("dimension", "").astype(str)
        prompts["prompt_row"] = prompts["prompt_id"].map(pid_to_row)
        prompts = prompts[prompts["prompt_row"].notna()].copy()
        prompts["prompt_row"] = prompts["prompt_row"].astype(int)
        pools_by_subject_dim = {
            (str(s), str(d)): g["prompt_row"].to_numpy(dtype=np.int64)
            for (s, d), g in prompts.groupby(["subject", "dimension"])
        }
        pools_by_dim = {str(d): g["prompt_row"].to_numpy(dtype=np.int64) for d, g in prompts.groupby("dimension")}

        meta_path = cross_dir / (f"{self.item_prefix}_routed_knowledge_with_temporal_meta.parquet" if include_temporal else "item_routed_knowledge_meta.parquet")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing routed knowledge meta: {meta_path}")
        meta = pd.read_parquet(meta_path)
        meta["article_id"] = meta["article_id"].astype(str)
        max_p = int(meta.groupby("article_id").size().max()) if len(meta) else 1
        n_items = len(self.article_ids)
        pos_rows = np.full((n_items, max_p), -1, dtype=np.int64)
        mask = np.zeros((n_items, max_p), dtype=np.bool_)
        neg_pools: list[list[np.ndarray]] = [[np.array([], dtype=np.int64) for _ in range(max_p)] for _ in range(n_items)]
        counters = np.zeros(n_items, dtype=np.int64)

        for _, r in meta.iterrows():
            aid = str(r.get("article_id", ""))
            item_idx = self.article_to_idx.get(aid)
            if item_idx is None:
                continue
            slot = int(counters[item_idx])
            if slot >= max_p:
                continue
            # prompt_ids can be a single id or a soft-merged list separated by ||; use the first selected id as positive.
            pids = [x for x in str(r.get("prompt_ids", "")).split("||") if x]
            if not pids:
                continue
            prow = pid_to_row.get(str(pids[0]))
            if prow is None:
                continue
            dim = str(r.get("dimension", ""))
            subj = str(r.get("subject", ""))
            neg = np.array([], dtype=np.int64)
            if self.prompt_negative_mode == "bottomk" and "negative_prompt_ids" in r.index:
                neg_ids = [x for x in str(r.get("negative_prompt_ids", "")).split("||") if x and x.lower() != "nan"]
                neg = np.array([pid_to_row[x] for x in neg_ids if x in pid_to_row and int(pid_to_row[x]) != int(prow)], dtype=np.int64)
            if len(neg) == 0:
                # Backward-compatible fallback for older routed meta files.
                # This should only happen if route_item_knowledge was not rerun
                # after the bottom-k patch.  New routed metadata stores explicit
                # least-similar prompt ids in negative_prompt_ids.
                pool = pools_by_subject_dim.get((subj, dim))
                if pool is None or len(pool) <= 1:
                    pool = pools_by_dim.get(dim, np.array([], dtype=np.int64))
                neg = np.array([x for x in pool.tolist() if int(x) != int(prow)], dtype=np.int64)
            pos_rows[item_idx, slot] = int(prow)
            mask[item_idx, slot] = True
            neg_pools[item_idx][slot] = neg
            counters[item_idx] += 1
        self.item_prompt_pos_rows = pos_rows
        self.item_prompt_mask = mask
        self.item_prompt_neg_pools = neg_pools

    def __len__(self) -> int:
        return len(self.windows)

    def _indices_from_ids(self, ids: Iterable[Any]) -> list[int]:
        out = []
        for x in ids:
            idx = self.article_to_idx.get(_article_key(x))
            if idx is not None:
                out.append(idx)
        return out

    def __getitem__(self, idx: int) -> dict:
        row = self.windows.iloc[idx]
        hist = self._indices_from_ids(parse_list_cell(row["history_article_ids"]))
        if self.max_history_items is not None:
            hist = hist[-self.max_history_items:]
        pos = sorted(set(self._indices_from_ids(parse_list_cell(row["target_article_ids"]))))
        if len(pos) == 0:
            pos = [int(self.rng.integers(0, len(self.article_ids)))]
        forbidden = set(hist) | set(pos)
        if self.num_negatives is None:
            # Full negative set: all item IDs except history and positives.
            neg = [int(x) for x in self.all_indices if int(x) not in forbidden]
        else:
            neg = self.neg_sampler.sample(self.num_negatives, forbidden=forbidden)
        candidates = pos + neg
        labels = np.zeros(len(candidates), dtype=np.float32)
        labels[:len(pos)] = 1.0 / max(len(pos), 1)
        return {
            "customer_id": row.get("customer_id", ""),
            "anchor_date": str(row.get("anchor_date", "")),
            "history_indices": hist,
            "candidate_indices": candidates,
            "labels": labels,
        }

    def sample_prompt_neg_rows(self, item_indices: np.ndarray, prompt_slots: np.ndarray) -> np.ndarray:
        out = np.full((*item_indices.shape, int(self.prompt_negatives_per_positive)), -1, dtype=np.int64)
        if self.item_prompt_neg_pools is None or self.prompt_negatives_per_positive <= 0:
            return out
        rng = np.random.default_rng(self.seed + np.random.randint(0, 10_000_000))
        it = np.nditer(item_indices, flags=["multi_index"])
        for x in it:
            item = int(x)
            slot = int(prompt_slots[it.multi_index])
            pool = self.item_prompt_neg_pools[item][slot] if 0 <= slot < len(self.item_prompt_neg_pools[item]) else np.array([], dtype=np.int64)
            if len(pool) == 0:
                continue
            if self.prompt_negative_mode == "bottomk":
                # pool is already ordered least-similar first by route_item_knowledge.
                draw = pool[: self.prompt_negatives_per_positive]
            else:
                replace = len(pool) < self.prompt_negatives_per_positive
                draw = rng.choice(pool, size=self.prompt_negatives_per_positive, replace=replace)
            out[it.multi_index + (slice(0, len(draw)),)] = draw.astype(np.int64)
        return out


def make_behavior_collate_fn(dataset: HMBehaviorDataset):
    def collate(batch: list[dict]) -> dict:
        bsz = len(batch)
        max_h = max(max(len(x["history_indices"]), 1) for x in batch)
        max_c = max(len(x["candidate_indices"]) for x in batch)
        labels = torch.zeros(bsz, max_c, dtype=torch.float32)
        candidate_mask = torch.zeros(bsz, max_c, dtype=torch.bool)
        history_mask = torch.zeros(bsz, max_h, dtype=torch.bool)
        candidate_indices = torch.zeros(bsz, max_c, dtype=torch.long)
        history_indices = torch.zeros(bsz, max_h, dtype=torch.long)
        candidate_style = torch.zeros(bsz, max_c, len(STYLES), dtype=torch.float32)

        for i, sample in enumerate(batch):
            h = sample["history_indices"] or []
            c = sample["candidate_indices"]
            if h:
                history_indices[i, :len(h)] = torch.tensor(h, dtype=torch.long)
            candidate_indices[i, :len(c)] = torch.tensor(c, dtype=torch.long)
            history_mask[i, :len(h)] = True
            candidate_mask[i, :len(c)] = True
            labels[i, :len(c)] = torch.from_numpy(sample["labels"])
            candidate_style[i, :len(c)] = torch.from_numpy(dataset.style_probs[c])

        base = torch.from_numpy(dataset.base_features)
        out = {
            "history_indices": history_indices,
            "candidate_indices": candidate_indices,
            "history_mask": history_mask,
            "candidate_mask": candidate_mask,
            "labels": labels,
            "candidate_style": candidate_style,
            "history_base_features": base[history_indices],
            "candidate_base_features": base[candidate_indices],
        }
        if dataset.use_knowledge and dataset.knowledge_factors is not None:
            kf = torch.from_numpy(dataset.knowledge_factors)
            out["history_knowledge_factors"] = kf[history_indices]
            out["candidate_knowledge_factors"] = kf[candidate_indices]
        if dataset.use_knowledge:
            kt = torch.from_numpy(dataset.knowledge_tokens)
            km = torch.from_numpy(dataset.knowledge_mask)
            out["history_knowledge"] = kt[history_indices]
            out["candidate_knowledge"] = kt[candidate_indices]
            out["history_knowledge_mask"] = km[history_indices]
            out["candidate_knowledge_mask"] = km[candidate_indices]

            # Prompt-level InfoNCE tensors.  Build them only for the items that
            # will actually contribute to prompt InfoNCE.  The previous version
            # prepared [B, all_candidates, prompts, negatives, D], which can OOM
            # even when prompt_infonce_on=positives because negative candidates
            # were later masked out but still materialized.
            pf = torch.from_numpy(dataset.prompt_features) if dataset.prompt_features is not None else None
            if pf is not None and dataset.item_prompt_pos_rows is not None:
                labels_np = labels.numpy()
                cand_mask_np = candidate_mask.numpy()
                if dataset.prompt_infonce_on == "candidates":
                    selected_positions = [np.flatnonzero(cand_mask_np[i]) for i in range(bsz)]
                else:
                    selected_positions = [np.flatnonzero((labels_np[i] > 0) & cand_mask_np[i]) for i in range(bsz)]

                # Optional emergency cap for very dense multi-positive rows or candidates mode.
                if dataset.prompt_infonce_max_items_per_batch is not None:
                    flat = [(i, int(p)) for i, arr in enumerate(selected_positions) for p in arr.tolist()]
                    if len(flat) > dataset.prompt_infonce_max_items_per_batch:
                        rng = np.random.default_rng(dataset.seed + len(batch))
                        keep_idx = set(rng.choice(len(flat), size=dataset.prompt_infonce_max_items_per_batch, replace=False).tolist())
                        kept = [[] for _ in range(bsz)]
                        for j, (i, ppos) in enumerate(flat):
                            if j in keep_idx:
                                kept[i].append(ppos)
                        selected_positions = [np.asarray(x, dtype=np.int64) for x in kept]

                max_pi = max((len(x) for x in selected_positions), default=0)
                if max_pi > 0:
                    prompt_item_positions = np.zeros((bsz, max_pi), dtype=np.int64)
                    prompt_item_valid = np.zeros((bsz, max_pi), dtype=np.bool_)
                    selected_item_idx = np.zeros((bsz, max_pi), dtype=np.int64)
                    for i, arr in enumerate(selected_positions):
                        if len(arr) == 0:
                            continue
                        prompt_item_positions[i, :len(arr)] = arr
                        prompt_item_valid[i, :len(arr)] = True
                        selected_item_idx[i, :len(arr)] = candidate_indices.numpy()[i, arr]

                    pos_rows_np = dataset.item_prompt_pos_rows[selected_item_idx]
                    prompt_mask_np = dataset.item_prompt_mask[selected_item_idx]
                    prompt_mask_np = prompt_mask_np & prompt_item_valid[..., None]
                    p_slots = np.broadcast_to(np.arange(pos_rows_np.shape[-1], dtype=np.int64), pos_rows_np.shape)
                    neg_rows_np = dataset.sample_prompt_neg_rows(selected_item_idx[..., None].repeat(pos_rows_np.shape[-1], axis=-1), p_slots)

                    safe_pos = np.where(pos_rows_np >= 0, pos_rows_np, 0)
                    safe_neg = np.where(neg_rows_np >= 0, neg_rows_np, 0)
                    out["prompt_item_positions"] = torch.from_numpy(prompt_item_positions).long()
                    out["prompt_item_valid"] = torch.from_numpy(prompt_item_valid).bool()
                    out["prompt_pos_features"] = pf[torch.from_numpy(safe_pos).long()]
                    out["prompt_neg_features"] = pf[torch.from_numpy(safe_neg).long()]
                    neg_valid = torch.from_numpy(neg_rows_np >= 0).bool()
                    out["prompt_neg_mask"] = neg_valid
                    out["prompt_mask"] = torch.from_numpy(prompt_mask_np).bool() & neg_valid.any(dim=-1)
        return out
    return collate
