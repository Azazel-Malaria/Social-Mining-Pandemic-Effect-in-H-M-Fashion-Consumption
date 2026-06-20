from __future__ import annotations

import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from util.io_utils import article_to_str, ensure_dir, find_hm_image_path, parse_bool
from util.style_taxonomy import infer_style_probs, style_columns

TEXT_COLS = [
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


def _join_text(row: pd.Series) -> str:
    vals = []
    for col in TEXT_COLS:
        v = row.get(col, "")
        if pd.notna(v) and str(v).strip():
            vals.append(str(v).strip())
    return "; ".join(vals)


def build_item_table(raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    articles_path = raw_dir / "articles.csv"
    if not articles_path.exists():
        raise FileNotFoundError(f"Missing {articles_path}. Run scripts/download_data.sh first.")
    articles = pd.read_csv(articles_path, dtype={"article_id": str})
    articles["article_id"] = articles["article_id"].map(article_to_str)
    for col in TEXT_COLS:
        if col not in articles.columns:
            articles[col] = ""
    articles["hm_clean_text"] = articles.apply(_join_text, axis=1)
    image_root = raw_dir / "images"
    articles["image_path"] = articles["article_id"].map(lambda x: find_hm_image_path(image_root, x))
    articles["has_image"] = articles["image_path"].map(lambda p: Path(p).exists())
    keep_cols = ["article_id", "product_code"] + TEXT_COLS + ["hm_clean_text", "image_path", "has_image"]
    keep_cols = [c for c in keep_cols if c in articles.columns]
    item_table = articles[keep_cols].copy()
    item_table.to_parquet(out_dir / "hm_item_text.parquet", index=False)
    return item_table


def build_style_weak_labels(item_table: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    probs = np.vstack([infer_style_probs(t) for t in item_table["hm_clean_text"].fillna("")])
    df = pd.DataFrame(probs, columns=style_columns())
    df.insert(0, "article_id", item_table["article_id"].values)
    df.to_parquet(out_dir / "hm_style_weak_labels.parquet", index=False)
    return df


def build_transactions(raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    tx_path = raw_dir / "transactions_train.csv"
    if not tx_path.exists():
        raise FileNotFoundError(f"Missing {tx_path}. Run scripts/download_data.sh first.")
    tx = pd.read_csv(tx_path, dtype={"article_id": str, "customer_id": str})
    tx["article_id"] = tx["article_id"].map(article_to_str)
    tx["t_dat"] = pd.to_datetime(tx["t_dat"])
    start_date = tx["t_dat"].min()
    tx["day_idx"] = (tx["t_dat"] - start_date).dt.days.astype(int)
    tx.sort_values(["customer_id", "t_dat"], inplace=True)
    tx.to_parquet(out_dir / "hm_transactions.parquet", index=False)
    customers_path = raw_dir / "customers.csv"
    if customers_path.exists():
        customers = pd.read_csv(customers_path, dtype={"customer_id": str})
        customers.to_parquet(out_dir / "hm_customers.parquet", index=False)
    return tx


def _split_by_anchor(anchor_dates: pd.Series) -> pd.Series:
    q_train = anchor_dates.quantile(0.80)
    q_val = anchor_dates.quantile(0.90)
    split = np.where(anchor_dates <= q_train, "train", np.where(anchor_dates <= q_val, "val", "test"))
    return pd.Series(split, index=anchor_dates.index)


def build_7day_windows(transactions: pd.DataFrame, out_dir: Path,
                       max_history_items: int = 80,
                       min_history_items: int = 1,
                       anchor_stride_days: int = 7,
                       max_users: int | None = None,
                       max_windows_per_user: int | None = None,
                       seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    grouped = transactions.groupby("customer_id", sort=False)
    if max_users is not None:
        selected_users = set(transactions["customer_id"].drop_duplicates().head(max_users))
    else:
        selected_users = None

    for customer_id, g in tqdm(grouped, desc="Build rolling 7-day windows"):
        if selected_users is not None and customer_id not in selected_users:
            continue
        g = g.sort_values("t_dat")
        dates = g["t_dat"].drop_duplicates().sort_values().to_list()
        if len(dates) <= 1:
            continue
        anchor_dates = []
        last_anchor = None
        for d in dates[1:]:
            if last_anchor is None or (d - last_anchor).days >= anchor_stride_days:
                anchor_dates.append(d)
                last_anchor = d
        if max_windows_per_user is not None and len(anchor_dates) > max_windows_per_user:
            idx = np.sort(rng.choice(len(anchor_dates), size=max_windows_per_user, replace=False))
            anchor_dates = [anchor_dates[i] for i in idx]

        for anchor in anchor_dates:
            hist = g[g["t_dat"] < anchor]
            fut = g[(g["t_dat"] >= anchor) & (g["t_dat"] < anchor + pd.Timedelta(days=7))]
            if len(hist) < min_history_items or len(fut) == 0:
                continue
            hist = hist.tail(max_history_items)
            rows.append({
                "customer_id": customer_id,
                "anchor_date": anchor,
                "history_article_ids": hist["article_id"].tolist(),
                "history_timestamps": hist["t_dat"].astype(str).tolist(),
                "target_article_ids": sorted(set(fut["article_id"].tolist())),
                "num_history": int(len(hist)),
                "num_target": int(fut["article_id"].nunique()),
            })
    windows = pd.DataFrame(rows)
    if windows.empty:
        raise RuntimeError("No training windows were generated. Lower min_history_items or check transactions data.")
    windows["anchor_date"] = pd.to_datetime(windows["anchor_date"])
    windows["split"] = _split_by_anchor(windows["anchor_date"])
    windows.to_parquet(out_dir / "hm_7day_windows.parquet", index=False)
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--hm_raw_dir", default=None)
    parser.add_argument("--hm_processed_dir", default=None)
    parser.add_argument("--max_history_items", type=int, default=80)
    parser.add_argument("--min_history_items", type=int, default=1)
    parser.add_argument("--anchor_stride_days", type=int, default=7)
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--max_windows_per_user", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_existing", default="0")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    raw_dir = Path(args.hm_raw_dir) if args.hm_raw_dir else data_root / "hm" / "raw"
    out_dir = Path(args.hm_processed_dir) if args.hm_processed_dir else data_root / "hm" / "processed"
    ensure_dir(out_dir)
    skip = parse_bool(args.skip_existing)

    item_path = out_dir / "hm_item_text.parquet"
    tx_path = out_dir / "hm_transactions.parquet"
    win_path = out_dir / "hm_7day_windows.parquet"

    if skip and item_path.exists():
        item_table = pd.read_parquet(item_path)
    else:
        item_table = build_item_table(raw_dir, out_dir)
    style_path = out_dir / "hm_style_weak_labels.parquet"
    if not (skip and style_path.exists()):
        build_style_weak_labels(item_table, out_dir)

    if skip and tx_path.exists():
        tx = pd.read_parquet(tx_path)
        tx["t_dat"] = pd.to_datetime(tx["t_dat"])
    else:
        tx = build_transactions(raw_dir, out_dir)

    if not (skip and win_path.exists()):
        build_7day_windows(
            tx,
            out_dir,
            max_history_items=args.max_history_items,
            min_history_items=args.min_history_items,
            anchor_stride_days=args.anchor_stride_days,
            max_users=args.max_users,
            max_windows_per_user=args.max_windows_per_user,
            seed=args.seed,
        )
    print(f"H&M preprocessing finished. Outputs are in {out_dir}")


if __name__ == "__main__":
    main()
