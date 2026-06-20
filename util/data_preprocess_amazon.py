from __future__ import annotations

import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from tqdm import tqdm

from util.io_utils import ensure_dir

CLOTHING_KEYWORDS = {
    "clothing", "fashion", "apparel", "shirt", "t-shirt", "tee", "dress", "skirt", "jeans",
    "trousers", "pants", "jacket", "coat", "sweater", "cardigan", "hoodie", "shorts", "leggings",
    "blouse", "top", "suit", "blazer", "underwear", "bra", "sock", "socks", "shoes", "sneaker", "boot",
}

# Hugging Face `hf download --local-dir ./data/amazon/raw` usually produces:
#   data/amazon/raw/raw/meta_categories/meta_Amazon_Fashion.jsonl
#   data/amazon/raw/raw/review_categories/Amazon_Fashion.jsonl
# plus local cache/bookkeeping files under:
#   data/amazon/raw/.cache/huggingface/download/*.metadata
# The cache files are NOT dataset payloads and must never be parsed as JSONL records.
EXCLUDED_PARTS = {".cache", "__pycache__", ".git"}
VALID_SUFFIXES = {".jsonl", ".gz"}


def is_payload_jsonl(path: Path) -> bool:
    """Return True only for real JSONL/JSONL.GZ dataset files, not HF cache metadata."""
    parts = set(path.parts)
    if parts & EXCLUDED_PARTS:
        return False
    name = path.name
    if name.endswith(".metadata") or name.endswith(".lock") or name.endswith(".incomplete"):
        return False
    return name.endswith(".jsonl") or name.endswith(".jsonl.gz")


def open_jsonl(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Iterate over JSONL records, skipping malformed lines and non-dict JSON values."""
    with open_jsonl(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                # Defensive guard: cache metadata or corrupted lines may be string/number/list.
                continue
            yield obj


def exact_candidate_paths(raw_dir: Path, category: str, meta: bool) -> list[Path]:
    """Prefer the actual Amazon-Reviews-2023 repo layout before any recursive fallback."""
    sub = "meta_categories" if meta else "review_categories"
    filename_stems = [f"meta_{category}" if meta else category]

    # Two common local layouts:
    # 1) hf download --local-dir ./data/amazon/raw  -> ./data/amazon/raw/raw/<sub>/...
    # 2) manually moved repo content               -> ./data/amazon/raw/<sub>/...
    roots = [raw_dir / "raw" / sub, raw_dir / sub]
    paths: list[Path] = []
    for root in roots:
        for stem in filename_stems:
            for suffix in (".jsonl", ".jsonl.gz"):
                p = root / f"{stem}{suffix}"
                if p.exists() and is_payload_jsonl(p):
                    paths.append(p)
    return paths


def find_files(raw_dir: Path, category: str, meta: bool) -> list[Path]:
    files = exact_candidate_paths(raw_dir, category, meta)
    if files:
        return sorted(dict.fromkeys(files))

    # Fallback for custom/manual layouts. Still strictly exclude .cache and non-payload files.
    prefix = f"meta_{category}" if meta else category
    candidates = []
    for p in raw_dir.rglob(f"{prefix}.jsonl*"):
        if is_payload_jsonl(p):
            candidates.append(p)
    return sorted(dict.fromkeys(candidates))


def clean_join(obj) -> str:
    if obj is None:
        return ""
    # pandas may use NaN floats for missing values; avoid stringifying them as "nan".
    if isinstance(obj, float) and np.isnan(obj):
        return ""
    if isinstance(obj, list):
        return " ".join(clean_join(x) for x in obj)
    if isinstance(obj, dict):
        return " ".join(f"{k}: {clean_join(v)}" for k, v in obj.items())
    return str(obj)


def is_clothing_related(row: dict) -> bool:
    if not isinstance(row, dict):
        return False
    text = " ".join([
        clean_join(row.get("main_category", "")),
        clean_join(row.get("title", "")),
        clean_join(row.get("features", "")),
        clean_join(row.get("description", "")),
        clean_join(row.get("categories", "")),
        clean_join(row.get("details", "")),
    ]).lower()
    return any(k in text for k in CLOTHING_KEYWORDS)


def load_meta(raw_dir: Path, categories: list[str], max_items: int | None = None) -> pd.DataFrame:
    rows = []
    for cat in categories:
        files = find_files(raw_dir, cat, meta=True)
        if not files:
            print(f"Warning: no meta file found for category={cat} under {raw_dir}")
        for path in files:
            print(f"Reading Amazon meta payload: {path}")
            for obj in tqdm(iter_jsonl(path), desc=f"Meta {path.name}"):
                if not is_clothing_related(obj):
                    continue
                obj = dict(obj)
                obj["source_category"] = cat
                obj["parent_asin"] = str(obj.get("parent_asin") or obj.get("asin") or "")
                if not obj["parent_asin"]:
                    continue
                obj["clean_text"] = " ".join([
                    clean_join(obj.get("title", "")), clean_join(obj.get("features", "")),
                    clean_join(obj.get("description", "")), clean_join(obj.get("categories", "")),
                    clean_join(obj.get("details", "")),
                ]).strip()
                obj["is_clothing_related"] = True
                rows.append(obj)
                if max_items is not None and len(rows) >= max_items:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def review_weight(row: dict) -> float:
    if not isinstance(row, dict):
        return 0.0
    verified = 1.0 if bool(row.get("verified_purchase", False)) else 0.5
    try:
        rating = float(row.get("rating", 0.0) or 0.0)
    except Exception:
        rating = 0.0
    helpful = row.get("helpful_vote", 0) or 0
    try:
        helpful = float(helpful)
    except Exception:
        helpful = 0.0
    return verified * max(rating - 3.0, 0.0) * np.log1p(helpful + 1.0)


def load_reviews(raw_dir: Path, categories: list[str], parent_asins: set[str],
                 max_reviews: int | None = None) -> pd.DataFrame:
    rows = []
    for cat in categories:
        files = find_files(raw_dir, cat, meta=False)
        if not files:
            print(f"Warning: no review file found for category={cat} under {raw_dir}")
        for path in files:
            print(f"Reading Amazon review payload: {path}")
            for obj in tqdm(iter_jsonl(path), desc=f"Reviews {path.name}"):
                parent = str(obj.get("parent_asin") or "")
                if parent not in parent_asins:
                    continue
                text = clean_join(obj.get("text", ""))
                if len(text.strip()) == 0:
                    continue
                obj = dict(obj)
                obj["parent_asin"] = parent
                obj["source_category"] = cat
                obj["clean_review_text"] = text.strip()
                obj["review_weight"] = float(review_weight(obj))
                rows.append(obj)
                if max_reviews is not None and len(rows) >= max_reviews:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)



def parse_review_datetime_series(df: pd.DataFrame) -> pd.Series:
    if "timestamp" in df.columns:
        raw = df["timestamp"]
        num = pd.to_numeric(raw, errors="coerce")
        # Amazon Reviews 2023 commonly stores timestamp in milliseconds.
        dt_ms = pd.to_datetime(num, unit="ms", errors="coerce", utc=False)
        dt_s = pd.to_datetime(num, unit="s", errors="coerce", utc=False)
        # Prefer millisecond conversion when value is large.
        dt = dt_ms.where(num.abs() > 10**11, dt_s)
        # Fill string dates if any.
        dt_str = pd.to_datetime(raw, errors="coerce", utc=False)
        return dt.fillna(dt_str)
    for col in ["date", "review_date", "time"]:
        if col in df.columns:
            return pd.to_datetime(df[col], errors="coerce", utc=False)
    return pd.Series(pd.NaT, index=df.index)


def hm_time_window(data_root: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    hm_tx = data_root / "hm" / "processed" / "hm_transactions.parquet"
    if not hm_tx.exists():
        raise FileNotFoundError(f"Cannot infer H&M time window because {hm_tx} does not exist. Run prepare_hm.sh first or pass --time_filter_from_hm 0 with --time_start/--time_end.")
    tx = pd.read_parquet(hm_tx)
    date_col = "t_dat" if "t_dat" in tx.columns else "date"
    d = pd.to_datetime(tx[date_col], errors="coerce")
    return d.min(), d.max()


def filter_reviews_by_time_and_sample(reviews: pd.DataFrame, data_root: Path, time_filter_from_hm: bool = True,
                                      time_start: str | None = None, time_end: str | None = None,
                                      review_sample_ratio: float = 1.0, sample_seed: int = 42) -> tuple[pd.DataFrame, dict]:
    report: dict = {"reviews_before_time_filter": int(len(reviews))}
    if reviews.empty:
        report.update({"reviews_after_time_filter": 0, "reviews_after_random_sample": 0})
        return reviews, report
    dt = parse_review_datetime_series(reviews)
    reviews = reviews.copy()
    reviews["review_datetime"] = dt
    if time_filter_from_hm or time_start or time_end:
        if time_start:
            start = pd.to_datetime(time_start)
        else:
            start, _ = hm_time_window(data_root)
        if time_end:
            end = pd.to_datetime(time_end)
        else:
            _, end = hm_time_window(data_root)
        mask = reviews["review_datetime"].notna() & (reviews["review_datetime"] >= start) & (reviews["review_datetime"] <= end)
        before = len(reviews)
        reviews = reviews[mask].copy()
        report.update({
            "time_start": str(start.date()) if pd.notna(start) else None,
            "time_end": str(end.date()) if pd.notna(end) else None,
            "dropped_by_time_filter": int(before - len(reviews)),
            "reviews_after_time_filter": int(len(reviews)),
            "time_drop_ratio": float((before - len(reviews)) / max(before, 1)),
        })
    else:
        report.update({"dropped_by_time_filter": 0, "reviews_after_time_filter": int(len(reviews)), "time_drop_ratio": 0.0})
    ratio = float(review_sample_ratio)
    if ratio < 1.0:
        if ratio <= 0:
            raise ValueError("review_sample_ratio must be in (0, 1] when sampling is enabled.")
        before = len(reviews)
        reviews = reviews.sample(frac=ratio, random_state=sample_seed).reset_index(drop=True)
        report.update({
            "review_sample_ratio": ratio,
            "dropped_by_random_sample": int(before - len(reviews)),
            "reviews_after_random_sample": int(len(reviews)),
            "sample_drop_ratio_after_time": float((before - len(reviews)) / max(before, 1)),
        })
    else:
        report.update({"review_sample_ratio": ratio, "dropped_by_random_sample": 0, "reviews_after_random_sample": int(len(reviews)), "sample_drop_ratio_after_time": 0.0})
    return reviews, report


_PRICE_MISSING = {"", "-", "—", "–", "--", "---", "n/a", "na", "none", "null", "nan", "unavailable", "currently unavailable"}

def parse_price(value) -> float:
    """Parse Amazon price into a float.

    Rules:
    - plain numeric values are returned as float;
    - placeholders such as '—' or '-' become NaN;
    - ranges such as '20-40', '$20 - $40', '20–40', '20 to 40' become the mean, e.g. 30.0;
    - currency symbols and commas are ignored.
    """
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            x = float(value)
            return x if np.isfinite(x) else float("nan")
        except Exception:
            return float("nan")
    if isinstance(value, (list, dict)):
        value = clean_join(value)
    text = str(value).strip()
    if text.lower() in _PRICE_MISSING:
        return float("nan")
    # Normalize dash variants and common range separators. Do not treat '-' as negative sign for prices.
    text_norm = (
        text.replace("−", "-")
            .replace("–", "-")
            .replace("—", "-")
            .replace("~", "-")
            .replace(" to ", "-")
            .replace(" TO ", "-")
    )
    # Extract unsigned numbers; this intentionally parses '20-40' as [20, 40], not [20, -40].
    nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", text_norm)]
    if not nums:
        return float("nan")
    if len(nums) >= 2:
        lo, hi = min(nums[0], nums[1]), max(nums[0], nums[1])
        return float((lo + hi) / 2.0)
    return float(nums[0])


def to_numeric_safe(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def stabilize_amazon_item_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Force known mixed-type Amazon metadata columns into parquet-safe dtypes."""
    df = df.copy()
    if "price" in df.columns:
        df["price"] = df["price"].map(parse_price).astype("float64")
    for col in ["average_rating", "rating_number", "store_rating", "bought_together_count"]:
        if col in df.columns:
            df[col] = to_numeric_safe(df[col]).astype("float64")
    # Keep IDs/text/categorical fields as strings after nested normalization.
    for col in ["parent_asin", "asin", "title", "main_category", "source_category", "clean_text"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


def stabilize_amazon_review_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Force known mixed-type Amazon review columns into parquet-safe dtypes."""
    df = df.copy()
    for col in ["rating", "helpful_vote", "review_weight"]:
        if col in df.columns:
            df[col] = to_numeric_safe(df[col]).astype("float64")
    for col in ["parent_asin", "source_category", "clean_review_text", "title", "text"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df

def normalize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make nested/list columns parquet-safe by serializing them into readable strings."""
    preferred_text_cols = {"features", "description", "categories", "details", "images", "videos"}

    def normalize_cell(x):
        if isinstance(x, (list, dict)):
            return clean_join(x)
        if isinstance(x, float) and np.isnan(x):
            return None
        return x

    for col in df.columns:
        if col in preferred_text_cols or df[col].dtype == "object":
            df[col] = df[col].map(normalize_cell)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--amazon_raw_dir", default=None)
    parser.add_argument("--amazon_processed_dir", default=None)
    parser.add_argument("--amazon_categories", nargs="+", default=["Amazon_Fashion", "Clothing_Shoes_and_Jewelry"])
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--max_reviews", type=int, default=None)
    parser.add_argument("--time_filter_from_hm", type=lambda x: str(x).lower() in {"1","true","yes","y"}, default=True)
    parser.add_argument("--time_start", default=None)
    parser.add_argument("--time_end", default=None)
    parser.add_argument("--review_sample_ratio", type=float, default=1.0)
    parser.add_argument("--sample_seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    raw_dir = Path(args.amazon_raw_dir) if args.amazon_raw_dir else data_root / "amazon" / "raw"
    out_dir = Path(args.amazon_processed_dir) if args.amazon_processed_dir else data_root / "amazon" / "processed"
    ensure_dir(out_dir)

    items = load_meta(raw_dir, args.amazon_categories, args.max_items)
    if items.empty:
        raise RuntimeError(
            f"No Amazon clothing items found under {raw_dir}. "
            "Expected files like raw/meta_categories/meta_Amazon_Fashion.jsonl or "
            "raw/raw/meta_categories/meta_Amazon_Fashion.jsonl."
        )
    items = stabilize_amazon_item_schema(normalize_object_columns(items))
    items.to_parquet(out_dir / "amazon_items_filtered.parquet", index=False)

    parent_asins = set(items["parent_asin"].astype(str))
    reviews = load_reviews(raw_dir, args.amazon_categories, parent_asins, args.max_reviews)
    if reviews.empty:
        print("Warning: no Amazon reviews matched filtered items; downstream retrieval corpus will be empty.")
        report = {"reviews_before_time_filter": 0, "reviews_after_time_filter": 0, "reviews_after_random_sample": 0}
    else:
        reviews, report = filter_reviews_by_time_and_sample(
            reviews, data_root=data_root, time_filter_from_hm=args.time_filter_from_hm,
            time_start=args.time_start, time_end=args.time_end,
            review_sample_ratio=args.review_sample_ratio, sample_seed=args.sample_seed,
        )
    reviews = stabilize_amazon_review_schema(normalize_object_columns(reviews))
    # To keep downstream retrieval small, retain only items with at least one retained review when possible.
    items_before_review_prune = len(items)
    if not reviews.empty:
        kept_asins = set(reviews["parent_asin"].astype(str))
        items = items[items["parent_asin"].astype(str).isin(kept_asins)].copy()
    report["items_before_review_prune"] = int(items_before_review_prune)
    report["items_after_review_prune"] = int(len(items))
    report["items_dropped_no_retained_review"] = int(items_before_review_prune - len(items))
    # Rewrite filtered item file after review pruning.
    items = stabilize_amazon_item_schema(normalize_object_columns(items))
    reviews = stabilize_amazon_review_schema(normalize_object_columns(reviews))
    items.to_parquet(out_dir / "amazon_items_filtered.parquet", index=False)
    reviews.to_parquet(out_dir / "amazon_reviews_filtered.parquet", index=False)
    (out_dir / "amazon_filter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("========== Amazon preprocessing/filtering report ==========")
    for k, v in report.items():
        print(f"{k}: {v}")
    print("=========================================================")
    print(f"Amazon preprocessing finished. Items={len(items)}, Reviews={len(reviews)}, outputs={out_dir}")


if __name__ == "__main__":
    main()
