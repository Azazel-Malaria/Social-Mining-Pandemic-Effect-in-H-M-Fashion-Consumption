#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User-level channel migration overview for H&M transactions.

This script is intentionally standalone and does not depend on social_analysis.py.
It builds a user-level summary of sales-channel migration and produces:
  1) a quadrant-style density/scatter plot;
  2) a binned-scatter plot with standard errors;
  3) summary CSV files for later reporting.

Core definitions for each user u:
  pre_share_u   = target-channel share in matched 2019 window
  covid_share_u = target-channel share in COVID 2020 window
  shift_u       = covid_share_u - pre_share_u

Recommended H&M default window:
  matched 2019: 2019-03-01 to 2019-09-22
  COVID 2020:   2020-03-01 to 2020-09-22
because the public H&M transaction file ends on 2020-09-22.
"""

from __future__ import annotations

# Keep BLAS-related libraries from oversubscribing threads on large servers.
# These must be set before importing numpy/scipy-like libraries.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "32")

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATE_COL = "t_dat"
USER_COL = "customer_id"
CHANNEL_COL = "sales_channel_id"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone user-level channel migration quadrant and binned-scatter plots."
    )
    parser.add_argument(
        "--transactions_path",
        type=str,
        required=True,
        help="Path to H&M transactions_train.csv/parquet, containing t_dat, customer_id, sales_channel_id.",
    )
    parser.add_argument(
        "--customers_path",
        type=str,
        default=None,
        help="Optional customers.csv/parquet. If provided, selected customer fields are merged into the summary.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for output figures and summary tables.",
    )
    parser.add_argument("--target_channel", type=int, default=2, help="Target sales_channel_id. Default: 2.")
    parser.add_argument("--pre_start", type=str, default="2019-03-01", help="Matched baseline start date, inclusive.")
    parser.add_argument("--pre_end", type=str, default="2019-09-22", help="Matched baseline end date, inclusive.")
    parser.add_argument("--covid_start", type=str, default="2020-03-01", help="COVID window start date, inclusive.")
    parser.add_argument("--covid_end", type=str, default="2020-09-22", help="COVID window end date, inclusive.")
    parser.add_argument(
        "--min_txn_per_window",
        type=int,
        default=1,
        help="Require at least this many transactions in both pre and COVID windows. Default: 1.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=2_000_000,
        help="CSV chunk size. Ignored for parquet input. Default: 2,000,000.",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=10,
        help="Number of baseline-share bins for binned scatter. Default: 10.",
    )
    parser.add_argument(
        "--binning",
        choices=["quantile", "equal_width"],
        default="quantile",
        help="How to bin baseline channel share. Default: quantile.",
    )
    parser.add_argument(
        "--quadrant_split",
        choices=["median", "mean", "0.5"],
        default="median",
        help="Vertical split for quadrant figure. Default: median baseline share.",
    )
    parser.add_argument(
        "--plot_kind",
        choices=["auto", "hexbin", "scatter"],
        default="auto",
        help="Quadrant plot type. auto uses hexbin when user count exceeds --hexbin_min_users.",
    )
    parser.add_argument(
        "--hexbin_min_users",
        type=int,
        default=80_000,
        help="Use hexbin under --plot_kind auto if filtered user count exceeds this threshold. Default: 80,000.",
    )
    parser.add_argument(
        "--scatter_sample",
        type=int,
        default=250_000,
        help="Maximum points to draw in scatter mode. If exceeded, sample deterministically. Default: 250,000.",
    )
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for scatter subsampling.")
    parser.add_argument(
        "--merge_customer_fields",
        type=str,
        default="age,club_member_status,fashion_news_frequency",
        help="Comma-separated customer columns to merge when --customers_path is provided.",
    )
    parser.add_argument(
        "--save_parquet",
        type=int,
        default=1,
        choices=[0, 1],
        help="Also save parquet outputs when parquet dependencies are available. Default: 1.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI. Default: 180.",
    )
    return parser.parse_args()


def _as_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).normalize()


def _read_transactions_aggregated(
    path: Path,
    target_channel: int,
    pre_start: pd.Timestamp,
    pre_end: pd.Timestamp,
    covid_start: pd.Timestamp,
    covid_end: pd.Timestamp,
    chunksize: int,
) -> pd.DataFrame:
    """Return per-user transaction counts in pre/COVID windows.

    Output columns:
      customer_id, pre_total, pre_target, covid_total, covid_target
    """
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path, columns=[DATE_COL, USER_COL, CHANNEL_COL])
        return _aggregate_one_frame(df, target_channel, pre_start, pre_end, covid_start, covid_end)

    if suffix not in {".csv", ".gz", ".zip"}:
        # pandas can still handle many csv-like paths; try CSV reader as fallback.
        pass

    partials: List[pd.DataFrame] = []
    usecols = [DATE_COL, USER_COL, CHANNEL_COL]
    dtypes = {USER_COL: "string", CHANNEL_COL: "int16"}

    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtypes,
        parse_dates=[DATE_COL],
        chunksize=chunksize,
    ):
        part = _aggregate_one_frame(chunk, target_channel, pre_start, pre_end, covid_start, covid_end)
        if len(part):
            partials.append(part)

    if not partials:
        return pd.DataFrame(columns=[USER_COL, "pre_total", "pre_target", "covid_total", "covid_target"])

    out = pd.concat(partials, ignore_index=True)
    out = out.groupby(USER_COL, as_index=False)[["pre_total", "pre_target", "covid_total", "covid_target"]].sum()
    return out


def _aggregate_one_frame(
    df: pd.DataFrame,
    target_channel: int,
    pre_start: pd.Timestamp,
    pre_end: pd.Timestamp,
    covid_start: pd.Timestamp,
    covid_end: pd.Timestamp,
) -> pd.DataFrame:
    if DATE_COL not in df.columns or USER_COL not in df.columns or CHANNEL_COL not in df.columns:
        missing = {DATE_COL, USER_COL, CHANNEL_COL}.difference(df.columns)
        raise ValueError(f"transactions file missing required columns: {sorted(missing)}")

    df = df[[DATE_COL, USER_COL, CHANNEL_COL]].copy()
    if not np.issubdtype(df[DATE_COL].dtype, np.datetime64):
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    pre_mask = (df[DATE_COL] >= pre_start) & (df[DATE_COL] <= pre_end)
    covid_mask = (df[DATE_COL] >= covid_start) & (df[DATE_COL] <= covid_end)
    keep = pre_mask | covid_mask
    if not keep.any():
        return pd.DataFrame(columns=[USER_COL, "pre_total", "pre_target", "covid_total", "covid_target"])

    sub = df.loc[keep, [USER_COL, CHANNEL_COL]].copy()
    sub["pre_total"] = pre_mask.loc[keep].astype("int8").to_numpy()
    sub["covid_total"] = covid_mask.loc[keep].astype("int8").to_numpy()
    is_target = (sub[CHANNEL_COL].astype(int) == int(target_channel)).astype("int8")
    sub["pre_target"] = sub["pre_total"].to_numpy() * is_target.to_numpy()
    sub["covid_target"] = sub["covid_total"].to_numpy() * is_target.to_numpy()
    return sub.groupby(USER_COL, as_index=False)[["pre_total", "pre_target", "covid_total", "covid_target"]].sum()


def _merge_customers(summary: pd.DataFrame, customers_path: Optional[str], fields: str) -> pd.DataFrame:
    if not customers_path:
        return summary
    path = Path(customers_path)
    want = [x.strip() for x in fields.split(",") if x.strip()]
    cols = [USER_COL] + want

    if path.suffix.lower() in {".parquet", ".pq"}:
        customers = pd.read_parquet(path)
    else:
        # Only read requested columns if possible. If some fields are missing, retry with all columns.
        try:
            customers = pd.read_csv(path, usecols=lambda c: c in cols, dtype={USER_COL: "string"})
        except Exception:
            customers = pd.read_csv(path, dtype={USER_COL: "string"})

    if USER_COL not in customers.columns:
        raise ValueError(f"customers file must contain {USER_COL}")
    keep_cols = [USER_COL] + [c for c in want if c in customers.columns]
    customers = customers[keep_cols].drop_duplicates(USER_COL)
    return summary.merge(customers, on=USER_COL, how="left")


def _build_user_summary(
    counts: pd.DataFrame,
    min_txn_per_window: int,
) -> pd.DataFrame:
    required = ["pre_total", "pre_target", "covid_total", "covid_target"]
    for c in required:
        if c not in counts.columns:
            counts[c] = 0
    summary = counts.copy()
    summary[required] = summary[required].fillna(0).astype(int)
    summary = summary[
        (summary["pre_total"] >= min_txn_per_window)
        & (summary["covid_total"] >= min_txn_per_window)
    ].copy()
    if summary.empty:
        return summary
    summary["pre_channel_target_share"] = summary["pre_target"] / summary["pre_total"].clip(lower=1)
    summary["covid_channel_target_share"] = summary["covid_target"] / summary["covid_total"].clip(lower=1)
    summary["channel_migration_shift"] = (
        summary["covid_channel_target_share"] - summary["pre_channel_target_share"]
    )
    summary["total_txn_both_windows"] = summary["pre_total"] + summary["covid_total"]
    return summary


def _quadrant_split_value(x: pd.Series, mode: str) -> float:
    if mode == "median":
        return float(x.median())
    if mode == "mean":
        return float(x.mean())
    if mode == "0.5":
        return 0.5
    raise ValueError(mode)


def _quadrant_table(df: pd.DataFrame, x_split: float) -> pd.DataFrame:
    x = df["pre_channel_target_share"].to_numpy()
    y = df["channel_migration_shift"].to_numpy()
    labels = np.where(x < x_split, "low baseline", "high baseline")
    labels2 = np.where(y >= 0, "positive migration", "negative migration")
    q = pd.DataFrame({"baseline_region": labels, "migration_region": labels2})
    out = q.value_counts().reset_index(name="n_users")
    out["share_users"] = out["n_users"] / max(1, len(df))
    return out.sort_values(["baseline_region", "migration_region"])


def _plot_quadrant(df: pd.DataFrame, out_dir: Path, x_split: float, args: argparse.Namespace) -> Path:
    x = df["pre_channel_target_share"].to_numpy(dtype=float)
    y = df["channel_migration_shift"].to_numpy(dtype=float)

    use_hexbin = args.plot_kind == "hexbin" or (args.plot_kind == "auto" and len(df) >= args.hexbin_min_users)
    fig, ax = plt.subplots(figsize=(9.8, 7.0))

    if use_hexbin:
        hb = ax.hexbin(x, y, gridsize=55, mincnt=1, bins="log")
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("log10(number of users)")
        out_path = out_dir / "user_channel_quadrant_hexbin.png"
    else:
        if len(df) > args.scatter_sample:
            draw = df.sample(args.scatter_sample, random_state=args.random_seed)
            x_draw = draw["pre_channel_target_share"].to_numpy(dtype=float)
            y_draw = draw["channel_migration_shift"].to_numpy(dtype=float)
        else:
            x_draw, y_draw = x, y
        ax.scatter(x_draw, y_draw, s=6, alpha=0.18, linewidths=0)
        out_path = out_dir / "user_channel_quadrant_scatter.png"

    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.axvline(x_split, linestyle="--", linewidth=1.2)
    ax.set_xlabel(f"Matched-2019 channel {args.target_channel} baseline share")
    ax.set_ylabel(f"COVID shift in channel {args.target_channel} share")
    ax.set_title("User-level channel migration overview")

    # Add quadrant shares in a compact text box.
    qtab = _quadrant_table(df, x_split)
    def _get_share(base: str, mig: str) -> float:
        m = (qtab["baseline_region"] == base) & (qtab["migration_region"] == mig)
        if m.any():
            return float(qtab.loc[m, "share_users"].iloc[0])
        return 0.0

    text = (
        f"low baseline / +shift: {_get_share('low baseline','positive migration'):.1%}\n"
        f"high baseline / +shift: {_get_share('high baseline','positive migration'):.1%}\n"
        f"low baseline / -shift: {_get_share('low baseline','negative migration'):.1%}\n"
        f"high baseline / -shift: {_get_share('high baseline','negative migration'):.1%}\n"
        f"N users = {len(df):,}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.7"},
        fontsize=9,
    )
    ax.set_xlim(-0.02, 1.02)
    y_abs = max(abs(np.nanquantile(y, 0.01)), abs(np.nanquantile(y, 0.99)), 0.05)
    ax.set_ylim(-min(1.05, y_abs * 1.25), min(1.05, y_abs * 1.25))
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    return out_path


def _make_bins(df: pd.DataFrame, num_bins: int, mode: str) -> pd.DataFrame:
    x = df["pre_channel_target_share"]
    work = df.copy()
    if mode == "quantile":
        # qcut may drop duplicate edges when shares are discrete; that's okay.
        work["baseline_bin"] = pd.qcut(x, q=num_bins, duplicates="drop")
    elif mode == "equal_width":
        work["baseline_bin"] = pd.cut(x, bins=np.linspace(0, 1, num_bins + 1), include_lowest=True)
    else:
        raise ValueError(mode)

    rows = []
    for b, g in work.dropna(subset=["baseline_bin"]).groupby("baseline_bin", observed=True):
        n = len(g)
        y = g["channel_migration_shift"].to_numpy(dtype=float)
        sd = float(np.nanstd(y, ddof=1)) if n > 1 else 0.0
        se = sd / math.sqrt(max(1, n))
        rows.append(
            {
                "baseline_bin": str(b),
                "n_users": n,
                "x_mean": float(g["pre_channel_target_share"].mean()),
                "x_median": float(g["pre_channel_target_share"].median()),
                "shift_mean": float(np.nanmean(y)),
                "shift_median": float(np.nanmedian(y)),
                "shift_sd": sd,
                "shift_se": se,
                "shift_ci95_low": float(np.nanmean(y) - 1.96 * se),
                "shift_ci95_high": float(np.nanmean(y) + 1.96 * se),
            }
        )
    return pd.DataFrame(rows).sort_values("x_mean")


def _plot_binned_scatter(bins: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> Path:
    out_path = out_dir / "user_channel_binned_scatter.png"
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    if bins.empty:
        ax.text(0.5, 0.5, "No valid bins", ha="center", va="center", transform=ax.transAxes)
    else:
        x = bins["x_mean"].to_numpy(dtype=float)
        y = bins["shift_mean"].to_numpy(dtype=float)
        yerr = 1.96 * bins["shift_se"].to_numpy(dtype=float)
        sizes = 30 + 220 * (bins["n_users"].to_numpy(dtype=float) / bins["n_users"].max())
        ax.errorbar(x, y, yerr=yerr, fmt="none", linewidth=1.1, capsize=3)
        ax.scatter(x, y, s=sizes, alpha=0.82)
        ax.plot(x, y, linewidth=1.0, alpha=0.65)
    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel(f"Matched-2019 channel {args.target_channel} baseline share, binned")
    ax.set_ylabel(f"Mean COVID shift in channel {args.target_channel} share")
    ax.set_title("Binned user channel migration")
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    return out_path


def _save_outputs(
    summary: pd.DataFrame,
    bins: pd.DataFrame,
    qtab: pd.DataFrame,
    out_dir: Path,
    save_parquet: bool,
) -> None:
    summary.to_csv(out_dir / "user_channel_migration_summary.csv", index=False)
    bins.to_csv(out_dir / "user_channel_binned_summary.csv", index=False)
    qtab.to_csv(out_dir / "user_channel_quadrant_summary.csv", index=False)
    if save_parquet:
        for name, df in [
            ("user_channel_migration_summary.parquet", summary),
            ("user_channel_binned_summary.parquet", bins),
            ("user_channel_quadrant_summary.parquet", qtab),
        ]:
            try:
                df.to_parquet(out_dir / name, index=False)
            except Exception:
                # Parquet support is optional. CSV is always saved.
                pass


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_start = _as_date(args.pre_start)
    pre_end = _as_date(args.pre_end)
    covid_start = _as_date(args.covid_start)
    covid_end = _as_date(args.covid_end)
    if pre_end < pre_start:
        raise ValueError("--pre_end must be >= --pre_start")
    if covid_end < covid_start:
        raise ValueError("--covid_end must be >= --covid_start")

    counts = _read_transactions_aggregated(
        Path(args.transactions_path),
        target_channel=args.target_channel,
        pre_start=pre_start,
        pre_end=pre_end,
        covid_start=covid_start,
        covid_end=covid_end,
        chunksize=args.chunksize,
    )
    summary = _build_user_summary(counts, min_txn_per_window=args.min_txn_per_window)
    summary = _merge_customers(summary, args.customers_path, args.merge_customer_fields)

    if summary.empty:
        raise RuntimeError(
            "No users remain after filtering. Try lowering --min_txn_per_window or checking date windows."
        )

    x_split = _quadrant_split_value(summary["pre_channel_target_share"], args.quadrant_split)
    qtab = _quadrant_table(summary, x_split)
    bins = _make_bins(summary, args.num_bins, args.binning)

    quadrant_fig = _plot_quadrant(summary, out_dir, x_split, args)
    binned_fig = _plot_binned_scatter(bins, out_dir, args)
    _save_outputs(summary, bins, qtab, out_dir, bool(args.save_parquet))

    manifest = {
        "transactions_path": str(Path(args.transactions_path).resolve()),
        "customers_path": str(Path(args.customers_path).resolve()) if args.customers_path else None,
        "output_dir": str(out_dir.resolve()),
        "target_channel": args.target_channel,
        "pre_window": [str(pre_start.date()), str(pre_end.date())],
        "covid_window": [str(covid_start.date()), str(covid_end.date())],
        "min_txn_per_window": args.min_txn_per_window,
        "n_users_after_filter": int(len(summary)),
        "overall_pre_share_mean": float(summary["pre_channel_target_share"].mean()),
        "overall_covid_share_mean": float(summary["covid_channel_target_share"].mean()),
        "overall_shift_mean": float(summary["channel_migration_shift"].mean()),
        "overall_shift_median": float(summary["channel_migration_shift"].median()),
        "quadrant_split": args.quadrant_split,
        "quadrant_split_value": float(x_split),
        "quadrant_figure": str(quadrant_fig.name),
        "binned_figure": str(binned_fig.name),
        "summary_csv": "user_channel_migration_summary.csv",
        "binned_summary_csv": "user_channel_binned_summary.csv",
        "quadrant_summary_csv": "user_channel_quadrant_summary.csv",
    }
    with open(out_dir / "user_channel_migration_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
