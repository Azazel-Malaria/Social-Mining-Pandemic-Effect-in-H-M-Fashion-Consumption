from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from util.io_utils import ensure_dir, parse_bool
from util.social_style_utils import add_social_style_scores, SOCIAL_STYLES


def zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return x * 0.0
    return (x - mu) / sd


def savefig(path: Path, dpi: int = 180) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def existing_score_cols(scores: pd.DataFrame) -> list[str]:
    return [c for c in scores.columns if c.endswith("_score") and c != "score"]


def load_scores_with_social_styles(root: Path) -> pd.DataFrame:
    score_path = root / "item_knowledge_scores.parquet"
    meta_path = root / "item_metadata.parquet"
    if score_path.exists():
        scores = pd.read_parquet(score_path)
        scores["article_id"] = scores["article_id"].astype(str)
    else:
        scores = pd.DataFrame()
    style_cols = [f"{s}_score" for s in SOCIAL_STYLES]
    missing_styles = any(c not in scores.columns for c in style_cols)
    if missing_styles and meta_path.exists():
        meta = pd.read_parquet(meta_path)
        meta["article_id"] = meta["article_id"].astype(str)
        style = add_social_style_scores(meta)[["article_id"] + style_cols]
        if scores.empty:
            scores = style
        else:
            scores = scores.merge(style, on="article_id", how="left", suffixes=("", "_derived"))
            for c in style_cols:
                alt = f"{c}_derived"
                if alt in scores.columns:
                    if c in scores.columns:
                        scores[c] = scores[c].where(scores[c].notna(), scores[alt])
                    else:
                        scores[c] = scores[alt]
                    scores = scores.drop(columns=[alt])
    return scores


def rebuild_timeseries_if_needed(root: Path) -> pd.DataFrame | None:
    ts_path = root / "knowledge_sales_timeseries.parquet"
    sales_path = root / "item_monthly_sales.parquet"
    if ts_path.exists():
        ts = pd.read_parquet(ts_path)
        # If formal/comfort etc are absent but metadata is available, rebuild with derived social styles.
        dims = set(ts.get("knowledge_dim", pd.Series(dtype=str)).astype(str).unique())
        if {"formal", "comfort", "homewear", "value"}.issubset(dims):
            return ts
    if not sales_path.exists():
        return pd.read_parquet(ts_path) if ts_path.exists() else None
    sales = pd.read_parquet(sales_path)
    sales["article_id"] = sales["article_id"].astype(str)
    scores = load_scores_with_social_styles(root)
    if scores.empty:
        return pd.read_parquet(ts_path) if ts_path.exists() else None
    score_cols = existing_score_cols(scores)
    if not score_cols:
        return pd.read_parquet(ts_path) if ts_path.exists() else None
    merged = sales[["article_id", "month", "sales_count"]].merge(scores[["article_id"] + score_cols], on="article_id", how="left")
    for c in score_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0).astype(float)
    total_sales = sales.groupby("month", as_index=False)["sales_count"].sum().rename(columns={"sales_count": "total_sales"})
    rows = []
    for c in score_cols:
        dim = c[:-6]
        tmp = merged.assign(weighted=merged["sales_count"] * merged[c]).groupby("month", as_index=False).agg(
            weighted_sales=("weighted", "sum"), raw_sales=("sales_count", "sum")
        )
        tmp["knowledge_dim"] = dim
        rows.append(tmp)
    ts = pd.concat(rows, ignore_index=True).merge(total_sales, on="month", how="left")
    ts["share"] = ts["weighted_sales"] / ts["total_sales"].replace(0, np.nan)
    ts.to_parquet(root / "knowledge_sales_timeseries.parquet", index=False)
    return ts


def make_tsne_or_pca(root: Path, fig_dir: Path, max_items: int = 4000, method: str = "tsne", seed: int = 42) -> list[Path]:
    emb_path = root / "item_embeddings.parquet"
    meta_path = root / "item_metadata.parquet"
    if not emb_path.exists() or not meta_path.exists():
        print(f"[WARN] Missing item_embeddings/item_metadata under {root}; skip UMAP/t-SNE.")
        return []
    emb_df = pd.read_parquet(emb_path)
    meta = pd.read_parquet(meta_path)
    emb_df["article_id"] = emb_df["article_id"].astype(str)
    meta["article_id"] = meta["article_id"].astype(str)
    emb_cols = [c for c in emb_df.columns if c.startswith("embedding_")]
    if len(emb_cols) < 2:
        print("[WARN] item_embeddings has fewer than 2 embedding columns; skip embedding plot.")
        return []
    df = emb_df[["article_id"] + emb_cols].merge(meta, on="article_id", how="left")
    if len(df) > max_items:
        df = df.sample(n=max_items, random_state=seed).reset_index(drop=True)
    X = df[emb_cols].to_numpy(dtype=np.float32)
    # PCA denoise first; t-SNE is optional and may be slow for large samples.
    from sklearn.decomposition import PCA
    X0 = PCA(n_components=min(50, X.shape[1], max(2, len(df) - 1)), random_state=seed).fit_transform(X)
    used_method = method.lower()
    try:
        if used_method == "tsne":
            from sklearn.manifold import TSNE
            coords = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=min(30, max(5, (len(df)-1)//3)), random_state=seed).fit_transform(X0)
        else:
            coords = X0[:, :2]
            used_method = "pca"
    except Exception as e:
        print(f"[WARN] t-SNE failed ({e}); fallback to PCA.")
        coords = X0[:, :2]
        used_method = "pca"
    out = []
    plot_df = df.copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]
    plot_df.to_parquet(root / "analysis" / f"item_space_{used_method}.parquet", index=False)

    # Product-group categorical scatter: plot top groups and gray out others via integer codes.
    if "product_group_name" in plot_df.columns:
        top = plot_df["product_group_name"].fillna("Unknown").astype(str).value_counts().head(12).index.tolist()
        group = plot_df["product_group_name"].fillna("Unknown").astype(str).where(lambda s: s.isin(top), "Other")
        cats = {v: i for i, v in enumerate(sorted(group.unique()))}
        c = group.map(cats).to_numpy()
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(plot_df["x"], plot_df["y"], c=c, s=5, alpha=0.65)
        handles, _ = sc.legend_elements(num=min(len(cats), 13))
        labels = list(cats.keys())[:len(handles)]
        if handles:
            plt.legend(handles, labels, loc="best", fontsize=7, frameon=True)
        plt.title(f"Item space by product group ({used_method.upper()})")
        plt.xlabel(f"{used_method.upper()} 1")
        plt.ylabel(f"{used_method.upper()} 2")
        p = fig_dir / f"item_space_{used_method}_product_group.png"
        savefig(p)
        out.append(p)

    scores = load_scores_with_social_styles(root)
    if not scores.empty:
        scores["article_id"] = scores["article_id"].astype(str)
        plot_df2 = plot_df[["article_id", "x", "y"]].merge(scores, on="article_id", how="left")
        for dim in ["formal", "comfort", "homewear", "value"]:
            col = f"{dim}_score"
            if col not in plot_df2.columns:
                continue
            plt.figure(figsize=(7, 6))
            vals = pd.to_numeric(plot_df2[col], errors="coerce").fillna(0.0)
            plt.scatter(plot_df2["x"], plot_df2["y"], c=vals, s=5, alpha=0.75)
            plt.colorbar(label=col)
            plt.title(f"Item space colored by {dim} score ({used_method.upper()})")
            plt.xlabel(f"{used_method.upper()} 1")
            plt.ylabel(f"{used_method.upper()} 2")
            p = fig_dir / f"item_space_{used_method}_{dim}_score.png"
            savefig(p)
            out.append(p)
    return out


def make_covid_fused_lines(root: Path, fig_dir: Path, style_dims: list[str]) -> list[Path]:
    out = []
    ts = rebuild_timeseries_if_needed(root)
    if ts is None or ts.empty:
        print("[WARN] Missing knowledge_sales_timeseries; skip COVID/style line figure.")
        return out
    covid_path = root / "analysis" / "covid_timeseries_monthly.parquet"
    if not covid_path.exists():
        print("[WARN] Missing analysis/covid_timeseries_monthly.parquet; skip COVID fused line. Provide --covid_csv to run_social_analysis.sh.")
        return out
    covid = pd.read_parquet(covid_path)
    months = sorted(set(ts["month"].astype(str)) & set(covid["month"].astype(str)))
    if not months:
        print("[WARN] No overlapping months between style timeseries and COVID data.")
        return out
    wide = ts[ts["knowledge_dim"].isin(style_dims)].pivot_table(index="month", columns="knowledge_dim", values="share", aggfunc="mean").reset_index()
    df = pd.DataFrame({"month": months}).merge(wide, on="month", how="left").merge(covid, on="month", how="left")
    covid_col = None
    for c in ["log1p_new_cases_smoothed_per_million", "log1p_new_cases_smoothed", "new_cases_smoothed_per_million", "new_cases_smoothed"]:
        if c in df.columns:
            covid_col = c
            break
    if covid_col is None:
        print("[WARN] COVID columns not found; skip COVID fused line.")
        return out
    plt.figure(figsize=(10, 5))
    plt.plot(df["month"], zscore(df[covid_col]), linewidth=2.5, label="COVID intensity")
    for dim in style_dims:
        if dim in df.columns:
            plt.plot(df["month"], zscore(df[dim]), marker="o", markersize=3, linewidth=1.5, label=f"{dim} share")
    plt.axvline("2020-03", linestyle="--", linewidth=1.0)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("z-score")
    plt.title("COVID intensity and style preference shares")
    plt.legend(fontsize=8)
    p = fig_dir / "covid_style_fused_lines.png"
    savefig(p)
    out.append(p)
    return out


def make_prepost_bar(root: Path, fig_dir: Path, top_n: int = 20) -> list[Path]:
    path = root / "analysis" / "pre_post_share_diff.parquet"
    if not path.exists():
        print("[WARN] Missing pre_post_share_diff.parquet; skip pre/post bar.")
        return []
    df = pd.read_parquet(path)
    if df.empty or "post_minus_pre" not in df.columns:
        return []
    df = df.sort_values("post_minus_pre", ascending=True).tail(top_n)
    plt.figure(figsize=(8, max(4, 0.28 * len(df))))
    plt.barh(df["knowledge_dim"].astype(str), df["post_minus_pre"].astype(float))
    plt.axvline(0, linewidth=1.0)
    plt.xlabel("Post minus pre share")
    plt.title("Preference share change after event month")
    p = fig_dir / "pre_post_preference_gap.png"
    savefig(p)
    return [p]


def make_seasonality_heatmap(root: Path, fig_dir: Path, style_dims: list[str], top_n: int = 20) -> list[Path]:
    ts = rebuild_timeseries_if_needed(root)
    if ts is None or ts.empty:
        print("[WARN] Missing knowledge_sales_timeseries; skip seasonality heatmap.")
        return []
    ts = ts.copy()
    ts["calendar_month"] = pd.PeriodIndex(ts["month"].astype(str), freq="M").month
    # Prefer interpretable style dims, then add top varying dimensions.
    var = ts.groupby("knowledge_dim")["share"].std().sort_values(ascending=False)
    dims = [d for d in style_dims if d in set(ts["knowledge_dim"].astype(str))]
    for d in var.index.tolist():
        if d not in dims:
            dims.append(d)
        if len(dims) >= top_n:
            break
    sub = ts[ts["knowledge_dim"].isin(dims)]
    heat = sub.pivot_table(index="knowledge_dim", columns="calendar_month", values="share", aggfunc="mean").reindex(dims)
    if heat.empty:
        return []
    # Row z-score to emphasize seasonal shape.
    arr = heat.to_numpy(dtype=float)
    mu = np.nanmean(arr, axis=1, keepdims=True)
    sd = np.nanstd(arr, axis=1, keepdims=True)
    arr_z = (arr - mu) / np.where(sd == 0, 1, sd)
    plt.figure(figsize=(10, max(4, 0.32 * len(heat))))
    plt.imshow(arr_z, aspect="auto")
    plt.colorbar(label="row z-score of monthly share")
    plt.yticks(np.arange(len(heat.index)), heat.index.astype(str), fontsize=8)
    plt.xticks(np.arange(12), [str(i) for i in range(1, 13)])
    plt.xlabel("Calendar month")
    plt.title("Fourier/seasonality diagnostic: monthly preference pattern")
    p = fig_dir / "seasonality_monthly_heatmap.png"
    savefig(p)
    return [p]



def month_diff(month_value, event_month: str) -> int:
    """Return integer month offset month_value - event_month.

    pandas Period subtraction can return DateOffset objects on some versions,
    so compute the difference explicitly from year/month fields.
    Accepts YYYY-MM strings, datetime-like values, or pandas Period values.
    """
    if pd.isna(month_value):
        return 0
    m = pd.Period(str(month_value)[:7], freq="M")
    ev = pd.Period(str(event_month)[:7], freq="M")
    return (m.year - ev.year) * 12 + (m.month - ev.month)

def make_event_study_proxy(root: Path, fig_dir: Path, style_dims: list[str], event_month: str = "2020-03") -> list[Path]:
    ts = rebuild_timeseries_if_needed(root)
    if ts is None or ts.empty:
        return []
    sub = ts[ts["knowledge_dim"].isin(style_dims)].copy()
    if sub.empty:
        return []
    sub["rel_month"] = [month_diff(m, event_month) for m in sub["month"]]
    plt.figure(figsize=(9, 5))
    for dim, g in sub.groupby("knowledge_dim"):
        g = g.sort_values("rel_month")
        base = g.loc[g["rel_month"] == -1, "share"].mean()
        if not np.isfinite(base):
            base = g[g["rel_month"] < 0]["share"].mean()
        y = g["share"] - base
        plt.plot(g["rel_month"], y, marker="o", markersize=3, linewidth=1.5, label=str(dim))
    plt.axvline(0, linestyle="--", linewidth=1.0)
    plt.axhline(0, linewidth=1.0)
    plt.xlabel(f"Months relative to {event_month}")
    plt.ylabel("Share minus pre-event baseline")
    plt.title("Event-study style preference dynamics")
    plt.legend(fontsize=8)
    p = fig_dir / "event_study_style_dynamics.png"
    savefig(p)
    return [p]


def compute_amazon_hm_compare(root: Path, data_root: Path | None, max_reviews: int = 300000, seed: int = 42) -> tuple[Path | None, Path | None]:
    if data_root is None:
        return None, None
    amazon_reviews = data_root / "amazon" / "processed" / "amazon_reviews_filtered.parquet"
    amazon_items = data_root / "amazon" / "processed" / "amazon_items_filtered.parquet"
    if not amazon_reviews.exists():
        print(f"[WARN] Missing {amazon_reviews}; skip Amazon-H&M comparison.")
        return None, None
    rev = pd.read_parquet(amazon_reviews)
    if len(rev) > max_reviews:
        rev = rev.sample(n=max_reviews, random_state=seed).reset_index(drop=True)
    if "review_datetime" in rev.columns:
        dt = pd.to_datetime(rev["review_datetime"], errors="coerce")
    elif "timestamp" in rev.columns:
        num = pd.to_numeric(rev["timestamp"], errors="coerce")
        dt = pd.to_datetime(num, unit="ms", errors="coerce")
        if dt.isna().mean() > 0.8:
            dt = pd.to_datetime(num, unit="s", errors="coerce")
    else:
        dt = pd.Series(pd.NaT, index=rev.index)
    rev["month"] = dt.dt.to_period("M").astype(str)
    text_cols = [c for c in ["title", "text", "clean_review_text"] if c in rev.columns]
    rev_text = rev[text_cols].fillna("").astype(str).agg(" ".join, axis=1) if text_cols else pd.Series([""] * len(rev))
    if amazon_items.exists() and "parent_asin" in rev.columns:
        items = pd.read_parquet(amazon_items)
        item_cols = [c for c in ["parent_asin", "title", "clean_text", "main_category"] if c in items.columns]
        if "parent_asin" in item_cols:
            item_text = items[item_cols].drop_duplicates("parent_asin")
            item_text["parent_asin"] = item_text["parent_asin"].astype(str)
            rev["parent_asin"] = rev["parent_asin"].astype(str)
            rev = rev.merge(item_text, on="parent_asin", how="left", suffixes=("", "_item"))
            extra_cols = [c for c in rev.columns if c.endswith("_item") or c in ["clean_text", "main_category"]]
            if extra_cols:
                rev_text = rev_text + " " + rev[extra_cols].fillna("").astype(str).agg(" ".join, axis=1)
    from util.social_style_utils import infer_social_style_scores
    arr = np.vstack([infer_social_style_scores(x) for x in rev_text]) if len(rev) else np.zeros((0, len(SOCIAL_STYLES)), dtype=np.float32)
    for j, s in enumerate(SOCIAL_STYLES):
        rev[f"{s}_score"] = arr[:, j]
    if "rating" in rev.columns:
        rating = pd.to_numeric(rev["rating"], errors="coerce").fillna(3.0).clip(1, 5)
    else:
        rating = pd.Series(3.0, index=rev.index)
    helpful = pd.to_numeric(rev["helpful_vote"], errors="coerce").fillna(0.0) if "helpful_vote" in rev.columns else pd.Series(0.0, index=rev.index)
    rev["weight"] = rating * np.log1p(1.0 + helpful)
    rev = rev[rev["month"].notna() & (rev["month"] != "NaT")]
    rows = []
    for s in SOCIAL_STYLES:
        tmp = rev.assign(weighted=rev[f"{s}_score"] * rev["weight"]).groupby("month", as_index=False).agg(
            amazon_pref=("weighted", "sum"), amazon_weight=("weight", "sum")
        )
        tmp["knowledge_dim"] = s
        tmp["amazon_pref"] = tmp["amazon_pref"] / tmp["amazon_weight"].replace(0, np.nan)
        rows.append(tmp[["month", "knowledge_dim", "amazon_pref"]])
    amazon_ts = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    hm_ts = rebuild_timeseries_if_needed(root)
    if hm_ts is None or hm_ts.empty:
        return None, None
    hm = hm_ts[hm_ts["knowledge_dim"].isin(SOCIAL_STYLES)][["month", "knowledge_dim", "share"]].rename(columns={"share": "hm_pref"})
    comp = hm.merge(amazon_ts, on=["month", "knowledge_dim"], how="inner")
    comp["gap_hm_minus_amazon"] = comp["hm_pref"] - comp["amazon_pref"]
    out_dir = ensure_dir(root / "analysis")
    ts_path = out_dir / "amazon_hm_preference_timeseries.parquet"
    gap_path = out_dir / "amazon_hm_preference_gap.parquet"
    comp.to_parquet(ts_path, index=False)
    gap = comp.groupby("knowledge_dim", as_index=False).agg(
        hm_pref=("hm_pref", "mean"), amazon_pref=("amazon_pref", "mean"), gap_hm_minus_amazon=("gap_hm_minus_amazon", "mean")
    )
    gap.to_parquet(gap_path, index=False)
    return ts_path, gap_path


def make_amazon_hm_figures(root: Path, fig_dir: Path, data_root: Path | None, max_reviews: int, seed: int) -> list[Path]:
    compute_amazon_hm_compare(root, data_root, max_reviews=max_reviews, seed=seed)
    out = []
    gap_path = root / "analysis" / "amazon_hm_preference_gap.parquet"
    ts_path = root / "analysis" / "amazon_hm_preference_timeseries.parquet"
    if gap_path.exists():
        gap = pd.read_parquet(gap_path).sort_values("gap_hm_minus_amazon")
        if not gap.empty:
            plt.figure(figsize=(8, max(4, 0.35 * len(gap))))
            plt.barh(gap["knowledge_dim"].astype(str), gap["gap_hm_minus_amazon"].astype(float))
            plt.axvline(0, linewidth=1.0)
            plt.xlabel("H&M purchase preference minus Amazon expression preference")
            plt.title("Amazon-H&M preference gap")
            p = fig_dir / "amazon_hm_preference_gap.png"
            savefig(p)
            out.append(p)
    if ts_path.exists():
        comp = pd.read_parquet(ts_path)
        if not comp.empty:
            # Cosine similarity over time between preference vectors.
            rows = []
            for month, g in comp.groupby("month"):
                hv = g.set_index("knowledge_dim")["hm_pref"].reindex(SOCIAL_STYLES).fillna(0).to_numpy(float)
                av = g.set_index("knowledge_dim")["amazon_pref"].reindex(SOCIAL_STYLES).fillna(0).to_numpy(float)
                den = np.linalg.norm(hv) * np.linalg.norm(av)
                sim = float(hv @ av / den) if den > 0 else np.nan
                rows.append({"month": month, "cosine_similarity": sim})
            sim = pd.DataFrame(rows).sort_values("month")
            sim.to_parquet(root / "analysis" / "amazon_hm_similarity_timeseries.parquet", index=False)
            plt.figure(figsize=(9, 4))
            plt.plot(sim["month"], sim["cosine_similarity"], marker="o", linewidth=1.8)
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Cosine similarity")
            plt.title("Amazon expression vs H&M purchase preference similarity")
            p = fig_dir / "amazon_hm_similarity_timeseries.png"
            savefig(p)
            out.append(p)
    return out


def make_all_figures(social_output_root: Path, data_root: Path | None = None, event_month: str = "2020-03",
                     embedding_method: str = "tsne", fig_max_items: int = 4000,
                     style_dims: str = "formal,comfort,homewear,value,casual,office",
                     run_amazon_hm_compare: bool = False, max_amazon_reviews: int = 300000,
                     seed: int = 42) -> list[Path]:
    root = Path(social_output_root)
    ensure_dir(root / "analysis")
    fig_dir = ensure_dir(root / "figures")
    dims = [x.strip() for x in style_dims.split(",") if x.strip()]
    made: list[Path] = []
    made += make_tsne_or_pca(root, fig_dir, max_items=fig_max_items, method=embedding_method, seed=seed)
    made += make_covid_fused_lines(root, fig_dir, dims)
    made += make_event_study_proxy(root, fig_dir, dims, event_month=event_month)
    made += make_prepost_bar(root, fig_dir)
    made += make_seasonality_heatmap(root, fig_dir, dims)
    if run_amazon_hm_compare:
        made += make_amazon_hm_figures(root, fig_dir, data_root, max_reviews=max_amazon_reviews, seed=seed)
    manifest = {"figures": [str(p) for p in made]}
    (fig_dir / "figures_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("========== Social figures finished ==========")
    for p in made:
        print(p)
    print(f"manifest: {fig_dir / 'figures_manifest.json'}")
    print("============================================")
    return made


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--social_output_root", default="./social_output")
    p.add_argument("--data_root", default="")
    p.add_argument("--event_month", default="2020-03")
    p.add_argument("--embedding_method", choices=["tsne", "pca"], default="tsne")
    p.add_argument("--fig_max_items", type=int, default=4000)
    p.add_argument("--style_dims", default="formal,comfort,homewear,value,casual,office")
    p.add_argument("--run_amazon_hm_compare", type=parse_bool, default=False)
    p.add_argument("--max_amazon_reviews", type=int, default=300000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    data_root = Path(args.data_root) if args.data_root else None
    make_all_figures(Path(args.social_output_root), data_root=data_root, event_month=args.event_month,
                     embedding_method=args.embedding_method, fig_max_items=args.fig_max_items,
                     style_dims=args.style_dims, run_amazon_hm_compare=bool(args.run_amazon_hm_compare),
                     max_amazon_reviews=args.max_amazon_reviews, seed=args.seed)


if __name__ == "__main__":
    main()
