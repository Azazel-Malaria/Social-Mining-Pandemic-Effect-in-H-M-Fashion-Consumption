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
from typing import Iterable

import numpy as np
import pandas as pd

from util.io_utils import ensure_dir, parse_bool, save_json

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover
    sm = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import mean_squared_error
    from sklearn.ensemble import RandomForestRegressor
except Exception:  # pragma: no cover
    PCA = KMeans = GaussianMixture = RandomForestRegressor = None
    mean_squared_error = None


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

DEFAULT_DIMS = ["formal", "office", "comfort", "homewear", "casual", "value"]
DEFAULT_COVID_VARS = [
    "covid_cases_index",
    "covid_deaths_index",
    "covid_stringency_index",
    "covid_reproduction_rate",
]


def _safe_name(x: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in ['_', '-'] else '_' for ch in str(x))[:140]


def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce').astype(float)
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return s * 0.0
    return (s - s.mean()) / sd


def _parse_list(s: str | None, default: list[str] | None = None) -> list[str]:
    if s is None or str(s).strip() == '':
        return list(default or [])
    return [x.strip() for x in str(s).split(',') if x.strip()]


def _score_col(dim: str, use_copula: bool = False) -> str:
    if dim.endswith('_score') or dim.endswith('_score_copula'):
        return dim
    return f'{dim}_score_copula' if use_copula else f'{dim}_score'


def _available_covid_vars(df: pd.DataFrame, requested: Iterable[str] | None = None) -> list[str]:
    req = list(requested or DEFAULT_COVID_VARS)
    return [c for c in req if c in df.columns and pd.to_numeric(df[c], errors='coerce').notna().sum() >= 3]


def _add_time_controls(df: pd.DataFrame, time_col: str = 'month') -> pd.DataFrame:
    out = df.copy()
    if time_col not in out.columns:
        return out
    if time_col == 'week':
        # String Period like 2020-02-24/2020-03-01; sort by string still mostly works but use categorical rank.
        vals = sorted(out[time_col].astype(str).dropna().unique().tolist())
        order = {v: i for i, v in enumerate(vals)}
        out['_trend'] = out[time_col].astype(str).map(order).astype(float)
        out['_calendar_month'] = out[time_col].astype(str).str[5:7].fillna('00')
    else:
        p = pd.PeriodIndex(out[time_col].astype(str), freq='M')
        out['_trend'] = (p.year - p.year.min()) * 12 + (p.month - 1)
        out['_calendar_month'] = p.month.astype(str)
    return out


def _fourier_controls(df: pd.DataFrame, order: int = 1, trend_col: str = '_trend') -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    cols: list[str] = []
    t = pd.to_numeric(out[trend_col], errors='coerce').fillna(0.0).to_numpy(dtype=float)
    for q in range(1, max(int(order), 0) + 1):
        s = f'_sin{q}'
        c = f'_cos{q}'
        out[s] = np.sin(2 * np.pi * q * t / 12.0)
        out[c] = np.cos(2 * np.pi * q * t / 12.0)
        cols.extend([s, c])
    return out, cols


def _ols(df: pd.DataFrame, y: str, xcols: list[str], fe_cols: list[str] | None = None, min_n: int = 10) -> dict:
    fe_cols = fe_cols or []
    cols = [y] + xcols + fe_cols
    d = df[cols].copy()
    for c in [y] + xcols:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[y] + xcols)
    if len(d) < min_n:
        return {'n': int(len(d)), 'ok': False, 'reason': 'not_enough_rows'}
    X_parts = [d[xcols].astype(float)] if xcols else []
    for fe in fe_cols:
        if fe in d.columns:
            dm = pd.get_dummies(d[fe].astype(str), prefix=fe, drop_first=True, dtype=float)
            if dm.shape[1] > 0:
                X_parts.append(dm)
    X = pd.concat(X_parts, axis=1) if X_parts else pd.DataFrame(index=d.index)
    X = sm.add_constant(X, has_constant='add') if sm is not None else X.assign(const=1.0)
    yv = d[y].astype(float)
    try:
        if sm is not None:
            res = sm.OLS(yv, X).fit(cov_type='HC1')
            ans = {'n': int(len(d)), 'ok': True, 'r2': float(getattr(res, 'rsquared', np.nan)), 'aic': float(getattr(res, 'aic', np.nan)), 'bic': float(getattr(res, 'bic', np.nan))}
            for x in xcols:
                ans[f'coef_{x}'] = float(res.params.get(x, np.nan))
                ans[f'se_{x}'] = float(res.bse.get(x, np.nan))
                ans[f'p_{x}'] = float(res.pvalues.get(x, np.nan))
            return ans
        beta = np.linalg.lstsq(X.to_numpy(dtype=float), yv.to_numpy(dtype=float), rcond=None)[0]
        ans = {'n': int(len(d)), 'ok': True}
        for x, b in zip(list(X.columns), beta):
            if x in xcols:
                ans[f'coef_{x}'] = float(b)
        return ans
    except Exception as exc:
        return {'n': int(len(d)), 'ok': False, 'reason': f'{type(exc).__name__}: {exc}'}


def _weighted_sample(values: np.ndarray, weights: np.ndarray, max_n: int, seed: int = 42) -> np.ndarray:
    values = np.asarray(values)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values).all(axis=1) if values.ndim == 2 else np.isfinite(values)
    weights = np.where(valid, weights, 0.0)
    if weights.sum() <= 0:
        idx = np.flatnonzero(valid)
        if len(idx) == 0:
            return values[:0]
        take = min(len(idx), max_n)
        rng = np.random.default_rng(seed)
        return values[rng.choice(idx, size=take, replace=False)]
    p = weights / weights.sum()
    take = min(max_n, len(values))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(values), size=take, replace=True, p=p)
    return values[idx]


def _empirical_cdf_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n, k = x.shape
    u = np.zeros_like(x)
    for j in range(k):
        order = np.argsort(x[:, j], kind='mergesort')
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1)
        u[:, j] = (ranks - 0.5) / max(n, 1)
    return np.clip(u, 1e-4, 1 - 1e-4)


def _norm_ppf_approx(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    a = np.array([-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00])
    b = np.array([-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01])
    c = np.array([-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00])
    d = np.array([7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00])
    plow = 0.02425
    phigh = 1 - plow
    x = np.zeros_like(u)
    mask = u < plow
    if np.any(mask):
        q = np.sqrt(-2 * np.log(u[mask])); x[mask] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    mask = (u >= plow) & (u <= phigh)
    if np.any(mask):
        q = u[mask] - 0.5; r = q*q; x[mask] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    mask = u > phigh
    if np.any(mask):
        q = np.sqrt(-2 * np.log(1 - u[mask])); x[mask] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    return x


def _js_divergence_hist(x: np.ndarray, y: np.ndarray, bins: int = 40) -> float:
    lo = np.nanmin([np.nanmin(x), np.nanmin(y)])
    hi = np.nanmax([np.nanmax(x), np.nanmax(y)])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0
    p, edges = np.histogram(x, bins=bins, range=(lo, hi), density=False)
    q, _ = np.histogram(y, bins=edges, density=False)
    p = p.astype(float) + 1e-12; q = q.astype(float) + 1e-12
    p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def _wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float)[np.isfinite(x)])
    y = np.sort(np.asarray(y, dtype=float)[np.isfinite(y)])
    if len(x) == 0 or len(y) == 0:
        return np.nan
    q = np.linspace(0, 1, min(len(x), len(y)))
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def _ensure_fig_dir(root: Path) -> Path:
    return ensure_dir(root / 'figures')


def _save_bar(df: pd.DataFrame, x: str, y: str, path: Path, title: str, top: int = 30):
    if plt is None or df.empty or x not in df.columns or y not in df.columns:
        return
    d = df[[x, y]].dropna().copy()
    d['_abs'] = pd.to_numeric(d[y], errors='coerce').abs()
    d = d.sort_values('_abs', ascending=False).head(top).sort_values(y)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(d))))
    ax.barh(d[x].astype(str), pd.to_numeric(d[y], errors='coerce'))
    ax.axvline(0, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(y)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_heatmap(mat: pd.DataFrame, path: Path, title: str):
    if plt is None or mat.empty:
        return
    fig, ax = plt.subplots(figsize=(max(5, 0.45 * mat.shape[1]), max(4, 0.35 * mat.shape[0])))
    im = ax.imshow(mat.to_numpy(dtype=float), aspect='auto')
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=45, ha='right')
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Experiment implementations
# -----------------------------------------------------------------------------


def exp_panel_check(args, root: Path, out: Path) -> dict:
    rows = []
    for name in [
        'item_monthly_panel', 'category_monthly_panel', 'prototype_monthly_panel', 'knowledge_monthly_panel',
        'knowledge_weekly_panel', 'channel_monthly_panel', 'channel_style_monthly_panel', 'user_monthly_panel', 'segment_style_monthly_panel',
        'item_knowledge_scores', 'item_knowledge_prototypes', 'item_embeddings'
    ]:
        p = root / f'{name}.parquet'
        if p.exists():
            try:
                df = pd.read_parquet(p)
                rows.append({'table': name, 'rows': len(df), 'cols': len(df.columns), 'columns': ','.join(map(str, df.columns[:60]))})
            except Exception as exc:
                rows.append({'table': name, 'rows': -1, 'cols': -1, 'columns': f'ERR {exc}'})
        else:
            rows.append({'table': name, 'rows': 0, 'cols': 0, 'columns': 'missing'})
    summary = pd.DataFrame(rows)
    summary.to_parquet(out / 'panel_check_summary.parquet', index=False)
    if (root / 'item_knowledge_scores.parquet').exists():
        scores = pd.read_parquet(root / 'item_knowledge_scores.parquet')
        sc = [c for c in scores.columns if c.endswith('_score') or c.endswith('_score_copula')]
        desc = scores[sc].describe().T.reset_index().rename(columns={'index':'score'}) if sc else pd.DataFrame()
        desc.to_parquet(out / 'style_score_summary.parquet', index=False)
    return {'tables': summary.to_dict(orient='records')}


def exp_prototype_covid(args, root: Path, out: Path) -> dict:
    panel_path = root / 'prototype_monthly_panel.parquet'
    if not panel_path.exists():
        raise FileNotFoundError(panel_path)
    panel = pd.read_parquet(panel_path)
    panel = _add_time_controls(panel, 'month')
    panel, fcols = _fourier_controls(panel, args.fourier_order)
    covid_vars = _available_covid_vars(panel, _parse_list(args.covid_vars, DEFAULT_COVID_VARS))
    rows = []
    for (dim, proto), g in panel.groupby(['dimension', 'prototype'], dropna=False):
        if g['month'].nunique() < args.min_periods:
            continue
        for cv in covid_vars:
            xcols = [cv, '_trend'] + fcols
            res = _ols(g, 'sales_share', xcols, min_n=args.min_periods)
            coef = res.get(f'coef_{cv}', np.nan)
            pval = res.get(f'p_{cv}', np.nan)
            rows.append({**res, 'dimension': dim, 'prototype': str(proto), 'covid_var': cv, 'coef': coef, 'pvalue': pval, 'mean_share': float(g['sales_share'].mean()), 'total_sales': float(g['sales_count'].sum())})
    ans = pd.DataFrame(rows)
    ans.to_parquet(out / 'prototype_covid_response.parquet', index=False)
    stat_rows = []
    outlier_rows = []
    for cv, g in ans.dropna(subset=['coef']).groupby('covid_var'):
        coef = g['coef'].to_numpy(dtype=float)
        if len(coef) == 0:
            continue
        mu = np.nanmean(coef); sd = np.nanstd(coef)
        kurt = float(np.nanmean(((coef - mu) / (sd + 1e-12)) ** 4)) if sd > 0 else np.nan
        med = np.nanmedian(coef); mad = np.nanmedian(np.abs(coef - med)) + 1e-12
        stat_rows.append({'covid_var': cv, 'n': len(coef), 'mean': float(mu), 'std': float(sd), 'variance': float(sd*sd), 'kurtosis': kurt, 'median': float(med), 'mad': float(mad)})
        z = (g['coef'] - med) / (1.4826 * mad)
        tmp = g.assign(robust_z=z)
        tmp = tmp[tmp['robust_z'].abs() >= args.outlier_bound].copy()
        outlier_rows.append(tmp)
    stats = pd.DataFrame(stat_rows)
    stats.to_parquet(out / 'prototype_response_moments.parquet', index=False)
    outliers = pd.concat(outlier_rows, ignore_index=True) if outlier_rows else pd.DataFrame()
    if len(outliers):
        # Cluster outlier prototypes by response vector + share + sales.
        piv = outliers.pivot_table(index=['dimension','prototype'], columns='covid_var', values='coef', aggfunc='mean').fillna(0.0)
        extra = outliers.groupby(['dimension','prototype'], as_index=True).agg(mean_share=('mean_share','mean'), total_sales=('total_sales','mean'))
        feat = piv.join(extra, how='left').fillna(0.0)
        if KMeans is not None and len(feat) >= 2:
            k = min(args.n_clusters, len(feat))
            lab = KMeans(n_clusters=k, random_state=args.seed, n_init='auto').fit_predict(feat.to_numpy(dtype=float))
            clusters = feat.reset_index()
            clusters['cluster'] = lab
        else:
            clusters = feat.reset_index(); clusters['cluster'] = 0
        clusters.to_parquet(out / 'prototype_outlier_clusters.parquet', index=False)
    outliers.to_parquet(out / 'prototype_response_outliers.parquet', index=False)
    fig = _ensure_fig_dir(out)
    if len(ans):
        # Volcano-style plot for primary covid var.
        cv = covid_vars[0] if covid_vars else ans['covid_var'].iloc[0]
        d = ans[ans['covid_var'] == cv].dropna(subset=['coef']).copy()
        if plt is not None and len(d):
            d['neglogp'] = -np.log10(pd.to_numeric(d['pvalue'], errors='coerce').fillna(1.0).clip(lower=1e-12))
            fig0, ax = plt.subplots(figsize=(7, 5))
            sizes = 10 + 80 * (np.log1p(d['total_sales']) / max(np.log1p(d['total_sales']).max(), 1e-8))
            ax.scatter(d['coef'], d['neglogp'], s=sizes, alpha=0.6)
            ax.axvline(0, linewidth=0.8)
            ax.set_title(f'Prototype response volcano: {cv}')
            ax.set_xlabel('COVID response coefficient')
            ax.set_ylabel('-log10(p-value)')
            fig0.tight_layout(); fig0.savefig(fig / 'prototype_response_volcano.png', dpi=180); plt.close(fig0)
        _save_bar(d.assign(label=d['dimension'].astype(str)+': '+d['prototype'].astype(str)), 'label', 'coef', fig / 'prototype_response_top_coefficients.png', f'Top prototype responses: {cv}', top=30)
    return {'n_results': int(len(ans)), 'n_outliers': int(len(outliers))}


def exp_style_distribution(args, root: Path, out: Path) -> dict:
    im = pd.read_parquet(root / 'item_monthly_panel.parquet')
    event = pd.Period(args.event_month, freq='M')
    pi = pd.PeriodIndex(im['month'].astype(str), freq='M')
    im['_period'] = pi
    dims = _parse_list(args.style_dims, DEFAULT_DIMS)
    cols = [_score_col(d, args.use_copula_scores) for d in dims]
    cols = [c for c in cols if c in im.columns]
    if not cols:
        raise RuntimeError('No requested style score columns found.')
    pre = im[im['_period'] < event]
    post = im[im['_period'] >= event]
    rows = []
    samples_pre = {}
    samples_post = {}
    for c in cols:
        xpre = _weighted_sample(pre[[c]].to_numpy(), pre['sales_count'].to_numpy(), args.max_distribution_samples, args.seed).reshape(-1)
        xpost = _weighted_sample(post[[c]].to_numpy(), post['sales_count'].to_numpy(), args.max_distribution_samples, args.seed + 1).reshape(-1)
        samples_pre[c] = xpre; samples_post[c] = xpost
        rows.append({'style_dim': c.replace('_score_copula','').replace('_score',''), 'score_col': c, 'pre_mean': float(np.nanmean(xpre)), 'post_mean': float(np.nanmean(xpost)), 'mean_shift': float(np.nanmean(xpost)-np.nanmean(xpre)), 'wasserstein': _wasserstein_1d(xpre, xpost), 'js_divergence': _js_divergence_hist(xpre, xpost, bins=args.hist_bins)})
    shift = pd.DataFrame(rows)
    shift.to_parquet(out / 'style_distribution_distance.parquet', index=False)
    # Multivariate samples for PCA/copula.
    mat_pre = _weighted_sample(pre[cols].to_numpy(dtype=float), pre['sales_count'].to_numpy(), args.max_distribution_samples, args.seed)
    mat_post = _weighted_sample(post[cols].to_numpy(dtype=float), post['sales_count'].to_numpy(), args.max_distribution_samples, args.seed + 1)
    # PCA center shift.
    if PCA is not None and len(mat_pre) + len(mat_post) >= 10 and len(cols) >= 2:
        X = np.vstack([mat_pre, mat_post])
        X = np.nan_to_num(X, nan=0.0)
        pca = PCA(n_components=2, random_state=args.seed).fit(X)
        zpre = pca.transform(mat_pre); zpost = pca.transform(mat_post)
        pca_rows = pd.DataFrame({
            'metric': ['pc1_mean_shift','pc2_mean_shift','pc_distance','explained_var_pc1','explained_var_pc2'],
            'value': [float(zpost[:,0].mean()-zpre[:,0].mean()), float(zpost[:,1].mean()-zpre[:,1].mean()), float(np.linalg.norm(zpost.mean(axis=0)-zpre.mean(axis=0))), float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1])]
        })
        pca_rows.to_parquet(out / 'style_pca_shift.parquet', index=False)
    # Gaussian copula correlations.
    def corr_copula(mat):
        u = _empirical_cdf_matrix(np.nan_to_num(mat, nan=0.0))
        z = _norm_ppf_approx(u)
        return pd.DataFrame(np.corrcoef(z, rowvar=False), index=[c.replace('_score_copula','').replace('_score','') for c in cols], columns=[c.replace('_score_copula','').replace('_score','') for c in cols])
    cop_pre = corr_copula(mat_pre)
    cop_post = corr_copula(mat_post)
    cop_delta = cop_post - cop_pre
    cop_pre.to_parquet(out / 'style_copula_corr_pre.parquet')
    cop_post.to_parquet(out / 'style_copula_corr_post.parquet')
    cop_delta.to_parquet(out / 'style_copula_corr_delta.parquet')
    pd.DataFrame([{'marginal_shift_mean_wasserstein': float(shift['wasserstein'].mean()), 'copula_shift_frobenius': float(np.linalg.norm(cop_delta.to_numpy(dtype=float), ord='fro'))}]).to_parquet(out / 'style_shift_indices.parquet', index=False)
    fig = _ensure_fig_dir(out)
    if plt is not None:
        # KDE-like hist density curves for top 3 dims by shift.
        top = shift.sort_values('wasserstein', ascending=False).head(min(3, len(shift)))['score_col'].tolist()
        fig0, ax = plt.subplots(figsize=(7, 4))
        for c in top:
            for label, arr, ls in [('pre', samples_pre[c], '-'), ('post', samples_post[c], '--')]:
                hist, edges = np.histogram(arr[np.isfinite(arr)], bins=args.hist_bins, range=(0, 1), density=True)
                centers = 0.5 * (edges[:-1] + edges[1:])
                ax.plot(centers, hist, linestyle=ls, label=f"{c.replace('_score_copula','').replace('_score','')} {label}")
        ax.set_title('Pre/post style score distributions')
        ax.set_xlabel('Style score')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        fig0.tight_layout(); fig0.savefig(fig / 'style_kde_prepost.png', dpi=180); plt.close(fig0)
        _save_heatmap(cop_delta, fig / 'style_copula_delta_heatmap.png', 'Post - pre Gaussian copula correlation')
    return {'n_dims': len(cols), 'marginal_shift': float(shift['wasserstein'].mean())}


def _twoway_demean(df: pd.DataFrame, cols: list[str], entity: str = 'article_id', time: str = 'month') -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        v = pd.to_numeric(out[c], errors='coerce')
        out[f'{c}__dm'] = v - v.groupby(out[entity]).transform('mean') - v.groupby(out[time]).transform('mean') + v.mean()
    return out


def _fast_dm_ols(df: pd.DataFrame, y_dm: str, x_dm: str, min_n: int = 500) -> dict:
    d = df[[y_dm, x_dm]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < min_n or d[x_dm].std() <= 1e-12:
        return {'n': int(len(d)), 'ok': False, 'reason': 'not_enough_rows_or_no_variation'}
    try:
        if sm is not None:
            X = sm.add_constant(d[[x_dm]], has_constant='add')
            res = sm.OLS(d[y_dm], X).fit(cov_type='HC1')
            return {'n': int(len(d)), 'ok': True, 'coef': float(res.params.get(x_dm, np.nan)), 'se': float(res.bse.get(x_dm, np.nan)), 'pvalue': float(res.pvalues.get(x_dm, np.nan)), 'r2': float(res.rsquared)}
        x = d[x_dm].to_numpy(dtype=float); y = d[y_dm].to_numpy(dtype=float)
        beta = float(np.dot(x, y) / max(np.dot(x, x), 1e-12))
        return {'n': int(len(d)), 'ok': True, 'coef': beta}
    except Exception as exc:
        return {'n': int(len(d)), 'ok': False, 'reason': f'{type(exc).__name__}: {exc}'}


def exp_exposure_response(args, root: Path, out: Path) -> dict:
    im = pd.read_parquet(root / 'item_monthly_panel.parquet')
    im = _add_time_controls(im, 'month')
    event = pd.Period(args.event_month, freq='M')
    pi = pd.PeriodIndex(im['month'].astype(str), freq='M')
    im['post'] = (pi >= event).astype(int)
    im['rel_month'] = (pi.year - event.year) * 12 + (pi.month - event.month)
    im['log_sales'] = np.log1p(pd.to_numeric(im['sales_count'], errors='coerce').fillna(0.0))
    # Keep this experiment bounded. Dense item-month panels can be large; two-way demeaning avoids dummy FE.
    if args.max_panel_rows and len(im) > args.max_panel_rows:
        im = im.sample(n=args.max_panel_rows, random_state=args.seed).reset_index(drop=True)
    dims = _parse_list(args.style_dims, ['formal','comfort','homewear'])
    rows = []
    trends = []
    for dim in dims:
        sc = _score_col(dim, args.use_copula_scores)
        if sc not in im.columns:
            continue
        vals = im[['article_id', sc]].drop_duplicates('article_id')
        hi_thr = vals[sc].quantile(args.high_quantile)
        lo_thr = vals[sc].quantile(args.low_quantile)
        group_map = vals.assign(exposure_group=np.where(vals[sc] >= hi_thr, 'high', np.where(vals[sc] <= lo_thr, 'low', 'middle')))[['article_id','exposure_group']]
        tmp = im.merge(group_map, on='article_id', how='left')
        tmp = tmp[tmp['exposure_group'].isin(['high','low'])].copy()
        tmp['high'] = (tmp['exposure_group'] == 'high').astype(float)
        tmp['high_x_post'] = tmp['high'] * tmp['post']
        tmp['score_x_post'] = pd.to_numeric(tmp[sc], errors='coerce').fillna(0.0) * tmp['post']
        dm = _twoway_demean(tmp, ['log_sales', 'high_x_post', 'score_x_post'], entity='article_id', time='month')
        res = _fast_dm_ols(dm, 'log_sales__dm', 'high_x_post__dm', min_n=args.min_rows_panel)
        rows.append({**res, 'style_dim': dim, 'score_col': sc, 'model': 'twoway_demeaned_high_x_post'})
        res2 = _fast_dm_ols(dm, 'log_sales__dm', 'score_x_post__dm', min_n=args.min_rows_panel)
        rows.append({**res2, 'style_dim': dim, 'score_col': sc, 'model': 'twoway_demeaned_score_x_post'})
        agg = tmp.groupby(['rel_month','exposure_group'], as_index=False)['sales_count'].sum()
        piv = agg.pivot(index='rel_month', columns='exposure_group', values='sales_count').fillna(0.0)
        if {'high','low'}.issubset(piv.columns):
            piv['log_high_low_gap'] = np.log1p(piv['high']) - np.log1p(piv['low'])
            piv['style_dim'] = dim
            trends.append(piv.reset_index())
    ans = pd.DataFrame(rows)
    ans.to_parquet(out / 'exposure_response_results.parquet', index=False)
    trends_df = pd.concat(trends, ignore_index=True) if trends else pd.DataFrame()
    trends_df.to_parquet(out / 'exposure_high_low_event_trends.parquet', index=False)
    fig = _ensure_fig_dir(out)
    if len(ans):
        _save_bar(ans, 'style_dim', 'coef', fig / 'exposure_response_coefficients.png', 'Style exposure post-COVID differential response')
    if plt is not None and len(trends_df):
        fig0, ax = plt.subplots(figsize=(7, 4))
        for dim, g in trends_df.groupby('style_dim'):
            ax.plot(g['rel_month'], g['log_high_low_gap'], marker='o', label=dim)
        ax.axvline(0, linewidth=0.8)
        ax.axhline(0, linewidth=0.8)
        ax.set_title('High-vs-low style exposure dynamic gap')
        ax.set_xlabel('Relative month to event')
        ax.set_ylabel('log(high sales) - log(low sales)')
        ax.legend(fontsize=8)
        fig0.tight_layout(); fig0.savefig(fig / 'exposure_event_study_gap.png', dpi=180); plt.close(fig0)
    return {'n_results': int(len(ans))}


def exp_lead_lag(args, root: Path, out: Path) -> dict:
    path = root / 'knowledge_weekly_panel.parquet'
    if not path.exists() or pd.read_parquet(path).empty:
        path = root / 'knowledge_monthly_panel.parquet'
        time_col = 'month'
    else:
        time_col = 'week'
    panel = pd.read_parquet(path)
    if panel.empty:
        raise RuntimeError('No knowledge weekly/monthly panel found.')
    value_col = 'weighted_share' if 'weighted_share' in panel.columns else 'share'
    dims = _parse_list(args.style_dims, DEFAULT_DIMS)
    covid_vars = _available_covid_vars(panel, _parse_list(args.covid_vars, DEFAULT_COVID_VARS))
    rows = []
    ardl_rows = []
    for dim in dims:
        g = panel[panel['knowledge_dim'].astype(str).str.replace('_copula','', regex=False) == dim].copy()
        if g.empty:
            continue
        g = g.sort_values(time_col).reset_index(drop=True)
        y = pd.to_numeric(g[value_col], errors='coerce')
        for cv in covid_vars:
            x = pd.to_numeric(g[cv], errors='coerce')
            for lag in range(-args.max_lead, args.max_lag + 1):
                # lag > 0 means COVID leads style by lag periods: corr(y_t, x_{t-lag})
                xs = x.shift(lag)
                ok = y.notna() & xs.notna()
                corr = float(np.corrcoef(y[ok], xs[ok])[0,1]) if ok.sum() >= args.min_periods and y[ok].std() > 0 and xs[ok].std() > 0 else np.nan
                rows.append({'knowledge_dim': dim, 'covid_var': cv, 'lag': lag, 'corr': corr, 'n': int(ok.sum()), 'time_col': time_col})
            # ARDL / predictive comparison.
            d = pd.DataFrame({'y': y, 'x': x})
            for p in range(1, args.ar_lags + 1):
                d[f'y_lag{p}'] = d['y'].shift(p)
            for l in range(0, args.max_lag + 1):
                d[f'x_lag{l}'] = d['x'].shift(l)
            d = d.dropna()
            if len(d) >= max(args.min_periods, 8):
                split = max(int(len(d) * 0.7), 3)
                train, test = d.iloc[:split], d.iloc[split:]
                base_cols = [f'y_lag{p}' for p in range(1, args.ar_lags + 1)]
                full_cols = base_cols + [f'x_lag{l}' for l in range(0, args.max_lag + 1)]
                res = _ols(d, 'y', full_cols, min_n=args.min_periods)
                # Simple OOS RMSE.
                rmse_base = rmse_full = np.nan
                if sm is not None and len(test) >= 2:
                    try:
                        mb = sm.OLS(train['y'], sm.add_constant(train[base_cols], has_constant='add')).fit()
                        mf = sm.OLS(train['y'], sm.add_constant(train[full_cols], has_constant='add')).fit()
                        pb = mb.predict(sm.add_constant(test[base_cols], has_constant='add'))
                        pf = mf.predict(sm.add_constant(test[full_cols], has_constant='add'))
                        rmse_base = float(np.sqrt(np.mean((test['y'].to_numpy()-pb.to_numpy())**2)))
                        rmse_full = float(np.sqrt(np.mean((test['y'].to_numpy()-pf.to_numpy())**2)))
                    except Exception:
                        pass
                ardl_rows.append({**res, 'knowledge_dim': dim, 'covid_var': cv, 'rmse_ar_only': rmse_base, 'rmse_ar_plus_covid': rmse_full, 'rmse_improvement': (rmse_base-rmse_full) if np.isfinite(rmse_base) and np.isfinite(rmse_full) else np.nan})
    corr_df = pd.DataFrame(rows)
    ardl_df = pd.DataFrame(ardl_rows)
    corr_df.to_parquet(out / 'lead_lag_correlation.parquet', index=False)
    ardl_df.to_parquet(out / 'ardl_granger_style_results.parquet', index=False)
    fig = _ensure_fig_dir(out)
    if len(corr_df):
        cv = covid_vars[0] if covid_vars else corr_df['covid_var'].iloc[0]
        mat = corr_df[corr_df['covid_var'] == cv].pivot_table(index='knowledge_dim', columns='lag', values='corr', aggfunc='mean')
        _save_heatmap(mat, fig / 'lead_lag_correlation_heatmap.png', f'Lead-lag correlation: {cv}')
    return {'n_corr': int(len(corr_df)), 'n_ardl': int(len(ardl_df))}


def exp_age_style(args, root: Path, out: Path) -> dict:
    up = pd.read_parquet(root / 'user_monthly_panel.parquet')
    if up.empty or 'age' not in up.columns:
        raise RuntimeError('user_monthly_panel with age is required. Re-run social inference with hm_customers.parquet available.')
    event = pd.Period(args.event_month, freq='M')
    pi = pd.PeriodIndex(up['month'].astype(str), freq='M')
    up['period_group'] = np.where(pi < event, 'pre', 'post')
    dims = _parse_list(args.style_dims, DEFAULT_DIMS)
    rows = []
    density_rows = []
    cop_rows = []
    for dim in dims:
        pref = f'{dim}_pref'
        if pref not in up.columns:
            pref = f'{dim}_copula_pref'
        if pref not in up.columns:
            continue
        d = up[['age','period_group',pref,'num_purchases']].copy()
        d['age'] = pd.to_numeric(d['age'], errors='coerce')
        d[pref] = pd.to_numeric(d[pref], errors='coerce')
        d = d.dropna(subset=['age', pref])
        if d.empty:
            continue
        # Weighted age distribution for style purchases: use pref*num_purchases as weight.
        for pg, g in d.groupby('period_group'):
            w = (pd.to_numeric(g[pref], errors='coerce').fillna(0.0) * pd.to_numeric(g['num_purchases'], errors='coerce').fillna(0.0)).to_numpy(dtype=float)
            a = g['age'].to_numpy(dtype=float)
            sample = _weighted_sample(a, w, args.max_distribution_samples, args.seed)
            if len(sample):
                density_rows.append(pd.DataFrame({'style_dim': dim, 'period_group': pg, 'age': sample}))
                rows.append({'style_dim': dim, 'period_group': pg, 'mean_age': float(np.mean(sample)), 'std_age': float(np.std(sample)), 'n_sample': int(len(sample))})
                if GaussianMixture is not None and len(sample) >= 30:
                    X = sample.reshape(-1, 1)
                    best = None
                    for k in range(1, min(args.gmm_components, 4) + 1):
                        try:
                            gm = GaussianMixture(n_components=k, random_state=args.seed).fit(X)
                            bic = gm.bic(X)
                            if best is None or bic < best[0]:
                                best = (bic, gm)
                        except Exception:
                            pass
                    if best is not None:
                        gm = best[1]
                        for j in range(gm.n_components):
                            rows.append({'style_dim': dim, 'period_group': pg, 'gmm_component': j, 'gmm_weight': float(gm.weights_[j]), 'gmm_mean_age': float(gm.means_[j,0]), 'gmm_std_age': float(np.sqrt(gm.covariances_[j].reshape(-1)[0])), 'bic': float(best[0])})
        # Age-style copula proxy: Spearman-style correlation after empirical CDF.
        for pg, g in d.groupby('period_group'):
            if len(g) >= args.min_periods:
                mat = g[['age', pref]].to_numpy(dtype=float)
                u = _empirical_cdf_matrix(mat)
                z = _norm_ppf_approx(u)
                rho = float(np.corrcoef(z, rowvar=False)[0,1])
                cop_rows.append({'style_dim': dim, 'period_group': pg, 'gaussian_copula_corr': rho, 'n': int(len(g))})
    pd.DataFrame(rows).to_parquet(out / 'age_style_gmm_results.parquet', index=False)
    dens = pd.concat(density_rows, ignore_index=True) if density_rows else pd.DataFrame()
    dens.to_parquet(out / 'age_style_density_samples.parquet', index=False)
    cop = pd.DataFrame(cop_rows)
    cop.to_parquet(out / 'age_style_copula_results.parquet', index=False)
    fig = _ensure_fig_dir(out)
    if plt is not None and len(dens):
        fig0, ax = plt.subplots(figsize=(7, 4))
        top_dims = dens['style_dim'].drop_duplicates().head(4).tolist()
        for dim in top_dims:
            for pg, ls in [('pre','-'),('post','--')]:
                arr = dens[(dens['style_dim']==dim)&(dens['period_group']==pg)]['age'].to_numpy(dtype=float)
                if len(arr):
                    hist, edges = np.histogram(arr, bins=30, range=(10, 80), density=True)
                    centers = 0.5*(edges[:-1]+edges[1:])
                    ax.plot(centers, hist, linestyle=ls, label=f'{dim} {pg}')
        ax.set_title('Age distributions by style and period')
        ax.set_xlabel('Age'); ax.set_ylabel('Density'); ax.legend(fontsize=8)
        fig0.tight_layout(); fig0.savefig(fig / 'age_style_kde.png', dpi=180); plt.close(fig0)
    if len(cop):
        mat = cop.pivot_table(index='style_dim', columns='period_group', values='gaussian_copula_corr', aggfunc='mean')
        if {'pre','post'}.issubset(mat.columns):
            mat['post_minus_pre'] = mat['post'] - mat['pre']
        _save_heatmap(mat, fig / 'age_style_copula_heatmap.png', 'Age-style Gaussian copula correlation')
    return {'n_density': int(len(dens)), 'n_copula': int(len(cop))}


def exp_channel_style(args, root: Path, out: Path) -> dict:
    ch = pd.read_parquet(root / 'channel_monthly_panel.parquet') if (root / 'channel_monthly_panel.parquet').exists() else pd.DataFrame()
    cs = pd.read_parquet(root / 'channel_style_monthly_panel.parquet') if (root / 'channel_style_monthly_panel.parquet').exists() else pd.DataFrame()
    rows = []
    if len(ch):
        ch = _add_time_controls(ch, 'month'); ch, fcols = _fourier_controls(ch, args.fourier_order)
        covid_vars = _available_covid_vars(ch, _parse_list(args.covid_vars, DEFAULT_COVID_VARS))
        for chan, g in ch.groupby('sales_channel_id'):
            for cv in covid_vars:
                res = _ols(g, 'sales_share', [cv, '_trend'] + fcols, min_n=args.min_periods)
                rows.append({**res, 'sales_channel_id': chan, 'covid_var': cv, 'model': 'channel_share'})
    pd.DataFrame(rows).to_parquet(out / 'channel_share_covid_results.parquet', index=False)
    style_rows = []
    if len(cs):
        cs = _add_time_controls(cs, 'month'); cs, fcols = _fourier_controls(cs, args.fourier_order)
        covid_vars = _available_covid_vars(cs, _parse_list(args.covid_vars, DEFAULT_COVID_VARS))
        for (chan, dim), g in cs.groupby(['sales_channel_id','knowledge_dim']):
            if str(dim).replace('_copula','') not in _parse_list(args.style_dims, DEFAULT_DIMS):
                continue
            for cv in covid_vars:
                res = _ols(g, 'weighted_share', [cv, '_trend'] + fcols, min_n=args.min_periods)
                style_rows.append({**res, 'sales_channel_id': chan, 'knowledge_dim': dim, 'covid_var': cv, 'model': 'channel_style_share'})
    style_df = pd.DataFrame(style_rows)
    style_df.to_parquet(out / 'channel_style_shift_results.parquet', index=False)
    # Pre/post heatmap.
    if len(cs):
        event = pd.Period(args.event_month, freq='M')
        pi = pd.PeriodIndex(cs['month'].astype(str), freq='M')
        tmp = cs.copy(); tmp['period_group'] = np.where(pi < event, 'pre', 'post')
        prepost = tmp.groupby(['sales_channel_id','knowledge_dim','period_group'], as_index=False)['weighted_share'].mean()
        piv = prepost.pivot_table(index=['sales_channel_id','knowledge_dim'], columns='period_group', values='weighted_share')
        if {'pre','post'}.issubset(piv.columns):
            piv['delta_post_pre'] = piv['post'] - piv['pre']
        piv.reset_index().to_parquet(out / 'channel_style_prepost.parquet', index=False)
        fig = _ensure_fig_dir(out)
        if 'delta_post_pre' in piv.columns:
            mat = piv['delta_post_pre'].unstack('knowledge_dim').fillna(0.0)
            _save_heatmap(mat, fig / 'channel_style_prepost_heatmap.png', 'Channel × style post-pre shift')
    if len(ch) and plt is not None:
        fig = _ensure_fig_dir(out)
        fig0, ax = plt.subplots(figsize=(7,4))
        for chan, g in ch.groupby('sales_channel_id'):
            ax.plot(g.sort_values('month')['month'], g.sort_values('month')['sales_share'], marker='o', label=f'channel {chan}')
        ax.set_title('Sales channel shares')
        ax.set_xlabel('Month'); ax.set_ylabel('Share'); ax.tick_params(axis='x', rotation=45); ax.legend()
        fig0.tight_layout(); fig0.savefig(fig / 'channel_share_lines.png', dpi=180); plt.close(fig0)
    return {'n_channel': int(len(rows)), 'n_channel_style': int(len(style_rows))}


def exp_member_news(args, root: Path, out: Path) -> dict:
    seg = pd.read_parquet(root / 'segment_style_monthly_panel.parquet') if (root / 'segment_style_monthly_panel.parquet').exists() else pd.DataFrame()
    if seg.empty:
        raise RuntimeError('segment_style_monthly_panel is missing or empty. Re-run social inference with customer metadata.')
    dims = _parse_list(args.style_dims, DEFAULT_DIMS)
    seg = seg[seg['knowledge_dim'].astype(str).str.replace('_copula','', regex=False).isin(dims)].copy()
    event = pd.Period(args.event_month, freq='M')
    pi = pd.PeriodIndex(seg['month'].astype(str), freq='M')
    seg['period_group'] = np.where(pi < event, 'pre', 'post')
    prepost = seg.groupby(['segment_field','segment_value','knowledge_dim','period_group'], as_index=False)['weighted_share'].mean()
    piv = prepost.pivot_table(index=['segment_field','segment_value','knowledge_dim'], columns='period_group', values='weighted_share')
    if {'pre','post'}.issubset(piv.columns):
        piv['delta_post_pre'] = piv['post'] - piv['pre']
    pp = piv.reset_index()
    pp.to_parquet(out / 'segment_style_prepost.parquet', index=False)
    # COVID response per segment/style.
    seg = _add_time_controls(seg, 'month'); seg, fcols = _fourier_controls(seg, args.fourier_order)
    covid_vars = _available_covid_vars(seg, _parse_list(args.covid_vars, DEFAULT_COVID_VARS))
    rows = []
    for (field, val, dim), g in seg.groupby(['segment_field','segment_value','knowledge_dim']):
        if g['month'].nunique() < args.min_periods:
            continue
        for cv in covid_vars:
            res = _ols(g, 'weighted_share', [cv, '_trend'] + fcols, min_n=args.min_periods)
            rows.append({**res, 'segment_field': field, 'segment_value': val, 'knowledge_dim': dim, 'covid_var': cv})
    pd.DataFrame(rows).to_parquet(out / 'segment_style_covid_response.parquet', index=False)
    fig = _ensure_fig_dir(out)
    if 'delta_post_pre' in pp.columns:
        for field, g in pp.groupby('segment_field'):
            mat = g.pivot_table(index='segment_value', columns='knowledge_dim', values='delta_post_pre', aggfunc='mean').fillna(0.0)
            _save_heatmap(mat, fig / f'{_safe_name(field)}_style_prepost_heatmap.png', f'{field} × style post-pre shift')
    return {'n_prepost': int(len(pp)), 'n_response': int(len(rows))}


def exp_seasonality(args, root: Path, out: Path) -> dict:
    results = []
    for file, id_cols, ycol, name in [
        ('category_monthly_panel.parquet', ['category_field','category_value'], 'sales_count', 'category'),
        ('prototype_monthly_panel.parquet', ['dimension','prototype'], 'sales_count', 'prototype'),
        ('knowledge_monthly_panel.parquet', ['knowledge_dim'], 'weighted_sales', 'knowledge'),
    ]:
        p = root / file
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue
        df = _add_time_controls(df, 'month'); df, fcols = _fourier_controls(df, args.fourier_order)
        df['_log_y'] = np.log1p(pd.to_numeric(df[ycol], errors='coerce').fillna(0.0))
        rows = []
        for key, g in df.groupby(id_cols, dropna=False):
            if not isinstance(key, tuple): key = (key,)
            if g['month'].nunique() < args.min_periods:
                continue
            res = _ols(g, '_log_y', ['_trend'] + fcols, min_n=args.min_periods)
            amp = 0.0
            for q in range(1, args.fourier_order+1):
                s = res.get(f'coef__sin{q}', 0.0); c = res.get(f'coef__cos{q}', 0.0)
                if np.isfinite(s) and np.isfinite(c): amp += s*s + c*c
            row = {**res, 'panel': name, 'season_amp': float(np.sqrt(amp))}
            for col, val in zip(id_cols, key): row[col] = val
            rows.append(row)
        ans = pd.DataFrame(rows)
        ans.to_parquet(out / f'seasonality_{name}_fourier_results.parquet', index=False)
        results.append(ans)
    allres = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if len(allres):
        fig = _ensure_fig_dir(out)
        lab_col = 'category_value' if 'category_value' in allres.columns else ('prototype' if 'prototype' in allres.columns else 'knowledge_dim')
        d = allres.copy(); d['label'] = d.get('panel','').astype(str)+': '+d[lab_col].astype(str)
        _save_bar(d, 'label', 'season_amp', fig / 'seasonality_strength_top.png', 'Top seasonal amplitudes', top=30)
    return {'n_results': int(len(allres))}


def exp_ml_heterogeneity(args, root: Path, out: Path) -> dict:
    im = pd.read_parquet(root / 'item_monthly_panel.parquet')
    dims = _parse_list(args.style_dims, DEFAULT_DIMS)
    cov = 'covid_cases_index' if 'covid_cases_index' in im.columns else None
    if cov is None:
        raise RuntimeError('No covid_cases_index found in item_monthly_panel.')
    im = _add_time_controls(im, 'month'); im, fcols = _fourier_controls(im, args.fourier_order)
    im['log_sales'] = np.log1p(pd.to_numeric(im['sales_count'], errors='coerce').fillna(0.0))
    betas = []
    for aid, g in im.groupby('article_id'):
        if g['month'].nunique() < args.min_periods:
            continue
        res = _ols(g, 'log_sales', [cov, '_trend'] + fcols, min_n=args.min_periods)
        betas.append({'article_id': aid, 'covid_response': res.get(f'coef_{cov}', np.nan), 'n': res.get('n', 0)})
    bdf = pd.DataFrame(betas).dropna(subset=['covid_response'])
    scores = pd.read_parquet(root / 'item_knowledge_scores.parquet')
    meta = pd.read_parquet(root / 'item_metadata.parquet')
    df = bdf.merge(scores, on='article_id', how='left').merge(meta, on='article_id', how='left')
    xcols = [c for c in [_score_col(d, args.use_copula_scores) for d in dims] if c in df.columns]
    cat_cols = [c for c in ['product_group_name','product_type_name','colour_group_name','garment_group_name'] if c in df.columns]
    X_parts = [df[xcols].apply(pd.to_numeric, errors='coerce').fillna(0.0)] if xcols else []
    for c in cat_cols:
        X_parts.append(pd.get_dummies(df[c].astype(str), prefix=c, dtype=float))
    if not X_parts:
        raise RuntimeError('No features for heterogeneity model.')
    X = pd.concat(X_parts, axis=1)
    y = df['covid_response'].to_numpy(dtype=float)
    if RandomForestRegressor is not None and len(df) >= 50:
        rf = RandomForestRegressor(n_estimators=args.rf_trees, max_depth=args.rf_max_depth, random_state=args.seed, n_jobs=-1).fit(X, y)
        imp = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)
    else:
        imp = pd.DataFrame({'feature': X.columns, 'importance': np.nan})
    imp.to_parquet(out / 'ml_heterogeneity_feature_importance.parquet', index=False)
    df[['article_id','covid_response'] + xcols + cat_cols].to_parquet(out / 'item_covid_response_by_item.parquet', index=False)
    fig = _ensure_fig_dir(out)
    if len(imp):
        _save_bar(imp, 'feature', 'importance', fig / 'ml_heterogeneity_feature_importance.png', 'Feature importance for item COVID response', top=30)
    return {'n_items': int(len(df)), 'n_features': int(X.shape[1])}


EXPERIMENTS = {
    'panel_check': exp_panel_check,
    'prototype_covid': exp_prototype_covid,
    'style_distribution': exp_style_distribution,
    'exposure_response': exp_exposure_response,
    'lead_lag': exp_lead_lag,
    'age_style': exp_age_style,
    'channel_style': exp_channel_style,
    'member_news': exp_member_news,
    'seasonality': exp_seasonality,
    'ml_heterogeneity': exp_ml_heterogeneity,
}
ALIASES = {
    '00': 'panel_check', '01': 'prototype_covid', '02': 'style_distribution', '03': 'exposure_response',
    '04': 'lead_lag', '05': 'age_style', '06': 'channel_style', '07': 'member_news', '08': 'seasonality', '09': 'ml_heterogeneity',
}


def run_experiment(args) -> None:
    social_root = Path(args.social_output_root)
    exp_name = args.exp_name or ALIASES.get(str(args.exp_id), str(args.exp_id))
    if exp_name not in EXPERIMENTS:
        raise ValueError(f'Unknown experiment {args.exp_id}/{args.exp_name}. Available: {sorted(EXPERIMENTS)}')
    out = ensure_dir(social_root / 'experiments' / exp_name)
    print(f'[social_experiment] Running {exp_name}. Output: {out}')
    result = EXPERIMENTS[exp_name](args, social_root, out)
    manifest = {
        'exp_name': exp_name,
        'exp_id': args.exp_id,
        'social_output_root': str(social_root),
        'result': result,
        'args': vars(args),
    }
    save_json(manifest, out / 'experiment_manifest.json')
    print('[social_experiment] Finished:', json.dumps(result, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description='Run exactly one H&M social-science experiment after social inference.')
    p.add_argument('--social_output_root', default='./social_output/k')
    p.add_argument('--data_root', default='./data')
    p.add_argument('--exp_id', default='01', help='00 panel_check, 01 prototype_covid, 02 style_distribution, 03 exposure_response, 04 lead_lag, 05 age_style, 06 channel_style, 07 member_news, 08 seasonality, 09 ml_heterogeneity')
    p.add_argument('--exp_name', default='', choices=[''] + sorted(EXPERIMENTS.keys()))
    p.add_argument('--event_month', default='2020-03')
    p.add_argument('--style_dims', default='formal,office,comfort,homewear,casual,value')
    p.add_argument('--covid_vars', default='covid_cases_index,covid_deaths_index,covid_stringency_index,covid_reproduction_rate')
    p.add_argument('--fourier_order', type=int, default=2)
    p.add_argument('--max_lag', type=int, default=4)
    p.add_argument('--max_lead', type=int, default=2)
    p.add_argument('--ar_lags', type=int, default=2)
    p.add_argument('--min_periods', type=int, default=6)
    p.add_argument('--min_rows_panel', type=int, default=500)
    p.add_argument('--max_panel_rows', type=int, default=500000, help='Cap dense item-month rows for expensive item-level exposure experiments; 0 disables sampling.')
    p.add_argument('--event_window', type=int, default=8)
    p.add_argument('--outlier_bound', type=float, default=2.5)
    p.add_argument('--n_clusters', type=int, default=4)
    p.add_argument('--hist_bins', type=int, default=40)
    p.add_argument('--max_distribution_samples', type=int, default=50000)
    p.add_argument('--use_copula_scores', type=parse_bool, default=False)
    p.add_argument('--high_quantile', type=float, default=0.75)
    p.add_argument('--low_quantile', type=float, default=0.25)
    p.add_argument('--gmm_components', type=int, default=3)
    p.add_argument('--rf_trees', type=int, default=200)
    p.add_argument('--rf_max_depth', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
