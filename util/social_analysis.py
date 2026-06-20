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




TASK1_RAW_AXIS_DIMS = [
    'formality_structure',
    'public_private_occasion',
    'comfort_ease',
    'practical_value',
    'trend_expressiveness',
    'design_complexity',
    'fit_looseness',
]

TASK1_COMPOSITE_DIMS = [
    'office_like',
    'homewear_like',
    'casual_like',
    'social_outing_like',
    'value_basic',
]

TASK1_HOME_PRACTICAL_DIMS = [
    'comfort_ease',
    'fit_looseness',
    'practical_value',
    'homewear_like',
    'value_basic',
]

TASK1_PUBLIC_FORMAL_DIMS = [
    'formality_structure',
    'public_private_occasion',
    'trend_expressiveness',
    'design_complexity',
    'office_like',
    'social_outing_like',
]

def _safe_name(x: object) -> str:
    return ''.join(ch if ch.isalnum() else '_' for ch in str(x)).strip('_')[:120] or 'x'


def _short_prototype_label(dim: object, proto: object, max_len: int = 74) -> str:
    """Compact labels for prototype plots.

    Routed prompt labels are often full prompt sentences such as
    "Sweater reflecting demand for everyday comfort and practical clothing".
    Long labels make coefficient plots unreadable, so the plotting path uses a
    shortened display label while all parquet outputs keep the full prototype.
    """
    d = str(dim)
    s = ' '.join(str(proto).replace('\n', ' ').split())
    replacements = [
        (' reflecting demand for ', ': '),
        (' indicating ', ': '),
        (' suitable for ', ': '),
        (' valued for ', ': '),
        (' used for ', ': '),
        (' with ', ': '),
        (' and ', ' & '),
    ]
    for a, b in replacements:
        s = s.replace(a, b)
    label = f'{d} | {s}'
    if len(label) > max_len:
        label = label[:max_len - 1].rstrip() + '…'
    return label


def _zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors='coerce').astype(float)
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return x * 0.0
    return (x - x.mean()) / sd


def _style_cols(df: pd.DataFrame, dims: str | Iterable[str] | None = None) -> list[str]:
    cols = [c for c in df.columns if c.endswith('_score') and not c.startswith('copula_calibrated_') and not c.endswith('_pos_score') and not c.endswith('_neg_score')]
    if dims:
        dset = {str(d).strip().replace('_score','') for d in (str(dims).split(',') if isinstance(dims, str) else dims) if str(d).strip()}
        cols = [c for c in cols if c.replace('_score','') in dset or c in dset]
    return cols


def _covid_vars(df: pd.DataFrame, preferred: str | None = None) -> list[str]:
    base = [
        'covid_cases_index','covid_deaths_index','covid_stringency_index','covid_reproduction_rate',
        'covid_cases_z','covid_deaths_z','covid_stringency_z','covid_reproduction_z'
    ]
    cols = [c for c in base if c in df.columns]
    if preferred and preferred in cols:
        return [preferred] + [c for c in cols if c != preferred]
    return cols


def _ensure_month_column(df: pd.DataFrame, table_name: str = 'panel') -> pd.DataFrame:
    """Return a copy with a string `month` column.

    Older or empty parquet files may not contain `month` even though the newer
    social experiments expect monthly panels.  We try a few safe conversions and
    otherwise raise a diagnostic error with the available columns.
    """
    out = df.copy()
    if 'month' in out.columns:
        out['month'] = out['month'].astype(str)
        return out
    for c in ['period', 'month_id', 'transaction_month']:
        if c in out.columns:
            out['month'] = out[c].astype(str)
            return out
    for c in ['t_dat', 'date', 'transaction_date']:
        if c in out.columns:
            out['month'] = pd.to_datetime(out[c], errors='coerce').dt.to_period('M').astype(str)
            return out
    raise KeyError(
        f"{table_name} does not contain a usable month column. "
        f"Available columns: {list(out.columns)}. "
        f"Please rerun scripts/run_social_inference.sh with the patched code, "
        f"or delete the stale {table_name} parquet under social_output_root."
    )


def _add_time_controls_month(df: pd.DataFrame, table_name: str = 'panel') -> pd.DataFrame:
    out = _ensure_month_column(df, table_name=table_name)
    p = pd.PeriodIndex(out['month'].astype(str), freq='M')
    out['_trend'] = (p.year * 12 + p.month).astype(float)
    out['_trend'] -= out['_trend'].min()
    out['_calendar_month'] = p.month.astype(str)
    return out


def _rebuild_prototype_monthly_panel(root: Path, args) -> pd.DataFrame:
    """Rebuild prototype-month panel from item_monthly_panel when needed.

    This handles stale/empty `prototype_monthly_panel.parquet` files produced by
    older inference runs.  If no *_proto columns exist, the default behavior is to keep an empty
    prototype panel.  Score-quantile pseudo-prototypes are generated only when
    --prototype_fallback_from_scores 1 is explicitly provided.
    """
    item_path = root / 'item_monthly_panel.parquet'
    if not item_path.exists():
        raise FileNotFoundError(
            f'Missing {item_path}; cannot rebuild prototype_monthly_panel. Run social inference first.'
        )
    im = pd.read_parquet(item_path)
    im = _ensure_month_column(im, table_name='item_monthly_panel')
    if 'sales_count' not in im.columns:
        raise KeyError(f'item_monthly_panel lacks sales_count. Available columns: {list(im.columns)}')
    proto_cols = [c for c in im.columns if c.endswith('_proto')]
    score_cols = _style_cols(im, args.style_dims)
    total_by_month = im.groupby('month', as_index=False)['sales_count'].sum().rename(columns={'sales_count':'total_sales'})
    rows = []
    for pc in proto_cols:
        dim = pc[:-6]
        tmp = im.dropna(subset=[pc]).copy()
        if tmp.empty:
            continue
        g = tmp.groupby([pc, 'month'], as_index=False)['sales_count'].sum().rename(columns={pc: 'prototype'})
        g['dimension'] = dim
        rows.append(g[['dimension', 'prototype', 'month', 'sales_count']])
    # Fallback: if no routed prompt prototypes are available, create high/low
    # score pseudo-prototypes.  This is marked with dimension style_quantile_*
    # so it is never mistaken for raw prompt prototypes.
    if not rows and bool(getattr(args, 'prototype_fallback_from_scores', False)) and score_cols:
        for sc in score_cols:
            dim = sc.replace('_score', '')
            vals = pd.to_numeric(im[sc], errors='coerce')
            hi = vals.quantile(float(getattr(args, 'prototype_high_quantile', 0.75)))
            lo = vals.quantile(float(getattr(args, 'prototype_low_quantile', 0.25)))
            tmp = im.copy()
            tmp['_pseudo_proto'] = pd.Series(pd.NA, index=tmp.index, dtype='object')
            tmp.loc[vals >= hi, '_pseudo_proto'] = f'high_{dim}'
            tmp.loc[vals <= lo, '_pseudo_proto'] = f'low_{dim}'
            tmp = tmp.dropna(subset=['_pseudo_proto'])
            if tmp.empty:
                continue
            g = tmp.groupby(['_pseudo_proto', 'month'], as_index=False)['sales_count'].sum().rename(columns={'_pseudo_proto': 'prototype'})
            g['dimension'] = f'style_quantile_{dim}'
            rows.append(g[['dimension', 'prototype', 'month', 'sales_count']])
    if not rows:
        return pd.DataFrame(columns=['dimension','prototype','month','sales_count','total_sales','sales_share'])
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.merge(total_by_month, on='month', how='left')
    panel['sales_share'] = panel['sales_count'] / panel['total_sales'].replace(0, np.nan)
    # Carry monthly COVID columns from item panel when available.
    covid_cols = [c for c in im.columns if c.startswith('covid_') or c.startswith('global_mobility_')]
    if covid_cols:
        cov = im[['month'] + covid_cols].drop_duplicates('month')
        panel = panel.merge(cov, on='month', how='left')
    return panel


def _load_prototype_monthly_panel(root: Path, args) -> pd.DataFrame:
    panel_path = root / 'prototype_monthly_panel.parquet'
    panel = pd.DataFrame()
    if panel_path.exists():
        try:
            panel = pd.read_parquet(panel_path)
        except Exception as exc:
            print(f'[WARN] Failed to read {panel_path}: {type(exc).__name__}: {exc}; rebuilding from item_monthly_panel.')
    if panel.empty or 'month' not in panel.columns or not {'dimension','prototype','sales_count','sales_share'}.issubset(panel.columns):
        print('[WARN] prototype_monthly_panel is empty/stale or lacks required columns; rebuilding from item_monthly_panel.')
        panel = _rebuild_prototype_monthly_panel(root, args)
        try:
            panel.to_parquet(panel_path, index=False)
            print(f'[social_experiments] Wrote rebuilt prototype panel to {panel_path}')
        except Exception as exc:
            print(f'[WARN] Could not overwrite rebuilt prototype panel: {type(exc).__name__}: {exc}')
    return _ensure_month_column(panel, table_name='prototype_monthly_panel')


def _rel_month(month: pd.Series, event_month: str) -> np.ndarray:
    p = pd.PeriodIndex(month.astype(str), freq='M')
    ev = pd.Period(event_month, freq='M')
    return ((p.year - ev.year) * 12 + (p.month - ev.month)).astype(int).to_numpy()


def _student_t_pvalue(t_stat: float, dof: int) -> float:
    """Two-sided p-value using Student-t when scipy is available.

    The first prototype-response patch used a normal approximation.  With only
    about 20-30 monthly observations per prototype, the normal approximation can
    make almost every coefficient look like p≈0 and collapse the volcano plot.
    """
    if not np.isfinite(t_stat):
        return np.nan
    dof = max(int(dof), 1)
    try:
        from scipy import stats
        return float(2.0 * stats.t.sf(abs(float(t_stat)), df=dof))
    except Exception:
        return float(math.erfc(abs(float(t_stat)) / math.sqrt(2.0)))


def _newey_west_covariance(X: np.ndarray, resid: np.ndarray, lag: int = 2) -> np.ndarray:
    """Newey-West HAC covariance for time-ordered OLS design X.

    This is intentionally lightweight and dependency-free.  It is used for the
    prototype response time-series regressions where residual autocorrelation is
    likely and conventional OLS standard errors are too optimistic.
    """
    X = np.asarray(X, dtype=float)
    resid = np.asarray(resid, dtype=float).reshape(-1)
    n, k = X.shape
    lag = int(max(0, min(lag, max(n - 1, 0))))
    xtx_inv = np.linalg.pinv(X.T @ X)
    xe = X * resid[:, None]
    S = xe.T @ xe
    for ell in range(1, lag + 1):
        weight = 1.0 - ell / (lag + 1.0)
        Gamma = xe[ell:].T @ xe[:-ell]
        S += weight * (Gamma + Gamma.T)
    # small-sample correction similar to HC1
    if n > k:
        S *= n / max(n - k, 1)
    cov = xtx_inv @ S @ xtx_inv
    return cov


def _ols(
    df: pd.DataFrame,
    y: str,
    xcols: list[str],
    fe_cols: list[str] | None = None,
    weight_col: str | None = None,
    min_n: int = 8,
    cov_type: str = 'classic',
    hac_lags: int = 2,
) -> dict:
    cols = [y] + xcols + (fe_cols or []) + ([weight_col] if weight_col else [])
    # Preserve time order when _trend is available; this is needed for HAC.
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if '_trend' in d.columns:
        d = d.sort_values('_trend')
    if len(d) < max(min_n, len(xcols) + 2):
        return {'status': 'too_few_rows', 'nobs': int(len(d))}
    # Skip degenerate outcomes.  Otherwise tiny numerical residuals produce
    # meaningless huge t-statistics.
    yy_raw = pd.to_numeric(d[y], errors='coerce').astype(float)
    if yy_raw.nunique(dropna=True) <= 1 or float(yy_raw.std(ddof=0)) <= 1e-12:
        return {'status': 'degenerate_outcome', 'nobs': int(len(d))}
    X_parts = [pd.Series(1.0, index=d.index, name='const')]
    for c in xcols:
        xc = pd.to_numeric(d[c], errors='coerce').astype(float).rename(c)
        if xc.nunique(dropna=True) <= 1:
            return {'status': f'degenerate_regressor:{c}', 'nobs': int(len(d))}
        X_parts.append(xc)
    for fc in fe_cols or []:
        dm = pd.get_dummies(d[fc].astype(str), prefix=fc, drop_first=True, dtype=float)
        if dm.shape[1] > 0:
            X_parts.append(dm)
    X = pd.concat(X_parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    yy = yy_raw.to_numpy(dtype=float)
    Xn = X.to_numpy(dtype=float)
    if weight_col:
        w = pd.to_numeric(d[weight_col], errors='coerce').fillna(1.0).clip(lower=1e-8).to_numpy(dtype=float)
        sw = np.sqrt(w)
        Xfit = Xn * sw[:, None]
        yfit = yy * sw
    else:
        w = None
        Xfit, yfit = Xn, yy
    try:
        beta, *_ = np.linalg.lstsq(Xfit, yfit, rcond=None)
        resid = yy - Xn @ beta
        n, k = Xn.shape
        dof = max(n - k, 1)
        if cov_type == 'hac' and weight_col is None:
            cov = _newey_west_covariance(Xn, resid, lag=hac_lags)
        elif weight_col:
            sigma2 = float(((resid ** 2) * w).sum() / dof)
            cov = np.linalg.pinv(Xfit.T @ Xfit) * sigma2
        else:
            sigma2 = float((resid @ resid) / dof)
            cov = np.linalg.pinv(Xn.T @ Xn) * sigma2
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        sst = float(((yy - yy.mean()) @ (yy - yy.mean())))
        r2 = float(1 - (resid @ resid) / max(sst, 1e-12))
        ans = {'status': 'ok', 'nobs': int(n), 'dof': int(dof), 'r2': r2, 'cov_type': cov_type}
        for c, b, se_i in zip(X.columns, beta, se):
            if c in xcols or c == 'const':
                t_stat = float(b / se_i) if se_i > 0 else np.nan
                ans[f'coef_{c}'] = float(b)
                ans[f'se_{c}'] = float(se_i)
                ans[f't_{c}'] = t_stat
                ans[f'p_{c}'] = _student_t_pvalue(t_stat, dof) if se_i > 0 else np.nan
        return ans
    except Exception as exc:
        return {'status': f'error:{type(exc).__name__}:{exc}', 'nobs': int(len(d))}


def _weighted_wasserstein_1d(x: np.ndarray, y: np.ndarray, wx: np.ndarray | None = None, wy: np.ndarray | None = None) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if wx is None: wx = np.ones_like(x)
    if wy is None: wy = np.ones_like(y)
    okx = np.isfinite(x) & np.isfinite(wx) & (wx > 0)
    oky = np.isfinite(y) & np.isfinite(wy) & (wy > 0)
    x, wx = x[okx], wx[okx]
    y, wy = y[oky], wy[oky]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    sx = np.argsort(x); sy = np.argsort(y)
    x, wx = x[sx], wx[sx] / wx.sum()
    y, wy = y[sy], wy[sy] / wy.sum()
    vals = np.sort(np.unique(np.concatenate([x, y])))
    if len(vals) <= 1:
        return 0.0
    Fx = np.searchsorted(x, vals[:-1], side='right')
    Fy = np.searchsorted(y, vals[:-1], side='right')
    cwx = np.concatenate([[0.0], np.cumsum(wx)])
    cwy = np.concatenate([[0.0], np.cumsum(wy)])
    dx = np.diff(vals)
    return float(np.sum(np.abs(cwx[Fx] - cwy[Fy]) * dx))


def _weighted_hist_prob(x: np.ndarray, w: np.ndarray, bins: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return np.zeros(len(bins) - 1)
    h, _ = np.histogram(x[ok], bins=bins, weights=w[ok])
    h = h.astype(float)
    if h.sum() > 0:
        h /= h.sum()
    return h


def _jsd_from_probs(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def _weighted_moments(x: np.ndarray, w: np.ndarray | None = None) -> dict:
    x = np.asarray(x, dtype=float)
    if w is None:
        w = np.ones_like(x)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    if len(x) == 0:
        return {'mean': np.nan, 'variance': np.nan, 'skewness': np.nan, 'kurtosis': np.nan}
    w = w / w.sum()
    mu = float(np.sum(w * x))
    xc = x - mu
    var = float(np.sum(w * xc ** 2))
    if var <= 1e-12:
        return {'mean': mu, 'variance': var, 'skewness': 0.0, 'kurtosis': 0.0}
    skew = float(np.sum(w * xc ** 3) / (var ** 1.5))
    kurt = float(np.sum(w * xc ** 4) / (var ** 2))
    return {'mean': mu, 'variance': var, 'skewness': skew, 'kurtosis': kurt}


def _safe_ratio(num: float, den: float, eps: float = 1e-12) -> float:
    num = float(num) if np.isfinite(num) else np.nan
    den = float(den) if np.isfinite(den) else np.nan
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= eps:
        return np.nan
    return float(num / den)


def _weighted_scale_stats(x: np.ndarray, w: np.ndarray | None = None) -> dict:
    """Weighted dispersion statistics for comparable effect-size reporting.

    Axis scores lie in [0, 1]. A raw YoY shift of 0.01--0.03 may look small,
    so we also report it relative to baseline SD and IQR.
    """
    x = np.asarray(x, dtype=float)
    if w is None:
        w = np.ones_like(x)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    if len(x) == 0:
        return {'sd': np.nan, 'iqr': np.nan, 'q25': np.nan, 'q75': np.nan}
    moments = _weighted_moments(x, w)
    var = float(moments.get('variance', np.nan))
    sd = math.sqrt(max(var, 0.0)) if np.isfinite(var) else np.nan
    q25, q75 = _weighted_quantile(x, w, [0.25, 0.75])
    iqr = float(q75 - q25) if np.isfinite(q25) and np.isfinite(q75) else np.nan
    return {'sd': float(sd), 'iqr': iqr, 'q25': float(q25), 'q75': float(q75)}


def _plot_heatmap(mat: pd.DataFrame, path: Path, title: str = '', figsize=(8, 6)) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat.to_numpy(dtype=float), aspect='auto')
    ax.set_xticks(np.arange(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=45, ha='right')
    ax.set_yticks(np.arange(mat.shape[0])); ax.set_yticklabels(mat.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_lines(df: pd.DataFrame, x: str, y: str, group: str, path: Path, title: str = '') -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, g in df.groupby(group):
        g = g.sort_values(x)
        ax.plot(g[x].astype(str), g[y], marker='o', label=str(key))
    ax.set_title(title)
    ax.set_xlabel(x); ax.set_ylabel(y)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _sample_weighted(df: pd.DataFrame, weight_col: str, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    w = pd.to_numeric(df[weight_col], errors='coerce').fillna(0.0).clip(lower=0)
    if w.sum() <= 0:
        return df.sample(n=n, random_state=seed).copy()
    return df.sample(n=n, weights=w, random_state=seed, replace=False).copy()



def _augment_prototype_response_results(results: pd.DataFrame, covid_vars: list[str], p_clip_min: float = 1e-16) -> pd.DataFrame:
    """Add validity, p-plot and 95% CI columns for prototype regressions."""
    out = results.copy()
    if out.empty:
        return out
    status_ok = out.get('status', pd.Series('', index=out.index)).astype(str).eq('ok')
    p_clip_min = float(p_clip_min)
    if not np.isfinite(p_clip_min) or p_clip_min <= 0:
        p_clip_min = 1e-16
    for cv in covid_vars:
        ccol = f'coef_{cv}'
        scol = f'se_{cv}'
        pcol = f'p_{cv}'
        if ccol not in out.columns:
            continue
        coef = pd.to_numeric(out[ccol], errors='coerce')
        se = pd.to_numeric(out[scol], errors='coerce') if scol in out.columns else pd.Series(np.nan, index=out.index)
        pval = pd.to_numeric(out[pcol], errors='coerce') if pcol in out.columns else pd.Series(np.nan, index=out.index)
        # p may underflow to exactly 0 for very large |t|. That is still a valid
        # regression result for ranking, but it must be clipped only for plotting.
        valid = status_ok & np.isfinite(coef) & np.isfinite(se) & (se > 0) & np.isfinite(pval) & (pval >= 0) & (pval <= 1)
        out[f'valid_{cv}'] = valid.astype(bool)
        clipped = pval.clip(lower=p_clip_min, upper=1.0)
        out[f'p_plot_{cv}'] = np.where(valid, clipped, np.nan)
        out[f'neglog10p_{cv}'] = np.where(valid, -np.log10(clipped), np.nan)
        out[f'ci95_low_{cv}'] = np.where(valid, coef - 1.96 * se, np.nan)
        out[f'ci95_high_{cv}'] = np.where(valid, coef + 1.96 * se, np.nan)
    return out


def _plot_top_prototype_responses(results: pd.DataFrame, cv: str, out_path: Path, top_n: int = 30) -> None:
    """Plot positive and negative prototype responses separately with 95% CI."""
    import matplotlib.pyplot as plt
    ccol, scol, valid_col = f'coef_{cv}', f'se_{cv}', f'valid_{cv}'
    if ccol not in results.columns or scol not in results.columns:
        return
    valid = results.get(valid_col, pd.Series(True, index=results.index)).astype(bool)
    d = results.loc[valid].copy()
    d[ccol] = pd.to_numeric(d[ccol], errors='coerce')
    d[scol] = pd.to_numeric(d[scol], errors='coerce')
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[ccol, scol])
    if d.empty:
        return
    n_each = max(5, int(top_n) // 2)
    pos = d[d[ccol] > 0].sort_values(ccol, ascending=False).head(n_each)
    neg = d[d[ccol] < 0].sort_values(ccol, ascending=True).head(n_each)
    panels = [('Most negative responses', neg), ('Most positive responses', pos)]
    max_rows = max([len(x) for _, x in panels] + [1])
    fig, axes = plt.subplots(1, 2, figsize=(16, max(4.5, 0.42 * max_rows)), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, (title, sub) in zip(axes, panels):
        if sub.empty:
            ax.text(0.5, 0.5, 'No valid coefficients', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue
        labels = [_short_prototype_label(r.dimension, r.prototype) for r in sub.itertuples(index=False)]
        y = np.arange(len(sub))
        x = sub[ccol].to_numpy(dtype=float)
        xerr = 1.96 * sub[scol].to_numpy(dtype=float)
        ax.barh(y, x, xerr=xerr, capsize=2)
        ax.axvline(0, linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(f'coef({cv}) with 95% CI')
    fig.suptitle(f'Prototype responses to {cv}', y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _plot_prototype_volcano(results: pd.DataFrame, cv: str, out_path: Path, label_n: int = 12) -> None:
    """Draw a cleaned volcano plot using finite p-values and a sane plotting floor."""
    import matplotlib.pyplot as plt
    ccol, ycol, valid_col = f'coef_{cv}', f'neglog10p_{cv}', f'valid_{cv}'
    if ccol not in results.columns:
        return
    valid = results.get(valid_col, pd.Series(True, index=results.index)).astype(bool)
    d = results.loc[valid].copy()
    d[ccol] = pd.to_numeric(d[ccol], errors='coerce')
    if ycol in d.columns:
        d[ycol] = pd.to_numeric(d[ycol], errors='coerce')
    else:
        pcol = f'p_{cv}'
        d[ycol] = -np.log10(pd.to_numeric(d[pcol], errors='coerce').clip(lower=1e-16, upper=1.0)) if pcol in d.columns else np.nan
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[ccol, ycol])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(d[ccol], d[ycol], s=20, alpha=0.75)
    ax.axvline(0, linewidth=1)
    ax.axhline(-math.log10(0.05), linestyle='--', linewidth=1)
    abscoef = d[ccol].abs()
    if len(abscoef) >= 5:
        eth = float(abscoef.quantile(0.90))
        if np.isfinite(eth) and eth > 0:
            ax.axvline(eth, linestyle=':', linewidth=1)
            ax.axvline(-eth, linestyle=':', linewidth=1)
    # Label only a few strong, non-overwhelming points.  If many p-values are
    # still clipped at the plot ceiling, labels at that ceiling are suppressed
    # because they become unreadable.
    label_n = max(0, int(label_n))
    if label_n > 0:
        dd = d.copy()
        ymax = float(dd[ycol].max()) if len(dd) else np.nan
        if np.isfinite(ymax):
            dd = dd[dd[ycol] < ymax - 1e-9] if (dd[ycol] >= ymax - 1e-9).mean() > 0.25 else dd
        dd['_rank_score'] = dd[ccol].abs() * (1.0 + dd[ycol].fillna(0.0))
        lab = dd.sort_values('_rank_score', ascending=False).head(label_n)
        for r in lab.itertuples(index=False):
            ax.annotate(
                _short_prototype_label(getattr(r, 'dimension'), getattr(r, 'prototype'), max_len=42),
                (getattr(r, ccol), getattr(r, ycol)),
                xytext=(4, 3), textcoords='offset points', fontsize=6, alpha=0.85,
            )
    ax.set_xlabel(f'coef({cv})')
    ax.set_ylabel('-log10(p), clipped for plotting')
    ax.set_title('Prototype response volcano')
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _plot_prototype_dimension_summary(results: pd.DataFrame, cv: str, out_path: Path) -> None:
    """Boxplot coefficient distributions by prototype dimension."""
    import matplotlib.pyplot as plt
    ccol, valid_col = f'coef_{cv}', f'valid_{cv}'
    if ccol not in results.columns:
        return
    valid = results.get(valid_col, pd.Series(True, index=results.index)).astype(bool)
    d = results.loc[valid, ['dimension', ccol]].copy()
    d[ccol] = pd.to_numeric(d[ccol], errors='coerce')
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=['dimension', ccol])
    if d.empty:
        return
    order = d.groupby('dimension')[ccol].median().sort_values().index.tolist()
    data = [d.loc[d['dimension'] == dim, ccol].to_numpy(dtype=float) for dim in order]
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.35 * len(order))))
    ax.boxplot(data, vert=False, labels=order, showfliers=True, flierprops={'markersize': 3, 'alpha': 0.6})
    ax.axvline(0, linewidth=1)
    
    # Coefficients have a very concentrated center with a few meaningful
    # outliers.  Symlog keeps both the central boxes and the extreme points visible.
    try:
        ax.set_xscale('symlog', linthresh=1e-4)
    except Exception:
        pass
    ax.set_xlabel(f'coef({cv})')
    ax.set_title(f'Prototype response distribution by dimension: {cv}')
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _weighted_mean_safe(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return np.nan
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok]))


def _weighted_quantile(x: np.ndarray, w: np.ndarray, qs: list[float] | np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    qs = np.asarray(qs, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return np.full_like(qs, np.nan, dtype=float)
    x = x[ok]
    w = w[ok]
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    return np.interp(qs, cw, x)


def _weighted_ks_1d(x: np.ndarray, y: np.ndarray, wx: np.ndarray, wy: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    wx = np.asarray(wx, dtype=float); wy = np.asarray(wy, dtype=float)
    okx = np.isfinite(x) & np.isfinite(wx) & (wx > 0)
    oky = np.isfinite(y) & np.isfinite(wy) & (wy > 0)
    x, wx = x[okx], wx[okx]
    y, wy = y[oky], wy[oky]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    sx = np.argsort(x); sy = np.argsort(y)
    x, wx = x[sx], wx[sx] / wx.sum()
    y, wy = y[sy], wy[sy] / wy.sum()
    vals = np.sort(np.unique(np.concatenate([x, y])))
    cwx = np.concatenate([[0.0], np.cumsum(wx)])
    cwy = np.concatenate([[0.0], np.cumsum(wy)])
    ix = np.searchsorted(x, vals, side='right')
    iy = np.searchsorted(y, vals, side='right')
    return float(np.max(np.abs(cwx[ix] - cwy[iy])))


def _parse_dim_list(x: str | Iterable[str] | None) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        raw = x.split(',')
    else:
        raw = list(x)
    return [str(v).strip().replace('_score', '') for v in raw if str(v).strip()]


def _resolve_score_cols(df: pd.DataFrame, dims: str | Iterable[str] | None, exclude_dims: str | Iterable[str] | None = None) -> list[str]:
    all_cols = _style_cols(df, None)
    wanted = _parse_dim_list(dims)
    excluded = set(_parse_dim_list(exclude_dims))
    if wanted:
        cols = [f'{d}_score' if not str(d).endswith('_score') else str(d) for d in wanted]
        cols = [c for c in cols if c in df.columns]
    else:
        cols = all_cols
    return [c for c in cols if c.replace('_score', '') not in excluded]


def _dim_csv(dims: Iterable[str]) -> str:
    return ','.join(str(d).strip().replace('_score', '') for d in dims if str(d).strip())


def _cols_for_dims(df: pd.DataFrame, dims: Iterable[str]) -> list[str]:
    return _resolve_score_cols(df, _dim_csv(dims), exclude_dims=None)


def _axis_ids(cols: Iterable[str]) -> list[str]:
    return [str(c).replace('_score', '') for c in cols]


def _load_item_monthly_with_scores(root: Path) -> pd.DataFrame:
    path = root / 'item_monthly_panel.parquet'
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}; run social inference first.')
    im = pd.read_parquet(path)
    im = _ensure_month_column(im, table_name='item_monthly_panel')
    if 'sales_count' not in im.columns:
        raise KeyError(f'item_monthly_panel lacks sales_count. Available columns: {list(im.columns)}')
    # Some stale panels may not include semantic scores.  Merge item_semantic_scores
    # when available, preserving existing columns.
    if not _style_cols(im, None):
        sem_path = root / 'item_semantic_scores.parquet'
        if sem_path.exists() and 'article_id' in im.columns:
            sem = pd.read_parquet(sem_path)
            score_cols = [c for c in _style_cols(sem, None) if c not in im.columns]
            if score_cols:
                im = im.merge(sem[['article_id'] + score_cols].drop_duplicates('article_id'), on='article_id', how='left')
    im['_weight'] = pd.to_numeric(im['sales_count'], errors='coerce').fillna(0.0).clip(lower=0)
    im = im[im['_weight'] > 0].copy()
    im['_period'] = pd.PeriodIndex(im['month'].astype(str), freq='M')
    return im


def _axis_monthly_wide(im: pd.DataFrame, score_cols: list[str], weight_col: str = '_weight') -> pd.DataFrame:
    rows = []
    for month, g in im.groupby('month'):
        w = pd.to_numeric(g[weight_col], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        row = {'month': str(month), 'total_sales': float(w.sum())}
        for sc in score_cols:
            row[sc] = _weighted_mean_safe(pd.to_numeric(g[sc], errors='coerce').to_numpy(dtype=float), w)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values('month').reset_index(drop=True)
    if len(out):
        out['_period'] = pd.PeriodIndex(out['month'].astype(str), freq='M')
    return out


def _task1_period_pairs(months: Iterable, event_month: str, covid_end_month: str | None = None) -> tuple[list[pd.Period], list[pd.Period]]:
    periods = sorted(pd.PeriodIndex([str(m) for m in months], freq='M').unique())
    available = set(periods)
    ev = pd.Period(event_month, freq='M')
    if covid_end_month and str(covid_end_month).strip():
        end = pd.Period(str(covid_end_month).strip(), freq='M')
    else:
        end = max(periods)
    covid = [p for p in periods if ev <= p <= end and (p - 12) in available]
    base = [p - 12 for p in covid]
    return covid, base


def _paired_weighted_shift(delta: np.ndarray, weights: np.ndarray) -> float:
    return _weighted_mean_safe(delta, weights)


def _bootstrap_month_ci(delta: np.ndarray, weights: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    delta = np.asarray(delta, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(delta) & np.isfinite(weights) & (weights > 0)
    delta, weights = delta[ok], weights[ok]
    if len(delta) <= 1 or int(n_boot) <= 1:
        val = _paired_weighted_shift(delta, weights)
        return val, val
    rng = np.random.default_rng(int(seed))
    vals = []
    idx = np.arange(len(delta))
    for _ in range(int(n_boot)):
        b = rng.choice(idx, size=len(idx), replace=True)
        vals.append(_paired_weighted_shift(delta[b], weights[b]))
    lo, hi = np.nanquantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _calendar_matched_shift(im: pd.DataFrame, sc: str, covid_periods: list[pd.Period], base_periods: list[pd.Period]) -> float:
    cov = im[im['_period'].isin(covid_periods)]
    bas = im[im['_period'].isin(base_periods)]
    mc = _weighted_mean_safe(pd.to_numeric(cov[sc], errors='coerce').to_numpy(float), cov['_weight'].to_numpy(float))
    mb = _weighted_mean_safe(pd.to_numeric(bas[sc], errors='coerce').to_numpy(float), bas['_weight'].to_numpy(float))
    return float(mc - mb) if np.isfinite(mc) and np.isfinite(mb) else np.nan


def _calendar_matched_bootstrap_ci(im: pd.DataFrame, sc: str, covid_periods: list[pd.Period], base_periods: list[pd.Period], n_boot: int, seed: int) -> tuple[float, float]:
    if len(covid_periods) <= 1 or int(n_boot) <= 1:
        val = _calendar_matched_shift(im, sc, covid_periods, base_periods)
        return val, val
    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(covid_periods))
    vals = []
    for _ in range(int(n_boot)):
        b = rng.choice(idx, size=len(idx), replace=True)
        cp = [covid_periods[j] for j in b]
        bp = [base_periods[j] for j in b]
        vals.append(_calendar_matched_shift(im, sc, cp, bp))
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def _build_residualized_scores(im: pd.DataFrame, target_cols: list[str], control_cols: list[str], args) -> pd.DataFrame:
    """Residualize main axis scores against season-sensitive axes and light category controls.

    This is a robustness diagnostic.  It is deliberately post-hoc and does not
    alter the original axis scores.  The residualized columns are named
    `<axis>_resid` and are used only for seasonality robustness plots/tables.
    """
    control_cols = [c for c in control_cols if c in im.columns and c not in target_cols]
    if not control_cols:
        return pd.DataFrame()
    key_col = 'article_id' if 'article_id' in im.columns else None
    base_cols = ([key_col] if key_col else []) + target_cols + control_cols
    cat_candidates = [
        'product_group_name', 'product_type_name', 'garment_group_name', 'section_name',
        'index_group_name', 'index_name', 'department_name'
    ]
    cat_cols = [c for c in cat_candidates if c in im.columns]
    base_cols += cat_cols
    if key_col:
        # Scores are item-level constants; aggregate to one row per article.
        agg = {c: 'mean' for c in target_cols + control_cols}
        for c in cat_cols:
            agg[c] = lambda s: s.dropna().astype(str).iloc[0] if len(s.dropna()) else 'NA'
        item = im[base_cols].groupby(key_col, as_index=False).agg(agg)
    else:
        item = im[base_cols].copy()
        item['_row_id_for_resid'] = np.arange(len(item))
        key_col = '_row_id_for_resid'
        im = im.copy()
        im[key_col] = item[key_col]
    X_parts = [pd.Series(1.0, index=item.index, name='const')]
    for c in control_cols:
        X_parts.append(pd.to_numeric(item[c], errors='coerce').astype(float).rename(c))
    max_cat_levels = int(getattr(args, 'task1_residual_max_cat_levels', 60))
    if bool(getattr(args, 'task1_category_residualize', True)):
        for c in cat_cols:
            s = item[c].astype(str).fillna('NA')
            top = set(s.value_counts().head(max_cat_levels).index)
            ss = s.where(s.isin(top), other='OTHER')
            dm = pd.get_dummies(ss, prefix=c, drop_first=True, dtype=float)
            if dm.shape[1] > 0:
                X_parts.append(dm)
    X = pd.concat(X_parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    out = item[[key_col]].copy()
    for sc in target_cols:
        y = pd.to_numeric(item[sc], errors='coerce').to_numpy(dtype=float)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        resid = np.full(len(item), np.nan, dtype=float)
        if ok.sum() >= X.shape[1] + 2:
            beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
            resid[ok] = y[ok] - X[ok] @ beta
        out[f'{sc.replace("_score", "")}_resid'] = resid
    if 'article_id' in out.columns and 'article_id' in im.columns:
        return out.drop_duplicates('article_id')
    return out


def _make_axis_long_timeseries(mon: pd.DataFrame, score_cols: list[str], covid_periods: list[pd.Period]) -> pd.DataFrame:
    if mon.empty:
        return pd.DataFrame()
    lookup = mon.set_index('_period')
    rows = []
    for _, row in mon.iterrows():
        p = row['_period']
        for sc in score_cols:
            base_p = p - 12
            base_val = lookup.loc[base_p, sc] if base_p in lookup.index else np.nan
            val = row[sc]
            rows.append({
                'month': str(row['month']),
                'axis_id': sc.replace('_score', ''),
                'axis_share': float(val) if np.isfinite(val) else np.nan,
                'total_sales': float(row['total_sales']),
                'yoy_baseline_month': str(base_p),
                'yoy_baseline_share': float(base_val) if np.isfinite(base_val) else np.nan,
                'yoy_delta': float(val - base_val) if np.isfinite(val) and np.isfinite(base_val) else np.nan,
                'is_covid_window': bool(p in set(covid_periods)),
            })
    return pd.DataFrame(rows)


def _plot_axis_yoy_forest(summary: pd.DataFrame, out_path: Path, value_col: str = 'raw_yoy_shift') -> None:
    import matplotlib.pyplot as plt
    if summary.empty or value_col not in summary.columns:
        return
    lo_col = 'raw_ci_low' if value_col == 'raw_yoy_shift' else f'{value_col}_ci_low'
    hi_col = 'raw_ci_high' if value_col == 'raw_yoy_shift' else f'{value_col}_ci_high'
    d = summary.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col]).copy()
    d = d.sort_values(value_col)
    if d.empty:
        return
    y = np.arange(len(d))
    x = d[value_col].to_numpy(float)
    if lo_col in d.columns and hi_col in d.columns:
        lo = d[lo_col].to_numpy(float)
        hi = d[hi_col].to_numpy(float)
        xerr = np.vstack([x - lo, hi - x])
        xerr = np.where(np.isfinite(xerr), np.maximum(xerr, 0.0), 0.0)
    else:
        xerr = None
    fig, ax = plt.subplots(figsize=(8, max(4, 0.42 * len(d))))
    ax.axvline(0, linewidth=1)
    ax.errorbar(x, y, xerr=xerr, fmt='o', capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d['axis_id'])
    ax.set_xlabel('YoY shift: 2020 COVID window minus matched 2019 months')
    ax.set_title('Axis-level COVID style shift')
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _plot_axis_effect_size(summary: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    if summary.empty:
        return
    cols = [c for c in ['raw_yoy_shift_std_base', 'raw_yoy_shift_iqr_base'] if c in summary.columns]
    if not cols:
        return
    d = summary.replace([np.inf, -np.inf], np.nan).copy()
    sort_col = 'raw_yoy_shift_std_base' if 'raw_yoy_shift_std_base' in d.columns else cols[0]
    d = d.dropna(subset=[sort_col]).sort_values(sort_col)
    if d.empty:
        return
    y = np.arange(len(d))
    fig, axes = plt.subplots(1, len(cols), figsize=(5.8 * len(cols), max(4.6, 0.42 * len(d))), squeeze=False)
    titles = {
        'raw_yoy_shift_std_base': 'YoY shift / baseline SD',
        'raw_yoy_shift_iqr_base': 'YoY shift / baseline IQR',
    }
    for ax, col in zip(axes.ravel(), cols):
        vals = d[col].to_numpy(float)
        ax.axvline(0, linewidth=1)
        ax.barh(y, vals)
        ax.set_yticks(y)
        ax.set_yticklabels(d['axis_id'].astype(str), fontsize=9)
        ax.set_title(titles.get(col, col))
        ax.set_xlabel('Normalized effect size')
        xmax = np.nanmax(np.abs(vals)) if len(vals) else 0.0
        for yy, val in zip(y, vals):
            if np.isfinite(val):
                dx = np.sign(val if val != 0 else 1.0) * max(xmax, 1e-9) * 0.025
                ax.text(val + dx, yy, f'{val:.2f}', va='center', fontsize=8)
        ax.margins(x=0.18)
    fig.suptitle('Axis-level COVID style shift: normalized effect sizes', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _plot_axis_monthly_yoy(ts: pd.DataFrame, out_path: Path, dims: list[str], covid_periods: list[pd.Period] | None = None) -> None:
    import matplotlib.pyplot as plt
    dims = [str(d).replace('_score', '') for d in dims]
    d = ts[ts['axis_id'].isin(dims)].dropna(subset=['yoy_delta']).copy()
    if d.empty:
        return
    # Preserve requested order while dropping missing dimensions.
    present = [x for x in dims if x in set(d['axis_id'])]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    months = sorted(d['month'].astype(str).unique())
    if covid_periods:
        covid_months = {str(p) for p in covid_periods}
        idxs = [i for i, m in enumerate(months) if m in covid_months]
        if idxs:
            ax.axvspan(min(idxs) - 0.5, max(idxs) + 0.5, alpha=0.08, label='COVID window')
    for axis_id in present:
        g = d[d['axis_id'] == axis_id].copy().sort_values('month')
        x = [months.index(str(m)) for m in g['month'].astype(str)]
        ax.plot(x, g['yoy_delta'], marker='o', linewidth=1.8, label=axis_id)
    ax.axhline(0, linewidth=1)
    ax.set_title('Monthly YoY axis-share changes')
    ax.set_xlabel('Month')
    ax.set_ylabel('Axis share minus same month last year')
    ax.set_xticks(np.arange(len(months)))
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.legend(fontsize=8, ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _plot_axis_distribution_curves(im: pd.DataFrame, score_cols: list[str], covid_periods: list[pd.Period], base_periods: list[pd.Period], bins: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    cols = [c for c in score_cols if c in im.columns][:min(12, len(score_cols))]
    if not cols:
        return
    ncols = 3 if len(cols) > 2 else len(cols)
    nrows = int(math.ceil(len(cols) / max(ncols, 1)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.35 * nrows), squeeze=False)
    base = im[im['_period'].isin(base_periods)]
    cov = im[im['_period'].isin(covid_periods)]
    mids = (bins[:-1] + bins[1:]) / 2
    for ax, sc in zip(axes.ravel(), cols):
        hb = _weighted_hist_prob(pd.to_numeric(base[sc], errors='coerce').to_numpy(float), base['_weight'].to_numpy(float), bins)
        hc = _weighted_hist_prob(pd.to_numeric(cov[sc], errors='coerce').to_numpy(float), cov['_weight'].to_numpy(float), bins)
        ax.plot(mids, hb, marker='o', linewidth=1.2, markersize=3, label='2019 matched')
        ax.plot(mids, hc, marker='o', linewidth=1.2, markersize=3, label='2020 COVID')
        ax.set_title(sc.replace('_score', ''), fontsize=10)
        ax.set_xlabel('score')
        ax.set_ylabel('weighted probability')
    for ax in axes.ravel()[len(cols):]:
        ax.set_axis_off()
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.suptitle('Calendar-matched axis-score distributions', y=1.055, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _plot_axis_distance_bar(dist: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    if dist.empty:
        return
    d = dist.replace([np.inf, -np.inf], np.nan).dropna(subset=['axis_id']).copy()
    if d.empty:
        return
    # Separate sorting per metric is easier to read than sharing a hidden y-axis.
    dw = d.sort_values('wasserstein', ascending=True).tail(min(len(d), 14))
    dj = d.sort_values('jsd', ascending=True).tail(min(len(d), 14))
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4.8, 0.42 * max(len(dw), len(dj)))))
    for ax, dd, metric, xlabel, title in [
        (axes[0], dw, 'wasserstein', 'Wasserstein-1', 'Distribution movement'),
        (axes[1], dj, 'jsd', 'Jensen-Shannon divergence', 'Distribution separation'),
    ]:
        y = np.arange(len(dd))
        ax.barh(y, dd[metric].to_numpy(float))
        ax.set_yticks(y)
        ax.set_yticklabels(dd['axis_id'].astype(str), fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        xmax = float(np.nanmax(dd[metric].to_numpy(float))) if len(dd) else 0.0
        for yy, val in zip(y, dd[metric].to_numpy(float)):
            if np.isfinite(val):
                ax.text(val + max(xmax, 1e-9) * 0.015, yy, f'{val:.4f}', va='center', fontsize=8)
        ax.margins(x=0.12)
    fig.suptitle('Axis distribution shift: 2019 matched vs 2020 COVID', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _plot_pca_migration(pca_df: pd.DataFrame, centers: pd.DataFrame, out_path: Path, seed: int = 42) -> None:
    import matplotlib.pyplot as plt
    if pca_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    # Low-alpha scatter gives the raw support; contour lines make overlap readable.
    for group, g in pca_df.groupby('period_group'):
        gg = g.sample(n=min(len(g), 2500), random_state=seed) if len(g) > 2500 else g
        ax.scatter(gg['pc1'], gg['pc2'], s=5, alpha=0.18, label=str(group))
        try:
            x = g['pc1'].to_numpy(float); y = g['pc2'].to_numpy(float)
            H, xe, ye = np.histogram2d(x, y, bins=45)
            if np.nanmax(H) > 0:
                Xc = (xe[:-1] + xe[1:]) / 2
                Yc = (ye[:-1] + ye[1:]) / 2
                levels = np.nanquantile(H[H > 0], [0.55, 0.75, 0.90])
                levels = np.unique(levels)
                if len(levels):
                    ax.contour(Xc, Yc, H.T, levels=levels, linewidths=1.0, alpha=0.7)
        except Exception:
            pass
    if {'period_group', 'pc1', 'pc2'}.issubset(centers.columns):
        cc = centers.set_index('period_group')
        if {'base_2019_matched', 'covid_2020'}.issubset(cc.index):
            x0, y0 = float(cc.loc['base_2019_matched', 'pc1']), float(cc.loc['base_2019_matched', 'pc2'])
            x1, y1 = float(cc.loc['covid_2020', 'pc1']), float(cc.loc['covid_2020', 'pc2'])
            ax.scatter([x0], [y0], s=120, marker='x', linewidths=2.5, label='base centroid')
            ax.scatter([x1], [y1], s=120, marker='x', linewidths=2.5, label='covid centroid')
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', lw=2.5))
            ax.text(x1, y1, '  shift', va='center', fontsize=9)
    ax.set_title('Raw-axis PCA migration: matched 2019 to COVID 2020')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _plot_seasonality_robustness(summary: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    if summary.empty or 'residualized_yoy_shift' not in summary.columns:
        return
    d = summary.replace([np.inf, -np.inf], np.nan).dropna(subset=['raw_yoy_shift', 'residualized_yoy_shift']).copy()
    if d.empty:
        return
    d = d.sort_values('raw_yoy_shift')
    x = np.arange(len(d))
    width = 0.42
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(d)), 5))
    ax.axhline(0, linewidth=1)
    ax.bar(x - width/2, d['raw_yoy_shift'], width, label='Raw YoY')
    ax.bar(x + width/2, d['residualized_yoy_shift'], width, label='Residualized vs thermal/color axes')
    ax.set_xticks(x)
    ax.set_xticklabels(d['axis_id'], rotation=45, ha='right')
    ax.set_ylabel('YoY shift')
    ax.set_title('Seasonality-exclusion robustness')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def exp_prototype_response(args) -> dict:
    """Task 1: merged axis-level COVID style/distribution shift with seasonality-robust measurement.

    The original v3 task-1 used routed-prompt prototypes as the analysis unit.
    That made the result sensitive to product type seasonality, e.g. sweater vs
    shorts.  The revised task-1 treats prototypes only as intermediate knowledge
    injection and moves the social-science analysis to continuous axis/composite
    scores.  The `prototype_response` exp_id is kept as a backward-compatible
    task-1 entry point; the former style_shift task is now merged into this full axis-level distribution-shift task.
    """
    root = Path(args.social_output_root)
    out = ensure_dir(root / 'analysis' / args.exp_id)
    im = _load_item_monthly_with_scores(root)

    season_dims = getattr(args, 'task1_season_control_dims', 'material_thermal_weight,color_temperature,color_lightness,material_softness')
    main_dims = getattr(args, 'task1_axis_dims', '') or getattr(args, 'style_dims', '')
    main_cols = _resolve_score_cols(im, main_dims, exclude_dims=season_dims)
    season_cols = _resolve_score_cols(im, season_dims, exclude_dims=None)
    raw_cols = _resolve_score_cols(im, getattr(args, 'task1_raw_axis_dims', _dim_csv(TASK1_RAW_AXIS_DIMS)), exclude_dims=season_dims)
    composite_cols = _resolve_score_cols(im, getattr(args, 'task1_composite_dims', _dim_csv(TASK1_COMPOSITE_DIMS)), exclude_dims=None)
    # PCA and copula are intentionally restricted to raw axes by default.
    # Composite scores such as casual_like=1-formality and office_like=formality*public
    # create mechanical dependence and should not drive the multivariate/coplua structure.
    pca_cols = _resolve_score_cols(im, getattr(args, 'task1_pca_dims', _dim_csv(TASK1_RAW_AXIS_DIMS)), exclude_dims=season_dims)
    copula_cols = _resolve_score_cols(im, getattr(args, 'task1_copula_dims', _dim_csv(TASK1_RAW_AXIS_DIMS)), exclude_dims=season_dims)
    if not main_cols:
        raise RuntimeError('No axis/composite score columns found for task 1. Rerun social inference v3 and check item_monthly_panel/item_semantic_scores.')
    if not pca_cols:
        pca_cols = raw_cols or main_cols
    if not copula_cols:
        copula_cols = raw_cols or main_cols

    covid_periods, base_periods = _task1_period_pairs(im['month'].unique(), args.event_month, getattr(args, 'task1_covid_end_month', ''))
    if not covid_periods:
        raise RuntimeError(
            f'No calendar-matched COVID months found for event_month={args.event_month}. '
            f'Task 1 needs both t and t-12 months in item_monthly_panel.'
        )
    bins = np.linspace(0, 1, int(args.hist_bins) + 1)
    n_boot = int(getattr(args, 'task1_bootstrap_n', 500))

    # Monthly weighted axis means and YoY deltas.
    mon = _axis_monthly_wide(im, main_cols, '_weight')
    ts = _make_axis_long_timeseries(mon, main_cols, covid_periods)
    ts.to_parquet(out / 'axis_monthly_yoy_timeseries.parquet', index=False)

    # Optional residualized scores for seasonality robustness.
    resid_summary = {}
    resid_cols = []
    if season_cols:
        resid_item = _build_residualized_scores(im, main_cols, season_cols, args)
        if len(resid_item) and 'article_id' in resid_item.columns and 'article_id' in im.columns:
            im_resid = im.merge(resid_item, on='article_id', how='left')
        elif len(resid_item) and '_row_id_for_resid' in resid_item.columns:
            im_resid = im.copy()
            im_resid['_row_id_for_resid'] = np.arange(len(im_resid))
            im_resid = im_resid.merge(resid_item, on='_row_id_for_resid', how='left')
        else:
            im_resid = pd.DataFrame()
        resid_cols = [f'{sc.replace("_score", "")}_resid' for sc in main_cols if f'{sc.replace("_score", "")}_resid' in im_resid.columns] if len(im_resid) else []
        if resid_cols:
            mon_resid = _axis_monthly_wide(im_resid, resid_cols, '_weight')
            lookup = mon_resid.set_index('_period')
            for sc, rc in zip(main_cols, resid_cols):
                deltas, weights = [], []
                for cp in covid_periods:
                    bp = cp - 12
                    if cp in lookup.index and bp in lookup.index:
                        deltas.append(float(lookup.loc[cp, rc] - lookup.loc[bp, rc]))
                        weights.append(float(lookup.loc[cp, 'total_sales']))
                resid_summary[sc.replace('_score', '')] = _paired_weighted_shift(np.asarray(deltas), np.asarray(weights))

    # Axis-level mean shift summary with standardized/IQR-normalized effect sizes.
    base = im[im['_period'].isin(base_periods)].copy()
    cov = im[im['_period'].isin(covid_periods)].copy()
    lookup = mon.set_index('_period')
    summary_rows = []
    for sc in main_cols:
        axis_id = sc.replace('_score', '')
        deltas, weights = [], []
        for cp in covid_periods:
            bp = cp - 12
            if cp in lookup.index and bp in lookup.index:
                dlt = float(lookup.loc[cp, sc] - lookup.loc[bp, sc])
                wt = float(lookup.loc[cp, 'total_sales'])
                if np.isfinite(dlt) and np.isfinite(wt) and wt > 0:
                    deltas.append(dlt); weights.append(wt)
        deltas = np.asarray(deltas, dtype=float); weights = np.asarray(weights, dtype=float)
        raw = _paired_weighted_shift(deltas, weights)
        raw_lo, raw_hi = _bootstrap_month_ci(deltas, weights, n_boot, int(args.seed))
        cm = _calendar_matched_shift(im, sc, covid_periods, base_periods)
        cm_lo, cm_hi = _calendar_matched_bootstrap_ci(im, sc, covid_periods, base_periods, n_boot, int(args.seed))
        xb = pd.to_numeric(base[sc], errors='coerce').to_numpy(float)
        wb = base['_weight'].to_numpy(float)
        scale = _weighted_scale_stats(xb, wb)
        base_sd = scale['sd']; base_iqr = scale['iqr']
        resid = resid_summary.get(axis_id, np.nan)
        summary_rows.append({
            'axis_id': axis_id,
            'score_col': sc,
            'covid_month_start': str(min(covid_periods)),
            'covid_month_end': str(max(covid_periods)),
            'base_month_start': str(min(base_periods)),
            'base_month_end': str(max(base_periods)),
            'num_matched_months': int(len(deltas)),
            'raw_yoy_shift': raw,
            'raw_ci_low': raw_lo,
            'raw_ci_high': raw_hi,
            'calendar_matched_shift': cm,
            'cm_ci_low': cm_lo,
            'cm_ci_high': cm_hi,
            'baseline_sd': base_sd,
            'baseline_iqr': base_iqr,
            'baseline_q25': scale['q25'],
            'baseline_q75': scale['q75'],
            'raw_yoy_shift_std_base': _safe_ratio(raw, base_sd),
            'raw_yoy_ci_low_std_base': _safe_ratio(raw_lo, base_sd),
            'raw_yoy_ci_high_std_base': _safe_ratio(raw_hi, base_sd),
            'raw_yoy_shift_iqr_base': _safe_ratio(raw, base_iqr),
            'raw_yoy_ci_low_iqr_base': _safe_ratio(raw_lo, base_iqr),
            'raw_yoy_ci_high_iqr_base': _safe_ratio(raw_hi, base_iqr),
            'calendar_matched_shift_std_base': _safe_ratio(cm, base_sd),
            'calendar_matched_shift_iqr_base': _safe_ratio(cm, base_iqr),
            'residualized_yoy_shift': resid,
            'residualized_yoy_shift_std_base': _safe_ratio(resid, base_sd),
            'residualized_yoy_shift_iqr_base': _safe_ratio(resid, base_iqr),
            'covid_purchase_weight': float(weights.sum()) if len(weights) else 0.0,
        })
    summary = pd.DataFrame(summary_rows).sort_values('raw_yoy_shift')
    summary.to_parquet(out / 'axis_yoy_shift_summary.parquet', index=False)
    summary.to_parquet(out / 'axis_effect_size_summary.parquet', index=False)

    # Distribution distances and quantile shifts.
    dist_rows = []
    qgrid = np.asarray([0.10, 0.25, 0.50, 0.75, 0.90])
    for sc in main_cols:
        xb = pd.to_numeric(base[sc], errors='coerce').to_numpy(float)
        xc = pd.to_numeric(cov[sc], errors='coerce').to_numpy(float)
        wb = base['_weight'].to_numpy(float)
        wc = cov['_weight'].to_numpy(float)
        hb = _weighted_hist_prob(xb, wb, bins)
        hc = _weighted_hist_prob(xc, wc, bins)
        qb = _weighted_quantile(xb, wb, qgrid)
        qc = _weighted_quantile(xc, wc, qgrid)
        mb = _weighted_moments(xb, wb)
        mc = _weighted_moments(xc, wc)
        w1 = _weighted_wasserstein_1d(xb, xc, wb, wc)
        mean_shift = mc['mean'] - mb['mean']
        scale = _weighted_scale_stats(xb, wb)
        base_sd = scale['sd']; base_iqr = scale['iqr']
        row = {
            'axis_id': sc.replace('_score', ''),
            'wasserstein': w1,
            'wasserstein_std_base': _safe_ratio(w1, base_sd),
            'wasserstein_iqr_base': _safe_ratio(w1, base_iqr),
            'jsd': _jsd_from_probs(hb, hc),
            'weighted_ks': _weighted_ks_1d(xb, xc, wb, wc),
            'mean_base': mb['mean'],
            'mean_covid': mc['mean'],
            'mean_shift': mean_shift,
            'mean_shift_std_base': _safe_ratio(mean_shift, base_sd),
            'mean_shift_iqr_base': _safe_ratio(mean_shift, base_iqr),
            'baseline_sd': base_sd,
            'baseline_iqr': base_iqr,
            'baseline_q25': scale['q25'],
            'baseline_q75': scale['q75'],
            'median_base': float(qb[2]) if len(qb) > 2 else np.nan,
            'median_covid': float(qc[2]) if len(qc) > 2 else np.nan,
            'base_purchase_weight': float(np.nansum(wb)),
            'covid_purchase_weight': float(np.nansum(wc)),
        }
        for q, bq, cq in zip(qgrid, qb, qc):
            qq = int(round(q * 100))
            qshift = float(cq - bq)
            row[f'q{qq}_base'] = float(bq)
            row[f'q{qq}_covid'] = float(cq)
            row[f'q{qq}_shift'] = qshift
            row[f'q{qq}_shift_std_base'] = _safe_ratio(qshift, base_sd)
            row[f'q{qq}_shift_iqr_base'] = _safe_ratio(qshift, base_iqr)
        dist_rows.append(row)
    dist = pd.DataFrame(dist_rows).sort_values('wasserstein', ascending=False)
    dist.to_parquet(out / 'axis_distribution_distance.parquet', index=False)

    # Multivariate PCA and MMD in the main axis space.
    both = pd.concat([
        base.assign(period_group='base_2019_matched'),
        cov.assign(period_group='covid_2020'),
    ], ignore_index=True)
    sample = _sample_weighted(both[['period_group', '_weight'] + pca_cols], '_weight', int(args.max_samples), int(args.seed))
    pca_df = pd.DataFrame(); centers = pd.DataFrame(); loadings = pd.DataFrame(); mmd = np.nan
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        X = sample[pca_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=float)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        pca = PCA(n_components=2, random_state=int(args.seed))
        Z = pca.fit_transform(Xs)
        pca_df = pd.DataFrame({'period_group': sample['period_group'].values, 'pc1': Z[:,0], 'pc2': Z[:,1], 'weight': sample['_weight'].values})
        centers = pca_df.groupby('period_group', as_index=False).agg(pc1=('pc1','mean'), pc2=('pc2','mean'), n=('pc1','size'))
        loadings = pd.DataFrame({
            'axis_id': [c.replace('_score','') for c in pca_cols],
            'pc1_loading': pca.components_[0],
            'pc2_loading': pca.components_[1],
        })
        evr = pd.DataFrame({'pc': ['pc1','pc2'], 'explained_variance_ratio': pca.explained_variance_ratio_})
        pca_df.to_parquet(out / 'axis_pca_distribution_points.parquet', index=False)
        centers.to_parquet(out / 'axis_pca_distribution_centers.parquet', index=False)
        loadings.to_parquet(out / 'axis_pca_loadings.parquet', index=False)
        evr.to_parquet(out / 'axis_pca_explained_variance.parquet', index=False)
        try:
            from sklearn.metrics.pairwise import rbf_kernel
            preX = sample[sample['period_group']=='base_2019_matched'][pca_cols].to_numpy(float)
            postX = sample[sample['period_group']=='covid_2020'][pca_cols].to_numpy(float)
            if len(preX) > 1 and len(postX) > 1:
                gamma = 1.0 / max(len(pca_cols), 1)
                mmd = float(rbf_kernel(preX, preX, gamma=gamma).mean() + rbf_kernel(postX, postX, gamma=gamma).mean() - 2 * rbf_kernel(preX, postX, gamma=gamma).mean())
        except Exception:
            mmd = np.nan
    except Exception as exc:
        pd.DataFrame([{'status': f'pca_failed:{type(exc).__name__}:{exc}'}]).to_parquet(out / 'axis_pca_distribution_points.parquet', index=False)
    pd.DataFrame([{'mmd_rbf': mmd, 'num_axis_dims': len(pca_cols), 'num_sampled_rows': int(len(sample)), 'dims_used': ','.join(_axis_ids(pca_cols))}]).to_parquet(out / 'axis_mmd_results.parquet', index=False)

    # Gaussian copula dependence shift for raw axes only by default.
    cop_base = _gaussian_copula_corr(base, copula_cols, int(args.max_samples), int(args.seed), '_weight')
    cop_cov = _gaussian_copula_corr(cov, copula_cols, int(args.max_samples), int(args.seed), '_weight')
    cop_delta = cop_cov - cop_base
    cop_base.to_csv(out / 'axis_copula_corr_base.csv')
    cop_cov.to_csv(out / 'axis_copula_corr_covid.csv')
    cop_delta.to_csv(out / 'axis_copula_corr_delta.csv')
    cop_long = cop_delta.stack().reset_index()
    cop_long.columns = ['axis_i', 'axis_j', 'corr_delta']
    cop_long = cop_long.merge(cop_base.stack().rename('corr_base').reset_index().rename(columns={'level_0':'axis_i','level_1':'axis_j'}), on=['axis_i','axis_j'], how='left')
    cop_long = cop_long.merge(cop_cov.stack().rename('corr_covid').reset_index().rename(columns={'level_0':'axis_i','level_1':'axis_j'}), on=['axis_i','axis_j'], how='left')
    cop_long.to_parquet(out / 'axis_copula_shift.parquet', index=False)
    pd.DataFrame([{'copula_shift_frobenius': float(np.linalg.norm(cop_delta.fillna(0).to_numpy(), ord='fro')), 'num_copula_dims': int(len(copula_cols)), 'dims_used': ','.join(_axis_ids(copula_cols))}]).to_parquet(out / 'axis_copula_shift_summary.parquet', index=False)

    if args.make_figures:
        _plot_axis_yoy_forest(summary, out / 'axis_yoy_shift_forest.png')
        _plot_axis_effect_size(summary, out / 'axis_effect_size_standardized.png')
        # Keep the legacy combined file, but add clearer grouped views.
        show_dims = [c.replace('_score','') for c in (raw_cols + composite_cols)[:min(8, len(raw_cols + composite_cols))]]
        _plot_axis_monthly_yoy(ts, out / 'axis_monthly_timeseries_yoy.png', show_dims)
        _plot_axis_monthly_yoy(ts, out / 'axis_monthly_yoy_home_practical.png', TASK1_HOME_PRACTICAL_DIMS, covid_periods=covid_periods)
        _plot_axis_monthly_yoy(ts, out / 'axis_monthly_yoy_public_formal.png', TASK1_PUBLIC_FORMAL_DIMS, covid_periods=covid_periods)
        # Main KDE output uses raw axes. Composite distributions are saved separately.
        _plot_axis_distribution_curves(im, raw_cols or main_cols, covid_periods, base_periods, bins, out / 'axis_kde_calendar_matched.png')
        _plot_axis_distribution_curves(im, raw_cols, covid_periods, base_periods, bins, out / 'axis_kde_calendar_matched_raw.png')
        _plot_axis_distribution_curves(im, composite_cols, covid_periods, base_periods, bins, out / 'axis_kde_calendar_matched_composite.png')
        _plot_axis_distance_bar(dist, out / 'axis_distribution_distance_bar.png')
        _plot_axis_distance_bar(dist[dist['axis_id'].isin(_axis_ids(raw_cols))], out / 'axis_distribution_distance_bar_raw.png')
        _plot_axis_distance_bar(dist[dist['axis_id'].isin(_axis_ids(composite_cols))], out / 'axis_distribution_distance_bar_composite.png')
        _plot_pca_migration(pca_df, centers, out / 'axis_pca_density_migration.png', int(args.seed))
        if len(cop_delta):
            _plot_heatmap(cop_delta, out / 'axis_copula_delta_heatmap.png', 'COVID - matched baseline copula correlation, raw axes only')
        _plot_seasonality_robustness(summary, out / 'axis_shift_seasonality_robustness.png')

    return {
        'task': 'axis_covid_style_shift_merged_with_distribution_shift',
        'axis_rows': int(len(summary)),
        'matched_months': int(len(covid_periods)),
        'covid_window': [str(min(covid_periods)), str(max(covid_periods))],
        'base_window': [str(min(base_periods)), str(max(base_periods))],
        'mmd_rbf': mmd,
        'pca_dims': _axis_ids(pca_cols),
        'copula_dims': _axis_ids(copula_cols),
    }


def _gaussian_copula_corr(df: pd.DataFrame, cols: list[str], max_samples: int, seed: int, weight_col: str | None = None) -> pd.DataFrame:
    d = df[cols + ([weight_col] if weight_col else [])].replace([np.inf,-np.inf], np.nan).dropna().copy()
    if len(d) == 0:
        return pd.DataFrame(index=cols, columns=cols, dtype=float)
    if max_samples and len(d) > max_samples:
        d = _sample_weighted(d, weight_col, max_samples, seed) if weight_col else d.sample(n=max_samples, random_state=seed)
    try:
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(n_quantiles=max(10, min(1000, len(d))), output_distribution='normal', random_state=seed)
        Z = qt.fit_transform(d[cols].to_numpy(dtype=float))
        Z = np.clip(Z, -6, 6)
        corr = np.corrcoef(Z, rowvar=False)
    except Exception:
        corr = d[cols].rank(pct=True).corr(method='spearman').to_numpy(dtype=float)
    return pd.DataFrame(corr, index=[c.replace('_score','') for c in cols], columns=[c.replace('_score','') for c in cols])


def exp_style_shift(args) -> dict:
    """Backward-compatible alias for merged Task 1.

    The former task-2 Axis Distribution Shift has been merged into Task 1.
    Calling --exp_id style_shift now runs the same calendar-matched,
    seasonality-robust axis COVID style/distribution-shift analysis as
    --exp_id axis_covid_style_shift or --exp_id prototype_response.
    """
    return exp_prototype_response(args)


def exp_exposure_response(args) -> dict:
    root = Path(args.social_output_root)
    out = ensure_dir(root / 'analysis' / args.exp_id)
    im = pd.read_parquet(root / 'item_monthly_panel.parquet')
    im['log_sales'] = np.log1p(pd.to_numeric(im['sales_count'], errors='coerce').fillna(0.0))
    im['rel_month'] = _rel_month(im['month'], args.event_month)
    im['post_event'] = (im['rel_month'] >= 0).astype(float)
    scols = _style_cols(im, args.style_dims)
    target_dims = [x.strip() for x in str(args.exposure_dims).split(',') if x.strip()]
    if not target_dims:
        target_dims = ['formal','comfort']
    rows = []
    event_rows = []
    trend_rows = []
    for dim in target_dims:
        sc = f'{dim}_score'
        if sc not in im.columns:
            continue
        item_score = im[['article_id', sc]].drop_duplicates('article_id').copy()
        qh = item_score[sc].quantile(float(args.high_quantile))
        ql = item_score[sc].quantile(float(args.low_quantile))
        item_score['exposure_group'] = np.where(item_score[sc] >= qh, 'high', np.where(item_score[sc] <= ql, 'low', 'mid'))
        tmp = im.merge(item_score[['article_id','exposure_group']], on='article_id', how='left')
        tmp = tmp[tmp['exposure_group'].isin(['high','low'])].copy()
        agg = tmp.groupby(['exposure_group','month','rel_month','post_event'], as_index=False).agg(sales_count=('sales_count','sum'))
        agg['log_sales'] = np.log1p(agg['sales_count'])
        agg['is_high'] = (agg['exposure_group'] == 'high').astype(float)
        agg['high_x_post'] = agg['is_high'] * agg['post_event']
        agg['_trend'] = agg['rel_month'] - agg['rel_month'].min()
        res = _ols(agg, 'log_sales', ['is_high','post_event','high_x_post','_trend'], min_n=6)
        res.update({'knowledge_dim': dim, 'model': 'aggregate_high_low_did_style', 'high_quantile': qh, 'low_quantile': ql})
        rows.append(res)
        trend = agg.copy(); trend['knowledge_dim'] = dim
        trend_rows.append(trend)
        # event study aggregate, using rel -1 as baseline.
        for l in range(-int(args.event_window), int(args.event_window)+1):
            if l == -1:
                continue
            cname = f'high_x_rel_{l:+d}'.replace('+','p').replace('-','m')
            agg[cname] = ((agg['rel_month'] == l) & (agg['exposure_group']=='high')).astype(float)
        xcols = [c for c in agg.columns if c.startswith('high_x_rel_')]
        evres = _ols(agg, 'log_sales', xcols + ['is_high'], fe_cols=['month'], min_n=8)
        for l in range(-int(args.event_window), int(args.event_window)+1):
            if l == -1:
                continue
            cname = f'high_x_rel_{l:+d}'.replace('+','p').replace('-','m')
            event_rows.append({'knowledge_dim': dim, 'rel_month': l, 'coef': evres.get(f'coef_{cname}', np.nan), 'se': evres.get(f'se_{cname}', np.nan), 't': evres.get(f't_{cname}', np.nan), 'nobs': evres.get('nobs', 0), 'status': evres.get('status','unknown')})
    results = pd.DataFrame(rows)
    events = pd.DataFrame(event_rows)
    trends = pd.concat(trend_rows, ignore_index=True) if trend_rows else pd.DataFrame()
    results.to_parquet(out / 'exposure_did_results.parquet', index=False)
    events.to_parquet(out / 'exposure_event_study_results.parquet', index=False)
    trends.to_parquet(out / 'exposure_high_low_trends.parquet', index=False)
    if args.make_figures:
        if len(trends):
            for dim, g in trends.groupby('knowledge_dim'):
                _plot_lines(g, 'month', 'log_sales', 'exposure_group', out / f'high_low_trend_{_safe_name(dim)}.png', f'High vs low {dim} sales')
        if len(events):
            import matplotlib.pyplot as plt
            for dim, g in events.groupby('knowledge_dim'):
                g = g.sort_values('rel_month')
                fig, ax = plt.subplots(figsize=(7,4))
                ax.axhline(0, linewidth=1); ax.axvline(-0.5, linestyle='--', linewidth=1)
                ax.errorbar(g['rel_month'], g['coef'], yerr=1.96*g['se'], marker='o', capsize=3)
                ax.set_title(f'Event-study style response: high {dim}')
                ax.set_xlabel('Relative month'); ax.set_ylabel('coef')
                fig.tight_layout(); fig.savefig(out / f'event_study_{_safe_name(dim)}.png', dpi=200); plt.close(fig)
    return {'did_rows': len(results), 'event_rows': len(events), 'trend_rows': len(trends)}



def _resolve_covid_csv(args) -> str:
    explicit = str(getattr(args, 'covid_csv', '') or '').strip()
    if explicit:
        return explicit
    default = Path(getattr(args, 'data_root', './data')) / 'external' / 'owid-covid-data.csv'
    return str(default)


def _parse_dim_list(x: str | Iterable[str] | None) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [t.strip().replace('_score', '') for t in x.split(',') if t.strip()]
    return [str(t).strip().replace('_score', '') for t in x if str(t).strip()]


def _read_parquet_columns(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        if columns:
            return pd.read_parquet(path, columns=columns)
    except Exception:
        pass
    return pd.read_parquet(path)


def _daily_covid_panel(covid_csv: str, covid_location: str, start_date: str, end_date: str, transform: str = 'delta7') -> pd.DataFrame:
    """Daily COVID variables and standardized COVID shocks.

    The style dynamics task uses shocks/changes by default, rather than levels,
    to reduce common-trend artifacts.  For each base `covid_*_index` column we
    create `<var>_shock`, standardized over the analysis window.
    """
    cpath = Path(covid_csv)
    if not cpath.exists():
        return pd.DataFrame({'date': pd.to_datetime([])})
    covid = pd.read_csv(cpath)
    if 'location' in covid.columns:
        covid = covid[covid['location'].astype(str) == str(covid_location)].copy()
    if covid.empty or 'date' not in covid.columns:
        return pd.DataFrame({'date': pd.to_datetime([])})
    covid['date'] = pd.to_datetime(covid['date'], errors='coerce')
    covid = covid.dropna(subset=['date']).sort_values('date')
    # Keep a little extra history so diff/rolling transformations are valid at the start.
    s = pd.to_datetime(start_date) - pd.Timedelta(days=35)
    e = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    covid = covid[(covid['date'] >= s) & (covid['date'] <= e)].copy()
    out = covid[['date']].drop_duplicates().copy()

    def pick(cols):
        for c in cols:
            if c in covid.columns:
                return pd.to_numeric(covid[c], errors='coerce')
        return None

    cases = pick(['new_cases_smoothed_per_million', 'new_cases_per_million', 'new_cases_smoothed', 'new_cases'])
    deaths = pick(['new_deaths_smoothed_per_million', 'new_deaths_per_million', 'new_deaths_smoothed', 'new_deaths'])
    if cases is not None:
        out['covid_cases_index'] = np.log1p(cases.fillna(0.0).clip(lower=0.0))
    if deaths is not None:
        out['covid_deaths_index'] = np.log1p(deaths.fillna(0.0).clip(lower=0.0))
    if 'stringency_index' in covid.columns:
        out['covid_stringency_index'] = pd.to_numeric(covid['stringency_index'], errors='coerce')
    if 'reproduction_rate' in covid.columns:
        out['covid_reproduction_rate'] = pd.to_numeric(covid['reproduction_rate'], errors='coerce')

    for c in [x for x in out.columns if x.startswith('covid_') and not x.endswith('_z') and not x.endswith('_shock')]:
        z = _zscore(pd.to_numeric(out[c], errors='coerce'))
        out[f'{c}_z'] = z
        base = pd.to_numeric(out[c], errors='coerce')
        if transform == 'level':
            shock = base.copy()
        elif transform == 'diff1':
            shock = base.diff(1)
        else:  # default delta7
            shock = base.diff(7)
        out[f'{c}_shock_raw'] = shock
        out[f'{c}_shock'] = _zscore(shock)
    # Trim back to the requested analysis window.
    out = out[(out['date'] >= pd.to_datetime(start_date)) & (out['date'] <= pd.to_datetime(end_date))].copy()
    return out.reset_index(drop=True)


def _parse_lag_bins(spec: str) -> list[tuple[str, int, int]]:
    ans = []
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            a, b = part.split(':', 1)
        elif '-' in part:
            a, b = part.split('-', 1)
        else:
            a = b = part
        lo, hi = int(a), int(b)
        if hi < lo:
            lo, hi = hi, lo
        ans.append((f'lag_{lo}_{hi}', lo, hi))
    return ans or [('lag_0_3', 0, 3), ('lag_4_7', 4, 7), ('lag_8_14', 8, 14), ('lag_15_28', 15, 28)]


def _load_scores_for_dims(root: Path, dims: list[str]) -> tuple[pd.DataFrame, list[str]]:
    scores_path = root / 'item_semantic_scores.parquet'
    if not scores_path.exists():
        scores_path = root / 'item_axis_scores.parquet'
    if not scores_path.exists():
        scores_path = root / 'item_knowledge_scores.parquet'
    if not scores_path.exists():
        raise FileNotFoundError(f'Missing item_semantic_scores/item_axis_scores under {root}.')
    scores = pd.read_parquet(scores_path)
    scores['article_id'] = scores['article_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    wanted = []
    for d in dims:
        c = d if d.endswith('_score') else f'{d}_score'
        if c in scores.columns and c not in wanted:
            wanted.append(c)
    if not wanted:
        raise KeyError(f'None of requested style dims are available in {scores_path}. requested={dims}; available={[c for c in scores.columns if c.endswith("_score")][:20]}...')
    keep = ['article_id'] + wanted
    return scores[keep].copy(), wanted


def _build_daily_axis_panel(args) -> pd.DataFrame:
    """Build daily, smoothed, seasonality-adjusted axis shares from transactions.

    Output is a wide daily panel with columns like
    `<dim>_share`, `<dim>_smooth`, `<dim>_deviation`, plus daily COVID variables.
    """
    root = Path(args.social_output_root)
    data_root = Path(args.data_root)
    start = str(getattr(args, 'lead_lag_start_date', '2019-10-01'))
    end = str(getattr(args, 'lead_lag_end_date', '2020-09-30'))
    roll_days = int(getattr(args, 'lead_lag_roll_days', 7))
    dims = _parse_dim_list(getattr(args, 'lead_lag_dims', '')) or _parse_dim_list(getattr(args, 'style_dims', ''))
    season_dims = _parse_dim_list(getattr(args, 'lead_lag_season_control_dims', ''))
    all_dims = []
    for d in dims + season_dims:
        if d and d not in all_dims:
            all_dims.append(d)
    scores, score_cols = _load_scores_for_dims(root, all_dims)

    tx_path = data_root / 'hm' / 'processed' / 'hm_transactions.parquet'
    if not tx_path.exists():
        raise FileNotFoundError(f'Missing transaction parquet: {tx_path}')
    tx = _read_parquet_columns(tx_path, columns=['t_dat', 'article_id'])
    tx['date'] = pd.to_datetime(tx['t_dat'], errors='coerce').dt.floor('D')
    tx = tx.dropna(subset=['date'])
    # Keep enough pre-window days for rolling means and lagged COVID shocks.
    pad = max(roll_days + int(getattr(args, 'lead_lag_max_lag_days', 28)) + 7, 45)
    start_pad = pd.to_datetime(start) - pd.Timedelta(days=pad)
    end_dt = pd.to_datetime(end)
    tx = tx[(tx['date'] >= start_pad) & (tx['date'] <= end_dt)].copy()
    tx['article_id'] = tx['article_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    d = tx[['date', 'article_id']].merge(scores, on='article_id', how='left')
    for c in score_cols:
        d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0.0)
    # Daily weighted numerator and denominator. Each transaction is one weight.
    agg = d.groupby('date', as_index=False).agg(transactions=('article_id', 'size'), **{c: (c, 'sum') for c in score_cols})
    calendar = pd.DataFrame({'date': pd.date_range(start_pad, end_dt, freq='D')})
    agg = calendar.merge(agg, on='date', how='left')
    agg['transactions'] = pd.to_numeric(agg['transactions'], errors='coerce').fillna(0.0)
    for c in score_cols:
        agg[c] = pd.to_numeric(agg[c], errors='coerce').fillna(0.0)
        dim = c.replace('_score', '')
        agg[f'{dim}_share'] = agg[c] / agg['transactions'].replace(0, np.nan)
        num_roll = agg[c].rolling(roll_days, min_periods=max(2, roll_days // 2)).sum()
        den_roll = agg['transactions'].rolling(roll_days, min_periods=max(2, roll_days // 2)).sum()
        agg[f'{dim}_smooth'] = num_roll / den_roll.replace(0, np.nan)

    # Merge daily COVID variables and trim to requested analysis window.
    covid = _daily_covid_panel(_resolve_covid_csv(args), str(getattr(args, 'covid_location', 'World')), start, end, transform=str(getattr(args, 'lead_lag_covid_transform', 'delta7')))
    panel = agg[(agg['date'] >= pd.to_datetime(start)) & (agg['date'] <= pd.to_datetime(end))].copy()
    if len(covid):
        panel = panel.merge(covid, on='date', how='left')
    panel['dow'] = panel['date'].dt.dayofweek.astype(str)
    panel['_trend'] = (panel['date'] - panel['date'].min()).dt.days.astype(float)
    if panel['_trend'].max() > 0:
        panel['_trend_scaled'] = panel['_trend'] / panel['_trend'].max()
    else:
        panel['_trend_scaled'] = 0.0
    panel['log_transactions'] = np.log1p(panel['transactions'].fillna(0.0))

    # Residualize target style axes against day-of-week, trend, transactions, and seasonal axes.
    target_score_cols = [c for c in score_cols if c.replace('_score', '') in dims]
    season_smooth_cols = [f'{d}_smooth' for d in season_dims if f'{d}_smooth' in panel.columns]
    for c in target_score_cols:
        dim = c.replace('_score', '')
        ycol = f'{dim}_smooth'
        xcols = []
        fe = []
        if bool(getattr(args, 'lead_lag_residualize_dow', True)):
            fe.append('dow')
        if bool(getattr(args, 'lead_lag_residualize_trend', True)):
            xcols.append('_trend_scaled')
        if bool(getattr(args, 'lead_lag_control_transactions', True)):
            xcols.append('log_transactions')
        if bool(getattr(args, 'lead_lag_residualize_season_axes', True)):
            xcols.extend([s for s in season_smooth_cols if not s.startswith(dim + '_')])
        tmp = panel[['date', ycol] + xcols + fe].copy()
        res = _ols(tmp, ycol, xcols, fe_cols=fe, min_n=max(30, len(xcols) + 10))
        # Reconstruct fitted values for residuals.  _ols does not return residuals.
        dfit = tmp[[ycol] + xcols + fe].replace([np.inf, -np.inf], np.nan).dropna().copy()
        if res.get('status') == 'ok' and len(dfit) >= max(30, len(xcols) + 10):
            X_parts = [pd.Series(1.0, index=dfit.index, name='const')]
            for xc in xcols:
                X_parts.append(pd.to_numeric(dfit[xc], errors='coerce').astype(float).rename(xc))
            for fc in fe:
                dm = pd.get_dummies(dfit[fc].astype(str), prefix=fc, drop_first=True, dtype=float)
                if dm.shape[1] > 0:
                    X_parts.append(dm)
            X = pd.concat(X_parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            yy = pd.to_numeric(dfit[ycol], errors='coerce').astype(float).to_numpy()
            beta, *_ = np.linalg.lstsq(X.to_numpy(dtype=float), yy, rcond=None)
            resid = pd.Series(yy - X.to_numpy(dtype=float) @ beta, index=dfit.index)
            panel[f'{dim}_deviation'] = np.nan
            panel.loc[dfit.index, f'{dim}_deviation'] = resid
        else:
            panel[f'{dim}_deviation'] = pd.to_numeric(panel[ycol], errors='coerce') - pd.to_numeric(panel[ycol], errors='coerce').mean()
    return panel.reset_index(drop=True)


def _ar_residual_series(s: pd.Series, p: int = 7, min_n: int = 30) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce').astype(float)
    df = pd.DataFrame({'y': s})
    p = int(max(0, p))
    for lag in range(1, p + 1):
        df[f'lag{lag}'] = s.shift(lag)
    d = df.dropna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if len(d) < max(min_n, p + 5) or d['y'].std(ddof=0) <= 1e-12:
        return s - s.mean()
    X = [pd.Series(1.0, index=d.index, name='const')]
    for lag in range(1, p + 1):
        X.append(d[f'lag{lag}'].astype(float))
    X = pd.concat(X, axis=1).to_numpy(dtype=float)
    y = d['y'].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    out.loc[d.index] = y - X @ beta
    return out


def _lead_lag_corr_rows(panel: pd.DataFrame, dims: list[str], covid_shocks: list[str], lags: list[int], y_suffix: str, corr_type: str) -> pd.DataFrame:
    rows = []
    for dim in dims:
        ycol = f'{dim}_{y_suffix}'
        if ycol not in panel.columns:
            continue
        y = pd.to_numeric(panel[ycol], errors='coerce')
        for cv in covid_shocks:
            if cv not in panel.columns:
                continue
            x = pd.to_numeric(panel[cv], errors='coerce')
            for lag in lags:
                xs = x.shift(lag)
                d = pd.DataFrame({'y': y, 'x': xs}).replace([np.inf, -np.inf], np.nan).dropna()
                corr = float(d['y'].corr(d['x'])) if len(d) >= 10 and d['y'].std(ddof=0) > 1e-12 and d['x'].std(ddof=0) > 1e-12 else np.nan
                rows.append({'knowledge_dim': dim, 'covid_var': cv.replace('_shock',''), 'covid_shock_col': cv, 'lag_days': int(lag), 'correlation': corr, 'nobs': int(len(d)), 'correlation_type': corr_type})
    return pd.DataFrame(rows)


def _lag_bin_regression(panel: pd.DataFrame, dims: list[str], covid_shocks: list[str], bins: list[tuple[str, int, int]], ar_lags: int = 7, hac_lags: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pred_rows = []
    for dim in dims:
        ycol = f'{dim}_deviation'
        if ycol not in panel.columns:
            continue
        for cv in covid_shocks:
            if cv not in panel.columns:
                continue
            tmp = panel[['date', ycol, cv]].copy().sort_values('date').reset_index(drop=True)
            tmp['y'] = pd.to_numeric(tmp[ycol], errors='coerce')
            xcols = []
            for p in range(1, int(ar_lags) + 1):
                cname = f'y_lag{p}'
                tmp[cname] = tmp['y'].shift(p)
                xcols.append(cname)
            bin_cols = []
            for bname, lo, hi in bins:
                cname = f'{cv}_{bname}'
                shifted = [pd.to_numeric(tmp[cv], errors='coerce').shift(l) for l in range(lo, hi + 1)]
                tmp[cname] = pd.concat(shifted, axis=1).mean(axis=1)
                xcols.append(cname)
                bin_cols.append((cname, bname, lo, hi))
            tmp['_trend'] = np.arange(len(tmp), dtype=float)
            res = _ols(tmp, 'y', xcols, min_n=max(45, len(xcols) + 10), cov_type='hac', hac_lags=int(hac_lags))
            for cname, bname, lo, hi in bin_cols:
                rows.append({
                    'knowledge_dim': dim,
                    'covid_var': cv.replace('_shock',''),
                    'covid_shock_col': cv,
                    'lag_bin': bname,
                    'lag_start': int(lo),
                    'lag_end': int(hi),
                    'coef': res.get(f'coef_{cname}', np.nan),
                    'se': res.get(f'se_{cname}', np.nan),
                    't': res.get(f't_{cname}', np.nan),
                    'p': res.get(f'p_{cname}', np.nan),
                    'nobs': res.get('nobs', 0),
                    'status': res.get('status', 'unknown'),
                    'model': 'daily_residualized_lag_bin_distributed_regression',
                })
            # Holdout prediction improvement: AR baseline vs AR+COVID bins.
            dreg = tmp[['y'] + xcols].replace([np.inf, -np.inf], np.nan).dropna().copy()
            ar_cols = [c for c in xcols if c.startswith('y_lag')]
            if len(dreg) >= max(60, len(xcols) + 20) and len(ar_cols) > 0:
                split = int(len(dreg) * 0.7)
                train, test = dreg.iloc[:split], dreg.iloc[split:]
                def fit_pred(cols):
                    Xtr = np.column_stack([np.ones(len(train))] + [pd.to_numeric(train[c], errors='coerce').to_numpy(dtype=float) for c in cols])
                    ytr = train['y'].to_numpy(dtype=float)
                    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
                    Xte = np.column_stack([np.ones(len(test))] + [pd.to_numeric(test[c], errors='coerce').to_numpy(dtype=float) for c in cols])
                    return Xte @ beta
                try:
                    yte = test['y'].to_numpy(dtype=float)
                    pred0 = fit_pred(ar_cols)
                    pred1 = fit_pred(xcols)
                    rmse0 = float(np.sqrt(np.mean((yte - pred0) ** 2)))
                    rmse1 = float(np.sqrt(np.mean((yte - pred1) ** 2)))
                    pred_rows.append({
                        'knowledge_dim': dim,
                        'covid_var': cv.replace('_shock',''),
                        'rmse_ar': rmse0,
                        'rmse_ar_covid': rmse1,
                        'rmse_improvement': rmse0 - rmse1,
                        'rmse_improvement_pct': (rmse0 - rmse1) / rmse0 if rmse0 > 1e-12 else np.nan,
                        'train_n': int(len(train)),
                        'test_n': int(len(test)),
                    })
                except Exception as exc:
                    pred_rows.append({'knowledge_dim': dim, 'covid_var': cv.replace('_shock',''), 'status': f'prediction_failed:{type(exc).__name__}:{exc}'})
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def _plot_lag_response_curves(corr: pd.DataFrame, out_path: Path, title: str, max_dims: int = 8) -> None:
    if corr.empty:
        return
    import matplotlib.pyplot as plt
    g = corr.dropna(subset=['correlation']).copy()
    if g.empty:
        return
    dims = list(g['knowledge_dim'].drop_duplicates())[:max_dims]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linestyle='--', linewidth=1)
    for dim in dims:
        dd = g[g['knowledge_dim'] == dim].sort_values('lag_days')
        ax.plot(dd['lag_days'], dd['correlation'], marker='o', linewidth=1.6, label=dim)
    ax.set_title(title)
    ax.set_xlabel('Lag in days (positive = COVID shock leads style deviation)')
    ax.set_ylabel('Correlation')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def _plot_lag_bin_forest(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df.empty:
        return
    import matplotlib.pyplot as plt
    g = df[df['status'].fillna('') == 'ok'].dropna(subset=['coef','se']).copy()
    if g.empty:
        return
    bins = list(g['lag_bin'].drop_duplicates())
    n = len(bins)
    fig, axes = plt.subplots(1, n, figsize=(max(4*n, 8), max(4, 0.32*g['knowledge_dim'].nunique()+2)), sharey=True)
    if n == 1:
        axes = [axes]
    order = (g.groupby('knowledge_dim')['coef'].mean().sort_values().index.tolist())
    ypos = np.arange(len(order))
    for ax, b in zip(axes, bins):
        dd = g[g['lag_bin']==b].set_index('knowledge_dim').reindex(order).reset_index()
        ax.axvline(0, linewidth=1)
        ax.errorbar(dd['coef'], ypos, xerr=1.96*dd['se'], fmt='o', capsize=3)
        ax.set_title(b.replace('lag_', '').replace('_', '–') + ' days')
        ax.set_xlabel('Coef')
        ax.set_yticks(ypos)
        ax.set_yticklabels(order)
    axes[0].invert_yaxis()
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def _plot_prediction_improvement(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df.empty or 'rmse_improvement_pct' not in df.columns:
        return
    import matplotlib.pyplot as plt
    g = df.dropna(subset=['rmse_improvement_pct']).copy()
    if g.empty:
        return
    g = g.sort_values('rmse_improvement_pct')
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35*len(g)+1)))
    ax.axvline(0, linewidth=1)
    ax.barh(g['knowledge_dim'], g['rmse_improvement_pct'])
    ax.set_title(title)
    ax.set_xlabel('RMSE improvement over AR baseline')
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def exp_lead_lag(args) -> dict:
    """Daily lead-lag dynamics for seasonality-adjusted axis deviations.

    This replaces the older weekly raw-share lead-lag path.  It builds daily
    transaction-weighted axis shares, applies 7-day smoothing and residualizes
    style levels against day-of-week, trend, transaction volume, and seasonal
    warm/color axes.  COVID variables are converted into standardized shocks
    (delta-7 by default).  Outputs focus on prewhitened correlations and lag-bin
    distributed regressions, while heatmaps are kept as supplementary views.
    """
    root = Path(args.social_output_root)
    out = ensure_dir(root / 'analysis' / args.exp_id)
    panel = _build_daily_axis_panel(args)
    panel.to_parquet(out / 'daily_axis_deviation_panel.parquet', index=False)

    dims = _parse_dim_list(getattr(args, 'lead_lag_dims', ''))
    if not dims:
        dims = [c.replace('_deviation', '') for c in panel.columns if c.endswith('_deviation')]
    dims = [d for d in dims if f'{d}_deviation' in panel.columns]
    requested_cv = _parse_dim_list(getattr(args, 'lead_lag_covid_vars', ''))
    if not requested_cv:
        requested_cv = ['covid_cases_index', 'covid_deaths_index', 'covid_reproduction_rate', 'covid_stringency_index']
    covid_shocks = []
    for cv in requested_cv:
        raw = cv if cv.startswith('covid_') else f'covid_{cv}'
        cshock = raw if raw.endswith('_shock') else f'{raw}_shock'
        if cshock in panel.columns and pd.to_numeric(panel[cshock], errors='coerce').notna().sum() >= 20:
            if pd.to_numeric(panel[cshock], errors='coerce').std(ddof=0) > 1e-12:
                covid_shocks.append(cshock)
    if not dims:
        raise RuntimeError('No lead-lag axis dimensions available. Check --lead_lag_dims and item_semantic_scores.')
    if not covid_shocks:
        raise RuntimeError('No valid daily COVID shock variables available. Check --covid_csv/--covid_location and --lead_lag_covid_vars.')

    placebo = int(getattr(args, 'lead_lag_placebo_days', 7))
    max_lag = int(getattr(args, 'lead_lag_max_lag_days', 28))
    lags = list(range(-placebo, max_lag + 1))
    # Residualized style deviation vs COVID shocks.
    corr = _lead_lag_corr_rows(panel, dims, covid_shocks, lags, y_suffix='deviation', corr_type='residualized')
    corr.to_parquet(out / 'lead_lag_correlation.parquet', index=False)

    # Prewhitened correlation: remove each series' own AR structure first.
    pw_panel = panel[['date']].copy()
    ar_p = int(getattr(args, 'lead_lag_prewhiten_lags', 7))
    for dim in dims:
        pw_panel[f'{dim}_pw'] = _ar_residual_series(panel[f'{dim}_deviation'], p=ar_p, min_n=45)
    for cv in covid_shocks:
        pw_panel[f'{cv}_pw'] = _ar_residual_series(panel[cv], p=ar_p, min_n=45)
    pw_rows = []
    for dim in dims:
        y = pd.to_numeric(pw_panel[f'{dim}_pw'], errors='coerce')
        for cv in covid_shocks:
            x = pd.to_numeric(pw_panel[f'{cv}_pw'], errors='coerce')
            for lag in lags:
                xs = x.shift(lag)
                d = pd.DataFrame({'y': y, 'x': xs}).replace([np.inf, -np.inf], np.nan).dropna()
                corr_val = float(d['y'].corr(d['x'])) if len(d) >= 10 and d['y'].std(ddof=0) > 1e-12 and d['x'].std(ddof=0) > 1e-12 else np.nan
                pw_rows.append({'knowledge_dim': dim, 'covid_var': cv.replace('_shock',''), 'covid_shock_col': cv, 'lag_days': int(lag), 'correlation': corr_val, 'nobs': int(len(d)), 'correlation_type': 'prewhitened'})
    pw_corr = pd.DataFrame(pw_rows)
    pw_corr.to_parquet(out / 'lead_lag_prewhitened_correlation.parquet', index=False)

    bins = _parse_lag_bins(getattr(args, 'lead_lag_bins', '0:3,4:7,8:14,15:28'))
    bin_reg, pred = _lag_bin_regression(
        panel,
        dims,
        covid_shocks,
        bins,
        ar_lags=int(getattr(args, 'lead_lag_ar_lags_daily', 7)),
        hac_lags=int(getattr(args, 'lead_lag_hac_lags_daily', 7)),
    )
    bin_reg.to_parquet(out / 'lead_lag_bin_regression.parquet', index=False)
    # Backward-compatible filename; now contains lag-bin distributed regressions rather than weekly ARDL.
    bin_reg.to_parquet(out / 'ardl_results.parquet', index=False)
    pred.to_parquet(out / 'lead_lag_prediction_improvement.parquet', index=False)

    if args.make_figures:
        for cv in covid_shocks:
            base_cv = cv.replace('_shock','')
            gg = corr[corr['covid_shock_col'] == cv]
            _plot_lag_response_curves(gg, out / f'lead_lag_response_curve_{_safe_name(base_cv)}.png', f'Daily residualized lead-lag: {base_cv}')
            gpw = pw_corr[pw_corr['covid_shock_col'] == cv]
            _plot_lag_response_curves(gpw, out / f'lead_lag_response_curve_prewhitened_{_safe_name(base_cv)}.png', f'Prewhitened daily lead-lag: {base_cv}')
            mat = gpw.pivot(index='knowledge_dim', columns='lag_days', values='correlation') if len(gpw) else pd.DataFrame()
            if len(mat):
                _plot_heatmap(mat, out / f'lead_lag_heatmap_prewhitened_{_safe_name(base_cv)}.png', f'Prewhitened lead-lag correlation: {base_cv}', figsize=(10, 5))
            _plot_lag_bin_forest(bin_reg[bin_reg['covid_shock_col'] == cv], out / f'lead_lag_bin_effect_forest_{_safe_name(base_cv)}.png', f'Daily lag-bin effects: {base_cv}')
            _plot_prediction_improvement(pred[pred['covid_var'] == base_cv], out / f'lead_lag_prediction_improvement_{_safe_name(base_cv)}.png', f'Prediction gain from {base_cv}')
    return {
        'daily_rows': int(len(panel)),
        'style_dims': dims,
        'covid_shocks': covid_shocks,
        'lead_lag_rows': int(len(corr)),
        'prewhitened_rows': int(len(pw_corr)),
        'lag_bin_rows': int(len(bin_reg)),
        'prediction_rows': int(len(pred)),
        'date_min': str(panel['date'].min().date()) if len(panel) else '',
        'date_max': str(panel['date'].max().date()) if len(panel) else '',
    }

def _parse_user_dims(dims: str | Iterable[str] | None, available_score_cols: list[str]) -> list[str]:
    if dims:
        wanted = [str(x).strip().replace('_score', '') for x in str(dims).split(',') if str(x).strip()]
    else:
        wanted = [c.replace('_score', '') for c in available_score_cols]
    out = []
    aset = set(available_score_cols)
    for d in wanted:
        c = d if d.endswith('_score') else f'{d}_score'
        if c in aset and c not in out:
            out.append(c)
    return out


def _choose_torch_device(mode: str = 'auto'):
    mode = str(mode or 'auto').lower()
    try:
        import torch
        if mode == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('user_use_cuda=cuda but torch.cuda.is_available() is False')
        if mode in {'cuda', 'auto'} and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    except Exception:
        if mode == 'cuda':
            raise
        return None


def _period_labels_for_user(d: pd.DataFrame, event_month: str, covid_end_month: str = '') -> tuple[pd.DataFrame, list[str], list[str]]:
    out = d.copy()
    months = pd.PeriodIndex(out['month'].astype(str), freq='M')
    ev = pd.Period(event_month, freq='M')
    if covid_end_month:
        end = pd.Period(covid_end_month, freq='M')
    else:
        end = months.max()
    covid_periods = [pd.Period(f'{y}-{m:02d}', freq='M') for y, m in []]
    covid_periods = list(pd.period_range(ev, end, freq='M'))
    base_periods = [p - 12 for p in covid_periods]
    period_map = {str(p): 'covid_2020' for p in covid_periods}
    period_map.update({str(p): 'base_2019_matched' for p in base_periods})
    out['period_group'] = out['month'].astype(str).map(period_map)
    out = out.dropna(subset=['period_group']).copy()
    return out, [str(p) for p in covid_periods], [str(p) for p in base_periods]


def _load_purchase_level(root: Path, data_root: Path, style_dims: str, max_rows: int, seed: int) -> pd.DataFrame:
    """Load transaction-level purchases joined with axis/composite scores and customer fields.

    max_rows<=0 means full transaction-level analysis.  The user heterogeneity task
    no longer relies on sklearn/OpenBLAS-heavy subsampling; CUDA-enabled binned KDE
    and torch GMM can process all rows in chunks.
    """
    tx_path = data_root / 'hm' / 'processed' / 'hm_transactions.parquet'
    cust_path = data_root / 'hm' / 'processed' / 'hm_customers.parquet'
    scores_path = root / 'item_semantic_scores.parquet'
    if not scores_path.exists():
        scores_path = root / 'item_axis_scores.parquet'
    if not scores_path.exists():
        scores_path = root / 'item_knowledge_scores.parquet'
    if not tx_path.exists() or not scores_path.exists():
        raise FileNotFoundError('Missing transactions or item_semantic_scores/item_axis_scores.')
    tx = pd.read_parquet(tx_path)
    tx['article_id'] = tx['article_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    tx['customer_id'] = tx['customer_id'].astype(str)
    tx['t_dat'] = pd.to_datetime(tx['t_dat'])
    tx['month'] = tx['t_dat'].dt.to_period('M').astype(str)
    scores = pd.read_parquet(scores_path)
    scores['article_id'] = scores['article_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    all_scols = _style_cols(scores, style_dims)
    cols = ['customer_id', 'article_id', 't_dat', 'month'] + ([c for c in ['sales_channel_id', 'price'] if c in tx.columns])
    d = tx[cols].merge(scores[['article_id'] + all_scols], on='article_id', how='left')
    if cust_path.exists():
        cust = pd.read_parquet(cust_path)
        cust['customer_id'] = cust['customer_id'].astype(str)
        ccols = [c for c in ['customer_id','age','club_member_status','fashion_news_frequency'] if c in cust.columns]
        d = d.merge(cust[ccols], on='customer_id', how='left')
    if max_rows and int(max_rows) > 0 and len(d) > int(max_rows):
        d = d.sample(n=int(max_rows), random_state=seed).reset_index(drop=True)
    for sc in all_scols:
        d[sc] = pd.to_numeric(d[sc], errors='coerce').astype('float32')
    return d


def _make_age_groups(df: pd.DataFrame, min_age: int, max_age: int) -> pd.DataFrame:
    out = df.copy()
    if 'age' not in out.columns:
        return out
    out['age'] = pd.to_numeric(out['age'], errors='coerce')
    out = out[(out['age'] >= float(min_age)) & (out['age'] <= float(max_age))].copy()
    bins = [float(min_age), 25, 35, 45, 55, 65, float(max_age) + 1]
    labels = ['16-24','25-34','35-44','45-54','55-64','65+']
    # If min/max are nonstandard, pd.cut still works; labels remain interpretable for defaults.
    out['age_group'] = pd.cut(out['age'], bins=bins, labels=labels, right=False, include_lowest=True)
    return out


def _transaction_weights_for_density(d: pd.DataFrame, mode: str) -> pd.Series:
    mode = str(mode or 'transaction').lower()
    if mode == 'user_balanced' and 'customer_id' in d.columns:
        cnt = d.groupby(['period_group','customer_id'])['article_id'].transform('size').astype(float).replace(0, np.nan)
        return (1.0 / cnt).fillna(0.0)
    return pd.Series(1.0, index=d.index, dtype='float64')


def _binned_kde2d(age: np.ndarray, score: np.ndarray, weight: np.ndarray, *, min_age: int, max_age: int, age_bins: int, score_bins: int, bw_age: float, bw_score: float, device_mode: str = 'auto') -> pd.DataFrame:
    """CUDA-friendly binned 2D KDE over age × axis score.

    This avoids O(N*G) pairwise KDE and does not call OpenBLAS.  We first build a
    weighted histogram, then smooth it with separable Gaussian kernels using
    torch conv2d on CUDA when available.
    """
    ok = np.isfinite(age) & np.isfinite(score) & np.isfinite(weight)
    age = age[ok].astype('float64')
    score = np.clip(score[ok].astype('float64'), 0.0, 1.0)
    weight = np.clip(weight[ok].astype('float64'), 0.0, None)
    if len(age) == 0 or weight.sum() <= 0:
        return pd.DataFrame(columns=['age_mid','score_mid','density'])
    age_edges = np.linspace(float(min_age), float(max_age), int(age_bins) + 1)
    score_edges = np.linspace(0.0, 1.0, int(score_bins) + 1)
    H, _, _ = np.histogram2d(age, score, bins=[age_edges, score_edges], weights=weight)
    if H.sum() <= 0:
        return pd.DataFrame(columns=['age_mid','score_mid','density'])
    H = H / H.sum()
    try:
        import torch
        import torch.nn.functional as F
        dev = _choose_torch_device(device_mode)
        if dev is None:
            raise RuntimeError('torch unavailable')
        x = torch.as_tensor(H, dtype=torch.float32, device=dev)[None, None, :, :]
        age_step = max((float(max_age) - float(min_age)) / max(int(age_bins), 1), 1e-6)
        score_step = 1.0 / max(int(score_bins), 1)
        sig_a = max(float(bw_age) / age_step, 0.5)
        sig_s = max(float(bw_score) / score_step, 0.5)
        def kernel1d(sig):
            radius = int(max(2, math.ceil(4.0 * sig)))
            grid = torch.arange(-radius, radius + 1, dtype=torch.float32, device=dev)
            k = torch.exp(-0.5 * (grid / float(sig)) ** 2)
            return k / k.sum()
        ka = kernel1d(sig_a).view(1, 1, -1, 1)
        ks = kernel1d(sig_s).view(1, 1, 1, -1)
        x = F.pad(x, (0, 0, ka.shape[2] // 2, ka.shape[2] // 2), mode='reflect')
        x = F.conv2d(x, ka)
        x = F.pad(x, (ks.shape[3] // 2, ks.shape[3] // 2, 0, 0), mode='reflect')
        x = F.conv2d(x, ks)
        D = x[0, 0].detach().cpu().numpy()
    except Exception:
        # CPU fallback: small separable convolution implemented in numpy.
        def kernel_np(sig):
            radius = int(max(2, math.ceil(4.0 * sig)))
            grid = np.arange(-radius, radius + 1, dtype='float64')
            k = np.exp(-0.5 * (grid / float(sig)) ** 2)
            return k / k.sum()
        age_step = max((float(max_age) - float(min_age)) / max(int(age_bins), 1), 1e-6)
        score_step = 1.0 / max(int(score_bins), 1)
        ka = kernel_np(max(float(bw_age) / age_step, 0.5))
        ks = kernel_np(max(float(bw_score) / score_step, 0.5))
        D = np.apply_along_axis(lambda v: np.convolve(v, ka, mode='same'), 0, H)
        D = np.apply_along_axis(lambda v: np.convolve(v, ks, mode='same'), 1, D)
    D = np.maximum(D, 0)
    if D.sum() > 0:
        D = D / D.sum()
    age_mid = (age_edges[:-1] + age_edges[1:]) / 2.0
    score_mid = (score_edges[:-1] + score_edges[1:]) / 2.0
    aa, ss = np.meshgrid(age_mid, score_mid, indexing='ij')
    return pd.DataFrame({'age_mid': aa.ravel(), 'score_mid': ss.ravel(), 'density': D.ravel()})


def _torch_diag_gmm_2d(X_np: np.ndarray, w_np: np.ndarray | None = None, n_components: int = 3, n_iter: int = 80, seed: int = 42, device_mode: str = 'auto') -> dict:
    """Small diagonal-covariance weighted GMM in torch, avoiding sklearn/OpenBLAS."""
    ok = np.isfinite(X_np).all(axis=1)
    X_np = X_np[ok].astype('float32')
    if w_np is None:
        w_np = np.ones(len(X_np), dtype='float32')
    else:
        w_np = np.asarray(w_np)[ok].astype('float32')
    w_np = np.clip(w_np, 0, None)
    if len(X_np) < max(20, int(n_components) * 5) or w_np.sum() <= 0:
        return {'ok': False, 'reason': 'insufficient_data'}
    try:
        import torch
        dev = _choose_torch_device(device_mode)
        if dev is None:
            raise RuntimeError('torch unavailable')
        torch.manual_seed(int(seed))
        X = torch.as_tensor(X_np, dtype=torch.float32, device=dev)
        w = torch.as_tensor(w_np / max(w_np.sum(), 1e-12), dtype=torch.float32, device=dev)
        K = int(max(1, min(n_components, len(X_np) // 10)))
        # Weighted random-ish initialization by quantiles along first principal-free dimension.
        idx = torch.linspace(0, len(X) - 1, steps=K).long().to(dev)
        order = torch.argsort(X[:, 0])
        means = X[order[idx]].clone()
        var0 = torch.var(X, dim=0, unbiased=False).clamp_min(1e-4)
        vars_ = var0.repeat(K, 1).clone()
        pis = torch.full((K,), 1.0 / K, device=dev)
        for _ in range(int(n_iter)):
            log_prob = []
            for k in range(K):
                lp = -0.5 * (((X - means[k]) ** 2 / vars_[k]).sum(dim=1) + torch.log(vars_[k]).sum() + 2 * math.log(2 * math.pi)) + torch.log(pis[k].clamp_min(1e-9))
                log_prob.append(lp)
            L = torch.stack(log_prob, dim=1)
            R = torch.softmax(L, dim=1) * w[:, None]
            Nk = R.sum(dim=0).clamp_min(1e-9)
            pis = Nk / Nk.sum()
            means = (R.T @ X) / Nk[:, None]
            for k in range(K):
                diff = X - means[k]
                vars_[k] = ((R[:, k:k+1] * diff * diff).sum(dim=0) / Nk[k]).clamp_min(1e-5)
        return {
            'ok': True,
            'weights': pis.detach().cpu().numpy(),
            'means': means.detach().cpu().numpy(),
            'vars': vars_.detach().cpu().numpy(),
        }
    except Exception as exc:
        return {'ok': False, 'reason': f'{type(exc).__name__}:{exc}'}


def _plot_user_heatmap(mat: pd.DataFrame, path: Path, title: str) -> None:
    if mat.empty:
        return
    _plot_heatmap(mat, path, title, figsize=(max(8, 0.75 * len(mat.columns) + 2), max(4, 0.45 * len(mat.index) + 2)))


def _plot_age_gradient(summary: pd.DataFrame, out: Path, dims: list[str], title: str, value_col: str = 'mean_delta') -> None:
    if summary.empty or 'age_group' not in summary.columns:
        return
    import matplotlib.pyplot as plt
    order = ['16-24','25-34','35-44','45-54','55-64','65+']
    fig, ax = plt.subplots(figsize=(10, 5))
    for dim in dims:
        g = summary[(summary['group_field'] == 'age_group') & (summary['knowledge_dim'] == dim)].copy()
        if g.empty:
            continue
        g['group_value'] = pd.Categorical(g['group_value'], categories=order, ordered=True)
        g = g.sort_values('group_value')
        ax.plot(g['group_value'].astype(str), g[value_col], marker='o', label=dim)
    ax.axhline(0, lw=1)
    ax.set_title(title)
    ax.set_xlabel('Age group')
    ax.set_ylabel(value_col)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def _plot_kde_diff(df: pd.DataFrame, out: Path, title: str) -> None:
    if df.empty:
        return
    import matplotlib.pyplot as plt
    piv = df.pivot(index='score_mid', columns='age_mid', values='density_diff').sort_index(ascending=True)
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    vmax = np.nanmax(np.abs(piv.to_numpy())) if np.isfinite(piv.to_numpy()).any() else 1.0
    im = ax.imshow(piv.to_numpy(), aspect='auto', origin='lower', cmap='coolwarm', vmin=-vmax, vmax=vmax,
                   extent=[float(piv.columns.min()), float(piv.columns.max()), float(piv.index.min()), float(piv.index.max())])
    ax.set_title(title)
    ax.set_xlabel('Age')
    ax.set_ylabel('Axis score')
    fig.colorbar(im, ax=ax, label='COVID density - 2019 matched density')
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def exp_user_heterogeneity(args) -> dict:
    root = Path(args.social_output_root)
    out = ensure_dir(root / 'analysis' / args.exp_id)

    user_dims_arg = getattr(args, 'user_axis_dims', '') or getattr(args, 'style_dims', '')
    max_rows = int(getattr(args, 'max_purchase_rows', 0) or 0)
    d = _load_purchase_level(root, Path(args.data_root), user_dims_arg + ',' + getattr(args, 'user_season_control_dims', ''), max_rows, int(args.seed))
    d = _make_age_groups(d, int(args.min_age), int(args.max_age))
    all_score_cols = _style_cols(d, None)
    scols = _parse_user_dims(user_dims_arg, all_score_cols)
    season_cols = _parse_user_dims(getattr(args, 'user_season_control_dims', ''), all_score_cols)
    if not scols:
        raise RuntimeError('No user heterogeneity axis columns found. Check item_semantic_scores.parquet and --user_axis_dims.')

    covid_end = getattr(args, 'user_covid_end_month', '') or getattr(args, 'task1_covid_end_month', '')
    d, covid_months, base_months = _period_labels_for_user(d, args.event_month, covid_end)
    if d.empty:
        raise RuntimeError('No purchase rows in matched base/COVID windows for user heterogeneity.')

    # ------------------------------------------------------------------
    # A. Group-level purchase-weighted axis shift.
    # ------------------------------------------------------------------
    group_fields = [x.strip() for x in str(getattr(args, 'user_group_fields', 'age_group,club_member_status,fashion_news_frequency')).split(',') if x.strip()]
    group_rows = []
    for gf in group_fields:
        if gf not in d.columns:
            continue
        dd = d.dropna(subset=[gf]).copy()
        if dd.empty:
            continue
        for (grp, period), g in dd.groupby([gf, 'period_group'], dropna=False):
            customers = int(g['customer_id'].nunique()) if 'customer_id' in g.columns else 0
            purchases = int(len(g))
            for sc in scols:
                group_rows.append({
                    'group_field': gf,
                    'group_value': str(grp),
                    'period_group': period,
                    'knowledge_dim': sc.replace('_score',''),
                    'style_share': float(pd.to_numeric(g[sc], errors='coerce').mean()),
                    'purchases': purchases,
                    'customers': customers,
                })
    groups = pd.DataFrame(group_rows)
    groups.to_parquet(out / 'user_group_axis_prepost.parquet', index=False)
    if len(groups):
        shift = groups.pivot_table(index=['group_field','group_value','knowledge_dim'], columns='period_group', values='style_share', aggfunc='first').reset_index()
        if {'base_2019_matched','covid_2020'}.issubset(shift.columns):
            shift['yoy_shift'] = shift['covid_2020'] - shift['base_2019_matched']
        counts = groups.pivot_table(index=['group_field','group_value','knowledge_dim'], columns='period_group', values='purchases', aggfunc='first').reset_index()
        counts = counts.rename(columns={'base_2019_matched':'base_purchases', 'covid_2020':'covid_purchases'})
        cust_counts = groups.pivot_table(index=['group_field','group_value','knowledge_dim'], columns='period_group', values='customers', aggfunc='first').reset_index()
        cust_counts = cust_counts.rename(columns={'base_2019_matched':'base_customers', 'covid_2020':'covid_customers'})
        shift = shift.merge(counts, on=['group_field','group_value','knowledge_dim'], how='left').merge(cust_counts, on=['group_field','group_value','knowledge_dim'], how='left')
        # Relative shift subtracts the all-market matched shift within each axis.
        all_rows = []
        for period, g in d.groupby('period_group'):
            for sc in scols:
                all_rows.append({'period_group': period, 'knowledge_dim': sc.replace('_score',''), 'style_share': float(g[sc].mean())})
        all_df = pd.DataFrame(all_rows).pivot(index='knowledge_dim', columns='period_group', values='style_share')
        if {'base_2019_matched','covid_2020'}.issubset(all_df.columns):
            all_shift = (all_df['covid_2020'] - all_df['base_2019_matched']).rename('overall_yoy_shift').reset_index()
            shift = shift.merge(all_shift, on='knowledge_dim', how='left')
            shift['relative_yoy_shift'] = shift['yoy_shift'] - shift['overall_yoy_shift']
        shift.to_parquet(out / 'user_group_axis_shift.parquet', index=False)
    else:
        shift = pd.DataFrame(); shift.to_parquet(out / 'user_group_axis_shift.parquet', index=False)

    # ------------------------------------------------------------------
    # B. User-level preference-shift heterogeneity.
    # Each user contributes one base/covid average per axis, then one delta.
    # ------------------------------------------------------------------
    user_cols = ['customer_id', 'period_group'] + scols
    meta_cols = [c for c in ['age','age_group','club_member_status','fashion_news_frequency'] if c in d.columns]
    user_period = d[user_cols + meta_cols].groupby(['customer_id','period_group'], as_index=False).agg(
        {**{sc: 'mean' for sc in scols}, **{c: 'first' for c in meta_cols}}
    )
    user_counts = d.groupby(['customer_id','period_group'], as_index=False).size().rename(columns={'size':'purchases'})
    user_period = user_period.merge(user_counts, on=['customer_id','period_group'], how='left')
    user_period.to_parquet(out / 'user_period_axis_profile.parquet', index=False)
    user_delta = None
    delta_frames = []
    base = user_period[user_period['period_group']=='base_2019_matched'].set_index('customer_id')
    cov = user_period[user_period['period_group']=='covid_2020'].set_index('customer_id')
    common = base.index.intersection(cov.index)
    if len(common):
        rows = []
        for uid in common:
            row = {'customer_id': uid}
            for c in meta_cols:
                row[c] = cov.loc[uid, c] if pd.notna(cov.loc[uid, c]) else base.loc[uid, c]
            row['base_purchases'] = int(base.loc[uid, 'purchases'])
            row['covid_purchases'] = int(cov.loc[uid, 'purchases'])
            for sc in scols:
                row[sc.replace('_score','') + '_base'] = float(base.loc[uid, sc])
                row[sc.replace('_score','') + '_covid'] = float(cov.loc[uid, sc])
                row[sc.replace('_score','') + '_delta'] = float(cov.loc[uid, sc] - base.loc[uid, sc])
            rows.append(row)
        user_delta = pd.DataFrame(rows)
    else:
        user_delta = pd.DataFrame()
    user_delta.to_parquet(out / 'user_level_axis_shift.parquet', index=False)

    user_group_summary_rows = []
    if len(user_delta):
        for gf in group_fields:
            if gf not in user_delta.columns:
                continue
            for grp, g in user_delta.dropna(subset=[gf]).groupby(gf, dropna=False):
                for sc in scols:
                    dim = sc.replace('_score','')
                    col = f'{dim}_delta'
                    vals = pd.to_numeric(g[col], errors='coerce').dropna()
                    if len(vals) == 0:
                        continue
                    user_group_summary_rows.append({
                        'group_field': gf,
                        'group_value': str(grp),
                        'knowledge_dim': dim,
                        'mean_delta': float(vals.mean()),
                        'median_delta': float(vals.median()),
                        'std_delta': float(vals.std(ddof=0)),
                        'users': int(len(vals)),
                        'mean_base_purchases': float(g['base_purchases'].mean()),
                        'mean_covid_purchases': float(g['covid_purchases'].mean()),
                    })
    user_summary = pd.DataFrame(user_group_summary_rows)
    user_summary.to_parquet(out / 'user_level_group_axis_shift.parquet', index=False)

    # ------------------------------------------------------------------
    # C. Transaction-level age × axis density: purchase-weighted or user-balanced.
    # ------------------------------------------------------------------
    kde_rows = []
    gmm_rows = []
    kde_dims = _parse_user_dims(getattr(args, 'user_kde_dims', ''), all_score_cols) or scols[:6]
    device_mode = getattr(args, 'user_use_cuda', 'auto')
    txn_weight = _transaction_weights_for_density(d, getattr(args, 'user_txn_weighting', 'user_balanced'))
    if bool(getattr(args, 'user_run_transaction_kde', True)) and 'age' in d.columns:
        for sc in kde_dims:
            dens_by_period = {}
            for period, g in d.dropna(subset=['age']).groupby('period_group'):
                w = txn_weight.loc[g.index].to_numpy(dtype='float64')
                dens = _binned_kde2d(
                    g['age'].to_numpy(dtype='float64'),
                    pd.to_numeric(g[sc], errors='coerce').to_numpy(dtype='float64'),
                    w,
                    min_age=int(args.min_age), max_age=int(args.max_age),
                    age_bins=int(getattr(args, 'user_kde_age_bins', 65)),
                    score_bins=int(getattr(args, 'user_kde_score_bins', 60)),
                    bw_age=float(getattr(args, 'user_kde_bandwidth_age', 2.5)),
                    bw_score=float(getattr(args, 'user_kde_bandwidth_score', 0.035)),
                    device_mode=device_mode,
                )
                dens['knowledge_dim'] = sc.replace('_score','')
                dens['period_group'] = period
                dens_by_period[period] = dens
                kde_rows.append(dens)
            if {'base_2019_matched','covid_2020'}.issubset(dens_by_period.keys()):
                b = dens_by_period['base_2019_matched'][['age_mid','score_mid','density']].rename(columns={'density':'density_base'})
                c = dens_by_period['covid_2020'][['age_mid','score_mid','density']].rename(columns={'density':'density_covid'})
                diff = b.merge(c, on=['age_mid','score_mid'], how='outer').fillna(0.0)
                diff['density_diff'] = diff['density_covid'] - diff['density_base']
                diff['knowledge_dim'] = sc.replace('_score','')
                if bool(getattr(args, 'make_figures', True)):
                    _plot_kde_diff(diff, out / f'transaction_age_axis_kde_diff_{_safe_name(sc.replace("_score", ""))}.png', f'COVID - 2019 matched purchase density: {sc.replace("_score", "")}')
    kde = pd.concat(kde_rows, ignore_index=True) if kde_rows else pd.DataFrame()
    kde.to_parquet(out / 'transaction_age_axis_kde.parquet', index=False)

    # Transaction-level torch GMM over (age, axis score), by period.
    if bool(getattr(args, 'user_run_transaction_gmm', True)) and 'age' in d.columns:
        for sc in kde_dims:
            for period, g in d.dropna(subset=['age']).groupby('period_group'):
                X = np.column_stack([
                    pd.to_numeric(g['age'], errors='coerce').to_numpy(dtype='float64'),
                    pd.to_numeric(g[sc], errors='coerce').to_numpy(dtype='float64'),
                ])
                w = txn_weight.loc[g.index].to_numpy(dtype='float64')
                # Standardize for EM stability, save means back on original scale.
                mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0); sd[sd <= 1e-8] = 1.0
                fit = _torch_diag_gmm_2d((X - mu) / sd, w, int(args.gmm_components), int(getattr(args, 'user_gmm_iter', 80)), int(args.seed), device_mode)
                if fit.get('ok'):
                    means = fit['means'] * sd + mu
                    vars_ = fit['vars'] * (sd ** 2)
                    for comp, pi in enumerate(fit['weights']):
                        gmm_rows.append({'level':'transaction', 'knowledge_dim': sc.replace('_score',''), 'period_group': period, 'component': comp, 'weight': float(pi), 'mean_age': float(means[comp,0]), 'mean_axis_score': float(means[comp,1]), 'sd_age': float(np.sqrt(vars_[comp,0])), 'sd_axis_score': float(np.sqrt(vars_[comp,1]))})
                else:
                    gmm_rows.append({'level':'transaction', 'knowledge_dim': sc.replace('_score',''), 'period_group': period, 'component': -1, 'status': fit.get('reason','failed')})

    # User-level torch GMM over (age, user-level axis delta).
    if bool(getattr(args, 'user_run_user_gmm', True)) and len(user_delta) and 'age' in user_delta.columns:
        for sc in kde_dims:
            dim = sc.replace('_score','')
            col = f'{dim}_delta'
            if col not in user_delta.columns:
                continue
            g = user_delta.dropna(subset=['age', col])
            if g.empty:
                continue
            X = np.column_stack([g['age'].to_numpy(dtype='float64'), g[col].to_numpy(dtype='float64')])
            mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0); sd[sd <= 1e-8] = 1.0
            fit = _torch_diag_gmm_2d((X - mu) / sd, np.ones(len(g)), int(args.gmm_components), int(getattr(args, 'user_gmm_iter', 80)), int(args.seed), device_mode)
            if fit.get('ok'):
                means = fit['means'] * sd + mu
                vars_ = fit['vars'] * (sd ** 2)
                for comp, pi in enumerate(fit['weights']):
                    gmm_rows.append({'level':'user_delta', 'knowledge_dim': dim, 'period_group': 'covid_minus_base', 'component': comp, 'weight': float(pi), 'mean_age': float(means[comp,0]), 'mean_axis_delta': float(means[comp,1]), 'sd_age': float(np.sqrt(vars_[comp,0])), 'sd_axis_delta': float(np.sqrt(vars_[comp,1]))})
            else:
                gmm_rows.append({'level':'user_delta', 'knowledge_dim': dim, 'period_group': 'covid_minus_base', 'component': -1, 'status': fit.get('reason','failed')})
    gmm = pd.DataFrame(gmm_rows)
    gmm.to_parquet(out / 'age_axis_gmm_results.parquet', index=False)

    # ------------------------------------------------------------------
    # Figures.
    # ------------------------------------------------------------------
    if bool(getattr(args, 'make_figures', True)):
        if len(shift) and 'yoy_shift' in shift.columns:
            for gf in shift['group_field'].dropna().unique():
                sub = shift[shift['group_field'] == gf]
                mat = sub.pivot(index='group_value', columns='knowledge_dim', values='yoy_shift')
                _plot_user_heatmap(mat, out / f'{_safe_name(gf)}_axis_yoy_shift_heatmap.png', f'{gf}: axis YoY shift')
                if 'relative_yoy_shift' in sub.columns:
                    mat_rel = sub.pivot(index='group_value', columns='knowledge_dim', values='relative_yoy_shift')
                    _plot_user_heatmap(mat_rel, out / f'{_safe_name(gf)}_axis_relative_shift_heatmap.png', f'{gf}: relative axis shift vs overall')
        if len(user_summary):
            plot_dims = [x.replace('_score','') for x in kde_dims[:min(6, len(kde_dims))]]
            _plot_age_gradient(user_summary, out / 'user_level_age_gradient_axis_shift.png', plot_dims, 'User-level age gradient of COVID style shift', 'mean_delta')

    manifest = {
        'purchase_rows_used': int(len(d)),
        'score_dims': [c.replace('_score','') for c in scols],
        'kde_dims': [c.replace('_score','') for c in kde_dims],
        'covid_months': covid_months,
        'base_months': base_months,
        'group_shift_rows': int(len(shift)),
        'user_delta_rows': int(len(user_delta)),
        'transaction_kde_rows': int(len(kde)),
        'gmm_rows': int(len(gmm)),
        'estimands': {
            'group_level': 'purchase-weighted group axis shift, with relative shift subtracting overall market shift',
            'user_level': 'one user contributes one base/COVID average and one axis delta',
            'transaction_level_kde': f'{getattr(args, "user_txn_weighting", "user_balanced")} transaction-level age × axis consumption density',
        },
    }
    save_json(manifest, out / 'user_heterogeneity_manifest.json')
    return manifest



def _parse_dim_names(dims: str | Iterable[str] | None, fallback: list[str] | None = None) -> list[str]:
    if dims:
        vals = [str(x).strip().replace('_score', '') for x in str(dims).split(',') if str(x).strip()]
    else:
        vals = list(fallback or [])
    out = []
    for v in vals:
        if v and v not in out:
            out.append(v)
    return out


def _infer_count_col(df: pd.DataFrame) -> str | None:
    for c in ['sales_count', 'transaction_count', 'purchase_count', 'n_transactions', 'count', 'cnt', 'volume', 'quantity']:
        if c in df.columns:
            return c
    return None


def _sort_monthly_panel(df: pd.DataFrame, table_name: str = 'panel') -> pd.DataFrame:
    out = _ensure_month_column(df, table_name=table_name).copy()
    out['_month_period'] = pd.PeriodIndex(out['month'].astype(str), freq='M')
    sort_cols = ['_month_period']
    for c in ['sales_channel_id', 'knowledge_dim']:
        if c in out.columns:
            sort_cols.append(c)
    out = out.sort_values(sort_cols).reset_index(drop=True)
    out['month'] = out['_month_period'].astype(str)
    return out


def _complete_channel_share_panel(ch: pd.DataFrame, fill_missing: bool = True) -> pd.DataFrame:
    """Sort channel-share panel and optionally add explicit zero-share rows for
    missing channel×month combinations.  This avoids matplotlib categorical-axis
    artifacts where a month present only in one channel is appended to the end of
    the line chart.
    """
    if len(ch) == 0:
        return ch.copy()
    out = _sort_monthly_panel(ch, table_name='channel_monthly_panel')
    if 'sales_channel_id' not in out.columns:
        return out.drop(columns=['_month_period'], errors='ignore')
    if 'channel_share' in out.columns:
        out['channel_share'] = pd.to_numeric(out['channel_share'], errors='coerce')
    count_col = _infer_count_col(out)
    if fill_missing and 'channel_share' in out.columns:
        months = pd.period_range(out['_month_period'].min(), out['_month_period'].max(), freq='M')
        channels = sorted(out['sales_channel_id'].dropna().unique().tolist())
        idx = pd.MultiIndex.from_product([months, channels], names=['_month_period', 'sales_channel_id'])
        base_cols = ['_month_period', 'sales_channel_id']
        keep_cols = [c for c in out.columns if c not in base_cols]
        tmp = out.set_index(base_cols)[keep_cols]
        # If duplicated rows exist, aggregate before reindexing.
        if not tmp.index.is_unique:
            agg = {'channel_share': 'sum'} if 'channel_share' in tmp.columns else {}
            if count_col:
                agg[count_col] = 'sum'
            other = [c for c in tmp.columns if c not in agg]
            for c in other:
                agg[c] = 'first'
            tmp = out.groupby(base_cols, as_index=True).agg(agg)
        tmp = tmp.reindex(idx).reset_index()
        tmp['month'] = tmp['_month_period'].astype(str)
        tmp['channel_missing_filled'] = tmp['channel_share'].isna()
        tmp['channel_share'] = tmp['channel_share'].fillna(0.0)
        if count_col and count_col in tmp.columns:
            tmp[count_col] = pd.to_numeric(tmp[count_col], errors='coerce').fillna(0.0)
        out = tmp
    out = out.sort_values(['_month_period', 'sales_channel_id']).reset_index(drop=True)
    out['month'] = out['_month_period'].astype(str)
    return out.drop(columns=['_month_period'], errors='ignore')


def _channel_share_quality(ch: pd.DataFrame) -> pd.DataFrame:
    if len(ch) == 0 or 'sales_channel_id' not in ch.columns:
        return pd.DataFrame()
    d = _sort_monthly_panel(ch, table_name='channel_monthly_panel')
    count_col = _infer_count_col(d)
    rows = []
    for m, g in d.groupby('_month_period'):
        share = pd.to_numeric(g.get('channel_share', pd.Series(dtype=float)), errors='coerce') if 'channel_share' in g.columns else pd.Series(dtype=float)
        row = {
            'month': str(m),
            'n_channels_observed': int(g['sales_channel_id'].nunique()),
            'share_sum': float(share.sum()) if len(share) else np.nan,
            'share_min': float(share.min()) if len(share) else np.nan,
            'share_max': float(share.max()) if len(share) else np.nan,
            'share_sum_abs_error': float(abs(share.sum() - 1.0)) if len(share) else np.nan,
        }
        if count_col:
            cnt = pd.to_numeric(g[count_col], errors='coerce').fillna(0.0)
            row['transaction_count_sum'] = float(cnt.sum())
            row['transaction_count_min_channel'] = float(cnt.min()) if len(cnt) else np.nan
        rows.append(row)
    q = pd.DataFrame(rows)
    if len(q):
        q['is_share_sum_bad'] = q['share_sum_abs_error'] > 1e-3
        if 'transaction_count_sum' in q.columns:
            q['is_low_count_month'] = q['transaction_count_sum'] < q['transaction_count_sum'].median() * 0.05
    return q


def _plot_month_lines(df: pd.DataFrame, x: str, y: str, group: str, path: Path, title: str = '') -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    d = _sort_monthly_panel(df, table_name='line_panel')
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, g in d.groupby(group):
        g = g.sort_values('_month_period')
        xx = g['_month_period'].dt.to_timestamp()
        ax.plot(xx, pd.to_numeric(g[y], errors='coerce'), marker='o', label=str(key))
    ax.set_title(title)
    ax.set_xlabel(x); ax.set_ylabel(y)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_centered_heatmap(mat: pd.DataFrame, path: Path, title: str = '', figsize=(9, 4.5)) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    arr = mat.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    fig, ax = plt.subplots(figsize=figsize)
    if len(finite):
        vmax = float(np.nanmax(np.abs(finite)))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax) if vmax > 0 else None
        im = ax.imshow(arr, aspect='auto', cmap='coolwarm', norm=norm)
    else:
        im = ax.imshow(arr, aspect='auto', cmap='coolwarm')
    ax.set_xticks(np.arange(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=45, ha='right')
    ax.set_yticks(np.arange(mat.shape[0])); ax.set_yticklabels(mat.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _label_channel_periods(df: pd.DataFrame, event_month: str, covid_end_month: str = '') -> tuple[pd.DataFrame, list[str], list[str]]:
    out = _sort_monthly_panel(df, table_name='channel_panel')
    months = pd.PeriodIndex(out['month'].astype(str), freq='M')
    ev = pd.Period(event_month, freq='M')
    end = pd.Period(covid_end_month, freq='M') if covid_end_month else months.max()
    covid_periods = list(pd.period_range(ev, end, freq='M'))
    base_periods = [p - 12 for p in covid_periods]
    period_map = {str(p): 'covid_2020' for p in covid_periods}
    period_map.update({str(p): 'base_2019_matched' for p in base_periods})
    out['period_group'] = out['month'].astype(str).map(period_map)
    out = out.dropna(subset=['period_group']).copy()
    out['month'] = pd.PeriodIndex(out['month'].astype(str), freq='M').astype(str)
    return out.drop(columns=['_month_period'], errors='ignore'), [str(p) for p in covid_periods], [str(p) for p in base_periods]


def _channel_style_yoy_shift(cs: pd.DataFrame, event_month: str, covid_end_month: str, dims: list[str]) -> pd.DataFrame:
    if len(cs) == 0:
        return pd.DataFrame()
    d, covid_months, base_months = _label_channel_periods(cs, event_month, covid_end_month)
    if dims:
        d = d[d['knowledge_dim'].astype(str).isin(dims)].copy()
    if len(d) == 0:
        return pd.DataFrame()
    d['style_share'] = pd.to_numeric(d['style_share'], errors='coerce')
    avg = d.groupby(['sales_channel_id', 'knowledge_dim', 'period_group'], as_index=False)['style_share'].mean()
    pvt = avg.pivot_table(index=['sales_channel_id', 'knowledge_dim'], columns='period_group', values='style_share').reset_index()
    if {'covid_2020', 'base_2019_matched'}.issubset(pvt.columns):
        pvt['yoy_shift'] = pvt['covid_2020'] - pvt['base_2019_matched']
    pvt['covid_months'] = ','.join(covid_months)
    pvt['base_months'] = ','.join(base_months)
    return pvt


def _channel_axis_decomposition(ch: pd.DataFrame, cs: pd.DataFrame, event_month: str, covid_end_month: str, dims: list[str]) -> pd.DataFrame:
    """Decompose aggregate axis shift into within-channel style shift,
    channel-mix shift, and an interaction term:

      ΔA = Σ s_base (A_covid - A_base)
         + Σ (s_covid - s_base) A_base
         + Σ (s_covid - s_base)(A_covid - A_base)
    """
    if len(ch) == 0 or len(cs) == 0:
        return pd.DataFrame()
    ch_l, covid_months, base_months = _label_channel_periods(ch, event_month, covid_end_month)
    cs_l, _, _ = _label_channel_periods(cs, event_month, covid_end_month)
    if dims:
        cs_l = cs_l[cs_l['knowledge_dim'].astype(str).isin(dims)].copy()
    if len(ch_l) == 0 or len(cs_l) == 0:
        return pd.DataFrame()
    ch_l['channel_share'] = pd.to_numeric(ch_l['channel_share'], errors='coerce')
    cs_l['style_share'] = pd.to_numeric(cs_l['style_share'], errors='coerce')
    s = ch_l.groupby(['sales_channel_id', 'period_group'], as_index=False)['channel_share'].mean()
    a = cs_l.groupby(['sales_channel_id', 'knowledge_dim', 'period_group'], as_index=False)['style_share'].mean()
    sp = s.pivot_table(index='sales_channel_id', columns='period_group', values='channel_share')
    ap = a.pivot_table(index=['sales_channel_id', 'knowledge_dim'], columns='period_group', values='style_share').reset_index()
    rows = []
    for dim, g in ap.groupby('knowledge_dim'):
        within = mix = interaction = total_base = total_covid = 0.0
        used = 0
        for _, r in g.iterrows():
            c = r['sales_channel_id']
            if c not in sp.index:
                continue
            sb = sp.loc[c].get('base_2019_matched', np.nan)
            scv = sp.loc[c].get('covid_2020', np.nan)
            ab = r.get('base_2019_matched', np.nan)
            acv = r.get('covid_2020', np.nan)
            if not all(np.isfinite(x) for x in [sb, scv, ab, acv]):
                continue
            ds = float(scv - sb); da = float(acv - ab)
            within += float(sb * da)
            mix += float(ds * ab)
            interaction += float(ds * da)
            total_base += float(sb * ab)
            total_covid += float(scv * acv)
            used += 1
        if used:
            rows.append({
                'knowledge_dim': dim,
                'within_channel_effect': within,
                'channel_mix_effect': mix,
                'interaction_effect': interaction,
                'total_decomposed_shift': within + mix + interaction,
                'total_direct_shift': total_covid - total_base,
                'decomposition_error': (within + mix + interaction) - (total_covid - total_base),
                'n_channels_used': used,
                'covid_months': ','.join(covid_months),
                'base_months': ','.join(base_months),
            })
    return pd.DataFrame(rows)


def _plot_channel_decomposition(dec: pd.DataFrame, path: Path, title: str = '') -> None:
    import matplotlib.pyplot as plt
    if len(dec) == 0:
        return
    d = dec.copy()
    d['_abs_total'] = d['total_direct_shift'].abs()
    d = d.sort_values('_abs_total', ascending=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(d) + 1.5)))
    left = np.zeros(len(d), dtype=float)
    for col, lab in [
        ('within_channel_effect', 'within-channel'),
        ('channel_mix_effect', 'channel-mix'),
        ('interaction_effect', 'interaction'),
    ]:
        vals = pd.to_numeric(d[col], errors='coerce').fillna(0.0).to_numpy()
        # Stacked bars with mixed signs are hard to read if stacked naively.
        # Plot as grouped thin bars around each axis instead.
    offsets = {'within_channel_effect': -0.22, 'channel_mix_effect': 0.0, 'interaction_effect': 0.22}
    labels = {'within_channel_effect': 'within-channel', 'channel_mix_effect': 'channel-mix', 'interaction_effect': 'interaction'}
    for col, off in offsets.items():
        ax.barh(y + off, d[col].astype(float), height=0.18, label=labels[col])
    ax.axvline(0, lw=1)
    ax.set_yticks(y); ax.set_yticklabels(d['knowledge_dim'])
    ax.set_xlabel('Matched COVID - 2019 axis-share contribution')
    ax.set_title(title or 'Channel-mediated axis-shift decomposition')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)



def _load_channel_transactions(data_root: Path, min_age: int = 16, max_age: int = 80) -> pd.DataFrame:
    """Load raw H&M transactions for channel-migration analysis.

    This task deliberately does not join item style/axis scores.  It estimates
    pandemic-induced channel substitution from purchase records only, with
    optional customer fields for heterogeneity analysis.
    """
    tx_path = data_root / 'hm' / 'processed' / 'hm_transactions.parquet'
    cust_path = data_root / 'hm' / 'processed' / 'hm_customers.parquet'
    if not tx_path.exists():
        raise FileNotFoundError(f'Missing transactions file: {tx_path}')
    cols = None
    tx = pd.read_parquet(tx_path)
    if 'sales_channel_id' not in tx.columns:
        raise KeyError(f'hm_transactions.parquet lacks sales_channel_id. Available columns: {list(tx.columns)}')
    if 't_dat' not in tx.columns:
        raise KeyError(f'hm_transactions.parquet lacks t_dat. Available columns: {list(tx.columns)}')
    keep = [c for c in ['customer_id','article_id','t_dat','sales_channel_id','price'] if c in tx.columns]
    d = tx[keep].copy()
    if 'customer_id' in d.columns:
        d['customer_id'] = d['customer_id'].astype(str)
    if 'article_id' in d.columns:
        d['article_id'] = d['article_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    d['date'] = pd.to_datetime(d['t_dat'], errors='coerce')
    d = d.dropna(subset=['date', 'sales_channel_id']).copy()
    d['month'] = d['date'].dt.to_period('M').astype(str)
    # Keep the original channel label for display, but create a stable string key
    # so CLI values such as --channel_target_id 2 match either integer 2 or string "2".
    d['sales_channel_id'] = d['sales_channel_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    if cust_path.exists() and 'customer_id' in d.columns:
        cust = pd.read_parquet(cust_path)
        cust['customer_id'] = cust['customer_id'].astype(str)
        ccols = [c for c in ['customer_id','age','club_member_status','fashion_news_frequency'] if c in cust.columns]
        d = d.merge(cust[ccols], on='customer_id', how='left')
    if 'age' in d.columns:
        d = _make_age_groups(d, int(min_age), int(max_age))
    return d


def _filter_channel_dates(d: pd.DataFrame, start_date: str = '', end_date: str = '', exclude_months: str = '') -> pd.DataFrame:
    out = d.copy()
    if start_date:
        out = out[out['date'] >= pd.to_datetime(start_date)].copy()
    if end_date:
        out = out[out['date'] <= pd.to_datetime(end_date)].copy()
    ex = {x.strip() for x in str(exclude_months or '').split(',') if x.strip()}
    if ex:
        out = out[~out['month'].astype(str).isin(ex)].copy()
    return out


def _channel_daily_panel_from_transactions(d: pd.DataFrame, target_channel: str, start_date: str = '', end_date: str = '') -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame()
    g = d.groupby(['date','sales_channel_id'], as_index=False).size().rename(columns={'size':'transactions'})
    p = g.pivot_table(index='date', columns='sales_channel_id', values='transactions', aggfunc='sum', fill_value=0.0)
    # Complete daily index so missing days are visible as zero transactions.
    s = pd.to_datetime(start_date) if start_date else p.index.min()
    e = pd.to_datetime(end_date) if end_date else p.index.max()
    if pd.isna(s) or pd.isna(e):
        return pd.DataFrame()
    p = p.reindex(pd.date_range(s, e, freq='D'), fill_value=0.0)
    p.index.name = 'date'
    total = p.sum(axis=1).astype(float)
    target = p[target_channel].astype(float) if target_channel in p.columns else pd.Series(0.0, index=p.index)
    out = pd.DataFrame({'date': p.index, 'target_channel_count': target.to_numpy(dtype=float), 'total_transactions': total.to_numpy(dtype=float)})
    out['target_channel_share'] = out['target_channel_count'] / out['total_transactions'].replace(0.0, np.nan)
    for c in p.columns:
        out[f'channel_{_safe_name(c)}_count'] = p[c].to_numpy(dtype=float)
        out[f'channel_{_safe_name(c)}_share'] = p[c].to_numpy(dtype=float) / out['total_transactions'].replace(0.0, np.nan)
    out['month'] = out['date'].dt.to_period('M').astype(str)
    out['dow'] = out['date'].dt.dayofweek.astype(str)
    out['_trend'] = (out['date'] - out['date'].min()).dt.days.astype(float)
    return out


def _channel_monthly_panel_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    count_cols = [c for c in daily.columns if c.startswith('channel_') and c.endswith('_count')]
    agg = daily.groupby('month', as_index=False).agg(
        target_channel_count=('target_channel_count','sum'),
        total_transactions=('total_transactions','sum'),
        active_days=('total_transactions', lambda x: int((pd.to_numeric(x, errors='coerce') > 0).sum())),
        calendar_days=('date','size'),
        **{c: (c, 'sum') for c in count_cols}
    )
    agg['target_channel_share'] = agg['target_channel_count'] / agg['total_transactions'].replace(0.0, np.nan)
    for c in count_cols:
        share_col = c.replace('_count','_share')
        agg[share_col] = agg[c] / agg['total_transactions'].replace(0.0, np.nan)
    agg['_month_period'] = pd.PeriodIndex(agg['month'].astype(str), freq='M')
    agg = agg.sort_values('_month_period').drop(columns=['_month_period']).reset_index(drop=True)
    return agg


def _channel_matched_yoy(monthly: pd.DataFrame, event_month: str, covid_end_month: str = '') -> tuple[pd.DataFrame, list[str], list[str]]:
    if monthly.empty:
        return pd.DataFrame(), [], []
    m = monthly.copy()
    p = pd.PeriodIndex(m['month'].astype(str), freq='M')
    m['_period'] = p
    ev = pd.Period(event_month, freq='M')
    end = pd.Period(covid_end_month, freq='M') if str(covid_end_month or '').strip() else p.max()
    covid_periods = list(pd.period_range(ev, end, freq='M'))
    base_periods = [x - 12 for x in covid_periods]
    base = m[m['_period'].isin(base_periods)].copy()
    cov = m[m['_period'].isin(covid_periods)].copy()
    base['_match_month'] = (base['_period'] + 12).astype(str)
    cov['_match_month'] = cov['_period'].astype(str)
    cols = ['target_channel_share','target_channel_count','total_transactions','active_days','calendar_days']
    keep = ['_match_month','month'] + [c for c in cols if c in m.columns]
    b = base[keep].rename(columns={c: f'base_{c}' for c in keep if c not in ['_match_month','month']}).rename(columns={'month':'base_month'})
    c = cov[keep].rename(columns={c: f'covid_{c}' for c in keep if c not in ['_match_month','month']}).rename(columns={'month':'covid_month'})
    out = c.merge(b, on='_match_month', how='left')
    out = out.rename(columns={'_match_month':'month'})
    if {'covid_target_channel_share','base_target_channel_share'}.issubset(out.columns):
        out['yoy_shift'] = out['covid_target_channel_share'] - out['base_target_channel_share']
        out['relative_yoy_shift_pct'] = out['yoy_shift'] / out['base_target_channel_share'].replace(0.0, np.nan)
    if {'covid_target_channel_count','base_target_channel_count'}.issubset(out.columns):
        out['target_count_yoy_change'] = out['covid_target_channel_count'] - out['base_target_channel_count']
    if {'covid_total_transactions','base_total_transactions'}.issubset(out.columns):
        out['total_count_yoy_change'] = out['covid_total_transactions'] - out['base_total_transactions']
    out['rel_month'] = [pd.Period(x, freq='M').ordinal - ev.ordinal for x in out['month']]
    return out.sort_values('month').reset_index(drop=True), [str(x) for x in covid_periods], [str(x) for x in base_periods]


def _plot_channel_share_and_counts(monthly: pd.DataFrame, path: Path, target_channel: str) -> None:
    import matplotlib.pyplot as plt
    if monthly.empty:
        return
    d = monthly.copy()
    d['_p'] = pd.PeriodIndex(d['month'].astype(str), freq='M')
    d = d.sort_values('_p')
    x = d['_p'].dt.to_timestamp()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(x, d['target_channel_share'], marker='o')
    axes[0].set_ylabel(f'channel {target_channel} share')
    axes[0].set_title('Sales-channel migration')
    count_cols = [c for c in d.columns if c.startswith('channel_') and c.endswith('_count')]
    for c in count_cols:
        lab = c[len('channel_'):-len('_count')]
        axes[1].plot(x, d[c], marker='o', label=lab)
    axes[1].set_ylabel('monthly transactions')
    axes[1].set_xlabel('month')
    axes[1].legend(title='channel', fontsize=8)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_channel_yoy(yoy: pd.DataFrame, path: Path, event_month: str, target_channel: str) -> None:
    import matplotlib.pyplot as plt
    if yoy.empty or 'yoy_shift' not in yoy.columns:
        return
    d = yoy.copy()
    d['_p'] = pd.PeriodIndex(d['month'].astype(str), freq='M')
    d = d.sort_values('_p')
    x = d['_p'].dt.to_timestamp()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhline(0, lw=1)
    ax.axvline(pd.Period(event_month, freq='M').to_timestamp(), linestyle='--', lw=1)
    ax.plot(x, d['yoy_shift'], marker='o')
    ax.set_title(f'Matched YoY shift in channel {target_channel} share')
    ax.set_ylabel('2020 share - matched 2019 share')
    ax.set_xlabel('month')
    for _, r in d.iterrows():
        if pd.notna(r.get('yoy_shift')):
            ax.annotate(f"{r['yoy_shift']:.2f}", (pd.Period(r['month'], freq='M').to_timestamp(), r['yoy_shift']), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=7)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _channel_daily_with_covid(args, daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return daily.copy()
    start = str(getattr(args, 'channel_regression_start_date', '') or daily['date'].min().date())
    end = str(getattr(args, 'channel_regression_end_date', '') or daily['date'].max().date())
    d = daily[(daily['date'] >= pd.to_datetime(start)) & (daily['date'] <= pd.to_datetime(end))].copy()
    covid = _daily_covid_panel(_resolve_covid_csv(args), str(getattr(args, 'covid_location', 'World')), start, end, transform=str(getattr(args, 'channel_covid_transform', 'delta7')))
    if not covid.empty:
        d = d.merge(covid, on='date', how='left')
    eps = float(getattr(args, 'channel_logit_eps', 1e-4))
    s = pd.to_numeric(d['target_channel_share'], errors='coerce').clip(lower=eps, upper=1.0-eps)
    d['logit_target_channel_share'] = np.log(s / (1.0 - s))
    d['lag1_logit_target_channel_share'] = d['logit_target_channel_share'].shift(1)
    d['log_total_transactions'] = np.log1p(pd.to_numeric(d['total_transactions'], errors='coerce').fillna(0.0))
    d['_trend2'] = d['_trend'] ** 2
    return d


def _channel_covid_regression(args, panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    requested = _parse_dim_list(getattr(args, 'channel_covid_vars', 'covid_cases_index,covid_deaths_index,covid_reproduction_rate,covid_stringency_index'))
    rows = []
    for base in requested:
        candidates = [f'{base}_shock', base]
        cv = next((c for c in candidates if c in panel.columns), None)
        if cv is None:
            continue
        d = panel.copy()
        # Drop zero-transaction days for regression because share is undefined.
        d = d[pd.to_numeric(d['total_transactions'], errors='coerce') > 0].copy()
        xcols = [cv]
        if bool(getattr(args, 'channel_regression_lagged_share', True)):
            xcols.append('lag1_logit_target_channel_share')
        if bool(getattr(args, 'channel_control_transactions', True)):
            xcols.append('log_total_transactions')
        fe = []
        if bool(getattr(args, 'channel_control_dow', True)):
            fe.append('dow')
        if bool(getattr(args, 'channel_control_trend', True)):
            xcols.extend(['_trend','_trend2'])
        res = _ols(d, 'logit_target_channel_share', xcols, fe_cols=fe, min_n=int(getattr(args, 'channel_min_daily_obs', 30)), cov_type='hac', hac_lags=int(getattr(args, 'channel_hac_lags_daily', 7)))
        rows.append({
            'covid_var': base,
            'regressor_used': cv,
            'coef': res.get(f'coef_{cv}', np.nan),
            'se': res.get(f'se_{cv}', np.nan),
            't': res.get(f't_{cv}', np.nan),
            'p': res.get(f'p_{cv}', np.nan),
            'nobs': res.get('nobs', 0),
            'r2': res.get('r2', np.nan),
            'status': res.get('status', 'unknown'),
            'outcome': 'logit_target_channel_share',
            'cov_type': res.get('cov_type', 'hac'),
        })
    return pd.DataFrame(rows)


def _channel_lag_bin_regression(args, panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    requested = _parse_dim_list(getattr(args, 'channel_covid_vars', 'covid_cases_index,covid_deaths_index,covid_reproduction_rate,covid_stringency_index'))
    bins = _parse_lag_bins(getattr(args, 'channel_lag_bins', '0:3,4:7,8:14,15:28'))
    rows = []
    for base in requested:
        cv = f'{base}_shock' if f'{base}_shock' in panel.columns else (base if base in panel.columns else None)
        if cv is None:
            continue
        d = panel.copy()
        d = d[pd.to_numeric(d['total_transactions'], errors='coerce') > 0].copy()
        xcols = []
        for lab, lo, hi in bins:
            cols = []
            for lag in range(lo, hi + 1):
                cname = f'{cv}_lag{lag}'
                d[cname] = pd.to_numeric(d[cv], errors='coerce').shift(lag)
                cols.append(cname)
            bcol = f'{cv}_{lab}'
            d[bcol] = d[cols].mean(axis=1)
            xcols.append(bcol)
        if bool(getattr(args, 'channel_regression_lagged_share', True)):
            xcols.append('lag1_logit_target_channel_share')
        if bool(getattr(args, 'channel_control_transactions', True)):
            xcols.append('log_total_transactions')
        if bool(getattr(args, 'channel_control_trend', True)):
            xcols.extend(['_trend','_trend2'])
        fe = ['dow'] if bool(getattr(args, 'channel_control_dow', True)) else []
        res = _ols(d, 'logit_target_channel_share', xcols, fe_cols=fe, min_n=int(getattr(args, 'channel_min_daily_obs', 30)), cov_type='hac', hac_lags=int(getattr(args, 'channel_hac_lags_daily', 7)))
        for lab, lo, hi in bins:
            bcol = f'{cv}_{lab}'
            rows.append({
                'covid_var': base,
                'regressor_used': cv,
                'lag_bin': lab.replace('lag_', '').replace('_', '-'),
                'lag_start': lo,
                'lag_end': hi,
                'coef': res.get(f'coef_{bcol}', np.nan),
                'se': res.get(f'se_{bcol}', np.nan),
                't': res.get(f't_{bcol}', np.nan),
                'p': res.get(f'p_{bcol}', np.nan),
                'nobs': res.get('nobs', 0),
                'r2': res.get('r2', np.nan),
                'status': res.get('status', 'unknown'),
            })
    return pd.DataFrame(rows)


def _plot_regression_forest(reg: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    if reg.empty or 'coef' not in reg.columns:
        return
    d = reg.copy().replace([np.inf, -np.inf], np.nan).dropna(subset=['coef'])
    if d.empty:
        return
    d = d.sort_values('coef')
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(d) + 1)))
    xerr = 1.96 * pd.to_numeric(d.get('se', pd.Series(np.nan, index=d.index)), errors='coerce').fillna(0.0).to_numpy()
    ax.errorbar(d['coef'].astype(float), y, xerr=xerr, fmt='o', capsize=3)
    ax.axvline(0, lw=1)
    ax.set_yticks(y); ax.set_yticklabels(d['covid_var'].astype(str))
    ax.set_xlabel('Effect on logit(channel target share), 95% CI')
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def _plot_lag_bin_effects(lag: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    if lag.empty or 'coef' not in lag.columns:
        return
    for cv, g in lag.groupby('covid_var'):
        d = g.copy().replace([np.inf, -np.inf], np.nan).dropna(subset=['coef']).sort_values('lag_start')
        if d.empty:
            continue
        x = np.arange(len(d))
        yerr = 1.96 * pd.to_numeric(d.get('se', pd.Series(np.nan, index=d.index)), errors='coerce').fillna(0.0).to_numpy()
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.errorbar(x, d['coef'].astype(float), yerr=yerr, marker='o', capsize=3)
        ax.axhline(0, lw=1)
        ax.set_xticks(x); ax.set_xticklabels(d['lag_bin'].astype(str))
        ax.set_xlabel('COVID shock lead window')
        ax.set_ylabel('Effect on logit(channel target share)')
        ax.set_title(f'Lag-bin channel response: {cv}')
        fig.tight_layout(); fig.savefig(out_dir / f'channel_lag_bin_effects_{_safe_name(cv)}.png', dpi=220); plt.close(fig)


def _channel_group_migration(d: pd.DataFrame, target_channel: str, event_month: str, covid_end_month: str, group_fields: str) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame()
    lab, covid_months, base_months = _period_labels_for_user(d, event_month, covid_end_month)
    if lab.empty:
        return pd.DataFrame()
    lab['is_target_channel'] = (lab['sales_channel_id'].astype(str) == str(target_channel)).astype(float)
    rows = []
    # Overall baseline used for relative group shifts.
    overall = lab.groupby('period_group', as_index=False).agg(
        target_share=('is_target_channel','mean'),
        transactions=('is_target_channel','size'),
        customers=('customer_id','nunique') if 'customer_id' in lab.columns else ('is_target_channel','size'),
    )
    op = overall.pivot_table(index=[], columns='period_group', values='target_share') if len(overall) else pd.DataFrame()
    overall_shift = np.nan
    if {'base_2019_matched','covid_2020'}.issubset(overall['period_group'].unique()):
        od = dict(zip(overall['period_group'], overall['target_share']))
        overall_shift = float(od.get('covid_2020', np.nan) - od.get('base_2019_matched', np.nan))
    fields = [x.strip() for x in str(group_fields or '').split(',') if x.strip()]
    for gf in fields:
        if gf not in lab.columns:
            continue
        tmp = lab.dropna(subset=[gf]).copy()
        if tmp.empty:
            continue
        agg = tmp.groupby([gf, 'period_group'], observed=False).agg(
            target_share=('is_target_channel','mean'),
            transactions=('is_target_channel','size'),
            customers=('customer_id','nunique') if 'customer_id' in tmp.columns else ('is_target_channel','size'),
        ).reset_index().rename(columns={gf: 'group_value'})
        piv = agg.pivot_table(index='group_value', columns='period_group', values='target_share').reset_index()
        for _, r in piv.iterrows():
            base_s = r.get('base_2019_matched', np.nan); cov_s = r.get('covid_2020', np.nan)
            shift = float(cov_s - base_s) if np.isfinite(base_s) and np.isfinite(cov_s) else np.nan
            rows.append({
                'group_field': gf,
                'group_value': str(r['group_value']),
                'base_target_share': base_s,
                'covid_target_share': cov_s,
                'yoy_shift': shift,
                'relative_yoy_shift': shift - overall_shift if np.isfinite(shift) and np.isfinite(overall_shift) else np.nan,
                'overall_yoy_shift': overall_shift,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        # Add count diagnostics.
        cnt = lab.copy()
        for gf in fields:
            if gf not in cnt.columns:
                continue
            c = cnt.dropna(subset=[gf]).groupby([gf, 'period_group'], observed=False).agg(
                transactions=('sales_channel_id','size'),
                customers=('customer_id','nunique') if 'customer_id' in cnt.columns else ('sales_channel_id','size'),
            ).reset_index().rename(columns={gf:'group_value'})
            cp = c.pivot_table(index='group_value', columns='period_group', values='transactions').reset_index().rename(columns={'base_2019_matched':'base_transactions','covid_2020':'covid_transactions'})
            cp['group_field'] = gf
            out = out.merge(cp, on=['group_field','group_value'], how='left')
    return out


def _plot_channel_group_bars(group: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    if group.empty or 'yoy_shift' not in group.columns:
        return
    for gf, g in group.groupby('group_field'):
        for metric, suffix, title_suffix in [('yoy_shift','yoy_shift','YoY shift'), ('relative_yoy_shift','relative_shift','relative shift vs overall')]:
            d = g.copy().replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])
            if d.empty:
                continue
            # Keep natural order for common age bins.
            age_order = ['16-24','25-34','35-44','45-54','55-64','65+']
            if gf == 'age_group':
                d['_ord'] = d['group_value'].map({v:i for i,v in enumerate(age_order)}).fillna(999)
                d = d.sort_values('_ord')
            else:
                d = d.sort_values(metric)
            fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.45 * len(d) + 1)))
            ax.barh(np.arange(len(d)), d[metric].astype(float))
            ax.axvline(0, lw=1)
            ax.set_yticks(np.arange(len(d))); ax.set_yticklabels(d['group_value'].astype(str))
            ax.set_xlabel('COVID 2020 - matched 2019 channel target share')
            ax.set_title(f'{gf}: channel migration {title_suffix}')
            fig.tight_layout(); fig.savefig(out_dir / f'channel_migration_{_safe_name(gf)}_{suffix}.png', dpi=220); plt.close(fig)


def exp_channel_style(args) -> dict:
    """COVID-induced channel migration.

    This replaces the earlier Channel × Axis Migration task.  The channel task is
    now intentionally style-free: it studies whether pandemic shocks changed the
    probability/share of purchasing through the target sales channel, when the
    shift occurred, and which user groups exhibit stronger migration.
    """
    root = Path(args.social_output_root)
    data_root = Path(args.data_root)
    out = ensure_dir(root / 'analysis' / args.exp_id)
    target = str(getattr(args, 'channel_target_id', '2')).replace('.0', '')
    start = str(getattr(args, 'channel_start_date', '') or '')
    end = str(getattr(args, 'channel_end_date', '') or '')
    exclude_months = str(getattr(args, 'channel_exclude_months', '') or '')
    covid_end = getattr(args, 'channel_covid_end_month', '') or getattr(args, 'task1_covid_end_month', '') or ''

    tx = _load_channel_transactions(data_root, int(getattr(args, 'min_age', 16)), int(getattr(args, 'max_age', 80)))
    tx = _filter_channel_dates(tx, start, end, exclude_months)
    if tx.empty:
        raise RuntimeError('No transactions remain for channel migration after date/month filters.')

    daily = _channel_daily_panel_from_transactions(tx, target, start, end)
    monthly = _channel_monthly_panel_from_daily(daily)
    yoy, covid_months, base_months = _channel_matched_yoy(monthly, args.event_month, covid_end)
    daily_covid = _channel_daily_with_covid(args, daily)
    reg = _channel_covid_regression(args, daily_covid)
    lag = _channel_lag_bin_regression(args, daily_covid)
    group = _channel_group_migration(tx, target, args.event_month, covid_end, getattr(args, 'channel_group_fields', 'age_group,club_member_status,fashion_news_frequency'))

    tx_diag = tx.groupby(['month','sales_channel_id'], as_index=False).size().rename(columns={'size':'transactions'})
    tx_diag.to_parquet(out / 'channel_monthly_counts_long.parquet', index=False)
    daily.to_parquet(out / 'channel_daily_panel.parquet', index=False)
    monthly.to_parquet(out / 'channel_monthly_panel.parquet', index=False)
    yoy.to_parquet(out / 'channel_monthly_matched_yoy.parquet', index=False)
    daily_covid.to_parquet(out / 'channel_daily_covid_panel.parquet', index=False)
    reg.to_parquet(out / 'channel_covid_regression.parquet', index=False)
    lag.to_parquet(out / 'channel_lag_bin_regression.parquet', index=False)
    group.to_parquet(out / 'channel_group_migration.parquet', index=False)

    # Backward-compatible QA: if inference-time channel panel exists, save its quality table too.
    ch_path = root / 'channel_monthly_panel.parquet'
    if ch_path.exists():
        try:
            ch_raw = pd.read_parquet(ch_path)
            _channel_share_quality(ch_raw).to_parquet(out / 'channel_share_quality_from_inference.parquet', index=False)
        except Exception:
            pass

    if bool(getattr(args, 'make_figures', True)):
        _plot_channel_share_and_counts(monthly, out / 'channel_share_and_counts_timeseries.png', target)
        _plot_channel_yoy(yoy, out / 'channel_monthly_matched_yoy_shift.png', args.event_month, target)
        _plot_regression_forest(reg, out / 'channel_covid_effect_forest.png', f'COVID effects on channel {target} migration')
        _plot_lag_bin_effects(lag, out)
        _plot_channel_group_bars(group, out)

    manifest = {
        'task': 'COVID-induced Channel Migration',
        'target_channel': target,
        'transaction_rows': int(len(tx)),
        'daily_rows': int(len(daily)),
        'monthly_rows': int(len(monthly)),
        'matched_yoy_rows': int(len(yoy)),
        'regression_rows': int(len(reg)),
        'lag_bin_rows': int(len(lag)),
        'group_rows': int(len(group)),
        'covid_months': covid_months,
        'base_months': base_months,
        'date_min': str(tx['date'].min().date()) if len(tx) else '',
        'date_max': str(tx['date'].max().date()) if len(tx) else '',
        'excluded_months': exclude_months,
        'estimands': {
            'monthly_matched_yoy': 'target-channel share in COVID 2020 months minus the same calendar months in 2019',
            'daily_covid_regression': 'HAC OLS of logit target-channel daily share on COVID shock variables with day-of-week/trend/volume controls',
            'lag_bin_regression': 'distributed-lag COVID shock bins predicting logit target-channel share',
            'group_migration': 'group-level target-channel share shift, with relative shift subtracting overall market shift',
        },
        'notes': {
            'style_free': 'This channel task does not use axis/style scores. Former channel-style/decomposition outputs are intentionally retired from the main channel experiment.',
            'channel_id_semantics': 'Do not label channel 1/2 as online/offline until the sales_channel_id mapping is externally confirmed.',
        },
    }
    save_json(manifest, out / 'channel_migration_manifest.json')
    return manifest


EXPERIMENTS = {
    'axis_covid_style_shift': exp_prototype_response,
    # Backward-compatible aliases for the merged task-1/task-2 analysis.
    'prototype_response': exp_prototype_response,
    'style_shift': exp_style_shift,
    'exposure_response': exp_exposure_response,
    'lead_lag': exp_lead_lag,
    'user_heterogeneity': exp_user_heterogeneity,
    'channel_style': exp_channel_style,
    'channel_migration': exp_channel_style,
}

ALL_EXPERIMENTS = [
    'axis_covid_style_shift',
    'exposure_response',
    'lead_lag',
    'user_heterogeneity',
    'channel_migration',
]


def run_one_experiment(args) -> dict:
    if args.exp_id == 'all':
        summary = {}
        for exp_id in ALL_EXPERIMENTS:
            fn = EXPERIMENTS[exp_id]
            local = argparse.Namespace(**vars(args))
            local.exp_id = exp_id
            print(f'========== Running social experiment: {exp_id} ==========')
            summary[exp_id] = fn(local)
        out = ensure_dir(Path(args.social_output_root) / 'analysis')
        save_json(summary, out / 'social_experiments_manifest.json')
        return summary
    if args.exp_id not in EXPERIMENTS:
        raise ValueError(f'Unknown exp_id={args.exp_id}. Available: {sorted(EXPERIMENTS)} or all')
    ans = EXPERIMENTS[args.exp_id](args)
    out = ensure_dir(Path(args.social_output_root) / 'analysis' / args.exp_id)
    save_json({'exp_id': args.exp_id, 'exp_name': args.exp_name, 'summary': ans, 'args': vars(args)}, out / 'manifest.json')
    return ans


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Run one social-analysis experiment at a time.')
    p.add_argument('--exp_id', default='axis_covid_style_shift', choices=list(EXPERIMENTS.keys()) + ['all'])
    p.add_argument('--exp_name', default='', help='Optional descriptive name for this run.')
    p.add_argument('--social_output_root', default='./social_output/k')
    p.add_argument('--data_root', default='./data')
    p.add_argument('--event_month', default='2020-03')
    p.add_argument('--style_dims', default='formality_structure,public_private_occasion,comfort_ease,practical_value,trend_expressiveness,material_softness,material_thermal_weight,material_durability,color_temperature,color_lightness,color_chroma,design_complexity,fit_looseness,office_like,homewear_like,casual_like,social_outing_like,value_basic')
    # Merged Task-1/Task-2 axis-level COVID style/distribution shift controls.  prototype_response/style_shift are kept as backward-compatible aliases.
    p.add_argument('--task1_axis_dims', default='formality_structure,public_private_occasion,comfort_ease,practical_value,trend_expressiveness,design_complexity,fit_looseness,office_like,homewear_like,casual_like,social_outing_like,value_basic', help='Main axis/composite dimensions for task 1. Seasonal warm/cool dimensions are excluded by default.')
    p.add_argument('--task1_raw_axis_dims', default='formality_structure,public_private_occasion,comfort_ease,practical_value,trend_expressiveness,design_complexity,fit_looseness', help='Raw, non-derived axes. Used by default for PCA/copula to avoid mechanical dependence from composite scores.')
    p.add_argument('--task1_composite_dims', default='office_like,homewear_like,casual_like,social_outing_like,value_basic', help='Derived composite dimensions used for interpretable univariate summaries but excluded from default PCA/copula.')
    p.add_argument('--task1_pca_dims', default='formality_structure,public_private_occasion,comfort_ease,practical_value,trend_expressiveness,design_complexity,fit_looseness', help='Dimensions used in task-1 PCA/MMD. Defaults to raw axes only.')
    p.add_argument('--task1_copula_dims', default='formality_structure,public_private_occasion,comfort_ease,practical_value,trend_expressiveness,design_complexity,fit_looseness', help='Dimensions used in task-1 Gaussian copula. Defaults to raw axes only.')
    p.add_argument('--task1_season_control_dims', default='material_thermal_weight,color_temperature,color_lightness,material_softness', help='Season-sensitive axes used only for residualization/robustness in task 1.')
    p.add_argument('--task1_covid_end_month', default='', help='Last COVID-window month for task 1. Empty means the latest available month in item_monthly_panel.')
    p.add_argument('--task1_bootstrap_n', type=int, default=500, help='Month-pair bootstrap repetitions for task-1 confidence intervals.')
    p.add_argument('--task1_category_residualize', type=parse_bool, default=True, help='Residualize task-1 axes using light product/category controls in addition to season-sensitive axes.')
    p.add_argument('--task1_residual_max_cat_levels', type=int, default=60, help='Maximum retained levels per category variable when residualizing task-1 axes.')
    p.add_argument('--primary_covid_var', default='covid_cases_index')
    p.add_argument('--make_figures', type=parse_bool, default=True)
    p.add_argument('--seed', type=int, default=42)
    # Common experiment controls.
    p.add_argument('--min_periods', type=int, default=8)
    p.add_argument('--top_n', type=int, default=30)
    p.add_argument('--outlier_z_bound', type=float, default=2.5)
    p.add_argument('--num_clusters', type=int, default=4)
    p.add_argument('--prototype_fallback_from_scores', type=parse_bool, default=False, help='Explicit fallback only: if prototype panel is empty/missing, build high/low pseudo-prototypes from style scores. Default False; normal prototype analysis uses routed prompt prototypes.')
    p.add_argument('--prototype_high_quantile', type=float, default=0.75)
    p.add_argument('--prototype_low_quantile', type=float, default=0.25)
    p.add_argument('--prototype_p_clip_min', type=float, default=1e-16, help='Lower bound used only when plotting -log10(p) for prototype volcano plots.')
    p.add_argument('--prototype_volcano_label_n', type=int, default=6, help='Number of extreme prototype points to annotate in the volcano plot.')
    p.add_argument('--prototype_hac_lags', type=int, default=2, help='Newey-West lag length for prototype-response monthly regressions. Use 0 to recover HC-style heteroskedastic robust SE.')
    # Style shift.
    p.add_argument('--hist_bins', type=int, default=30)
    p.add_argument('--max_samples', type=int, default=50000)
    # Exposure response.
    p.add_argument('--exposure_dims', default='formality_structure,comfort_ease,homewear_like,office_like')
    p.add_argument('--high_quantile', type=float, default=0.75)
    p.add_argument('--low_quantile', type=float, default=0.25)
    p.add_argument('--event_window', type=int, default=8)
    # Lead-lag.
    p.add_argument('--max_lead', type=int, default=2)
    p.add_argument('--max_lag', type=int, default=4)
    p.add_argument('--run_ardl', type=parse_bool, default=True)
    p.add_argument('--ar_lags', type=int, default=2)
    # Daily Lead-Lag dynamic association.  The old weekly raw-share path is replaced by this daily residualized-axis design.
    p.add_argument('--covid_csv', default='', help='OWID daily COVID CSV. Empty defaults to data_root/external/owid-covid-data.csv.')
    p.add_argument('--covid_location', default='World')
    p.add_argument('--lead_lag_start_date', default='2019-10-01')
    p.add_argument('--lead_lag_end_date', default='2020-09-30')
    p.add_argument('--lead_lag_dims', default='comfort_ease,homewear_like,fit_looseness,value_basic,office_like,public_private_occasion,social_outing_like,trend_expressiveness,design_complexity,formality_structure,practical_value')
    p.add_argument('--lead_lag_season_control_dims', default='material_thermal_weight,color_temperature,color_lightness,material_softness')
    p.add_argument('--lead_lag_covid_vars', default='covid_cases_index,covid_deaths_index,covid_reproduction_rate,covid_stringency_index')
    p.add_argument('--lead_lag_covid_transform', default='delta7', choices=['delta7','diff1','level'])
    p.add_argument('--lead_lag_roll_days', type=int, default=7)
    p.add_argument('--lead_lag_max_lag_days', type=int, default=28)
    p.add_argument('--lead_lag_placebo_days', type=int, default=7)
    p.add_argument('--lead_lag_bins', default='0:3,4:7,8:14,15:28')
    p.add_argument('--lead_lag_prewhiten_lags', type=int, default=7)
    p.add_argument('--lead_lag_ar_lags_daily', type=int, default=7)
    p.add_argument('--lead_lag_hac_lags_daily', type=int, default=7)
    p.add_argument('--lead_lag_residualize_dow', type=parse_bool, default=True)
    p.add_argument('--lead_lag_residualize_trend', type=parse_bool, default=True)
    p.add_argument('--lead_lag_residualize_season_axes', type=parse_bool, default=True)
    p.add_argument('--lead_lag_control_transactions', type=parse_bool, default=True)
    # User heterogeneity.  This task is now split into group-level, user-level,
    # and transaction-level age × axis density estimands.  max_purchase_rows<=0
    # means full transaction-level analysis.
    p.add_argument('--max_purchase_rows', type=int, default=0)
    p.add_argument('--min_age', type=int, default=16)
    p.add_argument('--max_age', type=int, default=80)
    p.add_argument('--age_bin_width', type=int, default=2)
    p.add_argument('--gmm_components', type=int, default=3)
    p.add_argument('--gmm_sample_size', type=int, default=30000)  # kept for backward compatibility; torch GMM does not sample by default.
    p.add_argument('--user_axis_dims', default='comfort_ease,homewear_like,fit_looseness,value_basic,office_like,public_private_occasion,social_outing_like,trend_expressiveness,design_complexity,formality_structure,practical_value')
    p.add_argument('--user_kde_dims', default='comfort_ease,office_like,trend_expressiveness,value_basic,homewear_like,public_private_occasion')
    p.add_argument('--user_season_control_dims', default='material_thermal_weight,color_temperature,color_lightness,material_softness')
    p.add_argument('--user_group_fields', default='age_group,club_member_status,fashion_news_frequency')
    p.add_argument('--user_covid_end_month', default='', help='Last COVID-window month for user heterogeneity; empty falls back to task1_covid_end_month or latest matched month.')
    p.add_argument('--user_use_cuda', default='auto', choices=['auto','cuda','cpu'])
    p.add_argument('--user_txn_weighting', default='user_balanced', choices=['transaction','user_balanced'])
    p.add_argument('--user_run_transaction_kde', type=parse_bool, default=True)
    p.add_argument('--user_run_transaction_gmm', type=parse_bool, default=True)
    p.add_argument('--user_run_user_gmm', type=parse_bool, default=True)
    p.add_argument('--user_kde_age_bins', type=int, default=65)
    p.add_argument('--user_kde_score_bins', type=int, default=60)
    p.add_argument('--user_kde_bandwidth_age', type=float, default=2.5)
    p.add_argument('--user_kde_bandwidth_score', type=float, default=0.035)
    p.add_argument('--user_gmm_iter', type=int, default=80)
    p.add_argument('--channel_axis_dims', default='comfort_ease,homewear_like,fit_looseness,value_basic,office_like,public_private_occasion,social_outing_like,trend_expressiveness,design_complexity,formality_structure,practical_value')
    p.add_argument('--channel_decomposition_dims', default='comfort_ease,homewear_like,fit_looseness,value_basic,office_like,public_private_occasion,social_outing_like,trend_expressiveness,design_complexity,formality_structure,practical_value')
    p.add_argument('--channel_plot_dims', default='comfort_ease,homewear_like,office_like,public_private_occasion,trend_expressiveness,design_complexity')
    p.add_argument('--channel_covid_end_month', default='', help='Last COVID-window month for channel matched YoY/decomposition; empty falls back to task1_covid_end_month or latest available channel month.')
    p.add_argument('--channel_fill_missing_channels', type=parse_bool, default=True, help='Fill missing channel×month rows with zero channel share so line plots use a true chronological axis and missing channels become explicit.')
    p.add_argument('--channel_plot_top_dims', type=int, default=6)

    # COVID-induced channel migration.  This replaces the old channel×axis task;
    # old channel_axis/channel_decomposition args are kept for backward CLI compatibility.
    p.add_argument('--channel_target_id', default='2', help='Target sales_channel_id whose share/migration is analyzed. Default 2.')
    p.add_argument('--channel_start_date', default='', help='Optional start date for channel migration. Empty uses first transaction date.')
    p.add_argument('--channel_end_date', default='', help='Optional end date for channel migration. Empty uses last transaction date.')
    p.add_argument('--channel_regression_start_date', default='2019-10-01', help='Start date for daily COVID-channel regression.')
    p.add_argument('--channel_regression_end_date', default='2020-09-30', help='End date for daily COVID-channel regression.')
    p.add_argument('--channel_exclude_months', default='', help='Comma-separated months to exclude, e.g. 2020-04,2020-09, for robustness.')
    p.add_argument('--channel_covid_vars', default='covid_cases_index,covid_deaths_index,covid_reproduction_rate,covid_stringency_index')
    p.add_argument('--channel_covid_transform', default='delta7', choices=['delta7','diff1','level'])
    p.add_argument('--channel_lag_bins', default='0:3,4:7,8:14,15:28')
    p.add_argument('--channel_hac_lags_daily', type=int, default=7)
    p.add_argument('--channel_min_daily_obs', type=int, default=30)
    p.add_argument('--channel_logit_eps', type=float, default=1e-4)
    p.add_argument('--channel_group_fields', default='age_group,club_member_status,fashion_news_frequency')
    p.add_argument('--channel_regression_lagged_share', type=parse_bool, default=True)
    p.add_argument('--channel_control_transactions', type=parse_bool, default=True)
    p.add_argument('--channel_control_dow', type=parse_bool, default=True)
    p.add_argument('--channel_control_trend', type=parse_bool, default=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    summary = run_one_experiment(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
