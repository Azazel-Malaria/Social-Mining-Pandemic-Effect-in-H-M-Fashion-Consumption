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
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from util.io_utils import ensure_dir, parse_bool, save_json

# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def _article_key(x) -> str:
    s = str(x)
    if s.endswith('.0'):
        s = s[:-2]
    return s.zfill(10) if s.isdigit() else s


def _safe_name(x: str) -> str:
    return ''.join(ch if ch.isalnum() else '_' for ch in str(x)).strip('_')[:80] or 'x'


def _zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors='coerce').astype(float)
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return x * 0.0
    return (x - x.mean()) / sd


def _period_month_index(month: pd.Series) -> np.ndarray:
    pi = pd.PeriodIndex(month.astype(str), freq='M')
    return (pi.year * 12 + pi.month).astype(float).to_numpy()


def _add_time_controls(df: pd.DataFrame, fourier_order: int = 0) -> pd.DataFrame:
    out = df.copy()
    pi = pd.PeriodIndex(out['month'].astype(str), freq='M')
    idx = (pi.year * 12 + pi.month).astype(float).to_numpy()
    out['_trend'] = idx - np.nanmin(idx) if len(idx) else idx
    out['_calendar_month'] = pi.month.astype(str)
    for h in range(1, int(fourier_order) + 1):
        out[f'_sin{h}'] = np.sin(2 * np.pi * h * idx / 12.0)
        out[f'_cos{h}'] = np.cos(2 * np.pi * h * idx / 12.0)
    return out


def _numeric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce').astype(float)
    return out


def _ols_np(df: pd.DataFrame, y: str, xcols: list[str], fe_cols: list[str] | None = None, min_n: int = 8) -> dict:
    cols = [y] + xcols + (fe_cols or [])
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(d) < max(min_n, len(xcols) + 2):
        return {'nobs': int(len(d)), 'status': 'too_few_rows'}
    parts = [pd.Series(1.0, index=d.index, name='const')]
    for c in xcols:
        parts.append(pd.to_numeric(d[c], errors='coerce').astype(float).rename(c))
    for fc in fe_cols or []:
        dm = pd.get_dummies(d[fc].astype(str), prefix=fc, drop_first=True, dtype=float)
        if dm.shape[1] > 0:
            parts.append(dm)
    X = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    yy = pd.to_numeric(d[y], errors='coerce').astype(float).to_numpy()
    Xn = X.to_numpy(dtype=float)
    try:
        beta, *_ = np.linalg.lstsq(Xn, yy, rcond=None)
        resid = yy - Xn @ beta
        n, k = Xn.shape
        dof = max(n - k, 1)
        sigma2 = float((resid @ resid) / dof)
        xtx_inv = np.linalg.pinv(Xn.T @ Xn)
        se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
        sst = max(float(((yy - yy.mean()) @ (yy - yy.mean()))), 1e-12)
        out = {'nobs': int(n), 'status': 'ok', 'r2': float(1.0 - (resid @ resid) / sst), 'num_parameters': int(k)}
        for c, b, s in zip(X.columns, beta, se):
            if c in xcols or c == 'const':
                out[f'coef_{c}'] = float(b)
                out[f'se_{c}'] = float(s)
                out[f't_{c}'] = float(b / s) if s > 0 else np.nan
        return out
    except Exception as exc:
        return {'nobs': int(len(d)), 'status': f'error:{type(exc).__name__}:{exc}'}


def _within_ols_np(df: pd.DataFrame, y: str, xcols: list[str], group_col: str, min_n: int = 30) -> dict:
    """Fast fixed-effect OLS by demeaning y and x within group_col."""
    cols = [group_col, y] + xcols
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(d) < max(min_n, len(xcols) + 2):
        return {'nobs': int(len(d)), 'status': 'too_few_rows'}
    for c in [y] + xcols:
        d[c] = pd.to_numeric(d[c], errors='coerce').astype(float)
        d[c] = d[c] - d.groupby(group_col)[c].transform('mean')
    X = d[xcols].to_numpy(dtype=float)
    yy = d[y].to_numpy(dtype=float)
    keep = np.isfinite(yy) & np.isfinite(X).all(axis=1)
    X, yy = X[keep], yy[keep]
    if len(yy) < max(min_n, len(xcols) + 2):
        return {'nobs': int(len(yy)), 'status': 'too_few_rows'}
    try:
        beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        resid = yy - X @ beta
        n, k = X.shape
        dof = max(n - k, 1)
        sigma2 = float((resid @ resid) / dof)
        xtx_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
        sst = max(float(((yy - yy.mean()) @ (yy - yy.mean()))), 1e-12)
        out = {'nobs': int(n), 'status': 'ok', 'r2_within': float(1.0 - (resid @ resid) / sst), 'num_parameters': int(k)}
        for c, b, s in zip(xcols, beta, se):
            out[f'coef_{c}'] = float(b)
            out[f'se_{c}'] = float(s)
            out[f't_{c}'] = float(b / s) if s > 0 else np.nan
        return out
    except Exception as exc:
        return {'nobs': int(len(d)), 'status': f'error:{type(exc).__name__}:{exc}'}


def _save_empty(path: Path, note: str = '') -> pd.DataFrame:
    df = pd.DataFrame([{'status': note or 'empty'}])
    df.to_parquet(path, index=False)
    return df


def _maybe_plot(out: Path, name: str, plot_func: Callable[[], None], make_figures: bool = True) -> None:
    if not make_figures:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig_dir = ensure_dir(out / 'figures')
        plot_func()
        plt.tight_layout()
        plt.savefig(fig_dir / f'{name}.png', dpi=220)
        plt.close()
    except Exception as exc:
        print(f'[WARN] Figure {name} failed: {type(exc).__name__}: {exc}')


def _standardize_for_plot(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[f'{c}_zplot'] = _zscore(out[c])
    return out


def _parse_style_dims(style_dims: str) -> list[str]:
    return [x.strip().replace('_score', '') for x in str(style_dims).split(',') if x.strip()]


def _score_col(dim: str) -> str:
    return dim if dim.endswith('_score') else f'{dim}_score'

# -----------------------------------------------------------------------------
# External data loaders
# -----------------------------------------------------------------------------

def load_or_build_covid_monthly(social_root: Path, covid_csv: str | None, covid_location: str = 'World') -> pd.DataFrame:
    if covid_csv:
        cpath = Path(covid_csv)
        if cpath.exists():
            covid = pd.read_csv(cpath)
            if 'location' in covid.columns:
                covid = covid[covid['location'].astype(str) == str(covid_location)].copy()
            if covid.empty:
                return pd.DataFrame({'month': []})
            covid['date'] = pd.to_datetime(covid['date'])
            covid['month'] = covid['date'].dt.to_period('M').astype(str)
            # OWID columns that are useful for mechanism analysis.
            raw_cols = [c for c in [
                'new_cases_smoothed_per_million', 'new_deaths_smoothed_per_million',
                'new_cases_per_million', 'new_deaths_per_million',
                'reproduction_rate', 'stringency_index',
                'icu_patients_per_million', 'hosp_patients_per_million',
                'positive_rate', 'people_vaccinated_per_hundred'
            ] if c in covid.columns]
            out = covid.groupby('month', as_index=False)[raw_cols].mean() if raw_cols else covid[['month']].drop_duplicates()
            case = 'new_cases_smoothed_per_million' if 'new_cases_smoothed_per_million' in out.columns else ('new_cases_per_million' if 'new_cases_per_million' in out.columns else None)
            death = 'new_deaths_smoothed_per_million' if 'new_deaths_smoothed_per_million' in out.columns else ('new_deaths_per_million' if 'new_deaths_per_million' in out.columns else None)
            if case:
                out['covid_cases_index'] = np.log1p(pd.to_numeric(out[case], errors='coerce').fillna(0.0).clip(lower=0))
                out['covid_cases_z'] = _zscore(out['covid_cases_index'])
            if death:
                out['covid_deaths_index'] = np.log1p(pd.to_numeric(out[death], errors='coerce').fillna(0.0).clip(lower=0))
                out['covid_deaths_z'] = _zscore(out['covid_deaths_index'])
            if 'stringency_index' in out.columns:
                out['covid_stringency_index'] = pd.to_numeric(out['stringency_index'], errors='coerce')
                out['covid_stringency_z'] = _zscore(out['covid_stringency_index'])
            if 'reproduction_rate' in out.columns:
                out['covid_reproduction_rate'] = pd.to_numeric(out['reproduction_rate'], errors='coerce')
                out['covid_reproduction_z'] = _zscore(out['covid_reproduction_rate'])
            for c in ['icu_patients_per_million', 'hosp_patients_per_million', 'positive_rate', 'people_vaccinated_per_hundred']:
                if c in out.columns:
                    out[f'covid_{c}'] = pd.to_numeric(out[c], errors='coerce')
                    out[f'covid_{c}_z'] = _zscore(out[f'covid_{c}'])
            return out
    # Fallback: COVID columns already in panel.
    existing = social_root / 'item_monthly_panel.parquet'
    if existing.exists():
        im = pd.read_parquet(existing)
        cols = [c for c in im.columns if c.startswith('covid_') or c == 'month']
        if len(cols) > 1:
            return im[cols].drop_duplicates('month').sort_values('month')
    return pd.DataFrame({'month': []})


def load_mobility_monthly(mobility_csv: str | None, mobility_country: str = '', mobility_region: str = '') -> pd.DataFrame:
    if not mobility_csv or not Path(mobility_csv).exists():
        return pd.DataFrame({'month': []})
    mob = pd.read_csv(mobility_csv)
    if 'date' not in mob.columns:
        return pd.DataFrame({'month': []})
    if mobility_country and 'country_region' in mob.columns:
        mob = mob[mob['country_region'].astype(str) == str(mobility_country)].copy()
    if mobility_region and 'sub_region_1' in mob.columns:
        mob = mob[mob['sub_region_1'].astype(str) == str(mobility_region)].copy()
    # Prefer country-level rows when possible.
    if 'sub_region_1' in mob.columns and not mobility_region:
        country_rows = mob['sub_region_1'].isna() | (mob['sub_region_1'].astype(str).str.lower() == 'nan')
        if country_rows.any():
            mob = mob[country_rows].copy()
    mob['date'] = pd.to_datetime(mob['date'])
    mob['month'] = mob['date'].dt.to_period('M').astype(str)
    cols = [c for c in [
        'retail_and_recreation_percent_change_from_baseline',
        'grocery_and_pharmacy_percent_change_from_baseline',
        'parks_percent_change_from_baseline',
        'transit_stations_percent_change_from_baseline',
        'workplaces_percent_change_from_baseline',
        'residential_percent_change_from_baseline'
    ] if c in mob.columns]
    out = mob.groupby('month', as_index=False)[cols].mean() if cols else mob[['month']].drop_duplicates()
    rename = {
        'retail_and_recreation_percent_change_from_baseline': 'mobility_retail_recreation',
        'workplaces_percent_change_from_baseline': 'mobility_workplaces',
        'residential_percent_change_from_baseline': 'mobility_residential',
        'transit_stations_percent_change_from_baseline': 'mobility_transit',
        'grocery_and_pharmacy_percent_change_from_baseline': 'mobility_grocery_pharmacy',
        'parks_percent_change_from_baseline': 'mobility_parks',
    }
    out = out.rename(columns=rename)
    for c in [c for c in out.columns if c.startswith('mobility_')]:
        out[f'{c}_z'] = _zscore(out[c])
    return out


def merge_external(ts: pd.DataFrame, covid: pd.DataFrame, mobility: pd.DataFrame | None = None) -> pd.DataFrame:
    out = ts.copy()
    if len(covid):
        out = out.merge(covid, on='month', how='left', suffixes=('', '_covid'))
    if mobility is not None and len(mobility) and len(mobility.columns) > 1:
        out = out.merge(mobility, on='month', how='left')
    return out

# -----------------------------------------------------------------------------
# Panel loaders
# -----------------------------------------------------------------------------

def load_item_panel(social_root: Path) -> pd.DataFrame:
    p = social_root / 'item_monthly_panel.parquet'
    if not p.exists():
        raise FileNotFoundError(f'Missing {p}; run scripts/run_social_inference.sh first.')
    return pd.read_parquet(p)


def load_category_panel(social_root: Path) -> pd.DataFrame:
    p = social_root / 'category_monthly_panel.parquet'
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def load_prototype_panel(social_root: Path) -> pd.DataFrame:
    p = social_root / 'prototype_monthly_panel.parquet'
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def load_knowledge_panel(social_root: Path) -> pd.DataFrame:
    p = social_root / 'knowledge_monthly_panel.parquet'
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def load_transactions(data_root: Path) -> pd.DataFrame:
    p = data_root / 'hm' / 'processed' / 'hm_transactions.parquet'
    if not p.exists():
        raise FileNotFoundError(f'Missing {p}')
    tx = pd.read_parquet(p)
    tx['article_id'] = tx['article_id'].map(_article_key)
    tx['t_dat'] = pd.to_datetime(tx['t_dat'])
    tx['month'] = tx['t_dat'].dt.to_period('M').astype(str)
    return tx

# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------

def exp_panel_check(args, out: Path) -> dict:
    social_root = Path(args.social_output_root)
    result = {'experiment': 'panel_check'}
    files = [
        'item_embeddings.parquet', 'item_metadata.parquet', 'item_knowledge_scores.parquet',
        'item_knowledge_prototypes.parquet', 'item_monthly_panel.parquet', 'category_monthly_panel.parquet',
        'prototype_monthly_panel.parquet', 'knowledge_monthly_panel.parquet', 'user_monthly_panel.parquet'
    ]
    rows = []
    for f in files:
        p = social_root / f
        if p.exists():
            try:
                df = pd.read_parquet(p)
                rows.append({'file': f, 'exists': True, 'rows': len(df), 'cols': len(df.columns)})
            except Exception as exc:
                rows.append({'file': f, 'exists': True, 'rows': -1, 'cols': -1, 'error': str(exc)})
        else:
            rows.append({'file': f, 'exists': False, 'rows': 0, 'cols': 0})
    summary = pd.DataFrame(rows)
    summary.to_parquet(out / 'panel_file_summary.parquet', index=False)
    result['files'] = rows
    try:
        im = load_item_panel(social_root)
        score_cols = [c for c in im.columns if c.endswith('_score')]
        stats = []
        for c in score_cols:
            s = pd.to_numeric(im[c], errors='coerce')
            stats.append({'score': c, 'mean': float(s.mean()), 'std': float(s.std()), 'min': float(s.min()), 'max': float(s.max()), 'missing_rate': float(s.isna().mean())})
        pd.DataFrame(stats).to_parquet(out / 'style_score_summary.parquet', index=False)
        result['num_item_month_rows'] = len(im)
        result['score_columns'] = score_cols
    except Exception as exc:
        result['panel_error'] = str(exc)
    return result


def exp_total_sales(args, out: Path) -> dict:
    im = load_item_panel(Path(args.social_output_root))
    covid = load_or_build_covid_monthly(Path(args.social_output_root), args.covid_csv, args.covid_location)
    mobility = load_mobility_monthly(args.mobility_csv, args.mobility_country, args.mobility_region)
    ts = im.groupby('month', as_index=False)['sales_count'].sum().rename(columns={'sales_count': 'total_sales'})
    if 'avg_price' in im.columns:
        price = im.groupby('month', as_index=False)['avg_price'].mean()
        ts = ts.merge(price, on='month', how='left')
    ts = merge_external(ts, covid, mobility)
    ts['log_total_sales'] = np.log1p(pd.to_numeric(ts['total_sales'], errors='coerce').fillna(0))
    ts = _add_time_controls(ts, args.fourier_order)
    ts.to_parquet(out / 'total_sales_timeseries.parquet', index=False)
    xbase = ['_trend'] + [f'_{k}{h}' for h in range(1, args.fourier_order + 1) for k in ['sin', 'cos']]
    candidates = [c for c in ['covid_cases_index', 'covid_deaths_index', 'covid_stringency_index', 'covid_reproduction_rate', 'mobility_retail_recreation', 'mobility_workplaces', 'mobility_residential'] if c in ts.columns]
    rows = []
    for cv in candidates:
        fit = _ols_np(ts, 'log_total_sales', [cv] + xbase, min_n=8)
        fit.update({'model': 'total_sales_dynamic_regression', 'covariate': cv, 'outcome': 'log_total_sales'})
        rows.append(fit)
    if len(candidates) >= 2:
        fit = _ols_np(ts, 'log_total_sales', candidates + xbase, min_n=8)
        fit.update({'model': 'total_sales_joint_dynamic_regression', 'covariate': '+'.join(candidates), 'outcome': 'log_total_sales'})
        rows.append(fit)
    # ARDL-style single covariate models with one lag for small monthly samples.
    for cv in candidates:
        t2 = ts.copy().sort_values('month')
        t2['lag_y1'] = t2['log_total_sales'].shift(1)
        t2[f'{cv}_lag1'] = t2[cv].shift(1)
        fit = _ols_np(t2, 'log_total_sales', ['lag_y1', cv, f'{cv}_lag1'] + xbase, min_n=8)
        fit.update({'model': 'total_sales_ardl_lag1', 'covariate': cv, 'outcome': 'log_total_sales'})
        rows.append(fit)
    res = pd.DataFrame(rows)
    res.to_parquet(out / 'total_sales_models.parquet', index=False)
    def plot():
        import matplotlib.pyplot as plt
        d = _standardize_for_plot(ts, ['total_sales'] + candidates[:3])
        for c in ['total_sales'] + candidates[:3]:
            pc = f'{c}_zplot'
            if pc in d.columns:
                plt.plot(d['month'], d[pc], marker='o', label=c)
        plt.axvline(args.event_month, linestyle='--')
        plt.xticks(rotation=45)
        plt.title('Total sales and pandemic indicators')
        plt.legend()
    _maybe_plot(out, 'total_sales_vs_pandemic', plot, args.make_figures)
    return {'experiment': 'total_sales', 'rows': len(res), 'timeseries_rows': len(ts), 'covariates': candidates}


def exp_channel_shift(args, out: Path) -> dict:
    tx = load_transactions(Path(args.data_root))
    if 'sales_channel_id' not in tx.columns:
        _save_empty(out / 'channel_share_timeseries.parquet', 'sales_channel_id missing')
        return {'experiment': 'channel_shift', 'status': 'sales_channel_id missing'}
    covid = load_or_build_covid_monthly(Path(args.social_output_root), args.covid_csv, args.covid_location)
    mobility = load_mobility_monthly(args.mobility_csv, args.mobility_country, args.mobility_region)
    ch = tx.groupby(['month', 'sales_channel_id'], as_index=False).size().rename(columns={'size': 'num_transactions'})
    total = tx.groupby('month', as_index=False).size().rename(columns={'size': 'total_transactions'})
    ch = ch.merge(total, on='month', how='left')
    ch['channel_share'] = ch['num_transactions'] / ch['total_transactions'].replace(0, np.nan)
    ch = merge_external(ch, covid, mobility)
    ch = _add_time_controls(ch, args.fourier_order)
    ch.to_parquet(out / 'channel_share_timeseries.parquet', index=False)
    xbase = ['_trend'] + [f'_{k}{h}' for h in range(1, args.fourier_order + 1) for k in ['sin', 'cos']]
    covs = [c for c in ['covid_cases_index', 'covid_deaths_index', 'covid_stringency_index', 'mobility_retail_recreation', 'mobility_workplaces', 'mobility_residential'] if c in ch.columns]
    rows = []
    for sid, g in ch.groupby('sales_channel_id'):
        for cv in covs:
            fit = _ols_np(g, 'channel_share', [cv] + xbase, min_n=8)
            fit.update({'model': 'channel_share_dynamic_regression', 'sales_channel_id': sid, 'covariate': cv})
            rows.append(fit)
        # Segmented regression.
        t = g.copy().sort_values('month')
        pi = pd.PeriodIndex(t['month'].astype(str), freq='M')
        ev = pd.Period(args.event_month, freq='M')
        rel = (pi.year - ev.year) * 12 + (pi.month - ev.month)
        t['post_event'] = (rel >= 0).astype(float)
        t['time_after_event'] = np.maximum(rel, 0).astype(float)
        fit = _ols_np(t, 'channel_share', ['_trend', 'post_event', 'time_after_event'] + [f'_{k}{h}' for h in range(1, args.fourier_order + 1) for k in ['sin', 'cos']], min_n=8)
        fit.update({'model': 'channel_share_interrupted_time_series', 'sales_channel_id': sid, 'covariate': 'event_month'})
        rows.append(fit)
    res = pd.DataFrame(rows)
    res.to_parquet(out / 'channel_shift_models.parquet', index=False)
    def plot():
        import matplotlib.pyplot as plt
        for sid, g in ch.groupby('sales_channel_id'):
            g = g.sort_values('month')
            plt.plot(g['month'], g['channel_share'], marker='o', label=f'channel {sid}')
        plt.axvline(args.event_month, linestyle='--')
        plt.xticks(rotation=45)
        plt.title('Sales channel share')
        plt.legend()
    _maybe_plot(out, 'channel_share_timeseries', plot, args.make_figures)
    return {'experiment': 'channel_shift', 'rows': len(res), 'channels': sorted(ch['sales_channel_id'].dropna().unique().tolist())}


def _share_regression_panel(panel: pd.DataFrame, value_col: str, id_cols: list[str], out_path: Path, prefix: str, args, top_n: int = 30) -> pd.DataFrame:
    if panel.empty or value_col not in panel.columns:
        return _save_empty(out_path / f'{prefix}_models.parquet', 'empty panel')
    panel = _add_time_controls(panel, args.fourier_order)
    # To keep small and meaningful, prioritize IDs with high total sales.
    if 'sales_count' in panel.columns:
        rank = panel.groupby(id_cols, dropna=False)['sales_count'].sum().reset_index().sort_values('sales_count', ascending=False).head(top_n)
        panel = panel.merge(rank[id_cols], on=id_cols, how='inner')
    covs = [c for c in ['covid_cases_index', 'covid_deaths_index', 'covid_stringency_index', 'covid_reproduction_rate'] if c in panel.columns]
    xbase = ['_trend'] + [f'_{k}{h}' for h in range(1, args.fourier_order + 1) for k in ['sin', 'cos']]
    rows = []
    for key, g in panel.groupby(id_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        if g['month'].nunique() < 8:
            continue
        for cv in covs:
            fit = _ols_np(g, value_col, [cv] + xbase, min_n=8)
            for c, v in zip(id_cols, key_tuple):
                fit[c] = v
            fit.update({'model': f'{prefix}_share_dynamic_regression', 'covariate': cv, 'outcome': value_col})
            rows.append(fit)
    res = pd.DataFrame(rows)
    res.to_parquet(out_path / f'{prefix}_models.parquet', index=False)
    return res


def exp_category_structure(args, out: Path) -> dict:
    panel = load_category_panel(Path(args.social_output_root))
    res = _share_regression_panel(panel, 'sales_share', ['category_field', 'category_value'], out, 'category_structure', args, top_n=args.top_n_categories)
    def plot():
        import matplotlib.pyplot as plt
        if panel.empty:
            return
        # Pre/post top changes by product_group_name if available.
        d = panel[panel['category_field'] == 'product_group_name'].copy()
        if d.empty:
            d = panel.copy()
        ev = pd.Period(args.event_month, freq='M')
        pi = pd.PeriodIndex(d['month'].astype(str), freq='M')
        d['post'] = pi >= ev
        piv = d.groupby(['category_value', 'post'])['sales_share'].mean().unstack()
        if True in piv.columns and False in piv.columns:
            delta = (piv[True] - piv[False]).sort_values()
            delta = pd.concat([delta.head(10), delta.tail(10)])
            plt.barh(delta.index.astype(str), delta.values)
            plt.title('Pre/post category share change')
    _maybe_plot(out, 'category_prepost_share_change', plot, args.make_figures)
    return {'experiment': 'category_structure', 'rows': len(res)}


def exp_style_share(args, out: Path) -> dict:
    kp = load_knowledge_panel(Path(args.social_output_root))
    if kp.empty:
        return {'experiment': 'style_share', 'status': 'knowledge_monthly_panel missing'}
    dims = _parse_style_dims(args.style_dims)
    kp = kp[kp['knowledge_dim'].isin(dims)].copy() if 'knowledge_dim' in kp.columns else kp
    covid = load_or_build_covid_monthly(Path(args.social_output_root), args.covid_csv, args.covid_location)
    kp = kp.drop(columns=[c for c in kp.columns if c.startswith('covid_')], errors='ignore')
    kp = merge_external(kp, covid)
    kp.to_parquet(out / 'style_share_timeseries.parquet', index=False)
    res = _share_regression_panel(kp, 'weighted_share', ['knowledge_dim'], out, 'style_share', args, top_n=999)
    def plot():
        import matplotlib.pyplot as plt
        for dim, g in kp.groupby('knowledge_dim'):
            if dim not in dims:
                continue
            g = g.sort_values('month')
            plt.plot(g['month'], _zscore(g['weighted_share']), marker='o', label=str(dim))
        if 'covid_stringency_index' in kp.columns:
            cv = kp[['month', 'covid_stringency_index']].drop_duplicates().sort_values('month')
            plt.plot(cv['month'], _zscore(cv['covid_stringency_index']), linestyle='--', label='stringency')
        elif 'covid_cases_index' in kp.columns:
            cv = kp[['month', 'covid_cases_index']].drop_duplicates().sort_values('month')
            plt.plot(cv['month'], _zscore(cv['covid_cases_index']), linestyle='--', label='cases')
        plt.axvline(args.event_month, linestyle='--')
        plt.xticks(rotation=45)
        plt.title('Style shares and pandemic intensity')
        plt.legend(fontsize=7)
    _maybe_plot(out, 'style_share_vs_pandemic', plot, args.make_figures)
    return {'experiment': 'style_share', 'rows': len(res), 'dims': dims}


def exp_prototype_shift(args, out: Path) -> dict:
    panel = load_prototype_panel(Path(args.social_output_root))
    res = _share_regression_panel(panel, 'sales_share', ['dimension', 'prototype'], out, 'prototype_shift', args, top_n=args.top_n_prototypes)
    # Distribution shift pre/post.
    rows = []
    if not panel.empty:
        ev = pd.Period(args.event_month, freq='M')
        pi = pd.PeriodIndex(panel['month'].astype(str), freq='M')
        d = panel.copy()
        d['period'] = np.where(pi >= ev, 'post', 'pre')
        for dim, g in d.groupby('dimension'):
            dist = g.groupby(['period', 'prototype'])['sales_count'].sum().unstack(fill_value=0.0)
            if {'pre', 'post'}.issubset(dist.index):
                pre = dist.loc['pre'].astype(float); post = dist.loc['post'].astype(float)
                pre = pre / max(pre.sum(), 1e-12); post = post / max(post.sum(), 1e-12)
                m = 0.5 * (pre + post)
                def kl(p, q):
                    mask = (p > 0) & (q > 0)
                    return float((p[mask] * np.log(p[mask] / q[mask])).sum())
                jsd = 0.5 * kl(pre, m) + 0.5 * kl(post, m)
                tv = 0.5 * float(np.abs(pre - post).sum())
                rows.append({'dimension': dim, 'js_divergence': jsd, 'total_variation': tv, 'num_prototypes': int(len(pre))})
    dist_res = pd.DataFrame(rows)
    dist_res.to_parquet(out / 'prototype_distribution_shift.parquet', index=False)
    def plot():
        import matplotlib.pyplot as plt
        if dist_res.empty:
            return
        top = dist_res.sort_values('js_divergence', ascending=False).head(20)
        plt.barh(top['dimension'].astype(str), top['js_divergence'])
        plt.title('Prototype pre/post distribution shift')
    _maybe_plot(out, 'prototype_distribution_shift', plot, args.make_figures)
    return {'experiment': 'prototype_shift', 'rows': len(res), 'dist_rows': len(dist_res)}


def exp_item_continuous_shock(args, out: Path) -> dict:
    im = load_item_panel(Path(args.social_output_root))
    im = _add_time_controls(im, args.fourier_order)
    im['log_sales'] = np.log1p(pd.to_numeric(im['sales_count'], errors='coerce').fillna(0.0))
    dims = _parse_style_dims(args.style_dims)
    covid_vars = [c for c in ['covid_cases_index', 'covid_deaths_index', 'covid_stringency_index'] if c in im.columns]
    rows = []
    for dim in dims:
        sc = _score_col(dim)
        if sc not in im.columns:
            continue
        im[sc] = pd.to_numeric(im[sc], errors='coerce').fillna(0.0)
        for cv in covid_vars:
            tmp = im[['article_id', 'month', 'log_sales', '_trend', sc, cv]].copy()
            tmp[f'{sc}__x__{cv}'] = tmp[sc] * tmp[cv]
            xcols = [cv, f'{sc}__x__{cv}', '_trend']
            # Add Fourier controls if requested.
            for h in range(1, args.fourier_order + 1):
                for kind in ['_sin', '_cos']:
                    cname = f'{kind}{h}'
                    tmp[cname] = im[cname]
                    xcols.append(cname)
            fit = _within_ols_np(tmp, 'log_sales', xcols, group_col='article_id', min_n=100)
            fit.update({'model': 'item_continuous_shock_within_item', 'knowledge_dim': dim, 'covariate': cv, 'interaction': f'{sc}__x__{cv}'})
            rows.append(fit)
    res = pd.DataFrame(rows)
    res.to_parquet(out / 'item_continuous_shock_models.parquet', index=False)
    def plot():
        import matplotlib.pyplot as plt
        if res.empty:
            return
        x = res[res['covariate'].isin(['covid_cases_index', 'covid_deaths_index', 'covid_stringency_index'])].copy()
        vals = []
        labels = []
        for _, r in x.iterrows():
            coef_col = f"coef_{r['interaction']}"
            vals.append(r.get(coef_col, np.nan))
            labels.append(f"{r['knowledge_dim']}×{r['covariate'].replace('covid_','')}")
        order = np.argsort(np.nan_to_num(vals))
        vals = np.array(vals)[order]
        labels = np.array(labels)[order]
        plt.barh(labels, vals)
        plt.title('Item-level style exposure × pandemic coefficients')
    _maybe_plot(out, 'item_continuous_shock_coefficients', plot, args.make_figures)
    return {'experiment': 'item_continuous_shock', 'rows': len(res)}


def exp_pfi_pca(args, out: Path) -> dict:
    kp = load_knowledge_panel(Path(args.social_output_root))
    if kp.empty or 'weighted_share' not in kp.columns:
        _save_empty(out / 'pfi_pca_timeseries.parquet', 'knowledge panel missing')
        return {'experiment': 'pfi_pca', 'status': 'knowledge panel missing'}
    dims = _parse_style_dims(args.style_dims)
    tab = kp[kp['knowledge_dim'].isin(dims)].pivot_table(index='month', columns='knowledge_dim', values='weighted_share', aggfunc='mean').sort_index()
    tab = tab.dropna(axis=1, how='all').fillna(method='ffill').fillna(method='bfill')
    if tab.shape[1] < 2:
        _save_empty(out / 'pfi_pca_timeseries.parquet', 'too few style dimensions')
        return {'experiment': 'pfi_pca', 'status': 'too few style dimensions'}
    X = tab.to_numpy(dtype=float)
    Xz = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
    U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
    scores = U[:, 0] * S[0]
    # orient PC1 so comfort/homewear are positive if possible
    load = pd.Series(Vt[0], index=tab.columns)
    orient_terms = [c for c in load.index if c in ['comfort', 'homewear', 'casual', 'value']]
    if orient_terms and load[orient_terms].mean() < 0:
        scores = -scores
        load = -load
    pfi = pd.DataFrame({'month': tab.index.astype(str), 'pandemic_fashion_index': scores})
    pfi = merge_external(pfi, load_or_build_covid_monthly(Path(args.social_output_root), args.covid_csv, args.covid_location))
    pfi = _add_time_controls(pfi, args.fourier_order)
    pfi.to_parquet(out / 'pfi_pca_timeseries.parquet', index=False)
    load_df = load.reset_index().rename(columns={'knowledge_dim': 'style_dim', 0: 'loading'})
    load_df.columns = ['style_dim', 'loading']
    load_df.to_parquet(out / 'pfi_pca_loadings.parquet', index=False)
    xbase = ['_trend'] + [f'_{k}{h}' for h in range(1, args.fourier_order + 1) for k in ['sin','cos']]
    covs = [c for c in ['covid_cases_index', 'covid_deaths_index', 'covid_stringency_index', 'covid_reproduction_rate'] if c in pfi.columns]
    rows = []
    for cv in covs:
        fit = _ols_np(pfi, 'pandemic_fashion_index', [cv] + xbase, min_n=8)
        fit.update({'model': 'pfi_pca_dynamic_regression', 'covariate': cv})
        rows.append(fit)
    res = pd.DataFrame(rows)
    res.to_parquet(out / 'pfi_pca_models.parquet', index=False)
    def plot():
        import matplotlib.pyplot as plt
        plt.bar(load_df['style_dim'], load_df['loading'])
        plt.xticks(rotation=45)
        plt.title('Pandemic Fashion Index PCA loadings')
    _maybe_plot(out, 'pfi_pca_loadings', plot, args.make_figures)
    def plot2():
        import matplotlib.pyplot as plt
        d = _standardize_for_plot(pfi, ['pandemic_fashion_index'] + covs[:2])
        for c in ['pandemic_fashion_index'] + covs[:2]:
            pc = f'{c}_zplot'
            if pc in d.columns:
                plt.plot(d['month'], d[pc], marker='o', label=c)
        plt.axvline(args.event_month, linestyle='--')
        plt.xticks(rotation=45)
        plt.title('Pandemic Fashion Index and COVID')
        plt.legend()
    _maybe_plot(out, 'pfi_pca_timeseries', plot2, args.make_figures)
    return {'experiment': 'pfi_pca', 'rows': len(res), 'num_dims': tab.shape[1]}


def exp_distribution_shift(args, out: Path) -> dict:
    im = load_item_panel(Path(args.social_output_root))
    ev = pd.Period(args.event_month, freq='M')
    pi = pd.PeriodIndex(im['month'].astype(str), freq='M')
    im = im.copy()
    im['period'] = np.where(pi >= ev, 'post', 'pre')
    dims = _parse_style_dims(args.style_dims)
    rows = []
    density_sample = []
    for dim in dims:
        sc = _score_col(dim)
        if sc not in im.columns:
            continue
        # Expand distribution by sales_count through weighted quantiles approximation.
        pre = im[im['period'] == 'pre'][[sc, 'sales_count']].dropna()
        post = im[im['period'] == 'post'][[sc, 'sales_count']].dropna()
        if pre.empty or post.empty:
            continue
        # Weighted sample for density/ECDF using cap to avoid huge memory.
        def weighted_sample(df, n=20000):
            w = pd.to_numeric(df['sales_count'], errors='coerce').fillna(0).clip(lower=0).to_numpy(dtype=float)
            v = pd.to_numeric(df[sc], errors='coerce').fillna(0).to_numpy(dtype=float)
            if w.sum() <= 0:
                p = None
            else:
                p = w / w.sum()
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(v), size=min(n, max(len(v), 1)), replace=True, p=p)
            return v[idx]
        a = weighted_sample(pre); b = weighted_sample(post)
        # Wasserstein via sorted samples, KS via ECDF grid.
        n = min(len(a), len(b))
        aa = np.sort(a[:n]); bb = np.sort(b[:n])
        w1 = float(np.mean(np.abs(aa - bb))) if n else np.nan
        grid = np.linspace(min(np.nanmin(a), np.nanmin(b)), max(np.nanmax(a), np.nanmax(b)), 100)
        ecdf_a = np.searchsorted(np.sort(a), grid, side='right') / max(len(a), 1)
        ecdf_b = np.searchsorted(np.sort(b), grid, side='right') / max(len(b), 1)
        ks = float(np.max(np.abs(ecdf_a - ecdf_b)))
        q75 = np.nanquantile(np.concatenate([a, b]), 0.75)
        high_pre = float(np.mean(a >= q75)); high_post = float(np.mean(b >= q75))
        rows.append({'knowledge_dim': dim, 'wasserstein_approx': w1, 'ks_approx': ks, 'high_quantile_threshold': float(q75), 'high_share_pre': high_pre, 'high_share_post': high_post, 'high_share_change': high_post - high_pre})
        for period, arr in [('pre', a[: min(5000, len(a))]), ('post', b[: min(5000, len(b))])]:
            density_sample.extend([{'knowledge_dim': dim, 'period': period, 'score': float(x)} for x in arr])
    res = pd.DataFrame(rows)
    res.to_parquet(out / 'style_distribution_shift.parquet', index=False)
    pd.DataFrame(density_sample).to_parquet(out / 'style_distribution_shift_sample.parquet', index=False)
    def plot():
        import matplotlib.pyplot as plt
        if res.empty:
            return
        top = res.sort_values('wasserstein_approx')
        plt.barh(top['knowledge_dim'], top['wasserstein_approx'])
        plt.title('Pre/post style distribution shift')
    _maybe_plot(out, 'style_distribution_shift_wasserstein', plot, args.make_figures)
    return {'experiment': 'distribution_shift', 'rows': len(res)}


def exp_event_study(args, out: Path) -> dict:
    im = load_item_panel(Path(args.social_output_root))
    im['log_sales'] = np.log1p(pd.to_numeric(im['sales_count'], errors='coerce').fillna(0.0))
    pi = pd.PeriodIndex(im['month'].astype(str), freq='M')
    ev = pd.Period(args.event_month, freq='M')
    im['rel_month'] = (pi.year - ev.year) * 12 + (pi.month - ev.month)
    im = im[(im['rel_month'] >= -args.event_window) & (im['rel_month'] <= args.event_window)].copy()
    dims = _parse_style_dims(args.style_dims)
    rows = []
    for dim in dims:
        sc = _score_col(dim)
        if sc not in im.columns:
            continue
        tmp = im[['article_id', 'month', 'rel_month', 'log_sales', sc]].copy()
        tmp[sc] = pd.to_numeric(tmp[sc], errors='coerce').fillna(0.0)
        # Within item: exact month FE are collinear with rel_month dummies alone, but interactions vary by item.
        xcols = []
        for l in range(-args.event_window, args.event_window + 1):
            if l == -1:
                continue
            cname = f'rel_{l:+d}'.replace('+', 'p').replace('-', 'm')
            tmp[cname] = ((tmp['rel_month'] == l).astype(float) * tmp[sc])
            xcols.append(cname)
        # Include month FE via get_dummies; window small so OK. No article dummies: use within item.
        # Demean interactions and y by item; month FE are not included in within to keep speed and avoid huge design.
        fit = _within_ols_np(tmp, 'log_sales', xcols, group_col='article_id', min_n=100)
        for l in range(-args.event_window, args.event_window + 1):
            if l == -1:
                continue
            cname = f'rel_{l:+d}'.replace('+', 'p').replace('-', 'm')
            rows.append({'knowledge_dim': dim, 'rel_month': l, 'coef': fit.get(f'coef_{cname}', np.nan), 'se': fit.get(f'se_{cname}', np.nan), 't': fit.get(f't_{cname}', np.nan), 'nobs': fit.get('nobs', 0), 'status': fit.get('status')})
    res = pd.DataFrame(rows)
    res.to_parquet(out / 'event_study_exposure_results.parquet', index=False)
    def plot():
        import matplotlib.pyplot as plt
        for dim, g in res.groupby('knowledge_dim'):
            g = g.sort_values('rel_month')
            plt.plot(g['rel_month'], g['coef'], marker='o', label=dim)
        plt.axhline(0, linestyle='--')
        plt.axvline(0, linestyle='--')
        plt.title('Event-study-style differential exposure')
        plt.legend(fontsize=7)
    _maybe_plot(out, 'event_study_exposure', plot, args.make_figures)
    return {'experiment': 'event_study', 'rows': len(res)}


def exp_lag_ardl(args, out: Path) -> dict:
    # ARDL on style shares and total sales, not item panel, to keep it fast.
    kp = load_knowledge_panel(Path(args.social_output_root))
    ts_total = load_item_panel(Path(args.social_output_root)).groupby('month', as_index=False)['sales_count'].sum().rename(columns={'sales_count': 'total_sales'})
    covid = load_or_build_covid_monthly(Path(args.social_output_root), args.covid_csv, args.covid_location)
    covs = [c for c in ['covid_cases_index', 'covid_deaths_index', 'covid_stringency_index', 'covid_reproduction_rate'] if c in covid.columns]
    rows = []
    # Total sales ARDL.
    t = merge_external(ts_total, covid).sort_values('month')
    t['log_total_sales'] = np.log1p(t['total_sales'])
    t = _add_time_controls(t, args.fourier_order)
    for cv in covs:
        d = t.copy()
        d['y_lag1'] = d['log_total_sales'].shift(1)
        xcols = ['y_lag1']
        for lag in range(0, args.max_lag + 1):
            cname = cv if lag == 0 else f'{cv}_lag{lag}'
            d[cname] = d[cv].shift(lag)
            xcols.append(cname)
        fit = _ols_np(d, 'log_total_sales', xcols + ['_trend'], min_n=8)
        fit.update({'model': 'ardl_total_sales', 'target': 'total_sales', 'covariate': cv})
        rows.append(fit)
    # Style share ARDL.
    if not kp.empty:
        dims = _parse_style_dims(args.style_dims)
        for dim, g in kp[kp['knowledge_dim'].isin(dims)].groupby('knowledge_dim'):
            g = merge_external(g[['month', 'weighted_share']], covid).sort_values('month')
            g = _add_time_controls(g, args.fourier_order)
            for cv in covs:
                d = g.copy()
                d['y_lag1'] = d['weighted_share'].shift(1)
                xcols = ['y_lag1']
                for lag in range(0, args.max_lag + 1):
                    cname = cv if lag == 0 else f'{cv}_lag{lag}'
                    d[cname] = d[cv].shift(lag)
                    xcols.append(cname)
                fit = _ols_np(d, 'weighted_share', xcols + ['_trend'], min_n=8)
                fit.update({'model': 'ardl_style_share', 'target': dim, 'covariate': cv})
                rows.append(fit)
    res = pd.DataFrame(rows)
    res.to_parquet(out / 'ardl_lag_models.parquet', index=False)
    return {'experiment': 'lag_ardl', 'rows': len(res), 'covariates': covs}


def exp_interrupted_ts(args, out: Path) -> dict:
    # Run ITS on total sales, channel share, and selected style shares.
    rows = []
    ev = pd.Period(args.event_month, freq='M')
    def add_its_vars(df):
        d = _add_time_controls(df, args.fourier_order)
        pi = pd.PeriodIndex(d['month'].astype(str), freq='M')
        rel = (pi.year - ev.year) * 12 + (pi.month - ev.month)
        d['post_event'] = (rel >= 0).astype(float)
        d['time_after_event'] = np.maximum(rel, 0).astype(float)
        return d
    # total
    im = load_item_panel(Path(args.social_output_root))
    ts = im.groupby('month', as_index=False)['sales_count'].sum().rename(columns={'sales_count': 'total_sales'})
    ts['log_total_sales'] = np.log1p(ts['total_sales'])
    ts = add_its_vars(ts)
    fit = _ols_np(ts, 'log_total_sales', ['_trend', 'post_event', 'time_after_event'] + [f'_{k}{h}' for h in range(1, args.fourier_order + 1) for k in ['sin','cos']], min_n=8)
    fit.update({'model': 'interrupted_time_series', 'target': 'log_total_sales'})
    rows.append(fit)
    # style
    kp = load_knowledge_panel(Path(args.social_output_root))
    dims = _parse_style_dims(args.style_dims)
    if not kp.empty:
        for dim, g in kp[kp['knowledge_dim'].isin(dims)].groupby('knowledge_dim'):
            g = add_its_vars(g.sort_values('month'))
            fit = _ols_np(g, 'weighted_share', ['_trend', 'post_event', 'time_after_event'] + [f'_{k}{h}' for h in range(1, args.fourier_order + 1) for k in ['sin','cos']], min_n=8)
            fit.update({'model': 'interrupted_time_series', 'target': f'style_{dim}'})
            rows.append(fit)
    res = pd.DataFrame(rows)
    res.to_parquet(out / 'interrupted_time_series_models.parquet', index=False)
    return {'experiment': 'interrupted_ts', 'rows': len(res)}


def exp_seasonality(args, out: Path) -> dict:
    results = []
    sources = [
        ('category', load_category_panel(Path(args.social_output_root)), ['category_field', 'category_value'], 'sales_share'),
        ('prototype', load_prototype_panel(Path(args.social_output_root)), ['dimension', 'prototype'], 'sales_share'),
        ('knowledge', load_knowledge_panel(Path(args.social_output_root)), ['knowledge_dim'], 'weighted_share'),
    ]
    for prefix, df, id_cols, ycol in sources:
        if df.empty or ycol not in df.columns:
            continue
        df = _add_time_controls(df, args.fourier_order)
        xcols = ['_trend'] + [f'_{k}{h}' for h in range(1, args.fourier_order + 1) for k in ['sin','cos']]
        rows = []
        # cap top groups for speed
        if 'sales_count' in df.columns:
            rank = df.groupby(id_cols, dropna=False)['sales_count'].sum().reset_index().sort_values('sales_count', ascending=False).head(max(args.top_n_categories, args.top_n_prototypes, 30))
            df2 = df.merge(rank[id_cols], on=id_cols, how='inner')
        else:
            df2 = df
        for key, g in df2.groupby(id_cols, dropna=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            fit = _ols_np(g, ycol, xcols, min_n=max(8, len(xcols) + 2))
            for c, v in zip(id_cols, key_tuple):
                fit[c] = v
            if fit.get('status') == 'ok':
                amp = 0.0
                for h in range(1, args.fourier_order + 1):
                    a = fit.get(f'coef__sin{h}', np.nan)
                    b = fit.get(f'coef__cos{h}', np.nan)
                    if np.isfinite(a) and np.isfinite(b):
                        amp += a*a + b*b
                fit['seasonality_amplitude'] = float(np.sqrt(amp))
            rows.append(fit)
        res = pd.DataFrame(rows)
        res.to_parquet(out / f'seasonality_{prefix}_fourier_results.parquet', index=False)
        results.append({'source': prefix, 'rows': len(res)})
    return {'experiment': 'seasonality', 'results': results}


def exp_channel_style(args, out: Path) -> dict:
    tx = load_transactions(Path(args.data_root))
    if 'sales_channel_id' not in tx.columns:
        return {'experiment': 'channel_style', 'status': 'sales_channel_id missing'}
    scores_path = Path(args.social_output_root) / 'item_knowledge_scores.parquet'
    if not scores_path.exists():
        return {'experiment': 'channel_style', 'status': 'item_knowledge_scores missing'}
    scores = pd.read_parquet(scores_path)
    scores['article_id'] = scores['article_id'].map(_article_key)
    dims = _parse_style_dims(args.style_dims)
    cols = ['article_id'] + [_score_col(d) for d in dims if _score_col(d) in scores.columns]
    tx = tx.merge(scores[cols], on='article_id', how='left')
    rows = []
    for ch, gch in tx.groupby('sales_channel_id'):
        total = gch.groupby('month').size().rename('total').reset_index()
        for dim in dims:
            sc = _score_col(dim)
            if sc not in gch.columns:
                continue
            tmp = gch.copy()
            tmp[sc] = pd.to_numeric(tmp[sc], errors='coerce').fillna(0.0)
            tmp['_weighted'] = tmp[sc]
            m = tmp.groupby('month', as_index=False)['_weighted'].sum().rename(columns={'_weighted': 'weighted_sales'})
            m = m.merge(total, on='month', how='left')
            m['style_share'] = m['weighted_sales'] / m['total'].replace(0, np.nan)
            m['sales_channel_id'] = ch
            m['knowledge_dim'] = dim
            rows.append(m)
    panel = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if panel.empty:
        return {'experiment': 'channel_style', 'status': 'empty channel-style panel'}
    panel = merge_external(panel, load_or_build_covid_monthly(Path(args.social_output_root), args.covid_csv, args.covid_location))
    panel.to_parquet(out / 'channel_style_panel.parquet', index=False)
    panel = _add_time_controls(panel, args.fourier_order)
    covs = [c for c in ['covid_cases_index', 'covid_deaths_index', 'covid_stringency_index'] if c in panel.columns]
    res_rows = []
    for (ch, dim), g in panel.groupby(['sales_channel_id', 'knowledge_dim']):
        for cv in covs:
            fit = _ols_np(g, 'style_share', [cv, '_trend'] + [f'_{k}{h}' for h in range(1, args.fourier_order + 1) for k in ['sin','cos']], min_n=8)
            fit.update({'model': 'channel_style_regression', 'sales_channel_id': ch, 'knowledge_dim': dim, 'covariate': cv})
            res_rows.append(fit)
    res = pd.DataFrame(res_rows)
    res.to_parquet(out / 'channel_style_models.parquet', index=False)
    def plot():
        import matplotlib.pyplot as plt
        pivot = panel.copy()
        ev = pd.Period(args.event_month, freq='M')
        pi = pd.PeriodIndex(pivot['month'].astype(str), freq='M')
        pivot['period'] = np.where(pi >= ev, 'post', 'pre')
        tab = pivot.groupby(['sales_channel_id','knowledge_dim','period'])['style_share'].mean().unstack()
        if {'pre', 'post'}.issubset(tab.columns):
            tab['delta'] = tab['post'] - tab['pre']
            top = tab.reset_index().pivot(index='knowledge_dim', columns='sales_channel_id', values='delta')
            plt.imshow(top.fillna(0).to_numpy(), aspect='auto')
            plt.yticks(range(len(top.index)), top.index)
            plt.xticks(range(len(top.columns)), [f'ch {c}' for c in top.columns])
            plt.colorbar(label='post - pre')
            plt.title('Channel × style pre/post change')
    _maybe_plot(out, 'channel_style_prepost_heatmap', plot, args.make_figures)
    return {'experiment': 'channel_style', 'rows': len(res)}


def exp_age_heterogeneity(args, out: Path) -> dict:
    tx = load_transactions(Path(args.data_root))
    cust_path = Path(args.data_root) / 'hm' / 'processed' / 'hm_customers.parquet'
    if not cust_path.exists() or 'customer_id' not in tx.columns:
        return {'experiment': 'age_heterogeneity', 'status': 'customers or customer_id missing'}
    cust = pd.read_parquet(cust_path)
    if 'age' not in cust.columns:
        return {'experiment': 'age_heterogeneity', 'status': 'age missing'}
    scores_path = Path(args.social_output_root) / 'item_knowledge_scores.parquet'
    if not scores_path.exists():
        return {'experiment': 'age_heterogeneity', 'status': 'item_knowledge_scores missing'}
    scores = pd.read_parquet(scores_path)
    scores['article_id'] = scores['article_id'].map(_article_key)
    tx = tx.merge(cust[['customer_id', 'age']], on='customer_id', how='left')
    tx['age'] = pd.to_numeric(tx['age'], errors='coerce')
    tx['age_group'] = pd.cut(tx['age'], bins=[0, 25, 40, 60, 120], labels=['<=25','26-40','41-60','60+'])
    dims = _parse_style_dims(args.style_dims)
    cols = ['article_id'] + [_score_col(d) for d in dims if _score_col(d) in scores.columns]
    tx = tx.merge(scores[cols], on='article_id', how='left')
    rows = []
    for (ag, month), gm in tx.dropna(subset=['age_group']).groupby(['age_group', 'month']):
        total = len(gm)
        row = {'age_group': str(ag), 'month': month, 'total_transactions': total}
        for dim in dims:
            sc = _score_col(dim)
            if sc in gm.columns:
                row[f'{dim}_share'] = float(pd.to_numeric(gm[sc], errors='coerce').fillna(0.0).sum() / max(total, 1))
        rows.append(row)
    panel = pd.DataFrame(rows)
    panel = merge_external(panel, load_or_build_covid_monthly(Path(args.social_output_root), args.covid_csv, args.covid_location))
    panel.to_parquet(out / 'age_style_panel.parquet', index=False)
    panel = _add_time_controls(panel, args.fourier_order)
    covs = [c for c in ['covid_cases_index', 'covid_deaths_index', 'covid_stringency_index'] if c in panel.columns]
    res_rows = []
    for ag, g in panel.groupby('age_group'):
        for dim in dims:
            y = f'{dim}_share'
            if y not in g.columns:
                continue
            for cv in covs:
                fit = _ols_np(g, y, [cv, '_trend'] + [f'_{k}{h}' for h in range(1, args.fourier_order + 1) for k in ['sin','cos']], min_n=8)
                fit.update({'model': 'age_style_regression', 'age_group': ag, 'knowledge_dim': dim, 'covariate': cv})
                res_rows.append(fit)
    res = pd.DataFrame(res_rows)
    res.to_parquet(out / 'age_style_models.parquet', index=False)
    return {'experiment': 'age_heterogeneity', 'rows': len(res)}


def exp_count_models(args, out: Path) -> dict:
    try:
        import statsmodels.api as sm
    except Exception as exc:
        _save_empty(out / 'count_models.parquet', f'statsmodels unavailable: {exc}')
        return {'experiment': 'count_models', 'status': f'statsmodels unavailable: {exc}'}
    # Prefer aggregated knowledge panel for speed.
    kp = load_knowledge_panel(Path(args.social_output_root))
    if kp.empty:
        return {'experiment': 'count_models', 'status': 'knowledge panel missing'}
    dims = _parse_style_dims(args.style_dims)
    kp = kp[kp['knowledge_dim'].isin(dims)].copy()
    kp = _add_time_controls(kp, args.fourier_order)
    cov = 'covid_cases_index' if 'covid_cases_index' in kp.columns else None
    if cov is None:
        return {'experiment': 'count_models', 'status': 'covid_cases_index missing'}
    rows = []
    for dim, g in kp.groupby('knowledge_dim'):
        y = pd.to_numeric(g.get('weighted_sales', g.get('raw_sales')), errors='coerce').fillna(0.0)
        X = pd.DataFrame({'const': 1.0, cov: pd.to_numeric(g[cov], errors='coerce').fillna(0.0), '_trend': g['_trend']})
        for h in range(1, args.fourier_order + 1):
            X[f'_sin{h}'] = g[f'_sin{h}']; X[f'_cos{h}'] = g[f'_cos{h}']
        for fam_name, fam in [('poisson', sm.families.Poisson()), ('negative_binomial', sm.families.NegativeBinomial())]:
            try:
                fit = sm.GLM(y, X, family=fam).fit(maxiter=100, disp=False)
                rows.append({'model': fam_name, 'knowledge_dim': dim, 'covariate': cov, 'coef': float(fit.params.get(cov, np.nan)), 'se': float(fit.bse.get(cov, np.nan)), 'pvalue': float(fit.pvalues.get(cov, np.nan)), 'nobs': int(fit.nobs), 'status': 'ok'})
            except Exception as exc:
                rows.append({'model': fam_name, 'knowledge_dim': dim, 'status': f'error:{type(exc).__name__}:{exc}'})
    res = pd.DataFrame(rows)
    res.to_parquet(out / 'count_models.parquet', index=False)
    return {'experiment': 'count_models', 'rows': len(res)}


def exp_ml_heterogeneity(args, out: Path) -> dict:
    im = load_item_panel(Path(args.social_output_root))
    dims = _parse_style_dims(args.style_dims)
    covid_var = 'covid_cases_index' if 'covid_cases_index' in im.columns else None
    if covid_var is None:
        return {'experiment': 'ml_heterogeneity', 'status': 'covid_cases_index missing'}
    # Estimate item-level COVID response using simple covariance slope per item.
    rows = []
    for aid, g in im.groupby('article_id', sort=False):
        if g['month'].nunique() < 6 or g['sales_count'].sum() <= 0:
            continue
        x = pd.to_numeric(g[covid_var], errors='coerce').astype(float)
        y = np.log1p(pd.to_numeric(g['sales_count'], errors='coerce').fillna(0.0).astype(float))
        if x.std() <= 1e-12:
            continue
        beta = float(np.cov(x, y, ddof=0)[0,1] / max(np.var(x), 1e-12))
        row = {'article_id': aid, 'covid_response_beta': beta, 'total_sales': float(g['sales_count'].sum())}
        first = g.iloc[0]
        for dim in dims:
            sc = _score_col(dim)
            if sc in g.columns:
                row[sc] = float(first.get(sc, np.nan))
        for c in ['product_group_name', 'product_type_name', 'garment_group_name', 'section_name', 'colour_group_name']:
            if c in g.columns:
                row[c] = first.get(c)
        rows.append(row)
        if len(rows) >= args.ml_max_items:
            break
    data = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=['covid_response_beta'])
    data.to_parquet(out / 'item_covid_response_estimates.parquet', index=False)
    if len(data) < 50:
        return {'experiment': 'ml_heterogeneity', 'status': 'too few item responses', 'items': len(data)}
    feature_cols = [_score_col(d) for d in dims if _score_col(d) in data.columns]
    cat_cols = [c for c in ['product_group_name', 'garment_group_name', 'section_name'] if c in data.columns]
    X = data[feature_cols].copy()
    for c in cat_cols:
        top = data[c].astype(str).value_counts().head(20).index
        for v in top:
            X[f'{c}={_safe_name(v)}'] = (data[c].astype(str) == v).astype(float)
    X = X.fillna(0.0)
    y = data['covid_response_beta'].astype(float)
    summary_rows = []
    try:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=args.seed)
        models = [
            ('random_forest', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=args.seed, n_jobs=-1)),
            ('gradient_boosting', GradientBoostingRegressor(random_state=args.seed, max_depth=3)),
        ]
        for name, model in models:
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            summary_rows.append({'model': name, 'r2': float(r2_score(yte, pred)), 'rmse': float(mean_squared_error(yte, pred, squared=False)), 'n_train': len(Xtr), 'n_test': len(Xte)})
            if hasattr(model, 'feature_importances_'):
                imp = pd.DataFrame({'model': name, 'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
                imp.to_parquet(out / f'{name}_feature_importance.parquet', index=False)
    except Exception as exc:
        summary_rows.append({'model': 'sklearn_models', 'status': f'skipped/error:{type(exc).__name__}:{exc}'})
    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(out / 'ml_heterogeneity_summary.parquet', index=False)
    return {'experiment': 'ml_heterogeneity', 'items': len(data), 'models': len(summary)}


def exp_mobility_mechanism(args, out: Path) -> dict:
    if not args.mobility_csv:
        return {'experiment': 'mobility_mechanism', 'status': 'mobility_csv not provided'}
    mobility = load_mobility_monthly(args.mobility_csv, args.mobility_country, args.mobility_region)
    if mobility.empty or len(mobility.columns) <= 1:
        return {'experiment': 'mobility_mechanism', 'status': 'empty mobility data'}
    kp = load_knowledge_panel(Path(args.social_output_root))
    if kp.empty:
        return {'experiment': 'mobility_mechanism', 'status': 'knowledge panel missing'}
    dims = [d for d in _parse_style_dims(args.style_dims) if d in ['office', 'formal', 'homewear', 'comfort']]
    panel = kp[kp['knowledge_dim'].isin(dims)].copy()
    panel = merge_external(panel, load_or_build_covid_monthly(Path(args.social_output_root), args.covid_csv, args.covid_location), mobility)
    panel.to_parquet(out / 'mobility_style_panel.parquet', index=False)
    panel = _add_time_controls(panel, args.fourier_order)
    covs = [c for c in ['mobility_workplaces', 'mobility_residential', 'mobility_retail_recreation', 'covid_cases_index'] if c in panel.columns]
    rows = []
    for dim, g in panel.groupby('knowledge_dim'):
        for cv in covs:
            fit = _ols_np(g, 'weighted_share', [cv, '_trend'] + [f'_{k}{h}' for h in range(1, args.fourier_order + 1) for k in ['sin','cos']], min_n=5)
            fit.update({'model': 'mobility_mechanism_regression', 'knowledge_dim': dim, 'covariate': cv})
            rows.append(fit)
    res = pd.DataFrame(rows)
    res.to_parquet(out / 'mobility_mechanism_models.parquet', index=False)
    return {'experiment': 'mobility_mechanism', 'rows': len(res), 'overlap_months': int(panel['month'].nunique())}

# -----------------------------------------------------------------------------
# Registry and dispatcher
# -----------------------------------------------------------------------------

EXPERIMENTS: dict[str, tuple[str, Callable[[argparse.Namespace, Path], dict]]] = {
    '00': ('panel_check', exp_panel_check),
    '01': ('total_sales', exp_total_sales),
    '02': ('channel_shift', exp_channel_shift),
    '03': ('category_structure', exp_category_structure),
    '04': ('style_share', exp_style_share),
    '05': ('prototype_shift', exp_prototype_shift),
    '06': ('item_continuous_shock', exp_item_continuous_shock),
    '07': ('pfi_pca', exp_pfi_pca),
    '08': ('distribution_shift', exp_distribution_shift),
    '09': ('event_study', exp_event_study),
    '10': ('lag_ardl', exp_lag_ardl),
    '11': ('interrupted_ts', exp_interrupted_ts),
    '12': ('seasonality', exp_seasonality),
    '13': ('channel_style', exp_channel_style),
    '14': ('age_heterogeneity', exp_age_heterogeneity),
    '15': ('count_models', exp_count_models),
    '16': ('ml_heterogeneity', exp_ml_heterogeneity),
    '17': ('mobility_mechanism', exp_mobility_mechanism),
}

NAME_TO_ID = {name: k for k, (name, _) in EXPERIMENTS.items()}


def list_experiments() -> pd.DataFrame:
    return pd.DataFrame([{'exp_id': k, 'exp_name': name} for k, (name, _) in sorted(EXPERIMENTS.items())])


def resolve_experiment(exp_id: str = '', exp_name: str = '') -> tuple[str, str, Callable[[argparse.Namespace, Path], dict]]:
    eid = str(exp_id or '').strip()
    ename = str(exp_name or '').strip()
    if ename:
        if ename not in NAME_TO_ID:
            raise ValueError(f'Unknown exp_name={ename}. Available: {sorted(NAME_TO_ID)}')
        eid = NAME_TO_ID[ename]
    if not eid:
        eid = '00'
    eid = eid.zfill(2) if eid.isdigit() else eid
    if eid not in EXPERIMENTS:
        raise ValueError(f'Unknown exp_id={eid}. Available: {sorted(EXPERIMENTS)}')
    name, fn = EXPERIMENTS[eid]
    return eid, name, fn


def run_single_experiment(args) -> None:
    social_root = Path(args.social_output_root)
    analysis_root = ensure_dir(social_root / 'analysis')
    if bool(args.list_experiments):
        df = list_experiments()
        
        try:
            df.to_parquet(analysis_root / 'experiment_registry.parquet', index=False)
        except Exception:
            df.to_csv(analysis_root / 'experiment_registry.csv', index=False)
        print(df.to_string(index=False))
        return
    eid, name, fn = resolve_experiment(args.exp_id, args.exp_name)
    exp_dir = ensure_dir(analysis_root / f'{eid}_{name}')
    print(f'[social_analysis] Running exp_id={eid}, exp_name={name}')
    result = fn(args, exp_dir)
    result.update({
        'exp_id': eid,
        'exp_name': name,
        'social_output_root': str(social_root),
        'output_dir': str(exp_dir),
    })
    save_json(result, exp_dir / 'manifest.json')
    # A flat index file makes it easy to find outputs.
    idx = analysis_root / 'last_experiment.json'
    save_json(result, idx)
    print('========== Social experiment finished =========')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f'Output: {exp_dir}')
    print('==============================================')


def run_covid_social_models(args) -> None:
    """Backward-compatible name. Now dispatches exactly one experiment by exp_id/exp_name."""
    run_single_experiment(args)


def add_social_analysis_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument('--social_output_root', default='./social_output/k')
    p.add_argument('--data_root', default='./data')
    p.add_argument('--exp_id', default='', help='Experiment id. Use --list_experiments 1 to view all ids.')
    p.add_argument('--exp_name', default='', help='Experiment name, e.g. total_sales, style_share, pfi_pca.')
    p.add_argument('--list_experiments', type=parse_bool, default=False)
    p.add_argument('--covid_csv', default='')
    p.add_argument('--covid_location', default='World')
    p.add_argument('--mobility_csv', default='')
    p.add_argument('--mobility_country', default='')
    p.add_argument('--mobility_region', default='')
    p.add_argument('--event_month', default='2020-03')
    p.add_argument('--event_window', type=int, default=8)
    p.add_argument('--style_dims', default='formal,office,comfort,homewear,casual,value')
    p.add_argument('--max_lag', type=int, default=2)
    p.add_argument('--fourier_order', type=int, default=2)
    p.add_argument('--top_n_categories', type=int, default=30)
    p.add_argument('--top_n_prototypes', type=int, default=40)
    p.add_argument('--ml_max_items', type=int, default=20000)
    p.add_argument('--count_model_max_rows', type=int, default=250000)  # kept for compatibility
    p.add_argument('--make_figures', type=parse_bool, default=True)
    p.add_argument('--seed', type=int, default=42)
    return p


def main():
    p = argparse.ArgumentParser(description='Run one social-science experiment at a time.')
    add_social_analysis_args(p)
    # Deprecated compatibility flags are accepted but no longer trigger all-in-one execution.
    p.add_argument('--run_count_models', type=parse_bool, default=False)
    p.add_argument('--run_cox', type=parse_bool, default=False)
    p.add_argument('--cox_max_rows', type=int, default=200000)
    args = p.parse_args()
    run_single_experiment(args)


if __name__ == '__main__':
    main()
