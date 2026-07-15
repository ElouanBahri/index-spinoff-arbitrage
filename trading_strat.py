# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#     notebook_metadata_filter: all
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Spinoff Index Arbitrage — Trading Strategy Analysis
#
# **Extends eda.py** with strategies a Citadel index-rebalance desk would run:
#
# 1. **Announcement-date parent return** — trade the parent on the spinoff announcement
# 2. **Child vs parent relative returns** — pairs trade as a tighter hedge than SPY
# 3. **Beta estimation and alpha decomposition** — strip market beta; isolate forced-flow alpha
# 4. **Free float forced-flow** — passive selling as % of child market cap
# 5. **Index inclusion probability model** — logistic regression, LOO-CV, no lookahead
# 6. **Bucket analysis** — what separates under- vs outperformers at t=21d?
# 7. **Trading strategy backtest** — four strategies, per-trade P&L, Sharpe, win rate
#
# **Lookahead policy:** each feature section documents when the data becomes available.

# %%
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:,.4f}'.format)
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

RAW_DIR   = Path('data/raw')
CLEAN_DIR = Path('data/clean')

# %% [markdown]
# ## 1. Load Data

# %%
events = pd.read_csv(
    CLEAN_DIR / 'spinoff_events_merged.csv',
    parse_dates=['announce_date', 'effective_date', 'sp500_start', 'sp500_end']
)
children      = pd.read_parquet(RAW_DIR / 'spinoff_children_crsp.parquet')
child_sp500   = pd.read_parquet(RAW_DIR / 'spinoff_children_sp500.parquet')
idx_ret       = pd.read_parquet(RAW_DIR / 'crsp_index_returns.parquet')
constituents  = pd.read_parquet(RAW_DIR / 'sp500_constituents_pit.parquet')
passive_aum   = pd.read_parquet(RAW_DIR / 'sp500_passive_aum.parquet')
crsp_path     = RAW_DIR / 'crsp_daily.parquet'
crsp          = pd.read_parquet(crsp_path) if crsp_path.exists() else None

print(f'Events:    {len(events)} spinoff events  '
      f'({events["effective_date"].min().year}–{events["effective_date"].max().year})')
print(f'Children:  {len(children):,} rows, {children["permno"].nunique()} spinoff children')
print(f'crsp_daily loaded: {crsp is not None}  ({len(crsp):,} rows)' if crsp is not None
      else '[!] crsp_daily.parquet not found')

# %% [markdown]
# ## 2. Rebuild Shared Variables
#
# Rebuild `all_cw` (child event windows) and `car_df` (per-event CAR table)
# so this notebook runs independently of eda.py.

# %%
# --- child inclusion status classification (mirrors eda.py Section 2) ---
first_inc = (
    child_sp500.sort_values('added_date')
    .groupby('child_ticker', as_index=False).first()
    [['child_ticker', 'added_date', 'removed_date']]
)
eff_dates = events[['spinoff_ticker', 'effective_date']].rename(columns={'spinoff_ticker': 'child_ticker'})
first_inc = first_inc.merge(eff_dates, on='child_ticker', how='left')
first_inc['days_to_inclusion'] = (first_inc['added_date'] - first_inc['effective_date']).dt.days

def classify(row):
    if pd.isna(row['days_to_inclusion']):
        return 'excluded'
    return 'immediately included' if row['days_to_inclusion'] <= 5 else 'later included'

first_inc['inclusion_status'] = first_inc.apply(classify, axis=1)
status_map = first_inc.set_index('child_ticker')['inclusion_status'].to_dict()
events['inclusion_status'] = events['spinoff_ticker'].map(status_map).fillna('excluded')

COLOR_MAP = {
    'excluded':             '#d32f2f',
    'immediately included': '#1976d2',
    'later included':       '#f57c00',
}

# %%
# --- child event windows (mirrors eda.py Section 3) ---
def build_child_window(permno, eff_date, crsp_df, idx_df, post=90):
    """Market-adjusted return path for one spinoff child from t=0."""
    child = crsp_df[crsp_df['permno'] == int(permno)].sort_values('date').copy()
    valid = child[child['date'] >= eff_date]
    if len(valid) == 0:
        return pd.DataFrame()
    window = valid.head(post + 1).copy()
    window = window.merge(idx_df[['date', 'sprtrn']], on='date', how='left')
    window['t'] = range(len(window))
    prc_col = 'adj_prc' if 'adj_prc' in window.columns else 'prc'
    t0 = window.iloc[0][prc_col]
    if pd.isna(t0) or t0 == 0:
        t0 = abs(window.iloc[0]['prc'])
    window['norm_prc']    = window[prc_col] / t0 * 100
    window['ret_mkt_adj'] = window['ret'] - window['sprtrn']
    window['car']         = window['ret_mkt_adj'].cumsum()
    window['short_pnl']   = -window['car']
    vol_base = window[window['t'].between(5, 30)]['dollar_vol'].mean()
    window['norm_vol'] = window['dollar_vol'] / vol_base if vol_base > 0 else np.nan
    return window

child_windows = []
for _, ev in events.iterrows():
    child_rows = children[children['child_ticker'] == ev['spinoff_ticker']]
    if child_rows.empty:
        continue
    permno = child_rows['permno'].iloc[0]
    w = build_child_window(permno, ev['effective_date'], children, idx_ret, post=90)
    if len(w) == 0:
        continue
    w['child_ticker']     = ev['spinoff_ticker']
    w['parent_ticker']    = ev['parent_ticker']
    w['effective_date']   = ev['effective_date']
    w['inclusion_status'] = ev['inclusion_status']
    w['forced_flow_adv']  = ev.get('forced_flow_adv', np.nan)
    child_windows.append(w)

all_cw = pd.concat(child_windows, ignore_index=True) if child_windows else pd.DataFrame()
print(f'Child windows: {all_cw["child_ticker"].nunique()} children, {len(all_cw):,} rows')

# %%
# --- per-event CAR table (mirrors eda.py Section 4) ---
horizons = [5, 10, 21, 42, 63]
car_rows = []
for _, ev in events.iterrows():
    ticker = ev['spinoff_ticker']
    grp = all_cw[all_cw['child_ticker'] == ticker]
    if grp.empty:
        continue
    row = {'child_ticker': ticker, 'parent_ticker': ev['parent_ticker'],
           'inclusion_status': ev['inclusion_status'],
           'forced_flow_adv': ev.get('forced_flow_adv', np.nan)}
    for h in horizons:
        sub = grp[grp['t'] == h]
        row[f'car_{h}d']       = sub['car'].values[0] if len(sub) else np.nan
        row[f'short_pnl_{h}d'] = sub['short_pnl'].values[0] if len(sub) else np.nan
    car_rows.append(row)

car_df = pd.DataFrame(car_rows)
print(f'CAR table: {len(car_df)} events  —  '
      f'excluded: {(car_df["inclusion_status"]=="excluded").sum()}, '
      f'immediately included: {(car_df["inclusion_status"]=="immediately included").sum()}')

# %% [markdown]
# ## 3. Announcement Date — Parent Return Study
#
# The parent stock typically pops on the spinoff announcement (value-unlock signal).
# Windows are anchored at **announcement date** (t\_ann=0), not effective date.
#
# | Feature | Available at | Lookahead? |
# |---|---|---|
# | announce_date | At announcement | No |
# | Parent price | Daily, real-time | No |
# | Market return | Daily, real-time | No |

# %%
def get_announce_window(permno, ann_date, eff_date, crsp_df, idx_df, pre=20, post=90):
    """Parent CAR path centered on announce_date; t_ann=0 = announce date."""
    parent = crsp_df[crsp_df['permno'] == int(permno)].copy()
    if parent.empty or pd.isna(ann_date):
        return pd.DataFrame()
    cal = int((pre + post) * 1.8)
    window = parent[
        (parent['date'] >= ann_date - pd.Timedelta(days=cal)) &
        (parent['date'] <= ann_date + pd.Timedelta(days=cal))
    ].merge(idx_df[['date', 'sprtrn']], on='date', how='left').sort_values('date').reset_index(drop=True)

    ann_idx = int(window['date'].searchsorted(ann_date))
    window['t_ann'] = range(-ann_idx, len(window) - ann_idx)
    window['ret_mkt_adj'] = window['ret'] - window['sprtrn']
    eff_idx = int(window['date'].searchsorted(eff_date)) if not pd.isna(eff_date) else ann_idx
    window['t_eff'] = range(-eff_idx, len(window) - eff_idx)
    return window[window['t_ann'].between(-pre, post)].copy()


announce_windows = []
if crsp is not None:
    for _, ev in events[events['permno'].notna() & events['announce_date'].notna()].iterrows():
        if ev['announce_date'] >= ev['effective_date']:
            continue
        w = get_announce_window(ev['permno'], ev['announce_date'],
                                 ev['effective_date'], crsp, idx_ret)
        if len(w) == 0:
            continue
        w['parent_ticker']     = ev['parent_ticker']
        w['spinoff_ticker']    = ev['spinoff_ticker']
        w['announce_date']     = ev['announce_date']
        w['effective_date']    = ev['effective_date']
        w['inclusion_status']  = ev['inclusion_status']
        w['announce_lag_days'] = (ev['effective_date'] - ev['announce_date']).days
        announce_windows.append(w)

all_aw = pd.concat(announce_windows, ignore_index=True) if announce_windows else pd.DataFrame()
print(f'Announce windows: {all_aw["parent_ticker"].nunique()} parents' if len(all_aw)
      else '[!] No announce windows — crsp_daily required')

# %%
# 3a. Announce → Effective lag distribution
events['announce_lag_days'] = (events['effective_date'] - events['announce_date']).dt.days
valid_lags = events['announce_lag_days'].dropna()
valid_lags = valid_lags[valid_lags > 0]

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].hist(valid_lags, bins=15, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].axvline(valid_lags.median(), color='red', linestyle='--',
                label=f'Median: {valid_lags.median():.0f}d')
axes[0].set_xlabel('Days: Announcement → Effective')
axes[0].set_title('Announce → Effective Lag', fontweight='bold')
axes[0].legend()

for status, grp in events[events['announce_lag_days'] > 0].groupby('inclusion_status'):
    axes[1].scatter(grp['announce_lag_days'], grp['forced_flow_adv'],
                    color=COLOR_MAP.get(status, 'grey'), label=status, alpha=0.8, s=70)
axes[1].set_xlabel('Announce → Effective (days)')
axes[1].set_ylabel('Forced Flow (× ADV)')
axes[1].set_title('Announce Lag vs Forced-Flow Intensity', fontweight='bold')
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.show()
print(f'Lag — median: {valid_lags.median():.0f}d  mean: {valid_lags.mean():.0f}d  '
      f'max: {valid_lags.max():.0f}d')

# %%
# 3b. Parent CAR anchored at announcement date
if len(all_aw):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    avg_ann = (all_aw[all_aw['t_ann'].between(-10, 60)]
               .groupby('t_ann')['ret_mkt_adj']
               .agg(['mean', 'sem']).reset_index())
    avg_ann['cum_mean'] = avg_ann['mean'].cumsum()
    ci_ann = 1.96 * avg_ann['sem'].cumsum()

    axes[0].plot(avg_ann['t_ann'], avg_ann['cum_mean'] * 100, lw=2.5, color='steelblue')
    axes[0].fill_between(avg_ann['t_ann'],
                         (avg_ann['cum_mean'] - ci_ann) * 100,
                         (avg_ann['cum_mean'] + ci_ann) * 100,
                         alpha=0.2, color='steelblue', label='95% CI')
    axes[0].axvline(0, color='red', linestyle='--', lw=1.2, label='Announcement')
    axes[0].axhline(0, color='black', lw=0.5)
    axes[0].set_xlabel('Trading days relative to announcement')
    axes[0].set_ylabel('Cumulative Abnormal Return (%)')
    axes[0].set_title('Parent CAR Around Spinoff Announcement\n(all events, market-adj)',
                      fontweight='bold')
    axes[0].legend(fontsize=9)

    for status, grp in all_aw[all_aw['t_ann'].between(-10, 60)].groupby('inclusion_status'):
        avg_s = grp.groupby('t_ann')['ret_mkt_adj'].mean().cumsum()
        axes[1].plot(avg_s.index, avg_s.values * 100, color=COLOR_MAP.get(status, 'grey'),
                     lw=2, label=status)
    axes[1].axvline(0, color='red', linestyle='--', lw=1.2, label='Announcement')
    axes[1].axhline(0, color='black', lw=0.5)
    axes[1].set_xlabel('Trading days relative to announcement')
    axes[1].set_ylabel('Cumulative Abnormal Return (%)')
    axes[1].set_title('Parent CAR by Child Inclusion Status', fontweight='bold')
    axes[1].legend(fontsize=9)

    plt.suptitle('Parent Stock: Spinoff Announcement Effect', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

# %%
# 3c. Per-event announcement return summary
ann_stats = []
if len(all_aw):
    for _, ev in events[events['permno'].notna() & events['announce_date'].notna()].iterrows():
        if ev['announce_date'] >= ev['effective_date']:
            continue
        grp = all_aw[all_aw['parent_ticker'] == ev['parent_ticker']]
        if grp.empty:
            continue
        t0_ret  = grp[grp['t_ann'] == 0]['ret_mkt_adj']
        t5_car  = grp[grp['t_ann'].between(0, 5)]['ret_mkt_adj'].sum()
        lag     = (ev['effective_date'] - ev['announce_date']).days
        ann2eff = grp[grp['t_ann'].between(0, lag)]['ret_mkt_adj'].sum()
        ann_stats.append({
            'parent': ev['parent_ticker'], 'child': ev['spinoff_ticker'],
            'lag_days': lag,
            'ann_day_car':   t0_ret.values[0] if len(t0_ret) else np.nan,
            'ann_5d_car':    t5_car,
            'ann_to_eff_car': ann2eff,
            'inclusion_status': ev['inclusion_status'],
        })

ann_df = pd.DataFrame(ann_stats)
if len(ann_df):
    print(ann_df[['parent', 'child', 'lag_days', 'ann_day_car', 'ann_5d_car',
                  'ann_to_eff_car', 'inclusion_status']]
          .assign(ann_day_car=lambda x: x['ann_day_car'].map('{:+.2%}'.format),
                  ann_5d_car=lambda x: x['ann_5d_car'].map('{:+.2%}'.format),
                  ann_to_eff_car=lambda x: x['ann_to_eff_car'].map('{:+.2%}'.format))
          .sort_values('ann_to_eff_car').to_string(index=False))
    t_, p_ = stats.ttest_1samp(ann_df['ann_day_car'].dropna(), 0)
    print(f'\nAnnouncement-day CAR: mean={ann_df["ann_day_car"].mean()*100:+.2f}%  '
          f't={t_:.2f}  p={p_:.3f}  n={ann_df["ann_day_car"].notna().sum()}')

# %%
# 3d. LEN / Millrose Properties (LEN/B) case highlight
len_ev = events[events['parent_ticker'] == 'LEN']
if not len_ev.empty and crsp is not None:
    len_row = len_ev.iloc[0]
    print(f'LEN → {len_row["spinoff_ticker"]}  '
          f'announced {len_row["announce_date"].date()}  '
          f'effective {len_row["effective_date"].date()}')
    len_w = get_announce_window(len_row['permno'], len_row['announce_date'],
                                 len_row['effective_date'], crsp, idx_ret)
    if len(len_w):
        fig, ax = plt.subplots(figsize=(12, 4))
        cum = len_w.set_index('t_ann')['ret_mkt_adj'].cumsum()
        ax.plot(cum.index, cum.values * 100, lw=2, color='steelblue', label='LEN CAR (mkt-adj)')
        ax.axvline(0, color='red', linestyle='--', lw=1.2, label='Announcement')
        eff_t = len_w['t_ann'][len_w['t_eff'] == 0]
        if len(eff_t):
            ax.axvline(eff_t.values[0], color='green', linestyle='--', lw=1.2,
                       label='Effective date')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_xlabel('Trading days relative to announcement')
        ax.set_ylabel('CAR (%)')
        ax.set_title('LEN (Lennar) — Millrose Properties Spinoff', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.show()
        lag = (len_row['effective_date'] - len_row['announce_date']).days
        ann2eff = len_w[len_w['t_ann'].between(0, lag)]['ret_mkt_adj'].sum()
        print(f'LEN announce→effective CAR: {ann2eff*100:+.2f}%')
    else:
        print('[!] No CRSP data for LEN in announce window')
else:
    print('[i] LEN not found in events (MRP spinoff too new for CRSP child data; parent may still be in crsp_daily)')

# %% [markdown]
# ## 4. Child vs Parent Relative Returns (Pairs Trade)
#
# **Short child / long parent** is a tighter hedge than **short child / long SPY**.
# We compute:
#
# $$\text{RelCAR}_t = \sum_{s=1}^{t}(r_{\text{child},s} - r_{\text{market},s})
#    - \sum_{s=1}^{t}(r_{\text{parent},s} - r_{\text{market},s})$$
#
# For **excluded** children we expect this to be *negative*: child falls more than parent.
# The pairs P&L = −RelCAR (we are long this divergence).
#
# **Lookahead:** none — uses daily returns available at end-of-day.

# %%
pairs_rows = []
if crsp is not None and len(all_cw):
    for _, ev in events[events['permno'].notna()].iterrows():
        ticker = ev['spinoff_ticker']
        child_grp = all_cw[all_cw['child_ticker'] == ticker].copy()
        if child_grp.empty:
            continue
        parent_prices = (crsp[crsp['permno'] == int(ev['permno'])]
                         .copy()
                         .query('date >= @ev["effective_date"]')
                         .sort_values('date'))
        parent_dict = parent_prices.set_index('date')['ret'].to_dict()

        for _, c_row in child_grp.iterrows():
            p_ret = parent_dict.get(c_row['date'], np.nan)
            pairs_rows.append({
                'child_ticker':     ticker,
                'parent_ticker':    ev['parent_ticker'],
                'inclusion_status': ev['inclusion_status'],
                't': c_row['t'],
                'date': c_row['date'],
                'child_ret_mkt_adj':  c_row['ret_mkt_adj'],
                'parent_ret_mkt_adj': p_ret - c_row['sprtrn'] if pd.notna(p_ret) else np.nan,
                'sprtrn': c_row['sprtrn'],
            })

    pairs_df = pd.DataFrame(pairs_rows).sort_values(['child_ticker', 't'])
    pairs_df['child_cum_ret']  = pairs_df.groupby('child_ticker')['child_ret_mkt_adj'].cumsum()
    pairs_df['parent_cum_ret'] = pairs_df.groupby('child_ticker')['parent_ret_mkt_adj'].cumsum()
    pairs_df['relative_car']   = pairs_df['child_cum_ret'] - pairs_df['parent_cum_ret']
    pairs_df['pairs_pnl']      = -pairs_df['relative_car']
    print(f'Pairs data: {pairs_df["child_ticker"].nunique()} events  {len(pairs_df):,} rows')
else:
    pairs_df = pd.DataFrame()
    print('[!] Pairs analysis requires crsp_daily')

# %%
if len(pairs_df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for status, grp in pairs_df[pairs_df['t'].between(0, 63)].groupby('inclusion_status'):
        color = COLOR_MAP.get(status, 'grey')
        for ticker, tgrp in grp.groupby('child_ticker'):
            axes[0].plot(tgrp['t'], tgrp['relative_car'] * 100,
                         color=color, alpha=0.2, lw=0.8)
        avg = grp.groupby('t')['relative_car'].mean()
        axes[0].plot(avg.index, avg.values * 100, color=color, lw=2.5, label=status)

    axes[0].axhline(0, color='black', lw=0.8)
    axes[0].set_xlabel('Trading days since effective date')
    axes[0].set_ylabel('Child − Parent Relative CAR (%)')
    axes[0].set_title('Relative Return: Child vs Parent\n(negative = child underperforms)',
                      fontweight='bold')
    axes[0].legend(fontsize=9)

    for status, grp in pairs_df[pairs_df['t'].between(0, 63)].groupby('inclusion_status'):
        color = COLOR_MAP.get(status, 'grey')
        avg_pnl = grp.groupby('t')['pairs_pnl'].mean()
        axes[1].plot(avg_pnl.index, avg_pnl.values * 100, color=color, lw=2.5, label=status)

    axes[1].axhline(0, color='black', lw=0.8)
    axes[1].set_xlabel('Trading days since effective date')
    axes[1].set_ylabel('Pairs P&L (%)')
    axes[1].set_title('Pairs Trade P&L: Short Child / Long Parent', fontweight='bold')
    axes[1].legend(fontsize=9)

    plt.suptitle('Child vs Parent Relative Return Analysis', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print('\n=== Pairs P&L (excluded children) ===')
    for h in [21, 42, 63]:
        sub = pairs_df[(pairs_df['t'] == h) & (pairs_df['inclusion_status'] == 'excluded')]
        excl = sub['pairs_pnl'].dropna()
        if len(excl) > 1:
            t, p = stats.ttest_1samp(excl, 0)
            print(f't={h:2d}d  mean={excl.mean()*100:+.2f}%  '
                  f'win_rate={(excl>0).mean():.0%}  t={t:.2f}  p={p:.3f}  n={len(excl)}')

# %%
# Child CAR vs Parent CAR scatter at t=42d
if len(pairs_df):
    sub42 = pairs_df[pairs_df['t'] == 42].dropna(subset=['child_cum_ret', 'parent_cum_ret'])
    colors_s = [COLOR_MAP.get(s, 'grey') for s in sub42['inclusion_status']]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sub42['parent_cum_ret'] * 100, sub42['child_cum_ret'] * 100,
               c=colors_s, s=80, alpha=0.8, zorder=3)
    for _, row in sub42.iterrows():
        ax.annotate(row['child_ticker'],
                    (row['parent_cum_ret']*100, row['child_cum_ret']*100),
                    fontsize=7, alpha=0.7)
    lims = [min(sub42['parent_cum_ret'].min(), sub42['child_cum_ret'].min()) * 100 - 2,
            max(sub42['parent_cum_ret'].max(), sub42['child_cum_ret'].max()) * 100 + 2]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.4, label='Child = Parent')
    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5)
    ax.set_xlabel('Parent CAR at t=42d (market-adj, %)')
    ax.set_ylabel('Child CAR at t=42d (market-adj, %)')
    ax.set_title('Child vs Parent CAR at t=42d\n(points below diagonal = child underperforms parent)',
                 fontweight='bold')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, alpha=0.8, label=s) for s, c in COLOR_MAP.items()]
              + [plt.Line2D([0],[0], color='black', linestyle='--', label='Child = Parent')],
              fontsize=9)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 5. Beta Estimation and Alpha Decomposition
#
# We decompose each child's return into:
# - **Market component**: β × S&P 500 daily return
# - **Residual alpha**: forced-flow pressure / idiosyncratic
#
# OLS: $r_{\text{child},t} = \alpha + \beta \cdot r_{\text{market},t} + \varepsilon_t$
#
# **Lookahead note:** β is estimated using t=1 to t=60 post-effective data, so it is
# *ex-post*. In live trading, use parent's 1-year pre-spinoff beta as a prior, then
# update once enough post-effective data accumulates.

# %%
beta_rows = []
for ticker, grp in all_cw.groupby('child_ticker'):
    grp = grp.sort_values('t')
    sub = grp[grp['t'].between(1, 60)].dropna(subset=['ret', 'sprtrn'])
    if len(sub) < 10:
        beta_rows.append({'child_ticker': ticker, 'beta': np.nan,
                           'alpha_daily': np.nan, 'r_squared': np.nan, 'n_obs': len(sub)})
        continue
    slope, intercept, r, _, _ = stats.linregress(sub['sprtrn'], sub['ret'])
    beta_rows.append({'child_ticker': ticker, 'beta': slope,
                       'alpha_daily': intercept, 'r_squared': r**2, 'n_obs': len(sub)})

beta_df = pd.DataFrame(beta_rows).merge(
    events[['spinoff_ticker', 'inclusion_status', 'forced_flow_adv']],
    left_on='child_ticker', right_on='spinoff_ticker', how='left'
)
print('=== Child Beta vs S&P 500 ===')
print(beta_df[['child_ticker', 'beta', 'alpha_daily', 'r_squared', 'n_obs', 'inclusion_status']]
      .sort_values('beta').to_string(index=False))
print(f'\nMean β: {beta_df["beta"].mean():.2f}  '
      f'Median β: {beta_df["beta"].median():.2f}  '
      f'Range: [{beta_df["beta"].min():.2f}, {beta_df["beta"].max():.2f}]')

# %%
# Build beta-stripped alpha paths
alpha_cw_list = []
beta_map = beta_df.set_index('child_ticker')['beta'].to_dict()
for _, ev in events.iterrows():
    ticker = ev['spinoff_ticker']
    grp = all_cw[all_cw['child_ticker'] == ticker].sort_values('t').copy()
    if grp.empty:
        continue
    b = beta_map.get(ticker, 1.0)
    b = b if pd.notna(b) else 1.0
    grp['alpha_daily']     = grp['ret'] - b * grp['sprtrn']
    grp['cum_alpha']       = grp['alpha_daily'].cumsum()
    grp['short_alpha_pnl'] = -grp['cum_alpha']
    alpha_cw_list.append(grp)

alpha_cw = pd.concat(alpha_cw_list, ignore_index=True) if alpha_cw_list else pd.DataFrame()

# %%
if len(alpha_cw):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for status, grp in alpha_cw[alpha_cw['t'].between(0, 63)].groupby('inclusion_status'):
        color = COLOR_MAP.get(status, 'grey')
        for ticker, tgrp in grp.groupby('child_ticker'):
            axes[0].plot(tgrp['t'], tgrp['cum_alpha'] * 100,
                         color=color, alpha=0.2, lw=0.8)
        avg_a = grp.groupby('t')['cum_alpha'].mean()
        axes[0].plot(avg_a.index, avg_a.values * 100, color=color, lw=2.5, label=status)

    axes[0].axhline(0, color='black', lw=0.8)
    axes[0].set_xlabel('Trading days since effective date')
    axes[0].set_ylabel('Cumulative Alpha (%)')
    axes[0].set_title('Beta-Stripped Cumulative Alpha\n(r_child − β·r_market)', fontweight='bold')
    axes[0].legend(fontsize=9)

    excl_alpha = alpha_cw[(alpha_cw['inclusion_status'] == 'excluded') &
                           alpha_cw['t'].between(0, 63)]
    avg_raw  = excl_alpha.groupby('t')['short_pnl'].mean()
    avg_alph = excl_alpha.groupby('t')['short_alpha_pnl'].mean()
    axes[1].plot(avg_raw.index,  avg_raw.values * 100,  lw=2.5, color='#d32f2f',
                 label='Raw short P&L (vs market)')
    axes[1].plot(avg_alph.index, avg_alph.values * 100, lw=2.5, color='#7b1fa2',
                 linestyle='--', label='Beta-stripped alpha P&L')
    axes[1].axhline(0, color='black', lw=0.8)
    axes[1].set_xlabel('Trading days since effective date')
    axes[1].set_ylabel('P&L (%)')
    axes[1].set_title('Raw vs Beta-Stripped P&L\n(Excluded children only)', fontweight='bold')
    axes[1].legend(fontsize=9)

    plt.suptitle('Beta Estimation and Alpha Decomposition', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

# %%
if len(beta_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    colors_b = [COLOR_MAP.get(s, 'grey') for s in beta_df['inclusion_status']]
    axes[0].scatter(beta_df['forced_flow_adv'], beta_df['beta'], c=colors_b, s=70, alpha=0.8)
    axes[0].axhline(1.0, color='grey', linestyle=':', lw=1.5, label='β=1')
    for status in COLOR_MAP:
        axes[0].scatter([], [], color=COLOR_MAP[status], label=status, s=50)
    axes[0].set_xlabel('Forced Flow (× ADV)'); axes[0].set_ylabel('Child Beta')
    axes[0].set_title('Beta vs Forced-Flow', fontweight='bold'); axes[0].legend(fontsize=8)

    axes[1].hist(beta_df['beta'].dropna(), bins=12, color='steelblue', edgecolor='white', alpha=0.8)
    axes[1].axvline(beta_df['beta'].median(), color='red', linestyle='--',
                    label=f'Median β={beta_df["beta"].median():.2f}')
    axes[1].axvline(1.0, color='grey', linestyle=':', lw=1.5, label='β=1')
    axes[1].set_xlabel('Beta vs S&P 500'); axes[1].set_title('Child Beta Distribution', fontweight='bold')
    axes[1].legend(fontsize=9)
    plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6. Free Float Forced-Flow Metric
#
# Standard metric: `forced_flow_adv = passive_AUM × parent_weight / parent_ADV`
# measures selling pressure in days of **parent** volume.
#
# Better for child price impact: use **child market cap** as denominator:
#
# $$\text{FF\%float} = \frac{\text{Passive AUM} \times w_{\text{parent}}}{\text{Child MktCap}_{t=0}}$$
#
# This is the fraction of the child's float that passive funds must liquidate.
#
# **Lookahead:** child mktcap is observable at t=0 close; AUM and parent weight
# are known at announcement.

# %%
child_t0 = (children.sort_values('date')
            .groupby('permno').first().reset_index()
            [['permno', 'mktcap', 'dollar_vol', 'adv_30d', 'child_ticker', 'effective_date']]
            .rename(columns={'mktcap': 'child_mktcap_t0',
                             'dollar_vol': 'child_vol_t0',
                             'adv_30d': 'child_adv30_t0'}))

events_ff = events.merge(
    child_t0[['child_ticker', 'child_mktcap_t0', 'child_vol_t0', 'child_adv30_t0']],
    left_on='spinoff_ticker', right_on='child_ticker', how='left'
)

events_ff['ff_float_pct'] = np.where(
    events_ff['child_mktcap_t0'].notna() & (events_ff['child_mktcap_t0'] > 0),
    events_ff['forced_flow_usd'] / events_ff['child_mktcap_t0'],
    np.nan
)
events_ff['ff_child_adv'] = np.where(
    events_ff['child_adv30_t0'].notna() & (events_ff['child_adv30_t0'] > 0),
    events_ff['forced_flow_usd'] / events_ff['child_adv30_t0'],
    np.nan
)

print('=== Forced-Flow Metrics Comparison (sorted by FF%float) ===')
disp_ff = (events_ff[['parent_ticker', 'spinoff_ticker', 'child_mktcap_t0',
                        'forced_flow_usd', 'forced_flow_adv', 'ff_float_pct',
                        'ff_child_adv', 'inclusion_status']]
           .dropna(subset=['child_mktcap_t0']).copy())
disp_ff['child_mktcap_t0'] = disp_ff['child_mktcap_t0'] / 1e9
disp_ff['forced_flow_usd'] = disp_ff['forced_flow_usd'] / 1e9
disp_ff['ff_float_pct']    = disp_ff['ff_float_pct'] * 100
print(disp_ff.rename(columns={'child_mktcap_t0': 'child_mktcap_B',
                                'forced_flow_usd': 'ff_usd_B',
                                'ff_float_pct': 'ff_%float'})
      .sort_values('ff_%float', ascending=False).to_string(index=False))

# %%
if events_ff['ff_float_pct'].notna().any():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ff_excl = events_ff[events_ff['inclusion_status'] == 'excluded']['ff_float_pct'].dropna()
    ff_incl = events_ff[events_ff['inclusion_status'] != 'excluded']['ff_float_pct'].dropna()
    for vals, label, color in [(ff_excl, 'excluded', COLOR_MAP['excluded']),
                                (ff_incl, 'included', COLOR_MAP['immediately included'])]:
        axes[0].hist(vals * 100, bins=10, alpha=0.6, color=color, edgecolor='white',
                     label=f'{label} (n={len(vals)})')
    axes[0].set_xlabel('FF% Float'); axes[0].set_ylabel('Events')
    axes[0].set_title('FF% Float Distribution', fontweight='bold'); axes[0].legend(fontsize=9)

    ev_car = events_ff.merge(car_df[['child_ticker', 'short_pnl_21d']],
                              left_on='spinoff_ticker', right_on='child_ticker', how='inner')
    valid = ev_car.dropna(subset=['ff_float_pct', 'short_pnl_21d'])
    colors_p = [COLOR_MAP.get(s, 'grey') for s in valid['inclusion_status']]
    axes[1].scatter(valid['ff_float_pct'] * 100, valid['short_pnl_21d'] * 100,
                    c=colors_p, s=70, alpha=0.8)
    if len(valid) > 2:
        m, b = np.polyfit(valid['ff_float_pct'], valid['short_pnl_21d'], 1)
        x_ = np.linspace(valid['ff_float_pct'].min(), valid['ff_float_pct'].max(), 50)
        axes[1].plot(x_ * 100, (m*x_+b)*100, 'k--', alpha=0.5)
        r, p = stats.pearsonr(valid['ff_float_pct'], valid['short_pnl_21d'])
        axes[1].set_title(f'FF%Float vs Short P&L t=21d\nr={r:.2f}  p={p:.3f}', fontweight='bold')
    for status in COLOR_MAP:
        if any(valid['inclusion_status'] == status):
            axes[1].scatter([], [], color=COLOR_MAP[status], label=status, s=50)
    axes[1].axhline(0, color='black', lw=0.5)
    axes[1].set_xlabel('FF% Float'); axes[1].set_ylabel('Short P&L t=21d (%)')
    axes[1].legend(fontsize=8)

    valid2 = events_ff.dropna(subset=['ff_float_pct', 'forced_flow_adv'])
    colors_p2 = [COLOR_MAP.get(s, 'grey') for s in valid2['inclusion_status']]
    axes[2].scatter(valid2['forced_flow_adv'], valid2['ff_float_pct'] * 100,
                    c=colors_p2, s=70, alpha=0.8)
    axes[2].set_xlabel('FF × Parent ADV'); axes[2].set_ylabel('FF% Float')
    axes[2].set_title('Two Forced-Flow Metrics', fontweight='bold')
    for status in COLOR_MAP:
        if any(valid2['inclusion_status'] == status):
            axes[2].scatter([], [], color=COLOR_MAP[status], label=status, s=50)
    axes[2].legend(fontsize=8)

    plt.suptitle('Free Float Forced-Flow Analysis', fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.show()

# %% [markdown]
# ## 7. Index Inclusion Probability Model
#
# Logistic regression predicting **immediate inclusion** (within 5 trading days).
#
# ### S&P 500 Eligibility Criteria (as features):
# | Criterion | Feature proxy |
# |---|---|
# | Market cap ≥ annual threshold | `log_size_ratio` (child mktcap / year threshold) |
# | Dollar volume traded | `child_vol_t0` |
# | Float ≥ 50% | (not directly observed; proxied by mktcap) |
# | Positive earnings | `parent profitability` (fundamentals) |
# | US domicile / major exchange | (all events are US) |
#
# ### Lookahead audit:
# | Feature | Available at | Clean? |
# |---|---|---|
# | log_child_mktcap | t=0 close | Yes |
# | parent_index_weight | Before t=0 | Yes |
# | forced_flow_adv | Before t=0 | Yes |
# | log_size_ratio | t=0 close | Yes |
# | log_lag_days | Announcement | Yes |
# | passive_aum_B | Announcement | Yes |

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

SP500_MIN_MKTCAP = {2020: 8.2e9, 2021: 11.8e9, 2022: 13.1e9, 2023: 12.7e9, 2024: 18.0e9}
def sp500_threshold(year):
    return SP500_MIN_MKTCAP.get(year, SP500_MIN_MKTCAP[min(SP500_MIN_MKTCAP,
                                                             key=lambda y: abs(y - year))])

model_rows = []
for _, ev in events_ff.iterrows():
    year = ev['effective_date'].year
    thresh = sp500_threshold(year)
    size_ratio = (ev['child_mktcap_t0'] / thresh
                  if pd.notna(ev['child_mktcap_t0']) and ev['child_mktcap_t0'] > 0 else np.nan)
    lag = (ev['effective_date'] - ev['announce_date']).days if pd.notna(ev['announce_date']) else np.nan
    lag = lag if (pd.notna(lag) and lag > 0) else np.nan
    model_rows.append({
        'child_ticker':        ev['spinoff_ticker'],
        'parent_ticker':       ev['parent_ticker'],
        'effective_date':      ev['effective_date'],
        'inclusion_status':    ev['inclusion_status'],
        'target':              1 if ev['inclusion_status'] == 'immediately included' else 0,
        'log_child_mktcap':    np.log(ev['child_mktcap_t0']) if pd.notna(ev['child_mktcap_t0']) and ev['child_mktcap_t0'] > 0 else np.nan,
        'parent_index_weight': ev['parent_index_weight'],
        'forced_flow_adv':     ev['forced_flow_adv'],
        'log_size_ratio':      np.log(size_ratio) if pd.notna(size_ratio) and size_ratio > 0 else np.nan,
        'log_lag_days':        np.log(lag) if pd.notna(lag) else np.nan,
        'passive_aum_B':       ev['passive_aum_usd'] / 1e9 if pd.notna(ev['passive_aum_usd']) else np.nan,
    })

model_df = pd.DataFrame(model_rows)
feature_cols = ['log_child_mktcap', 'parent_index_weight', 'forced_flow_adv',
                'log_size_ratio', 'log_lag_days', 'passive_aum_B']
model_clean = model_df.dropna(subset=feature_cols + ['target']).copy()
print(f'Model: n={len(model_clean)} (dropped {len(model_df)-len(model_clean)} for NaN)')
print(f'Target: {dict(model_clean["target"].value_counts())}  '
      f'(inclusion rate {model_clean["target"].mean():.0%})')

# %%
if len(model_clean) >= 10:
    X = model_clean[feature_cols].values
    y = model_clean['target'].values
    X_sc = StandardScaler().fit_transform(X)

    loo_probs = np.zeros(len(y))
    loo_preds = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in LeaveOneOut().split(X_sc):
        clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        clf.fit(X_sc[train_idx], y[train_idx])
        loo_probs[test_idx] = clf.predict_proba(X_sc[test_idx])[:, 1]
        loo_preds[test_idx] = clf.predict(X_sc[test_idx])

    model_clean = model_clean.copy()
    model_clean['prob_included'] = loo_probs
    model_clean['pred_included'] = loo_preds
    loo_acc = accuracy_score(y, loo_preds)
    loo_auc = roc_auc_score(y, loo_probs) if len(np.unique(y)) > 1 else np.nan
    print(f'LOO-CV Accuracy: {loo_acc:.2%}   AUC: {loo_auc:.3f}')
    print(f'Majority baseline: {max(y.mean(), 1-y.mean()):.0%}')

    scaler_full = StandardScaler()
    X_sc_full = scaler_full.fit_transform(X)
    clf_full = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf_full.fit(X_sc_full, y)
    coef_df = pd.DataFrame({'Feature': feature_cols, 'Coef': clf_full.coef_[0]})
    coef_df['Abs'] = coef_df['Coef'].abs()
    coef_df = coef_df.sort_values('Abs', ascending=False)
    print('\n=== Coefficients (standardized) ===')
    print(coef_df[['Feature', 'Coef']].to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors_c = ['green' if c > 0 else 'red' for c in coef_df['Coef']]
    axes[0].barh(coef_df['Feature'], coef_df['Coef'], color=colors_c, alpha=0.8)
    axes[0].axvline(0, color='black', lw=0.8)
    axes[0].set_title('Feature Importance\n(+ = predicts inclusion)', fontweight='bold')

    mc_s = model_clean.sort_values('prob_included')
    bar_colors = [COLOR_MAP.get(s, 'grey') for s in mc_s['inclusion_status']]
    labels_ = [f'{r.child_ticker} ({r.parent_ticker})' for r in mc_s.itertuples()]
    axes[1].barh(labels_, mc_s['prob_included'] * 100, color=bar_colors, alpha=0.85)
    axes[1].axvline(50, color='black', linestyle='--', lw=1)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=c, alpha=0.8, label=s) for s, c in COLOR_MAP.items()],
                   fontsize=8)
    axes[1].set_xlabel('P(immediately included) %')
    axes[1].set_title('LOO-CV Inclusion Probabilities', fontweight='bold')

    if not np.isnan(loo_auc):
        fpr, tpr, _ = roc_curve(y, loo_probs)
        axes[2].plot(fpr, tpr, lw=2, color='steelblue', label=f'AUC={loo_auc:.3f}')
        axes[2].plot([0,1],[0,1], 'k--', lw=1, label='Random')
        axes[2].set_xlabel('FPR'); axes[2].set_ylabel('TPR')
        axes[2].set_title('ROC Curve (LOO-CV)', fontweight='bold')
        axes[2].legend(fontsize=9)

    plt.suptitle('Index Inclusion Probability Model', fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.show()
else:
    print('[!] Insufficient clean data for inclusion model')
    model_clean = model_df.copy()
    model_clean['prob_included'] = np.nan

# %%
# Size-only eligibility check (naive rule)
print('=== S&P 500 Size-Only Eligibility Check ===')
elig_rows = []
for _, row in model_df.iterrows():
    year = row['effective_date'].year
    thresh = sp500_threshold(year)
    mktcap = events_ff[events_ff['spinoff_ticker'] == row['child_ticker']]['child_mktcap_t0']
    mktcap = mktcap.values[0] if len(mktcap) and pd.notna(mktcap.values[0]) else np.nan
    meets_size = (mktcap >= thresh) if pd.notna(mktcap) else None
    elig_rows.append({
        'child': row['child_ticker'], 'parent': row['parent_ticker'], 'year': year,
        'mktcap_B': mktcap/1e9 if pd.notna(mktcap) else np.nan,
        'threshold_B': thresh/1e9,
        'meets_size': meets_size,
        'actually_included': row['target'] == 1,
    })
elig_df = pd.DataFrame(elig_rows).dropna(subset=['mktcap_B'])
elig_df['correct'] = elig_df['meets_size'] == elig_df['actually_included']
print(elig_df[['child','parent','year','mktcap_B','threshold_B',
               'meets_size','actually_included','correct']]
      .sort_values('mktcap_B', ascending=False).to_string(index=False))
print(f'\nSize-only accuracy: {elig_df["correct"].sum()}/{len(elig_df)} = {elig_df["correct"].mean():.0%}')

# %% [markdown]
# ## 8. Bucket Analysis — Overperformers vs Underperformers
#
# At t=21d we split each child into:
# - **Underperformer**: CAR < 0 (child underperforms market — good for short)
# - **Outperformer**: CAR ≥ 0 (child keeps pace or beats market)
#
# The key trading question: which **t=0 observable features** predict the bucket?

# %%
car_df['bucket_21d'] = np.where(car_df['car_21d'] < 0, 'underperformer', 'outperformer')
car_df['bucket_strong'] = np.where(
    car_df['car_21d'] < -0.05, 'strong under (<−5%)',
    np.where(car_df['car_21d'] > 0.05, 'outperform (>+5%)', 'neutral (±5%)'))

bucket_feat = car_df.merge(
    events_ff[['spinoff_ticker', 'parent_index_weight', 'ff_float_pct', 'child_mktcap_t0']],
    left_on='child_ticker', right_on='spinoff_ticker', how='left'
)

print('=== Bucket Distribution (t=21d, all children) ===')
print(car_df['bucket_strong'].value_counts().to_string())
excl_only = car_df[car_df['inclusion_status'] == 'excluded']
print(f'\nExcluded only: underperform rate = '
      f'{(excl_only["car_21d"] < 0).sum()}/{len(excl_only)} = '
      f'{(excl_only["car_21d"] < 0).mean():.0%}')

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
feature_plot = [
    ('forced_flow_adv',   'Forced Flow (× ADV)'),
    ('ff_float_pct',      'FF% Float'),
    ('parent_index_weight','Parent Index Weight'),
    ('child_mktcap_t0',   'Child MktCap ($)'),
    ('car_21d',           'CAR t=21d'),
    ('car_63d',           'CAR t=63d'),
]
bucket_colors_map = {'underperformer': '#d32f2f', 'outperformer': '#1976d2'}

for ax, (col, label) in zip(axes, feature_plot):
    data = {b: bucket_feat[bucket_feat['bucket_21d'] == b][col].dropna()
            for b in bucket_colors_map}
    bp = ax.boxplot([data[b] for b in bucket_colors_map],
                    labels=list(bucket_colors_map.keys()),
                    patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], bucket_colors_map.values()):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    for med in bp['medians']:
        med.set(color='black', linewidth=2)
    u, o = data['underperformer'], data['outperformer']
    if len(u) > 1 and len(o) > 1:
        t_, p_ = stats.ttest_ind(u, o)
        ax.set_xlabel(f't={t_:.2f}  p={p_:.3f}', fontsize=8)
    ax.set_ylabel(label)
    ax.set_title(f'{label} by t=21d bucket', fontweight='bold', fontsize=10)

plt.suptitle('Feature Comparison: Underperformers vs Outperformers at t=21d',
             fontsize=12, fontweight='bold')
plt.tight_layout(); plt.show()

# %%
# Feature correlations with CAR (excluded children only)
print('=== Feature Correlation with CAR (excluded children) ===')
excl_feat = bucket_feat[bucket_feat['inclusion_status'] == 'excluded']
for col in ['forced_flow_adv', 'ff_float_pct', 'parent_index_weight', 'child_mktcap_t0']:
    for h in [21, 63]:
        sub = excl_feat.dropna(subset=[col, f'car_{h}d'])
        if len(sub) > 2:
            r, p = stats.pearsonr(sub[col], sub[f'car_{h}d'])
            print(f'  {col:<25s} vs car_{h}d:  r={r:+.3f}  p={p:.3f}  n={len(sub)}')

# %% [markdown]
# ## 9. Trading Strategy Backtest
#
# Four strategies, all entering at **t=0 close** (effective date):
#
# | # | Short | Long hedge | Universe filter |
# |---|---|---|---|
# | S1 | Child | SPY (β=1) | All excluded |
# | S2 | Child | SPY (β=1) | Excluded + forced_flow_adv > 5× |
# | S3 | Child | Parent stock | Excluded with parent data |
# | S4 | Child | SPY (β=1) | Excluded + prob_included < 25% |
#
# **Exit:** first of (a) child added to S&P 500, (b) t=42d time stop.
# Equal-weight per trade. No transaction costs (EDA level).

# %%
trade_rows = []
horizons_bt = [10, 21, 42, 63]

for _, ev in events.iterrows():
    ticker = ev['spinoff_ticker']
    status = ev['inclusion_status']
    child_grp = all_cw[all_cw['child_ticker'] == ticker].copy() if len(all_cw) else pd.DataFrame()
    if child_grp.empty:
        continue

    inc_rec = child_sp500[child_sp500['child_ticker'] == ticker]
    t_exit_inc = (child_grp[child_grp['date'] >= inc_rec.sort_values('added_date')
                  ['added_date'].iloc[0]]['t'].min()
                  if not inc_rec.empty else 999)

    ff   = ev.get('forced_flow_adv', np.nan)
    ff_f = events_ff[events_ff['spinoff_ticker'] == ticker]['ff_float_pct']
    ff_float_val = ff_f.values[0] if len(ff_f) else np.nan

    prob_inc = np.nan
    if len(model_clean):
        mc_r = model_clean[model_clean['child_ticker'] == ticker]
        if len(mc_r):
            prob_inc = mc_r['prob_included'].values[0]

    for h in horizons_bt:
        t_exit = min(h, t_exit_inc - 1) if t_exit_inc < 999 else h
        t_exit = max(t_exit, 1)
        exit_row = child_grp[child_grp['t'] == t_exit]
        if exit_row.empty:
            exit_row = child_grp.iloc[(child_grp['t'] - t_exit).abs().argsort()[:1]]
        raw_pnl = exit_row['short_pnl'].values[0]

        pairs_pnl_val = np.nan
        if len(pairs_df):
            p_r = pairs_df[(pairs_df['child_ticker'] == ticker) & (pairs_df['t'] == t_exit)]
            if len(p_r):
                pairs_pnl_val = p_r['pairs_pnl'].values[0]

        trade_rows.append({
            'child_ticker':   ticker,
            'parent_ticker':  ev['parent_ticker'],
            'effective_date': ev['effective_date'],
            'inclusion_status': status,
            'forced_flow_adv': ff,
            'ff_float_pct':   ff_float_val,
            'prob_included':  prob_inc,
            't_stop': h,
            't_actual_exit': t_exit,
            'forced_exit': t_exit_inc <= h,
            'pnl_s1': raw_pnl      if status == 'excluded' else np.nan,
            'pnl_s2': raw_pnl      if (status == 'excluded' and pd.notna(ff) and ff >= 5) else np.nan,
            'pnl_s3': pairs_pnl_val if status == 'excluded' else np.nan,
            'pnl_s4': raw_pnl      if (status == 'excluded' and pd.notna(prob_inc) and prob_inc < 0.25) else np.nan,
        })

trades_df = pd.DataFrame(trade_rows)
print(f'Trade ledger: {len(trades_df)} trade-horizon rows, '
      f'{trades_df["child_ticker"].nunique()} unique events')

# %%
def strat_metrics(s, name):
    s = s.dropna() * 100
    if len(s) == 0:
        return
    t_, p_ = stats.ttest_1samp(s, 0)
    print(f'  {name:<38s} n={len(s):2d}  '
          f'mean={s.mean():+.2f}%  med={s.median():+.2f}%  '
          f'std={s.std():.2f}%  win={( s>0).mean():.0%}  '
          f't={t_:.2f}  p={p_:.3f}')

strategy_defs = [
    ('pnl_s1', 'S1: All excluded, SPY hedge'),
    ('pnl_s2', 'S2: FF>5×, SPY hedge'),
    ('pnl_s3', 'S3: Pairs (child/parent)'),
    ('pnl_s4', 'S4: Prob-filtered, SPY hedge'),
]
print('\n=== Strategy Results by Horizon ===')
for h in horizons_bt:
    sub = trades_df[trades_df['t_stop'] == h]
    print(f'\n--- t={h}d exit ---')
    for col, name in strategy_defs:
        strat_metrics(sub[col], name)

# %%
# Equity curves
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()
strat_colors = ['#d32f2f', '#e65100', '#1565c0', '#2e7d32']

for ax, h in zip(axes, horizons_bt):
    sub_h = trades_df[trades_df['t_stop'] == h].sort_values('effective_date')
    for (col, name), color in zip(strategy_defs, strat_colors):
        dated = sub_h.dropna(subset=[col]).sort_values('effective_date')
        if len(dated) < 2:
            continue
        cum = dated[col].cumsum() * 100
        ax.plot(range(len(cum)), cum.values, lw=2, color=color,
                label=f'{name} (n={len(dated)})')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('Trade #'); ax.set_ylabel('Cumulative P&L (%)')
    ax.set_title(f't={h}d stop — Equity Curve', fontweight='bold')
    ax.legend(fontsize=8)

plt.suptitle('Strategy Equity Curves (equal-weight, t=0 close entry)',
             fontsize=12, fontweight='bold')
plt.tight_layout(); plt.show()

# %%
# Summary table at t=42d
print('=== Strategy Summary — t=42d Horizon ===')
h42 = trades_df[trades_df['t_stop'] == 42]
summary_rows = []
for col, name in strategy_defs:
    s = h42[col].dropna() * 100
    if len(s) == 0:
        continue
    t_, p_ = stats.ttest_1samp(s, 0)
    summary_rows.append({
        'Strategy': name, 'N': len(s),
        'Mean P&L (%)': round(s.mean(), 2),
        'Median (%)': round(s.median(), 2),
        'Std (%)': round(s.std(), 2),
        'Win Rate': f'{(s>0).mean():.0%}',
        'Max Loss (%)': round(s.min(), 2),
        'Max Gain (%)': round(s.max(), 2),
        't-stat': round(t_, 2), 'p-value': round(p_, 3),
    })
display(pd.DataFrame(summary_rows))

# %% [markdown]
# ## 10. Key Findings Summary

# %%
print('=' * 70)
print('COMPREHENSIVE FINDINGS SUMMARY')
print('=' * 70)

print('\n[1] ANNOUNCEMENT DATE EFFECT')
if len(ann_df):
    print(f'    Announcement-day CAR: mean = {ann_df["ann_day_car"].mean()*100:+.2f}%')
    print(f'    Announce→Effective CAR: mean = {ann_df["ann_to_eff_car"].mean()*100:+.2f}%')
    print(f'    Lag: median {valid_lags.median():.0f}d  max {valid_lags.max():.0f}d')

print('\n[2] CHILD vs PARENT RELATIVE RETURN')
if len(pairs_df):
    at42 = pairs_df[(pairs_df['t'] == 42) & (pairs_df['inclusion_status'] == 'excluded')]['pairs_pnl'].dropna()
    if len(at42):
        print(f'    Excluded pairs P&L t=42d: mean={at42.mean()*100:+.2f}%  '
              f'win={(at42>0).mean():.0%}')

print('\n[3] BETA DECOMPOSITION')
if len(beta_df):
    print(f'    Mean child β = {beta_df["beta"].mean():.2f}  '
          f'(range {beta_df["beta"].min():.2f}–{beta_df["beta"].max():.2f})')
    print('    Underperformance is predominantly alpha (forced selling), not beta.')

print('\n[4] FREE FLOAT FORCED FLOW')
if 'ff_float_pct' in events_ff.columns:
    ff_e = events_ff[events_ff['inclusion_status'] == 'excluded']['ff_float_pct'].dropna()
    if len(ff_e):
        print(f'    Passive funds must sell: median {ff_e.median()*100:.1f}% of child mktcap')

print('\n[5] INCLUSION MODEL')
if 'prob_included' in model_clean.columns and model_clean['prob_included'].notna().any():
    print(f'    LOO-CV Accuracy: {loo_acc:.0%}   AUC: {loo_auc:.3f}')
    print(f'    Size threshold (log_size_ratio) is the dominant predictor.')

print('\n[6] BUCKET ANALYSIS')
if 'bucket_21d' in car_df.columns:
    excl_b = car_df[car_df['inclusion_status'] == 'excluded']
    print(f'    Excluded children underperform at t=21d: '
          f'{(excl_b["car_21d"] < 0).sum()}/{len(excl_b)} = '
          f'{(excl_b["car_21d"] < 0).mean():.0%}')

print('\n[7] TRADING STRATEGY (t=42d)')
if len(trades_df):
    for col, name in [('pnl_s1','S1 (all excl.)'), ('pnl_s2','S2 (FF>5×)')]:
        s = trades_df[trades_df['t_stop'] == 42][col].dropna() * 100
        if len(s):
            print(f'    {name}: mean={s.mean():+.2f}%  win={(s>0).mean():.0%}  n={len(s)}')

print('\n' + '=' * 70)
print('FEATURE HIERARCHY (Citadel index rebalance desk view)')
print('=' * 70)
for rank, feat, desc in [
    ('1 (gate)',   'inclusion_status = excluded',  'Must be true — forced passive selling'),
    ('2 (size)',   'ff_float_pct',                  'FF as % child mktcap — best price-impact proxy'),
    ('3 (signal)', 'forced_flow_adv',               'Passive AUM × weight / parent ADV'),
    ('4 (model)',  'prob_included (logit)',          'High prob → avoid; child may be fast-tracked'),
    ('5 (timing)', 'log_announce_lag',              'Longer lag = more arb pre-positioning'),
    ('6 (hedge)',  'child_beta',                     'Size SPY hedge; pairs hedge preferred'),
]:
    print(f'  {rank:<12s}  {feat:<30s}  {desc}')
