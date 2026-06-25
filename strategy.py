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
# # Spinoff Index Arbitrage — Trading Strategy
#
# ## Thesis
#
# When a company spins off a subsidiary, S&P 500 passive funds face a **mechanical, non-discretionary flow**:
#
# - **Excluded child** (not added to S&P 500): passive funds receive child shares but *must sell* them — forced selling pressure → **SHORT the child**
# - **Included child** (immediately added to S&P 500): passive funds receive shares and keep them → no forced selling, skip child leg
# - **Parent deletion**: if the parent is subsequently removed from the S&P 500, passive funds *must sell* the parent → **SHORT the parent**
#
# ## Position Sizing
#
# Size is driven by the forced-flow signal:
#
# $$\text{Signal} = \frac{\text{Passive AUM} \times w_{\text{parent}}}{\text{Child ADV}}$$
#
# Higher signal → more selling pressure → larger position.
# We also test sizing by raw parent index weight.
#
# ## Regression Proof
#
# We prove the signal using:  Simple OLS · Multiple OLS · Polynomial · LASSO · Ridge

# %%
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from pathlib import Path

from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import r2_score
import statsmodels.api as sm

pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:,.4f}'.format)
plt.rcParams['figure.figsize'] = (13, 5)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
sns.set_palette('husl')

RAW_DIR   = Path('data/raw')
CLEAN_DIR = Path('data/clean')

# %% [markdown]
# ## 1. Data Loading & Feature Engineering

# %%
events = pd.read_csv(
    CLEAN_DIR / 'spinoff_events_merged.csv',
    parse_dates=['announce_date', 'effective_date', 'sp500_start', 'sp500_end']
)

children   = pd.read_parquet(RAW_DIR / 'spinoff_children_crsp.parquet')
child_sp500 = pd.read_parquet(RAW_DIR / 'spinoff_children_sp500.parquet')
idx_ret    = pd.read_parquet(RAW_DIR / 'crsp_index_returns.parquet')
crsp_path  = RAW_DIR / 'crsp_daily.parquet'
crsp       = pd.read_parquet(crsp_path) if crsp_path.exists() else None

print(f"Events:   {len(events)}")
print(f"Children CRSP: {children['permno'].nunique()} children, {len(children):,} rows")
print(f"Parent CRSP:   {'loaded' if crsp is not None else 'NOT FOUND — run repull_data.py'}")

# %%
# --- Classify child inclusion status ---
first_inc = (
    child_sp500.sort_values('added_date')
    .groupby('child_ticker', as_index=False).first()
    [['child_ticker', 'added_date']]
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

# --- Announce lag ---
events['announce_lag_days'] = (events['effective_date'] - events['announce_date']).dt.days

# --- Parent deletion within 1 year ---
def deleted_within_1yr(row, constituents=None, window=365):
    if constituents is None:
        return False
    if pd.isna(row.get('permno')):
        return False
    from pathlib import Path
    return False  # placeholder if sp500_constituents_pit not loaded

const_path = RAW_DIR / 'sp500_constituents_pit.parquet'
if const_path.exists():
    constituents = pd.read_parquet(const_path)
    def check_deleted(row, window_days=365):
        val = row.get('permno')
        if pd.isna(val):
            return None
        memberships = constituents[constituents['permno'] == val]
        if memberships.empty:
            return None
        eff = row['effective_date']
        deleted = memberships[
            (memberships['end_date'] >= eff) &
            (memberships['end_date'] <= eff + pd.Timedelta(days=window_days)) &
            (~memberships['still_active'])
        ]
        return len(deleted) > 0
    events['parent_deleted_1yr'] = events.apply(check_deleted, axis=1)
else:
    events['parent_deleted_1yr'] = False

COLOR_MAP = {
    'excluded':             '#d32f2f',
    'immediately included': '#1976d2',
    'later included':       '#f57c00',
}

print("\nInclusion breakdown:")
print(events['inclusion_status'].value_counts())
print(f"\nParent deleted within 1yr: {events['parent_deleted_1yr'].sum()} events")

# %% [markdown]
# ## 2. Build Return Windows
#
# For each event compute the market-adjusted cumulative return (CAR) of the child
# at horizons t = 5, 10, 21, 42, 63 trading days.
# **Short P&L = −CAR** (we are short the child, long the market).

# %%
HORIZONS = [5, 10, 21, 42, 63]

def build_child_window(permno, eff_date, crsp_df, idx_df, post=90):
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
        t0 = window.iloc[0]['prc'].abs()
    window['norm_prc']    = window[prc_col] / t0 * 100
    window['ret_mkt_adj'] = window['ret'] - window['sprtrn']
    window['car']         = window['ret_mkt_adj'].cumsum()
    window['short_pnl']   = -window['car']
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

# Build CAR summary per event
car_rows = []
for _, ev in events.iterrows():
    ticker = ev['spinoff_ticker']
    grp = all_cw[all_cw['child_ticker'] == ticker]
    if grp.empty:
        continue
    row = {
        'child_ticker':       ticker,
        'parent_ticker':      ev['parent_ticker'],
        'effective_date':     ev['effective_date'],
        'inclusion_status':   ev['inclusion_status'],
        'forced_flow_adv':    ev.get('forced_flow_adv', np.nan),
        'forced_flow_usd':    ev.get('forced_flow_usd', np.nan),
        'parent_index_weight': ev.get('parent_index_weight', np.nan),
        'parent_mktcap_usd':  ev.get('parent_mktcap_usd', np.nan),
        'announce_lag_days':  ev.get('announce_lag_days', np.nan),
        'parent_deleted_1yr': ev.get('parent_deleted_1yr', False),
        'in_sp500_at_spinoff': ev.get('in_sp500_at_spinoff', True),
    }
    for h in HORIZONS:
        sub = grp[grp['t'] == h]
        row[f'car_{h}d']       = sub['car'].values[0]       if len(sub) else np.nan
        row[f'short_pnl_{h}d'] = sub['short_pnl'].values[0] if len(sub) else np.nan
    car_rows.append(row)

car_df = pd.DataFrame(car_rows)
print(f"Return table: {len(car_df)} events with price data")
print(car_df[['child_ticker','inclusion_status','forced_flow_adv',
              'short_pnl_5d','short_pnl_21d','short_pnl_63d']].to_string())

# %% [markdown]
# ## 3. Strategy Logic
#
# | Condition | Action | Sizing |
# |---|---|---|
# | Child **excluded** from S&P 500 | **Short child** + long S&P 500 | ∝ `forced_flow_adv` × `parent_index_weight` |
# | Child **immediately included** | Skip child leg | — |
# | Parent **deleted within 1yr** | **Short parent** + long S&P 500 | ∝ `parent_index_weight` |
#
# We backtest two sizing variants:
# - **Equal-weight**: every eligible trade gets weight = 1
# - **Signal-weight**: weight ∝ `forced_flow_adv` (rescaled to mean=1)

# %%
# --- Strategy filter: only trade excluded children ---
excl = car_df[car_df['inclusion_status'] == 'excluded'].copy()
excl['signal_weight'] = excl['forced_flow_adv'] / excl['forced_flow_adv'].mean()

print(f"Eligible trades (excluded children): {len(excl)}")
print(f"Signal-weight range: {excl['signal_weight'].min():.2f}× – {excl['signal_weight'].max():.2f}×")

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Panel 1: signal-weighted vs equal-weight bar chart of returns at t=21d
ax = axes[0]
x   = np.arange(len(excl))
ew  = excl['short_pnl_21d'].values * 100
sw  = (excl['short_pnl_21d'] * excl['signal_weight']).values * 100
colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in ew]
ax.bar(x - 0.2, ew, 0.4, color=colors, alpha=0.7, label='Equal-weight')
ax.bar(x + 0.2, sw, 0.4, color='steelblue', alpha=0.6, label='Signal-weight')
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(excl['child_ticker'].values, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Short P&L at t=21d (%)')
ax.set_title('Per-Trade Short P&L — Excluded Children\n(t=21 trading days)', fontweight='bold')
ax.legend()

# Panel 2: cumulative P&L over time
ax = axes[1]
for h in HORIZONS:
    col = f'short_pnl_{h}d'
    valid = excl.dropna(subset=[col])
    ew_cum = valid[col].mean() * 100
    sw_cum = (valid[col] * valid['signal_weight']).sum() / valid['signal_weight'].sum() * 100
    ax.plot(h, ew_cum, 'o', color='#2ecc71', markersize=9)
    ax.plot(h, sw_cum, 's', color='steelblue', markersize=9)

ax.axhline(0, color='black', lw=0.8)
eq_patch  = mpatches.Patch(color='#2ecc71', label='Equal-weight avg')
sw_patch  = mpatches.Patch(color='steelblue', label='Signal-weight avg')
ax.legend(handles=[eq_patch, sw_patch])
ax.set_xlabel('Holding period (trading days)')
ax.set_ylabel('Average Short P&L (%)')
ax.set_title('Average Return by Holding Period\n(excluded children only)', fontweight='bold')
ax.set_xticks(HORIZONS)

plt.suptitle('Short Excluded Spinoff Children — Strategy P&L', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Strategy Summary Statistics

# %%
print('=' * 70)
print('STRATEGY PERFORMANCE — SHORT EXCLUDED CHILDREN')
print('=' * 70)
for h in HORIZONS:
    col = f'short_pnl_{h}d'
    vals = excl[col].dropna()
    sw   = excl.loc[vals.index, 'signal_weight']
    ew_mean = vals.mean()
    sw_mean = (vals * sw).sum() / sw.sum()
    win_rate = (vals > 0).mean()
    t_stat, p_val = stats.ttest_1samp(vals, 0)
    print(f"\n  t={h:2d}d  (n={len(vals)})")
    print(f"    Equal-weight mean:   {ew_mean*100:+.2f}%   Sharpe*: {ew_mean/vals.std():.2f}")
    print(f"    Signal-weight mean:  {sw_mean*100:+.2f}%")
    print(f"    Win rate:            {win_rate:.0%}")
    print(f"    t-stat:  {t_stat:.2f}   p-value: {p_val:.3f}   sig: {'YES ***' if p_val<0.01 else 'YES **' if p_val<0.05 else 'YES *' if p_val<0.1 else 'NO'}")

print('\n* Sharpe computed per-trade (not annualised) for reference')

# %%
# vs. included children — two-sample test at t=21
print('\n=== Two-sample test: excluded vs immediately included at t=21d ===')
excl_21 = car_df[car_df['inclusion_status'] == 'excluded']['short_pnl_21d'].dropna()
incl_21 = car_df[car_df['inclusion_status'] == 'immediately included']['short_pnl_21d'].dropna()

if len(excl_21) > 1 and len(incl_21) > 1:
    t, p = stats.ttest_ind(excl_21, incl_21)
    print(f"  Excluded  (n={len(excl_21)}): mean = {excl_21.mean()*100:+.2f}%")
    print(f"  Included  (n={len(incl_21)}): mean = {incl_21.mean()*100:+.2f}%")
    print(f"  Spread:  {(excl_21.mean()-incl_21.mean())*100:+.2f}%")
    print(f"  t = {t:.2f}   p = {p:.3f}   Significant: {'YES' if p<0.05 else 'NO'}")
else:
    print("  Not enough data for two-sample test")

# %% [markdown]
# ## 5. Position Sizing Analysis
#
# Two sizing models are compared:
# - **Model A — Index-weight sizing**: position size ∝ `parent_index_weight`
# - **Model B — Forced-flow sizing**: position size ∝ `forced_flow_adv`
#
# We check whether larger positions on higher-signal trades improve risk-adjusted returns.

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, col, label, color in zip(
    axes,
    ['parent_index_weight', 'forced_flow_adv'],
    ['Parent Index Weight', 'Forced Flow (× ADV)'],
    ['#8e44ad', '#e67e22']
):
    valid = excl.dropna(subset=[col, 'short_pnl_21d'])
    x = valid[col].values
    y = valid['short_pnl_21d'].values * 100

    ax.scatter(x, y, color=color, s=80, alpha=0.8, zorder=3)

    # Label each point
    for _, row in valid.iterrows():
        ax.annotate(
            row['child_ticker'],
            (row[col], row['short_pnl_21d'] * 100),
            fontsize=7, ha='left', va='bottom',
            xytext=(3, 3), textcoords='offset points'
        )

    # Regression line
    if len(valid) > 2:
        m, b = np.polyfit(x, y, 1)
        x_ = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_, m * x_ + b, '--', color=color, alpha=0.7, lw=1.5)
        r, p = stats.pearsonr(x, y)
        ax.set_title(
            f'Short P&L t=21d vs {label}\nr = {r:.2f}   p = {p:.3f}',
            fontweight='bold'
        )

    ax.axhline(0, color='black', lw=0.7)
    ax.set_xlabel(label)
    ax.set_ylabel('Short P&L t=21d (%)')

plt.suptitle('Position Sizing Signal vs Short Returns (excluded children)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Parent Leg — Short Deleted Parents
#
# When the parent is removed from the S&P 500 within ~1 year of the spinoff,
# passive funds must sell the parent as well. We test whether shorting the parent
# at the spinoff effective date captures this additional alpha.

# %%
if crsp is not None and events['parent_deleted_1yr'].any():
    def build_parent_window(permno, eff_date, crsp_df, idx_df, post=90):
        parent = crsp_df[crsp_df['permno'] == int(permno)].sort_values('date').copy()
        valid  = parent[parent['date'] >= eff_date]
        if len(valid) == 0:
            return pd.DataFrame()
        window = valid.head(post + 1).copy()
        window = window.merge(idx_df[['date', 'sprtrn']], on='date', how='left')
        window['t'] = range(len(window))
        t0 = window.iloc[0]['adj_prc']
        if pd.isna(t0) or t0 == 0:
            t0 = window.iloc[0]['prc'].abs()
        window['ret_mkt_adj'] = window['ret'] - window['sprtrn']
        window['car']         = window['ret_mkt_adj'].cumsum()
        window['short_pnl']   = -window['car']
        return window

    parent_deleted = events[events['parent_deleted_1yr'] == True].copy()
    parent_rows = []
    for _, ev in parent_deleted.iterrows():
        if pd.isna(ev['permno']):
            continue
        w = build_parent_window(int(ev['permno']), ev['effective_date'], crsp, idx_ret, post=90)
        if len(w) == 0:
            continue
        w['parent_ticker'] = ev['parent_ticker']
        parent_rows.append(w)

    if parent_rows:
        all_pw = pd.concat(parent_rows, ignore_index=True)
        avg    = all_pw.groupby('t')[['car', 'short_pnl']].mean().reset_index()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ticker, grp in all_pw.groupby('parent_ticker'):
            axes[0].plot(grp['t'], grp['short_pnl'] * 100, alpha=0.4, lw=1)
            axes[1].plot(grp['t'], grp['car'] * 100, alpha=0.4, lw=1)
        axes[0].plot(avg['t'], avg['short_pnl'] * 100, 'k-', lw=2.5, label='Average')
        axes[1].plot(avg['t'], avg['car'] * 100, 'k-', lw=2.5, label='Average')
        for ax, title, ylabel in zip(
            axes,
            ['Short P&L — Deleted Parents', 'Child CAR — Deleted Parents'],
            ['Short P&L (%)', 'Cumulative Abnormal Return (%)']
        ):
            ax.axhline(0, color='black', lw=0.7, linestyle='--')
            ax.axvline(0, color='red', lw=1, linestyle='--')
            ax.set_xlabel('Trading days since spinoff effective date')
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight='bold')
            ax.legend()
        plt.suptitle('Parent Short — S&P 500 Deletion Events', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

        print('=== Parent short P&L (deleted within 1yr) ===')
        for h in HORIZONS:
            sub = all_pw[all_pw['t'] == h].groupby('parent_ticker')['short_pnl'].mean()
            if len(sub) == 0:
                continue
            t_stat, p_val = stats.ttest_1samp(sub, 0) if len(sub) > 1 else (np.nan, np.nan)
            print(f"  t={h:2d}d  mean={sub.mean()*100:+.2f}%  n={len(sub)}  "
                  f"p={p_val:.3f}" if not np.isnan(p_val) else f"  t={h:2d}d  mean={sub.mean()*100:+.2f}%  n={len(sub)}")
    else:
        print("[!] No parent price data available for deleted parents")
else:
    print("[!] Either crsp_daily.parquet missing or no parent deletion events — skipping parent leg")

# %% [markdown]
# ## 7. Regression Analysis
#
# We use four regression frameworks to prove the forced-flow signal:
#
# 1. **Simple OLS** — `short_pnl_21d ~ forced_flow_adv`
# 2. **Multiple OLS** — `short_pnl_21d ~ forced_flow_adv + parent_index_weight + announce_lag_days`
# 3. **Polynomial (degree 2)** — adds squared terms
# 4. **LASSO** — L1 regularisation (feature selection)
# 5. **Ridge** — L2 regularisation (shrinkage)
#
# All models use Leave-One-Out CV given the small sample size (~30 events).

# %%
# --- Prepare regression dataset ---
REG_TARGET  = 'short_pnl_21d'
REG_FEATURES = ['forced_flow_adv', 'parent_index_weight', 'announce_lag_days']

reg_df = car_df[car_df['inclusion_status'] == 'excluded'].copy()
reg_df = reg_df.dropna(subset=[REG_TARGET] + REG_FEATURES)

# Winsorise forced_flow_adv at 99th percentile to reduce outlier leverage
ff_cap = reg_df['forced_flow_adv'].quantile(0.99)
reg_df['forced_flow_adv'] = reg_df['forced_flow_adv'].clip(upper=ff_cap)

y = reg_df[REG_TARGET].values * 100         # in percent
X_raw = reg_df[REG_FEATURES].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print(f"Regression sample: n = {len(reg_df)}")
print(f"Features: {REG_FEATURES}")
print(f"Target: {REG_TARGET} (in %)")
print(f"\nTarget stats:  mean={y.mean():.2f}%  std={y.std():.2f}%  "
      f"min={y.min():.2f}%  max={y.max():.2f}%")

# %%
# --- Utility: LOO R² ---
def loo_r2(model, X, y):
    loo = LeaveOneOut()
    preds = cross_val_score(model, X, y, cv=loo, scoring='r2')
    return preds.mean()

# %% [markdown]
# ### 7.1 Simple OLS — `short_pnl_21d ~ forced_flow_adv`

# %%
X1 = sm.add_constant(reg_df[['forced_flow_adv']].values)
ols1 = sm.OLS(y, X1).fit()
print(ols1.summary())

fig, ax = plt.subplots(figsize=(9, 5))
x_plot = np.linspace(reg_df['forced_flow_adv'].min(), reg_df['forced_flow_adv'].max(), 100)
y_hat  = ols1.params[0] + ols1.params[1] * x_plot
ci     = ols1.get_prediction(sm.add_constant(x_plot)).summary_frame(alpha=0.05)

ax.fill_between(x_plot, ci['mean_ci_lower'], ci['mean_ci_upper'], alpha=0.15, color='steelblue', label='95% CI')
ax.plot(x_plot, y_hat, color='steelblue', lw=2, label=f'OLS  (β={ols1.params[1]:.2f}, p={ols1.pvalues[1]:.3f})')
ax.scatter(reg_df['forced_flow_adv'], y, color='#d32f2f', s=70, zorder=5, label='Observations')
for _, row in reg_df.iterrows():
    ax.annotate(row['child_ticker'], (row['forced_flow_adv'], row[REG_TARGET]*100),
                fontsize=7, xytext=(3, 3), textcoords='offset points')
ax.axhline(0, color='black', lw=0.7)
ax.set_xlabel('Forced Flow (× ADV)')
ax.set_ylabel('Short P&L t=21d (%)')
ax.set_title('Simple OLS: Short P&L ~ Forced Flow\n(excluded children)', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()

loo_r2_1 = loo_r2(LinearRegression(), reg_df[['forced_flow_adv']].values, y)
print(f"\nIn-sample R²:  {ols1.rsquared:.3f}")
print(f"LOO R²:        {loo_r2_1:.3f}")

# %% [markdown]
# ### 7.2 Multiple OLS — `short_pnl_21d ~ forced_flow_adv + parent_index_weight + announce_lag_days`

# %%
X2 = sm.add_constant(X_scaled)
ols2 = sm.OLS(y, X2).fit()
print(ols2.summary())

# Coefficient plot
fig, ax = plt.subplots(figsize=(8, 4))
feat_names = ['const'] + REG_FEATURES
coefs  = ols2.params
errors = ols2.bse * 1.96   # 95% CI
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in coefs]
ax.barh(feat_names, coefs, xerr=errors, color=colors, alpha=0.7, capsize=5)
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('Standardised coefficient (95% CI)')
ax.set_title('Multiple OLS — Standardised Coefficients\n(excluded children, t=21d)', fontweight='bold')
plt.tight_layout()
plt.show()

loo_r2_2 = loo_r2(LinearRegression(), X_scaled, y)
print(f"\nIn-sample R²:  {ols2.rsquared:.3f}")
print(f"LOO R²:        {loo_r2_2:.3f}")

# %% [markdown]
# ### 7.3 Polynomial Regression (degree 2) — captures non-linear signal

# %%
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(reg_df[['forced_flow_adv']].values)
X_poly_sc = StandardScaler().fit_transform(X_poly)

X3 = sm.add_constant(X_poly_sc)
ols3 = sm.OLS(y, X3).fit()
print(f"Polynomial OLS (deg=2):  R²={ols3.rsquared:.3f}  "
      f"adj-R²={ols3.rsquared_adj:.3f}  AIC={ols3.aic:.1f}")

# Plot
x_seq = np.linspace(reg_df['forced_flow_adv'].min(), reg_df['forced_flow_adv'].max(), 200).reshape(-1, 1)
X_seq_poly = StandardScaler().fit(X_poly).transform(poly.transform(x_seq))
y_poly = sm.OLS(y, sm.add_constant(X_poly_sc)).fit().predict(sm.add_constant(X_seq_poly))

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(reg_df['forced_flow_adv'], y, color='#d32f2f', s=70, zorder=5, label='Observations')
ax.plot(x_seq, y_poly, color='#8e44ad', lw=2, label=f'Polynomial deg=2  (R²={ols3.rsquared:.3f})')
# Also plot simple OLS for comparison
ax.plot(x_plot, y_hat, '--', color='steelblue', lw=1.5, alpha=0.7, label=f'Simple OLS  (R²={ols1.rsquared:.3f})')
ax.axhline(0, color='black', lw=0.7)
ax.set_xlabel('Forced Flow (× ADV)')
ax.set_ylabel('Short P&L t=21d (%)')
ax.set_title('Polynomial vs Simple OLS\n(excluded children, t=21d)', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()

loo_r2_3 = loo_r2(Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)),
                              ('sc', StandardScaler()),
                              ('lr', LinearRegression())]),
                   reg_df[['forced_flow_adv']].values, y)
print(f"LOO R²:        {loo_r2_3:.3f}")

# %% [markdown]
# ### 7.4 LASSO — L1 Regularisation (feature selection)
#
# LASSO shrinks small coefficients to exactly zero, effectively selecting the most informative features.

# %%
lasso_cv = LassoCV(cv=LeaveOneOut(), max_iter=10000)
lasso_cv.fit(X_scaled, y)

lasso_coefs = pd.Series(lasso_cv.coef_, index=REG_FEATURES)
print(f"LASSO optimal α: {lasso_cv.alpha_:.4f}")
print("\nNon-zero coefficients:")
print(lasso_coefs[lasso_coefs != 0].to_string())
print("\nZeroed out (not informative):")
print(lasso_coefs[lasso_coefs == 0].index.tolist())

fig, ax = plt.subplots(figsize=(8, 4))
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in lasso_coefs]
ax.bar(REG_FEATURES, lasso_coefs.values, color=colors, alpha=0.8)
ax.axhline(0, color='black', lw=0.8)
ax.set_ylabel('LASSO coefficient (standardised)')
ax.set_title(f'LASSO Coefficients  (α={lasso_cv.alpha_:.4f})\n'
             f'Zeroed features = not informative for short P&L', fontweight='bold')
plt.tight_layout()
plt.show()

y_lasso = lasso_cv.predict(X_scaled)
r2_lasso_is = r2_score(y, y_lasso)
r2_lasso_loo = loo_r2(Lasso(alpha=lasso_cv.alpha_, max_iter=10000), X_scaled, y)
print(f"\nIn-sample R²:  {r2_lasso_is:.3f}")
print(f"LOO R²:        {r2_lasso_loo:.3f}")

# %% [markdown]
# ### 7.5 Ridge — L2 Regularisation (coefficient shrinkage)
#
# Ridge keeps all features but shrinks coefficients proportionally, reducing overfitting.

# %%
ridge_cv = RidgeCV(alphas=np.logspace(-3, 4, 100), cv=LeaveOneOut())
ridge_cv.fit(X_scaled, y)

ridge_coefs = pd.Series(ridge_cv.coef_, index=REG_FEATURES)
print(f"Ridge optimal α: {ridge_cv.alpha_:.4f}")
print("\nCoefficients:")
print(ridge_coefs.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Coefficient plot
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in ridge_coefs]
axes[0].bar(REG_FEATURES, ridge_coefs.values, color=colors, alpha=0.8)
axes[0].axhline(0, color='black', lw=0.8)
axes[0].set_ylabel('Ridge coefficient (standardised)')
axes[0].set_title(f'Ridge Coefficients  (α={ridge_cv.alpha_:.4f})', fontweight='bold')

# Predicted vs actual
y_ridge = ridge_cv.predict(X_scaled)
r2_ridge_is = r2_score(y, y_ridge)
axes[1].scatter(y, y_ridge, color='#1976d2', s=70, alpha=0.8)
lim = [min(y.min(), y_ridge.min()) - 1, max(y.max(), y_ridge.max()) + 1]
axes[1].plot(lim, lim, 'k--', lw=1)
axes[1].set_xlabel('Actual short P&L (%)')
axes[1].set_ylabel('Ridge predicted (%)')
axes[1].set_title(f'Ridge: Predicted vs Actual  (R²={r2_ridge_is:.3f})', fontweight='bold')

plt.tight_layout()
plt.show()

r2_ridge_loo = loo_r2(Ridge(alpha=ridge_cv.alpha_), X_scaled, y)
print(f"\nIn-sample R²:  {r2_ridge_is:.3f}")
print(f"LOO R²:        {r2_ridge_loo:.3f}")

# %% [markdown]
# ## 8. Regression Model Comparison

# %%
model_summary = pd.DataFrame([
    {'Model': 'Simple OLS',         'Features': 'forced_flow_adv',                        'In-sample R²': ols1.rsquared,       'LOO R²': loo_r2_1, 'AIC': ols1.aic},
    {'Model': 'Multiple OLS',       'Features': 'FF + weight + lag',                      'In-sample R²': ols2.rsquared,       'LOO R²': loo_r2_2, 'AIC': ols2.aic},
    {'Model': 'Polynomial (deg=2)', 'Features': 'FF + FF²',                               'In-sample R²': ols3.rsquared,       'LOO R²': loo_r2_3, 'AIC': ols3.aic},
    {'Model': 'LASSO',              'Features': 'FF + weight + lag (L1 selected)',         'In-sample R²': r2_lasso_is,          'LOO R²': r2_lasso_loo, 'AIC': None},
    {'Model': 'Ridge',              'Features': 'FF + weight + lag (L2 shrunk)',           'In-sample R²': r2_ridge_is,          'LOO R²': r2_ridge_loo, 'AIC': None},
])
model_summary = model_summary.set_index('Model')
print(model_summary.to_string())

fig, ax = plt.subplots(figsize=(10, 4))
x      = np.arange(len(model_summary))
width  = 0.35
ax.bar(x - width/2, model_summary['In-sample R²'], width, label='In-sample R²', color='steelblue', alpha=0.8)
ax.bar(x + width/2, model_summary['LOO R²'],       width, label='LOO R²',       color='#e67e22',   alpha=0.8)
ax.axhline(0, color='black', lw=0.7)
ax.set_xticks(x)
ax.set_xticklabels(model_summary.index, rotation=15, ha='right')
ax.set_ylabel('R²')
ax.set_title('Model Comparison: In-Sample vs Leave-One-Out R²\n'
             '(LOO R² penalises overfitting — prefer models where both are high)', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Final Strategy Summary

# %%
print('=' * 70)
print('SPINOFF INDEX ARBITRAGE — STRATEGY SUMMARY')
print('=' * 70)

print('\n--- UNIVERSE ---')
total   = len(events)
excl_n  = (events['inclusion_status'] == 'excluded').sum()
incl_n  = (events['inclusion_status'] == 'immediately included').sum()
later_n = (events['inclusion_status'] == 'later included').sum()
del_n   = events['parent_deleted_1yr'].sum()
print(f"  Total spinoff events (2020-2025):   {total}")
print(f"  Child excluded (trade ✓):           {excl_n}  ({excl_n/total:.0%})")
print(f"  Child immediately included (skip):  {incl_n}  ({incl_n/total:.0%})")
print(f"  Child later included:               {later_n}  ({later_n/total:.0%})")
print(f"  Parent deleted within 1yr (trade ✓):{del_n}  ({del_n/total:.0%})")

print('\n--- SIGNAL (forced_flow_adv, excluded children) ---')
ff = events.loc[events['inclusion_status']=='excluded','forced_flow_adv'].dropna()
print(f"  Median:  {ff.median():.1f}× ADV")
print(f"  Mean:    {ff.mean():.1f}× ADV")
print(f"  Max:     {ff.max():.1f}× ADV")
print(f"  >10× ADV: {(ff>=10).sum()} events  |  >5× ADV: {(ff>=5).sum()} events")

print('\n--- CHILD SHORT RETURNS (excluded, equal-weight) ---')
for h in [5, 10, 21, 63]:
    col = f'short_pnl_{h}d'
    vals = excl[col].dropna()
    t, p = stats.ttest_1samp(vals, 0)
    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    print(f"  t={h:2d}d:  mean={vals.mean()*100:+.2f}%  win={( vals>0).mean():.0%}  "
          f"t={t:.2f}  p={p:.3f} {sig}")

print('\n--- REGRESSION PROOF (best model by LOO R²) ---')
best = model_summary['LOO R²'].idxmax()
row  = model_summary.loc[best]
print(f"  Best model:   {best}")
print(f"  Features:     {row['Features']}")
print(f"  In-sample R²: {row['In-sample R²']:.3f}")
print(f"  LOO R²:       {row['LOO R²']:.3f}")

print('\n--- SIZING RECOMMENDATION ---')
print('  Size ∝ forced_flow_adv (higher signal → more selling pressure → larger short)')
print('  Cap at 10× ADV to avoid outlier crowding; hedge with long S&P 500 futures')
print('  Entry:  effective date close')
print('  Exits:  child added to S&P 500 | 21-day base | 63-day max | +10% stop-loss')

print('\n--- NEXT STEPS ---')
print('  1. Deletion probability model (logistic on parent fundamentals)')
print('  2. Composite signal: FF_ADV × (1 − deletion_prob_parent)')
print('  3. Transaction cost model: bid-ask + market impact vs signal strength')
print('  4. Portfolio backtest with proper position sizing and rebalancing')
