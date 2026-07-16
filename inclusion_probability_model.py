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
# # Index Inclusion Probability Model
#
# **Question:** at the moment a spinoff becomes effective, can we estimate the
# probability that the child gets added to the S&P 500 **immediately**
# (within 5 trading days), using only information available at that moment?
#
# This matters for the long strategy in `spinoff_post_completion_strategy.ipynb`:
# a child that is unlikely to be added keeps facing passive-fund selling
# pressure after completion (a headwind for a long position), while a child
# likely to be added avoids — or even reverses — that drag. `prob_included`
# is a candidate signal for tilting/screening which children to hold long.
#
# This notebook does **only** the modeling step: build the label, build the
# features, fit the model, evaluate it honestly, and hand back a clean table
# of per-event probabilities. Whether/how to fold `prob_included` into the
# long strategy is a separate decision for a later notebook — this one is not
# a backtest.
#
# ## Method: logistic regression, evaluated by leave-one-out cross-validation
#
# **Why logistic regression.** The target is binary (included immediately /
# not), and what we actually want out of the model is a *probability*, not
# just a class label — logistic regression is the standard tool for that: it
# fits a linear model in log-odds space,
#
# $$\text{logit}(p) = \log\frac{p}{1-p} = \beta_0 + \beta_1 x_1 + \dots + \beta_6 x_6,
# \qquad p = P(\text{immediately included})$$
#
# and its coefficients are directly interpretable (sign and, once features are
# standardized, relative magnitude). With only a few features and a small
# sample, a simple linear-in-log-odds model is the right level of complexity —
# there isn't enough data to justify a more flexible, higher-variance model
# (e.g. a tree ensemble).
#
# **Why leave-one-out CV.** The full sample is small — see the count below —
# so a conventional train/test split would waste data and give a noisy,
# arbitrary estimate depending on which rows land in the test fold.
# Leave-one-out (LOO) instead holds out **one event at a time**, refits the
# model on all the others, and predicts that one held-out event. Repeating
# this for every event gives an out-of-sample probability for *every* row
# without ever letting a model see the event it's predicting — this is the
# `prob_included` column in the final table, and it's what the metrics below
# are computed from. It uses the data as efficiently as possible while still
# being a genuine out-of-sample estimate.
#
# **Why standardize features first.** Logistic regression coefficients are
# only comparable to each other (to judge which feature matters most) if the
# features are on the same scale — otherwise a feature measured in billions
# would mechanically get a tiny coefficient next to one measured in single
# digits. We standardize (zero mean, unit variance) via `StandardScaler`,
# fit *inside* each LOO training fold so no information about the held-out
# event's own scale leaks into its prediction.
#
# ## Features and lookahead audit
#
# Every feature must be knowable at the effective date (when we'd actually
# put a trade on) — nothing here is allowed to peek at what happens after.
#
# | Feature | Definition | Known at effective date? |
# |---|---|---|
# | `log_child_mktcap` | log(child market cap at t=0 close) | Yes — t=0 close |
# | `log_size_ratio` | log(child mktcap / that year's S&P 500 min-mktcap threshold) | Yes — t=0 close |
# | `parent_index_weight` | parent's weight in the S&P 500 pre-spinoff | Yes — pre-effective |
# | `forced_flow_adv` | passive AUM × parent weight ÷ parent ADV | Yes — pre-effective |
# | `log_lag_days` | log(days from announcement to effective date) | Yes — known once announced |
# | `passive_aum_B` | total passive S&P 500 AUM ($B) at announcement | Yes — pre-effective |
#
# **Target:** `target = 1` if the child is added to the S&P 500 within 5
# trading days of the effective date ("immediately included"), else `0`
# (excluded, or added later — both are lumped together because both mean the
# child does *not* get the immediate passive-inflow benefit).

# %%
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:,.4f}'.format)

RAW_DIR   = Path('data/raw')
CLEAN_DIR = Path('data/clean')

# %% [markdown]
# ## 1. Load data

# %%
events      = pd.read_csv(
    CLEAN_DIR / 'spinoff_events_merged.csv',
    parse_dates=['announce_date', 'effective_date', 'sp500_start', 'sp500_end']
)
children    = pd.read_parquet(RAW_DIR / 'spinoff_children_crsp.parquet')
child_sp500 = pd.read_parquet(RAW_DIR / 'spinoff_children_sp500.parquet')

print(f'Events loaded: {len(events)}')
print(f'Children CRSP: {children["permno"].nunique()} spinoff children, {len(children):,} rows')
print(f'S&P 500 membership records for children: {len(child_sp500)}')

# %% [markdown]
# ## 2. Build the target label
#
# A child is **immediately included** if its first S&P 500 addition date is
# within 5 trading days of the spinoff's effective date. Everything else
# (never added, or added later) is `0`.

# %%
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
events['target'] = (events['inclusion_status'] == 'immediately included').astype(int)

print(events['inclusion_status'].value_counts().to_string())
print(f'\nImmediate-inclusion rate: {events["target"].mean():.0%}')

# %% [markdown]
# ## 3. Build features
#
# `log_size_ratio` needs the S&P 500 minimum market-cap eligibility threshold
# for the year the spinoff became effective — these are S&P Dow Jones
# Indices' published minimums, hand-entered here since they aren't in a data
# pull.

# %%
SP500_MIN_MKTCAP = {2020: 8.2e9, 2021: 11.8e9, 2022: 13.1e9, 2023: 12.7e9, 2024: 18.0e9}

def sp500_threshold(year):
    return SP500_MIN_MKTCAP.get(
        year, SP500_MIN_MKTCAP[min(SP500_MIN_MKTCAP, key=lambda y: abs(y - year))]
    )

child_t0 = (
    children.sort_values('date')
    .groupby('permno').first().reset_index()
    [['permno', 'mktcap', 'child_ticker']]
    .rename(columns={'mktcap': 'child_mktcap_t0'})
)

model_df = events.merge(
    child_t0[['child_ticker', 'child_mktcap_t0']],
    left_on='spinoff_ticker', right_on='child_ticker', how='left'
)

model_df['year'] = model_df['effective_date'].dt.year
model_df['size_threshold']  = model_df['year'].map(sp500_threshold)
model_df['log_child_mktcap'] = np.where(
    model_df['child_mktcap_t0'] > 0, np.log(model_df['child_mktcap_t0']), np.nan)
model_df['log_size_ratio'] = np.where(
    model_df['child_mktcap_t0'] > 0,
    np.log(model_df['child_mktcap_t0'] / model_df['size_threshold']), np.nan)
lag = (model_df['effective_date'] - model_df['announce_date']).dt.days
model_df['log_lag_days'] = np.where(lag > 0, np.log(lag), np.nan)
model_df['passive_aum_B'] = model_df['passive_aum_usd'] / 1e9

feature_cols = [
    'log_child_mktcap', 'log_size_ratio', 'parent_index_weight',
    'forced_flow_adv', 'log_lag_days', 'passive_aum_B',
]

model_clean = model_df.dropna(subset=feature_cols + ['target']).copy().reset_index(drop=True)
print(f'Usable sample: {len(model_clean)} events '
      f'(dropped {len(model_df) - len(model_clean)} for missing features)')
print(f'Target balance in usable sample: {dict(model_clean["target"].value_counts())} '
      f'(inclusion rate {model_clean["target"].mean():.0%})')

# %% [markdown]
# ## 4. Fit with leave-one-out cross-validation
#
# For each event: scale features using only the *other* events, fit logistic
# regression on the other events, predict this event's probability. The
# resulting `prob_included` is a genuine out-of-sample estimate for every row
# — no event ever influences its own prediction.

# %%
X = model_clean[feature_cols].values
y = model_clean['target'].values

loo_probs = np.zeros(len(y))
loo_preds = np.zeros(len(y), dtype=int)

for train_idx, test_idx in LeaveOneOut().split(X):
    scaler = StandardScaler().fit(X[train_idx])
    X_train = scaler.transform(X[train_idx])
    X_test  = scaler.transform(X[test_idx])

    clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf.fit(X_train, y[train_idx])

    loo_probs[test_idx] = clf.predict_proba(X_test)[:, 1]
    loo_preds[test_idx] = clf.predict(X_test)

model_clean['prob_included'] = loo_probs
model_clean['pred_included'] = loo_preds
model_clean['correct'] = (model_clean['pred_included'] == model_clean['target'])

print('LOO-CV fitting complete.')

# %% [markdown]
# ## 5. Full-sample fit — for interpreting coefficients only
#
# This second fit uses *all* events at once. It is **not** used to produce
# `prob_included` above (that would leak information) — it exists only so we
# can read off which features the model leans on.

# %%
scaler_full = StandardScaler()
X_full = scaler_full.fit_transform(X)
clf_full = LogisticRegression(C=1.0, max_iter=500, random_state=42)
clf_full.fit(X_full, y)

coef_df = pd.DataFrame({
    'feature': feature_cols,
    'standardized_coef': clf_full.coef_[0],
})
coef_df['abs_coef'] = coef_df['standardized_coef'].abs()
coef_df = coef_df.sort_values('abs_coef', ascending=False).drop(columns='abs_coef')
coef_df['direction'] = np.where(coef_df['standardized_coef'] > 0,
                                 '+ → more likely included', '− → less likely included')

print('=== Standardized coefficients (full-sample fit, interpretation only) ===')
print(coef_df.to_string(index=False))

# %% [markdown]
# ## 6. Metrics table
#
# All metrics are computed from the LOO out-of-sample predictions
# (`prob_included` / `pred_included` at a 0.5 threshold), not the full-sample
# fit — this is the honest read of how well the model would have done
# predicting each event without having seen its outcome.

# %%
majority_baseline = max(y.mean(), 1 - y.mean())
tn, fp, fn, tp = confusion_matrix(y, loo_preds).ravel()

metrics_table = pd.DataFrame([{
    'N (events)':            len(y),
    'Inclusion rate':        f'{y.mean():.0%}',
    'Majority-class baseline': f'{majority_baseline:.0%}',
    'LOO Accuracy':          f'{accuracy_score(y, loo_preds):.0%}',
    'Precision':             f'{precision_score(y, loo_preds, zero_division=0):.2f}',
    'Recall':                f'{recall_score(y, loo_preds, zero_division=0):.2f}',
    'F1':                    f'{f1_score(y, loo_preds, zero_division=0):.2f}',
    'AUC':                   f'{roc_auc_score(y, loo_probs):.3f}' if len(np.unique(y)) > 1 else 'n/a',
    'True Positives':        tp,
    'False Positives':       fp,
    'True Negatives':        tn,
    'False Negatives':       fn,
}]).T.rename(columns={0: 'value'})

print('=== LOO-CV Metrics ===')
display(metrics_table)

print('\nReading precision/recall here: "positive" = predicted immediately included.')
print(f'  Precision = {tp}/{tp+fp} = of the events the model called "included", '
      f'how many really were.')
print(f'  Recall    = {tp}/{tp+fn} = of the events truly included, '
      f'how many the model caught.')

# %% [markdown]
# ## 7. Full sample: probabilities and outcomes
#
# One row per spinoff event: the features that went in, the LOO-CV
# probability that came out, the resulting 0.5-threshold classification, the
# true label, and whether the model got it right. Sorted by predicted
# probability, highest first.

# %%
results_table = model_clean[[
    'spinoff_ticker', 'parent_ticker', 'effective_date',
    'log_child_mktcap', 'log_size_ratio', 'parent_index_weight',
    'forced_flow_adv', 'log_lag_days', 'passive_aum_B',
    'prob_included', 'pred_included', 'inclusion_status', 'target', 'correct',
]].sort_values('prob_included', ascending=False).reset_index(drop=True)

results_table_display = results_table.copy()
results_table_display['prob_included'] = (results_table_display['prob_included'] * 100).round(1)
results_table_display['pred_included']  = results_table_display['pred_included'].map({1: 'included', 0: 'not included'})
results_table_display = results_table_display.rename(columns={
    'spinoff_ticker': 'child', 'parent_ticker': 'parent',
    'prob_included': 'P(included) %', 'pred_included': 'model call',
    'inclusion_status': 'actual status', 'target': 'actual label', 'correct': 'model correct',
})

print(f'=== Full sample: {len(results_table_display)} events ===')
display(results_table_display)

# %% [markdown]
# ## 8. Save the results table
#
# Persist the per-event probability table so it can be joined against other
# notebooks without re-running the model.

# %%
results_out_path = CLEAN_DIR / 'inclusion_probability_results.csv'
results_table_display.to_csv(results_out_path, index=False)
print(f'Saved {len(results_table_display)} rows to {results_out_path}')

# %% [markdown]
# ## 9. Does `prob_included` say anything about the **parent's** post-completion CAR?
#
# `spinoff_post_completion_strategy.ipynb` runs a long-the-parent strategy:
# buy the parent stock at the spinoff's effective date, hold up to 252 trading
# days, hedge with an estimated beta (`hedged_car = ret − β·sprtrn`,
# cumulative). This section rebuilds that same parent CAR path — same beta
# estimation, same construction — purely to test one question: **is a child
# more likely to be added to the S&P 500 associated with a better or worse
# outcome for the parent that spun it off?**
#
# The mechanism this would test: if the market already prices in the child's
# eventual index inclusion (or exclusion) around the spinoff, that
# information might also show up in how the parent re-rates post-completion
# (e.g. a "clean" spinoff with an index-eligible child vs. a messier one).
# This is exploratory — a low/insignificant correlation is a legitimate,
# useful answer (it says the parent leg and the child's inclusion odds are
# separate bets), not a failed analysis.

# %%
crsp    = pd.read_parquet(RAW_DIR / 'crsp_daily.parquet').sort_values(['permno', 'date']).reset_index(drop=True)
idx_ret = pd.read_parquet(RAW_DIR / 'crsp_index_returns.parquet')
by_permno = {p: g.reset_index(drop=True) for p, g in crsp.groupby('permno')}

events_parent = events.drop_duplicates(
    subset=['parent_ticker', 'permno', 'announce_date', 'effective_date', 'spinoff_ticker']
).reset_index(drop=True)
events_parent = events_parent[
    (events_parent['effective_date'] - events_parent['announce_date']).dt.days >= 0
].reset_index(drop=True)

def estimate_beta(permno, before_date, window=252, min_obs=60):
    g = by_permno.get(permno)
    if g is None:
        return 1.0
    pos = g.index[g['date'] < before_date]
    if len(pos) < min_obs:
        return 1.0
    i0 = pos[-1]
    w = g.iloc[max(0, i0 - window):i0 + 1].merge(idx_ret[['date', 'sprtrn']], on='date', how='left').dropna(subset=['ret', 'sprtrn'])
    if len(w) < min_obs:
        return 1.0
    beta = np.cov(w['ret'], w['sprtrn'])[0, 1] / np.var(w['sprtrn'])
    return float(np.clip(beta, 0.3, 2.5))

events_parent['beta'] = [
    estimate_beta(p, d) for p, d in zip(events_parent['permno'], events_parent['announce_date'])
]

HOLD = 252
paths = []
for _, ev in events_parent.iterrows():
    g = by_permno.get(ev['permno'])
    if g is None:
        continue
    pos = g.index[g['date'] <= ev['effective_date']]
    if len(pos) == 0:
        continue
    i0 = pos[-1]
    w = g.iloc[i0:min(len(g), i0 + HOLD + 1)].merge(idx_ret[['date', 'sprtrn']], on='date', how='left')
    if len(w) < HOLD + 1:
        continue
    w['t'] = np.arange(len(w))
    w['naive_car']  = (w['ret'] - w['sprtrn']).cumsum()
    w['hedged_car'] = (w['ret'] - ev['beta'] * w['sprtrn']).cumsum()
    w['parent_ticker'], w['spinoff_ticker'] = ev['parent_ticker'], ev['spinoff_ticker']
    paths.append(w)

all_paths = pd.concat(paths, ignore_index=True) if paths else pd.DataFrame()
print(f'Parent post-completion paths built: {all_paths["spinoff_ticker"].nunique() if len(all_paths) else 0} events '
      f'(full {HOLD}-day path available)')

# %%
car_horizons = [21, 63, 126, 252]
parent_car = (
    all_paths[all_paths['t'].isin(car_horizons)]
    [['parent_ticker', 'spinoff_ticker', 't', 'naive_car', 'hedged_car']]
)
parent_car_wide = parent_car.pivot(index=['parent_ticker', 'spinoff_ticker'], columns='t',
                                    values=['naive_car', 'hedged_car'])
parent_car_wide.columns = [f'{kind}_{h}d' for kind, h in parent_car_wide.columns]
parent_car_wide = parent_car_wide.reset_index()

# %% [markdown]
# ### Join `prob_included` (child) onto the parent's CAR (same spinoff event)

# %%
joined = results_table[['spinoff_ticker', 'parent_ticker', 'prob_included', 'inclusion_status']].merge(
    parent_car_wide, on=['spinoff_ticker', 'parent_ticker'], how='inner'
)
print(f'Events with both prob_included and full parent CAR path: {len(joined)}')

joined_display = joined.copy()
joined_display['prob_included'] = (joined_display['prob_included'] * 100).round(1)
for col in [c for c in joined_display.columns if c.endswith('d')]:
    joined_display[col] = (joined_display[col] * 100).round(2)
joined_display = joined_display.rename(columns={
    'spinoff_ticker': 'child', 'parent_ticker': 'parent', 'prob_included': 'P(included) %',
})

print('=== Per-event: P(child included) vs parent post-completion CAR (%) ===')
display(joined_display.sort_values('P(included) %', ascending=False))

# %% [markdown]
# ### Correlation table: `prob_included` vs. parent CAR, by horizon

# %%
corr_rows = []
for h in car_horizons:
    for kind, label in [('naive_car', 'naive (unhedged)'), ('hedged_car', 'beta-hedged')]:
        col = f'{kind}_{h}d'
        sub = joined.dropna(subset=['prob_included', col])
        if len(sub) > 2:
            r, p = stats.pearsonr(sub['prob_included'], sub[col])
        else:
            r, p = np.nan, np.nan
        corr_rows.append({
            'horizon (trading days)': h,
            'parent CAR type': label,
            'n': len(sub),
            'pearson r': round(r, 3) if pd.notna(r) else np.nan,
            'p-value': round(p, 3) if pd.notna(p) else np.nan,
        })

corr_table = pd.DataFrame(corr_rows)
print('=== Correlation: prob_included vs. parent post-completion CAR ===')
display(corr_table)
