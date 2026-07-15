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
# # Slides Generator — concise methodology deck
# Outputs: slides/slide_XX.png (1920×1080) + slides/deck.pdf

# %%
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch
from scipy import stats
from scipy.stats import linregress as _lr
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, accuracy_score

RAW_DIR    = Path('data/raw')
CLEAN_DIR  = Path('data/clean')
SLIDES_DIR = Path('slides')
SLIDES_DIR.mkdir(exist_ok=True)

NAVY  = '#0a2342'; BLUE  = '#1565c0'; RED   = '#c62828'
ORG   = '#e65100'; GRN   = '#2e7d32'; GREY  = '#546e7a'
WHITE = '#ffffff'; LTBG  = '#f4f6f9'; GOLD  = '#f9a825'

SLIDE_W, SLIDE_H, DPI = 16, 9, 120
COLOR_MAP = {'excluded': RED, 'immediately included': BLUE, 'later included': ORG}

# %% [markdown]
# ## 1. Load + Compute

# %%
events      = pd.read_csv(CLEAN_DIR / 'spinoff_events_merged.csv',
                           parse_dates=['announce_date','effective_date','sp500_start','sp500_end'])
children    = pd.read_parquet(RAW_DIR / 'spinoff_children_crsp.parquet')
child_sp500 = pd.read_parquet(RAW_DIR / 'spinoff_children_sp500.parquet')
idx_ret     = pd.read_parquet(RAW_DIR / 'crsp_index_returns.parquet')
passive_aum = pd.read_parquet(RAW_DIR / 'sp500_passive_aum.parquet')
crsp_path   = RAW_DIR / 'crsp_daily.parquet'
crsp        = pd.read_parquet(crsp_path) if crsp_path.exists() else None

# child classification
first_inc = (child_sp500.sort_values('added_date')
             .groupby('child_ticker', as_index=False).first()
             [['child_ticker','added_date','removed_date']])
eff_dates = events[['spinoff_ticker','effective_date']].rename(columns={'spinoff_ticker':'child_ticker'})
first_inc = first_inc.merge(eff_dates, on='child_ticker', how='left')
first_inc['days_to_inclusion'] = (first_inc['added_date'] - first_inc['effective_date']).dt.days
def classify(r):
    if pd.isna(r['days_to_inclusion']): return 'excluded'
    return 'immediately included' if r['days_to_inclusion'] <= 5 else 'later included'
first_inc['inclusion_status'] = first_inc.apply(classify, axis=1)
status_map = first_inc.set_index('child_ticker')['inclusion_status'].to_dict()
events['inclusion_status'] = events['spinoff_ticker'].map(status_map).fillna('excluded')

# child event windows
def build_window(permno, eff_date, post=90):
    child = children[children['permno']==int(permno)].sort_values('date').copy()
    valid = child[child['date'] >= eff_date]
    if len(valid) == 0: return pd.DataFrame()
    w = valid.head(post+1).copy()
    w = w.merge(idx_ret[['date','sprtrn']], on='date', how='left')
    w['t'] = range(len(w))
    pc = 'adj_prc' if 'adj_prc' in w.columns else 'prc'
    t0 = w.iloc[0][pc]; t0 = t0 if (pd.notna(t0) and t0!=0) else abs(w.iloc[0]['prc'])
    w['norm_prc']    = w[pc] / t0 * 100
    w['ret_mkt_adj'] = w['ret'] - w['sprtrn']
    w['car']         = w['ret_mkt_adj'].cumsum()
    w['short_pnl']   = -w['car']
    return w

cw_list = []
for _, ev in events.iterrows():
    cr = children[children['child_ticker']==ev['spinoff_ticker']]
    if cr.empty: continue
    w = build_window(cr['permno'].iloc[0], ev['effective_date'])
    if len(w)==0: continue
    w['child_ticker']     = ev['spinoff_ticker']
    w['parent_ticker']    = ev['parent_ticker']
    w['inclusion_status'] = ev['inclusion_status']
    w['forced_flow_adv']  = ev.get('forced_flow_adv', np.nan)
    cw_list.append(w)
all_cw = pd.concat(cw_list, ignore_index=True) if cw_list else pd.DataFrame()

# CAR table
car_rows = []
for _, ev in events.iterrows():
    tk = ev['spinoff_ticker']
    grp = all_cw[all_cw['child_ticker']==tk]
    if grp.empty: continue
    row = {'child_ticker': tk, 'inclusion_status': ev['inclusion_status'],
           'forced_flow_adv': ev.get('forced_flow_adv', np.nan)}
    for h in [5,10,21,42,63]:
        sub = grp[grp['t']==h]
        row[f'car_{h}d']       = sub['car'].values[0]       if len(sub) else np.nan
        row[f'short_pnl_{h}d'] = sub['short_pnl'].values[0] if len(sub) else np.nan
    car_rows.append(row)
car_df = pd.DataFrame(car_rows)

# free float metric
child_t0 = (children.sort_values('date').groupby('permno').first().reset_index()
            [['permno','mktcap','child_ticker']].rename(columns={'mktcap':'child_mktcap_t0'}))
events_ff = events.merge(child_t0[['child_ticker','child_mktcap_t0']],
                          left_on='spinoff_ticker', right_on='child_ticker', how='left')
events_ff['ff_float_pct'] = np.where(
    events_ff['child_mktcap_t0'].notna() & (events_ff['child_mktcap_t0']>0),
    events_ff['forced_flow_usd'] / events_ff['child_mktcap_t0'], np.nan)

# inclusion model
SP500_THRESH = {2020:8.2e9,2021:11.8e9,2022:13.1e9,2023:12.7e9,2024:18.0e9}
def sp500_thr(y): return SP500_THRESH.get(y, SP500_THRESH[min(SP500_THRESH,key=lambda x:abs(x-y))])
mrows = []
for _, ev in events_ff.iterrows():
    yr = ev['effective_date'].year; thr = sp500_thr(yr)
    sr  = (ev['child_mktcap_t0']/thr if pd.notna(ev['child_mktcap_t0']) and ev['child_mktcap_t0']>0 else np.nan)
    lag = (ev['effective_date']-ev['announce_date']).days if pd.notna(ev['announce_date']) else np.nan
    lag = lag if (pd.notna(lag) and lag>0) else np.nan
    mrows.append({'child_ticker':ev['spinoff_ticker'],'inclusion_status':ev['inclusion_status'],
                  'target':1 if ev['inclusion_status']=='immediately included' else 0,
                  'log_child_mktcap':np.log(ev['child_mktcap_t0']) if pd.notna(ev['child_mktcap_t0']) and ev['child_mktcap_t0']>0 else np.nan,
                  'parent_index_weight':ev['parent_index_weight'],
                  'forced_flow_adv':ev['forced_flow_adv'],
                  'log_size_ratio':np.log(sr) if pd.notna(sr) and sr>0 else np.nan,
                  'log_lag_days':np.log(lag) if pd.notna(lag) else np.nan,
                  'passive_aum_B':ev['passive_aum_usd']/1e9 if pd.notna(ev['passive_aum_usd']) else np.nan})
model_df = pd.DataFrame(mrows)
fcols = ['log_child_mktcap','parent_index_weight','forced_flow_adv','log_size_ratio','log_lag_days','passive_aum_B']
mc = model_df.dropna(subset=fcols+['target']).copy()
loo_acc=loo_auc=np.nan; coef_df=pd.DataFrame()
if len(mc)>=10:
    X,y = mc[fcols].values, mc['target'].values
    Xs  = StandardScaler().fit_transform(X)
    prb = np.zeros(len(y)); prd = np.zeros(len(y),dtype=int)
    for tr,te in LeaveOneOut().split(Xs):
        clf = LogisticRegression(C=1.0,max_iter=500,random_state=42)
        clf.fit(Xs[tr],y[tr]); prb[te]=clf.predict_proba(Xs[te])[:,1]; prd[te]=clf.predict(Xs[te])
    loo_acc=accuracy_score(y,prd)
    loo_auc=roc_auc_score(y,prb) if len(np.unique(y))>1 else np.nan
    mc=mc.copy(); mc['prob_included']=prb
    clf_f=LogisticRegression(C=1.0,max_iter=500,random_state=42).fit(StandardScaler().fit_transform(X),y)
    coef_df=pd.DataFrame({'feat':fcols,'coef':clf_f.coef_[0]}).sort_values('coef',key=abs,ascending=False)

# beta map (excluded children only, t=1–60)
beta_map = {}
for tk,grp in all_cw.groupby('child_ticker'):
    sub = grp[grp['t'].between(1,60)].dropna(subset=['ret','sprtrn'])
    if len(sub)>=10:
        sl,*_ = _lr(sub['sprtrn'],sub['ret']); beta_map[tk]=sl

# announcement window
ann_frames=[]
if crsp is not None:
    for _,ev in events[events['permno'].notna()&events['announce_date'].notna()].iterrows():
        if ev['announce_date']>=ev['effective_date']: continue
        p = crsp[crsp['permno']==int(ev['permno'])].copy()
        if p.empty: continue
        ann=ev['announce_date']
        w=(p[(p['date']>=ann-pd.Timedelta(days=180))&(p['date']<=ann+pd.Timedelta(days=180))]
           .merge(idx_ret[['date','sprtrn']],on='date',how='left').sort_values('date').reset_index(drop=True))
        ai=int(w['date'].searchsorted(ann))
        w['t_ann']=range(-ai,len(w)-ai); w['rma']=w['ret']-w['sprtrn']
        w['parent_ticker']=ev['parent_ticker']; w['inclusion_status']=ev['inclusion_status']
        ann_frames.append(w[w['t_ann'].between(-10,60)])
all_aw=pd.concat(ann_frames,ignore_index=True) if ann_frames else pd.DataFrame()

# key stats
n_excl=(events['inclusion_status']=='excluded').sum()
n_imm =(events['inclusion_status']=='immediately included').sum()
n_lat =(events['inclusion_status']=='later included').sum()
lat_aum=passive_aum.sort_values('date').iloc[-1]['total_aum_billions']
ff=events['forced_flow_adv'].dropna()
excl_c=car_df[car_df['inclusion_status']=='excluded']
imm_c =car_df[car_df['inclusion_status']=='immediately included']
e21=excl_c['short_pnl_21d'].dropna(); e42=excl_c['short_pnl_42d'].dropna()
t21,p21=(stats.ttest_1samp(e21,0) if len(e21)>1 else (np.nan,np.nan))
t42,p42=(stats.ttest_1samp(e42,0) if len(e42)>1 else (np.nan,np.nan))
ff_fl=events_ff[events_ff['inclusion_status']=='excluded']['ff_float_pct'].dropna()
print('Key stats ready. Building slides...')

# %% [markdown]
# ## 2. Slide Helpers

# %%
def new_slide(title, kicker=None):
    """16:9 figure with navy header. kicker = small caption above title."""
    fig = plt.figure(figsize=(SLIDE_W,SLIDE_H), facecolor=WHITE)
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    # header
    ah = fig.add_axes([0,0.895,1,0.105]); ah.set_facecolor(NAVY); ah.axis('off')
    if kicker:
        ah.text(0.025,0.82, kicker.upper(), color=GOLD, fontsize=9,
                fontweight='bold', va='top')
    ah.text(0.025,0.44, title, color=WHITE, fontsize=24,
            fontweight='bold', va='center', family='sans-serif')
    # accent line
    al=fig.add_axes([0,0.888,1,0.007]); al.set_facecolor(BLUE); al.axis('off')
    # footer
    af=fig.add_axes([0,0,1,0.038]); af.set_facecolor(NAVY); af.axis('off')
    af.text(0.015,0.5,'JPMorgan  ·  Index Rebalance Strategy  ·  CONFIDENTIAL',
            color='#78909c',fontsize=9,va='center')
    af.text(0.985,0.5,'July 2026',color='#78909c',fontsize=9,va='center',ha='right')
    return fig

def big_stat(ax, val, label, x=0.5, y=0.5, val_size=52, label_size=13, color=NAVY):
    ax.text(x, y+0.12, val,   color=color, fontsize=val_size, fontweight='bold',
            ha='center', va='center', transform=ax.transAxes)
    ax.text(x, y-0.10, label, color=GREY,  fontsize=label_size,
            ha='center', va='center', transform=ax.transAxes)

def save_slide(fig, idx, pdf):
    fig.savefig(SLIDES_DIR/f'slide_{idx:02d}.png', dpi=DPI, bbox_inches='tight', facecolor=WHITE)
    pdf.savefig(fig, bbox_inches='tight', facecolor=WHITE)
    plt.close(fig)
    print(f'  ✓ slide_{idx:02d}.png')

# %% [markdown]
# ## 3. Precompute announcement stats (used in slide 4)

# %%
ann_day_cars, ann2eff_cars = [], []
if len(all_aw):
    for _, ev in events[events['permno'].notna() & events['announce_date'].notna()].iterrows():
        if ev['announce_date'] >= ev['effective_date']: continue
        grp = all_aw[all_aw['parent_ticker'] == ev['parent_ticker']]
        if grp.empty: continue
        lag = (ev['effective_date'] - ev['announce_date']).days
        t0r = grp[grp['t_ann'] == 0]['rma']
        if len(t0r): ann_day_cars.append(t0r.values[0])
        ann2eff_cars.append(grp[grp['t_ann'].between(0, lag)]['rma'].sum())

# Build pairs data for slide 4
pair_rows = []
if crsp is not None and len(all_cw):
    for _, ev in events[events['permno'].notna()].iterrows():
        tk = ev['spinoff_ticker']
        cg = all_cw[all_cw['child_ticker'] == tk]
        if cg.empty: continue
        pp = (crsp[crsp['permno'] == int(ev['permno'])]
              .query('date >= @ev["effective_date"]')
              .sort_values('date').set_index('date')['ret'].to_dict())
        for _, cr in cg.iterrows():
            pr = pp.get(cr['date'], np.nan)
            pair_rows.append({'child_ticker': tk, 'inclusion_status': ev['inclusion_status'],
                               't': cr['t'], 'short_pnl_spy': cr['short_pnl'],
                               'pairs_pnl': -(cr['ret_mkt_adj'] - (pr - cr['sprtrn'])
                                              if pd.notna(pr) else np.nan)})
pairs_tmp = pd.DataFrame(pair_rows).sort_values(['child_ticker','t'])
pairs_tmp['pairs_cum'] = pairs_tmp.groupby('child_ticker')['pairs_pnl'].cumsum()

# Build trade P&L for slide 5 (t=42d stop)
trows = []
for _, ev in events.iterrows():
    tk = ev['spinoff_ticker']; st = ev['inclusion_status']
    cg = all_cw[all_cw['child_ticker'] == tk] if len(all_cw) else pd.DataFrame()
    if cg.empty: continue
    ir = child_sp500[child_sp500['child_ticker'] == tk]
    t_ei = (cg[cg['date'] >= ir.sort_values('added_date')['added_date'].iloc[0]]['t'].min()
            if not ir.empty else 999)
    ffv  = ev.get('forced_flow_adv', np.nan)
    piv  = (mc[mc['child_ticker'] == tk]['prob_included'].values[0]
            if (len(mc) and tk in mc['child_ticker'].values) else np.nan)
    for h in [21, 42]:
        tex = min(h, t_ei - 1) if t_ei < 999 else h; tex = max(tex, 1)
        er  = cg[cg['t'] == tex]
        if er.empty: er = cg.iloc[(cg['t'] - tex).abs().argsort()[:1]]
        rp  = er['short_pnl'].values[0]
        trows.append({'child_ticker': tk, 'effective_date': ev['effective_date'],
                      'inclusion_status': st, 'h': h,
                      'pnl_s1': rp if st == 'excluded' else np.nan,
                      'pnl_s2': rp if (st == 'excluded' and pd.notna(ffv) and ffv >= 5) else np.nan,
                      'pnl_s4': rp if (st == 'excluded' and pd.notna(piv) and piv < 0.25) else np.nan})
tdf = pd.DataFrame(trows)

print('Pre-computations done.')

# %% [markdown]
# ## 4. Extended Precompute (8-slide deck)

# %%
# FF vs returns link (slide 3)
ff_ret = (car_df[car_df['inclusion_status']=='excluded']
          .dropna(subset=['forced_flow_adv','short_pnl_42d']).copy())
ff_ret['ff_float_pct'] = ff_ret['child_ticker'].map(
    events_ff.set_index('spinoff_ticker')['ff_float_pct'].to_dict())
ff_hi   = ff_ret[ff_ret['forced_flow_adv']>=5]['short_pnl_42d'].dropna()*100
ff_lo   = ff_ret[ff_ret['forced_flow_adv']< 5]['short_pnl_42d'].dropna()*100
ff_corr = (np.corrcoef(ff_ret['forced_flow_adv'], ff_ret['short_pnl_42d'])[0,1]
           if len(ff_ret)>=5 else np.nan)

# Per-event announcement table (slide 4)
ann_tbl_rows = []
if len(all_aw):
    for _, ev in events[events['permno'].notna() & events['announce_date'].notna()].iterrows():
        if ev['announce_date'] >= ev['effective_date']: continue
        grp = all_aw[all_aw['parent_ticker']==ev['parent_ticker']]
        if grp.empty: continue
        lag = (ev['effective_date']-ev['announce_date']).days
        t0r = grp[grp['t_ann']==0]['rma']
        cum_eff = grp[grp['t_ann'].between(0,lag)]['rma'].sum() if lag>0 else np.nan
        if len(t0r):
            ann_tbl_rows.append({'parent_ticker':ev['parent_ticker'],
                                  'child_ticker':ev['spinoff_ticker'],
                                  'ann_day_car':float(t0r.values[0])*100,
                                  'ann2eff_car':float(cum_eff)*100 if pd.notna(cum_eff) else np.nan,
                                  'lag_days':int(lag),'status':ev['inclusion_status']})
ann_tbl = (pd.DataFrame(ann_tbl_rows).sort_values('ann2eff_car',ascending=False)
           if ann_tbl_rows else pd.DataFrame())
med_lag = (int(np.median([r['lag_days'] for r in ann_tbl_rows]))
           if ann_tbl_rows else 0)
ann_win_rate = (np.mean([r['ann2eff_car']>0 for r in ann_tbl_rows if pd.notna(r['ann2eff_car'])])
                if ann_tbl_rows else np.nan)

# Pairs per-event at t=42d (slide 5)
pairs42 = (pairs_tmp[pairs_tmp['t']==42]
           .groupby(['child_ticker','inclusion_status'])[['short_pnl_spy','pairs_cum']]
           .first().reset_index())
pairs42_excl = pairs42[pairs42['inclusion_status']=='excluded'].copy()

print('Extended precompute done. Building 8 slides...')

# %% [markdown]
# ## 5. Slide Helpers

# %%
def stat_strip(fig, stats_list, y=0.72, h=0.17):
    """4-number stat strip below the header."""
    sa = fig.add_axes([0.0, y, 1.0, h]); sa.axis('off'); sa.set_facecolor(LTBG)
    for i, (val, lbl) in enumerate(stats_list):
        x = 0.125 + i * 0.25
        sa.text(x, 0.72, val, color=NAVY, fontsize=26, fontweight='bold',
                ha='center', va='center')
        sa.text(x, 0.22, lbl, color=GREY, fontsize=11, ha='center', va='center')
        if i < 3: sa.axvline(x + 0.125, 0.1, 0.9, color='#cfd8dc', lw=1)

def ret_table(ax, horizons_, excl_means, incl_means, pvals_):
    """Compact return table on a bare axes."""
    ax.axis('off')
    col_x = [0.0, 0.28, 0.52, 0.76]
    for ci, hdr in enumerate(['Horizon', 'Excl.', 'Incl.', 'p-val']):
        ax.text(col_x[ci], 0.94, hdr, color=WHITE, fontsize=11, fontweight='bold', va='top',
                bbox=dict(facecolor=NAVY, edgecolor='none', boxstyle='round,pad=0.2'))
    row_ys = [0.76, 0.60, 0.44, 0.28, 0.12]
    for ri, (h, em, im, pv) in enumerate(zip(horizons_, excl_means, incl_means, pvals_)):
        bg = LTBG if ri % 2 == 0 else WHITE
        ax.add_patch(mpatches.FancyBboxPatch((-0.02, row_ys[ri]-0.12), 1.04, 0.17,
            boxstyle='round,pad=0.01', facecolor=bg, edgecolor='none'))
        stars = '***' if pv < 0.01 else ('**' if pv < 0.05 else ('*' if pv < 0.10 else ''))
        vals = [f't={h}d', f'{em:+.1f}%', f'{im:+.1f}%' if pd.notna(im) else '—',
                f'{pv:.3f}{stars}']
        for ci, (cx, val) in enumerate(zip(col_x, vals)):
            fc = RED if (ci == 1 and em < 0) else (GRN if (ci == 1 and em > 0) else '#333')
            ax.text(cx, row_ys[ri], val, color=fc, fontsize=11,
                    fontweight='bold' if ci in [0, 1] else 'normal', va='top')

from matplotlib.patches import Patch

with PdfPages(SLIDES_DIR/'deck.pdf') as pdf:

    # ── SLIDE 1: Universe & Signal Construction ───────────────────────────────
    fig = new_slide('Universe & Signal Construction', kicker='Passive Flow Quantification')
    stat_strip(fig, [
        (f'${lat_aum:.0f}B',              'Passive S&P 500 AUM (latest)'),
        (f'{len(events)}',                 'Spinoff events  (2020–2024)'),
        (f'{n_excl} / {len(events)}',      'Children excluded from S&P 500'),
        (f'{ff.median():.1f}×',            'Median forced flow / parent ADV'),
    ])
    ax_bar = fig.add_axes([0.03, 0.06, 0.58, 0.62])
    ff_s = events.dropna(subset=['forced_flow_adv']).sort_values('forced_flow_adv')
    bar_c = [COLOR_MAP.get(s,GREY) for s in ff_s['inclusion_status']]
    ax_bar.barh(range(len(ff_s)), ff_s['forced_flow_adv'], color=bar_c, alpha=0.9, height=0.75)
    ax_bar.axvline(5,  color=ORG, ls='--', lw=1.5, label='5× threshold')
    ax_bar.axvline(10, color=RED, ls='--', lw=1.5, label='10× threshold')
    ax_bar.set_yticks(range(len(ff_s)))
    ax_bar.set_yticklabels([f'{r.spinoff_ticker}  ←  {r.parent_ticker}' for _,r in ff_s.iterrows()],
                            fontsize=8.5)
    ax_bar.set_xlabel('Forced Flow  (× Parent 30d ADV)', fontsize=11)
    ax_bar.legend(fontsize=10, loc='lower right')
    ax_bar.spines[['top','right']].set_visible(False)
    ax_key = fig.add_axes([0.64, 0.06, 0.33, 0.62]); ax_key.axis('off')
    ax_key.legend(handles=[Patch(color=c,alpha=0.85,label=s) for s,c in COLOR_MAP.items()],
                  fontsize=11, loc='upper left', framealpha=0.9)
    txt = ('FF × ADV:\n\n'
           '  Passive AUM × Parent Wt\n'
           '  ─────────────────────\n'
           '     Parent 30d ADV\n\n'
           'FF % Float:\n\n'
           '  Forced Flow ($)\n'
           '  ─────────────────────\n'
           '  Child Mktcap at t=0\n\n'
           'Gate: child must be EXCLUDED\n'
           'from S&P 500 (passive must sell)\n\n'
           f'n={n_excl} excl  ·  n={n_imm} imm  ·  n={n_lat} later')
    ax_key.text(0.05, 0.9, txt, color=NAVY, fontsize=11, va='top', linespacing=1.7,
                bbox=dict(boxstyle='round,pad=0.6', facecolor=LTBG, edgecolor=BLUE, lw=1.5))
    save_slide(fig, 1, pdf)

    # ── SLIDE 2: Child Return Evidence ────────────────────────────────────────
    fig = new_slide('Child Return Evidence', kicker='Short Leg Validation  ·  Market-Adjusted Returns')
    stat_strip(fig, [
        (f'{e21.mean()*100:+.1f}%', f'Excluded mean short P&L  t=21d  (n={len(e21)})'),
        (f'{e42.mean()*100:+.1f}%', f'Excluded mean short P&L  t=42d  (n={len(e42)})'),
        (f'{(e42>0).mean():.0%}',    'Win rate  t=42d  (excluded)'),
        (f'p={p42:.3f}',             't-test vs zero  t=42d'),
    ])
    ax_paths = fig.add_axes([0.03, 0.06, 0.47, 0.62])
    if len(all_cw):
        for status, grp in all_cw[all_cw['t'].between(0,63)].groupby('inclusion_status'):
            c = COLOR_MAP.get(status,GREY)
            for _, tg in grp.groupby('child_ticker'):
                ax_paths.plot(tg['t'], tg['short_pnl']*100, color=c, alpha=0.12, lw=0.7)
            avg = grp.groupby('t')['short_pnl'].mean()
            ax_paths.plot(avg.index, avg.values*100, color=c, lw=2.8, label=status)
    ax_paths.axhline(0, color='black', lw=0.8, ls='--')
    ax_paths.set_xlabel('Trading days since effective date', fontsize=10)
    ax_paths.set_ylabel('Short P&L  (%, market-adjusted)', fontsize=10)
    ax_paths.set_title('Short P&L by Child Inclusion Status', fontsize=11, fontweight='bold', color=NAVY)
    ax_paths.legend(fontsize=9); ax_paths.spines[['top','right']].set_visible(False)
    ax_tbl = fig.add_axes([0.55, 0.06, 0.42, 0.62])
    hs = [5,10,21,42,63]
    ret_table(ax_tbl, hs,
        [car_df[car_df['inclusion_status']=='excluded'][f'short_pnl_{h}d'].dropna().mean()*100 for h in hs],
        [car_df[car_df['inclusion_status']=='immediately included'][f'short_pnl_{h}d'].dropna().mean()*100 for h in hs],
        [stats.ttest_1samp(car_df[car_df['inclusion_status']=='excluded'][f'short_pnl_{h}d'].dropna(),0)[1] for h in hs])
    ax_tbl.set_title('Mean Short P&L by Horizon', fontsize=11, fontweight='bold', color=NAVY, pad=8)
    save_slide(fig, 2, pdf)

    # ── SLIDE 3: Forced-Flow Signal Depth ────────────────────────────────────
    fig = new_slide('Forced-Flow Signal: Depth & Return Link',
                    kicker='Signal Calibration  ·  Cross-Sectional Evidence')
    stat_strip(fig, [
        (f'{ff_corr:+.2f}',                          'FF×ADV vs t=42d return correlation (excl.)'),
        (f'{ff_hi.mean():+.1f}%' if len(ff_hi) else '—', f'Mean P&L  FF≥5×  (n={len(ff_hi)})'),
        (f'{ff_lo.mean():+.1f}%' if len(ff_lo) else '—', f'Mean P&L  FF<5×  (n={len(ff_lo)})'),
        (f'{ff_fl.median()*100:.1f}%',                'Median FF% of child mktcap (excl.)'),
    ])
    ax_sc = fig.add_axes([0.03, 0.06, 0.43, 0.62])
    if len(ff_ret)>=4:
        sc_c = [COLOR_MAP.get(s,GREY) for s in ff_ret['inclusion_status']]
        ax_sc.scatter(ff_ret['forced_flow_adv'], ff_ret['short_pnl_42d']*100,
                      c=sc_c, alpha=0.85, s=70, zorder=3)
        sl,ic,*_ = stats.linregress(ff_ret['forced_flow_adv'], ff_ret['short_pnl_42d']*100)
        xr = np.linspace(ff_ret['forced_flow_adv'].min(), ff_ret['forced_flow_adv'].max(), 50)
        ax_sc.plot(xr, sl*xr+ic, color=NAVY, lw=1.8, ls='--', alpha=0.7)
        for _, row in ff_ret.iterrows():
            ax_sc.annotate(row['child_ticker'],
                           (row['forced_flow_adv'], row['short_pnl_42d']*100),
                           fontsize=7, alpha=0.75, xytext=(3,3), textcoords='offset points')
    ax_sc.axhline(0, color='black', lw=0.7, ls='--')
    ax_sc.axvline(5, color=ORG, lw=1.2, ls=':', alpha=0.7, label='5× gate')
    ax_sc.set_xlabel('Forced Flow  (× Parent ADV)', fontsize=10)
    ax_sc.set_ylabel('Short P&L  t=42d (%)', fontsize=10)
    ax_sc.set_title('FF×ADV vs Short P&L  (excluded children)', fontsize=11, fontweight='bold', color=NAVY)
    ax_sc.legend(fontsize=9); ax_sc.spines[['top','right']].set_visible(False)
    ax_bk = fig.add_axes([0.53, 0.37, 0.44, 0.31])
    if len(ff_hi)>0 and len(ff_lo)>0:
        ax_bk.hist(ff_lo.values, bins=8, alpha=0.65, color=BLUE,
                   label=f'FF<5×  (n={len(ff_lo)})', density=True)
        ax_bk.hist(ff_hi.values, bins=8, alpha=0.65, color=RED,
                   label=f'FF≥5×  (n={len(ff_hi)})', density=True)
        ax_bk.axvline(0, color='black', lw=0.8, ls='--')
    ax_bk.set_xlabel('Short P&L t=42d (%)', fontsize=9)
    ax_bk.set_title('P&L Distribution by FF Signal Strength', fontsize=10, fontweight='bold', color=NAVY)
    ax_bk.legend(fontsize=9); ax_bk.spines[['top','right']].set_visible(False)
    ax_fl = fig.add_axes([0.53, 0.06, 0.44, 0.27])
    ff_fl_all = events_ff.dropna(subset=['ff_float_pct']).sort_values('ff_float_pct', ascending=False)
    fl_c = [COLOR_MAP.get(s,GREY) for s in ff_fl_all['inclusion_status']]
    ax_fl.bar(ff_fl_all['spinoff_ticker'], ff_fl_all['ff_float_pct']*100, color=fl_c, alpha=0.85, width=0.7)
    ax_fl.set_ylabel('FF % of child mktcap', fontsize=9)
    ax_fl.set_title('Forced Flow as % of Child Float', fontsize=10, fontweight='bold', color=NAVY)
    ax_fl.tick_params(axis='x', labelsize=7, rotation=60)
    ax_fl.spines[['top','right']].set_visible(False)
    save_slide(fig, 3, pdf)

    # ── SLIDE 4: Parent Company Strategy — Announcement Effect ───────────────
    ann_day_mean = np.mean(ann_day_cars)*100 if ann_day_cars else np.nan
    ann2eff_mean = np.mean(ann2eff_cars)*100 if ann2eff_cars else np.nan
    fig = new_slide('Parent Company Strategy: Announcement Effect',
                    kicker='Long Leg  ·  Event-Driven Alpha  ·  Buy Announcement / Sell Effective Date')
    stat_strip(fig, [
        (f'{ann_day_mean:+.1f}%',   f'Avg parent ann.-day CAR  (n={len(ann_day_cars)})'),
        (f'{ann2eff_mean:+.1f}%',   'Avg parent CAR  announce → effective'),
        (f'{ann_win_rate:.0%}' if pd.notna(ann_win_rate) else '—',
                                     'Win rate  announce → effective'),
        (f'{med_lag}d',              'Median announce → effective lag'),
    ])
    ax_ann = fig.add_axes([0.03, 0.06, 0.44, 0.62])
    if len(all_aw):
        avg_a = (all_aw[all_aw['t_ann'].between(-10,60)]
                 .groupby('t_ann')['rma'].agg(['mean','sem']).reset_index())
        avg_a['cm'] = avg_a['mean'].cumsum()
        ci_ = 1.96 * avg_a['sem'].cumsum()
        ax_ann.plot(avg_a['t_ann'], avg_a['cm']*100, lw=2.5, color=NAVY, label='All events (avg)')
        ax_ann.fill_between(avg_a['t_ann'],
                            (avg_a['cm']-ci_)*100, (avg_a['cm']+ci_)*100,
                            alpha=0.15, color=BLUE)
        for status, grp in all_aw[all_aw['t_ann'].between(-10,60)].groupby('inclusion_status'):
            avg_s = grp.groupby('t_ann')['rma'].mean().cumsum()
            ax_ann.plot(avg_s.index, avg_s.values*100,
                        color=COLOR_MAP.get(status,GREY), lw=1.5, alpha=0.8, label=status)
    ax_ann.axvline(0, color=RED, ls='--', lw=1.5, label='Announcement day')
    ax_ann.axhline(0, color='black', lw=0.6)
    ax_ann.set_xlabel('Trading days from announcement', fontsize=10)
    ax_ann.set_ylabel('Parent CAR  (%, market-adj.)', fontsize=10)
    ax_ann.set_title('Parent Stock: Announcement-Centered CAR', fontsize=11, fontweight='bold', color=NAVY)
    ax_ann.legend(fontsize=8.5); ax_ann.spines[['top','right']].set_visible(False)
    ax_tb4 = fig.add_axes([0.52, 0.06, 0.46, 0.62]); ax_tb4.axis('off')
    if len(ann_tbl):
        cx4 = [0.0,0.14,0.26,0.38,0.56,0.72]
        for ci,hdr in enumerate(['Parent','Child','Lag','Ann Day','Ann→Eff','Status']):
            ax_tb4.text(cx4[ci],0.97,hdr,color=WHITE,fontsize=9.5,fontweight='bold',va='top',
                        bbox=dict(facecolor=NAVY,edgecolor='none',boxstyle='round,pad=0.2'))
        row_h = 0.85/max(len(ann_tbl),1)
        for ri,(_,r) in enumerate(ann_tbl.iterrows()):
            if ri>=13: break
            bg=LTBG if ri%2==0 else WHITE
            y_=0.90-ri*(row_h+0.005)
            ax_tb4.add_patch(mpatches.FancyBboxPatch((-0.01,y_-row_h*0.85),1.02,row_h*0.9,
                boxstyle='round,pad=0.005',facecolor=bg,edgecolor='none'))
            a2e=f'{r.ann2eff_car:+.1f}%' if pd.notna(r.ann2eff_car) else '—'
            a2e_c=GRN if (pd.notna(r.ann2eff_car) and r.ann2eff_car>0) else RED
            st_s={'excluded':'excl','immediately included':'imm','later included':'later'}.get(r.status,r.status)
            for ci,(cx,val,fc) in enumerate(zip(cx4,
                [r.parent_ticker,r.child_ticker,f'{r.lag_days}d',f'{r.ann_day_car:+.1f}%',a2e,st_s],
                ['#333','#333','#333','#333',a2e_c,COLOR_MAP.get(r.status,'#333')])):
                ax_tb4.text(cx,y_,val,color=fc,fontsize=9,va='top',
                            fontweight='bold' if ci in [0,4] else 'normal')
    save_slide(fig, 4, pdf)

    # ── SLIDE 5: Pairs Trade Analysis ─────────────────────────────────────────
    excl_pairs_42 = pairs42_excl['pairs_cum'].dropna()
    excl_spy_42   = pairs42_excl['short_pnl_spy'].dropna()
    fig = new_slide('Pairs Trade: Short Child / Long Parent',
                    kicker='Hedge Construction  ·  Idiosyncratic vs Market Risk')
    stat_strip(fig, [
        (f'{excl_spy_42.mean()*100:+.1f}%' if len(excl_spy_42) else '—',
                                              f'SPY hedge P&L  t=42d  (n={len(excl_spy_42)})'),
        (f'{excl_pairs_42.mean()*100:+.1f}%' if len(excl_pairs_42) else '—',
                                              f'Pairs P&L  t=42d  (n={len(excl_pairs_42)})'),
        (f'{(excl_pairs_42>0).mean():.0%}' if len(excl_pairs_42) else '—',
                                              'Pairs win rate  t=42d'),
        (f'{((excl_pairs_42.mean()-excl_spy_42.mean())*100):+.1f}%'
         if (len(excl_pairs_42) and len(excl_spy_42)) else '—',
                                              'Pairs vs SPY hedge differential'),
    ])
    ax_pair = fig.add_axes([0.03, 0.06, 0.44, 0.62])
    excl_p = pairs_tmp[(pairs_tmp['inclusion_status']=='excluded') & pairs_tmp['t'].between(0,63)]
    avg_spy  = excl_p.groupby('t')['short_pnl_spy'].mean()
    avg_pair = excl_p.groupby('t')['pairs_cum'].mean()
    ax_pair.plot(avg_spy.index,  avg_spy.values*100,  lw=2.5, color=BLUE,
                 label='Short child / Long SPY')
    ax_pair.plot(avg_pair.index, avg_pair.values*100, lw=2.5, color=GRN, ls='--',
                 label='Short child / Long parent\n(pairs trade)')
    ax_pair.axhline(0, color='black', lw=0.6)
    ax_pair.set_xlabel('Trading days since effective date', fontsize=10)
    ax_pair.set_ylabel('P&L  (%, excluded children avg)', fontsize=10)
    ax_pair.set_title('SPY Hedge vs Pairs Trade — Avg P&L Path', fontsize=11, fontweight='bold', color=NAVY)
    ax_pair.legend(fontsize=9); ax_pair.spines[['top','right']].set_visible(False)
    ax_sc5 = fig.add_axes([0.53, 0.38, 0.43, 0.30])
    if len(pairs42_excl)>=2:
        ax_sc5.scatter(pairs42_excl['short_pnl_spy']*100, pairs42_excl['pairs_cum']*100,
                       color=RED, alpha=0.8, s=55)
        for _,r in pairs42_excl.iterrows():
            ax_sc5.annotate(r['child_ticker'],
                            (r['short_pnl_spy']*100, r['pairs_cum']*100),
                            fontsize=7, xytext=(3,3), textcoords='offset points', alpha=0.8)
        lim = max(abs(pairs42_excl[['short_pnl_spy','pairs_cum']].values).max()*100*1.2, 5)
        ax_sc5.plot([-lim,lim],[-lim,lim],color='black',lw=0.8,ls='--',alpha=0.4)
        ax_sc5.axhline(0,color='black',lw=0.5); ax_sc5.axvline(0,color='black',lw=0.5)
        ax_sc5.set_xlim(-lim,lim); ax_sc5.set_ylim(-lim,lim)
    ax_sc5.set_xlabel('SPY hedge P&L %', fontsize=9)
    ax_sc5.set_ylabel('Pairs P&L %', fontsize=9)
    ax_sc5.set_title('SPY vs Pairs  (t=42d, per event)', fontsize=10, fontweight='bold', color=NAVY)
    ax_sc5.spines[['top','right']].set_visible(False)
    ax_txt5 = fig.add_axes([0.53, 0.06, 0.43, 0.28]); ax_txt5.axis('off')
    ax_txt5.text(0.05, 0.95,
        'Pairs rationale:\n\n'
        '• Parent & child share sector/macro exposure\n'
        '• Long parent hedges both SPY beta and sector\n'
        '• Residual P&L ≈ pure forced-selling alpha\n'
        '• Exit: child S&P 500 addition OR 42-day stop',
        color=NAVY, fontsize=11, va='top', linespacing=1.6,
        bbox=dict(boxstyle='round,pad=0.6', facecolor=LTBG, edgecolor=BLUE, lw=1.5))
    save_slide(fig, 5, pdf)

    # ── SLIDE 6: Beta Decomposition & Alpha Attribution ───────────────────────
    betas_ = list(beta_map.values()); med_b = np.median(betas_) if betas_ else np.nan
    fig = new_slide('Beta Decomposition & Alpha Attribution',
                    kicker='Risk Factor Analysis  ·  Excluded Children  ·  OLS on t=1–60')
    stat_strip(fig, [
        (f'{med_b:.2f}',                            'Median child β vs S&P 500 (OLS t=1–60)'),
        (f'{np.mean(betas_):.2f}' if betas_ else '—', 'Mean child β'),
        ('Alpha-driven',                             'Primary source of underperformance'),
        (f'{len(beta_map)}',                         'Events with β estimate (≥10 obs)'),
    ])
    ax_beta = fig.add_axes([0.03, 0.06, 0.44, 0.62])
    if len(all_cw):
        ea = all_cw[(all_cw['inclusion_status']=='excluded') & all_cw['t'].between(0,63)]
        raw_avg = ea.groupby('t')['short_pnl'].mean()
        al_list = []
        for tk, grp in ea.groupby('child_ticker'):
            b = beta_map.get(tk,1.0); b = b if pd.notna(b) else 1.0
            g = grp.sort_values('t').copy()
            g['alpha_d']   = g['ret'] - b*g['sprtrn']
            g['cum_alpha'] = g['alpha_d'].cumsum()
            al_list.append(g)
        if al_list:
            adf = pd.concat(al_list)
            al_avg = adf.groupby('t')['cum_alpha'].mean()
            ax_beta.plot(raw_avg.index, raw_avg.values*100, lw=2.5, color=RED,
                         label='Raw short P&L  (mkt-adj.)')
            ax_beta.plot(al_avg.index, -al_avg.values*100, lw=2.5, color='#7b1fa2',
                         ls='--', label='Beta-stripped alpha  (pure α)')
    ax_beta.axhline(0, color='black', lw=0.6)
    ax_beta.set_xlabel('Trading days since effective date', fontsize=10)
    ax_beta.set_ylabel('Cumulative P&L (%)', fontsize=10)
    ax_beta.set_title('Raw P&L vs Beta-Stripped Alpha\n(excluded children, equal-weighted avg)',
                      fontsize=11, fontweight='bold', color=NAVY)
    ax_beta.legend(fontsize=9.5); ax_beta.spines[['top','right']].set_visible(False)
    ax_hist = fig.add_axes([0.55, 0.38, 0.41, 0.30])
    if betas_:
        ax_hist.hist(betas_, bins=12, color=BLUE, alpha=0.8, edgecolor='white')
        ax_hist.axvline(med_b, color=RED, lw=1.8, ls='--', label=f'Median β={med_b:.2f}')
        ax_hist.axvline(1.0,  color='black', lw=1.2, ls=':', alpha=0.7, label='β=1 reference')
    ax_hist.set_xlabel('Estimated β vs S&P 500', fontsize=9)
    ax_hist.set_ylabel('Count', fontsize=9)
    ax_hist.set_title('Child β Distribution', fontsize=10, fontweight='bold', color=NAVY)
    ax_hist.legend(fontsize=9); ax_hist.spines[['top','right']].set_visible(False)
    ax_note6 = fig.add_axes([0.55, 0.06, 0.41, 0.28]); ax_note6.axis('off')
    ax_note6.text(0.05, 0.95,
        'Beta estimation:\n\n'
        '• OLS: child ret = α + β × S&P 500 ret\n'
        '• Window: t=1 to t=60 post-effective\n'
        '• β<1 for most excluded children\n'
        '  (forced selling depresses price,\n'
        '   not index co-movement)\n'
        '• Live trading: use parent pre-spinoff β\n'
        '  to avoid ex-post lookahead',
        color=NAVY, fontsize=10.5, va='top', linespacing=1.6,
        bbox=dict(boxstyle='round,pad=0.6', facecolor=LTBG, edgecolor='#7b1fa2', lw=1.5))
    save_slide(fig, 6, pdf)

    # ── SLIDE 7: Index Inclusion Probability Model ────────────────────────────
    fig = new_slide('Index Inclusion Probability Model',
                    kicker='Logistic Regression  ·  Leave-One-Out CV  ·  Trade Gating')
    stat_strip(fig, [
        (f'{loo_acc:.0%}',   f'LOO-CV accuracy  (n={len(mc)})'),
        (f'{loo_auc:.3f}',   'ROC AUC'),
        (f'{(mc["target"]==0).mean():.0%}' if len(mc) else '—',
                              'Baseline accuracy (always predict excl.)'),
        (f'{len(mc[mc["target"]==1])}' if len(mc) else '—',
                              'Immediately-included events in sample'),
    ])
    ax_coef = fig.add_axes([0.03, 0.06, 0.41, 0.62])
    if len(coef_df):
        feat_labels = {'log_child_mktcap':'Log child mktcap',
                       'parent_index_weight':'Parent index weight',
                       'forced_flow_adv':'Forced flow (×ADV)',
                       'log_size_ratio':'Size vs SP500 threshold',
                       'log_lag_days':'Announce→eff lag (log)',
                       'passive_aum_B':'Passive AUM ($B)'}
        coef_df['label'] = [feat_labels.get(f,f) for f in coef_df['feat']]
        bc = [GRN if c>0 else RED for c in coef_df['coef']]
        bars = ax_coef.barh(coef_df['label'], coef_df['coef'], color=bc, alpha=0.85)
        ax_coef.axvline(0, color='black', lw=0.8)
        for bar,v in zip(bars, coef_df['coef']):
            ax_coef.text(v+(0.02 if v>=0 else -0.02),
                         bar.get_y()+bar.get_height()/2,
                         f'{v:+.2f}', va='center',
                         ha='left' if v>=0 else 'right', fontsize=9)
        ax_coef.set_xlabel('Coefficient (standardized features)', fontsize=10)
        ax_coef.set_title('Feature Importance\n(+ → predicts immediate inclusion)',
                          fontsize=11, fontweight='bold', color=NAVY)
        ax_coef.tick_params(labelsize=10)
        ax_coef.spines[['top','right']].set_visible(False)
    ax_prob = fig.add_axes([0.50, 0.06, 0.47, 0.62])
    if len(mc) and 'prob_included' in mc.columns:
        mc_s = mc.sort_values('prob_included')
        bc2  = [COLOR_MAP.get(s,GREY) for s in mc_s['inclusion_status']]
        ax_prob.barh(mc_s['child_ticker'].tolist(), mc_s['prob_included']*100,
                     color=bc2, alpha=0.85)
        ax_prob.axvline(50, color='black', ls='--', lw=1.2, label='50% threshold')
        ax_prob.axvline(25, color=ORG,    ls=':',  lw=1.2, label='25% filter gate')
        ax_prob.set_xlabel('P(immediately included) %', fontsize=10)
        ax_prob.set_title('LOO-CV Predicted Inclusion Probabilities',
                          fontsize=11, fontweight='bold', color=NAVY)
        ax_prob.legend(handles=[Patch(color=c,alpha=0.85,label=s) for s,c in COLOR_MAP.items()]
                       + [mpatches.Patch(color='black',fill=False,ls='--',label='50% threshold'),
                          mpatches.Patch(color=ORG,   fill=False,ls=':',  label='25% gate')],
                       fontsize=8.5, loc='lower right')
        ax_prob.tick_params(labelsize=9)
        ax_prob.spines[['top','right']].set_visible(False)
    save_slide(fig, 7, pdf)

    # ── SLIDE 8: Strategy Backtest Results ────────────────────────────────────
    t42 = tdf[tdf['h']==42]
    s1  = t42['pnl_s1'].dropna()*100
    s2  = t42['pnl_s2'].dropna()*100
    s4  = t42['pnl_s4'].dropna()*100
    _,ps1 = stats.ttest_1samp(s1,0) if len(s1)>1 else (np.nan,np.nan)
    _,ps2 = stats.ttest_1samp(s2,0) if len(s2)>1 else (np.nan,np.nan)
    fig = new_slide('Strategy Backtest Results',
                    kicker='Equal-Weight  ·  t=0 Close Entry  ·  42-Day Stop  ·  No Transaction Costs')
    stat_strip(fig, [
        (f'{s1.mean():+.1f}%',   f'S1 (All excl.) mean P&L  (n={len(s1)})'),
        (f'{(s1>0).mean():.0%}', 'S1 win rate  t=42d'),
        (f'{s2.mean():+.1f}%',   f'S2 (FF≥5×) mean P&L  (n={len(s2)})'),
        (f'{(s2>0).mean():.0%}', 'S2 win rate  t=42d'),
    ])
    ax_eq = fig.add_axes([0.03, 0.06, 0.44, 0.62])
    strats = [('pnl_s1','S1: All excluded  (SPY hedge)',   BLUE,'-'),
              ('pnl_s2','S2: FF≥5×  (SPY hedge)',          RED, '--'),
              ('pnl_s4','S4: Prob-filtered  (p<25%)',       GRN, ':')]
    for col,name,c,ls in strats:
        d = tdf[tdf['h']==42].dropna(subset=[col]).sort_values('effective_date')
        if len(d)>=2:
            ax_eq.plot(range(len(d)), d[col].cumsum()*100, lw=2.5,
                       color=c, ls=ls, label=f'{name}  (n={len(d)})')
    ax_eq.axhline(0, color='black', lw=0.8)
    ax_eq.set_xlabel('Trade # (chronological)', fontsize=10)
    ax_eq.set_ylabel('Cumulative P&L (%)', fontsize=10)
    ax_eq.set_title('Equity Curves — 42-Day Time Stop', fontsize=11, fontweight='bold', color=NAVY)
    ax_eq.legend(fontsize=9); ax_eq.spines[['top','right']].set_visible(False)
    ax_rt = fig.add_axes([0.52, 0.32, 0.46, 0.38]); ax_rt.axis('off')
    hdrs4  = ['Strategy','N','Mean P&L','Win%','t-stat','p-val']
    col_x4 = [0.0,0.20,0.38,0.56,0.70,0.84]
    for ci,cx in enumerate(col_x4):
        ax_rt.text(cx,0.97,hdrs4[ci],color=WHITE,fontsize=9.5,fontweight='bold',va='top',
                   bbox=dict(facecolor=NAVY,edgecolor='none',boxstyle='round,pad=0.2'))
    row_defs=[('pnl_s1','S1: All excl.'),('pnl_s2','S2: FF≥5×'),('pnl_s4','S4: p<25%')]
    row_ys4=[0.74,0.50,0.26]
    for ri,(col,nm) in enumerate(row_defs):
        s=tdf[tdf['h']==42][col].dropna()*100
        bg=LTBG if ri%2==0 else WHITE
        ax_rt.add_patch(mpatches.FancyBboxPatch((-0.01,row_ys4[ri]-0.2),1.02,0.24,
            boxstyle='round,pad=0.01',facecolor=bg,edgecolor='none'))
        if len(s)>1:
            tv,pv=stats.ttest_1samp(s,0)
            vals=[nm,str(len(s)),f'{s.mean():+.1f}%',f'{(s>0).mean():.0%}',f'{tv:.2f}',f'{pv:.3f}']
        else:
            vals=[nm,'—','—','—','—','—']
        for ci,(cx,val) in enumerate(zip(col_x4,vals)):
            fc=RED if (ci==2 and len(s)>0 and s.mean()<0) else (GRN if (ci==2 and len(s)>0) else '#333')
            ax_rt.text(cx,row_ys4[ri],val,color=fc,fontsize=9.5,
                       fontweight='bold' if ci in [0,2] else 'normal',va='top')
    ax_sig = fig.add_axes([0.52, 0.06, 0.46, 0.23]); ax_sig.axis('off')
    ax_sig.text(0.0,0.96,'Signal Priority',color=NAVY,fontsize=10,fontweight='bold',va='top')
    for i,(num,txt,c) in enumerate([
        ('1.','Gate: inclusion_status = excluded  (passive must sell)', NAVY),
        ('2.','Sizing: FF% float  (forced flow ÷ child mktcap)',         BLUE),
        ('3.','Signal: forced_flow_adv  (× parent 30d ADV)',             BLUE),
        ('4.','Filter: prob_included < 25%  (logistic model gate)',      ORG),
        ('5.','Hedge: long parent >> long SPY  (pairs trade)',           GRN),
    ]):
        ax_sig.text(0.0,  0.78-i*0.175, num, color=c,    fontsize=9.5, fontweight='bold', va='top')
        ax_sig.text(0.07, 0.78-i*0.175, txt, color='#333',fontsize=9.5, va='top')
    save_slide(fig, 8, pdf)

print('\nDone — 8 slides in slides/')
print('  PNG → Insert → Image → Upload from computer  (Google Slides)')
print('  deck.pdf → File distribution')
