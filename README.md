# Spinoff Index Arbitrage

Systematic, event-driven strategy research around S&P 500 index mechanics for
corporate spinoffs: forced passive-fund flows at spinoff completion, and the
parent's return in the year that follows.

## Thesis

When a company spins off a subsidiary, S&P 500 passive funds face a
**mechanical, non-discretionary flow**:

* **Excluded child** (not added to the S&P 500): passive funds receive child
  shares but *must sell* them → forced selling pressure → short the child.
* **Included child** (immediately added to the S&P 500): passive funds keep
  the shares they receive → no forced selling, skip the child leg.
* **Parent, post-completion**: separately, parent stocks show a strong
  positive market-adjusted return in the 12 months after a spinoff completes
  — the long leg the project ultimately builds on.

The **final strategy** (`notebooks/final_strategy.ipynb`) is a long-only book
of parent stocks after spinoff completion, sized by each spinoff child's
probability of immediate S&P 500 inclusion (`prob_included`), which is
predictive of the parent's 252-day return (Pearson r ≈ 0.67, p ≈ 0.004,
n = 16).

## Repository Structure

```
.
├── data/
│   ├── raw/                     # WRDS/CRSP/Compustat pulls (parquet/csv)
│   └── clean/                   # Cleaned + merged event-level datasets
├── pipeline/                    # Data acquisition & cleaning scripts (run in order, from repo root)
│   ├── repull_data.py           # 1. Full WRDS pull: S&P 500 history, CRSP prices, ETF AUM
│   ├── pull_etf_aum.py          # 1b. Passive S&P 500 fund AUM (WRDS Mutual Fund DB)
│   ├── clean_data.py            # 2. Clean raw Bloomberg spinoff event export
│   ├── merge_data.py            # 3. Merge events with CRSP/Compustat, compute forced-flow feature
│   └── pull_strategy_data.py    # 4. Pull spinoff-child prices/membership + parent fundamentals
├── notebooks/                   # Analysis notebooks, in the order the research progressed
│   ├── eda.ipynb                # Exploratory analysis of the spinoff child (short leg)
│   ├── strategy.ipynb           # Original strategy: short child (forced flow) + short parent on deletion
│   ├── spinoff_announce_effective_runup.ipynb      # Parent return, announce → effective date
│   ├── spinoff_post_completion_strategy.ipynb      # Parent long post-completion + risk overlays
│   ├── inclusion_probability_model.ipynb           # Model: P(child added to S&P 500 immediately)
│   ├── inclusion_prob_sizing_classification.ipynb  # Exploratory: sizing + classification on prob_included
│   └── final_strategy.ipynb     # ★ Final, decision-ready strategy (parent long, sized by prob_included)
├── results/                     # Figures exported from final_strategy.ipynb
├── slides/                      # Presentation decks (PDF)
├── environment.yaml             # Conda environment spec
├── pyproject.toml               # ruff / pyright config
└── .env                         # WRDS_USERNAME / WRDS_PASSWORD (gitignored, not committed)
```

Notebooks with a matching `.py` file in the same folder (`eda`, `strategy`,
`inclusion_probability_model`, `inclusion_prob_sizing_classification`) are
paired via [jupytext](https://jupytext.readthedocs.io/) percent format — the
`.py` file is a plain-text, diff-friendly mirror of the notebook's code and
markdown cells, not a separate script.

## How the Project Fits Together

1. **`pipeline/`** pulls and cleans the raw data (requires WRDS access — see
   below). Outputs land in `data/raw/` and `data/clean/`.
2. **`notebooks/eda.ipynb`** and **`notebooks/strategy.ipynb`** explore the
   original short-the-child, forced-flow thesis.
3. **`notebooks/spinoff_announce_effective_runup.ipynb`** and
   **`notebooks/spinoff_post_completion_strategy.ipynb`** find and stress-test
   a stronger, separate result: parent stocks rally after spinoff completion.
4. **`notebooks/inclusion_probability_model.ipynb`** builds `prob_included` —
   a leave-one-out cross-validated estimate of whether the spinoff child gets
   added to the S&P 500 immediately — and saves it to
   `data/clean/inclusion_probability_results.csv`.
5. **`notebooks/inclusion_prob_sizing_classification.ipynb`** joins
   `prob_included` against the parent's post-completion return and tests two
   ways to use it (position sizing, and a decision-tree classifier).
6. **`notebooks/final_strategy.ipynb`** is the trimmed, decision-ready result:
   only the position-sizing approach survives review intact (the classifier
   had leakage and a structural blind spot in the exploratory notebook); a
   corrected, leakage-free version of the tree is added back as a small
   overlay. This is the strategy presented in `slides/`.

## Environment

This repo uses **Python == 3.12.13**.

### 1. Virtual Environment

```bash
conda env create -f environment.yaml
conda activate index_spinoff_arbitrage_env
```

### 2. WRDS Credentials

Data-pulling scripts in `pipeline/` need WRDS access. Create a `.env` file at
the repo root (gitignored):

```
WRDS_USERNAME=your_wrds_username
WRDS_PASSWORD=your_wrds_password
```

### 3. Running the Pipeline

All commands below are run from the repo root:

```bash
python pipeline/repull_data.py
python pipeline/pull_etf_aum.py
python pipeline/clean_data.py
python pipeline/merge_data.py
python pipeline/pull_strategy_data.py
```

Large raw pulls (`crsp_daily.parquet`, `ccm_link.parquet`,
`crsp_index_returns.parquet`, `sp500_passive_funds.parquet`) are gitignored
and regenerated by `repull_data.py`; everything else in `data/` is checked
in so the notebooks run without WRDS access.

### 4. Running the Notebooks

Open any notebook in `notebooks/` with Jupyter — each resolves `data/` via a
path relative to the repo root (`../data/raw`, `../data/clean`), so no extra
configuration is needed as long as the notebook is launched from inside
`notebooks/`.
