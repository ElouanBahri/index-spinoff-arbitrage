"""
Pull all remaining data needed to build the spinoff index-arb trading strategy.

What this adds on top of repull_data.py:
  1. Spinoff child CRSP daily prices/volume  — the actual short leg
  2. Spinoff child S&P 500 membership        — exit signal (forced selling ends
                                               when child is added to index)
  3. Compustat annual fundamentals (parents) — features for deletion prob model

Spinoff children not yet in CRSP (future/very recent events, excluded):
  OXY/WS  — OXY warrants, not a standard equity spinoff
  MRP     — Millrose Properties (LEN/B, Jan 2025), too new
  SNDK    — New SanDisk (WDC, Feb 2025), too new
  RAL     — Ralliant Corp (FTV, Jun 2025), too new
  SOLS    — Solaris (HON, Oct 2025), future event

Run:
  python pull_strategy_data.py

Outputs (data/raw/):
  spinoff_children_crsp.parquet     — daily OHLCV for all spinoff children
  spinoff_children_sp500.parquet    — S&P 500 membership intervals for children
  parent_fundamentals.parquet       — Compustat annual fundamentals for parents
"""

import os
import builtins
import getpass
from pathlib import Path

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

WRDS_USERNAME = os.getenv("WRDS_USERNAME", "")
WRDS_PASSWORD = os.getenv("WRDS_PASSWORD", "")

_real_input   = builtins.input
_real_getpass = getpass.getpass

def _auto_input(prompt=""):
    if "username" in prompt.lower() and WRDS_USERNAME:
        print(f"{prompt}{WRDS_USERNAME}")
        return WRDS_USERNAME
    return _real_input(prompt)

def _auto_getpass(prompt="Password: ", stream=None):
    return WRDS_PASSWORD if WRDS_PASSWORD else _real_getpass(prompt, stream=stream)

builtins.input    = _auto_input
getpass.getpass   = _auto_getpass

import wrds
import pandas as pd

RAW_DIR   = Path("data/raw")
CLEAN_DIR = Path("data/clean")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Spinoff child permno → (ticker, parent_ticker, effective_date) mapping
# OXY/WS excluded (warrants). 2025 events excluded (not yet in CRSP).
# ---------------------------------------------------------------------------
CHILD_MAP = {
    18771: ("CHNG",  "MCK",   "2020-03-10"),  # Change Healthcare
    19283: ("ARNC",  "HWM",   "2020-04-01"),  # Arconic Corp
    19285: ("CARR",  "RTX",   "2020-04-03"),  # Carrier Global
    19286: ("OTIS",  "RTX",   "2020-04-03"),  # Otis Worldwide
    17783: ("CHX",   "ECL",   "2020-06-04"),  # ChampionX (was Apergy/APY)
    19807: ("VNT",   "FTV",   "2020-10-09"),  # Vontier Corp
    20057: ("VTRS",  "PFE",   "2020-11-17"),  # Viatris Inc
    21124: ("OGN",   "MRK",   "2021-06-03"),  # Organon & Co
    21356: ("DTM",   "DTE",   "2021-07-01"),  # DT Midstream
    21903: ("SLVM",  "IP",    "2021-10-01"),  # Sylvamo Corp
    92257: ("VMW",   "DELL",  "2021-11-02"),  # VMware Inc
    22092: ("KD",    "IBM",   "2021-11-04"),  # Kyndryl Holdings
    22264: ("ONL",   "O",     "2021-11-15"),  # Orion Office REIT
    22623: ("CEG",   "EXC",   "2022-02-02"),  # Constellation Energy
    22757: ("ZIMV",  "ZBH",   "2022-03-01"),  # Zimvie Inc
    22879: ("EMBC",  "BDX",   "2022-04-01"),  # Embecta Corp
    22976: ("WBD",   "T",     "2022-04-11"),  # Warner Bros Discovery
    23570: ("GEHC",  "GE",    "2023-01-04"),  # GE HealthCare
    21124: ("OGN",   "MRK",   "2021-06-03"),  # (dedup)
    23877: ("ATMU",  "CMI",   "2024-03-08"),  # Atmus Filtration
    23942: ("FTRE",  "LH",    "2023-07-03"),  # Fortrea Holdings
    24174: ("VLTO",  "DHR",   "2023-10-02"),  # Veralto Corp
    24877: ("SOLV",  "MMM",   "2024-04-01"),  # Solventum Corp
    24878: ("GEV",   "GE",    "2024-04-02"),  # GE Vernova
    25434: ("AMTM",  "J",     "2024-09-30"),  # Amentum Holdings
}

CHILD_PERMNOS = sorted(set(CHILD_MAP.keys()))


def connect() -> wrds.Connection:
    print(f"Connecting to WRDS as '{WRDS_USERNAME}'...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME, wrds_password=WRDS_PASSWORD)
    print("  Connected.")
    return db


# ---------------------------------------------------------------------------
# 1. Spinoff child CRSP daily prices
# ---------------------------------------------------------------------------

def pull_child_crsp(db) -> pd.DataFrame:
    """
    Pull daily CRSP prices for spinoff children from 90 days before the
    earliest effective date through present.
    """
    permno_str = ",".join(str(p) for p in CHILD_PERMNOS)
    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    df = db.raw_sql(f"""
        SELECT permno, date, prc, ret, retx, vol, shrout, cfacpr, cfacshr,
               bid, ask
        FROM crsp.dsf
        WHERE permno IN ({permno_str})
          AND date >= '2019-01-01'
          AND date <= '{today}'
        ORDER BY permno, date
    """, date_cols=["date"])

    df["adj_prc"]    = df["prc"].abs() / df["cfacpr"].replace(0, pd.NA)
    df["mktcap"]     = df["prc"].abs() * df["shrout"] * 1_000
    df["dollar_vol"] = df["vol"] * df["prc"].abs()
    df["spread_pct"] = ((df["ask"] - df["bid"]) / df["prc"].abs()).clip(0, 0.5)

    df = df.sort_values(["permno", "date"])
    df["adv_30d"] = (
        df.groupby("permno")["dollar_vol"]
        .transform(lambda x: x.rolling(30, min_periods=5).mean())
    )

    # Attach ticker and parent labels
    child_meta = pd.DataFrame(
        [(p, t, par, eff) for p, (t, par, eff) in CHILD_MAP.items()],
        columns=["permno", "child_ticker", "parent_ticker", "effective_date"]
    ).drop_duplicates("permno")
    child_meta["effective_date"] = pd.to_datetime(child_meta["effective_date"])
    df = df.merge(child_meta, on="permno", how="left")

    print(f"[child_crsp] {len(df):,} rows, {df['permno'].nunique()} children")
    for _, row in child_meta.iterrows():
        subset = df[df["permno"] == row["permno"]]
        if len(subset):
            print(f"  {row['child_ticker']:5s} ({row['parent_ticker']:5s}) "
                  f"{subset['date'].min().date()} → {subset['date'].max().date()}"
                  f"  {len(subset)} rows")
        else:
            print(f"  {row['child_ticker']:5s} ({row['parent_ticker']:5s})  NO DATA")
    return df


# ---------------------------------------------------------------------------
# 2. Spinoff child S&P 500 membership (exit signal)
# ---------------------------------------------------------------------------

def pull_child_sp500_membership(db) -> pd.DataFrame:
    """
    Check which spinoff children ever appeared in the S&P 500.
    When a child is ADDED to the index, passive buying begins — forced selling stops.
    This is the primary exit signal for the short.
    """
    permno_str = ",".join(str(p) for p in CHILD_PERMNOS)
    df = db.raw_sql(f"""
        SELECT permno, start AS added_date, ending AS removed_date
        FROM crsp.msp500list
        WHERE permno IN ({permno_str})
        ORDER BY permno, start
    """, date_cols=["added_date", "removed_date"])

    child_meta = pd.DataFrame(
        [(p, t, par) for p, (t, par, _) in CHILD_MAP.items()],
        columns=["permno", "child_ticker", "parent_ticker"]
    ).drop_duplicates("permno")
    df = df.merge(child_meta, on="permno", how="left")

    print(f"[child_sp500] {len(df)} membership records for "
          f"{df['permno'].nunique()} children")
    print(df[["child_ticker", "parent_ticker", "added_date",
               "removed_date"]].to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# 3. Compustat annual fundamentals for parents (deletion prob model)
# ---------------------------------------------------------------------------

def pull_parent_fundamentals(db) -> pd.DataFrame:
    """
    Annual Compustat fundamentals for parent companies.

    Features for the deletion probability model:
      mkvalt  — market cap at fiscal year-end (S&P 500 uses float-adj mktcap for eligibility)
      revt    — revenue (market breadth proxy)
      at      — total assets
      lt      — long-term debt
      csho    — shares outstanding
      prcc_f  — fiscal year-end price
      roa     — return on assets (profitability signal)
      sale    — sales / revenue (alternative to revt)
      sic     — sector classification
    """
    events = pd.read_csv(CLEAN_DIR / "spinoff_events_merged.csv",
                         parse_dates=["effective_date"])
    ccm    = pd.read_parquet(RAW_DIR / "ccm_link.parquet")
    ccm    = (ccm.sort_values("linkdt")
                 .drop_duplicates("permno", keep="last")
                 [["permno", "gvkey"]])

    # Map parent permnos to gvkeys
    parent_permnos = pd.to_numeric(events["permno"], errors="coerce").dropna().astype(int).unique()
    perm_df = pd.DataFrame({"permno": parent_permnos})
    perm_gvkey = perm_df.merge(ccm, on="permno", how="left")
    gvkeys = perm_gvkey["gvkey"].dropna().unique().tolist()
    gvkey_str = ",".join(f"'{g}'" for g in gvkeys)

    df = db.raw_sql(f"""
        SELECT gvkey, datadate, fyear,
               mkvalt, csho, prcc_f,
               revt, sale, at, lt, ni, oibdp,
               sich AS sic, naicsh AS naics
        FROM comp.funda
        WHERE gvkey IN ({gvkey_str})
          AND datadate >= '2018-01-01'
          AND indfmt = 'INDL'
          AND datafmt = 'STD'
          AND popsrc = 'D'
          AND consol  = 'C'
        ORDER BY gvkey, datadate
    """, date_cols=["datadate"])

    df["roa"]    = df["ni"] / df["at"].replace(0, pd.NA)
    df["ebitda"] = df["oibdp"]
    df["leverage"] = df["lt"] / df["at"].replace(0, pd.NA)

    # Re-attach permno and parent ticker
    df = df.merge(perm_gvkey[["gvkey","permno"]].drop_duplicates(), on="gvkey", how="left")
    ticker_map = (events[["permno","parent_ticker"]]
                  .drop_duplicates("permno")
                  .assign(permno=lambda x: pd.to_numeric(x["permno"], errors="coerce")
                          .astype("Int64")))
    df = df.merge(ticker_map, on="permno", how="left")

    print(f"[fundamentals] {len(df)} annual obs, "
          f"{df['gvkey'].nunique()} companies, "
          f"{df['datadate'].min().year}–{df['datadate'].max().year}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    db = connect()

    print("\n[1/3] Pulling spinoff child CRSP daily prices...")
    children = pull_child_crsp(db)
    children.to_parquet(RAW_DIR / "spinoff_children_crsp.parquet", index=False)

    print("\n[2/3] Checking spinoff child S&P 500 membership...")
    child_sp500 = pull_child_sp500_membership(db)
    child_sp500.to_parquet(RAW_DIR / "spinoff_children_sp500.parquet", index=False)

    print("\n[3/3] Pulling Compustat parent fundamentals...")
    fundamentals = pull_parent_fundamentals(db)
    fundamentals.to_parquet(RAW_DIR / "parent_fundamentals.parquet", index=False)

    db.close()

    print("\n=== Done. Files written to data/raw/ ===")
    print("  spinoff_children_crsp.parquet")
    print("  spinoff_children_sp500.parquet")
    print("  parent_fundamentals.parquet")
    print()
    print("Missing (not yet in CRSP — 2025 events):")
    print("  OXY/WS  — OXY warrants, excluded (not equity)")
    print("  MRP     — Millrose Properties (LEN/B Jan 2025)")
    print("  SNDK    — New SanDisk (WDC Feb 2025)")
    print("  RAL     — Ralliant Corp (FTV Jun 2025)")
    print("  SOLS    — Solaris (HON Oct 2025, future)")


if __name__ == "__main__":
    main()
