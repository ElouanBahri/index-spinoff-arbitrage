"""
Full WRDS re-pull — replaces and extends the original pull_data.ipynb run.

Key changes vs. original pull:
  1. S&P 500 constituent history: uses crsp.msp500list (permno-based, tracks
     actual additions/deletions) instead of comp.idxcst_his (only current members).
     Captures all ~817 unique permnos active in the index from 2010 onward.
  2. CRSP daily prices: pulls from 2010-01-01 for ALL constituent permnos PLUS
     the spinoff parent permnos from cleaned_primary_spinoffs.csv.
  3. ETF AUM: pulls passive S&P 500 fund AUM from CRSP Mutual Fund DB using
     the correct tables (monthly_tna / monthly_returns, not fund_summary).

Run:
  python repull_data.py

Outputs (all written to data/raw/):
  sp500_constituents_pit.parquet        — S&P 500 member history from 2010
  ccm_link.parquet                      — Compustat CCM link (gvkey->permno)
  crsp_daily.parquet                    — daily prices, all constituent + parent permnos
  crsp_index_returns.parquet
  sp500_passive_funds_hdr.parquet
  sp500_passive_funds.parquet
  sp500_passive_aum.parquet
"""

import os
import builtins
import getpass
from pathlib import Path

# Load .env before anything else
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

WRDS_USERNAME = os.getenv("WRDS_USERNAME", "")
WRDS_PASSWORD = os.getenv("WRDS_PASSWORD", "")

# The installed WRDS library always calls input()/getpass() even when credentials
# are passed explicitly. Patch both so the script runs without a terminal.
_real_input   = builtins.input
_real_getpass = getpass.getpass

def _auto_input(prompt=""):
    if "username" in prompt.lower() and WRDS_USERNAME:
        print(f"{prompt}{WRDS_USERNAME}")
        return WRDS_USERNAME
    return _real_input(prompt)

def _auto_getpass(prompt="Password: ", stream=None):
    if WRDS_PASSWORD:
        return WRDS_PASSWORD
    return _real_getpass(prompt, stream=stream)

builtins.input    = _auto_input
getpass.getpass   = _auto_getpass

import wrds
import pandas as pd

RAW_DIR   = Path("data/raw")
CLEAN_DIR = Path("data/clean")
RAW_DIR.mkdir(parents=True, exist_ok=True)

CRSP_START       = "2010-01-01"
CONSTITUENT_FROM = "2010-01-01"   # pull members active at any point from here

# Fund name patterns that identify S&P 500 passive funds.
# Note: use %% for literal % in psycopg2 queries to avoid parameter-placeholder errors.
SP500_NAME_PATTERNS = [
    "S&P 500", "S&P500", "SP 500", "SP500",
    "500 INDEX", "STANDARD & POOR",
]


def connect() -> wrds.Connection:
    print(f"Connecting to WRDS as '{WRDS_USERNAME}'...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME, wrds_password=WRDS_PASSWORD)
    print("  Connected.")
    return db


# ---------------------------------------------------------------------------
# S&P 500 constituents — via CRSP msp500list (permno-based, tracks deletions)
# ---------------------------------------------------------------------------

def pull_sp500_members(db) -> pd.DataFrame:
    """
    Full S&P 500 constituent history via crsp.msp500list.

    Unlike comp.idxcst_his (which only returns current members on this account),
    msp500list tracks historical additions and deletions and returns ~817 unique
    permnos active since 2010.
    """
    df = db.raw_sql(f"""
        SELECT permno, start AS start_date, ending AS end_date
        FROM crsp.msp500list
        WHERE ending >= '{CONSTITUENT_FROM}'
        ORDER BY permno, start
    """, date_cols=["start_date", "end_date"])

    today = pd.Timestamp.today().normalize()
    df["still_active"] = df["end_date"] >= today

    print(f"[sp500_members] {len(df)} membership intervals, "
          f"{df['permno'].nunique()} unique permnos")
    return df


def attach_permno_names(db, members: pd.DataFrame) -> pd.DataFrame:
    """Join CRSP stock names (ticker, company name) to the member list."""
    permnos = members["permno"].unique().tolist()
    perm_str = ",".join(str(p) for p in permnos)
    names = db.raw_sql(f"""
        SELECT DISTINCT ON (permno)
            permno,
            ticker,
            comnam AS company_name,
            shrcd,
            exchcd,
            naics
        FROM crsp.dsenames
        WHERE permno IN ({perm_str})
        ORDER BY permno, namedt DESC
    """)
    return members.merge(names, on="permno", how="left")


# ---------------------------------------------------------------------------
# CCM link (kept for Compustat joins in merge_data.py)
# ---------------------------------------------------------------------------

def pull_ccm_link(db) -> pd.DataFrame:
    df = db.raw_sql("""
        SELECT
            gvkey,
            lpermno  AS permno,
            linktype,
            linkprim,
            linkdt,
            linkenddt
        FROM crsp.ccmxpf_lnkhist
        WHERE linktype IN ('LU', 'LC')
          AND linkprim IN ('P', 'C')
    """, date_cols=["linkdt", "linkenddt"])

    df["linkenddt"] = df["linkenddt"].fillna(pd.Timestamp.today().normalize())
    print(f"[ccm_link] {len(df)} link records, {df['permno'].nunique()} unique permnos")
    return df


# ---------------------------------------------------------------------------
# CRSP daily prices
# ---------------------------------------------------------------------------

def pull_crsp_daily(db, permnos: list[int], chunk_size: int = 500) -> pd.DataFrame:
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    permnos  = sorted(set(int(p) for p in permnos if pd.notna(p)))
    chunks   = [permnos[i:i + chunk_size] for i in range(0, len(permnos), chunk_size)]

    frames = []
    for i, chunk in enumerate(chunks):
        permno_str = ",".join(str(p) for p in chunk)
        df = db.raw_sql(f"""
            SELECT permno, date, prc, ret, retx, vol, shrout, cfacpr, cfacshr
            FROM crsp.dsf
            WHERE permno IN ({permno_str})
              AND date BETWEEN '{CRSP_START}' AND '{end_date}'
            ORDER BY permno, date
        """, date_cols=["date"])
        frames.append(df)
        print(f"  chunk {i + 1}/{len(chunks)} -> {len(df):,} rows")

    daily = pd.concat(frames, ignore_index=True)
    daily["adj_prc"]    = daily["prc"].abs() / daily["cfacpr"]
    daily["mktcap"]     = daily["prc"].abs() * daily["shrout"] * 1_000
    daily["dollar_vol"] = daily["vol"] * daily["prc"].abs()

    daily = daily.sort_values(["permno", "date"])
    daily["adv_30d"] = (
        daily.groupby("permno")["dollar_vol"]
        .transform(lambda x: x.rolling(30, min_periods=5).mean())
    )
    print(f"[crsp_daily] {len(daily):,} rows total, "
          f"{daily['permno'].nunique()} permnos")
    return daily


def pull_index_returns(db) -> pd.DataFrame:
    df = db.raw_sql(f"""
        SELECT date, vwretd, ewretd, sprtrn
        FROM crsp.dsi
        WHERE date >= '{CRSP_START}'
        ORDER BY date
    """, date_cols=["date"])
    print(f"[index_returns] {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Passive ETF AUM (CRSP Mutual Fund DB)
# ---------------------------------------------------------------------------

def pull_passive_aum(db) -> tuple:
    """
    Pull passive S&P 500 fund headers and monthly TNA.

    Uses crsp_q_mutualfunds.monthly_tna (not fund_summary) which has the
    correct mtna column. LIKE clauses use %% to escape % from psycopg2.
    """
    name_conditions = " OR ".join(
        f"UPPER(fund_name) LIKE '%%{p}%%'" for p in SP500_NAME_PATTERNS
    )

    headers = db.raw_sql(f"""
        SELECT crsp_fundno, fund_name, ticker,
               index_fund_flag, et_flag, dead_flag
        FROM crsp_q_mutualfunds.fund_hdr
        WHERE index_fund_flag = 'D'
          AND ({name_conditions})
        ORDER BY crsp_fundno
    """)

    print(f"[fund_headers] {len(headers)} passive S&P 500 share classes "
          f"({(headers['et_flag'] == 'F').sum()} ETFs, "
          f"{(headers['et_flag'] != 'F').sum()} mutual funds)")

    fundnos    = headers["crsp_fundno"].dropna().astype(int).tolist()
    chunk_size = 500
    chunks     = [fundnos[i:i + chunk_size] for i in range(0, len(fundnos), chunk_size)]

    tna_frames = []
    ret_frames = []
    for i, chunk in enumerate(chunks):
        fundno_str = ",".join(str(f) for f in chunk)

        tna = db.raw_sql(f"""
            SELECT crsp_fundno, caldt AS date, mtna AS tna_millions
            FROM crsp_q_mutualfunds.monthly_tna
            WHERE crsp_fundno IN ({fundno_str})
              AND caldt >= '{CRSP_START}'
            ORDER BY crsp_fundno, caldt
        """, date_cols=["date"])
        tna_frames.append(tna)

        ret = db.raw_sql(f"""
            SELECT crsp_fundno, caldt AS date, mret AS monthly_ret
            FROM crsp_q_mutualfunds.monthly_returns
            WHERE crsp_fundno IN ({fundno_str})
              AND caldt >= '{CRSP_START}'
            ORDER BY crsp_fundno, caldt
        """, date_cols=["date"])
        ret_frames.append(ret)

        print(f"[fund_data] chunk {i + 1}/{len(chunks)} -> "
              f"{len(tna)} TNA rows, {len(ret)} return rows")

    monthly_tna  = pd.concat(tna_frames,  ignore_index=True)
    monthly_ret  = pd.concat(ret_frames,  ignore_index=True)
    monthly      = monthly_tna.merge(monthly_ret, on=["crsp_fundno", "date"], how="outer")
    monthly      = monthly.merge(
        headers[["crsp_fundno", "fund_name", "ticker", "et_flag"]],
        on="crsp_fundno", how="left"
    )

    # forward-fill TNA within each fund (up to 3 months) then aggregate
    monthly = monthly.sort_values(["crsp_fundno", "date"])
    monthly["tna_millions"] = (
        monthly.groupby("crsp_fundno")["tna_millions"]
        .transform(lambda x: x.ffill(limit=3))
    )

    agg = (
        monthly.dropna(subset=["tna_millions"])
        .groupby("date")
        .agg(total_aum_millions=("tna_millions", "sum"),
             num_funds=("crsp_fundno", "nunique"))
        .reset_index()
    )
    agg["total_aum_billions"] = agg["total_aum_millions"] / 1_000

    latest = agg.sort_values("date").iloc[-1]
    print(f"[passive_aum] {len(agg)} monthly obs | latest: "
          f"${latest['total_aum_billions']:.0f}B across "
          f"{int(latest['num_funds'])} funds ({latest['date'].date()})")

    return headers, monthly, agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    db = connect()

    # --- S&P 500 member history (via CRSP, permno-based) ---
    print("\n[1/6] Pulling S&P 500 constituent history (via crsp.msp500list)...")
    members = pull_sp500_members(db)
    members = attach_permno_names(db, members)
    members.to_parquet(RAW_DIR / "sp500_constituents_pit.parquet")

    # --- CCM link (for Compustat joins) ---
    print("\n[2/6] Pulling CCM link table...")
    link = pull_ccm_link(db)
    link.to_parquet(RAW_DIR / "ccm_link.parquet")

    # --- CRSP daily prices ---
    print("\n[3/6] Pulling CRSP daily prices...")
    constituent_permnos = set(members["permno"].dropna().astype(int))

    spinoff_path = CLEAN_DIR / "cleaned_primary_spinoffs.csv"
    if spinoff_path.exists():
        spinoffs = pd.read_csv(spinoff_path)
        parent_permnos = set(
            pd.to_numeric(spinoffs["PERMNO"], errors="coerce").dropna().astype(int)
        )
        all_permnos = constituent_permnos | parent_permnos
        extra = parent_permnos - constituent_permnos
        print(f"  Constituent permnos: {len(constituent_permnos)}, "
              f"parent permnos added: {len(extra)}, total: {len(all_permnos)}")
    else:
        all_permnos = constituent_permnos
        print(f"  Constituent permnos: {len(all_permnos)}")

    daily = pull_crsp_daily(db, list(all_permnos))
    daily.to_parquet(RAW_DIR / "crsp_daily.parquet")

    # --- Index returns ---
    print("\n[4/6] Pulling CRSP index returns...")
    index_ret = pull_index_returns(db)
    index_ret.to_parquet(RAW_DIR / "crsp_index_returns.parquet")

    # --- Passive ETF AUM ---
    print("\n[5/6] Pulling passive S&P 500 fund AUM...")
    try:
        hdr, monthly_tna, aum = pull_passive_aum(db)
        hdr.to_parquet(RAW_DIR / "sp500_passive_funds_hdr.parquet")
        monthly_tna.to_parquet(RAW_DIR / "sp500_passive_funds.parquet")
        aum.to_parquet(RAW_DIR / "sp500_passive_aum.parquet")
    except Exception as e:
        import traceback
        print(f"  [!] AUM pull failed: {e}")
        traceback.print_exc()
        print("  Skipping AUM — fix and re-run pull_etf_aum.py if needed.")

    db.close()

    # --- Rebuild merged dataset ---
    print("\n[6/6] Rebuilding merged spinoff dataset with forced-flow features...")
    import subprocess
    result = subprocess.run(["python3", "merge_data.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("[!] merge_data.py error:", result.stderr[:500])

    print("\nAll done. Files written to data/raw/ and data/clean/")


if __name__ == "__main__":
    main()
