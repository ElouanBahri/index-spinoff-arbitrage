"""
Full WRDS re-pull — replaces and extends the original pull_data.ipynb run.

Fixes vs. original pull:
  1. S&P 500 constituent history: removes start_date='1996-01-01' filter so
     pre-1996 members (IBM, PFE, MRK, GE, etc.) are included. Pulls from 1990.
  2. CRSP daily prices: pulls from 2019-01-01 for ALL constituent permnos PLUS
     the spinoff parent permnos from cleaned_primary_spinoffs.csv.
  3. ETF AUM: pulls passive S&P 500 fund AUM from CRSP Mutual Fund DB.
  4. Credentials: reads WRDS_USERNAME from .env / environment, looks up
     password from ~/.pgpass — no interactive prompt needed.

Run:
  python repull_data.py

Outputs (all written to data/raw/):
  sp500_constituents_pit.parquet        — full S&P 500 member history from 1990
  sp500_constituents_with_permno.parquet
  ccm_link.parquet                      (re-pulled; same filter, same data)
  crsp_daily.parquet                    — daily prices for all constituent +
                                          spinoff parent permnos, 2019-present
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

WRDS_USERNAME = os.getenv("WRDS_USERNAME", "vedantbhagat")
WRDS_PASSWORD = os.getenv("WRDS_PASSWORD", "")

# The installed WRDS library always calls input()/getpass() even when credentials
# are passed explicitly. Patch both so the script runs without a terminal.
_real_input = builtins.input

def _auto_input(prompt=""):
    if "username" in prompt.lower() and WRDS_USERNAME:
        print(f"{prompt}{WRDS_USERNAME}")
        return WRDS_USERNAME
    return _real_input(prompt)

builtins.input = _auto_input
# Password: leave getpass alone — WRDS web password must be typed interactively

import wrds
import pandas as pd

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
RAW_DIR.mkdir(parents=True, exist_ok=True)
CRSP_START = "2010-01-01"        # daily prices start date (10yr window before first spinoff)
CONSTITUENT_START = "1990-01-01" # constituent history start — captures all pre-1996 members

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
# Constituents
# ---------------------------------------------------------------------------

def pull_sp500_constituents(db) -> pd.DataFrame:
    """
    Full point-in-time S&P 500 membership from 1990 onward.
    Previous pull used start_date='1996-01-01' which excluded any company
    that was already in the index before 1996 (IBM, PFE, MRK, GE, etc.).
    """
    df = db.raw_sql(f"""
        SELECT
            gvkey,
            "from"  AS start_date,
            thru    AS end_date
        FROM comp.idxcst_his
        WHERE gvkeyx = '000003'
          AND "from" >= '{CONSTITUENT_START}'
        ORDER BY "from"
    """, date_cols=["start_date", "end_date"])

    df["end_date"] = df["end_date"].fillna(pd.Timestamp.today().normalize())
    df["still_active"] = df["end_date"] == pd.Timestamp.today().normalize()

    print(f"[constituents] {len(df)} membership records, "
          f"{df['gvkey'].nunique()} unique gvkeys")
    return df


def attach_company_names(db, constituents: pd.DataFrame) -> pd.DataFrame:
    gvkeys = constituents["gvkey"].unique().tolist()
    gvkey_str = ",".join(f"'{g}'" for g in gvkeys)
    names = db.raw_sql(f"""
        SELECT gvkey, conm AS company_name, tic AS ticker, cusip, cik, sic, naics
        FROM comp.names
        WHERE gvkey IN ({gvkey_str})
    """)
    names = names.drop_duplicates(subset=["gvkey"], keep="last")
    return constituents.merge(names, on="gvkey", how="left")


# ---------------------------------------------------------------------------
# CCM link
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


def map_gvkey_to_permno(constituents: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    merged = constituents.merge(link, on="gvkey", how="left")
    overlap = (merged["linkdt"] <= merged["end_date"]) & (
        merged["linkenddt"] >= merged["start_date"]
    )
    merged = merged[overlap].copy()
    merged = merged.drop_duplicates(subset=["gvkey", "start_date", "permno"])
    print(f"[gvkey_to_permno] {len(merged)} rows, "
          f"{merged['permno'].nunique()} unique permnos")
    return merged


# ---------------------------------------------------------------------------
# CRSP daily prices
# ---------------------------------------------------------------------------

def pull_crsp_daily(db, permnos: list[int], chunk_size: int = 500) -> pd.DataFrame:
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    permnos = sorted(set(int(p) for p in permnos if pd.notna(p)))
    chunks = [permnos[i:i + chunk_size] for i in range(0, len(permnos), chunk_size)]

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
    daily["adj_prc"] = daily["prc"].abs() / daily["cfacpr"]
    daily["mktcap"] = daily["prc"].abs() * daily["shrout"] * 1_000
    daily["dollar_vol"] = daily["vol"] * daily["prc"].abs()

    # rolling 30-day ADV
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

def pull_passive_aum(db) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull passive S&P 500 fund headers and monthly TNA."""

    name_conditions = " OR ".join(
        f"UPPER(fund_name) LIKE '%{p}%'" for p in SP500_NAME_PATTERNS
    )

    headers = db.raw_sql(f"""
        SELECT crsp_fundno, fund_name, nasdaq AS ticker,
               index_fund_flag, et_flag, dead_flag
        FROM crsp_q_mutualfunds.fund_hdr
        WHERE index_fund_flag = 'D'
          AND ({name_conditions})
        ORDER BY crsp_fundno
    """)

    print(f"[fund_headers] {len(headers)} passive S&P 500 share classes "
          f"({(headers['et_flag'] == 'F').sum()} ETFs, "
          f"{(headers['et_flag'] != 'F').sum()} mutual funds)")

    fundnos = headers["crsp_fundno"].dropna().astype(int).tolist()
    chunk_size = 500
    chunks = [fundnos[i:i + chunk_size] for i in range(0, len(fundnos), chunk_size)]

    frames = []
    for i, chunk in enumerate(chunks):
        fundno_str = ",".join(str(f) for f in chunk)
        df = db.raw_sql(f"""
            SELECT crsp_fundno, caldt AS date,
                   mtna AS tna_millions, mret AS monthly_ret
            FROM crsp_q_mutualfunds.fund_summary
            WHERE crsp_fundno IN ({fundno_str})
              AND caldt >= '2010-01-01'
            ORDER BY crsp_fundno, caldt
        """, date_cols=["date"])
        frames.append(df)

    monthly = pd.concat(frames, ignore_index=True)
    monthly = monthly.merge(
        headers[["crsp_fundno", "fund_name", "ticker", "et_flag"]],
        on="crsp_fundno", how="left"
    )

    # forward-fill within each fund (up to 3 months) then aggregate
    monthly["tna_millions"] = (
        monthly.sort_values("date")
        .groupby("crsp_fundno")["tna_millions"]
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

    # --- Constituents ---
    print("\n[1/6] Pulling S&P 500 constituent history (1990-present)...")
    constituents = pull_sp500_constituents(db)
    constituents = attach_company_names(db, constituents)
    constituents.to_parquet(RAW_DIR / "sp500_constituents_pit.parquet")

    # --- CCM link ---
    print("\n[2/6] Pulling CCM link table...")
    link = pull_ccm_link(db)
    link.to_parquet(RAW_DIR / "ccm_link.parquet")

    mapped = map_gvkey_to_permno(constituents, link)
    mapped.to_parquet(RAW_DIR / "sp500_constituents_with_permno.parquet")

    # --- CRSP daily prices ---
    print("\n[3/6] Pulling CRSP daily prices...")
    constituent_permnos = set(mapped["permno"].dropna().astype(int))

    # also include spinoff parent permnos from cleaned_primary_spinoffs.csv
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
        print(f"  [!] AUM pull failed: {e}")
        print("  Skipping AUM — run pull_etf_aum.py separately if needed.")

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
