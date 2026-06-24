"""
Pull passive S&P 500 fund AUM from WRDS CRSP Mutual Fund database.

Identifies passive S&P 500 tracking funds (ETFs + mutual funds) by:
  1. index_fund_flag = 'D' (CRSP designation for index fund)
  2. Fund name contains 'S&P 500' or '500 INDEX' etc.

Aggregates monthly total net assets across all qualifying funds to produce
a total passive S&P 500 AUM time series used to estimate forced-flow selling
pressure at each spinoff inclusion/exclusion event.

Output: data/raw/sp500_passive_aum.parquet
        data/raw/sp500_passive_funds.parquet   (fund-level monthly detail)

Requires WRDS access. Run once; results are cached locally.
"""

import os
import wrds
import pandas as pd
from pathlib import Path

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

WRDS_USERNAME = os.getenv("WRDS_USERNAME", "vedantbhagat")
WRDS_PASSWORD = os.getenv("WRDS_PASSWORD")

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Keywords that identify S&P 500 tracking mandates in fund names
SP500_NAME_PATTERNS = [
    "S&P 500", "S&P500", "SP 500", "SP500",
    "500 INDEX", "500 INDEX FUND", "STANDARD & POOR",
    "STANDARD AND POOR",
]

# Well-known S&P 500 ETF tickers — used to sanity-check the fund filter
KNOWN_SP500_ETF_TICKERS = {"SPY", "IVV", "VOO", "SPLG", "CSPX", "IUSA"}


def inspect_mf_schema(db):
    """Print schema for key CRSP mutual fund tables."""
    for schema, table in [
        ("crsp_q_mutualfunds", "fund_hdr"),
        ("crsp_q_mutualfunds", "fund_summary"),
        ("crsp_q_mutualfunds", "fund_style"),
    ]:
        try:
            cols = db.raw_sql(f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = '{schema}' AND table_name = '{table}'
                ORDER BY ordinal_position
            """)
            print(f"\n[{schema}.{table}]")
            print(cols.to_string(index=False))
        except Exception as e:
            print(f"[{schema}.{table}] ERROR: {e}")


def pull_sp500_fund_headers(db) -> pd.DataFrame:
    """
    Retrieve fund headers for passive S&P 500 tracking funds.

    index_fund_flag = 'D' identifies index funds in CRSP.
    We also accept funds whose name pattern matches S&P 500 keywords
    even if the flag is missing (some older funds aren't flagged correctly).
    """
    name_conditions = " OR ".join(
        f"UPPER(fund_name) LIKE '%{p}%'" for p in SP500_NAME_PATTERNS
    )

    df = db.raw_sql(f"""
        SELECT
            crsp_fundno,
            fund_name,
            nasdaq         AS ticker,
            index_fund_flag,
            et_flag,
            dead_flag,
            delist_cd
        FROM crsp_q_mutualfunds.fund_hdr
        WHERE index_fund_flag = 'D'
           OR ({name_conditions})
        ORDER BY crsp_fundno
    """)

    # Keep only those that are (a) index funds OR (b) match name — this cast
    # is wide intentionally; we'll tighten by name below
    name_upper = df["fund_name"].str.upper().fillna("")
    is_sp500_name = name_upper.apply(
        lambda n: any(p in n for p in SP500_NAME_PATTERNS)
    )
    is_index_fund = df["index_fund_flag"] == "D"
    df = df[is_index_fund & is_sp500_name].copy()

    print(f"[fund_headers] {len(df)} passive S&P 500 fund share classes")
    print(f"  ETFs: {(df['et_flag'] == 'F').sum()}, "
          f"Mutual funds: {(df['et_flag'] != 'F').sum()}")

    known_found = set(df["ticker"].dropna()) & KNOWN_SP500_ETF_TICKERS
    print(f"  Known ETFs found: {sorted(known_found)}")

    df.to_parquet(RAW_DIR / "sp500_passive_funds_hdr.parquet")
    return df


def pull_fund_monthly_tna(db, fund_headers: pd.DataFrame,
                           start_date: str = "2018-01-01") -> pd.DataFrame:
    """
    Monthly total net assets (mtna, in $M) for each qualifying fund share class.

    We pull from 2018 to cover all spinoff events in the Bloomberg dataset.
    mtna is month-end TNA in millions of dollars.
    """
    fundnos = fund_headers["crsp_fundno"].dropna().astype(int).tolist()
    chunk_size = 500
    chunks = [fundnos[i:i + chunk_size] for i in range(0, len(fundnos), chunk_size)]

    frames = []
    for i, chunk in enumerate(chunks):
        fundno_str = ",".join(str(f) for f in chunk)
        df = db.raw_sql(f"""
            SELECT
                crsp_fundno,
                caldt       AS date,
                mtna        AS tna_millions,
                mret        AS monthly_ret
            FROM crsp_q_mutualfunds.fund_summary
            WHERE crsp_fundno IN ({fundno_str})
              AND caldt >= '{start_date}'
            ORDER BY crsp_fundno, caldt
        """, date_cols=["date"])
        frames.append(df)
        print(f"[fund_tna] chunk {i + 1}/{len(chunks)} -> {len(df)} rows")

    monthly = pd.concat(frames, ignore_index=True)

    # Join fund names for readability
    monthly = monthly.merge(
        fund_headers[["crsp_fundno", "fund_name", "ticker", "et_flag"]],
        on="crsp_fundno",
        how="left",
    )

    monthly.to_parquet(RAW_DIR / "sp500_passive_funds.parquet")
    print(f"[fund_tna] {len(monthly):,} monthly obs, "
          f"{monthly['crsp_fundno'].nunique()} funds, "
          f"date range: {monthly['date'].min().date()} -> {monthly['date'].max().date()}")
    return monthly


def build_aggregate_aum(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly TNA across all passive S&P 500 funds.

    Produces a monthly series of total passive AUM (in $B) used to
    scale the forced-flow estimate at each spinoff event.

    tna_millions can be NaN for months with no report — forward-fill
    within each fund for up to 3 months, then drop remaining NaNs
    before summing, to avoid understating AUM when funds skip a month.
    """
    monthly = monthly.copy()
    monthly["tna_millions"] = (
        monthly.sort_values("date")
        .groupby("crsp_fundno")["tna_millions"]
        .transform(lambda x: x.ffill(limit=3))
    )

    agg = (
        monthly.dropna(subset=["tna_millions"])
        .groupby("date")
        .agg(
            total_aum_millions=("tna_millions", "sum"),
            num_funds=("crsp_fundno", "nunique"),
        )
        .reset_index()
    )
    agg["total_aum_billions"] = agg["total_aum_millions"] / 1_000

    agg.to_parquet(RAW_DIR / "sp500_passive_aum.parquet")
    print(f"[passive_aum] {len(agg)} monthly observations")
    print(f"  Latest AUM: ${agg.iloc[-1]['total_aum_billions']:.0f}B "
          f"across {agg.iloc[-1]['num_funds']} funds ({agg.iloc[-1]['date'].date()})")
    return agg


def main():
    db = wrds.Connection(wrds_username=WRDS_USERNAME, wrds_password=WRDS_PASSWORD)

    print("Inspecting CRSP Mutual Fund schema...")
    inspect_mf_schema(db)

    print("\nPulling passive S&P 500 fund headers...")
    headers = pull_sp500_fund_headers(db)

    print("\nPulling monthly TNA...")
    monthly = pull_fund_monthly_tna(db, headers)

    print("\nAggregating to total passive AUM...")
    aum = build_aggregate_aum(monthly)

    db.close()
    print("\nDone. Files written to data/raw/")
    print("  sp500_passive_funds_hdr.parquet")
    print("  sp500_passive_funds.parquet")
    print("  sp500_passive_aum.parquet")
    return aum


if __name__ == "__main__":
    aum = main()
    print()
    print(aum.tail(12).to_string(index=False))
