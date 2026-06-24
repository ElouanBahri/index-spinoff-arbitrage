"""
Merge cleaned Bloomberg spinoff events with WRDS/CRSP data.

Input:  data/clean/cleaned_primary_spinoffs.csv   (one row per spinoff event, manually curated)
        data/raw/*.parquet                          (WRDS pulls from pull_data.ipynb)

Outputs:
  data/clean/spinoff_events_merged.csv    — one row per event, WRDS metadata joined
  data/clean/spinoff_price_windows.parquet — long-format daily prices for each parent,
                                             filtered to [event_date - 90, event_date + 90]

Run:
  python merge_data.py               # uses only local parquet files
  python merge_data.py --pull-crsp   # also connects to WRDS to pull missing parent prices
"""

import argparse
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")

EVENT_WINDOW_DAYS = 90  # trading days on each side of effective date


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_spinoff_events() -> pd.DataFrame:
    df = pd.read_csv(
        CLEAN_DIR / "cleaned_primary_spinoffs.csv",
        parse_dates=["Announce/Declared Date", "Effective Date"],
    )
    df = df.rename(columns={
        "Action Type": "action_type",
        "Security ID": "security_id",
        "Parent Ticker": "parent_ticker",
        "PERMNO": "permno",
        "Announce/Declared Date": "announce_date",
        "Effective Date": "effective_date",
        "Amd Flag": "amd_flag",
        "Name": "parent_name",
        "Spun-off Company name": "spinoff_name",
        "Spun off Company ticker": "spinoff_ticker",
        "Terms": "spinoff_ratio",
    })
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")

    # Strip " US Equity" suffix from spinoff ticker for readability
    df["spinoff_ticker"] = df["spinoff_ticker"].str.replace(" US Equity", "", regex=False).str.strip()

    return df


def load_ccm_link() -> pd.DataFrame:
    ccm = pd.read_parquet(RAW_DIR / "ccm_link.parquet")
    ccm["permno"] = pd.to_numeric(ccm["permno"], errors="coerce").astype("Int64")
    # Take one gvkey per permno (most recent link wins — handles restructurings)
    return (
        ccm.sort_values("linkdt")
        .drop_duplicates(subset=["permno"], keep="last")
        [["permno", "gvkey"]]
    )


def load_constituents() -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / "sp500_constituents_pit.parquet")[
        ["gvkey", "start_date", "end_date", "still_active",
         "company_name", "ticker", "cusip", "sic", "naics"]
    ]


def load_crsp_daily() -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / "crsp_daily.parquet")


def load_index_returns() -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / "crsp_index_returns.parquet")


# ---------------------------------------------------------------------------
# WRDS pull for parent permnos missing from local CRSP file
# ---------------------------------------------------------------------------

def pull_missing_crsp(missing_permnos: list[int], start_date: str = "2019-01-01") -> pd.DataFrame:
    """Connect to WRDS and pull daily CRSP data for permnos not in local parquet."""
    import wrds

    db = wrds.Connection()
    permno_str = ",".join(str(p) for p in missing_permnos)
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    df = db.raw_sql(f"""
        SELECT permno, date, prc, ret, retx, vol, shrout, cfacpr, cfacshr
        FROM crsp.dsf
        WHERE permno IN ({permno_str})
          AND date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY permno, date
    """, date_cols=["date"])

    db.close()

    df["adj_prc"] = df["prc"].abs() / df["cfacpr"]
    df["mktcap"] = df["prc"].abs() * df["shrout"] * 1_000
    df["dollar_vol"] = df["vol"] * df["prc"].abs()
    return df


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def build_events_table(events: pd.DataFrame, ccm: pd.DataFrame, constituents: pd.DataFrame) -> pd.DataFrame:
    """Join spinoff events with WRDS identifiers and S&P 500 membership dates."""

    # Step 1: permno → gvkey via CCM (one gvkey per permno, most recent link)
    merged = events.merge(ccm, on="permno", how="left")

    # Step 2: gvkey → constituent info.
    # A gvkey can have multiple membership windows; keep the record whose window
    # covers the spinoff effective date, or the latest record if none match.
    const_cols = constituents[["gvkey", "start_date", "end_date", "still_active",
                                "company_name", "ticker", "cusip", "sic", "naics"]].copy()
    const_cols = const_cols.rename(columns={
        "start_date": "sp500_start",
        "end_date": "sp500_end",
        "company_name": "company_name_wrds",
        "ticker": "ticker_wrds",
    })

    # Join all constituent records for each matched gvkey
    joined = merged.merge(const_cols, on="gvkey", how="left")

    # Flag rows where the constituent window covers the spinoff date
    has_const = joined["sp500_start"].notna()
    in_window = has_const & (
        (joined["sp500_start"] <= joined["effective_date"]) &
        (joined["sp500_end"] >= joined["effective_date"])
    )
    joined["in_sp500_at_spinoff"] = in_window

    # For each event keep: the in-window record if it exists, else any constituent
    # record (latest), else the single null-const row
    def pick_best(grp):
        if grp["in_sp500_at_spinoff"].any():
            return grp[grp["in_sp500_at_spinoff"]]
        if grp["sp500_start"].notna().any():
            return grp.sort_values("sp500_start").tail(1)
        return grp.head(1)

    result = (
        joined.groupby(["parent_name", "effective_date", "spinoff_name"], group_keys=False)
        .apply(pick_best)
        .reset_index(drop=True)
    )

    col_order = [
        "parent_name", "parent_ticker", "permno", "gvkey",
        "spinoff_name", "spinoff_ticker", "spinoff_ratio",
        "announce_date", "effective_date",
        "company_name_wrds", "ticker_wrds", "cusip", "sic", "naics",
        "sp500_start", "sp500_end", "still_active", "in_sp500_at_spinoff",
    ]
    return result[col_order].sort_values("effective_date").reset_index(drop=True)


def build_price_windows(events: pd.DataFrame, crsp: pd.DataFrame,
                         index_ret: pd.DataFrame) -> pd.DataFrame:
    """
    For each spinoff event, extract daily CRSP prices for the parent in
    [effective_date - WINDOW, effective_date + WINDOW] calendar days.
    Returns a long-format DataFrame tagged with event metadata.
    """
    trading_dates = pd.Series(crsp["date"].sort_values().unique())
    frames = []

    for _, ev in events.iterrows():
        permno = ev["permno"]
        eff_date = ev["effective_date"]

        parent_prices = crsp[crsp["permno"] == permno].copy()
        if parent_prices.empty:
            continue

        # Use calendar-day window (approximate; CRSP has ~252 trading days/yr)
        cal_window = pd.Timedelta(days=EVENT_WINDOW_DAYS * 1.5)
        window = parent_prices[
            (parent_prices["date"] >= eff_date - cal_window) &
            (parent_prices["date"] <= eff_date + cal_window)
        ].copy()

        # Trading-day offset relative to effective date
        all_td = trading_dates[
            (trading_dates >= eff_date - cal_window) &
            (trading_dates <= eff_date + cal_window)
        ]
        eff_idx = all_td.searchsorted(eff_date)
        td_map = {d: i - eff_idx for i, d in enumerate(all_td)}
        window["t"] = window["date"].map(td_map)

        # Attach event metadata
        window["parent_name"] = ev["parent_name"]
        window["parent_ticker"] = ev["parent_ticker"]
        window["spinoff_name"] = ev["spinoff_name"]
        window["spinoff_ticker"] = ev["spinoff_ticker"]
        window["effective_date"] = eff_date

        frames.append(window)

    if not frames:
        return pd.DataFrame()

    price_windows = pd.concat(frames, ignore_index=True)

    # Join S&P 500 index return (sprtrn) for market-adjusted return computation
    price_windows = price_windows.merge(
        index_ret[["date", "sprtrn", "vwretd"]],
        on="date",
        how="left",
    )
    price_windows["ret_mkt_adj"] = price_windows["ret"] - price_windows["sprtrn"]

    return price_windows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(pull_crsp: bool = False):
    print("Loading Bloomberg spinoff events...")
    events = load_spinoff_events()
    print(f"  {len(events)} spinoff events")

    print("Loading WRDS tables...")
    ccm = load_ccm_link()
    constituents = load_constituents()
    crsp = load_crsp_daily()
    index_ret = load_index_returns()

    # Optionally pull CRSP for parent permnos not in local file
    local_permnos = set(crsp["permno"].unique())
    event_permnos = set(events["permno"].dropna().astype(int))
    missing = sorted(event_permnos - local_permnos)

    if missing:
        print(f"\n  {len(missing)} parent PERMNOs not in local CRSP file: {missing}")
        if pull_crsp:
            print("  Connecting to WRDS to pull missing prices...")
            new_crsp = pull_missing_crsp(missing)
            crsp = pd.concat([crsp, new_crsp], ignore_index=True)
            crsp.to_parquet(RAW_DIR / "crsp_daily.parquet")
            print(f"  Pulled and appended. Total CRSP rows: {len(crsp):,}")
        else:
            print("  Run with --pull-crsp to fetch them from WRDS.")
    else:
        print("  All parent PERMNOs present in local CRSP file.")

    print("\nBuilding events table...")
    events_merged = build_events_table(events, ccm, constituents)
    out_events = CLEAN_DIR / "spinoff_events_merged.csv"
    events_merged.to_csv(out_events, index=False)
    matched = events_merged["gvkey"].notna().sum()
    in_sp500 = events_merged["in_sp500_at_spinoff"].sum()
    print(f"  {matched}/{len(events_merged)} events matched to WRDS gvkey")
    print(f"  {in_sp500}/{len(events_merged)} parents confirmed in S&P 500 at spinoff date")
    print(f"  Saved -> {out_events}")

    print("\nBuilding price windows...")
    price_windows = build_price_windows(events_merged, crsp, index_ret)
    if not price_windows.empty:
        out_prices = CLEAN_DIR / "spinoff_price_windows.parquet"
        price_windows.to_parquet(out_prices, index=False)
        covered = price_windows["parent_ticker"].nunique()
        print(f"  Price data for {covered}/{len(events_merged)} parents")
        print(f"  Saved -> {out_prices}")
    else:
        print("  No price data available yet — run with --pull-crsp.")

    print("\nDone.")
    return events_merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pull-crsp", action="store_true",
                        help="Connect to WRDS and pull CRSP data for missing parent permnos")
    args = parser.parse_args()
    result = main(pull_crsp=args.pull_crsp)
    print()
    print(result.to_string())
