"""
Merge cleaned Bloomberg spinoff events with WRDS/CRSP data and compute
the forced-flow feature for each event.

Inputs:
  data/clean/cleaned_primary_spinoffs.csv   (one row per event, manually curated)
  data/raw/crsp_daily.parquet               (CRSP daily prices for S&P 500 members)
  data/raw/crsp_index_returns.parquet       (daily S&P 500 index returns)
  data/raw/ccm_link.parquet                 (CRSP-Compustat permno→gvkey mapping)
  data/raw/sp500_constituents_pit.parquet   (S&P 500 point-in-time membership)
  data/raw/sp500_passive_aum.parquet        (monthly passive S&P 500 AUM — from pull_etf_aum.py)

Outputs:
  data/clean/spinoff_events_merged.csv      — one row per event with WRDS metadata
                                              and forced-flow estimates
  data/clean/spinoff_price_windows.parquet  — long-format daily parent prices,
                                              ±90 days around each spinoff date

Run order:
  1. python pull_data.ipynb           (pulls CRSP/Compustat via WRDS — already done)
  2. python pull_etf_aum.py           (pulls passive fund AUM via WRDS)
  3. python merge_data.py             (combine; add --pull-crsp to also fetch parent prices)

Flags:
  --pull-crsp   Connect to WRDS and fetch CRSP daily prices for parent company
                PERMNOs not yet in the local crsp_daily.parquet (25 of 26 parents
                were pre-1996 S&P 500 members, so they were missed by the original pull).
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")

EVENT_WINDOW_DAYS = 90  # calendar days on each side of effective date


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
    df["spinoff_ticker"] = df["spinoff_ticker"].str.replace(" US Equity", "", regex=False).str.strip()
    return df


def load_ccm_link() -> pd.DataFrame:
    ccm = pd.read_parquet(RAW_DIR / "ccm_link.parquet")
    ccm["permno"] = pd.to_numeric(ccm["permno"], errors="coerce").astype("Int64")
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


def load_passive_aum() -> pd.DataFrame | None:
    path = RAW_DIR / "sp500_passive_aum.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


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
# Events table: join spinoff events to WRDS identifiers
# ---------------------------------------------------------------------------

def build_events_table(events: pd.DataFrame, ccm: pd.DataFrame,
                        constituents: pd.DataFrame) -> pd.DataFrame:
    """Join spinoff events with WRDS identifiers and S&P 500 membership dates."""

    merged = events.merge(ccm, on="permno", how="left")

    const_cols = constituents[["gvkey", "start_date", "end_date", "still_active",
                                "company_name", "ticker", "cusip", "sic", "naics"]].copy()
    const_cols = const_cols.rename(columns={
        "start_date": "sp500_start",
        "end_date": "sp500_end",
        "company_name": "company_name_wrds",
        "ticker": "ticker_wrds",
    })

    joined = merged.merge(const_cols, on="gvkey", how="left")

    has_const = joined["sp500_start"].notna()
    in_window = has_const & (
        (joined["sp500_start"] <= joined["effective_date"]) &
        (joined["sp500_end"] >= joined["effective_date"])
    )
    joined["in_sp500_at_spinoff"] = in_window

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


# ---------------------------------------------------------------------------
# Forced-flow feature
# ---------------------------------------------------------------------------

def compute_index_weight(crsp: pd.DataFrame, permno: int,
                          ref_date: pd.Timestamp) -> float | None:
    """
    Parent's weight in S&P 500 on ref_date, approximated as:
        parent mktcap / sum(all S&P 500 constituent mktcaps)

    Uses the closest trading day at or before ref_date.
    Returns None if the parent's permno has no price data.
    """
    # snap to closest available date at or before ref_date
    available = crsp[crsp["date"] <= ref_date]["date"]
    if available.empty:
        return None
    snap_date = available.max()

    day = crsp[crsp["date"] == snap_date]
    parent_row = day[day["permno"] == permno]
    if parent_row.empty or parent_row["mktcap"].isna().all():
        return None

    parent_mktcap = parent_row["mktcap"].iloc[0]
    total_mktcap = day["mktcap"].sum()
    if total_mktcap == 0:
        return None

    return parent_mktcap / total_mktcap


def compute_adv(crsp: pd.DataFrame, permno: int,
                ref_date: pd.Timestamp, window: int = 30) -> float | None:
    """
    Parent's average daily dollar volume over the `window` trading days
    ending on (or just before) ref_date.
    """
    parent = crsp[(crsp["permno"] == permno) & (crsp["date"] <= ref_date)]
    if parent.empty:
        return None
    recent = parent.sort_values("date").tail(window)
    adv = recent["dollar_vol"].replace(0, np.nan).mean()
    return adv if pd.notna(adv) else None


def build_forced_flow_features(events: pd.DataFrame, crsp: pd.DataFrame,
                                passive_aum: pd.DataFrame | None) -> pd.DataFrame:
    """
    Compute forced-flow estimates for each spinoff event and append as columns.

    Columns added:
      parent_mktcap_usd      — parent market cap 1 trading day before effective date
      sp500_total_mktcap_usd — sum of all S&P 500 constituent mktcaps on same day
      parent_index_weight    — parent_mktcap / sp500_total_mktcap
      passive_aum_usd        — total passive S&P 500 AUM in the month before effective date
      forced_flow_usd        — passive_aum_usd × parent_index_weight
                               ($ amount passive funds hold of parent — estimates
                               forced selling pressure if parent is removed from index)
      parent_adv_usd         — parent's 30-day avg daily dollar volume before event
      forced_flow_adv        — forced_flow_usd / parent_adv_usd
                               (days of parent trading volume the forced selling represents)
    """
    rows = []

    for _, ev in events.iterrows():
        permno = int(ev["permno"]) if pd.notna(ev["permno"]) else None
        eff_date = ev["effective_date"]
        # Use one trading day before the effective date (pre-spinoff snapshot)
        ref_date = eff_date - pd.Timedelta(days=1)

        row = {
            "parent_mktcap_usd": None,
            "sp500_total_mktcap_usd": None,
            "parent_index_weight": None,
            "passive_aum_usd": None,
            "forced_flow_usd": None,
            "parent_adv_usd": None,
            "forced_flow_adv": None,
        }

        if permno is not None and not crsp.empty:
            available = crsp[crsp["date"] <= ref_date]["date"]
            if not available.empty:
                snap_date = available.max()
                day = crsp[crsp["date"] == snap_date]
                parent_row = day[day["permno"] == permno]

                if not parent_row.empty:
                    parent_mktcap = parent_row["mktcap"].iloc[0]
                    total_mktcap = day["mktcap"].sum()
                    row["parent_mktcap_usd"] = parent_mktcap
                    row["sp500_total_mktcap_usd"] = total_mktcap
                    if total_mktcap > 0:
                        row["parent_index_weight"] = parent_mktcap / total_mktcap

                adv = compute_adv(crsp, permno, ref_date)
                row["parent_adv_usd"] = adv

        # Passive AUM: last month-end on or before ref_date
        if passive_aum is not None and not passive_aum.empty:
            aum_before = passive_aum[passive_aum["date"] <= ref_date]
            if not aum_before.empty:
                latest_aum = aum_before.sort_values("date").iloc[-1]
                # total_aum_millions → convert to dollars
                row["passive_aum_usd"] = latest_aum["total_aum_millions"] * 1e6

        # Forced-flow
        if row["passive_aum_usd"] is not None and row["parent_index_weight"] is not None:
            row["forced_flow_usd"] = row["passive_aum_usd"] * row["parent_index_weight"]

        if row["forced_flow_usd"] is not None and row["parent_adv_usd"] is not None:
            if row["parent_adv_usd"] > 0:
                row["forced_flow_adv"] = row["forced_flow_usd"] / row["parent_adv_usd"]

        rows.append(row)

    ff = pd.DataFrame(rows, index=events.index)
    return pd.concat([events, ff], axis=1)


# ---------------------------------------------------------------------------
# Price windows
# ---------------------------------------------------------------------------

def build_price_windows(events: pd.DataFrame, crsp: pd.DataFrame,
                         index_ret: pd.DataFrame) -> pd.DataFrame:
    """
    For each spinoff event, extract daily CRSP prices for the parent in
    [effective_date - WINDOW, effective_date + WINDOW] calendar days.
    Returns a long-format DataFrame with trading-day offset (t=0 on effective date).
    """
    trading_dates = pd.Series(crsp["date"].sort_values().unique())
    frames = []

    for _, ev in events.iterrows():
        permno = ev["permno"]
        eff_date = ev["effective_date"]

        parent_prices = crsp[crsp["permno"] == permno].copy()
        if parent_prices.empty:
            continue

        cal_window = pd.Timedelta(days=EVENT_WINDOW_DAYS * 1.5)
        window = parent_prices[
            (parent_prices["date"] >= eff_date - cal_window) &
            (parent_prices["date"] <= eff_date + cal_window)
        ].copy()

        all_td = trading_dates[
            (trading_dates >= eff_date - cal_window) &
            (trading_dates <= eff_date + cal_window)
        ]
        eff_idx = all_td.searchsorted(eff_date)
        td_map = {d: i - eff_idx for i, d in enumerate(all_td)}
        window["t"] = window["date"].map(td_map)

        window["parent_name"] = ev["parent_name"]
        window["parent_ticker"] = ev["parent_ticker"]
        window["spinoff_name"] = ev["spinoff_name"]
        window["spinoff_ticker"] = ev["spinoff_ticker"]
        window["effective_date"] = eff_date

        frames.append(window)

    if not frames:
        return pd.DataFrame()

    price_windows = pd.concat(frames, ignore_index=True)
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

    passive_aum = load_passive_aum()
    if passive_aum is None:
        print("  [!] sp500_passive_aum.parquet not found — run pull_etf_aum.py first")
        print("      Forced-flow estimates will be partial (index weight only, no AUM)")
    else:
        latest = passive_aum.sort_values("date").iloc[-1]
        print(f"  Passive AUM loaded: ${latest['total_aum_billions']:.0f}B "
              f"as of {latest['date'].date()}")

    # Optionally pull CRSP for parent permnos not in local file
    local_permnos = set(crsp["permno"].unique())
    event_permnos = set(events["permno"].dropna().astype(int))
    missing = sorted(event_permnos - local_permnos)

    if missing:
        print(f"\n  {len(missing)} parent PERMNOs not in local CRSP file")
        if pull_crsp:
            print("  Connecting to WRDS to pull missing prices...")
            new_crsp = pull_missing_crsp(missing)
            crsp = pd.concat([crsp, new_crsp], ignore_index=True)
            crsp.to_parquet(RAW_DIR / "crsp_daily.parquet")
            print(f"  Pulled and appended. Total CRSP rows: {len(crsp):,}")
        else:
            print("  Run with --pull-crsp to fetch them (needed for full forced-flow estimates).")
    else:
        print("  All parent PERMNOs present in local CRSP file.")

    print("\nBuilding events table...")
    events_merged = build_events_table(events, ccm, constituents)
    print(f"  {events_merged['gvkey'].notna().sum()}/{len(events_merged)} events matched to WRDS gvkey")
    print(f"  {events_merged['in_sp500_at_spinoff'].sum()}/{len(events_merged)} parents confirmed in S&P 500 at spinoff date")

    print("\nComputing forced-flow features...")
    events_merged = build_forced_flow_features(events_merged, crsp, passive_aum)
    has_ff = events_merged["forced_flow_usd"].notna().sum()
    has_weight = events_merged["parent_index_weight"].notna().sum()
    print(f"  Index weight computed for {has_weight}/{len(events_merged)} events")
    print(f"  Forced-flow (weight × AUM) computed for {has_ff}/{len(events_merged)} events")

    out_events = CLEAN_DIR / "spinoff_events_merged.csv"
    events_merged.to_csv(out_events, index=False)
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
        print("  No price data yet — run with --pull-crsp.")

    print("\nDone.")
    return events_merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pull-crsp", action="store_true",
        help="Connect to WRDS to pull CRSP prices for the 25 parent PERMNOs "
             "not yet in local crsp_daily.parquet",
    )
    args = parser.parse_args()
    result = main(pull_crsp=args.pull_crsp)
    print()
    ff_cols = ["parent_ticker", "effective_date", "parent_index_weight",
               "passive_aum_usd", "forced_flow_usd", "forced_flow_adv"]
    print(result[[c for c in ff_cols if c in result.columns]].to_string())
