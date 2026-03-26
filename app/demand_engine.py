"""
app/demand_engine.py
─────────────────────
Standalone Pandas analytics layer for demand spike detection.

Run directly in PyCharm:
    Right-click → Run 'demand_engine'

Or via CLI:
    python -m app.demand_engine --csv data/kaggle_airline_demand.csv --fare 520 --seats 42
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════

def load_search_log(path: str) -> pd.DataFrame:
    """
    Load a search log CSV.

    Supported formats:
      - sample_searches.csv       → columns: timestamp, route, search_count
      - kaggle_airline_demand.csv → columns: route, demand_tier, searches, ...
      - flash_fare_report.csv     → pre-computed fare table
      - Any CSV with route + searches columns
    """
    df = pd.read_csv(path)

    # Auto-detect timestamp column
    ts_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower() or 'window' in c.lower()]
    if ts_cols:
        df[ts_cols[0]] = pd.to_datetime(df[ts_cols[0]], errors='coerce')
        df = df.rename(columns={ts_cols[0]: 'timestamp'})
    else:
        # No timestamp — synthesize one
        df['timestamp'] = pd.date_range('2025-06-15 06:00:00', periods=len(df), freq='1min')

    # Auto-detect search count column
    search_cols = [c for c in df.columns if 'search' in c.lower() and 'count' in c.lower() or c == 'searches']
    if search_cols and 'search_count' not in df.columns:
        df = df.rename(columns={search_cols[0]: 'search_count'})
    elif 'searches' in df.columns and 'search_count' not in df.columns:
        df = df.rename(columns={'searches': 'search_count'})

    if 'search_count' not in df.columns:
        df['search_count'] = 10  # fallback

    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def load_kaggle_dataset(path: str) -> pd.DataFrame:
    """
    Load the enriched Kaggle-style dataset (kaggle_airline_demand.csv).
    Handles the full schema with demand_tier, flash_fare, load_factor, etc.
    """
    df = pd.read_csv(path)

    # Parse any datetime columns
    for col in df.columns:
        if 'time' in col.lower() or 'date' in col.lower() or 'window' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass

    return df


# ════════════════════════════════════════════════════════════════
# CORE ANALYTICS
# ════════════════════════════════════════════════════════════════

def expand_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand batch rows (timestamp, route, search_count) into
    individual search event rows for fine-grained rolling analysis.
    """
    rows = []
    for _, row in df.iterrows():
        count = int(row.get('search_count', 1))
        for _ in range(count):
            rows.append({"timestamp": row['timestamp'], "route": row['route']})
    return pd.DataFrame(rows)


def rolling_demand(events: pd.DataFrame, window_sec: int = 60) -> pd.DataFrame:
    """
    Compute demand metrics per route using Pandas resampling.

    Algorithm:
        1. Resample events into 1-minute bins → searches per bin
        2. Compute 5-period rolling baseline (prior 5 minutes)
        3. spike_ratio = current_bin / baseline
        4. Classify into demand tier

    Returns a tidy DataFrame with columns:
        route, window_start, searches, baseline, spike_ratio, demand_tier
    """
    if events.empty:
        return pd.DataFrame(columns=['route', 'window_start', 'searches', 'baseline', 'spike_ratio', 'demand_tier'])

    results = []
    for route, grp in events.groupby("route"):
        grp = grp.sort_values("timestamp").copy()
        grp = grp.set_index("timestamp")
        grp["count"] = 1

        # 1-minute bins
        resampled = grp["count"].resample("1min").sum().fillna(0).reset_index()
        resampled.columns = ["window_start", "searches"]

        # Rolling 5-minute baseline
        resampled["baseline"] = (
            resampled["searches"]
            .shift(1)
            .rolling(5, min_periods=1)
            .mean()
            .fillna(1)
        )
        resampled["baseline"]    = resampled["baseline"].clip(lower=1)
        resampled["spike_ratio"] = (resampled["searches"] / resampled["baseline"]).round(2)
        resampled["demand_tier"] = resampled.apply(_classify_tier_row, axis=1)
        resampled["route"]       = route
        results.append(resampled)

    combined = pd.concat(results, ignore_index=True)
    return combined[["route", "window_start", "searches", "baseline", "spike_ratio", "demand_tier"]]


def _classify_tier_row(row) -> str:
    if row["spike_ratio"] >= 3.0 or row["searches"] >= 150:
        return "SURGE"
    if row["spike_ratio"] >= 2.0 or row["searches"] >= 80:
        return "HIGH"
    if row["spike_ratio"] >= 1.3 or row["searches"] >= 30:
        return "MEDIUM"
    return "LOW"


# ════════════════════════════════════════════════════════════════
# FLASH FARE ENGINE
# ════════════════════════════════════════════════════════════════

def flash_fare_table(
    demand_df:  pd.DataFrame,
    base_fare:  float = 520.0,
    seats:      int   = 42,
) -> pd.DataFrame:
    """
    Apply the Flash Fare algorithm across all rows of the demand DataFrame.

    Pricing Formula:
        total_mult  = demand_mult + spike_premium + scarcity_premium
        flash_fare  = base_fare × total_mult

    Multiplier components:
        demand_mult      LOW=1.00 / MEDIUM=1.05 / HIGH=1.12 / SURGE=1.20
        spike_premium    min((spike_ratio-1) × 0.03,  0.08)
        scarcity_premium (1 - seats/200) × 0.10

    Flash window (urgency):
        LOW=30 min / MEDIUM=20 min / HIGH=12 min / SURGE=7 min
    """
    TIER_MULT   = {"LOW": 1.00, "MEDIUM": 1.05, "HIGH": 1.12, "SURGE": 1.20}
    TIER_WINDOW = {"LOW": 30,   "MEDIUM": 20,   "HIGH": 12,   "SURGE": 7}

    df           = demand_df.copy()
    seats_factor = max(0.0, 1 - seats / 200)

    df["demand_mult"]    = df["demand_tier"].map(TIER_MULT)
    df["spike_premium"]  = df["spike_ratio"].apply(
        lambda s: min((s - 1) * 0.03, 0.08) if s > 1 else 0.0
    )
    df["scarcity_prem"]  = seats_factor * 0.10
    df["total_mult"]     = (df["demand_mult"] + df["spike_premium"] + df["scarcity_prem"]).round(4)
    df["flash_fare"]     = (base_fare * df["total_mult"]).round(2)
    df["uplift_pct"]     = ((df["flash_fare"] - base_fare) / base_fare * 100).round(1)
    df["window_min"]     = df["demand_tier"].map(TIER_WINDOW)
    df["base_fare"]      = base_fare

    return df[[
        "route", "window_start", "searches", "spike_ratio", "demand_tier",
        "base_fare", "flash_fare", "uplift_pct", "window_min"
    ]]


# ════════════════════════════════════════════════════════════════
# REVENUE ANALYTICS
# ════════════════════════════════════════════════════════════════

def revenue_leakage_report(fare_df: pd.DataFrame, conversion_rate: float = 0.03) -> dict:
    """
    Estimate revenue leakage recovered by Flash Fare vs static pricing.

    Assumption: each search has a `conversion_rate` chance of booking.

    Returns:
        high_surge_windows        — count of HIGH/SURGE windows
        total_static_revenue_usd  — revenue if static pricing was used
        total_optimized_revenue   — revenue with Flash Fare
        revenue_recovered_usd     — difference (recovered leakage)
        leakage_recovery_pct      — % improvement
        detail                    — per-row detail DataFrame as records
    """
    surge_mask = fare_df["demand_tier"].isin(["HIGH", "SURGE"])
    missed     = fare_df[surge_mask].copy()

    missed["potential_bookings"] = (missed["searches"] * conversion_rate).round(1)
    missed["static_revenue"]     = missed["potential_bookings"] * missed["base_fare"]
    missed["optimized_revenue"]  = missed["potential_bookings"] * missed["flash_fare"]
    missed["leakage_recovered"]  = missed["optimized_revenue"] - missed["static_revenue"]

    total_static    = missed["static_revenue"].sum()
    total_optimized = missed["optimized_revenue"].sum()
    total_recovered = missed["leakage_recovered"].sum()
    leakage_pct     = (total_recovered / total_static * 100) if total_static > 0 else 0.0

    return {
        "high_surge_windows":        int(surge_mask.sum()),
        "total_static_revenue_usd":  round(total_static, 2),
        "total_optimized_revenue_usd": round(total_optimized, 2),
        "revenue_recovered_usd":     round(total_recovered, 2),
        "leakage_recovery_pct":      round(leakage_pct, 1),
        "detail":                    missed.to_dict(orient="records"),
    }


def route_performance_summary(fare_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fare metrics by route for dashboard KPIs."""
    return (
        fare_df.groupby("route")
        .agg(
            total_searches = ("searches", "sum"),
            avg_spike       = ("spike_ratio", "mean"),
            surge_windows   = ("demand_tier", lambda x: (x == "SURGE").sum()),
            avg_flash_fare  = ("flash_fare", "mean"),
            avg_uplift      = ("uplift_pct", "mean"),
            peak_spike      = ("spike_ratio", "max"),
        )
        .round(2)
        .reset_index()
        .sort_values("total_searches", ascending=False)
    )


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AeroSignal Demand Engine — Batch CSV Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.demand_engine --csv data/sample_searches.csv
  python -m app.demand_engine --csv data/kaggle_airline_demand.csv --fare 620 --seats 30
  python -m app.demand_engine --csv data/my_data.csv --out results/fare_report.csv
        """
    )
    parser.add_argument("--csv",   default="data/kaggle_airline_demand.csv", help="Input CSV path")
    parser.add_argument("--fare",  type=float, default=520.0,                help="Base fare in USD")
    parser.add_argument("--seats", type=int,   default=42,                   help="Seats remaining")
    parser.add_argument("--out",   default="data/flash_fare_report.csv",     help="Output CSV path")
    parser.add_argument("--conv",  type=float, default=0.03,                 help="Conversion rate (default 0.03)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  AeroSignal Demand Engine")
    print(f"{'='*60}")
    print(f"  Input  : {args.csv}")
    print(f"  Fare   : ${args.fare:.2f}")
    print(f"  Seats  : {args.seats}")
    print(f"  Output : {args.out}")
    print(f"{'='*60}\n")

    print("📂  Loading data...")
    raw  = load_search_log(args.csv)
    print(f"    {len(raw):,} rows loaded, {raw['route'].nunique()} routes\n")

    print("🔄  Expanding events & computing rolling demand...")
    events = expand_events(raw)
    demand = rolling_demand(events)
    print(f"    {len(demand):,} time windows computed\n")

    print("💲  Applying Flash Fare algorithm...")
    fares = flash_fare_table(demand, base_fare=args.fare, seats=args.seats)

    print("📊  Generating revenue leakage report...")
    report = revenue_leakage_report(fares, conversion_rate=args.conv)

    fares.to_csv(args.out, index=False)
    print(f"\n✅  Flash fare table saved → {args.out}")

    print(f"\n{'─'*50}")
    print(f"  Revenue Leakage Report")
    print(f"{'─'*50}")
    print(f"  High/Surge windows   : {report['high_surge_windows']:>8,}")
    print(f"  Static pricing rev   : ${report['total_static_revenue_usd']:>12,.2f}")
    print(f"  Optimized rev        : ${report['total_optimized_revenue_usd']:>12,.2f}")
    print(f"  Revenue recovered    : ${report['revenue_recovered_usd']:>12,.2f}")
    print(f"  Leakage recovery     : {report['leakage_recovery_pct']:>7.1f}%")
    print(f"{'─'*50}\n")

    print("  Route Performance Summary:")
    summary = route_performance_summary(fares)
    print(summary.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
