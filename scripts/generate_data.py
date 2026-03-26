"""
scripts/generate_data.py
─────────────────────────
Generate all sample datasets for AeroSignal.

Run directly in PyCharm:
    Right-click → Run 'generate_data'

Or via run_all.py:
    python run_all.py  → choose option 5
"""

import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)


def generate_search_log(
    routes: list[str] = None,
    n_minutes: int = 120,
    seed: int = 42,
    output_path: str = "data/sample_searches.csv",
) -> pd.DataFrame:
    """
    Generate a realistic search log CSV with demand spikes injected at known windows.

    Columns: timestamp, route, search_count
    """
    random.seed(seed)
    np.random.seed(seed)

    if routes is None:
        routes = ["NYC-LON", "LAX-TYO", "CHI-PAR", "MIA-MAD", "SFO-SYD"]

    start = pd.Timestamp("2025-06-15 06:00:00")
    rows  = []

    for route in routes:
        base_load = random.randint(10, 25)
        for minute in range(n_minutes):
            ts = start + pd.Timedelta(minutes=minute)
            # Inject 3 demand spikes per route
            if 20 <= minute <= 25:     count = random.randint(120, 200)
            elif 55 <= minute <= 60:   count = random.randint(60, 100)
            elif 90 <= minute <= 95:   count = random.randint(180, 280)
            else:                      count = random.randint(base_load - 5, base_load + 10)
            rows.append({"timestamp": ts.isoformat(), "route": route, "search_count": max(1, count)})

    df = pd.DataFrame(rows)
    path = ROOT / output_path
    df.to_csv(path, index=False)
    print(f"  ✅ sample_searches.csv  →  {len(df):,} rows  ({df['route'].nunique()} routes)")
    return df


def generate_kaggle_dataset(
    routes: list[str] = None,
    n_minutes: int = 120,
    base_fare: float = 520.0,
    seed: int = 42,
    output_path: str = "data/kaggle_airline_demand.csv",
) -> pd.DataFrame:
    """
    Generate the enriched Kaggle-style airline demand dataset.

    Columns:
        route, window_start, searches, spike_ratio, demand_tier,
        base_fare, flash_fare, uplift_pct, window_min,
        conversion_rate, bookings, static_revenue, flash_revenue,
        revenue_recovered, competitor_price, price_advantage,
        load_factor, days_to_departure
    """
    random.seed(seed)
    np.random.seed(seed)

    if routes is None:
        routes = ["NYC-LON", "LAX-TYO", "CHI-PAR", "MIA-MAD", "SFO-SYD"]

    # First generate search log, then compute demand metrics on it
    search_df = generate_search_log(routes=routes, n_minutes=n_minutes, seed=seed,
                                    output_path="data/sample_searches.csv")

    from app.demand_engine import load_search_log, expand_events, rolling_demand, flash_fare_table
    raw    = load_search_log(str(ROOT / "data" / "sample_searches.csv"))
    events = expand_events(raw)
    demand = rolling_demand(events)
    fares  = flash_fare_table(demand, base_fare=base_fare, seats=42)

    # Enrich with Kaggle-style columns
    n = len(fares)
    tier_conv = {'LOW': 0.02, 'MEDIUM': 0.04, 'HIGH': 0.07, 'SURGE': 0.11}

    fares['conversion_rate']   = fares['demand_tier'].map(tier_conv) + np.random.normal(0, 0.005, n)
    fares['conversion_rate']   = fares['conversion_rate'].clip(0.01, 0.15).round(4)
    fares['bookings']          = (fares['searches'] * fares['conversion_rate']).round().astype(int)
    fares['static_revenue']    = fares['bookings'] * fares['base_fare']
    fares['flash_revenue']     = fares['bookings'] * fares['flash_fare']
    fares['revenue_recovered'] = fares['flash_revenue'] - fares['static_revenue']
    fares['competitor_price']  = (fares['base_fare'] * np.random.uniform(0.95, 1.15, n)).round(2)
    fares['price_advantage']   = ((fares['flash_fare'] - fares['competitor_price']) / fares['competitor_price'] * 100).round(2)
    fares['load_factor']       = np.random.uniform(0.55, 0.97, n).round(3)
    fares['days_to_departure'] = np.random.randint(1, 90, n)

    path = ROOT / output_path
    fares.to_csv(path, index=False)
    print(f"  ✅ kaggle_airline_demand.csv  →  {len(fares):,} rows")
    return fares


def generate_flash_fare_report(output_path: str = "data/flash_fare_report.csv") -> pd.DataFrame:
    """Re-generate flash fare report from the Kaggle dataset."""
    from app.demand_engine import load_search_log, expand_events, rolling_demand, flash_fare_table

    raw    = load_search_log(str(ROOT / "data" / "sample_searches.csv"))
    events = expand_events(raw)
    demand = rolling_demand(events)
    fares  = flash_fare_table(demand, base_fare=520.0, seats=42)

    path = ROOT / output_path
    fares.to_csv(path, index=False)
    print(f"  ✅ flash_fare_report.csv  →  {len(fares):,} rows")
    return fares


def generate_all(routes: list[str] = None, seed: int = 42):
    """Generate all datasets in the correct order."""
    print(f"\n{'='*55}")
    print(f"  AeroSignal — Dataset Generator")
    print(f"{'='*55}\n")
    generate_kaggle_dataset(routes=routes, seed=seed)
    generate_flash_fare_report()
    print(f"\n  All datasets saved to: {DATA}\n")


# ── PyCharm Direct Run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_all()
