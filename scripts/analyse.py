"""
scripts/analyse.py
───────────────────
Batch Demand Analysis — load a CSV, run the full demand + fare engine,
print a rich report, and save outputs.

Run directly in PyCharm:
    Right-click → Run 'analyse'

Or via run_all.py:
    python run_all.py  → choose option 6
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from app.demand_engine import (
    load_search_log, expand_events, rolling_demand,
    flash_fare_table, revenue_leakage_report, route_performance_summary
)

# ── Config — edit these to change the analysis ────────────────────────────────
DEFAULT_CSV   = "data/kaggle_airline_demand.csv"
DEFAULT_FARE  = 520.0
DEFAULT_SEATS = 42
DEFAULT_CONV  = 0.03   # 3% conversion rate


def run_analysis(
    csv_path:    str   = DEFAULT_CSV,
    base_fare:   float = DEFAULT_FARE,
    seats:       int   = DEFAULT_SEATS,
    conv_rate:   float = DEFAULT_CONV,
    save_output: bool  = True,
) -> dict:
    """
    Full batch analysis pipeline.

    Returns a dict with:
        fares     — flash fare DataFrame
        report    — revenue leakage report dict
        summary   — route performance DataFrame
    """
    path = ROOT / csv_path if not Path(csv_path).is_absolute() else Path(csv_path)

    print(f"\n{'═'*60}")
    print(f"  AeroSignal — Batch Demand Analysis")
    print(f"{'═'*60}")
    print(f"  Dataset  : {path.name}")
    print(f"  Base fare: ${base_fare:,.2f}")
    print(f"  Seats    : {seats}")
    print(f"  Conv rate: {conv_rate:.1%}")
    print(f"{'═'*60}\n")

    # ── Load & process ────────────────────────────────────────────
    print("📂  Loading dataset...")
    raw = load_search_log(str(path))
    print(f"    {len(raw):,} rows | {raw['route'].nunique()} routes\n")

    print("🔄  Computing rolling demand windows...")
    events = expand_events(raw)
    demand = rolling_demand(events)
    print(f"    {len(demand):,} time windows\n")

    print("💲  Applying Flash Fare algorithm...")
    fares = flash_fare_table(demand, base_fare=base_fare, seats=seats)

    print("📊  Computing revenue leakage report...")
    report  = revenue_leakage_report(fares, conversion_rate=conv_rate)
    summary = route_performance_summary(fares)

    # ── Print report ──────────────────────────────────────────────
    _print_report(fares, report, summary)

    # ── Save outputs ──────────────────────────────────────────────
    if save_output:
        out_dir = ROOT / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        fares_path   = out_dir / "flash_fare_analysis.csv"
        summary_path = out_dir / "route_summary.csv"

        fares.to_csv(fares_path, index=False)
        summary.to_csv(summary_path, index=False)
        print(f"\n✅  Saved → {fares_path.name}")
        print(f"✅  Saved → {summary_path.name}")

    return {"fares": fares, "report": report, "summary": summary}


def _print_report(fares: pd.DataFrame, report: dict, summary: pd.DataFrame):
    """Print a formatted terminal report."""
    tier_counts = fares['demand_tier'].value_counts()

    print(f"\n{'─'*60}")
    print(f"  Demand Tier Distribution")
    print(f"{'─'*60}")
    tier_colors = {'SURGE': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
    for tier in ['SURGE', 'HIGH', 'MEDIUM', 'LOW']:
        count = tier_counts.get(tier, 0)
        pct   = count / len(fares) * 100
        bar   = '█' * int(pct / 2)
        print(f"  {tier_colors[tier]} {tier:<8} {count:>5}  {bar:<30} {pct:.1f}%")

    print(f"\n{'─'*60}")
    print(f"  Revenue Leakage Report")
    print(f"{'─'*60}")
    print(f"  High/Surge windows     : {report['high_surge_windows']:>10,}")
    print(f"  Static pricing revenue : ${report['total_static_revenue_usd']:>12,.2f}")
    print(f"  Flash fare revenue     : ${report['total_optimized_revenue_usd']:>12,.2f}")
    print(f"  Revenue recovered      : ${report['revenue_recovered_usd']:>12,.2f}  ← captured leakage")
    print(f"  Recovery %             : {report['leakage_recovery_pct']:>10.1f}%")

    print(f"\n{'─'*60}")
    print(f"  Route Performance Summary")
    print(f"{'─'*60}")
    print(summary.to_string(index=False))

    # Top 5 surge windows
    surges = fares[fares['demand_tier'] == 'SURGE'].nlargest(5, 'spike_ratio')
    if not surges.empty:
        print(f"\n{'─'*60}")
        print(f"  Top Surge Events (by spike ratio)")
        print(f"{'─'*60}")
        print(surges[['route', 'window_start', 'searches', 'spike_ratio', 'flash_fare', 'uplift_pct']].to_string(index=False))

    print(f"\n{'═'*60}\n")


# ── PyCharm Direct Run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_analysis()
