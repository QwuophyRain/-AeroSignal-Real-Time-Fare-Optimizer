"""
notebooks/quickstart.py
────────────────────────
AeroSignal Quickstart — Interactive notebook-style script.
Run each section independently in PyCharm using Ctrl+Shift+E (Execute in Console)
or right-click any block → "Execute Selection in Python Console"

This file walks through the entire system:
  1. Generate data
  2. Demand analysis
  3. Flash fare computation
  4. ML training
  5. Live API calls
  6. Prediction inference
"""

# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 1 — Setup & Imports                            ║
# ╚══════════════════════════════════════════════════════════╝
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("✅ Imports OK — AeroSignal ready")
print(f"   Project root: {ROOT}")

# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 2 — Generate Sample Data                       ║
# ╚══════════════════════════════════════════════════════════╝
from scripts.generate_data import generate_all

generate_all()

# Peek at the data
df = pd.read_csv(ROOT / "data" / "kaggle_airline_demand.csv")
print(f"\n📊 Dataset shape: {df.shape}")
print(df.head(5))
print("\nDemand tier distribution:")
print(df['demand_tier'].value_counts())

# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 3 — Demand Analysis                            ║
# ╚══════════════════════════════════════════════════════════╝
from app.demand_engine import (
    load_search_log, expand_events, rolling_demand,
    flash_fare_table, revenue_leakage_report, route_performance_summary
)

raw    = load_search_log(str(ROOT / "data" / "sample_searches.csv"))
events = expand_events(raw)
demand = rolling_demand(events)
fares  = flash_fare_table(demand, base_fare=520.0, seats=42)

print(f"\n📈 Rolling demand windows: {len(demand)}")
print(demand[demand['demand_tier'] == 'SURGE'].head(5))

# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 4 — Revenue Report                             ║
# ╚══════════════════════════════════════════════════════════╝
report  = revenue_leakage_report(fares)
summary = route_performance_summary(fares)

print("\n💰 Revenue Leakage Report:")
for k, v in report.items():
    if k != 'detail':
        print(f"   {k:<35} {v}")

print("\n🗺  Route Performance:")
print(summary.to_string(index=False))

# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 5 — Quick Chart                                ║
# ╚══════════════════════════════════════════════════════════╝
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor('#0d1320')

# Revenue comparison
ax = axes[0]
ax.set_facecolor('#0d1320')
routes = summary['route'].tolist()
x = np.arange(len(routes))
ax.bar(x - 0.2, summary['avg_flash_fare'], 0.4, label='Avg Flash Fare', color='#00d4ff', alpha=0.8)
ax.bar(x + 0.2, [520]*len(routes), 0.4, label='Base Fare', color='#7c5cfc', alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(routes, color='#6b7a96')
ax.set_title('Flash Fare vs Base Fare by Route', color='#e8edf5')
ax.legend(facecolor='#141c2e', labelcolor='#e8edf5')
ax.tick_params(colors='#6b7a96')
for s in ax.spines.values(): s.set_color('#1e2a3a')

# Tier distribution
ax2 = axes[1]
ax2.set_facecolor('#0d1320')
tier_counts = fares['demand_tier'].value_counts()
colors = {'LOW': '#6bcb77', 'MEDIUM': '#ffd93d', 'HIGH': '#ff9f43', 'SURGE': '#ff4d6d'}
bars = ax2.bar(tier_counts.index, tier_counts.values,
               color=[colors.get(t, '#aaa') for t in tier_counts.index], alpha=0.85)
ax2.set_title('Demand Tier Distribution', color='#e8edf5')
ax2.tick_params(colors='#6b7a96')
for s in ax2.spines.values(): s.set_color('#1e2a3a')

plt.tight_layout()
plt.savefig(ROOT / "outputs" / "quickstart_charts.png", dpi=120, facecolor='#0d1320')
plt.show()
print("\n✅ Chart saved → outputs/quickstart_charts.png")

# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 6 — ML Training                                ║
# ╚══════════════════════════════════════════════════════════╝
from ml.pipeline import run_full_pipeline, predict_tier

model, features = run_full_pipeline(csv_path="data/kaggle_airline_demand.csv")
print(f"\n🤖 Trained on features: {features}")

# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 7 — Make Predictions                           ║
# ╚══════════════════════════════════════════════════════════╝
test_cases = [
    dict(spike_ratio=0.8,  load_factor=0.72, uplift_pct=7.9,  days_to_departure=45, searches=12),
    dict(spike_ratio=1.6,  load_factor=0.81, uplift_pct=14.5, days_to_departure=14, searches=55),
    dict(spike_ratio=2.4,  load_factor=0.88, uplift_pct=22.0, days_to_departure=7,  searches=90),
    dict(spike_ratio=8.0,  load_factor=0.94, uplift_pct=35.9, days_to_departure=2,  searches=210),
]

print("\n🔮 Demand Tier Predictions:")
print(f"  {'Spike':>6} {'Load':>6} {'DTD':>5} {'Predicted':>10} {'Confidence':>12}")
print(f"  {'─'*50}")
for tc in test_cases:
    result = predict_tier(**tc)
    print(f"  {tc['spike_ratio']:>6.1f}x {tc['load_factor']:>5.0%} {tc['days_to_departure']:>5}d "
          f"  {result['predicted_tier']:>9}   {result['confidence']:>10.1%}")

# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 8 — API Demo (start server first!)             ║
# ╚══════════════════════════════════════════════════════════╝
# NOTE: Start the API server first (run_all.py → option 1)
# Then run this block to make live API calls.

try:
    import httpx

    BASE = "http://localhost:8000"
    client = httpx.Client(timeout=5.0)

    # Record searches
    r = client.post(f"{BASE}/search/record", json={"route": "NYC-LON", "search_count": 150})
    print(f"\n📡 Recorded 150 searches: {r.json()['status']}")

    # Get flash fare
    r = client.post(f"{BASE}/fare/flash", json={
        "route": "NYC-LON", "base_fare_usd": 520.0, "seats_remaining": 42
    })
    rec = r.json()
    print(f"💲 Flash Fare: ${rec['recommendation']['flash_fare_usd']}")
    print(f"   Tier: {rec['demand']['demand_tier']}")
    print(f"   Action: {rec['action']}")

    client.close()

except Exception as e:
    print(f"\n⚠️  API not running — start server first with run_all.py option 1")
    print(f"   Error: {e}")

print("\n✅ Quickstart complete!")
