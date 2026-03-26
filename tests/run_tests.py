"""
tests/run_tests.py
───────────────────
Standalone test runner — works without pytest installed.

Run in PyCharm: Right-click → Run 'run_tests'
Or: python tests/run_tests.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from datetime import datetime, timedelta, timezone
import pandas as pd

from app.demand_engine import (
    expand_events, rolling_demand, flash_fare_table,
    revenue_leakage_report, route_performance_summary, _classify_tier_row
)

# ── Assertion helpers ─────────────────────────────────────────────────────────
def eq(a, b):    assert a == b,  f"Expected {b!r}, got {a!r}"
def gt(a, b):    assert a > b,   f"Expected {a} > {b}"
def lt(a, b):    assert a < b,   f"Expected {a} < {b}"
def gte(a, b):   assert a >= b,  f"Expected {a} >= {b}"
def all_gte(s, v): assert (s >= v).all(), f"Not all values >= {v}"
def all_eq(s, v):  assert (s == v).all(), f"Not all values == {v}"
def has_type(x, t): assert isinstance(x, t), f"Expected {t.__name__}, got {type(x).__name__}"
def has_col(df, c): assert c in df.columns, f"Missing column: {c!r}"
def has_key(d, k):  assert k in d, f"Missing key: {k!r}"

# ── Test runner ───────────────────────────────────────────────────────────────
results = []

def test(name, fn):
    try:
        fn()
        results.append(("PASS", name, ""))
    except AssertionError as e:
        results.append(("FAIL", name, str(e)))
    except Exception as e:
        results.append(("ERROR", name, f"{type(e).__name__}: {e}"))

# ── Fixtures ─────────────────────────────────────────────────────────────────
now = datetime.now(timezone.utc).replace(tzinfo=None)

demand_fixture = pd.DataFrame({
    'route':        ['NYC-LON'] * 4,
    'window_start': [now - timedelta(minutes=i) for i in range(4)],
    'searches':     [10, 80, 200, 15],
    'baseline':     [10, 10, 10, 10],
    'spike_ratio':  [1.0, 2.0, 5.0, 1.0],
    'demand_tier':  ['LOW', 'HIGH', 'SURGE', 'LOW'],
})

# ════════════════════════════════════════════════════════════════
# TIER CLASSIFICATION
# ════════════════════════════════════════════════════════════════
def r(searches, spike): return pd.Series({'searches': searches, 'spike_ratio': spike})

test("SURGE by spike ≥3.0",      lambda: eq(_classify_tier_row(r(5, 4.0)), 'SURGE'))
test("SURGE by volume ≥150",      lambda: eq(_classify_tier_row(r(200, 1.0)), 'SURGE'))
test("HIGH by spike ≥2.0",        lambda: eq(_classify_tier_row(r(5, 2.2)), 'HIGH'))
test("HIGH by volume ≥80",        lambda: eq(_classify_tier_row(r(90, 1.0)), 'HIGH'))
test("MEDIUM by spike ≥1.3",      lambda: eq(_classify_tier_row(r(5, 1.5)), 'MEDIUM'))
test("MEDIUM by volume ≥30",      lambda: eq(_classify_tier_row(r(35, 1.0)), 'MEDIUM'))
test("LOW default",                lambda: eq(_classify_tier_row(r(5, 0.9)), 'LOW'))
test("LOW zero searches",          lambda: eq(_classify_tier_row(r(0, 0.0)), 'LOW'))

# ════════════════════════════════════════════════════════════════
# FLASH FARE TABLE
# ════════════════════════════════════════════════════════════════
fares = flash_fare_table(demand_fixture, base_fare=500.0, seats=42)

test("Output row count matches",   lambda: eq(len(fares), 4))
test("Required columns present",   lambda: [has_col(fares, c) for c in ['flash_fare','uplift_pct','window_min','base_fare']])
test("Flash fare >= base fare",    lambda: all_gte(fares['flash_fare'], 500.0))
test("SURGE fare > LOW fare",      lambda: gt(
    fares[fares['demand_tier']=='SURGE']['flash_fare'].mean(),
    fares[fares['demand_tier']=='LOW']['flash_fare'].mean()
))
test("SURGE window < LOW window",  lambda: lt(
    fares[fares['demand_tier']=='SURGE']['window_min'].iloc[0],
    fares[fares['demand_tier']=='LOW']['window_min'].iloc[0]
))
test("Base fare stored correctly", lambda: all_eq(
    flash_fare_table(demand_fixture, base_fare=620.0, seats=42)['base_fare'], 620.0
))
test("Uplift_pct ≥ 0",            lambda: all_gte(fares['uplift_pct'], 0.0))

def _scarcity_test():
    lo = flash_fare_table(demand_fixture, base_fare=500.0, seats=10)
    hi = flash_fare_table(demand_fixture, base_fare=500.0, seats=190)
    for tier in ['LOW', 'HIGH', 'SURGE']:
        lf = lo[lo['demand_tier']==tier]['flash_fare'].mean()
        hf = hi[hi['demand_tier']==tier]['flash_fare'].mean()
        gte(lf, hf)  # fewer seats → higher price

test("Scarcity raises price",      _scarcity_test)

# ════════════════════════════════════════════════════════════════
# REVENUE LEAKAGE REPORT
# ════════════════════════════════════════════════════════════════
report = revenue_leakage_report(fares)

test("Required keys present",      lambda: [has_key(report, k) for k in [
    'high_surge_windows', 'revenue_recovered_usd', 'leakage_recovery_pct',
    'total_static_revenue_usd', 'total_optimized_revenue_usd', 'detail'
]])
test("Optimized ≥ static revenue", lambda: gte(
    report['total_optimized_revenue_usd'], report['total_static_revenue_usd']
))
test("Revenue recovered ≥ 0",     lambda: gte(report['revenue_recovered_usd'], 0))
test("Leakage pct ≥ 0",           lambda: gte(report['leakage_recovery_pct'], 0))
test("Detail is a list",           lambda: has_type(report['detail'], list))

def _low_only_test():
    low_only = demand_fixture[demand_fixture['demand_tier']=='LOW'].copy()
    low_fares = flash_fare_table(low_only, base_fare=500.0, seats=42)
    r = revenue_leakage_report(low_fares)
    eq(r['high_surge_windows'], 0)

test("No HIGH/SURGE → 0 windows", _low_only_test)

# ════════════════════════════════════════════════════════════════
# ROUTE PERFORMANCE SUMMARY
# ════════════════════════════════════════════════════════════════
summary = route_performance_summary(fares)

test("Returns DataFrame",          lambda: has_type(summary, pd.DataFrame))
test("Has route column",           lambda: has_col(summary, 'route'))
test("Has surge_windows column",   lambda: has_col(summary, 'surge_windows'))
test("Surge windows > 0",          lambda: gt(
    summary[summary['route']=='NYC-LON']['surge_windows'].iloc[0], 0
))

# ════════════════════════════════════════════════════════════════
# EXPAND EVENTS
# ════════════════════════════════════════════════════════════════
df_batch = pd.DataFrame({'timestamp': [now], 'route': ['NYC-LON'], 'search_count': [7]})
events = expand_events(df_batch)

test("Expand: correct row count",  lambda: eq(len(events), 7))
test("Expand: has timestamp col",  lambda: has_col(events, 'timestamp'))
test("Expand: has route col",      lambda: has_col(events, 'route'))

# ════════════════════════════════════════════════════════════════
# ROLLING DEMAND
# ════════════════════════════════════════════════════════════════
test_events = pd.DataFrame({
    'timestamp': [now - timedelta(seconds=i) for i in range(50)],
    'route': 'NYC-LON'
})

rolling = rolling_demand(test_events)
test("Rolling: returns DataFrame",  lambda: has_type(rolling, pd.DataFrame))
test("Rolling: required columns",   lambda: [has_col(rolling, c) for c in
    ['route', 'window_start', 'searches', 'baseline', 'spike_ratio', 'demand_tier']])
test("Rolling: spike_ratio ≥ 0",    lambda: all_gte(rolling['spike_ratio'], 0))
test("Rolling: baseline ≥ 1",       lambda: all_gte(rolling['baseline'], 1))
test("Rolling: empty input → empty",lambda: eq(len(rolling_demand(pd.DataFrame(columns=['timestamp','route']))), 0))

# ════════════════════════════════════════════════════════════════
# KAGGLE CSV LOADING
# ════════════════════════════════════════════════════════════════
from app.demand_engine import load_search_log

for csv_name in ['kaggle_airline_demand.csv', 'sample_searches.csv', 'flash_fare_report.csv']:
    path = ROOT / 'data' / csv_name
    if path.exists():
        test(f"Load {csv_name}", lambda p=path: gt(len(load_search_log(str(p))), 0))
    else:
        test(f"Load {csv_name} (skip — not found)", lambda: None)

# ════════════════════════════════════════════════════════════════
# RESULTS
# ════════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  AeroSignal Test Results")
print(f"{'═'*60}")

pass_count  = sum(1 for s,_,_ in results if s == "PASS")
fail_count  = sum(1 for s,_,_ in results if s in ("FAIL","ERROR"))

for status, name, msg in results:
    icon = "✅" if status == "PASS" else "❌"
    line = f"  {icon} {name}"
    if msg:
        line += f"\n       ↳ {msg}"
    print(line)

print(f"\n{'─'*60}")
print(f"  {pass_count} passed  |  {fail_count} failed  |  {len(results)} total")
print(f"{'═'*60}\n")

if fail_count > 0:
    sys.exit(1)
