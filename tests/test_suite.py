"""
tests/test_suite.py
────────────────────
Full pytest test suite for AeroSignal.

Run in PyCharm:
    Right-click on tests/ folder → "Run pytest in tests"
    OR right-click this file → "Run 'pytest for test_suite'"

Run from terminal:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov-report=html
"""

import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.demand_engine import (
    load_search_log, expand_events, rolling_demand,
    flash_fare_table, revenue_leakage_report,
    route_performance_summary, _classify_tier_row
)


# ════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_events():
    """50 events for NYC-LON over the last 60 seconds."""
    now = datetime.utcnow()
    return pd.DataFrame({
        "timestamp": [now - timedelta(seconds=i) for i in range(50)],
        "route":     "NYC-LON",
    })


@pytest.fixture
def multi_route_events():
    """100 events each for 3 routes."""
    now = datetime.utcnow()
    rows = []
    for route in ["NYC-LON", "LAX-TYO", "CHI-PAR"]:
        for i in range(100):
            rows.append({"timestamp": now - timedelta(seconds=i), "route": route})
    return pd.DataFrame(rows)


@pytest.fixture
def sample_demand():
    """Pre-built demand DataFrame with all 4 tiers."""
    return pd.DataFrame({
        "route":        ["NYC-LON"] * 4,
        "window_start": [datetime.utcnow() - timedelta(minutes=i) for i in range(4)],
        "searches":     [10, 80, 200, 15],
        "baseline":     [10, 10, 10, 10],
        "spike_ratio":  [1.0, 2.0, 5.0, 1.0],
        "demand_tier":  ["LOW", "HIGH", "SURGE", "LOW"],
    })


# ════════════════════════════════════════════════════════════════
# TIER CLASSIFICATION
# ════════════════════════════════════════════════════════════════

class TestTierClassification:

    @pytest.mark.parametrize("searches,spike,expected", [
        (200, 4.0, "SURGE"),
        (150, 1.0, "SURGE"),  # volume threshold
        (90,  2.5, "HIGH"),
        (80,  1.0, "HIGH"),   # volume threshold
        (35,  1.5, "MEDIUM"),
        (30,  1.0, "MEDIUM"), # volume threshold
        (10,  1.0, "LOW"),
        (5,   0.8, "LOW"),
        (0,   0.0, "LOW"),
    ])
    def test_classify_tiers(self, searches, spike, expected):
        row = pd.Series({"searches": searches, "spike_ratio": spike})
        assert _classify_tier_row(row) == expected

    def test_surge_beats_searches_threshold(self):
        row = pd.Series({"searches": 5, "spike_ratio": 3.5})
        assert _classify_tier_row(row) == "SURGE"

    def test_high_spike_beats_searches_threshold(self):
        row = pd.Series({"searches": 5, "spike_ratio": 2.1})
        assert _classify_tier_row(row) == "HIGH"


# ════════════════════════════════════════════════════════════════
# ROLLING DEMAND
# ════════════════════════════════════════════════════════════════

class TestRollingDemand:

    def test_returns_dataframe(self, sample_events):
        result = rolling_demand(sample_events)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, sample_events):
        result = rolling_demand(sample_events)
        required = {"route", "window_start", "searches", "baseline", "spike_ratio", "demand_tier"}
        assert required.issubset(set(result.columns))

    def test_empty_input(self):
        empty = pd.DataFrame(columns=["timestamp", "route"])
        result = rolling_demand(empty)
        assert result.empty

    def test_multiple_routes(self, multi_route_events):
        result = rolling_demand(multi_route_events)
        assert set(result["route"].unique()) == {"NYC-LON", "LAX-TYO", "CHI-PAR"}

    def test_spike_ratio_positive(self, sample_events):
        result = rolling_demand(sample_events)
        assert (result["spike_ratio"] >= 0).all()

    def test_demand_tier_valid_values(self, sample_events):
        result = rolling_demand(sample_events)
        valid_tiers = {"LOW", "MEDIUM", "HIGH", "SURGE"}
        assert set(result["demand_tier"].unique()).issubset(valid_tiers)

    def test_baseline_clipped_at_one(self, sample_events):
        result = rolling_demand(sample_events)
        assert (result["baseline"] >= 1).all()


# ════════════════════════════════════════════════════════════════
# FLASH FARE TABLE
# ════════════════════════════════════════════════════════════════

class TestFlashFareTable:

    def test_output_shape(self, sample_demand):
        result = flash_fare_table(sample_demand, base_fare=500.0, seats=50)
        assert len(result) == 4

    def test_required_output_columns(self, sample_demand):
        result = flash_fare_table(sample_demand, base_fare=500.0, seats=50)
        assert "flash_fare" in result.columns
        assert "uplift_pct" in result.columns
        assert "window_min" in result.columns

    def test_surge_higher_than_low(self, sample_demand):
        result = flash_fare_table(sample_demand, base_fare=500.0, seats=42)
        surge_fare = result[result["demand_tier"] == "SURGE"]["flash_fare"].mean()
        low_fare   = result[result["demand_tier"] == "LOW"]["flash_fare"].mean()
        assert surge_fare > low_fare

    def test_surge_shorter_window_than_low(self, sample_demand):
        result = flash_fare_table(sample_demand, base_fare=500.0, seats=42)
        surge_win = result[result["demand_tier"] == "SURGE"]["window_min"].iloc[0]
        low_win   = result[result["demand_tier"] == "LOW"]["window_min"].iloc[0]
        assert surge_win < low_win

    def test_flash_fare_above_base(self, sample_demand):
        result = flash_fare_table(sample_demand, base_fare=500.0, seats=42)
        assert (result["flash_fare"] >= 500.0).all()

    def test_scarcity_increases_price(self, sample_demand):
        low_seats  = flash_fare_table(sample_demand, base_fare=500.0, seats=10)
        high_seats = flash_fare_table(sample_demand, base_fare=500.0, seats=190)
        # Same tier rows: low_seats should always price >= high_seats
        for tier in ["LOW", "MEDIUM", "HIGH", "SURGE"]:
            lf = low_seats[low_seats["demand_tier"] == tier]["flash_fare"].mean()
            hf = high_seats[high_seats["demand_tier"] == tier]["flash_fare"].mean()
            assert lf >= hf, f"Failed for tier {tier}: {lf} < {hf}"

    def test_base_fare_stored_correctly(self, sample_demand):
        result = flash_fare_table(sample_demand, base_fare=620.0, seats=42)
        assert (result["base_fare"] == 620.0).all()


# ════════════════════════════════════════════════════════════════
# REVENUE LEAKAGE REPORT
# ════════════════════════════════════════════════════════════════

class TestRevenueLeakageReport:

    @pytest.fixture
    def fare_table(self, sample_demand):
        return flash_fare_table(sample_demand, base_fare=500.0, seats=42)

    def test_has_required_keys(self, fare_table):
        report = revenue_leakage_report(fare_table)
        for key in ["high_surge_windows", "revenue_recovered_usd", "leakage_recovery_pct",
                    "total_static_revenue_usd", "total_optimized_revenue_usd"]:
            assert key in report

    def test_optimized_gte_static(self, fare_table):
        report = revenue_leakage_report(fare_table)
        if report["high_surge_windows"] > 0:
            assert report["total_optimized_revenue_usd"] >= report["total_static_revenue_usd"]

    def test_recovered_non_negative(self, fare_table):
        report = revenue_leakage_report(fare_table)
        assert report["revenue_recovered_usd"] >= 0

    def test_pct_non_negative(self, fare_table):
        report = revenue_leakage_report(fare_table)
        assert report["leakage_recovery_pct"] >= 0

    def test_detail_is_list(self, fare_table):
        report = revenue_leakage_report(fare_table)
        assert isinstance(report["detail"], list)

    def test_no_surge_returns_zeros(self, sample_demand):
        low_only = sample_demand[sample_demand["demand_tier"] == "LOW"].copy()
        fares    = flash_fare_table(low_only, base_fare=500.0, seats=42)
        report   = revenue_leakage_report(fares)
        assert report["high_surge_windows"] == 0


# ════════════════════════════════════════════════════════════════
# ROUTE PERFORMANCE SUMMARY
# ════════════════════════════════════════════════════════════════

class TestRoutePerformanceSummary:

    def test_returns_dataframe(self, sample_demand):
        fares   = flash_fare_table(sample_demand)
        summary = route_performance_summary(fares)
        assert isinstance(summary, pd.DataFrame)

    def test_has_route_column(self, sample_demand):
        fares   = flash_fare_table(sample_demand)
        summary = route_performance_summary(fares)
        assert "route" in summary.columns

    def test_surge_windows_counted(self, sample_demand):
        fares   = flash_fare_table(sample_demand)
        summary = route_performance_summary(fares)
        nyc     = summary[summary["route"] == "NYC-LON"].iloc[0]
        assert nyc["surge_windows"] >= 1   # we have one SURGE row in fixture


# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════

class TestDataLoading:

    def test_load_kaggle_csv(self):
        path = ROOT / "data" / "kaggle_airline_demand.csv"
        if path.exists():
            df = load_search_log(str(path))
            assert "route" in df.columns
            assert len(df) > 0

    def test_load_sample_searches(self):
        path = ROOT / "data" / "sample_searches.csv"
        if path.exists():
            df = load_search_log(str(path))
            assert len(df) > 0

    def test_expand_events_count(self):
        df = pd.DataFrame({
            "timestamp":    [datetime.utcnow()],
            "route":        ["NYC-LON"],
            "search_count": [5],
        })
        events = expand_events(df)
        assert len(events) == 5

    def test_expand_events_columns(self):
        df = pd.DataFrame({
            "timestamp":    [datetime.utcnow()],
            "route":        ["NYC-LON"],
            "search_count": [1],
        })
        events = expand_events(df)
        assert "timestamp" in events.columns
        assert "route" in events.columns


# ════════════════════════════════════════════════════════════════
# API TESTS (if FastAPI is available)
# ════════════════════════════════════════════════════════════════

try:
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    class TestAPI:

        def test_health_check(self):
            r = client.get("/")
            assert r.status_code == 200
            assert r.json()["status"] == "operational"

        def test_record_search(self):
            r = client.post("/search/record", json={"route": "NYC-LON", "search_count": 50})
            assert r.status_code == 200
            assert r.json()["searches_recorded"] == 50

        def test_demand_metrics(self):
            client.post("/search/record", json={"route": "TST-RTE", "search_count": 30})
            r = client.get("/demand/metrics/TST-RTE")
            assert r.status_code == 200
            data = r.json()
            assert "spike_ratio" in data
            assert "demand_tier" in data

        def test_flash_fare(self):
            client.post("/search/record", json={"route": "NYC-LON", "search_count": 100})
            r = client.post("/fare/flash", json={
                "route": "NYC-LON", "base_fare_usd": 520.0, "seats_remaining": 42
            })
            assert r.status_code == 200
            rec = r.json()["recommendation"]
            assert rec["flash_fare_usd"] >= 520.0
            assert "expires_at" in rec

        def test_simulate(self):
            r = client.post("/simulate", json={
                "route": "SIM-RTE", "searches_per_minute": 100, "duration_seconds": 30
            })
            assert r.status_code == 200
            assert r.json()["simulated_searches"] > 0

        def test_list_routes(self):
            r = client.get("/routes")
            assert r.status_code == 200
            assert "active_routes" in r.json()

        def test_flash_fare_surge_higher_than_low(self):
            """SURGE route should get higher fare than LOW route."""
            # Inject many searches → SURGE
            client.post("/search/record", json={"route": "HGH-DMD", "search_count": 500})
            r_high = client.post("/fare/flash", json={
                "route": "HGH-DMD", "base_fare_usd": 500.0, "seats_remaining": 10
            })
            # Fresh route with no searches → LOW
            r_low = client.post("/fare/flash", json={
                "route": "LOW-DMD", "base_fare_usd": 500.0, "seats_remaining": 180
            })
            high_fare = r_high.json()["recommendation"]["flash_fare_usd"]
            low_fare  = r_low.json()["recommendation"]["flash_fare_usd"]
            assert high_fare > low_fare

except ImportError:
    pass  # FastAPI / httpx not installed — skip API tests
