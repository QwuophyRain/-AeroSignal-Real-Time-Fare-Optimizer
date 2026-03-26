"""
app/main.py
───────────
Real-Time Fare Optimizer — FastAPI Application

Run directly in PyCharm:
    Right-click → Run 'main'  (runs the __main__ block at the bottom)

Or via uvicorn from terminal:
    uvicorn app.main:app --reload --port 8000

Swagger UI → http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import random
from collections import deque
import threading

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="✈️ AeroSignal Fare Optimizer API",
    description=(
        "Captures latent demand from shoppers who search but don't book — "
        "converting demand spikes into Flash Fares before revenue leaks to competitors."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Search Event Store ────────────────────────────────────────────────────────
SEARCH_WINDOW_SECONDS = 60
MAX_EVENTS_PER_ROUTE  = 5000


class SearchStore:
    """Thread-safe rolling window of search events, keyed by route."""

    def __init__(self):
        self._lock   = threading.Lock()
        self._events: dict[str, deque] = {}
        self._total_recorded: int = 0

    def record(self, route: str, count: int = 1) -> None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with self._lock:
            if route not in self._events:
                self._events[route] = deque(maxlen=MAX_EVENTS_PER_ROUTE)
            for _ in range(count):
                self._events[route].append(now)
            self._total_recorded += count

    def get_dataframe(self, route: str) -> pd.DataFrame:
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=SEARCH_WINDOW_SECONDS * 10)
        with self._lock:
            events = list(self._events.get(route, []))
        if not events:
            return pd.DataFrame(columns=["timestamp"])
        df = pd.DataFrame({"timestamp": events})
        df = df[df["timestamp"] >= cutoff].copy()
        return df.sort_values("timestamp").reset_index(drop=True)

    def all_routes(self) -> list[str]:
        with self._lock:
            return list(self._events.keys())

    @property
    def total_recorded(self) -> int:
        with self._lock:
            return self._total_recorded


store = SearchStore()


# ── Demand Analytics (Pandas Core) ───────────────────────────────────────────

def compute_demand_metrics(route: str) -> dict:
    """
    Compute real-time demand metrics for a route using a rolling Pandas window.

    Returns:
        searches_last_60s  — raw search count in last 60 seconds
        baseline_per_min   — 5-minute rolling average (expected baseline)
        spike_ratio        — current / baseline (>1 = above normal)
        velocity_per_sec   — searches/second trend (linear regression slope)
        demand_tier        — LOW | MEDIUM | HIGH | SURGE
    """
    df  = store.get_dataframe(route)
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    window_60s  = now - timedelta(seconds=60)
    window_5min = now - timedelta(seconds=300)

    searches_60s  = int((df["timestamp"] >= window_60s).sum())
    searches_5min = int((df["timestamp"] >= window_5min).sum())

    baseline_per_min = max(searches_5min / 5, 1)
    spike_ratio      = round(searches_60s / baseline_per_min, 2)

    # Linear regression velocity over 60s window
    recent   = df[df["timestamp"] >= window_60s].copy()
    velocity = 0.0
    if len(recent) >= 2:
        recent["t"]   = (recent["timestamp"] - recent["timestamp"].min()).dt.total_seconds()
        recent["idx"] = range(len(recent))
        slope         = np.polyfit(recent["t"], recent["idx"], 1)
        velocity      = round(float(slope[0]), 3)

    tier = _classify_tier(spike_ratio, searches_60s)

    return {
        "searches_last_60s": searches_60s,
        "baseline_per_min":  round(baseline_per_min, 1),
        "spike_ratio":       spike_ratio,
        "velocity_per_sec":  velocity,
        "demand_tier":       tier,
    }


def _classify_tier(spike_ratio: float, searches: int) -> str:
    if spike_ratio >= 3.0 or searches >= 150:
        return "SURGE"
    if spike_ratio >= 2.0 or searches >= 80:
        return "HIGH"
    if spike_ratio >= 1.3 or searches >= 30:
        return "MEDIUM"
    return "LOW"


def compute_flash_fare(base_fare: float, metrics: dict, seats_remaining: int) -> dict:
    """
    Flash Fare Pricing Algorithm:

        flash_fare = base_fare × (demand_mult + spike_premium + scarcity_premium + velocity_bonus)

    Components:
        demand_mult      — tier multiplier:  1.00 / 1.05 / 1.12 / 1.20
        spike_premium    — (spike_ratio-1) × 0.03, capped at 0.08
        scarcity_premium — (1 - seats/200) × 0.10
        velocity_bonus   — velocity × 0.02, capped at 0.05
    """
    tier         = metrics["demand_tier"]
    spike        = metrics["spike_ratio"]
    velocity     = metrics["velocity_per_sec"]
    seats_factor = max(0.0, 1 - seats_remaining / 200)

    demand_mult      = {"LOW": 1.00, "MEDIUM": 1.05, "HIGH": 1.12, "SURGE": 1.20}[tier]
    spike_premium    = min((spike - 1) * 0.03, 0.08) if spike > 1 else 0.0
    scarcity_premium = seats_factor * 0.10
    velocity_bonus   = min(velocity * 0.02, 0.05) if velocity > 0 else 0.0

    total_mult  = demand_mult + spike_premium + scarcity_premium + velocity_bonus
    flash_price = round(base_fare * total_mult, 2)
    uplift_pct  = round((flash_price - base_fare) / base_fare * 100, 1)

    window_min = {"LOW": 30, "MEDIUM": 20, "HIGH": 12, "SURGE": 7}[tier]

    return {
        "base_fare_usd":        base_fare,
        "flash_fare_usd":       flash_price,
        "price_multiplier":     round(total_mult, 4),
        "uplift_pct":           uplift_pct,
        "flash_window_minutes": window_min,
        "conversion_confidence": {"LOW": "Low", "MEDIUM": "Moderate", "HIGH": "High", "SURGE": "Very High"}[tier],
        "expires_at":           (datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=window_min)).isoformat() + "Z",
    }


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class SearchEvent(BaseModel):
    route:        str = Field(..., example="NYC-LON",  description="Origin-Destination pair")
    search_count: int = Field(..., ge=1, le=10000, example=100, description="Number of searches in batch")


class FlashFareRequest(BaseModel):
    route:           str   = Field(..., example="NYC-LON")
    base_fare_usd:   float = Field(..., gt=0,   example=520.00, description="Current published fare in USD")
    seats_remaining: int   = Field(..., ge=0, le=500, example=42,  description="Available seats on this flight")


class SimulateRequest(BaseModel):
    route:               str = Field(..., example="NYC-LON")
    searches_per_minute: int = Field(..., ge=1, le=2000, example=120)
    duration_seconds:    int = Field(60, ge=10, le=300)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"], summary="Health check")
def root():
    return {
        "service":   "AeroSignal Fare Optimizer API",
        "version":   "2.0.0",
        "status":    "operational",
        "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        "endpoints": {
            "record_search":  "POST /search/record",
            "flash_fare":     "POST /fare/flash",
            "demand_metrics": "GET  /demand/metrics/{route}",
            "simulate":       "POST /simulate",
            "routes":         "GET  /routes",
            "docs":           "GET  /docs",
        },
    }


@app.post("/search/record", tags=["Ingestion"], summary="Ingest search events")
def record_search(event: SearchEvent):
    """Ingest a batch of search events for a route into the rolling window store."""
    route = event.route.upper()
    store.record(route, event.search_count)
    return {
        "status":            "recorded",
        "route":             route,
        "searches_recorded": event.search_count,
        "total_in_store":    store.total_recorded,
        "timestamp":         datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z",
    }


@app.get("/demand/metrics/{route}", tags=["Analytics"], summary="Get live demand metrics")
def get_demand_metrics(route: str):
    """Return real-time Pandas-computed demand metrics for a specific route."""
    route   = route.upper()
    metrics = compute_demand_metrics(route)
    return {"route": route, **metrics, "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"}


@app.post("/fare/flash", tags=["Pricing"], summary="Get Flash Fare recommendation")
def get_flash_fare(req: FlashFareRequest):
    """
    Core pricing endpoint.

    Given live search pressure + published fare + seat inventory,
    returns an optimized Flash Fare designed to convert latent searchers
    before they migrate to a competitor.
    """
    route    = req.route.upper()
    metrics  = compute_demand_metrics(route)
    rec      = compute_flash_fare(req.base_fare_usd, metrics, req.seats_remaining)

    return {
        "route":          route,
        "demand":         metrics,
        "recommendation": rec,
        "action":         _action_message(metrics["demand_tier"], rec["flash_window_minutes"]),
        "timestamp":      datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z",
    }


@app.post("/simulate", tags=["Testing"], summary="Simulate a demand spike")
def simulate_demand(req: SimulateRequest):
    """Inject synthetic search events — useful for demos and load testing."""
    route               = req.route.upper()
    batches             = max(1, req.duration_seconds // 5)
    searches_per_batch  = max(1, req.searches_per_minute * 5 // 60)

    injected = 0
    for _ in range(batches):
        jitter = random.randint(-searches_per_batch // 4, searches_per_batch // 4)
        count  = max(1, searches_per_batch + jitter)
        store.record(route, count)
        injected += count

    metrics = compute_demand_metrics(route)
    return {
        "route":            route,
        "simulated_searches": injected,
        "duration_seconds": req.duration_seconds,
        "demand_result":    metrics,
        "message":          f"Injected {injected} search events in {batches} batches.",
    }


@app.get("/routes", tags=["Monitoring"], summary="List all active routes")
def list_routes():
    """Return all routes currently tracked in the search store with live demand tiers."""
    routes  = store.all_routes()
    summary = []
    for r in routes:
        m = compute_demand_metrics(r)
        summary.append({
            "route":             r,
            "demand_tier":       m["demand_tier"],
            "searches_last_60s": m["searches_last_60s"],
            "spike_ratio":       m["spike_ratio"],
        })
    summary.sort(key=lambda x: x["searches_last_60s"], reverse=True)
    return {
        "active_routes":   len(routes),
        "total_recorded":  store.total_recorded,
        "routes":          summary,
        "timestamp":       datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z",
    }


def _action_message(tier: str, window: int) -> str:
    messages = {
        "SURGE":  f"🔴 SURGE DETECTED — Activate Flash Fare NOW. Window closes in {window} min.",
        "HIGH":   f"🟠 HIGH DEMAND   — Deploy Flash Fare within {window} min to capture searchers.",
        "MEDIUM": f"🟡 ELEVATED      — Flash Fare recommended. Monitor for escalation ({window} min).",
        "LOW":    f"🟢 STABLE        — No Flash Fare action needed. Standard pricing optimal.",
    }
    return messages[tier]


# ── PyCharm Direct Run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n✈️  AeroSignal API Server")
    print("   Docs  → http://localhost:8000/docs")
    print("   Press Ctrl+C to stop\n")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
