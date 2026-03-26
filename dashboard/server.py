"""
dashboard/server.py
────────────────────
Flask server that serves the AeroSignal HTML dashboard
and provides a JSON API endpoint for the frontend.

Run directly in PyCharm:
    Right-click → Run 'server'

Or via run_all.py:
    python run_all.py  → choose option 2
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, send_file, jsonify, request
import pandas as pd
import numpy as np


def build_dashboard_json(csv_path: str) -> dict:
    """
    Compute all dashboard metrics from a CSV dataset.
    Supports kaggle_airline_demand.csv, flash_fare_report.csv,
    sample_searches.csv, or any compatible format.
    """
    df = pd.read_csv(csv_path)

    # ── Column normalization ──────────────────────────────────────
    col_lower = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=col_lower)

    def find_col(df, options):
        for o in options:
            if o in df.columns:
                return o
        return None

    route_col   = find_col(df, ['route', 'od_pair', 'origin_dest', 'route_code'])
    tier_col    = find_col(df, ['demand_tier', 'tier', 'demand_level'])
    search_col  = find_col(df, ['searches', 'search_count', 'volume'])
    spike_col   = find_col(df, ['spike_ratio', 'spike', 'demand_spike'])
    ff_col      = find_col(df, ['flash_fare', 'flash_price', 'optimized_fare'])
    bf_col      = find_col(df, ['base_fare', 'base_price', 'static_fare', 'fare'])
    uplift_col  = find_col(df, ['uplift_pct', 'uplift', 'price_increase_pct'])
    load_col    = find_col(df, ['load_factor', 'load', 'occupancy'])
    book_col    = find_col(df, ['bookings', 'conversions', 'booked'])
    dtd_col     = find_col(df, ['days_to_departure', 'days_to_depart', 'dtd'])
    ts_col      = find_col(df, ['window_start', 'timestamp', 'time', 'date'])

    # Fallbacks
    if route_col:  df['_route']   = df[route_col].astype(str)
    else:          df['_route']   = 'UNKNOWN'
    if tier_col:   df['_tier']    = df[tier_col].str.upper().str.strip()
    else:          df['_tier']    = 'LOW'
    if search_col: df['_srch']    = pd.to_numeric(df[search_col], errors='coerce').fillna(0)
    else:          df['_srch']    = 10
    if spike_col:  df['_spike']   = pd.to_numeric(df[spike_col], errors='coerce').fillna(1)
    else:          df['_spike']   = 1.0
    if ff_col:     df['_ff']      = pd.to_numeric(df[ff_col], errors='coerce').fillna(520)
    else:          df['_ff']      = 560
    if bf_col:     df['_bf']      = pd.to_numeric(df[bf_col], errors='coerce').fillna(520)
    else:          df['_bf']      = 520
    if uplift_col: df['_uplift']  = pd.to_numeric(df[uplift_col], errors='coerce').fillna(8)
    else:          df['_uplift']  = ((df['_ff'] - df['_bf']) / df['_bf'] * 100).round(1)
    if load_col:   df['_load']    = pd.to_numeric(df[load_col], errors='coerce').fillna(0.75)
    else:          df['_load']    = 0.75
    if book_col:   df['_book']    = pd.to_numeric(df[book_col], errors='coerce').fillna(0)
    else:          df['_book']    = (df['_srch'] * 0.034).round().astype(int)
    if dtd_col:    df['_dtd']     = pd.to_numeric(df[dtd_col], errors='coerce').fillna(30)
    else:          df['_dtd']     = 30

    df['_static_rev'] = df['_bf'] * df['_book']
    df['_flash_rev']  = df['_ff'] * df['_book']

    # ── Aggregations ──────────────────────────────────────────────
    routes = df['_route'].unique().tolist()

    route_summary = []
    for r in routes:
        sub = df[df['_route'] == r]
        route_summary.append({
            'route':          r,
            'total_searches': int(sub['_srch'].sum()),
            'total_bookings': int(sub['_book'].sum()),
            'flash_revenue':  round(float(sub['_flash_rev'].sum()), 2),
            'static_revenue': round(float(sub['_static_rev'].sum()), 2),
            'recovered':      round(float((sub['_flash_rev'] - sub['_static_rev']).sum()), 2),
            'avg_spike':      round(float(sub['_spike'].mean()), 2),
            'surge_windows':  int((sub['_tier'] == 'SURGE').sum()),
            'avg_load':       round(float(sub['_load'].mean()), 3),
        })

    tier_dist = df['_tier'].value_counts().to_dict()

    # Hourly breakdown (if timestamp available)
    hourly = []
    if ts_col:
        try:
            df['_ts'] = pd.to_datetime(df[ts_col], errors='coerce')
            df['_hr'] = df['_ts'].dt.hour
            h = df.groupby('_hr').agg(
                searches=('_srch', 'sum'),
                bookings=('_book', 'sum'),
                flash_revenue=('_flash_rev', 'sum'),
                surge_count=('_tier', lambda x: (x == 'SURGE').sum())
            ).reset_index().rename(columns={'_hr': 'hour'})
            hourly = h.to_dict(orient='records')
        except Exception:
            pass

    if not hourly:
        # Synthesize hourly spread
        total = int(df['_srch'].sum())
        for h in range(6, 22):
            mult = 1.2 if 8 <= h <= 11 or 18 <= h <= 20 else 0.8
            hourly.append({'hour': h, 'searches': int(total / 16 * mult),
                           'bookings': int(total / 16 * mult * 0.034),
                           'flash_revenue': total / 16 * mult * 0.034 * 560,
                           'surge_count': 2 if 8 <= h <= 11 else 0})

    # Top surge events
    surge_df = df[df['_tier'] == 'SURGE'].nlargest(10, '_spike')
    surge_list = []
    for _, row in surge_df.iterrows():
        surge_list.append({
            'route':       row['_route'],
            'window_start': str(row[ts_col]) if ts_col and ts_col in row else '-',
            'searches':    int(row['_srch']),
            'spike_ratio': round(float(row['_spike']), 2),
            'flash_fare':  round(float(row['_ff']), 2),
            'uplift_pct':  round(float(row['_uplift']), 1),
        })

    # ML features sample
    ml_features = []
    for _, row in df.sample(min(200, len(df)), random_state=42).iterrows():
        ml_features.append({
            'spike_ratio':      round(float(row['_spike']), 2),
            'load_factor':      round(float(row['_load']), 3),
            'uplift_pct':       round(float(row['_uplift']), 1),
            'days_to_departure': int(row['_dtd']),
            'tier':             str(row['_tier']),
        })

    total_flash  = float(df['_flash_rev'].sum())
    total_static = float(df['_static_rev'].sum())
    recovered    = total_flash - total_static

    return {
        'routes':          routes,
        'route_summary':   route_summary,
        'tier_dist':       tier_dist,
        'hourly':          hourly,
        'total_flash':     round(total_flash, 2),
        'total_static':    round(total_static, 2),
        'recovered':       round(recovered, 2),
        'recovered_pct':   round(recovered / max(total_static, 1) * 100, 1),
        'total_searches':  int(df['_srch'].sum()),
        'total_bookings':  int(df['_book'].sum()),
        'surge_events':    int((df['_tier'] == 'SURGE').sum()),
        'avg_conversion':  round(float(df['_book'].sum() / max(df['_srch'].sum(), 1)) * 100, 2),
        'surge_list':      surge_list,
        'ml_features':     ml_features,
        'row_count':       len(df),
        'dataset':         Path(csv_path).name,
    }


def create_app():
    """Create and configure the Flask application."""
    flask_app = Flask(__name__, static_folder=str(ROOT / "dashboard"))

    DEFAULT_CSV = str(ROOT / "data" / "kaggle_airline_demand.csv")

    @flask_app.route("/")
    def index():
        """Serve the main dashboard HTML."""
        return send_file(ROOT / "dashboard" / "index.html")

    @flask_app.route("/api/dashboard")
    def api_dashboard():
        """
        JSON endpoint — returns full dashboard dataset.

        Query params:
            csv  — path to CSV file (default: kaggle_airline_demand.csv)
        """
        csv_path = request.args.get("csv", DEFAULT_CSV)
        if not Path(csv_path).is_absolute():
            csv_path = str(ROOT / csv_path)

        if not Path(csv_path).exists():
            return jsonify({"error": f"File not found: {csv_path}"}), 404

        try:
            data = build_dashboard_json(csv_path)
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/api/datasets")
    def list_datasets():
        """List all CSV files in the data directory."""
        data_dir = ROOT / "data"
        csvs = [f.name for f in data_dir.glob("*.csv")] if data_dir.exists() else []
        return jsonify({"datasets": csvs, "directory": str(data_dir)})

    @flask_app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "AeroSignal Dashboard"})

    return flask_app


# ── PyCharm Direct Run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import webbrowser, threading
    print("\n🌐  AeroSignal Dashboard")
    print("   Open → http://localhost:5050")
    print("   Press Ctrl+C to stop\n")
    app = create_app()
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:5050")).start()
    app.run(host="0.0.0.0", port=5050, debug=False)
