"""
╔══════════════════════════════════════════════════════════════╗
║           AEROSIGNAL — Real-Time Fare Optimizer              ║
║                  PyCharm Entry Point                         ║
╚══════════════════════════════════════════════════════════════╝

HOW TO RUN IN PYCHARM:
─────────────────────
  Right-click this file → "Run 'run_all'"
  OR press Shift+F10

  Choose a mode via the MENU that appears, or set MODE below.

AVAILABLE MODES:
  1  → API Server only          (FastAPI on http://localhost:8000)
  2  → Dashboard only           (Browser UI on http://localhost:5050)
  3  → API + Dashboard together (both servers simultaneously)
  4  → ML Training Pipeline     (train, evaluate, save model)
  5  → Generate Sample Data     (create/refresh CSV datasets)
  6  → Run Demand Analysis      (batch CSV → flash fare report)
  7  → Run All Tests            (pytest suite)
  8  → Full Demo                (data → API → dashboard → ML)
"""

import os
import sys
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

# ── Set working directory to project root ────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION — Edit these defaults
# ══════════════════════════════════════════════════════════════════
MODE         = None           # Set to 1-8 to skip the menu, e.g. MODE = 3
API_PORT     = 8000
DASH_PORT    = 5050
CSV_PATH     = "data/kaggle_airline_demand.csv"
BASE_FARE    = 520.0
SEATS        = 42
AUTO_BROWSER = True          # Open browser automatically
# ══════════════════════════════════════════════════════════════════


def banner():
    print("""
\033[36m╔══════════════════════════════════════════════════════════╗
║          ✈  AEROSIGNAL FARE OPTIMIZER  ✈                  ║
╠══════════════════════════════════════════════════════════╣
║  Real-time demand spike detection + Flash Fare engine    ║
╚══════════════════════════════════════════════════════════╝\033[0m
""")


def menu() -> int:
    print("\033[33m  Select a mode to run:\033[0m")
    options = [
        ("1", "API Server          →  FastAPI  http://localhost:{}".format(API_PORT)),
        ("2", "Dashboard           →  Browser  http://localhost:{}".format(DASH_PORT)),
        ("3", "API + Dashboard     →  Both servers together"),
        ("4", "ML Training         →  Train demand-tier classifier"),
        ("5", "Generate Data       →  Create sample CSV datasets"),
        ("6", "Demand Analysis     →  Batch CSV → flash fare report"),
        ("7", "Run Tests           →  pytest test suite"),
        ("8", "Full Demo           →  Data → API → Dashboard → ML"),
    ]
    for key, desc in options:
        print(f"  \033[36m[{key}]\033[0m  {desc}")
    print()
    choice = input("  Enter choice (1-8): ").strip()
    return int(choice) if choice.isdigit() else 1


def run_api():
    """Start the FastAPI server with uvicorn."""
    print(f"\n\033[36m▶  Starting API server on http://localhost:{API_PORT}\033[0m")
    print(f"   Swagger docs → http://localhost:{API_PORT}/docs\n")
    import uvicorn
    from app.main import app
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="info")


def run_dashboard():
    """Start the Flask dashboard server and open browser."""
    print(f"\n\033[36m▶  Starting Dashboard on http://localhost:{DASH_PORT}\033[0m\n")
    from dashboard.server import create_app
    flask_app = create_app()
    if AUTO_BROWSER:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{DASH_PORT}")).start()
    flask_app.run(host="0.0.0.0", port=DASH_PORT, debug=False, use_reloader=False)


def run_api_and_dashboard():
    """Run both servers concurrently using threads."""
    print("\n\033[36m▶  Starting API + Dashboard together\033[0m")
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    time.sleep(1.5)  # let API boot first
    run_dashboard()   # dashboard blocks main thread


def run_ml():
    """Execute the full ML training pipeline."""
    print("\n\033[36m▶  Running ML Training Pipeline\033[0m\n")
    from ml.pipeline import run_full_pipeline
    run_full_pipeline(csv_path=CSV_PATH)


def run_generate_data():
    """Generate sample datasets."""
    print("\n\033[36m▶  Generating Sample Datasets\033[0m\n")
    from scripts.generate_data import generate_all
    generate_all()


def run_demand_analysis():
    """Run batch demand analysis on a CSV file."""
    print("\n\033[36m▶  Running Demand Analysis\033[0m\n")
    from scripts.analyse import run_analysis
    run_analysis(csv_path=CSV_PATH, base_fare=BASE_FARE, seats=SEATS)


def run_tests():
    """Run pytest suite."""
    print("\n\033[36m▶  Running Test Suite\033[0m\n")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=str(ROOT)
    )
    sys.exit(result.returncode)


def run_full_demo():
    """Full end-to-end demo: generate → analyse → ML → serve."""
    print("\n\033[36m▶  Full Demo Mode\033[0m\n")
    print("  Step 1/3: Generating data...")
    run_generate_data()
    print("\n  Step 2/3: Running demand analysis...")
    run_demand_analysis()
    print("\n  Step 3/3: Training ML model...")
    run_ml()
    print("\n  Launching Dashboard + API...\n")
    run_api_and_dashboard()


if __name__ == "__main__":
    banner()
    mode = MODE or menu()
    dispatch = {
        1: run_api,
        2: run_dashboard,
        3: run_api_and_dashboard,
        4: run_ml,
        5: run_generate_data,
        6: run_demand_analysis,
        7: run_tests,
        8: run_full_demo,
    }
    fn = dispatch.get(mode)
    if fn:
        fn()
    else:
        print(f"\033[31m  Invalid choice: {mode}\033[0m")
        sys.exit(1)
