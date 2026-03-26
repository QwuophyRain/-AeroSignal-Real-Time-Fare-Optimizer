# ✈️ AeroSignal — Real-Time Fare Optimizer

> **Captures latent demand from shoppers who search but don't book — converting demand spikes into Flash Fares before revenue leaks to competitors.**

---

## 🚀 PyCharm Setup (Step-by-Step)

### Step 1 — Open Project
```
File → Open → select the aerosignal/ folder
```

### Step 2 — Create Virtual Environment
```
File → Settings → Project → Python Interpreter
  → Add Interpreter → Virtualenv → New
  → Base interpreter: Python 3.11+
  → Location: aerosignal/venv
  → OK
```

### Step 3 — Install Dependencies
Open PyCharm Terminal (`Alt+F12`) and run:
```bash
pip install -r requirements.txt
```

### Step 4 — Mark Sources Root
Right-click `aerosignal/` folder in Project panel → `Mark Directory as` → `Sources Root`

### Step 5 — Run the Project
**Open `run_all.py`** → Right-click → **Run 'run_all'**

A menu appears in the terminal — choose a mode:

| Key | Mode | What Opens |
|-----|------|------------|
| `1` | API Server | http://localhost:8000/docs |
| `2` | Dashboard | http://localhost:5050 |
| `3` | API + Dashboard | Both simultaneously |
| `4` | ML Training | Trains classifier, saves charts |
| `5` | Generate Data | Creates fresh CSV datasets |
| `6` | Demand Analysis | Batch CSV → fare report |
| `7` | Run Tests | Full pytest suite |
| `8` | Full Demo | Everything end-to-end |

---

## 📁 Project Structure

```
aerosignal/
│
├── run_all.py                  ← START HERE — main PyCharm launcher
│
├── app/
│   ├── main.py                 ← FastAPI server (run directly too)
│   └── demand_engine.py        ← Pandas analytics core (run directly too)
│
├── dashboard/
│   ├── server.py               ← Flask dashboard server (run directly too)
│   └── index.html              ← AeroSignal UI (open in browser)
│
├── ml/
│   └── pipeline.py             ← ML training pipeline (run directly too)
│
├── scripts/
│   ├── generate_data.py        ← Dataset generator (run directly too)
│   └── analyse.py              ← Batch analysis runner (run directly too)
│
├── notebooks/
│   └── quickstart.py           ← Interactive walkthrough (run section by section)
│
├── tests/
│   └── test_suite.py           ← pytest test suite
│
├── data/
│   ├── kaggle_airline_demand.csv   ← Main dataset (600 rows, 18 cols)
│   ├── sample_searches.csv         ← Search event log
│   └── flash_fare_report.csv       ← Pre-computed fare table
│
├── models/                     ← Saved ML model artifacts (created on training)
├── outputs/                    ← Analysis CSVs + ML charts (created on run)
│
├── requirements.txt
└── .env.example                ← Copy to .env to configure
```

---

## 🏃 Running Individual Files Directly

Every Python file has a `if __name__ == "__main__":` block. Right-click any file → **Run** in PyCharm:

| File | What it does |
|------|-------------|
| `run_all.py` | Interactive menu — choose any mode |
| `app/main.py` | Start API server on port 8000 |
| `dashboard/server.py` | Start dashboard on port 5050 + open browser |
| `ml/pipeline.py` | Train ML models, save charts & model |
| `scripts/generate_data.py` | Generate all sample CSV datasets |
| `scripts/analyse.py` | Batch demand analysis, print report |
| `notebooks/quickstart.py` | Run section-by-section in Python Console |

---

## 📡 API Endpoints

Once the API server is running (`run_all.py` → `1`):

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/docs` | Swagger UI — test all endpoints |
| `GET` | `/` | Health check |
| `POST` | `/search/record` | Ingest search events |
| `GET` | `/demand/metrics/{route}` | Live demand metrics |
| `POST` | `/fare/flash` | Get Flash Fare recommendation |
| `POST` | `/simulate` | Inject synthetic demand |
| `GET` | `/routes` | All active routes + tiers |

### Quick API Test (in PyCharm Terminal)
```bash
# Record 150 searches
curl -X POST http://localhost:8000/search/record \
  -H "Content-Type: application/json" \
  -d '{"route": "NYC-LON", "search_count": 150}'

# Get flash fare recommendation
curl -X POST http://localhost:8000/fare/flash \
  -H "Content-Type: application/json" \
  -d '{"route": "NYC-LON", "base_fare_usd": 520.00, "seats_remaining": 42}'
```

---

## 🤖 ML Pipeline

The ML pipeline (`ml/pipeline.py`) trains a multi-class demand tier classifier:

**Target**: `demand_tier` → `LOW / MEDIUM / HIGH / SURGE`

**Features engineered**:
| Feature | Description |
|---------|-------------|
| `spike_ratio` | Current searches / 5-min baseline |
| `load_factor` | Seat occupancy (0-1) |
| `uplift_pct` | Flash fare % above base |
| `days_to_departure` | Booking urgency |
| `searches_log` | Log-scaled volume |
| `spike_x_load` | Interaction: spike × occupancy |
| `is_last_minute` | Binary: DTD < 7 days |
| `spike_squared` | Non-linear spike signal |
| `urgency_score` | spike_ratio / days_to_departure |

**Models trained**: Random Forest, Gradient Boosting, Logistic Regression, XGBoost

**Charts saved to** `outputs/ml_charts/`:
- `1_model_comparison.png` — CV vs Test accuracy bars
- `2_confusion_matrix.png` — Normalized confusion matrix
- `3_feature_importance.png` — Feature importance bars
- `4_spike_distribution.png` — Spike ratio histogram by tier
- `5_uplift_scatter.png` — Days to departure vs uplift scatter

### Predict a single instance:
```python
from ml.pipeline import predict_tier

result = predict_tier(
    spike_ratio=4.2,
    load_factor=0.89,
    uplift_pct=28.0,
    days_to_departure=3,
    searches=180
)
print(result)
# → {'predicted_tier': 'SURGE', 'confidence': 0.9731, 'probabilities': {...}}
```

---

## 📊 Dashboard

The browser UI (`dashboard/index.html`) has 4 pages:

| Page | Contents |
|------|----------|
| **Overview** | KPI cards, revenue bar chart, demand tier doughnut, hourly traffic, route table |
| **Monitoring** | Route cards with live tiers, surge alert feed, uplift histogram, load factor chart |
| **Traffic** | Route/tier filters, stacked volume chart, conversion rate bars, pricing pressure scatter, top surge table |
| **ML Analytics** | Feature importance, confusion matrix, model metrics, scatter plots, class distribution |

**CSV Upload**: Drop any compatible CSV onto the upload bar — the dashboard rebuilds itself instantly.

Compatible column names (auto-detected):
- Route: `route`, `od_pair`, `origin_dest`, `route_code`
- Tier: `demand_tier`, `tier`, `demand_level`
- Searches: `searches`, `search_count`, `volume`
- Fare: `flash_fare`, `flash_price`, `optimized_fare`

---

## 🧪 Running Tests

```bash
# In PyCharm Terminal
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=app --cov-report=html

# Run just API tests
pytest tests/ -v -k "TestAPI"

# Run just ML tests
pytest tests/ -v -k "TestFlashFare"
```

---

## 💡 Flash Fare Algorithm

```
flash_fare = base_fare × total_multiplier

total_multiplier = demand_mult          # LOW=1.00 / MED=1.05 / HIGH=1.12 / SURGE=1.20
                 + spike_premium        # min((spike-1) × 0.03, 0.08)
                 + scarcity_premium     # (1 - seats/200) × 0.10
                 + velocity_bonus       # min(velocity × 0.02, 0.05)

flash_window     = 30 / 20 / 12 / 7 min   # shorter = more urgency
```

---

## 🗄️ Using Your Own Kaggle Dataset

1. Download any airline pricing/demand dataset from [kaggle.com](https://www.kaggle.com)
2. Drop the CSV into the `data/` folder
3. Update `CSV_PATH` in `run_all.py`:
   ```python
   CSV_PATH = "data/your_dataset.csv"
   ```
4. Or drag-drop onto the dashboard Upload bar
5. The system auto-detects column names and rebuilds all charts

**Recommended Kaggle datasets**:
- "Airline Demand Forecasting" — demand + route data
- "Flight Price Prediction" — fare + date features
- "Airline Passenger Satisfaction" — load factor proxies

---

## ⚙️ Configuration

Copy `.env.example` → `.env` and edit:

```bash
cp .env.example .env
```

Key settings:
```env
API_PORT=8000
DASHBOARD_PORT=5050
DEFAULT_CSV=data/kaggle_airline_demand.csv
BASE_FARE=520.0
SEATS_REMAINING=42
SURGE_SPIKE_THRESHOLD=3.0
```

---

## 🐍 Python Version

Requires **Python 3.10+** (uses `match` syntax and modern type hints).
Tested on Python 3.11 and 3.13.
