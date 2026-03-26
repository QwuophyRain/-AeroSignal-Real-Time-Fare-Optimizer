"""
ml/pipeline.py
───────────────
Machine Learning Pipeline — Demand Tier Classifier

Run directly in PyCharm:
    Right-click → Run 'pipeline'

Or via run_all.py:
    python run_all.py  → choose option 4

What this does:
    1. Load the Kaggle airline demand dataset
    2. Engineer features (spike_ratio, load_factor, uplift_pct, days_to_departure)
    3. Train XGBoost + sklearn classifiers
    4. Cross-validate and compare models
    5. Generate evaluation report (confusion matrix, feature importance)
    6. Save best model as models/demand_tier_classifier.pkl
    7. Export predictions to data/ml_predictions.csv
"""

import sys
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for PyCharm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics          import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)
from sklearn.pipeline         import Pipeline
import joblib

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).parent.parent
MODELS = ROOT / "models"
DATA   = ROOT / "data"
FIGS   = ROOT / "outputs" / "ml_charts"

sys.path.insert(0, str(ROOT))

TIER_ORDER = ["LOW", "MEDIUM", "HIGH", "SURGE"]


# ════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════

def load_and_engineer(csv_path: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Load dataset and engineer ML features.

    Features used:
        spike_ratio        — primary demand signal
        load_factor        — seat occupancy pressure
        uplift_pct         — current price uplift
        days_to_departure  — booking urgency
        searches_log       — log-transformed search volume
        spike_x_load       — interaction feature
        is_last_minute     — binary: days_to_departure < 7

    Target:
        demand_tier  → LOW / MEDIUM / HIGH / SURGE
    """
    df = pd.read_csv(csv_path)

    # Map column aliases
    col_map = {
        'search_count': 'searches',
        'search_volume': 'searches',
        'volume': 'searches',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ensure required columns exist with fallbacks
    if 'searches' not in df.columns:
        df['searches'] = 10
    if 'spike_ratio' not in df.columns:
        df['spike_ratio'] = 1.0
    if 'load_factor' not in df.columns:
        df['load_factor'] = 0.75
    if 'uplift_pct' not in df.columns:
        df['uplift_pct'] = 8.0
    if 'days_to_departure' not in df.columns:
        df['days_to_departure'] = 30
    if 'demand_tier' not in df.columns:
        raise ValueError("Dataset must have a 'demand_tier' column (LOW/MEDIUM/HIGH/SURGE)")

    # Feature engineering
    df['searches_log']    = np.log1p(df['searches'])
    df['spike_x_load']    = df['spike_ratio'] * df['load_factor']
    df['is_last_minute']  = (df['days_to_departure'] < 7).astype(int)
    df['spike_squared']   = df['spike_ratio'] ** 2
    df['urgency_score']   = df['spike_ratio'] / (df['days_to_departure'].clip(lower=1))

    features = [
        'spike_ratio', 'load_factor', 'uplift_pct', 'days_to_departure',
        'searches_log', 'spike_x_load', 'is_last_minute', 'spike_squared', 'urgency_score'
    ]

    # Encode target
    df['demand_tier'] = df['demand_tier'].str.upper().str.strip()
    df = df[df['demand_tier'].isin(TIER_ORDER)]

    X = df[features].fillna(0)
    y = df['demand_tier']

    return X, y, features


# ════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════

def build_models() -> dict:
    """Return dict of model_name → sklearn Pipeline."""
    models = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=8,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=5,
                learning_rate=0.08, subsample=0.8, random_state=42
            ))
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=500, C=1.0, random_state=42,
                multi_class='multinomial', solver='lbfgs'
            ))
        ]),
    }

    # Add XGBoost if available
    try:
        from xgboost import XGBClassifier
        le = LabelEncoder()
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric='mlogloss',
                random_state=42, n_jobs=-1
            ))
        ])
    except ImportError:
        print("  ℹ️  XGBoost not installed — skipping (pip install xgboost)")

    return models


# ════════════════════════════════════════════════════════════════
# EVALUATION
# ════════════════════════════════════════════════════════════════

def evaluate_models(models: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Train all models, cross-validate, return comparison DataFrame."""
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows   = []
    fitted = {}

    print(f"\n  {'Model':<22} {'CV Acc':>8} {'Test Acc':>10} {'F1-Macro':>10}")
    print(f"  {'─'*22} {'─'*8} {'─'*10} {'─'*10}")

    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

        # Fit on full train, evaluate on test
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc      = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

        rows.append({
            "Model":       name,
            "CV Accuracy": round(cv_scores.mean(), 4),
            "CV Std":      round(cv_scores.std(),  4),
            "Test Accuracy": round(acc,      4),
            "F1 Macro":    round(f1_macro,   4),
        })
        fitted[name] = (model, y_pred)

        print(f"  {name:<22} {cv_scores.mean():>7.1%}  {acc:>9.1%}  {f1_macro:>9.1%}")

    return pd.DataFrame(rows), fitted


# ════════════════════════════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════════════════════════════

def save_charts(best_model, best_name: str, X_test, y_test, y_pred, features: list,
                X_full, y_full, comparison_df: pd.DataFrame):
    """Generate and save all evaluation charts."""
    FIGS.mkdir(parents=True, exist_ok=True)

    # ── 1. Model Comparison Bar Chart ─────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#0d1320')
    ax.set_facecolor('#0d1320')

    x   = np.arange(len(comparison_df))
    w   = 0.35
    b1  = ax.bar(x - w/2, comparison_df["CV Accuracy"],  w, label="CV Accuracy",   color='#00d4ff', alpha=0.85)
    b2  = ax.bar(x + w/2, comparison_df["Test Accuracy"], w, label="Test Accuracy", color='#7c5cfc', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["Model"], color='#6b7a96', fontsize=9)
    ax.set_ylabel("Accuracy", color='#6b7a96')
    ax.set_title("Model Comparison — CV vs Test Accuracy", color='#e8edf5', fontsize=12)
    ax.legend(facecolor='#141c2e', labelcolor='#e8edf5', fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.tick_params(colors='#6b7a96')
    for spine in ax.spines.values():
        spine.set_color('#1e2a3a')
    ax.set_ylim(0.5, 1.05)
    plt.tight_layout()
    plt.savefig(FIGS / "1_model_comparison.png", dpi=140, facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Saved → outputs/ml_charts/1_model_comparison.png")

    # ── 2. Confusion Matrix ────────────────────────────────────
    tiers  = [t for t in TIER_ORDER if t in y_test.unique()]
    cm     = confusion_matrix(y_test, y_pred, labels=tiers, normalize='true')

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#0d1320')
    ax.set_facecolor('#0d1320')

    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=tiers, yticklabels=tiers,
                ax=ax, linewidths=0.5, linecolor='#1e2a3a',
                annot_kws={'size': 11, 'color': 'white'})
    ax.set_title(f"Confusion Matrix — {best_name}", color='#e8edf5', fontsize=12)
    ax.set_xlabel("Predicted", color='#6b7a96')
    ax.set_ylabel("Actual",    color='#6b7a96')
    ax.tick_params(colors='#6b7a96')
    plt.tight_layout()
    plt.savefig(FIGS / "2_confusion_matrix.png", dpi=140, facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Saved → outputs/ml_charts/2_confusion_matrix.png")

    # ── 3. Feature Importance ─────────────────────────────────
    clf = best_model.named_steps.get('clf')
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        idx         = np.argsort(importances)
        colors_fi   = ['#ff4d6d' if importances[i] > 0.15 else '#00d4ff' if importances[i] > 0.05 else '#6b7a96' for i in idx]

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#0d1320')
        ax.set_facecolor('#0d1320')
        ax.barh([features[i] for i in idx], importances[idx], color=colors_fi, alpha=0.85)
        ax.set_title("Feature Importance", color='#e8edf5', fontsize=12)
        ax.set_xlabel("Importance Score", color='#6b7a96')
        ax.tick_params(colors='#6b7a96')
        for spine in ax.spines.values():
            spine.set_color('#1e2a3a')
        plt.tight_layout()
        plt.savefig(FIGS / "3_feature_importance.png", dpi=140, facecolor=fig.get_facecolor())
        plt.close()
        print(f"    Saved → outputs/ml_charts/3_feature_importance.png")

    # ── 4. Spike Ratio Distribution by Tier ───────────────────
    full_df = X_full.copy()
    full_df['demand_tier'] = y_full.values

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#0d1320')
    ax.set_facecolor('#0d1320')

    tier_colors = {'LOW': '#6bcb77', 'MEDIUM': '#ffd93d', 'HIGH': '#ff9f43', 'SURGE': '#ff4d6d'}
    for tier in TIER_ORDER:
        subset = full_df[full_df['demand_tier'] == tier]['spike_ratio']
        if not subset.empty:
            subset.clip(upper=20).plot.hist(
                bins=25, alpha=0.55, ax=ax,
                label=tier, color=tier_colors.get(tier, '#aaa')
            )

    ax.set_title("Spike Ratio Distribution by Demand Tier", color='#e8edf5', fontsize=12)
    ax.set_xlabel("Spike Ratio", color='#6b7a96')
    ax.set_ylabel("Count",       color='#6b7a96')
    ax.legend(facecolor='#141c2e', labelcolor='#e8edf5', fontsize=9)
    ax.tick_params(colors='#6b7a96')
    for spine in ax.spines.values():
        spine.set_color('#1e2a3a')
    plt.tight_layout()
    plt.savefig(FIGS / "4_spike_distribution.png", dpi=140, facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Saved → outputs/ml_charts/4_spike_distribution.png")

    # ── 5. Uplift % vs Days to Departure Scatter ───────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#0d1320')
    ax.set_facecolor('#0d1320')

    for tier in TIER_ORDER:
        mask = full_df['demand_tier'] == tier
        ax.scatter(
            full_df.loc[mask, 'days_to_departure'],
            full_df.loc[mask, 'uplift_pct'],
            alpha=0.4, s=18, label=tier,
            color=tier_colors.get(tier, '#aaa')
        )
    ax.set_title("Days to Departure vs Uplift % (Pricing Pressure)", color='#e8edf5', fontsize=12)
    ax.set_xlabel("Days to Departure", color='#6b7a96')
    ax.set_ylabel("Flash Fare Uplift %", color='#6b7a96')
    ax.legend(facecolor='#141c2e', labelcolor='#e8edf5', fontsize=9)
    ax.tick_params(colors='#6b7a96')
    for spine in ax.spines.values():
        spine.set_color('#1e2a3a')
    plt.tight_layout()
    plt.savefig(FIGS / "5_uplift_scatter.png", dpi=140, facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Saved → outputs/ml_charts/5_uplift_scatter.png")


# ════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════

def run_full_pipeline(csv_path: str = "data/kaggle_airline_demand.csv"):
    """
    Execute the full ML training pipeline.

    Steps:
        1. Load + feature engineer
        2. Train/test split (80/20, stratified)
        3. Train + cross-validate 3-4 models
        4. Select best model by F1-macro
        5. Save charts + model artifact
        6. Export predictions CSV
    """
    csv_path = str(ROOT / csv_path) if not Path(csv_path).is_absolute() else csv_path
    MODELS.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "ml_charts").mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  AeroSignal ML Training Pipeline")
    print(f"{'='*60}")
    print(f"  Dataset : {csv_path}\n")

    # 1. Load data
    print("📂  Loading + engineering features...")
    X, y, features = load_and_engineer(csv_path)
    print(f"    {len(X):,} samples · {len(features)} features · {y.nunique()} classes")
    print(f"    Class balance: {dict(y.value_counts())}\n")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    # 3. Train models
    print("🤖  Training models (5-fold cross-validation)...\n")
    models           = build_models()
    comparison_df, fitted = evaluate_models(models, X_train, X_test, y_train, y_test)

    # 4. Select best by F1
    best_name  = comparison_df.loc[comparison_df["F1 Macro"].idxmax(), "Model"]
    best_model = fitted[best_name][0]
    y_pred     = fitted[best_name][1]

    print(f"\n  ★  Best model: {best_name}  (F1={comparison_df.loc[comparison_df['Model']==best_name,'F1 Macro'].iloc[0]:.4f})")

    # 5. Full classification report
    print(f"\n  Classification Report ({best_name}):")
    print("  " + "─" * 56)
    report_str = classification_report(y_test, y_pred, target_names=TIER_ORDER, zero_division=0)
    for line in report_str.strip().split('\n'):
        print("  " + line)

    # 6. Save charts
    print("\n📊  Generating evaluation charts...")
    save_charts(best_model, best_name, X_test, y_test, y_pred, features, X, y, comparison_df)

    # 7. Save model
    model_path = MODELS / "demand_tier_classifier.pkl"
    joblib.dump({
        "model":    best_model,
        "features": features,
        "classes":  TIER_ORDER,
        "trained":  pd.Timestamp.now().isoformat(),
        "dataset":  csv_path,
        "metrics":  comparison_df[comparison_df["Model"] == best_name].to_dict(orient="records")[0],
    }, model_path)
    print(f"\n💾  Model saved → {model_path}")

    # 8. Export predictions
    test_df = X_test.copy()
    test_df["actual_tier"]    = y_test.values
    test_df["predicted_tier"] = y_pred
    test_df["correct"]        = test_df["actual_tier"] == test_df["predicted_tier"]
    out_path = DATA / "ml_predictions.csv"
    test_df.to_csv(out_path, index=False)
    print(f"📄  Predictions saved → {out_path}")

    # 9. Model comparison table
    print(f"\n{'─'*60}")
    print(f"  Model Comparison Summary")
    print(f"{'─'*60}")
    print(comparison_df.to_string(index=False))
    print(f"{'─'*60}\n")

    return best_model, features


def load_model(model_path: str = "models/demand_tier_classifier.pkl"):
    """Load a saved model artifact for inference."""
    path = ROOT / model_path
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}\nRun the ML pipeline first.")
    return joblib.load(path)


def predict_tier(spike_ratio: float, load_factor: float, uplift_pct: float,
                 days_to_departure: int, searches: int = 50,
                 model_path: str = "models/demand_tier_classifier.pkl") -> dict:
    """
    Make a single demand tier prediction.

    Example:
        from ml.pipeline import predict_tier
        result = predict_tier(spike_ratio=2.5, load_factor=0.82, uplift_pct=14.0, days_to_departure=7)
        print(result)
    """
    artifact = load_model(model_path)
    model    = artifact["model"]
    features = artifact["features"]

    searches_log  = np.log1p(searches)
    spike_x_load  = spike_ratio * load_factor
    is_last_minute = int(days_to_departure < 7)
    spike_squared  = spike_ratio ** 2
    urgency_score  = spike_ratio / max(days_to_departure, 1)

    row = pd.DataFrame([[
        spike_ratio, load_factor, uplift_pct, days_to_departure,
        searches_log, spike_x_load, is_last_minute, spike_squared, urgency_score
    ]], columns=features)

    tier       = model.predict(row)[0]
    proba      = model.predict_proba(row)[0]
    classes    = model.classes_

    return {
        "predicted_tier": tier,
        "confidence":     round(float(proba.max()), 4),
        "probabilities":  dict(zip(classes, proba.round(4).tolist())),
    }


# ── PyCharm Direct Run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_full_pipeline()
