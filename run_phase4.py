from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.portfolio import build_investment_universe, construct_all_portfolios
from src.risk import compute_correlation_matrix, compute_portfolio_metrics, compute_stock_risk_metrics


DEFAULT_SETTINGS = {
    "paths": {
        "processed_data_dir": "data/processed",
        "outputs_dir": "outputs",
    },
    "portfolio": {
        "risk_free_rate": 0.0,
    },
}


def load_settings() -> dict:
    config_path = Path("config/settings.yaml")
    if not config_path.exists():
        return DEFAULT_SETTINGS

    try:
        import yaml
    except ModuleNotFoundError:
        return DEFAULT_SETTINGS

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    merged = DEFAULT_SETTINGS.copy()
    merged["paths"] = {**DEFAULT_SETTINGS["paths"], **loaded.get("paths", {})}
    merged["portfolio"] = {**DEFAULT_SETTINGS["portfolio"], **loaded.get("portfolio", {})}
    return merged


def main() -> int:
    settings = load_settings()
    paths = settings.get("paths", {})
    portfolio_settings = settings.get("portfolio", {})

    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))
    outputs_dir = Path(paths.get("outputs_dir", "outputs"))
    outputs_dir.mkdir(parents=True, exist_ok=True)

    feature_path = processed_dir / "phase2_model_features.csv"
    if not feature_path.exists():
        print("Phase 2 feature file was not found.")
        print("Run python run_phase1.py and python run_phase2.py after placing the allowed Kaggle CSVs in data/raw.")
        return 1

    features = pd.read_csv(feature_path, parse_dates=["date"])
    risk_free_rate = float(portfolio_settings.get("risk_free_rate", 0.0))

    prediction_path = outputs_dir / "phase3_predictions.csv"
    predictions = pd.read_csv(prediction_path, parse_dates=["date"]) if prediction_path.exists() else pd.DataFrame()

    risk_metrics = compute_stock_risk_metrics(features, risk_free_rate=risk_free_rate)
    universe = build_investment_universe(risk_metrics, predictions)
    allocations = construct_all_portfolios(universe)
    portfolio_metrics = compute_portfolio_metrics(features, allocations, risk_free_rate=risk_free_rate)
    correlation = compute_correlation_matrix(features)

    risk_metrics.to_csv(outputs_dir / "phase4_stock_risk_metrics.csv", index=False)
    universe.to_csv(outputs_dir / "phase4_investment_universe.csv", index=False)
    allocations.to_csv(outputs_dir / "phase4_portfolio_allocations.csv", index=False)
    portfolio_metrics.to_csv(outputs_dir / "phase4_portfolio_metrics.csv", index=False)
    correlation.to_csv(outputs_dir / "phase4_return_correlation.csv")

    print("Phase 4 complete.")
    print("Saved outputs:")
    print("- outputs/phase4_stock_risk_metrics.csv")
    print("- outputs/phase4_investment_universe.csv")
    print("- outputs/phase4_portfolio_allocations.csv")
    print("- outputs/phase4_portfolio_metrics.csv")
    print("- outputs/phase4_return_correlation.csv")
    if not portfolio_metrics.empty:
        print("Portfolio metrics:")
        print(portfolio_metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

