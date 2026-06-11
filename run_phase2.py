from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import load_price_history
from src.indicators import build_phase2_features
from src.preprocessing import add_basic_market_features, clean_price_history


DEFAULT_SETTINGS = {
    "paths": {
        "raw_data_dir": "data/raw",
        "processed_data_dir": "data/processed",
        "outputs_dir": "outputs",
    },
    "eda": {
        "rolling_window_days": 21,
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
    merged["eda"] = {**DEFAULT_SETTINGS["eda"], **loaded.get("eda", {})}
    return merged


def load_phase1_or_raw(paths: dict, rolling_window: int) -> pd.DataFrame:
    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))
    phase1_path = processed_dir / "phase1_features.csv"

    if phase1_path.exists():
        df = pd.read_csv(phase1_path, parse_dates=["date"])
        return df.sort_values(["symbol", "date"]).reset_index(drop=True)

    raw_dir = Path(paths.get("raw_data_dir", "data/raw"))
    prices = load_price_history(raw_dir)
    if prices.empty:
        return prices

    clean_prices = clean_price_history(prices)
    return add_basic_market_features(clean_prices, rolling_window=rolling_window)


def main() -> int:
    settings = load_settings()
    paths = settings.get("paths", {})
    eda_settings = settings.get("eda", {})

    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))
    outputs_dir = Path(paths.get("outputs_dir", "outputs"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    prices = load_phase1_or_raw(paths, rolling_window=int(eda_settings.get("rolling_window_days", 21)))
    if prices.empty:
        print("No Phase 1 features or raw price-history CSV files were found.")
        print("Place the allowed Kaggle CSV files in data/raw, then run python run_phase1.py and python run_phase2.py.")
        return 1

    features = build_phase2_features(prices)
    features.to_csv(processed_dir / "phase2_model_features.csv", index=False)

    feature_summary = pd.DataFrame(
        {
            "column": features.columns,
            "non_null_count": [features[col].notna().sum() for col in features.columns],
            "missing_count": [features[col].isna().sum() for col in features.columns],
            "dtype": [str(features[col].dtype) for col in features.columns],
        }
    )
    feature_summary.to_csv(outputs_dir / "phase2_feature_summary.csv", index=False)

    target_cols = [col for col in features.columns if col.startswith("future_return_") or col.startswith("target_direction_")]
    print("Phase 2 complete.")
    print(f"Rows: {len(features):,}")
    print(f"Stocks: {features['symbol'].nunique():,}")
    print(f"Feature columns: {len(features.columns):,}")
    print("Prediction targets:")
    for col in target_cols:
        print(f"- {col}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
