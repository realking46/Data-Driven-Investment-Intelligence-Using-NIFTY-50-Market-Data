from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models import train_and_evaluate


DEFAULT_SETTINGS = {
    "paths": {
        "processed_data_dir": "data/processed",
        "outputs_dir": "outputs",
    },
    "modeling": {
        "target_column": "future_return_5d",
        "ridge_alpha": 5.0,
        "test_fraction": 0.2,
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
    merged["modeling"] = {**DEFAULT_SETTINGS["modeling"], **loaded.get("modeling", {})}
    return merged


def main() -> int:
    settings = load_settings()
    paths = settings.get("paths", {})
    modeling = settings.get("modeling", {})

    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))
    outputs_dir = Path(paths.get("outputs_dir", "outputs"))
    outputs_dir.mkdir(parents=True, exist_ok=True)

    feature_path = processed_dir / "phase2_model_features.csv"
    if not feature_path.exists():
        print("Phase 2 feature file was not found.")
        print("Run python run_phase1.py and python run_phase2.py after placing the allowed Kaggle CSVs in data/raw.")
        return 1

    features = pd.read_csv(feature_path, parse_dates=["date"])
    target_column = modeling.get("target_column", "future_return_5d")
    alpha = float(modeling.get("ridge_alpha", 5.0))
    test_fraction = float(modeling.get("test_fraction", 0.2))

    metrics, predictions, importance = train_and_evaluate(
        features,
        target_column=target_column,
        alpha=alpha,
        test_fraction=test_fraction,
    )

    metrics.to_csv(outputs_dir / "phase3_model_metrics.csv", index=False)
    predictions.to_csv(outputs_dir / "phase3_predictions.csv", index=False)
    importance.to_csv(outputs_dir / "phase3_feature_importance.csv", index=False)

    print("Phase 3 complete.")
    print(f"Target: {target_column}")
    print("Model metrics:")
    print(metrics[["model", "mae", "rmse", "r2", "directional_accuracy"]].to_string(index=False))
    print("Saved outputs:")
    print("- outputs/phase3_model_metrics.csv")
    print("- outputs/phase3_predictions.csv")
    print("- outputs/phase3_feature_importance.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

