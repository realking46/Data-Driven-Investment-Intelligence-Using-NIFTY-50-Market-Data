from __future__ import annotations

from pathlib import Path

import pandas as pd


DATASETS = {
    "phase1_features": "data/processed/phase1_features.csv",
    "phase2_features": "data/processed/phase2_model_features.csv",
    "market_summary": "outputs/phase1_market_summary.csv",
    "stock_summary": "outputs/phase1_stock_summary.csv",
    "model_metrics": "outputs/phase3_model_metrics.csv",
    "predictions": "outputs/phase3_predictions.csv",
    "feature_importance": "outputs/phase3_feature_importance.csv",
    "stock_risk": "outputs/phase4_stock_risk_metrics.csv",
    "investment_universe": "outputs/phase4_investment_universe.csv",
    "portfolio_allocations": "outputs/phase4_portfolio_allocations.csv",
    "portfolio_metrics": "outputs/phase4_portfolio_metrics.csv",
    "return_correlation": "outputs/phase4_return_correlation.csv",
    "explainable_insights": "outputs/phase5_explainable_insights.csv",
}


DATE_COLUMNS = {
    "phase1_features": ["date"],
    "phase2_features": ["date"],
    "predictions": ["date"],
    "model_metrics": ["train_start", "train_end", "test_start", "test_end"],
}


def read_csv_if_exists(path: str | Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(csv_path, parse_dates=parse_dates or [])
    except ValueError:
        return pd.read_csv(csv_path)


def load_dashboard_data() -> dict[str, pd.DataFrame]:
    return {
        name: read_csv_if_exists(path, DATE_COLUMNS.get(name))
        for name, path in DATASETS.items()
    }


def available_symbols(*frames: pd.DataFrame) -> list[str]:
    symbols: set[str] = set()
    for frame in frames:
        if not frame.empty and "symbol" in frame.columns:
            symbols.update(frame["symbol"].dropna().astype(str).unique().tolist())
    return sorted(symbols)


def latest_rows(frame: pd.DataFrame, group_col: str = "symbol", date_col: str = "date") -> pd.DataFrame:
    if frame.empty or group_col not in frame.columns or date_col not in frame.columns:
        return pd.DataFrame()

    out = frame.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out.sort_values([group_col, date_col]).groupby(group_col, as_index=False).tail(1)


def percent(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def number(value: float | int | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"
