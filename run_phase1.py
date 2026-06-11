from __future__ import annotations

from pathlib import Path

from src.data_loader import load_metadata, load_price_history
from src.eda import plot_correlation_heatmap, plot_price_history, plot_risk_return, save_market_summary
from src.preprocessing import add_basic_market_features, build_stock_summary, clean_price_history


DEFAULT_SETTINGS = {
    "paths": {
        "raw_data_dir": "data/raw",
        "processed_data_dir": "data/processed",
        "outputs_dir": "outputs",
        "figures_dir": "outputs/figures",
    },
    "eda": {
        "sample_symbols": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"],
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


def main() -> int:
    settings = load_settings()
    paths = settings.get("paths", {})
    eda_settings = settings.get("eda", {})

    raw_dir = Path(paths.get("raw_data_dir", "data/raw"))
    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))
    outputs_dir = Path(paths.get("outputs_dir", "outputs"))
    figures_dir = Path(paths.get("figures_dir", "outputs/figures"))

    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    prices = load_price_history(raw_dir)
    metadata = load_metadata(raw_dir)

    if prices.empty:
        print("No price-history CSV files were found in data/raw.")
        print("Download and extract the allowed Kaggle dataset CSV files into data/raw, then run this again.")
        return 1

    clean_prices = clean_price_history(prices)
    features = add_basic_market_features(
        clean_prices,
        rolling_window=int(eda_settings.get("rolling_window_days", 21)),
    )
    stock_summary = build_stock_summary(features)

    clean_prices.to_csv(processed_dir / "phase1_clean_prices.csv", index=False)
    features.to_csv(processed_dir / "phase1_features.csv", index=False)
    stock_summary.to_csv(outputs_dir / "phase1_stock_summary.csv", index=False)
    save_market_summary(features, outputs_dir / "phase1_market_summary.csv")

    if not metadata.empty:
        metadata.to_csv(processed_dir / "phase1_metadata.csv", index=False)

    sample_symbols = eda_settings.get("sample_symbols") or None
    generated_figures = [
        plot_price_history(features, figures_dir, sample_symbols),
        plot_risk_return(stock_summary, figures_dir),
        plot_correlation_heatmap(features, figures_dir),
    ]
    generated_figures = [str(path) for path in generated_figures if path is not None]

    print("Phase 1 complete.")
    print(f"Rows loaded: {len(prices):,}")
    print(f"Rows after cleaning: {len(clean_prices):,}")
    print(f"Stocks found: {clean_prices['symbol'].nunique():,}")
    print(f"Date range: {clean_prices['date'].min().date()} to {clean_prices['date'].max().date()}")
    if generated_figures:
        print("Generated figures:")
        for figure in generated_figures:
            print(f"- {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
