from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_market_summary(features: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    if features.empty:
        summary = pd.DataFrame()
    else:
        summary = pd.DataFrame(
            {
                "metric": [
                    "stocks",
                    "rows",
                    "first_date",
                    "last_date",
                    "average_daily_return",
                    "average_daily_volatility",
                ],
                "value": [
                    features["symbol"].nunique(),
                    len(features),
                    features["date"].min(),
                    features["date"].max(),
                    features["daily_return"].mean(),
                    features["daily_return"].std(),
                ],
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def plot_price_history(features: pd.DataFrame, figures_dir: str | Path, symbols: list[str] | None = None) -> Path | None:
    if features.empty:
        return None

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ModuleNotFoundError:
        print("Skipping price-history chart because matplotlib/seaborn is not installed.")
        return None

    figures_path = ensure_dir(figures_dir)
    available = features["symbol"].dropna().unique().tolist()
    selected = [s for s in (symbols or available[:5]) if s in available]
    if not selected:
        selected = available[:5]

    plot_df = features[features["symbol"].isin(selected)]

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=plot_df, x="date", y="close", hue="symbol", linewidth=1.4)
    plt.title("Historical Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.tight_layout()

    out = figures_path / "phase1_price_history.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_risk_return(summary: pd.DataFrame, figures_dir: str | Path) -> Path | None:
    if summary.empty or {"annualized_return_estimate", "annualized_volatility"}.difference(summary.columns):
        return None

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ModuleNotFoundError:
        print("Skipping risk-return chart because matplotlib/seaborn is not installed.")
        return None

    figures_path = ensure_dir(figures_dir)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=summary,
        x="annualized_volatility",
        y="annualized_return_estimate",
        size="observations",
        sizes=(30, 180),
        alpha=0.75,
    )
    plt.title("Risk vs Return Snapshot")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return Estimate")
    plt.tight_layout()

    out = figures_path / "phase1_risk_return.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_correlation_heatmap(features: pd.DataFrame, figures_dir: str | Path, top_n: int = 15) -> Path | None:
    if features.empty:
        return None

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ModuleNotFoundError:
        print("Skipping correlation heatmap because matplotlib/seaborn is not installed.")
        return None

    figures_path = ensure_dir(figures_dir)
    top_symbols = (
        features.groupby("symbol")["volume"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    returns = features[features["symbol"].isin(top_symbols)].pivot(index="date", columns="symbol", values="daily_return")
    corr = returns.corr()
    if corr.empty:
        return None

    plt.figure(figsize=(11, 8))
    sns.heatmap(corr, cmap="RdYlGn", center=0, linewidths=0.2)
    plt.title("Return Correlation Heatmap")
    plt.tight_layout()

    out = figures_path / "phase1_correlation_heatmap.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return out
