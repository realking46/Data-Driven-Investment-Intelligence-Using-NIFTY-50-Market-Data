from __future__ import annotations

import numpy as np
import pandas as pd


def clean_price_history(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()

    df = prices.copy()
    df = df.dropna(subset=["date", "symbol", "open", "high", "low", "close", "volume"])

    if "series" in df.columns:
        equity_rows = df["series"].astype(str).str.upper().eq("EQ")
        if equity_rows.any():
            df = df[equity_rows]

    df = df[df["volume"] >= 0]
    df = df[df["high"] >= df["low"]]
    df = df[df["close"] > 0]
    df = df.sort_values(["symbol", "date"])
    df = df.drop_duplicates(subset=["symbol", "date"], keep="last")
    return df.reset_index(drop=True)


def add_basic_market_features(prices: pd.DataFrame, rolling_window: int = 21) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()

    df = prices.sort_values(["symbol", "date"]).copy()
    grouped = df.groupby("symbol", group_keys=False)

    df["daily_return"] = grouped["close"].pct_change()
    df["log_return"] = grouped["close"].transform(lambda s: np.log(s).diff())
    df["close_open_return"] = (df["close"] - df["open"]) / df["open"]
    df["high_low_spread"] = (df["high"] - df["low"]) / df["close"]
    df["volume_change"] = grouped["volume"].pct_change()

    if "turnover" in df.columns:
        df["turnover_change"] = grouped["turnover"].pct_change()

    df[f"rolling_{rolling_window}d_return"] = grouped["close"].pct_change(rolling_window)
    df[f"rolling_{rolling_window}d_volatility"] = grouped["daily_return"].transform(
        lambda s: s.rolling(rolling_window, min_periods=max(5, rolling_window // 3)).std()
    )
    df[f"moving_average_{rolling_window}d"] = grouped["close"].transform(
        lambda s: s.rolling(rolling_window, min_periods=max(5, rolling_window // 3)).mean()
    )

    running_max = grouped["close"].cummax()
    df["drawdown"] = (df["close"] / running_max) - 1
    return df.replace([np.inf, -np.inf], np.nan)


def build_stock_summary(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()

    summary = (
        features.groupby("symbol")
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            observations=("date", "count"),
            latest_close=("close", "last"),
            mean_daily_return=("daily_return", "mean"),
            volatility=("daily_return", "std"),
            max_drawdown=("drawdown", "min"),
            avg_volume=("volume", "mean"),
        )
        .reset_index()
    )
    summary["annualized_return_estimate"] = summary["mean_daily_return"] * 252
    summary["annualized_volatility"] = summary["volatility"] * np.sqrt(252)
    summary["simple_sharpe_proxy"] = summary["annualized_return_estimate"] / summary["annualized_volatility"]
    return summary.sort_values("simple_sharpe_proxy", ascending=False)

