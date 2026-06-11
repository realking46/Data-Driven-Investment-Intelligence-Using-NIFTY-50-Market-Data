from __future__ import annotations

import numpy as np
import pandas as pd


def _grouped_close(df: pd.DataFrame):
    return df.sort_values(["symbol", "date"]).groupby("symbol", group_keys=False)["close"]


def add_moving_averages(prices: pd.DataFrame, windows: tuple[int, ...] = (7, 21, 50, 100, 200)) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()

    df = prices.sort_values(["symbol", "date"]).copy()
    grouped = _grouped_close(df)

    for window in windows:
        min_periods = max(3, window // 3)
        df[f"sma_{window}"] = grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
        df[f"ema_{window}"] = grouped.transform(lambda s: s.ewm(span=window, adjust=False, min_periods=min_periods).mean())
        df[f"price_to_sma_{window}"] = (df["close"] / df[f"sma_{window}"]) - 1

    return df.replace([np.inf, -np.inf], np.nan)


def add_rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()

    df = prices.sort_values(["symbol", "date"]).copy()

    def compute_rsi(close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    df[f"rsi_{window}"] = _grouped_close(df).transform(compute_rsi)
    return df.replace([np.inf, -np.inf], np.nan)


def add_macd(prices: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()

    df = prices.sort_values(["symbol", "date"]).copy()

    def compute_macd(close: pd.Series) -> pd.DataFrame:
        ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame(
            {
                "macd": macd_line,
                "macd_signal": signal_line,
                "macd_histogram": histogram,
            },
            index=close.index,
        )

    macd = df.groupby("symbol", group_keys=False)["close"].apply(compute_macd)
    for col in ["macd", "macd_signal", "macd_histogram"]:
        df[col] = macd[col]
    return df.replace([np.inf, -np.inf], np.nan)


def add_bollinger_bands(prices: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()

    df = prices.sort_values(["symbol", "date"]).copy()
    grouped = _grouped_close(df)
    min_periods = max(5, window // 3)

    middle = grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
    rolling_std = grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).std())

    df[f"bb_middle_{window}"] = middle
    df[f"bb_upper_{window}"] = middle + (num_std * rolling_std)
    df[f"bb_lower_{window}"] = middle - (num_std * rolling_std)
    df[f"bb_width_{window}"] = (df[f"bb_upper_{window}"] - df[f"bb_lower_{window}"]) / middle
    df[f"bb_position_{window}"] = (df["close"] - df[f"bb_lower_{window}"]) / (
        df[f"bb_upper_{window}"] - df[f"bb_lower_{window}"]
    )
    return df.replace([np.inf, -np.inf], np.nan)


def add_momentum_features(prices: pd.DataFrame, windows: tuple[int, ...] = (5, 10, 21, 63)) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()

    df = prices.sort_values(["symbol", "date"]).copy()
    grouped = _grouped_close(df)

    for window in windows:
        df[f"return_{window}d"] = grouped.pct_change(window)
        df[f"momentum_{window}d"] = grouped.transform(lambda s: s - s.shift(window))

    return df.replace([np.inf, -np.inf], np.nan)


def add_lagged_features(prices: pd.DataFrame, lags: tuple[int, ...] = (1, 2, 3, 5, 10)) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()

    df = prices.sort_values(["symbol", "date"]).copy()
    grouped = df.groupby("symbol", group_keys=False)

    base_columns = ["daily_return", "log_return", "volume_change", "high_low_spread"]
    for col in base_columns:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag_{lag}"] = grouped[col].shift(lag)

    return df.replace([np.inf, -np.inf], np.nan)


def add_prediction_targets(prices: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 21)) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()

    df = prices.sort_values(["symbol", "date"]).copy()
    grouped = _grouped_close(df)

    for horizon in horizons:
        future_close = grouped.shift(-horizon)
        df[f"future_close_{horizon}d"] = future_close
        df[f"future_return_{horizon}d"] = (future_close / df["close"]) - 1
        df[f"target_direction_{horizon}d"] = np.where(df[f"future_return_{horizon}d"] > 0, 1, 0)
        df.loc[df[f"future_return_{horizon}d"].isna(), f"target_direction_{horizon}d"] = np.nan

    return df.replace([np.inf, -np.inf], np.nan)


def build_phase2_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = add_moving_averages(prices)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_momentum_features(df)
    df = add_lagged_features(df)
    df = add_prediction_targets(df)
    return df

