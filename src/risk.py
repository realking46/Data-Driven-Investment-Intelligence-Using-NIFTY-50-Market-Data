from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def build_returns_matrix(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()

    if "daily_return" in features.columns:
        returns = features.pivot(index="date", columns="symbol", values="daily_return")
    else:
        close = features.pivot(index="date", columns="symbol", values="close")
        returns = close.pct_change()

    return returns.sort_index().replace([np.inf, -np.inf], np.nan)


def max_drawdown(return_series: pd.Series) -> float:
    clean = return_series.dropna()
    if clean.empty:
        return np.nan

    cumulative = (1 + clean).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1
    return float(drawdown.min())


def sortino_ratio(return_series: pd.Series, risk_free_rate: float = 0.0) -> float:
    clean = return_series.dropna()
    if clean.empty:
        return np.nan

    daily_rf = risk_free_rate / TRADING_DAYS
    excess = clean - daily_rf
    downside = excess[excess < 0]
    downside_deviation = downside.std() * np.sqrt(TRADING_DAYS)
    if downside_deviation == 0 or np.isnan(downside_deviation):
        return np.nan
    return float((excess.mean() * TRADING_DAYS) / downside_deviation)


def compute_stock_risk_metrics(features: pd.DataFrame, risk_free_rate: float = 0.0) -> pd.DataFrame:
    returns = build_returns_matrix(features)
    if returns.empty:
        return pd.DataFrame()

    rows = []
    daily_rf = risk_free_rate / TRADING_DAYS

    for symbol in returns.columns:
        series = returns[symbol].dropna()
        if series.empty:
            continue

        annualized_return = series.mean() * TRADING_DAYS
        annualized_volatility = series.std() * np.sqrt(TRADING_DAYS)
        excess_return = (series - daily_rf).mean() * TRADING_DAYS
        sharpe = excess_return / annualized_volatility if annualized_volatility != 0 else np.nan
        mdd = max_drawdown(series)

        rows.append(
            {
                "symbol": symbol,
                "observations": int(series.count()),
                "annualized_return": float(annualized_return),
                "annualized_volatility": float(annualized_volatility),
                "sharpe_ratio": float(sharpe) if not np.isnan(sharpe) else np.nan,
                "sortino_ratio": sortino_ratio(series, risk_free_rate=risk_free_rate),
                "max_drawdown": mdd,
                "risk_adjusted_return": float(annualized_return / abs(mdd)) if mdd not in (0, np.nan) and not np.isnan(mdd) else np.nan,
                "avg_daily_return": float(series.mean()),
                "daily_volatility": float(series.std()),
            }
        )

    return pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)


def compute_correlation_matrix(features: pd.DataFrame) -> pd.DataFrame:
    returns = build_returns_matrix(features)
    if returns.empty:
        return pd.DataFrame()
    return returns.corr()


def compute_portfolio_metrics(
    features: pd.DataFrame,
    allocations: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    returns = build_returns_matrix(features)
    if returns.empty or allocations.empty:
        return pd.DataFrame()

    rows = []
    daily_rf = risk_free_rate / TRADING_DAYS

    for profile, group in allocations.groupby("profile"):
        weights = group.set_index("symbol")["weight"]
        common_symbols = [symbol for symbol in weights.index if symbol in returns.columns]
        if not common_symbols:
            continue

        aligned_returns = returns[common_symbols].dropna(how="all").fillna(0)
        aligned_weights = weights.loc[common_symbols]
        aligned_weights = aligned_weights / aligned_weights.sum()
        portfolio_returns = aligned_returns @ aligned_weights

        annualized_return = portfolio_returns.mean() * TRADING_DAYS
        annualized_volatility = portfolio_returns.std() * np.sqrt(TRADING_DAYS)
        excess_return = (portfolio_returns - daily_rf).mean() * TRADING_DAYS
        sharpe = excess_return / annualized_volatility if annualized_volatility != 0 else np.nan
        mdd = max_drawdown(portfolio_returns)

        rows.append(
            {
                "profile": profile,
                "stocks": len(common_symbols),
                "annualized_return": float(annualized_return),
                "annualized_volatility": float(annualized_volatility),
                "sharpe_ratio": float(sharpe) if not np.isnan(sharpe) else np.nan,
                "sortino_ratio": sortino_ratio(portfolio_returns, risk_free_rate=risk_free_rate),
                "max_drawdown": mdd,
            }
        )

    return pd.DataFrame(rows)

