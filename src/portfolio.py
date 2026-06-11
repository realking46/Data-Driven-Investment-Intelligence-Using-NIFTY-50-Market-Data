from __future__ import annotations

import numpy as np
import pandas as pd


PROFILE_CONFIG = {
    "Conservative": {
        "top_n": 8,
        "max_weight": 0.22,
        "score_weights": {
            "predicted_return": 0.20,
            "sharpe_ratio": 0.35,
            "annualized_volatility": -0.25,
            "max_drawdown": 0.20,
        },
    },
    "Balanced": {
        "top_n": 10,
        "max_weight": 0.28,
        "score_weights": {
            "predicted_return": 0.35,
            "sharpe_ratio": 0.30,
            "annualized_volatility": -0.20,
            "max_drawdown": 0.15,
        },
    },
    "Aggressive": {
        "top_n": 10,
        "max_weight": 0.35,
        "score_weights": {
            "predicted_return": 0.50,
            "sharpe_ratio": 0.25,
            "annualized_volatility": -0.10,
            "max_drawdown": 0.15,
        },
    },
}


def latest_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["symbol", "predicted_return"])

    df = predictions.copy()
    prediction_column = "best_model_prediction" if "best_model_prediction" in df.columns else "ridge_prediction"
    if prediction_column not in df.columns:
        return pd.DataFrame(columns=["symbol", "predicted_return"])

    df["date"] = pd.to_datetime(df["date"])
    latest = df.sort_values(["symbol", "date"]).groupby("symbol", as_index=False).tail(1)
    return latest[["symbol", prediction_column]].rename(columns={prediction_column: "predicted_return"})


def _normalize(series: pd.Series) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan)
    min_value = clean.min()
    max_value = clean.max()
    if pd.isna(min_value) or pd.isna(max_value) or min_value == max_value:
        return pd.Series(0.5, index=series.index)
    return (clean - min_value) / (max_value - min_value)


def build_investment_universe(risk_metrics: pd.DataFrame, predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    if risk_metrics.empty:
        return pd.DataFrame()

    universe = risk_metrics.copy()
    if predictions is not None and not predictions.empty:
        universe = universe.merge(latest_predictions(predictions), on="symbol", how="left")
    else:
        universe["predicted_return"] = universe["annualized_return"] / 252

    universe["predicted_return"] = universe["predicted_return"].fillna(universe["annualized_return"] / 252)
    universe = universe.dropna(subset=["annualized_volatility", "max_drawdown"])
    return universe.reset_index(drop=True)


def score_universe(universe: pd.DataFrame, profile: str) -> pd.DataFrame:
    if universe.empty:
        return universe.copy()

    config = PROFILE_CONFIG[profile]
    scored = universe.copy()

    scored["predicted_return_norm"] = _normalize(scored["predicted_return"])
    scored["sharpe_ratio_norm"] = _normalize(scored["sharpe_ratio"])
    scored["annualized_volatility_norm"] = _normalize(scored["annualized_volatility"])
    scored["max_drawdown_norm"] = _normalize(scored["max_drawdown"])

    scored["investment_score"] = 0.0
    for metric, weight in config["score_weights"].items():
        scored["investment_score"] += scored[f"{metric}_norm"] * weight

    return scored.sort_values("investment_score", ascending=False).reset_index(drop=True)


def _cap_and_normalize_weights(raw_scores: pd.Series, max_weight: float) -> pd.Series:
    scores = raw_scores.clip(lower=0)
    if scores.sum() == 0:
        weights = pd.Series(1 / len(scores), index=scores.index)
    else:
        weights = scores / scores.sum()

    for _ in range(20):
        over_cap = weights > max_weight
        if not over_cap.any():
            break
        capped_total = max_weight * over_cap.sum()
        remaining = weights[~over_cap]
        if remaining.empty:
            weights[:] = 1 / len(weights)
            break
        weights[over_cap] = max_weight
        weights[~over_cap] = remaining / remaining.sum() * (1 - capped_total)

    return weights / weights.sum()


def construct_portfolio(universe: pd.DataFrame, profile: str) -> pd.DataFrame:
    scored = score_universe(universe, profile)
    if scored.empty:
        return pd.DataFrame()

    config = PROFILE_CONFIG[profile]
    selected = scored.head(config["top_n"]).copy()
    shifted_scores = selected["investment_score"] - selected["investment_score"].min() + 0.01
    selected["weight"] = _cap_and_normalize_weights(shifted_scores, config["max_weight"]).to_numpy()
    selected["profile"] = profile

    explanation_parts = []
    for _, row in selected.iterrows():
        reason = (
            f"score={row['investment_score']:.3f}; "
            f"predicted_return={row['predicted_return']:.4f}; "
            f"sharpe={row['sharpe_ratio']:.3f}; "
            f"volatility={row['annualized_volatility']:.3f}; "
            f"max_drawdown={row['max_drawdown']:.3f}"
        )
        explanation_parts.append(reason)
    selected["allocation_reason"] = explanation_parts

    columns = [
        "profile",
        "symbol",
        "weight",
        "investment_score",
        "predicted_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "allocation_reason",
    ]
    return selected[columns].sort_values(["profile", "weight"], ascending=[True, False]).reset_index(drop=True)


def construct_all_portfolios(universe: pd.DataFrame) -> pd.DataFrame:
    portfolios = [construct_portfolio(universe, profile) for profile in PROFILE_CONFIG]
    portfolios = [portfolio for portfolio in portfolios if not portfolio.empty]
    if not portfolios:
        return pd.DataFrame()
    return pd.concat(portfolios, ignore_index=True)
