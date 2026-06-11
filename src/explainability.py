from __future__ import annotations

import pandas as pd


def _fmt_pct(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _fmt_num(value: float | int | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def explain_portfolio_allocations(allocations: pd.DataFrame) -> pd.DataFrame:
    if allocations.empty:
        return pd.DataFrame(columns=["category", "profile", "symbol", "insight", "evidence"])

    rows = []
    for _, row in allocations.iterrows():
        symbol = row.get("symbol")
        profile = row.get("profile")
        insight = f"{symbol} is included in the {profile} portfolio because it scores well on the selected risk-return criteria."
        evidence = (
            f"Weight {_fmt_pct(row.get('weight'))}; "
            f"investment score {_fmt_num(row.get('investment_score'))}; "
            f"predicted return {_fmt_pct(row.get('predicted_return'))}; "
            f"Sharpe {_fmt_num(row.get('sharpe_ratio'))}; "
            f"volatility {_fmt_pct(row.get('annualized_volatility'))}; "
            f"max drawdown {_fmt_pct(row.get('max_drawdown'))}."
        )
        rows.append(
            {
                "category": "Portfolio Allocation",
                "profile": profile,
                "symbol": symbol,
                "insight": insight,
                "evidence": evidence,
            }
        )

    return pd.DataFrame(rows)


def explain_risk_flags(stock_risk: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    if stock_risk.empty:
        return pd.DataFrame(columns=["category", "profile", "symbol", "insight", "evidence"])

    rows = []
    high_vol = stock_risk.sort_values("annualized_volatility", ascending=False).head(top_n)
    severe_drawdown = stock_risk.sort_values("max_drawdown", ascending=True).head(top_n)
    strong_sharpe = stock_risk.sort_values("sharpe_ratio", ascending=False).head(top_n)

    for _, row in high_vol.iterrows():
        rows.append(
            {
                "category": "Risk Flag",
                "profile": "All",
                "symbol": row.get("symbol"),
                "insight": f"{row.get('symbol')} has elevated historical volatility.",
                "evidence": f"Annualized volatility is {_fmt_pct(row.get('annualized_volatility'))}.",
            }
        )

    for _, row in severe_drawdown.iterrows():
        rows.append(
            {
                "category": "Risk Flag",
                "profile": "All",
                "symbol": row.get("symbol"),
                "insight": f"{row.get('symbol')} experienced one of the deeper historical drawdowns.",
                "evidence": f"Maximum drawdown is {_fmt_pct(row.get('max_drawdown'))}.",
            }
        )

    for _, row in strong_sharpe.iterrows():
        rows.append(
            {
                "category": "Positive Signal",
                "profile": "All",
                "symbol": row.get("symbol"),
                "insight": f"{row.get('symbol')} has one of the stronger historical risk-adjusted return profiles.",
                "evidence": f"Sharpe ratio is {_fmt_num(row.get('sharpe_ratio'))}; annualized return is {_fmt_pct(row.get('annualized_return'))}.",
            }
        )

    return pd.DataFrame(rows).drop_duplicates(subset=["category", "symbol", "insight"]).reset_index(drop=True)


def explain_model_importance(feature_importance: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if feature_importance.empty:
        return pd.DataFrame(columns=["category", "profile", "symbol", "insight", "evidence"])

    ranking_column = "absolute_coefficient" if "absolute_coefficient" in feature_importance.columns else "importance"
    top = feature_importance.sort_values(ranking_column, ascending=False).head(top_n)
    rows = []
    for _, row in top.iterrows():
        feature = row.get("feature")
        model = row.get("model", "model")
        coefficient = row.get("coefficient")
        importance = row.get("importance")
        if pd.notna(coefficient):
            direction = "positive" if coefficient >= 0 else "negative"
            evidence = f"The standardized coefficient is {_fmt_num(coefficient, 5)}, giving it a {direction} relationship with the target in {model}."
        else:
            evidence = f"The reported feature importance is {_fmt_num(importance, 5)} in {model}."
        rows.append(
            {
                "category": "Model Explanation",
                "profile": "All",
                "symbol": "All",
                "insight": f"{feature} is an influential feature in the return predictor.",
                "evidence": evidence,
            }
        )
    return pd.DataFrame(rows)


def build_explainable_insights(
    allocations: pd.DataFrame,
    stock_risk: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> pd.DataFrame:
    frames = [
        explain_portfolio_allocations(allocations),
        explain_risk_flags(stock_risk),
        explain_model_importance(feature_importance),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["category", "profile", "symbol", "insight", "evidence"])
    return pd.concat(frames, ignore_index=True)


def write_insights_markdown(insights: pd.DataFrame, output_path: str) -> None:
    lines = [
        "# Explainable Investment Insights",
        "",
        "These statements are generated from model outputs, portfolio allocations, and historical risk metrics.",
        "They are intended for educational decision support, not financial advice.",
        "",
    ]

    if insights.empty:
        lines.append("No insights are available yet. Run Phases 1-5 after adding the allowed Kaggle data.")
    else:
        for category, group in insights.groupby("category", sort=False):
            lines.extend([f"## {category}", ""])
            for _, row in group.iterrows():
                prefix = row.get("symbol", "All")
                profile = row.get("profile", "All")
                lines.append(f"- **{prefix}** ({profile}): {row.get('insight')} {row.get('evidence')}")
            lines.append("")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines).strip() + "\n")
