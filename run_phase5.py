from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.explainability import build_explainable_insights, write_insights_markdown


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> int:
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    allocations = read_csv(outputs_dir / "phase4_portfolio_allocations.csv")
    stock_risk = read_csv(outputs_dir / "phase4_stock_risk_metrics.csv")
    feature_importance = read_csv(outputs_dir / "phase3_feature_importance.csv")

    if allocations.empty and stock_risk.empty and feature_importance.empty:
        print("No Phase 3 or Phase 4 outputs were found.")
        print("Run python run_phase1.py through python run_phase4.py first.")
        return 1

    insights = build_explainable_insights(allocations, stock_risk, feature_importance)
    insights.to_csv(outputs_dir / "phase5_explainable_insights.csv", index=False)
    write_insights_markdown(insights, str(outputs_dir / "phase5_explainable_insights.md"))

    print("Phase 5 complete.")
    print(f"Insights generated: {len(insights):,}")
    print("Saved outputs:")
    print("- outputs/phase5_explainable_insights.csv")
    print("- outputs/phase5_explainable_insights.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

