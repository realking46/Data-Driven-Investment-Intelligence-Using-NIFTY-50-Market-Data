# Data-Driven Investment Intelligence Using NIFTY-50 Market Data

## Executive Summary

This project builds a complete AI-assisted investment intelligence workflow for NIFTY-50 historical market data. It converts raw stock-market CSV files into cleaned data, technical indicators, prediction targets, model comparisons, portfolio allocations, risk analytics, explainable insight statements, and an interactive dashboard.

The strongest part of the system is the decision-support layer: it does not only forecast returns, but combines prediction, historical risk-adjusted performance, drawdown behavior, and profile-specific allocation logic. The final dashboard and report are designed to help a user understand why a stock or portfolio was recommended.

Key outputs:

- Working Streamlit prototype
- Reproducible five-phase pipeline
- Completed technical report
- Model comparison results
- Conservative, Balanced, and Aggressive portfolios
- Explainable insight outputs

## 1. Problem Understanding

This project develops an educational investment intelligence platform using historical NIFTY-50 market data. The goal is to support data-driven investment decisions through historical analysis, technical indicators, return prediction, risk assessment, portfolio construction, and explainable insights.

The solution is designed as a decision-support system, not as financial advice or an automated trading system. It uses only the organizer-provided historical market dataset and avoids live market APIs, news data, social media sentiment, proprietary financial data, and external alternative datasets.

## 2. Dataset

The project uses the Kaggle NIFTY-50 Stock Market Dataset provided in the problem statement. After extraction, the dataset contains one CSV file per stock.

Pipeline summary:

- Raw rows loaded: 470,384
- Cleaned rows used: 235,192
- Stocks found: 65
- Date range: 2000-01-03 to 2021-04-30

The data includes daily open, high, low, close, traded volume, and turnover-style market fields. Rows were cleaned by parsing dates, standardizing numeric fields, removing invalid price/volume records, filtering equity-series rows where available, sorting by symbol/date, and removing duplicate symbol-date records.

## 3. Exploratory Data Analysis

Phase 1 generates market-level and stock-level summaries. The EDA focuses on:

- Historical price movement
- Daily and rolling returns
- Volatility patterns
- Drawdown behavior
- Risk-return positioning
- Return correlation between stocks

Generated artifacts:

- `outputs/phase1_market_summary.csv`
- `outputs/phase1_stock_summary.csv`
- `outputs/figures/phase1_price_history.png`
- `outputs/figures/phase1_risk_return.png`
- `outputs/figures/phase1_correlation_heatmap.png`

These outputs provide the initial market overview used by the dashboard and later portfolio logic.

Important EDA outputs include:

- Historical price charts for selected stocks
- Risk-return scatter plot
- Return correlation heatmap
- Stock-level summary table with return, volatility, drawdown, and Sharpe-style estimates

The purpose of the EDA is to identify market behavior before modeling, so that model outputs can be interpreted in the context of historical volatility and drawdown risk.

## 4. Feature Engineering

Phase 2 transforms cleaned historical data into model-ready features. The system creates:

- Daily returns and log returns
- Close-open returns
- High-low spread
- Rolling 21-day return and volatility
- Simple moving averages
- Exponential moving averages
- Price-to-moving-average ratios
- RSI
- MACD, MACD signal, and MACD histogram
- Bollinger Band width and position
- Momentum features
- Lagged return and volume features
- Forward-return targets for 1-day, 5-day, and 21-day horizons

The final Phase 2 dataset contains 235,192 rows and 87 feature columns.

Feature engineering follows the competition constraint that only the provided market data may be used. All indicators are derived from open, high, low, close, volume, turnover, date, and symbol fields. No external market data, financial API data, sentiment data, or news data is used.

## 5. Stock Predictor Engine

The target used for the main experiment is `future_return_5d`, a 5-day forward return. The split is time-based to avoid training on future observations:

- Training period: 2000-01-03 to 2017-01-06
- Test period: 2017-01-09 to 2021-04-23
- Training rows: 182,878
- Test rows: 51,989

Models tested:

| Model | MAE | RMSE | R2 | Directional Accuracy |
|---|---:|---:|---:|---:|
| Symbol mean baseline | 0.033955 | 0.051309 | -0.001602 | 52.24% |
| Extra Trees | 0.033970 | 0.051400 | -0.005160 | 51.90% |
| Ridge Regression | 0.034763 | 0.052359 | -0.043010 | 51.30% |
| XGBoost Regressor | 0.035564 | 0.053267 | -0.079495 | 50.23% |
| Hist Gradient Boosting | 0.037950 | 0.057212 | -0.245333 | 50.45% |

The symbol-mean baseline performs best by RMSE. This is an important modeling insight: on this 5-day return target, simple historical symbol-level behavior is difficult to beat using the current feature set. XGBoost was added to the comparison, but it did not outperform the baseline. The project therefore uses the best model by test RMSE for downstream portfolio construction and reports the comparison transparently.

This result is useful rather than a failure: it shows that short-horizon stock returns are noisy and that model complexity alone does not guarantee better investment intelligence. The project therefore treats forecasting as one signal among several, not as the sole decision rule.

## 6. Portfolio Construction

Phase 4 builds three investor-profile portfolios:

- Conservative
- Balanced
- Aggressive

The scoring logic combines:

- Predicted return
- Sharpe ratio
- Annualized volatility
- Maximum drawdown

The profile weights change by investor type. Conservative portfolios emphasize lower volatility and drawdown, Balanced portfolios combine risk and return, and Aggressive portfolios give more weight to expected return and momentum.

Top allocations:

| Profile | Top Stocks |
|---|---|
| Conservative | HINDUNILVR, NESTLEIND, MARUTI, SHREECEM, UPL |
| Balanced | UPL, SHREECEM, HINDUNILVR, NESTLEIND |
| Aggressive | UPL, SHREECEM, EICHERMOT |

Allocation weights are capped to avoid one-stock concentration. This makes the portfolios more realistic for decision support and helps each profile reflect a diversified set of risk-return trade-offs.

## 7. Risk Assessment

Risk metrics include:

- Annualized return
- Annualized volatility
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Return correlation

Portfolio-level results:

| Profile | Stocks | Annualized Return | Annualized Volatility | Sharpe | Sortino | Max Drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Aggressive | 10 | 21.74% | 18.10% | 1.201 | 1.571 | -45.67% |
| Balanced | 10 | 18.83% | 15.31% | 1.230 | 1.642 | -38.25% |
| Conservative | 8 | 17.37% | 13.77% | 1.261 | 1.790 | -29.81% |

This behavior is aligned with the profile design: Conservative has the lowest volatility and drawdown, while Aggressive has the highest estimated annualized return.

Strong historical risk-adjusted stocks by Sharpe ratio include BHARTI, UTIBANK, SHREECEM, BAJFINANCE, NESTLEIND, HINDUNILVR, BAJAJFINSV, and MARUTI.

## 8. Explainability

The system generates explainable insight statements from:

- Portfolio allocations
- Stock risk metrics
- Model feature importance

Phase 5 generated 53 insight statements. Each statement includes a category, profile, symbol, human-readable insight, and quantitative evidence. These explanations are visible in the dashboard's Insights tab and saved in:

- `outputs/phase5_explainable_insights.csv`
- `outputs/phase5_explainable_insights.md`

Example explanation style:

> HINDUNILVR is included in the Conservative portfolio because it scores well on the selected risk-return criteria, supported by its portfolio weight, investment score, predicted return, Sharpe ratio, volatility, and maximum drawdown.

The explanation layer is intentionally evidence-based. It avoids unsupported language such as "guaranteed return" or "best stock" and instead presents the quantitative reason each signal appears in the system.

## 9. Working Prototype

The project includes a Streamlit dashboard with these sections:

- Overview
- Stock Intelligence
- Predictor
- Portfolio
- Risk
- Insights

Run command:

```bash
streamlit run app.py
```

The dashboard reads generated CSV outputs and updates after rerunning the pipeline.

The dashboard is organized around investor workflow:

1. Understand the market.
2. Inspect an individual stock.
3. Review predictor performance.
4. Compare portfolio allocations.
5. Analyze risk.
6. Read explanation statements.

## 10. Key Insights

The current system shows that:

- Simple baseline prediction remains hard to beat for short-horizon 5-day returns.
- Extra Trees nearly matches the baseline and is the best nonlinear model in the current experiment.
- Conservative portfolio design successfully reduces volatility and drawdown.
- Balanced portfolio has a strong risk-return trade-off with Sharpe above 1.2.
- Aggressive allocation increases annualized return but accepts materially higher drawdown.
- Explainability improves trust by connecting each allocation to measurable return and risk signals.

## 11. Limitations

- Historical performance does not guarantee future returns.
- The model does not use live market information, news, sentiment, macroeconomic indicators, or fundamentals.
- The prediction target is short horizon and noisy, making high R2 difficult.
- The portfolio optimizer uses a transparent scoring rule rather than a full mean-variance optimization framework.
- Some stocks in the dataset have shorter trading histories than others.
- Results should be interpreted as educational decision support, not investment advice.

## 12. Evaluation Criteria Alignment

The project aligns with the official evaluation criteria as follows:

| Evaluation Area | Project Response |
|---|---|
| Quality of Investment Insights | EDA summaries, risk-return analysis, portfolio metrics, and generated explanations |
| Technical Innovation | Multi-phase pipeline combining indicators, model comparison, risk analytics, allocation scoring, and dashboard |
| Portfolio & Risk Analytics | Conservative/Balanced/Aggressive portfolios with Sharpe, Sortino, volatility, drawdown, and correlation |
| Explainability & Transparency | Evidence-backed insights and transparent reporting of model limitations |
| Reproducibility & Documentation | README, phase runners, modular source code, report, and dashboard instructions |

## 13. Reproducibility

After placing the allowed dataset files in `data/raw`, run:

```bash
python run_phase1.py
python run_phase2.py
python run_phase3.py
python run_phase4.py
python run_phase5.py
streamlit run app.py
```

Main outputs are stored in `data/processed/` and `outputs/`. The project includes a README, source modules, reproducible phase runners, dashboard app, and report files.

## 14. Conclusion

The final system satisfies the mandatory requirements of the problem statement: stock behavior prediction, portfolio construction, risk assessment, explainability, and a working prototype. The project also documents an important practical finding: for the current 5-day return target, the simple baseline outperforms more complex models, so investment intelligence should combine prediction with risk analytics and transparent reasoning rather than relying on forecasting alone.

Future improvements could include more robust walk-forward validation, additional investor constraints, a full mean-variance optimization module, richer sector-level analysis, and model calibration for longer investment horizons.
