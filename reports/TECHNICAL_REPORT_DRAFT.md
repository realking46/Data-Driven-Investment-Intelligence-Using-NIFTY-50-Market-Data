# Data-Driven Investment Intelligence Using NIFTY-50 Market Data

## 1. Problem Understanding

This project builds an educational investment decision-support platform using the organizer-provided NIFTY-50 historical market datasets. The objective is not to provide financial advice or rely on live market signals, but to transform historical price and volume data into interpretable insights for stock analysis, risk assessment, and portfolio construction.

## 2. Dataset and Constraints

Allowed datasets:

- NIFTY-50 Stock Market Dataset
- Organizer-provided India NSE historical dataset

The solution does not use live market APIs, news data, social media sentiment, proprietary financial data, or alternative external datasets.

## 3. Exploratory Data Analysis

To be completed after running Phase 1 on the provided CSV files.

Recommended content:

- Number of stocks and observations
- Date range covered
- Stock-wise closing price trends
- Return distributions
- Volatility comparison
- Drawdown observations
- Volume and turnover patterns

Generated files:

- `outputs/phase1_market_summary.csv`
- `outputs/phase1_stock_summary.csv`
- `outputs/figures/`

## 4. Feature Engineering

Phase 2 generates historical-data-only features:

- Daily returns and log returns
- Moving averages and exponential moving averages
- RSI
- MACD
- Bollinger Bands
- Momentum features
- Lagged returns
- Forward-return prediction targets

Generated file:

- `data/processed/phase2_model_features.csv`

## 5. Stock Predictor Engine

Phase 3 trains and evaluates:

- Symbol-mean baseline
- Ridge-regression return predictor

Primary target:

- 5-day future return

Evaluation metrics:

- MAE
- RMSE
- R2 score
- Directional accuracy

Generated files:

- `outputs/phase3_model_metrics.csv`
- `outputs/phase3_predictions.csv`
- `outputs/phase3_feature_importance.csv`

## 6. Portfolio Construction Logic

Phase 4 builds three investor profiles:

- Conservative
- Balanced
- Aggressive

The allocation score combines:

- Predicted return
- Sharpe ratio
- Annualized volatility
- Maximum drawdown

Generated file:

- `outputs/phase4_portfolio_allocations.csv`

## 7. Risk Assessment Methodology

Risk metrics:

- Annualized volatility
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Risk-adjusted return
- Return correlation

Generated files:

- `outputs/phase4_stock_risk_metrics.csv`
- `outputs/phase4_portfolio_metrics.csv`
- `outputs/phase4_return_correlation.csv`

## 8. Explainability

Phase 5 converts model, portfolio, and risk outputs into readable evidence-backed insight statements.

Generated files:

- `outputs/phase5_explainable_insights.csv`
- `outputs/phase5_explainable_insights.md`

## 9. Prototype

The Streamlit dashboard includes:

- Market overview
- Stock intelligence
- Predictor results
- Portfolio allocations
- Risk analytics
- Explainable insights

Run:

```bash
streamlit run app.py
```

## 10. Key Insights

To be completed after running the full pipeline on the actual allowed dataset.

Recommended insight types:

- Stocks with strong risk-adjusted historical returns
- Stocks with high volatility or severe drawdown risk
- Differences between Conservative, Balanced, and Aggressive allocations
- Model features that most influence predictions
- Cases where predicted return and historical risk disagree

## 11. Limitations

- Historical performance does not guarantee future results.
- The model uses only organizer-provided historical market data.
- No live market information, news, macroeconomic indicators, or sentiment data are used.
- The predictor is designed for decision support, not autonomous trading.
- Portfolio allocations depend on simplified scoring rules and should be interpreted cautiously.

## 12. Reproducibility

Pipeline:

```bash
python run_phase1.py
python run_phase2.py
python run_phase3.py
python run_phase4.py
python run_phase5.py
streamlit run app.py
```

The project is reproducible from the public repository once the allowed Kaggle CSV files are placed in `data/raw/`.

