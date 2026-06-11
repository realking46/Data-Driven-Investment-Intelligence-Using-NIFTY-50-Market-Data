# NIFTY-50 Investment Intelligence

An educational investment decision-support platform for the Cult Open Projects 2026 problem statement: **Data-Driven Investment Intelligence Using NIFTY-50 Market Data**.

The system transforms historical NIFTY-50 stock data into market insights, technical indicators, return predictions, risk metrics, portfolio allocations, and explainable recommendation statements. It is designed for decision support and analysis, not financial advice.

## Project Highlights

- Uses only the organizer-provided NIFTY-50 Kaggle dataset.
- Processes 470,384 raw rows into 235,192 cleaned rows.
- Covers 65 stocks from 2000-01-03 to 2021-04-30.
- Generates 87 model-ready feature columns.
- Compares baseline, linear, and tree-based return predictors.
- Builds Conservative, Balanced, and Aggressive portfolios.
- Computes volatility, Sharpe ratio, Sortino ratio, maximum drawdown, and correlations.
- Generates 53 explainable insight statements.
- Includes a Streamlit dashboard prototype.
- Includes a completed technical report PDF.

## Final Results Summary

### Predictor Results

Target: 5-day future return (`future_return_5d`)

| Model | MAE | RMSE | R2 | Directional Accuracy |
|---|---:|---:|---:|---:|
| Symbol mean baseline | 0.033955 | 0.051309 | -0.001602 | 52.24% |
| Extra Trees | 0.033970 | 0.051400 | -0.005160 | 51.90% |
| Ridge Regression | 0.034763 | 0.052359 | -0.043010 | 51.30% |
| XGBoost Regressor | 0.035564 | 0.053267 | -0.079495 | 50.23% |
| Hist Gradient Boosting | 0.037950 | 0.057212 | -0.245333 | 50.45% |

The symbol-mean baseline performed best by RMSE. This is reported transparently because short-horizon stock-return prediction is noisy, and stronger models, including XGBoost, did not improve the current target.


### Portfolio Results

| Profile | Stocks | Annualized Return | Annualized Volatility | Sharpe | Sortino | Max Drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Aggressive | 10 | 21.74% | 18.10% | 1.201 | 1.571 | -45.67% |
| Balanced | 10 | 18.83% | 15.31% | 1.230 | 1.642 | -38.25% |
| Conservative | 8 | 17.37% | 13.77% | 1.261 | 1.790 | -29.81% |

The portfolio behavior matches the intended investor profiles: Conservative has the lowest volatility and drawdown, while Aggressive targets higher return with higher risk.

## Dataset Setup

Download the allowed Kaggle dataset and place the extracted CSV files inside:

```text
data/raw/
```

Allowed dataset:

- NIFTY-50 Stock Market Dataset: https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data

Optional organizer-provided dataset:

- India Stock Data NSE 1990-2020: https://www.kaggle.com/datasets/stoicstatic/india-stock-data-nse-1990-2020

Do not use live market data, financial APIs, news data, social media sentiment, proprietary financial data, or other external market datasets.

## Installation

```bash
pip install -r requirements.txt
```

If your default `python` does not have the dependencies installed, run the commands from the Python environment where `requirements.txt` was installed.

## Run the Full Pipeline

```bash
python run_phase1.py
python run_phase2.py
python run_phase3.py
python run_phase4.py
python run_phase5.py
```

### Phase Outputs

Phase 1: data cleaning and EDA

- `data/processed/phase1_clean_prices.csv`
- `data/processed/phase1_features.csv`
- `outputs/phase1_market_summary.csv`
- `outputs/phase1_stock_summary.csv`
- `outputs/figures/`

Phase 2: technical indicators and model features

- `data/processed/phase2_model_features.csv`
- `outputs/phase2_feature_summary.csv`

Phase 3: stock predictor engine

- `outputs/phase3_model_metrics.csv`
- `outputs/phase3_predictions.csv`
- `outputs/phase3_feature_importance.csv`

Phase 4: portfolio and risk analytics

- `outputs/phase4_stock_risk_metrics.csv`
- `outputs/phase4_investment_universe.csv`
- `outputs/phase4_portfolio_allocations.csv`
- `outputs/phase4_portfolio_metrics.csv`
- `outputs/phase4_return_correlation.csv`

Phase 5: explainable insights

- `outputs/phase5_explainable_insights.csv`
- `outputs/phase5_explainable_insights.md`

## Run the Dashboard

```bash
streamlit run app.py
```

Dashboard sections:

- Overview
- Stock Intelligence
- Predictor
- Portfolio
- Risk
- Insights

## Reports

Final report files:

- `reports/TECHNICAL_REPORT.md`
- `reports/technical_report.pdf`

The PDF report is under the 12-page competition limit.

## Project Structure

```text
config/              Project settings
data/raw/            Organizer-provided raw CSV files, not committed
data/processed/      Generated cleaned/features data, not committed by default
notebooks/           EDA starter notebook
src/                 Reusable project modules
outputs/             Generated summaries, metrics, and figures
reports/             Technical report files
models/              Saved model files in later extensions
```

## Reproducibility Notes

- Raw Kaggle files are excluded from Git by default.
- Large generated outputs are excluded from Git by default.
- The code regenerates all outputs from the allowed dataset.
- The dashboard reads generated CSV outputs and updates after rerunning the pipeline.

## Disclaimer

This system is for educational decision support using historical data. It is not financial advice, and historical performance does not guarantee future returns.
