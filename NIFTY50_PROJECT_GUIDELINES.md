# Data-Driven Investment Intelligence Using NIFTY-50 Market Data

## Competition Guidelines

Project: AI-powered investment intelligence platform using the provided NIFTY-50 market datasets.

Team format: Up to 2 participants.

Final submission: Exactly one public link, either a Google Drive folder or a GitHub repository. For team submissions, each member must individually submit the same project link through the form.

Deadline: 11:59 PM, 12 June 2026.

Submission must remain publicly accessible throughout evaluation.

## Allowed Data

Use only the datasets provided by the organizers:

- NIFTY-50 Stock Market Dataset: https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data
- Additional India NSE dataset: https://www.kaggle.com/datasets/stoicstatic/india-stock-data-nse-1990-2020

Do not use:

- Live market data
- Financial APIs
- News datasets
- Social media sentiment data
- Proprietary financial data
- Alternative market datasets

## Core Project Objective

Build a practical decision-support system that helps investors analyze NIFTY-50 stocks, assess portfolio risk, and make data-driven investment decisions.

The project should go beyond simple stock-price prediction. The strongest version should combine:

- Historical stock analysis
- Predictive modeling
- Portfolio construction
- Risk assessment
- Explainable recommendations
- A usable prototype interface

## Mandatory Modules

### 1. Stock Predictor Engine

Goal: Forecast future stock behavior using historical market data.

Recommended prediction targets:

- Next-day or next-week return
- Price movement direction: up or down
- Future closing price
- Volatility estimate

Suggested models:

- Linear Regression or Ridge Regression baseline
- Random Forest or XGBoost-style tree model
- LSTM or GRU model only if time permits

Evaluation metrics:

- MAE
- RMSE
- R2 score
- Directional accuracy

### 2. Portfolio Construction Module

Goal: Recommend portfolio allocations for different investor profiles.

Investor profiles:

- Conservative: lower volatility, stable returns, lower drawdown
- Balanced: moderate risk and return
- Aggressive: higher return potential with higher risk

Suggested logic:

- Rank stocks by expected return, volatility, Sharpe ratio, and drawdown
- Apply allocation constraints so one stock cannot dominate the portfolio
- Generate weights for each profile
- Explain why each selected stock was included

### 3. Risk Assessment Module

Goal: Quantify stock and portfolio risk using historical data.

Required or recommended risk metrics:

- Volatility
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Risk-adjusted return
- Correlation between selected stocks

## Optional Modules

### Explainable AI Framework

Explain recommendations using clear evidence such as:

- Recent momentum
- Volatility trend
- Sharpe ratio ranking
- Drawdown history
- Sector diversification
- Feature importance from model outputs

### Personalized Investment Strategies

Allow users to choose:

- Risk tolerance
- Investment horizon
- Preferred sectors
- Number of stocks in portfolio

### Market Anomaly Detection

Detect:

- Sudden volatility spikes
- Extreme drawdowns
- Unusual volume changes
- Abnormal returns

### Deployment

Best practical option: Streamlit dashboard.

Alternative options:

- Flask web app
- Local desktop app
- Cloud-hosted dashboard

## Recommended Final Product

We should build a Streamlit-based investment intelligence dashboard with these pages:

1. Market Overview
   - NIFTY-50 stock list
   - Sector-wise performance
   - Historical price and return trends
   - Volume and turnover patterns

2. Stock Intelligence
   - Select a stock
   - View price chart, returns, volatility, moving averages, RSI, MACD, and Bollinger Bands
   - Show prediction and confidence-style explanation

3. Portfolio Builder
   - Select investor type: Conservative, Balanced, Aggressive
   - Generate recommended allocation
   - Show expected return, volatility, Sharpe ratio, and max drawdown

4. Risk Analyzer
   - Compare stocks
   - Display risk-return scatter plot
   - Show correlation heatmap
   - Highlight high-risk assets

5. Explainability & Insights
   - Explain top recommendations
   - Show the quantitative reason behind each recommendation
   - Summarize success cases, limitations, and assumptions

## Suggested Repository Structure

```text
nifty50-investment-intelligence/
  README.md
  requirements.txt
  app.py
  config/
    settings.yaml
  data/
    raw/
    processed/
  notebooks/
    01_eda.ipynb
    02_feature_engineering.ipynb
    03_modeling.ipynb
    04_portfolio_risk.ipynb
  src/
    data_loader.py
    preprocessing.py
    features.py
    indicators.py
    models.py
    portfolio.py
    risk.py
    explainability.py
    visualization.py
  reports/
    technical_report.pdf
    presentation.pdf
  models/
  outputs/
```

## Feature Engineering Plan

Use only transformations derived from the provided market data:

- Daily returns
- Log returns
- Moving averages: 7-day, 21-day, 50-day, 100-day, 200-day
- Exponential moving averages
- RSI
- MACD
- Bollinger Bands
- Rolling volatility
- Momentum indicators
- Volume change
- Turnover change
- Lagged returns
- Drawdown

## Model Strategy

Start simple and make the solution reliable:

1. Baseline model
   - Predict next-period return using simple historical average or linear regression.

2. Main machine learning model
   - Train a tree-based model using technical indicators and lagged features.

3. Directional prediction
   - Convert future return into up/down movement and report directional accuracy.

4. Portfolio integration
   - Use predicted return plus risk metrics to rank stocks.

The report should clearly state that predictions are historical-data-based decision support, not financial advice.

## Portfolio Strategy

Suggested scoring formula:

```text
Investment Score =
  0.35 * normalized expected return
+ 0.25 * normalized Sharpe ratio
- 0.20 * normalized volatility
- 0.20 * normalized maximum drawdown
```

Adjust weights by investor profile:

- Conservative: prioritize low volatility and low drawdown
- Balanced: balance return and risk
- Aggressive: prioritize expected return and momentum

## Evaluation Criteria Alignment

The official evaluation weights are:

- Quality of Investment Insights: 25%
- Technical Innovation: 20%
- Portfolio & Risk Analytics: 20%
- Explainability & Transparency: 20%
- Reproducibility & Documentation: 15%

To score well, the project should emphasize:

- Clear insights, not just charts
- Evidence-backed recommendations
- Strong portfolio and risk logic
- Explainable outputs
- Clean README and reproducible code

## Deliverables Checklist

Required:

- Working prototype
- Technical report as PDF, maximum 12 pages
- Public GitHub repository
- README.md with setup, installation, run instructions, and result reproduction steps

Recommended extras:

- Short demo video or screenshots
- Saved model files
- Example output portfolios
- Presentation PDF

## Build Roadmap

### Phase 1: Data and EDA

- Download provided datasets
- Clean and combine stock files
- Add company metadata and sector information
- Analyze price, volume, returns, volatility, and sector patterns

### Phase 2: Indicators and Features

- Create technical indicators
- Create lagged features
- Create prediction targets
- Save processed data

### Phase 3: Modeling

- Train baseline model
- Train main ML model
- Evaluate MAE, RMSE, R2, and directional accuracy
- Compare models and explain trade-offs

### Phase 4: Portfolio and Risk

- Calculate risk metrics
- Build investor-profile allocation logic
- Add correlation and diversification analysis
- Generate portfolio recommendations

### Phase 5: Dashboard

- Build Streamlit app
- Add pages for overview, stock analysis, portfolio builder, risk analyzer, and explanations
- Add charts and recommendation summaries

### Phase 6: Report and Submission

- Write 12-page technical report
- Prepare README
- Verify reproducibility
- Make repository public
- Submit one accessible link

## Recommended Project Positioning

Title: Data-Driven Investment Intelligence Using NIFTY-50 Market Data

One-line description:

An explainable investment decision-support platform that uses historical NIFTY-50 market data to forecast stock behavior, construct risk-aware portfolios, and generate evidence-backed investment insights.

Important disclaimer:

This system is designed for educational decision support using historical data and should not be treated as financial advice.
