# Project Readiness Audit

Project: Data-Driven Investment Intelligence Using NIFTY-50 Market Data

Audit date: 11 June 2026

## Overall Status

Ready for submission packaging.

The project satisfies the core problem-statement requirements:

- Working prototype
- Technical report PDF
- Public GitHub-ready repository structure
- README with setup and reproduction instructions
- Stock predictor engine
- Portfolio construction module
- Risk assessment module
- Explainability layer
- Reproducible outputs generated from the allowed NIFTY-50 dataset

## Problem Statement Requirement Check

| Requirement | Status | Evidence |
|---|---|---|
| Use provided NIFTY-50 dataset | Pass | Official Kaggle NIFTY-50 dataset downloaded/extracted under `data/raw/`; raw data is excluded from Git |
| No live APIs/news/social/proprietary data | Pass | Pipeline uses CSV price/volume data only |
| Stock predictor engine | Pass | `run_phase3.py`, `src/models.py`, `outputs/phase3_model_metrics.csv` |
| Forecast future stock behavior | Pass | 5-day future return target and multiple model comparisons |
| Model evaluation | Pass | MAE, RMSE, R2, and directional accuracy reported |
| Portfolio construction | Pass | `run_phase4.py`, `src/portfolio.py`, Conservative/Balanced/Aggressive allocations |
| Risk assessment | Pass | Volatility, Sharpe, Sortino, max drawdown, correlations |
| Explainability | Pass | `run_phase5.py`, `src/explainability.py`, 53 generated insight statements |
| Working prototype | Pass | `app.py` Streamlit dashboard with Overview, Stock Intelligence, Predictor, Portfolio, Risk, Insights |
| Technical report PDF | Pass | `reports/technical_report.pdf`, 8 pages |
| README | Pass | `README.md` includes setup, pipeline, dashboard, results, report files |
| Reproducibility | Pass | Phase runners regenerate outputs from the allowed dataset |

## Generated Results

Dataset:

- Raw rows loaded: 470,384
- Cleaned rows used: 235,192
- Stocks found: 65
- Date range: 2000-01-03 to 2021-04-30
- Feature columns: 87

Best prediction model by RMSE:

- Symbol mean baseline
- RMSE: 0.051309
- Directional accuracy: 52.24%

Portfolio metrics:

| Profile | Annualized Return | Annualized Volatility | Sharpe | Sortino | Max Drawdown |
|---|---:|---:|---:|---:|---:|
| Aggressive | 21.74% | 18.10% | 1.201 | 1.571 | -45.67% |
| Balanced | 18.83% | 15.31% | 1.230 | 1.642 | -38.25% |
| Conservative | 17.37% | 13.77% | 1.261 | 1.790 | -29.81% |

## Required Submission Files

Recommended files to include in GitHub:

- `README.md`
- `requirements.txt`
- `.gitignore`
- `app.py`
- `run_phase1.py`
- `run_phase2.py`
- `run_phase3.py`
- `run_phase4.py`
- `run_phase5.py`
- `src/`
- `config/`
- `notebooks/`
- `reports/TECHNICAL_REPORT.md`
- `reports/technical_report.pdf`
- `NIFTY50_PROJECT_GUIDELINES.md`
- `GITHUB_PUSH_COMMANDS.md`
- `data/raw/README.md`

Files intentionally excluded from Git:

- Raw Kaggle dataset files
- Large processed data files
- Large generated output CSVs
- Python cache files
- Local environment folders

## Remaining Optional Improvements

These are optional, not blockers:

- Host the Streamlit dashboard publicly.
- Add screenshots if the submission format benefits from them.
- Add a short demo video.
- Add more advanced walk-forward validation.
- Add sector-level portfolio constraints if sector metadata is expanded.

## Backdated Commit Request

A backdated Git commit should not be created to imply work was done on 5 June if that is not the true commit date. The recommended alternative is a normal current-date commit plus this readiness audit and the project report, which transparently show the project structure, pipeline, outputs, and reproducibility.
