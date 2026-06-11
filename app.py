from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard_data import available_symbols, latest_rows, load_dashboard_data, number, percent


st.set_page_config(
    page_title="NIFTY-50 Investment Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #d9dee7;
        border-radius: 6px;
        padding: 0.65rem 0.75rem;
        background: #fbfcfe;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #17202a !important;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #e6e9ef;
        border-radius: 6px;
    }
    .status-note {
        border-left: 4px solid #315f72;
        background: #f6f8fa;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0 1rem 0;
        color: #1f2933;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_data() -> dict[str, pd.DataFrame]:
    return load_dashboard_data()


def show_pipeline_status(data: dict[str, pd.DataFrame]) -> None:
    required = [
        ("Phase 1", "phase1_features"),
        ("Phase 2", "phase2_features"),
        ("Phase 3", "model_metrics"),
        ("Phase 4", "portfolio_allocations"),
    ]
    cols = st.columns(len(required))
    for col, (label, key) in zip(cols, required):
        ready = not data[key].empty
        col.metric(label, "Ready" if ready else "Pending")

    if all(data[key].empty for _, key in required):
        st.markdown(
            """
            <div class="status-note">
            Add the allowed Kaggle CSV files to <b>data/raw</b>, then run
            <b>python run_phase1.py</b>, <b>python run_phase2.py</b>,
            <b>python run_phase3.py</b>, and <b>python run_phase4.py</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )


def overview_tab(data: dict[str, pd.DataFrame], selected_symbols: list[str]) -> None:
    features = data["phase1_features"]
    stock_summary = data["stock_summary"]

    if features.empty:
        st.info("Phase 1 outputs are not available yet.")
        return

    filtered = features[features["symbol"].isin(selected_symbols)] if selected_symbols else features
    latest = latest_rows(filtered)

    cols = st.columns(4)
    cols[0].metric("Stocks", f"{features['symbol'].nunique():,}")
    cols[1].metric("Rows", f"{len(features):,}")
    cols[2].metric("Start", str(pd.to_datetime(features["date"]).min().date()))
    cols[3].metric("End", str(pd.to_datetime(features["date"]).max().date()))

    chart_data = (
        filtered.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
        .sort_index()
    )
    st.subheader("Closing Price")
    st.line_chart(chart_data, height=340)

    st.subheader("Latest Snapshot")
    display_cols = [col for col in ["symbol", "date", "close", "daily_return", "rolling_21d_volatility", "drawdown"] if col in latest.columns]
    st.dataframe(latest[display_cols], use_container_width=True, hide_index=True)

    if not stock_summary.empty:
        st.subheader("Stock Summary")
        st.dataframe(stock_summary, use_container_width=True, hide_index=True)


def stock_tab(data: dict[str, pd.DataFrame], symbols: list[str]) -> None:
    features = data["phase2_features"] if not data["phase2_features"].empty else data["phase1_features"]
    if features.empty:
        st.info("Stock intelligence will appear after Phase 1 and Phase 2 run.")
        return

    symbol = st.sidebar.selectbox("Stock", symbols, index=0 if symbols else None)
    stock = features[features["symbol"] == symbol].sort_values("date")
    if stock.empty:
        st.info("No rows found for the selected stock.")
        return

    latest = stock.iloc[-1]
    cols = st.columns(5)
    cols[0].metric("Close", number(latest.get("close"), 2))
    cols[1].metric("Daily Return", percent(latest.get("daily_return")))
    cols[2].metric("RSI", number(latest.get("rsi_14"), 2))
    cols[3].metric("MACD", number(latest.get("macd"), 4))
    cols[4].metric("Drawdown", percent(latest.get("drawdown")))

    price_cols = [col for col in ["close", "sma_21", "sma_50", "sma_200"] if col in stock.columns]
    st.subheader("Price and Moving Averages")
    st.line_chart(stock.set_index("date")[price_cols], height=340)

    indicator_cols = [
        col
        for col in ["daily_return", "rolling_21d_volatility", "rsi_14", "macd", "macd_signal", "bb_position_20", "return_21d"]
        if col in stock.columns
    ]
    st.subheader("Indicator History")
    st.dataframe(stock[["date", *indicator_cols]].tail(60), use_container_width=True, hide_index=True)


def predictor_tab(data: dict[str, pd.DataFrame]) -> None:
    metrics = data["model_metrics"]
    predictions = data["predictions"]
    importance = data["feature_importance"]

    if metrics.empty:
        st.info("Predictor results will appear after Phase 3 runs.")
        return

    st.subheader("Model Metrics")
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    metric_view = metrics.set_index("model")[[col for col in ["mae", "rmse", "r2", "directional_accuracy"] if col in metrics.columns]]
    st.bar_chart(metric_view, height=300)

    if not importance.empty:
        st.subheader("Feature Importance")
        importance_score = "absolute_coefficient" if "absolute_coefficient" in importance.columns else "importance"
        importance_display = importance.copy()
        importance_display[importance_score] = pd.to_numeric(importance_display[importance_score], errors="coerce")
        importance_display = importance_display.sort_values(importance_score, ascending=False)
        importance_view = importance_display.head(20).set_index("feature")[importance_score]
        st.bar_chart(importance_view, height=360)
        st.dataframe(importance_display.head(40), use_container_width=True, hide_index=True)

    if not predictions.empty:
        st.subheader("Prediction Sample")
        st.dataframe(predictions.sort_values("date", ascending=False).head(100), use_container_width=True, hide_index=True)


def portfolio_tab(data: dict[str, pd.DataFrame]) -> None:
    allocations = data["portfolio_allocations"]
    portfolio_metrics = data["portfolio_metrics"]

    if allocations.empty:
        st.info("Portfolio allocations will appear after Phase 4 runs.")
        return

    profiles = allocations["profile"].dropna().unique().tolist()
    profile = st.sidebar.radio("Investor Profile", profiles, horizontal=False)
    selected = allocations[allocations["profile"] == profile].sort_values("weight", ascending=False)

    st.subheader(f"{profile} Allocation")
    allocation_chart = selected.set_index("symbol")["weight"]
    st.bar_chart(allocation_chart, height=320)

    display_cols = [
        "symbol",
        "weight",
        "investment_score",
        "predicted_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "allocation_reason",
    ]
    display_cols = [col for col in display_cols if col in selected.columns]
    st.dataframe(selected[display_cols], use_container_width=True, hide_index=True)

    if not portfolio_metrics.empty:
        st.subheader("Portfolio Risk Metrics")
        st.dataframe(portfolio_metrics, use_container_width=True, hide_index=True)


def risk_tab(data: dict[str, pd.DataFrame]) -> None:
    stock_risk = data["stock_risk"]
    correlation = data["return_correlation"]

    if stock_risk.empty:
        st.info("Risk analytics will appear after Phase 4 runs.")
        return

    cols = st.columns(4)
    cols[0].metric("Best Sharpe", number(stock_risk["sharpe_ratio"].max(), 3))
    cols[1].metric("Median Volatility", percent(stock_risk["annualized_volatility"].median()))
    cols[2].metric("Worst Drawdown", percent(stock_risk["max_drawdown"].min()))
    cols[3].metric("Tracked Stocks", f"{stock_risk['symbol'].nunique():,}")

    st.subheader("Risk and Return")
    scatter_source = stock_risk[["symbol", "annualized_volatility", "annualized_return"]].dropna()
    if not scatter_source.empty:
        st.scatter_chart(
            scatter_source,
            x="annualized_volatility",
            y="annualized_return",
            color="symbol",
            height=360,
        )

    st.subheader("Stock Risk Metrics")
    st.dataframe(stock_risk, use_container_width=True, hide_index=True)

    if not correlation.empty:
        st.subheader("Return Correlation")
        st.dataframe(correlation, use_container_width=True)


def insights_tab(data: dict[str, pd.DataFrame]) -> None:
    insights = data["explainable_insights"]

    if insights.empty:
        st.info("Explainable insights will appear after Phase 5 runs.")
        return

    categories = insights["category"].dropna().unique().tolist()
    selected_categories = st.multiselect("Insight Categories", categories, default=categories)
    filtered = insights[insights["category"].isin(selected_categories)] if selected_categories else insights

    for category, group in filtered.groupby("category", sort=False):
        st.subheader(category)
        for _, row in group.head(20).iterrows():
            symbol = row.get("symbol", "All")
            profile = row.get("profile", "All")
            st.markdown(f"**{symbol}** ({profile})")
            st.write(row.get("insight", ""))
            st.caption(row.get("evidence", ""))

    st.subheader("Insight Table")
    st.dataframe(filtered, use_container_width=True, hide_index=True)


def main() -> None:
    data = cached_data()
    symbols = available_symbols(data["phase1_features"], data["phase2_features"], data["stock_risk"])

    st.title("NIFTY-50 Investment Intelligence")
    show_pipeline_status(data)

    with st.sidebar:
        st.header("Controls")
        default_symbols = symbols[:5]
        selected_symbols = st.multiselect("Overview Stocks", symbols, default=default_symbols)
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    tabs = st.tabs(["Overview", "Stock Intelligence", "Predictor", "Portfolio", "Risk", "Insights"])
    with tabs[0]:
        overview_tab(data, selected_symbols)
    with tabs[1]:
        stock_tab(data, symbols)
    with tabs[2]:
        predictor_tab(data)
    with tabs[3]:
        portfolio_tab(data)
    with tabs[4]:
        risk_tab(data)
    with tabs[5]:
        insights_tab(data)


if __name__ == "__main__":
    main()
