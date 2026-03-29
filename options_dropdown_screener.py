import streamlit as st

from config import TICKERS
from options_screener import analyze_single_ticker, run_screener


def render_app():
    st.set_page_config(page_title="Options Screener", layout="wide")

    st.title("Advanced Options Screener")

    mode = st.selectbox(
        "Select Data Period",
        ["3 Months (Short-Term)", "2 Years (Historical Volatility)"]
    )

    period = "3mo" if "3 Months" in mode else "2y"

    expiry_days = st.selectbox(
        "Select Expiry Length (days)",
        [5, 15, 30, 45, 60, 90, 120, 180]
    )

    st.markdown("---")

    if st.button("Run Full Screener"):
        with st.spinner("Analyzing tickers..."):
            df = run_screener(period=period, expiry_days=expiry_days)

        st.success("Analysis Complete")
        st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.subheader("Single Ticker Analysis")

    selected_ticker = st.selectbox("Select Ticker", TICKERS)

    if st.button("Analyze Ticker"):
        data = analyze_single_ticker(selected_ticker, period=period, expiry_days=expiry_days)

        if data:
            st.subheader(f"{selected_ticker} Analysis")

            col1, col2, col3 = st.columns(3)
            col1.metric("Price", f"${data['price']}")
            col2.metric("Direction", data["direction"])
            col3.metric("Score", data["score"])

            st.write(f"Expected Move %: **{data['expected_move_pct']}%**")
            st.write(f"Expected Range: **${data['expected_low']} to ${data['expected_high']}**")
            st.write(f"Suggested Strike: **${data['suggested_strike']}**")
        else:
            st.warning("No data available.")


if __name__ == "__main__":
    render_app()
