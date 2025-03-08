import streamlit as st
import vectorbt as vbt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import pytz
from datetime import datetime

# Convert date to datetime with timezone
def convert_to_timezone_aware(date_obj):
    return datetime.combine(date_obj, datetime.min.time()).replace(tzinfo=pytz.UTC)

# Streamlit interface
st.set_page_config(page_title='VectorBT Backtesting', layout='wide')
st.title("📊 VectorBT Backtesting")

# Sidebar for inputs
with st.sidebar:
    st.header("Strategy Controls")

    # Select a stock ticker
    symbol = st.selectbox("Select a stock ticker:", 
                          ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC-USD", "ETH-USD"], 
                          index=0)

    start_date = st.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))

    # EMA settings
    short_ema_period = st.number_input("Short EMA Period", value=10, min_value=1)
    long_ema_period = st.number_input("Long EMA Period", value=20, min_value=1)

    st.header("Backtesting Controls")
    initial_equity = st.number_input("Initial Equity", value=100000)
    size = st.text_input("Position Size", value='50')  
    size_type = st.selectbox("Size Type", ["amount", "value", "percent"], index=2)  
    fees = st.number_input("Fees (as %)", value=0.12, format="%.4f")
    direction = st.selectbox("Direction", ["longonly", "shortonly", "both"], index=0)

    backtest_clicked = st.button("Backtest")

if backtest_clicked:
    start_date_tz = convert_to_timezone_aware(start_date)
    end_date_tz = convert_to_timezone_aware(end_date)

    # Fetch market data
    data = vbt.YFData.download(symbol, start=start_date_tz, end=end_date_tz).get('Close')

    if data is None or data.empty:
        st.error(f"⚠️ No data found for {symbol}. Try another ticker.")
    else:
        # Compute EMAs and signals
        short_ema = vbt.MA.run(data, short_ema_period, short_name='fast', ewm=True)
        long_ema = vbt.MA.run(data, long_ema_period, short_name='slow', ewm=True)
        entries = short_ema.ma_crossed_above(long_ema)
        exits = short_ema.ma_crossed_below(long_ema)

        # Convert size
        size_value = float(size) / 100.0 if size_type == 'percent' else float(size)

        # Run portfolio
        portfolio = vbt.Portfolio.from_signals(
            data, entries, exits,
            direction=direction,
            size=size_value,
            size_type=size_type,
            fees=fees / 100,
            init_cash=initial_equity,
            freq='1D'
        )

        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Backtesting Stats", "List of Trades", 
                                                "Equity Curve", "Drawdown", "Portfolio Plot"])

        with tab1:
            st.markdown("**Backtesting Stats:**")
            stats_df = pd.DataFrame(portfolio.stats(), columns=['Value'])
            st.dataframe(stats_df)

        with tab2:
            st.markdown("**List of Trades:**")
            trades_df = portfolio.trades.records_readable.round(2)
            st.dataframe(trades_df)

        with tab3:
            equity_trace = go.Scatter(x=portfolio.value().index, y=portfolio.value(), mode='lines', name='Equity', line=dict(color='green'))
            equity_fig = go.Figure(data=[equity_trace])
            st.plotly_chart(equity_fig)

        with tab4:
            drawdown_trace = go.Scatter(x=portfolio.drawdown().index, y=portfolio.drawdown() * 100, mode='lines', name='Drawdown', fill='tozeroy', line=dict(color='red'))
            drawdown_fig = go.Figure(data=[drawdown_trace])
            st.plotly_chart(drawdown_fig)

        with tab5:
            st.plotly_chart(portfolio.plot())
