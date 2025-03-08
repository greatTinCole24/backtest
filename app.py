import streamlit as st
import vectorbt as vbt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import pytz
from datetime import datetime
import pandas_ta as ta  # Ensure pandas_ta is installed

# Convert date to datetime with timezone
def convert_to_timezone_aware(date_obj):
    return datetime.combine(date_obj, datetime.min.time()).replace(tzinfo=pytz.UTC)

# Streamlit interface
st.set_page_config(page_title='VectorBT Backtesting', layout='wide')
st.title("📊 VectorBT Backtesting with MACD + Forecast Oscillator Strategy")

# Sidebar for inputs
with st.sidebar:
    st.header("Strategy Controls")

    # Select a stock ticker
    symbol = st.selectbox("Select a stock ticker:", 
                          ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC-USD", "ETH-USD"], 
                          index=0)

    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))

    # Timeframe Selection
    timeframe = st.selectbox("Select Timeframe", ["1m", "5m", "1h", "1d"], index=3)

    # Take Profit / Stop Loss
    tp = st.number_input("Take Profit (%)", value=5, min_value=1)
    sl = st.number_input("Stop Loss (%)", value=2, min_value=1)

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

    # Ensure valid timeframe selection for stocks (Yahoo Finance limitation)
    if symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]:
        valid_timeframe = "1d"  # Stocks only support daily or higher timeframes
    else:
        valid_timeframe = timeframe  # Crypto supports all timeframes

    # Fetch market data
    data = vbt.YFData.download(symbol, start=start_date_tz, end=end_date_tz, interval=valid_timeframe).get('Close')

    if data is None or data.empty:
        st.error(f"⚠️ No data found for {symbol}. Try another ticker.")
    else:
        # MACD Calculation
        fast_length = 12
        slow_length = 26
        signal_length = 9
        macd_df = data.ta.macd(fast=fast_length, slow=slow_length, signal=signal_length)
        macd_line = macd_df["MACD_12_26_9"]
        signal_line = macd_df["MACDs_12_26_9"]

        # 9 EMA Calculation
        ema9 = vbt.MA.run(data, 9, short_name='ema9', ewm=True).ma

        # Forecast Oscillator Calculation
        forecast_length = 14
        lrc = vbt.MA.run(data, forecast_length).ma
        lrc1 = vbt.MA.run(data.shift(1), forecast_length).ma
        lrs = (lrc - lrc1)
        TSF = lrc + lrs
        fosc = 100 * (data - TSF.shift(1)) / data

        # Define Forecast Oscillator Trend
        fosc_increasing = fosc > fosc.shift(1)  
        fosc_decreasing = fosc < fosc.shift(1)  

        # Buy and Sell Signals (MACD + Forecast Oscillator + 9 EMA Confirmation)
        buy_signal = (macd_line > signal_line) & (macd_line < 0) & (data > ema9) & fosc_increasing
        sell_signal = (macd_line < signal_line) & (macd_line > 0) & (data < ema9) & fosc_decreasing

        # Convert size
        size_value = float(size) / 100.0 if size_type == 'percent' else float(size)

        # Run portfolio with TP & SL
        portfolio = vbt.Portfolio.from_signals(
            data, buy_signal, sell_signal,
            direction=direction,
            size=size_value,
            size_type=size_type,
            fees=fees / 100,
            init_cash=initial_equity,
            freq='1D',
            sl_stop=sl / 100,
            tp_stop=tp / 100
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
