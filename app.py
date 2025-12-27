import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION
# ==========================================
# VISUAL CONFIRMATION: If you don't see "v2.0" in the title, the app hasn't updated yet.
st.set_page_config(page_title="Market Analyzer v2.0", layout="wide")

# Fallback Key (Since we know this works)
API_KEY = "HO1Gg4eZ38sEt6MhH0SKI7XrhmGjjrX8"
BASE_URL = "https://financialmodelingprep.com/api/v3"

# ==========================================
# 2. THE FIX: NEW DATA ENGINE
# ==========================================

@st.cache_data(ttl=3600*24)
def get_daily_price_history(ticker):
    # CRITICAL FIX: Using 'historical-chart/1day' (Allowed for New Users)
    # The old code used 'historical-price-full' which causes the "Legacy" error.
    url = f"{BASE_URL}/historical-chart/1day/{ticker}?apikey={API_KEY}"
    
    try:
        data = requests.get(url).json()
        
        # New Endpoint returns a LIST, not a Dictionary.
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').set_index('date')
            return df['close']
        else:
            return None
    except: return None

@st.cache_data(ttl=3600*24)
def get_key_metrics(ticker):
    url = f"{BASE_URL}/key-metrics/{ticker}?period=quarter&limit=80&apikey={API_KEY}"
    try:
        data = requests.get(url).json()
        if isinstance(data, list):
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').set_index('date')
            return df
        return pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(ttl=3600*24)
def get_ratios(ticker):
    url = f"{BASE_URL}/ratios/{ticker}?period=quarter&limit=80&apikey={API_KEY}"
    try:
        data = requests.get(url).json()
        if isinstance(data, list):
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').set_index('date')
            return df
        return pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(ttl=3600*24)
def get_full_data_merged(ticker):
    prices = get_daily_price_history(ticker)
    if prices is None: return None
    ratios = get_ratios(ticker)
    metrics = get_key_metrics(ticker)
    
    df = pd.DataFrame(index=prices.index)
    df['price'] = prices
    
    # Merge Ratios
    if not ratios.empty:
        cols = ['priceEarningsRatio', 'pegRatio']
        existing_cols = [c for c in cols if c in ratios.columns]
        if existing_cols:
            df = pd.merge_asof(df, ratios[existing_cols], left_index=True, right_index=True, direction='backward')
            
    # Merge Metrics
    if not metrics.empty:
        cols_map = {'enterpriseValueOverEBITDA': 'ev_ebitda', 'roic': 'roic', 'freeCashFlowYield': 'fcf_yield'}
        available_cols = [c for c in cols_map.keys() if c in metrics.columns]
        if available_cols:
            metrics_subset = metrics[available_cols].rename(columns=cols_map)
            df = pd.merge_asof(df, metrics_subset, left_index=True, right_index=True, direction='backward')
    
    return df

def calculate_score(row):
    score = 0
    pe = row.get('priceEarningsRatio', 35)
    if pd.isna(pe): pe = 35
    if pe < 20: score += 10
    elif 20 <= pe <= 30: score += 5
    
    ev = row.get('ev_ebitda', 25)
    if pd.isna(ev): ev = 25
    if ev < 12: score += 15
    elif 12 <= ev <= 20: score += 7.5
    
    peg = row.get('pegRatio', 3)
    if pd.isna(peg): peg = 3
    if peg < 1: score += 25
    elif 1 <= peg <= 1.5: score += 12.5
    
    roic = row.get('roic', 0)
    if pd.isna(roic): roic = 0
    if roic < 1.0: roic = roic * 100 
    if roic > 20: score += 30
    elif 10 <= roic <= 20: score += 15
    
    fcf = row.get('fcf_yield', 0)
    if pd.isna(fcf): fcf = 0
    if fcf < 1.0: fcf = fcf * 100 
    if fcf > 5: score += 20
    elif 3 <= fcf <= 5: score += 10
    return score

# ==========================================
# 3. APP UI
# ==========================================

st.title("✅ Market Analyzer v2.0 (New Data Engine)")
st.markdown("---")

# Navigation
page = st.sidebar.radio("Navigate", ["Individual Stock Analyzer", "Trade Tester"])

if page == "Individual Stock Analyzer":
    st.header("Individual Stock Analysis")
    ticker_input = st.text_input("Enter Ticker", "NVDA").upper()
    
    if ticker_input:
        with st.spinner(f"Fetching V2 Data for {ticker_input}..."):
            df = get_full_data_merged(ticker_input)
            
            if df is not None and not df.empty:
                # Resample and Score
                df_weekly = df.resample('W-FRI').last().ffill()
                df_weekly['Score'] = df_weekly.apply(calculate_score, axis=1)
                latest = df_weekly.iloc[-1]
                
                st.subheader(f"{ticker_input} Score: {latest['Score']}/100")
                
                # Metrics Row
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Price", f"${latest['price']:.2f}")
                c2.metric("P/E", f"{latest.get('priceEarningsRatio', 0):.2f}")
                c3.metric("ROIC", f"{latest.get('roic', 0)*100:.1f}%")
                c4.metric("PEG", f"{latest.get('pegRatio', 0):.2f}")
                
                # Score Chart
                fig = px.line(df_weekly, x=df_weekly.index, y="Score", title="Historical Quality Score")
                fig.add_hrect(y0=80, y1=100, line_width=0, fillcolor="green", opacity=0.1)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("Still no data. Please wait 1 minute for Streamlit to clear cache.")

elif page == "Trade Tester":
    st.header("Trade Backtester")
    t_ticker = st.text_input("Ticker", "MSFT").upper()
    t_date = st.date_input("Buy Date", datetime.today() - timedelta(days=365))
    
    if st.button("Simulate"):
        df = get_full_data_merged(t_ticker)
        if df is not None:
            t_date_ts = pd.Timestamp(t_date)
            try:
                idx = df.index.get_indexer([t_date_ts], method='nearest')[0]
                row_buy = df.iloc[idx]
                row_now = df.iloc[-1]
                
                buy_score = calculate_score(row_buy)
                profit = ((row_now['price'] - row_buy['price']) / row_buy['price']) * 100
                
                st.success(f"Bought at Score: {buy_score}/100")
                st.metric("Total Return", f"{profit:.2f}%")
            except: st.error("Date out of range")
