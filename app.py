import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Market Analyzer v2.1 (Debug)", layout="wide")

# Hardcoded Key for Stability
API_KEY = "HO1Gg4eZ38sEt6MhH0SKI7XrhmGjjrX8"
BASE_URL = "https://financialmodelingprep.com/api/v3"

# ==========================================
# 2. DATA ENGINE (With Error Printing)
# ==========================================

@st.cache_data(ttl=3600*24)
def get_daily_price_history(ticker):
    # STRATEGY: Try 4-Hour charts.
    # This is the most compatible endpoint for new FMP accounts.
    url = f"{BASE_URL}/historical-chart/4hour/{ticker}?apikey={API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # DEBUG: If it's an error dictionary, print it!
        if isinstance(data, dict) and "Error Message" in data:
            st.error(f"⚠️ API ERROR for {ticker}: {data['Error Message']}")
            return None
            
        # If it's a list (Success)
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').set_index('date')
            return df['close']
        
        # If list is empty
        st.warning(f"⚠️ API returned empty list for {ticker}.")
        return None
        
    except Exception as e:
        st.error(f"System Error: {e}")
        return None

@st.cache_data(ttl=3600*24)
def get_full_data_merged(ticker):
    prices = get_daily_price_history(ticker)
    if prices is None: return None
    
    # Create simple DF with Price
    df = pd.DataFrame(index=prices.index)
    df['price'] = prices
    
    # Get Fundamentals (Ratios)
    try:
        r_url = f"{BASE_URL}/ratios/{ticker}?period=quarter&limit=40&apikey={API_KEY}"
        r_data = requests.get(r_url).json()
        if isinstance(r_data, list):
            r_df = pd.DataFrame(r_data)
            r_df['date'] = pd.to_datetime(r_df['date'])
            r_df = r_df.sort_values('date').set_index('date')
            
            # Merge logic
            cols = ['priceEarningsRatio', 'pegRatio']
            existing = [c for c in cols if c in r_df.columns]
            if existing:
                df = pd.merge_asof(df, r_df[existing], left_index=True, right_index=True, direction='backward')
    except: pass
    
    return df

def calculate_score(row):
    # Simplified Score for Testing
    score = 0
    pe = row.get('priceEarningsRatio', 35)
    if pd.isna(pe): pe = 35
    if pe < 20: score += 20
    elif 20 <= pe <= 30: score += 10
    
    peg = row.get('pegRatio', 3)
    if pd.isna(peg): peg = 3
    if peg < 1: score += 30
    elif 1 <= peg <= 1.5: score += 15
    
    return score

# ==========================================
# 3. APP UI
# ==========================================

st.title("🛠️ Market Analyzer v2.1 (Debug Mode)")
st.info("If you see an error below, please copy-paste it to the chat.")

ticker_input = st.text_input("Enter Ticker", "NVDA").upper()

if ticker_input:
    df = get_full_data_merged(ticker_input)
    
    if df is not None and not df.empty:
        # Calculate Score
        latest = df.iloc[-1]
        try:
            score = calculate_score(latest)
        except: score = 0
        
        st.subheader(f"{ticker_input} - Status: ACTIVE")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Price", f"${latest['price']:.2f}")
        c2.metric("P/E Ratio", f"{latest.get('priceEarningsRatio', 0):.2f}")
        c3.metric("Quality Score", f"{score}/50")
        
        # Chart
        st.subheader("Price Chart (4-Hour)")
        st.line_chart(df['price'])
        
    else:
        st.error("❌ Data Load Failed. See error messages above.")
