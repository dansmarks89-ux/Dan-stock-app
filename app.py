import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from datetime import datetime

# ==========================================
# 1. CONFIGURATION & STORAGE
# ==========================================
st.set_page_config(page_title="Market Analyzer Pro (FMP)", layout="wide")

# File to store your long-term history
DB_FILE = "market_data.json"

def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, 'r') as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, 'w') as f:
        json.dump(data, f)

def update_history(ticker, score, price):
    db = load_db()
    today = datetime.now().strftime("%Y-%m-%d")
    
    if ticker not in db:
        db[ticker] = []
    
    # Check if we already have an entry for today to avoid duplicates
    history = db[ticker]
    if not history or history[-1]['date'] != today:
        history.append({
            "date": today,
            "score": score,
            "price": price
        })
        db[ticker] = history
        save_db(db)

# ==========================================
# 2. DATA ENGINE (FMP PREMIUM)
# ==========================================
# Use your Premium Key here
API_KEY = "HO1Gg4eZ38sEt6MhH0SKI7XrhmGjjrX8" 
BASE_URL = "https://financialmodelingprep.com/api/v3"

@st.cache_data(ttl=300)
def get_fmp_data(ticker):
    """
    Fetches Premium Data from FMP.
    Returns: (HistoryDF, MetricsDict, RatiosDict)
    """
    # 1. Price History
    try:
        url_hist = f"{BASE_URL}/historical-price-full/{ticker}?serietype=line&apikey={API_KEY}"
        r_hist = requests.get(url_hist).json()
        if "historical" in r_hist:
            hist_df = pd.DataFrame(r_hist['historical'])
            hist_df['date'] = pd.to_datetime(hist_df['date'])
            hist_df = hist_df.sort_values('date')
        else:
            hist_df = pd.DataFrame()
    except: hist_df = pd.DataFrame()

    # 2. Key Metrics (TTM) - For ROIC, FCF Yield, EV/EBITDA
    try:
        url_metrics = f"{BASE_URL}/key-metrics-ttm/{ticker}?apikey={API_KEY}"
        metrics = requests.get(url_metrics).json()
        if isinstance(metrics, list) and len(metrics) > 0:
            m_data = metrics[0]
        else: m_data = {}
    except: m_data = {}

    # 3. Ratios (TTM) - For PE, PEG
    try:
        url_ratios = f"{BASE_URL}/ratios-ttm/{ticker}?apikey={API_KEY}"
        ratios = requests.get(url_ratios).json()
        if isinstance(ratios, list) and len(ratios) > 0:
            r_data = ratios[0]
        else: r_data = {}
    except: r_data = {}  # <--- THIS WAS THE MISSING LINE
        
    return hist_df, m_data, r_data

def safe_float(val):
    try:
        if val is None: return None
        return float(val)
    except: return None

# ==========================================
# 3. SCORING LOGIC (FMP Edition)
# ==========================================

def calculate_fmp_score(metrics, ratios):
    earned = 0
    possible = 0
    log = {}
    
    # 1. P/E Ratio (Weight: 10)
    pe = safe_float(ratios.get('priceEarningsRatioTTM'))
    if pe:
        possible += 10
        pts = 0
        if pe < 20: pts = 10
        elif 20 <= pe <= 30: pts = 5
        earned += pts
        log["P/E Ratio"] = f"{pe:.2f} ({pts}/10)"
    else: log["P/E"] = "N/A"

    # 2. EV / EBITDA (Weight: 15)
    ev = safe_float(metrics.get('enterpriseValueOverEBITDATTM'))
    if ev:
        possible += 15
        pts = 0
        if ev < 12: pts = 15
        elif 12 <= ev <= 20: pts = 7.5
        earned += pts
        log["EV/EBITDA"] = f"{ev:.2f} ({pts}/15)"
    else: log["EV/EBITDA"] = "N/A"

    # 3. PEG Ratio (Weight: 25)
    peg = safe_float(ratios.get('pegRatioTTM'))
    if peg:
        possible += 25
        pts = 0
        if peg < 1: pts = 25
        elif 1 <= peg <= 1.5: pts = 12.5
        earned += pts
        log["PEG Ratio"] = f"{peg:.2f} ({pts}/25)"
    else: log["PEG Ratio"] = "N/A"

    # 4. ROIC (Weight: 30) - FMP has this pre-calculated!
    roic = safe_float(metrics.get('roicTTM'))
    if roic:
        roic = roic * 100 # FMP usually gives decimal 0.15 -> 15%
        possible += 30
        pts = 0
        if roic > 20: pts = 30
        elif 10 <= roic <= 20: pts = 15
        earned += pts
        log["ROIC"] = f"{roic:.1f}% ({pts}/30)"
    else: log["ROIC"] = "N/A"

    # 5. FCF Yield (Weight: 20) - The Best Metric is Back
    fcfy = safe_float(metrics.get('freeCashFlowYieldTTM'))
    if fcfy:
        fcfy = fcfy * 100
        possible += 20
        pts = 0
        if fcfy > 5: pts = 20
        elif 3 <= fcfy <= 5: pts = 10
        earned += pts
        log["FCF Yield"] = f"{fcfy:.1f}% ({pts}/20)"
    else: log["FCF Yield"] = "N/A"

    score = int((earned / possible) * 100) if possible > 0 else 0
    return score, log

# ==========================================
# 4. APP UI
# ==========================================

st.title("💎 Market Analyzer Pro (FMP Premium)")
st.caption("Powered by Financial Modeling Prep (Premium Tier)")

# TABS
tab_analysis, tab_watchlist, tab_sectors = st.tabs(["🔍 Stock Analysis", "⭐ My Watchlist", "📊 Sector ETFs"])

# --- TAB 1: ANALYSIS ---
with tab_analysis:
    c1, c2 = st.columns([3, 1])
    ticker = c1.text_input("Analyze Ticker", "NVDA").upper()
    
    if ticker:
        # Fetch Data
        hist, metrics, ratios = get_fmp_data(ticker)
        
        if not hist.empty and metrics:
            # Calculate Score
            score, breakdown = calculate_fmp_score(metrics, ratios)
            curr_price = hist['close'].iloc[-1]
            
            # SAVE TO HISTORY AUTOMATICALLY
            update_history(ticker, score, curr_price)
            
            # Display
            col_score, col_chart = st.columns([1, 2])
            
            with col_score:
                st.metric("Quality Score", f"{score}/100", delta=None)
                st.write("### Score Metrics")
                st.table(pd.DataFrame(list(breakdown.items()), columns=["Metric", "Value"]))
                
                st.success(f"✅ Data point saved for {ticker}")

            with col_chart:
                st.subheader(f"{ticker} Price Action")
                st.line_chart(hist.set_index('date')['close'])
                
                # Show Historical Score Graph if available
                db = load_db()
                if ticker in db and len(db[ticker]) > 1:
                    st.write("### Your Quality Score History")
                    score_df = pd.DataFrame(db[ticker])
                    
                    fig = px.line(score_df, x='date', y='score', markers=True, title=f"How {ticker}'s Score Changed")
                    fig.update_yaxes(range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)
                elif ticker in db:
                    st.caption("Analyze this stock again on a future date to see the Score History graph appear here.")

        else:
            st.error("Could not fetch data. FMP might be blocking access or Ticker is invalid.")
            st.info(f"Debug: History Empty? {hist.empty} | Metrics? {bool(metrics)}")

# --- TAB 2: WATCHLIST ---
with tab_watchlist:
    st.header("Your Tracked Stocks")
    db = load_db()
    
    if db:
        # Create a summary table
        watchlist_data = []
        for t, history in db.items():
            latest = history[-1]
            watchlist_data.append({
                "Ticker": t,
                "Last Checked": latest['date'],
                "Latest Score": latest['score'],
                "Latest Price": f"${latest['price']:.2f}",
                "Data Points": len(history)
            })
        
        wl_df = pd.DataFrame(watchlist_data)
        st.dataframe(wl_df, use_container_width=True)
        
        # Deep Dive Graph
        st.markdown("---")
        st.subheader("Compare Scores Over Time")
        selected_ticker = st.selectbox("Select Ticker to Graph", list(db.keys()))
        
        if selected_ticker:
            history = db[selected_ticker]
            if len(history) > 0:
                h_df = pd.DataFrame(history)
                fig = px.line(h_df, x='date', y='score', markers=True, title=f"{selected_ticker} Quality Score Trend")
                fig.update_yaxes(range=[0, 100])
                st.plotly_chart(fig)
            else:
                st.info("Not enough data points yet.")
    else:
        st.info("Your watchlist is empty. Go to the 'Stock Analysis' tab and search for a ticker to start tracking it!")

# --- TAB 3: SECTORS (ETFs ONLY) ---
with tab_sectors:
    st.header("Sector Health (ETF Proxies)")
    
    SECTOR_ETFS = {
        "Technology": "XLK",
        "Healthcare": "XLV",
        "Financials": "XLF",
        "Energy": "XLE",
        "Semiconductors": "SMH",
        "Communications": "XLC",
        "Consumer Disc.": "XLY",
        "Industrials": "XLI",
        "Real Estate": "XLRE"
    }
    
    selected_sector = st.selectbox("Choose Sector", list(SECTOR_ETFS.keys()))
    etf_ticker = SECTOR_ETFS[selected_sector]
    
    if st.button(f"Analyze {selected_sector} ({etf_ticker})"):
        hist, metrics, ratios = get_fmp_data(etf_ticker)
        
        if not hist.empty:
            # 1. Price Trend
            st.subheader(f"{selected_sector} Trend ({etf_ticker})")
            
            # Calculate simple MA
            hist['MA50'] = hist['close'].rolling(50).mean()
            curr = hist['close'].iloc[-1]
            ma50 = hist['MA50'].iloc[-1]
            
            trend = "BULLISH 🟢" if curr > ma50 else "BEARISH 🔴"
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"${curr:.2f}")
            c2.metric("Trend Status", trend)
            c3.metric("50-Day MA", f"${ma50:.2f}")
            
            # Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist['date'], y=hist['close'], name='Price'))
            fig.add_trace(go.Scatter(x=hist['date'], y=hist['MA50'], name='50-Day MA', line=dict(color='orange')))
            st.plotly_chart(fig, use_container_width=True)
            
            # ETF Fundamentals (FMP sometimes has these for ETFs, sometimes not)
            if metrics:
                st.write("### ETF Fundamentals (Weighted Avg)")
                pe = ratios.get('priceEarningsRatioTTM', 'N/A')
                div = metrics.get('dividendYieldTTM', 0) * 100 if metrics.get('dividendYieldTTM') else 'N/A'
                
                ec1, ec2 = st.columns(2)
                ec1.metric("Sector P/E", f"{pe:.2f}" if pe != 'N/A' else "N/A")
                ec2.metric("Dividend Yield", f"{div:.2f}%" if div != 'N/A' else "N/A")
        else:
            st.error("Could not load Sector Data.")
