import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from datetime import datetime

# ==========================================
# 1. CONFIGURATION & DATABASE
# ==========================================
st.set_page_config(page_title="Alpha Vantage Pro", layout="wide")

DB_FILE = "market_data.json"

def load_db():
    if not os.path.exists(DB_FILE): return {}
    with open(DB_FILE, 'r') as f: return json.load(f)

def save_db(data):
    with open(DB_FILE, 'w') as f: json.dump(data, f)

def update_history(ticker, score, price):
    db = load_db()
    today = datetime.now().strftime("%Y-%m-%d")
    if ticker not in db: db[ticker] = []
    
    # Avoid duplicate entries for the same day
    history = db[ticker]
    if not history or history[-1]['date'] != today:
        history.append({"date": today, "score": score, "price": price})
        db[ticker] = history
        save_db(db)

# ==========================================
# 2. DATA ENGINE (ALPHA VANTAGE PREMIUM)
# ==========================================

# Default Key (You can change this in the Sidebar)
DEFAULT_KEY = "O1U1GWC8OQ4RRBL1" 

@st.cache_data(ttl=300)
def get_alpha_data(ticker, api_key):
    """
    Fetches 3 distinct datasets from Alpha Vantage.
    Only possible with Premium (due to rate limits).
    """
    base = "https://www.alphavantage.co/query"
    
    # 1. OVERVIEW (Ratios)
    try:
        r_ov = requests.get(f"{base}?function=OVERVIEW&symbol={ticker}&apikey={api_key}").json()
    except: r_ov = {}

    # 2. CASH FLOW (For FCF Yield)
    try:
        r_cf = requests.get(f"{base}?function=CASH_FLOW&symbol={ticker}&apikey={api_key}").json()
    except: r_cf = {}
    
    # 3. DAILY PRICE (Charts)
    # Using 'outputsize=compact' returns last 100 days (faster)
    # 'outputsize=full' returns 20 years (slower)
    try:
        r_price = requests.get(f"{base}?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={api_key}").json()
        ts = r_price.get('Time Series (Daily)', {})
        if ts:
            df = pd.DataFrame.from_dict(ts, orient='index')
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            # Rename for convenience
            df = df.rename(columns={'4. close': 'close'})
        else:
            df = pd.DataFrame()
    except: df = pd.DataFrame()
        
    return df, r_ov, r_cf

def safe_float(val):
    try:
        if val is None or val == "None" or val == "-": return None
        return float(val)
    except: return None

# ==========================================
# 3. SCORING LOGIC (With FCF Yield)
# ==========================================

def calculate_alpha_score(overview, cash_flow):
    earned = 0
    possible = 0
    log = {}
    
    # 1. Forward P/E (Weight: 10)
    pe = safe_float(overview.get('ForwardPE'))
    if pe:
        possible += 10
        pts = 0
        if pe < 20: pts = 10
        elif 20 <= pe <= 30: pts = 5
        earned += pts
        log["Forward P/E"] = f"{pe:.2f} ({pts}/10)"
    else: log["Forward P/E"] = "N/A"
    
    # 2. EV / EBITDA (Weight: 15)
    ev_ebitda = safe_float(overview.get('EVToEBITDA'))
    if ev_ebitda:
        possible += 15
        pts = 0
        if ev_ebitda < 12: pts = 15
        elif 12 <= ev_ebitda <= 20: pts = 7.5
        earned += pts
        log["EV/EBITDA"] = f"{ev_ebitda:.2f} ({pts}/15)"
    else: log["EV/EBITDA"] = "N/A"
    
    # 3. PEG Ratio (Weight: 25)
    peg = safe_float(overview.get('PEGRatio'))
    if peg:
        possible += 25
        pts = 0
        if peg < 1: pts = 25
        elif 1 <= peg <= 1.5: pts = 12.5
        earned += pts
        log["PEG Ratio"] = f"{peg:.2f} ({pts}/25)"
    else: log["PEG Ratio"] = "N/A"
    
    # 4. ROE (Weight: 30) - Mega-Cap Safe
    roe = safe_float(overview.get('ReturnOnEquityTTM'))
    if roe:
        # Fix logic for Apple/NVO (decimals vs percents)
        if roe < 5: roe = roe * 100
        possible += 30
        pts = 0
        if roe > 20: pts = 30
        elif 10 <= roe <= 20: pts = 15
        earned += pts
        log["ROE (Quality)"] = f"{roe:.1f}% ({pts}/30)"
    else: log["ROE"] = "N/A"
    
    # 5. FCF YIELD (Weight: 20) - CALCULATED MANUALLY
    # Formula: (OpCashFlow - CapEx) / MarketCap
    try:
        annual_reports = cash_flow.get('annualReports', [])
        if annual_reports:
            latest = annual_reports[0]
            ocf = safe_float(latest.get('operatingCashflow'))
            capex = safe_float(latest.get('capitalExpenditures'))
            mcap = safe_float(overview.get('MarketCapitalization'))
            
            if ocf and capex and mcap:
                fcf = ocf - capex
                fcf_yield = (fcf / mcap) * 100
                
                possible += 20
                pts = 0
                if fcf_yield > 5: pts = 20
                elif 3 <= fcf_yield <= 5: pts = 10
                earned += pts
                log["FCF Yield"] = f"{fcf_yield:.1f}% ({pts}/20)"
            else: log["FCF Yield"] = "Data Missing"
        else: log["FCF Yield"] = "N/A (No Report)"
    except: log["FCF Yield"] = "Calc Error"
    
    score = int((earned / possible) * 100) if possible > 0 else 0
    return score, log

# ==========================================
# 4. APP UI
# ==========================================

st.title("🦅 Alpha Pro (Premium Edition)")
st.caption("Powered by Alpha Vantage Premium")

# Sidebar
with st.sidebar:
    st.header("Settings")
    av_key = st.text_input("API Key", value=DEFAULT_KEY, type="password")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Analyzer", "⭐ Watchlist", "📊 Sectors"])

# --- TAB 1: ANALYZER ---
with tab1:
    c1, c2 = st.columns([3,1])
    ticker = c1.text_input("Enter Ticker", "MRK").upper()
    
    if ticker and av_key:
        with st.spinner("Fetching Premium Data..."):
            hist, ov, cf = get_alpha_data(ticker, av_key)
            
        if not hist.empty and ov:
            score, breakdown = calculate_alpha_score(ov, cf)
            curr_price = hist['close'].iloc[-1]
            
            # SAVE TO DB
            update_history(ticker, score, curr_price)
            
            # UI
            col_score, col_chart = st.columns([1, 2])
            with col_score:
                st.metric("Alpha Score", f"{score}/100")
                st.write("### Score Metrics")
                st.table(pd.DataFrame(list(breakdown.items()), columns=["Metric", "Value"]))
                st.success(f"✅ Saved {ticker} to Watchlist")
                
            with col_chart:
                st.subheader(f"{ticker} Price Chart")
                # Filter to last 2 years for cleanliness
                subset = hist.tail(500)
                st.line_chart(subset['close'])
                
                # History Graph
                db = load_db()
                if ticker in db and len(db[ticker]) > 1:
                    st.write("### Your Score History")
                    h_df = pd.DataFrame(db[ticker])
                    fig = px.line(h_df, x='date', y='score', markers=True, title="Score Trend")
                    fig.update_yaxes(range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Data not found. Check Ticker or API Key.")

# --- TAB 2: WATCHLIST ---
with tab2:
    st.header("My Watchlist")
    db = load_db()
    if db:
        # Table
        rows = []
        for t, h in db.items():
            last = h[-1]
            rows.append({"Ticker": t, "Date": last['date'], "Score": last['score'], "Price": f"${last['price']:.2f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        
        # Graph
        st.markdown("---")
        st.subheader("Deep Dive")
        sel = st.selectbox("Select Ticker", list(db.keys()))
        if sel:
            h = db[sel]
            if len(h) > 0:
                fig = px.line(pd.DataFrame(h), x='date', y='score', markers=True, title=f"{sel} History")
                fig.update_yaxes(range=[0, 100])
                st.plotly_chart(fig)
    else:
        st.info("Watchlist is empty. Analyze some stocks!")

# --- TAB 3: SECTORS ---
with tab3:
    st.header("Sector ETF Analysis")
    SECTORS = {
        "Tech": "XLK", "Healthcare": "XLV", "Financials": "XLF", 
        "Energy": "XLE", "Semis": "SMH", "Real Estate": "XLRE"
    }
    sel_sec = st.selectbox("Choose Sector", list(SECTORS.keys()))
    t_sec = SECTORS[sel_sec]
    
    if st.button(f"Analyze {sel_sec}"):
        with st.spinner("Fetching Sector Data..."):
            s_hist, _, _ = get_alpha_data(t_sec, av_key)
            
        if not s_hist.empty:
            s_hist['MA50'] = s_hist['close'].rolling(50).mean()
            curr = s_hist['close'].iloc[-1]
            ma50 = s_hist['MA50'].iloc[-1]
            trend = "🟢 BULLISH" if curr > ma50 else "🔴 BEARISH"
            
            c1, c2 = st.columns(2)
            c1.metric("Trend", trend)
            c2.metric("Price vs 50MA", f"${curr:.2f} / ${ma50:.2f}")
            
            st.line_chart(s_hist.tail(252)[['close', 'MA50']])
