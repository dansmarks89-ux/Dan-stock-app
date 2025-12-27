import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Market Analyzer v6.0 (Hybrid Engine)", layout="wide")

DEFAULT_SECTORS = {
    "Tech": ["NVDA", "AAPL", "MSFT", "AVGO", "ORCL"],
    "Healthcare": ["LLY", "UNH", "JNJ", "NVO", "MRK"],
    "Financials": ["JPM", "V", "MA", "BAC", "WFC"],
}
if 'sectors' not in st.session_state:
    st.session_state['sectors'] = DEFAULT_SECTORS

# ==========================================
# 2. DATA ENGINES
# ==========================================

# --- ENGINE A: YAHOO (For Charts) ---
@st.cache_data(ttl=900)
def get_chart_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")
        return hist
    except: return None

# --- ENGINE B: ALPHA VANTAGE (For Accurate Ratios) ---
@st.cache_data(ttl=3600*24)
def get_alpha_fundamentals(ticker, api_key):
    # 1. Company Overview (PE, PEG, EBITDA, ROE)
    url_overview = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}"
    
    try:
        r = requests.get(url_overview)
        data = r.json()
        
        # Check if limit reached or invalid key
        if "Note" in data or "Information" in data:
            return None, "API Limit Reached (25 calls/day) or Invalid Key."
        if not data:
            return None, "Ticker not found in Alpha Vantage."
            
        return data, None
    except Exception as e:
        return None, str(e)

def safe_float(val):
    try:
        if val is None or val == "None" or val == "-": return None
        return float(val)
    except: return None

# ==========================================
# 3. SCORING LOGIC (Using Alpha Vantage Data)
# ==========================================

def calculate_hybrid_score(av_data):
    """Calculates score using Alpha Vantage's clean data"""
    earned = 0
    possible = 0
    log = {}
    
    # 1. Forward P/E
    pe = safe_float(av_data.get('ForwardPE'))
    if pe:
        possible += 10
        pts = 0
        if pe < 20: pts = 10
        elif 20 <= pe <= 30: pts = 5
        earned += pts
        log["Forward P/E"] = f"{pe:.2f} ({pts}/10)"
    else: log["Forward P/E"] = "N/A"
    
    # 2. EV / EBITDA
    ev_ebitda = safe_float(av_data.get('EVToEBITDA'))
    if ev_ebitda:
        possible += 15
        pts = 0
        if ev_ebitda < 12: pts = 15
        elif 12 <= ev_ebitda <= 20: pts = 7.5
        earned += pts
        log["EV/EBITDA"] = f"{ev_ebitda:.2f} ({pts}/15)"
    else: log["EV/EBITDA"] = "N/A"
    
    # 3. PEG Ratio
    peg = safe_float(av_data.get('PEGRatio'))
    if peg:
        possible += 25
        pts = 0
        if peg < 1: pts = 25
        elif 1 <= peg <= 1.5: pts = 12.5
        earned += pts
        log["PEG Ratio"] = f"{peg:.2f} ({pts}/25)"
    else: log["PEG Ratio"] = "N/A"
    
    # 4. ROIC (Using ReturnOnEquityTTM as AlphaVantage Proxy is cleaner)
    # AlphaVantage normalizes this, so it's safer than Yahoo's raw math
    roe = safe_float(av_data.get('ReturnOnEquityTTM'))
    if roe:
        roe = roe * 100 if roe < 1 else roe # Handle decimal vs %
        possible += 30
        pts = 0
        if roe > 20: pts = 30
        elif 10 <= roe <= 20: pts = 15
        earned += pts
        log["ROE (Quality Proxy)"] = f"{roe:.1f}% ({pts}/30)"
    else: log["ROE"] = "N/A"
    
    # 5. Operating Margin (Replacement for FCF Yield)
    # Why? FCF requires a second API call (burning your limits). 
    # Operating Margin is a great proxy for Cash Efficiency.
    op_margin = safe_float(av_data.get('OperatingMarginTTM'))
    if op_margin:
        op_margin = op_margin * 100 if op_margin < 1 else op_margin
        possible += 20
        pts = 0
        if op_margin > 20: pts = 20
        elif 10 <= op_margin <= 20: pts = 10
        earned += pts
        log["Operating Margin"] = f"{op_margin:.1f}% ({pts}/20)"
    else: log["Op Margin"] = "N/A"
    
    score = int((earned / possible) * 100) if possible > 0 else 0
    return score, log

# ==========================================
# 4. APP UI
# ==========================================

st.title("🦅 Market Analyzer v6.0 (Hybrid Engine)")
st.caption("Charts by Yahoo (Unlimited) | Scoring by Alpha Vantage (Accurate)")

# Sidebar for API Key
with st.sidebar:
    st.header("Settings")
    av_key = st.text_input("Alpha Vantage Key", type="password", help="Get free key at alphavantage.co")
    if not av_key:
        st.warning("⚠️ Enter Key for Accurate Scores")
        # Use a demo key for IBM tests only
        if st.checkbox("Use Demo Key (IBM Only)"):
            av_key = "demo"

ticker = st.text_input("Enter Ticker", "IBM").upper()

if ticker:
    # 1. Get Chart (Yahoo)
    hist = get_chart_data(ticker)
    if hist is not None and not hist.empty:
        curr_price = hist['Close'].iloc[-1]
        
        # 2. Get Score (Alpha Vantage)
        if av_key:
            av_data, error = get_alpha_fundamentals(ticker, av_key)
            
            if av_data:
                score, breakdown = calculate_hybrid_score(av_data)
                
                # --- DISPLAY ---
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.metric("Hybrid Quality Score", f"{score}/100")
                    st.write("### Alpha Vantage Data")
                    st.table(pd.DataFrame(list(breakdown.items()), columns=["Metric", "Value"]))
                with c2:
                    st.subheader(f"Price Chart (${curr_price:.2f})")
                    st.line_chart(hist['Close'])
            else:
                st.warning(f"⚠️ Scoring Unavailable: {error}")
                st.line_chart(hist['Close'])
        else:
            st.info("ℹ️ Enter Alpha Vantage Key in Sidebar to see Scores.")
            st.line_chart(hist['Close'])
            
    else:
        st.error("Ticker not found in Yahoo Finance.")
