import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Market Analyzer v6.2 (Mega-Cap Safe)", layout="wide")

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

@st.cache_data(ttl=900)
def get_chart_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")
        return hist
    except: return None

@st.cache_data(ttl=3600*24)
def get_alpha_fundamentals(ticker, api_key):
    url_overview = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}"
    try:
        r = requests.get(url_overview)
        data = r.json()
        if "Note" in data: return None, "⚠️ API Limit Reached."
        if not data: return None, "Ticker not found."
        return data, None
    except Exception as e:
        return None, str(e)

def safe_float(val):
    try:
        if val is None or val == "None" or val == "-": return None
        return float(val)
    except: return None

# ==========================================
# 3. SCORING LOGIC (Fixed for Apple ROE)
# ==========================================

def calculate_hybrid_score(av_data):
    earned = 0
    possible = 0
    log = {}
    
    # 1. Forward P/E (Weight: 10)
    pe = safe_float(av_data.get('ForwardPE'))
    if pe:
        possible += 10
        pts = 0
        if pe < 20: pts = 10
        elif 20 <= pe <= 30: pts = 5
        earned += pts
        log["Forward P/E"] = f"{pe:.2f} ({pts}/10)"
    else: log["Forward P/E"] = "N/A"
    
    # 2. EV / EBITDA (Weight: 15)
    ev_ebitda = safe_float(av_data.get('EVToEBITDA'))
    if ev_ebitda:
        possible += 15
        pts = 0
        if ev_ebitda < 12: pts = 15
        elif 12 <= ev_ebitda <= 20: pts = 7.5
        earned += pts
        log["EV/EBITDA"] = f"{ev_ebitda:.2f} ({pts}/15)"
    else: log["EV/EBITDA"] = "N/A"
    
    # 3. PEG Ratio (Weight: 25)
    peg = safe_float(av_data.get('PEGRatio'))
    if peg:
        possible += 25
        pts = 0
        if peg < 1: pts = 25
        elif 1 <= peg <= 1.5: pts = 12.5
        earned += pts
        log["PEG Ratio"] = f"{peg:.2f} ({pts}/25)"
    else: log["PEG Ratio"] = "N/A"
    
    # 4. ROE (Weight: 30) - THE FIX
    roe = safe_float(av_data.get('ReturnOnEquityTTM'))
    if roe:
        # LOGIC FIX:
        # If ROE is e.g. 0.15 -> It becomes 15.0%
        # If ROE is e.g. 1.70 (Apple) -> It becomes 170.0%
        # If ROE is e.g. 15.0 (Already percent) -> It stays 15.0%
        if roe < 5: 
            roe = roe * 100
            
        possible += 30
        pts = 0
        if roe > 20: pts = 30
        elif 10 <= roe <= 20: pts = 15
        earned += pts
        log["ROE (Quality)"] = f"{roe:.1f}% ({pts}/30)"
    else: log["ROE"] = "N/A"
    
    # 5. Operating Margin (Weight: 20)
    op_margin = safe_float(av_data.get('OperatingMarginTTM'))
    if op_margin:
        # Same fix for Margin just in case
        if op_margin < 5: op_margin = op_margin * 100
            
        possible += 20
        pts = 0
        if op_margin > 20: pts = 20
        elif 10 <= op_margin <= 20: pts = 10
        earned += pts
        log["Op Margin"] = f"{op_margin:.1f}% ({pts}/20)"
    else: log["Op Margin"] = "N/A"
    
    score = int((earned / possible) * 100) if possible > 0 else 0
    return score, log

# ==========================================
# 4. APP UI
# ==========================================

st.title("🦅 Market Analyzer v6.2 (Mega-Cap Safe)")

with st.sidebar:
    st.header("API Settings")
    av_key = st.text_input("Alpha Vantage Key", value="O1U1GWC8OQ4RRBL1", type="password")

ticker = st.text_input("Enter Ticker", "AAPL").upper()

if ticker:
    hist = get_chart_data(ticker)
    
    if hist is not None and not hist.empty:
        curr_price = hist['Close'].iloc[-1]
        
        if av_key:
            with st.spinner("Fetching Professional Data..."):
                av_data, error = get_alpha_fundamentals(ticker, av_key)
            
            if av_data:
                score, breakdown = calculate_hybrid_score(av_data)
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.metric("Hybrid Quality Score", f"{score}/100")
                    st.write("### Alpha Vantage Data")
                    st.table(pd.DataFrame(list(breakdown.items()), columns=["Metric", "Value"]))
                with c2:
                    st.subheader(f"Price Chart (${curr_price:.2f})")
                    st.line_chart(hist['Close'])
            else:
                st.warning(f"Data Unavailable: {error}")
                st.line_chart(hist['Close'])
    else:
        st.error("Ticker not found.")
