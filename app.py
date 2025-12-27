import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Market Analyzer v5.0", layout="wide")

# Default Sector Definitions
DEFAULT_SECTORS = {
    "Tech": ["VGT", "NVDA", "AAPL", "MSFT", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "CSCO", "ACN", "TXN", "IBM"],
    "Healthcare": ["XLV", "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "AMGN", "PFE"],
    "Financials": ["XLF", "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "SPGI", "AXP"],
    "Communication": ["XLC", "NFLX", "DIS", "CMCSA", "TMUS", "T", "VZ", "CHTR", "EA", "TTWO"],
    "Consumer Discretionary": ["XLY", "HD", "MCD", "BKNG", "TJX", "LOW", "SBUX", "NKE", "GM", "CMG"],
    "Consumer Staples": ["XLP", "WMT", "PG", "COST", "KO", "PEP", "PM", "MO", "CL", "TGT", "MDLZ", "KMB"],
    "Energy": ["XLE", "XOM", "CVX", "COP", "SLB", "EOG", "WMB", "MPC", "PSX", "VLO", "OXY"],
    "Industrials": ["XLI", "GE", "CAT", "RTX", "UBER", "UNP", "HON", "ETN", "BA", "DE", "LMT"],
    "Utilities": ["XLU", "NEE", "SO", "DUK", "CEG", "AEP", "SRE", "D", "EXC", "XEL"],
    "Real Estate": ["XLRE", "PLD", "AMT", "EQIX", "WELL", "SPG", "O", "DLR", "PSA", "CCI"],
    "Materials": ["XLB", "LIN", "SHW", "FCX", "NEM", "ECL", "NUE", "APD", "MLM", "VMC"],
    "MAG 7": ["TSLA", "GOOG", "NVDA", "AAPL", "MSFT", "META", "AMZN"],
    "AI": ["AIQ", "NVDA", "AMD", "AVGO", "TSM", "MU", "MSFT", "GOOGL", "ORCL", "PLTR", "META"]
}

if 'sectors' not in st.session_state:
    st.session_state['sectors'] = DEFAULT_SECTORS

# ==========================================
# 2. DATA ENGINE (Robust)
# ==========================================

@st.cache_data(ttl=900) # Cache for 15 mins
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    
    # 1. Fetch History (Robust)
    try:
        hist = stock.history(period="2y")
    except: 
        hist = pd.DataFrame()
        
    # 2. Fetch Info (Robust)
    try:
        info = stock.info
    except: 
        info = {}
        
    return hist, info

def safe_float(val):
    try:
        if val is None or val == "None": return None
        return float(val)
    except: return None

# ==========================================
# 3. SCORING LOGIC (Fixed Math)
# ==========================================

def calculate_smart_score(info):
    earned_points = 0
    total_possible = 0
    metrics_log = {}
    
    # --- 1. Forward P/E (Weight: 10) ---
    pe = safe_float(info.get('forwardPE'))
    if pe is not None:
        total_possible += 10
        pts = 0
        if pe < 20: pts = 10
        elif 20 <= pe <= 30: pts = 5
        earned_points += pts
        metrics_log["Forward P/E"] = f"{pe:.2f} ({pts}/10)"
    else:
        metrics_log["Forward P/E"] = "N/A"

    # --- 2. EV / EBITDA (Weight: 15) ---
    ev_ebitda = safe_float(info.get('enterpriseToEbitda'))
    if ev_ebitda is not None:
        total_possible += 15
        pts = 0
        if ev_ebitda < 12: pts = 15
        elif 12 <= ev_ebitda <= 20: pts = 7.5
        earned_points += pts
        metrics_log["EV/EBITDA"] = f"{ev_ebitda:.2f} ({pts}/15)"
    else:
        metrics_log["EV/EBITDA"] = "N/A"

    # --- 3. PEG Ratio (Weight: 25) ---
    peg = safe_float(info.get('pegRatio'))
    if peg is not None:
        total_possible += 25
        pts = 0
        if peg < 1: pts = 25
        elif 1 <= peg <= 1.5: pts = 12.5
        earned_points += pts
        metrics_log["PEG Ratio"] = f"{peg:.2f} ({pts}/25)"
    else:
        metrics_log["PEG Ratio"] = "N/A"

    # --- 4. ROIC (Weight: 30) ---
    roic = None
    try:
        rev = safe_float(info.get('totalRevenue'))
        margin = safe_float(info.get('operatingMargins'))
        
        # CORRECTED MATH: Use Total Equity (or Book Value * Shares)
        equity = safe_float(info.get('totalStockholderEquity'))
        if equity is None:
            bv = safe_float(info.get('bookValue'))
            shares = safe_float(info.get('sharesOutstanding'))
            if bv and shares: equity = bv * shares
            
        debt = safe_float(info.get('totalDebt'))
        
        if rev and margin and equity:
            op_income = rev * margin
            nopat = op_income * 0.79 # 21% Tax Assumption
            invested_capital = equity + (debt if debt else 0)
            if invested_capital > 0:
                roic = (nopat / invested_capital) * 100
    except: pass
    
    # Fallback to ROE
    is_proxy = False
    if roic is None:
        roic = safe_float(info.get('returnOnEquity'))
        if roic: 
            roic = roic * 100
            is_proxy = True

    if roic is not None:
        if roic > 100 and not is_proxy: roic = 100 # Cap logic
        total_possible += 30
        pts = 0
        if roic > 20: pts = 30
        elif 10 <= roic <= 20: pts = 15
        earned_points += pts
        label = "ROE (Proxy)" if is_proxy else "ROIC (Calc)"
        metrics_log[label] = f"{roic:.1f}% ({pts}/30)"
    else:
        metrics_log["ROIC"] = "N/A"

    # --- 5. FCF Yield (Weight: 20) ---
    fcf = safe_float(info.get('freeCashflow'))
    mcap = safe_float(info.get('marketCap'))
    
    if fcf is not None and mcap is not None and mcap > 0:
        total_possible += 20
        fcf_yield = (fcf / mcap) * 100
        pts = 0
        if fcf_yield > 5: pts = 20
        elif 3 <= fcf_yield <= 5: pts = 10
        earned_points += pts
        metrics_log["FCF Yield"] = f"{fcf_yield:.1f}% ({pts}/20)"
    else:
        metrics_log["FCF Yield"] = "N/A"

    # Final Calc
    if total_possible > 0:
        final_score = (earned_points / total_possible) * 100
    else:
        final_score = 0
        
    return int(final_score), metrics_log

# ==========================================
# 4. APP UI
# ==========================================

st.title("📈 Market Analyzer v5.0")
st.markdown("---")

page = st.sidebar.radio("Navigate", ["Individual Stock Analyzer", "Sector Trends", "Trade Tester"])

if page == "Individual Stock Analyzer":
    st.header("Individual Stock Scoring")
    c1, c2 = st.columns([3, 1])
    ticker = c1.text_input("Enter Ticker", "NVDA").upper()
    
    if c2.button("Reload"):
        st.cache_data.clear()
        st.rerun()

    if ticker:
        with st.spinner("Analyzing..."):
            hist, info = get_stock_data(ticker)
            
            # Check 1: Do we have History?
            if hist is not None and not hist.empty:
                
                # Check 2: Do we have Info for scoring?
                if info and 'symbol' in info: 
                    score, breakdown = calculate_smart_score(info)
                    
                    # --- UI Layout ---
                    col_metrics, col_chart = st.columns([1, 2])
                    
                    with col_metrics:
                        st.metric("Quality Score", f"{score}/100")
                        st.write("### Scorecard")
                        st.table(pd.DataFrame(list(breakdown.items()), columns=["Metric", "Value"]))
                        
                    with col_chart:
                        curr_price = hist['Close'].iloc[-1]
                        st.subheader(f"Price Chart (${curr_price:.2f})")
                        st.line_chart(hist['Close'])
                else:
                    # History works, but Info failed
                    st.warning("⚠️ Chart loaded, but Fundamental Data (Scoring) is unavailable from Yahoo right now.")
                    st.line_chart(hist['Close'])
            else:
                st.error("❌ Ticker not found (or Yahoo is blocking this specific request).")

elif page == "Sector Trends":
    st.header("Sector Dashboard")
    sector_name = st.selectbox("Select Sector", list(st.session_state['sectors'].keys()))
    tickers = st.session_state['sectors'][sector_name]
    
    if st.button("Analyze Sector"):
        with st.spinner("Analyzing..."):
            above_sma = 0
            total_stocks = 0
            fig = go.Figure()
            
            for t in tickers:
                hist, _ = get_stock_data(t) # Only need history here
                if hist is not None and not hist.empty:
                    sma50 = hist['Close'].rolling(50).mean().iloc[-1]
                    curr = hist['Close'].iloc[-1]
                    if curr > sma50: above_sma += 1
                    total_stocks += 1
                    norm = (hist['Close'] / hist['Close'].iloc[0] - 1) * 100
                    fig.add_trace(go.Scatter(x=hist.index, y=norm, name=t))
            
            if total_stocks > 0:
                breadth = (above_sma / total_stocks) * 100
                st.metric("Breadth (% > 50 SMA)", f"{breadth:.1f}%")
                st.plotly_chart(fig, use_container_width=True)

elif page == "Trade Tester":
    st.header("Backtest Simulator")
    c1, c2 = st.columns(2)
    t_ticker = c1.text_input("Ticker", "MSFT").upper()
    t_date = c2.date_input("Buy Date", datetime.today() - timedelta(days=365))
    
    if st.button("Run Simulation"):
        hist, _ = get_stock_data(t_ticker)
        if hist is not None and not hist.empty:
            try:
                hist.index = hist.index.tz_localize(None)
                search_date = pd.Timestamp(t_date)
                
                # Ensure date is within range
                if search_date < hist.index[0] or search_date > hist.index[-1]:
                    st.error("Date is outside available data range (last 2 years).")
                else:
                    idx = hist.index.get_indexer([search_date], method='nearest')[0]
                    buy = hist.iloc[idx]['Close']
                    curr = hist.iloc[-1]['Close']
                    ret = ((curr - buy)/buy)*100
                    
                    st.success(f"Trade Result: {t_ticker}")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Buy Price", f"${buy:.2f}")
                    c2.metric("Current Price", f"${curr:.2f}")
                    c3.metric("Total Return", f"{ret:.2f}%")
            except Exception as e: st.error(f"Error: {e}")
        else:
            st.error("Could not fetch data.")
