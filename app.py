import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Market Analyzer (Yahoo Engine)", layout="wide")

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

if 'trade_log' not in st.session_state:
    st.session_state['trade_log'] = []

# ==========================================
# 2. DATA ENGINE (Yahoo Finance)
# ==========================================

@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    """Fetches Price History and Key Ratios from Yahoo"""
    try:
        stock = yf.Ticker(ticker)
        
        # 1. Get History (1 Year)
        hist = stock.history(period="2y")
        
        # 2. Get Info (Fundamentals)
        info = stock.info
        
        return hist, info
    except:
        return None, None

def safe_get(data, key, default=0):
    """Safely extracts data from Yahoo Info dict"""
    val = data.get(key)
    return val if val is not None else default

# ==========================================
# 3. SCORING LOGIC
# ==========================================

def calculate_score(info):
    score = 0
    
    # 1. Forward P/E
    pe = safe_get(info, 'forwardPE', 35)
    if pe < 20: score += 10
    elif 20 <= pe <= 30: score += 5
    
    # 2. EV / EBITDA
    ev_ebitda = safe_get(info, 'enterpriseToEbitda', 25)
    if ev_ebitda < 12: score += 15
    elif 12 <= ev_ebitda <= 20: score += 7.5
    
    # 3. PEG Ratio
    peg = safe_get(info, 'pegRatio', 3)
    if peg < 1: score += 25
    elif 1 <= peg <= 1.5: score += 12.5
    
    # 4. ROIC Proxy (Using ROE as Yahoo doesn't explicitly have ROIC in standard info)
    # We will use Return on Equity as the closest reliable proxy available free
    roe = safe_get(info, 'returnOnEquity', 0) * 100
    if roe > 20: score += 30
    elif 10 <= roe <= 20: score += 15
    
    # 5. FCF Yield (Free Cash Flow / Market Cap)
    fcf = safe_get(info, 'freeCashflow', 0)
    mcap = safe_get(info, 'marketCap', 1)
    fcf_yield = (fcf / mcap) * 100 if mcap > 0 else 0
    
    if fcf_yield > 5: score += 20
    elif 3 <= fcf_yield <= 5: score += 10
    
    return score, {
        "P/E (Fwd)": pe,
        "EV/EBITDA": ev_ebitda,
        "PEG": peg,
        "ROE (ROIC Proxy)": roe,
        "FCF Yield": fcf_yield
    }

# ==========================================
# 4. APP UI
# ==========================================

st.title("🦅 Market Analyzer (Powered by Yahoo)")
st.markdown("---")

page = st.sidebar.radio("Navigate", ["Individual Stock Analyzer", "Sector Trends", "Trade Tester"])

if page == "Individual Stock Analyzer":
    st.header("Individual Stock Scoring")
    ticker = st.text_input("Enter Ticker", "NVDA").upper()
    
    if ticker:
        with st.spinner(f"Fetching data for {ticker}..."):
            hist, info = get_stock_data(ticker)
            
            if info and len(hist) > 0:
                # Calculate Score
                score, metrics = calculate_score(info)
                
                # Display Score
                st.subheader(f"Quality Score: {score}/100")
                
                # Metrics Row
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Price", f"${hist.iloc[-1]['Close']:.2f}")
                c2.metric("Fwd P/E", f"{metrics['P/E (Fwd)']:.2f}")
                c3.metric("PEG", f"{metrics['PEG']:.2f}")
                c4.metric("ROE", f"{metrics['ROE (ROIC Proxy)']:.1f}%")
                c5.metric("FCF Yield", f"{metrics['FCF Yield']:.1f}%")
                
                # Charts
                tab1, tab2 = st.tabs(["Price Chart", "Score Detail"])
                
                with tab1:
                    st.line_chart(hist['Close'])
                
                with tab2:
                    st.write("### Scoring Breakdown")
                    st.json(metrics)
            else:
                st.error("Ticker not found.")

elif page == "Sector Trends":
    st.header("Sector Dashboard")
    
    sector_name = st.selectbox("Select Sector", list(st.session_state['sectors'].keys()))
    tickers = st.session_state['sectors'][sector_name]
    
    if st.button("Analyze Sector"):
        with st.spinner("Analyzing Sector Breadth... (This takes a moment)"):
            above_sma = 0
            total_stocks = 0
            
            fig = go.Figure()
            
            for t in tickers:
                hist, info = get_stock_data(t)
                if hist is not None and not hist.empty:
                    # Calculate SMA 50
                    sma50 = hist['Close'].rolling(50).mean().iloc[-1]
                    curr = hist['Close'].iloc[-1]
                    
                    if curr > sma50:
                        above_sma += 1
                    total_stocks += 1
                    
                    # Normalize for Chart
                    norm_price = (hist['Close'] / hist['Close'].iloc[0] - 1) * 100
                    fig.add_trace(go.Scatter(x=hist.index, y=norm_price, name=t))
            
            if total_stocks > 0:
                breadth = (above_sma / total_stocks) * 100
                st.metric("Sector Breadth (% > 50 SMA)", f"{breadth:.1f}%")
                st.plotly_chart(fig, use_container_width=True)

elif page == "Trade Tester":
    st.header("Trade Backtester")
    st.caption("Note: Yahoo Finance provides current scores. Historical scoring is approximated by Price change.")
    
    col1, col2 = st.columns(2)
    t_ticker = col1.text_input("Ticker", "MSFT").upper()
    t_date = col2.date_input("Buy Date", datetime.today() - timedelta(days=365))
    
    if st.button("Simulate"):
        hist, info = get_stock_data(t_ticker)
        if hist is not None:
            # Find closest date
            try:
                # Convert date to tz-naive for comparison
                search_date = pd.Timestamp(t_date).tz_localize(None)
                hist.index = hist.index.tz_localize(None)
                
                # Get closest index
                idx = hist.index.get_indexer([search_date], method='nearest')[0]
                buy_price = hist.iloc[idx]['Close']
                curr_price = hist.iloc[-1]['Close']
                
                profit = ((curr_price - buy_price) / buy_price) * 100
                
                st.success(f"Simulated Trade: {t_ticker}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Buy Price", f"${buy_price:.2f}")
                c2.metric("Current Price", f"${curr_price:.2f}")
                c3.metric("Total Return", f"{profit:.2f}%")
                
            except Exception as e:
                st.error(f"Error calculating date: {e}")
