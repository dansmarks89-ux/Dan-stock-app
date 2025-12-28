import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Alpha Pro v10.0", layout="wide")

DEFAULT_KEY = "GLN6L0BRQIEN59OL"

SECTOR_PE_BENCHMARKS = {
    "TECHNOLOGY": 28.5, "HEALTHCARE": 19.2, "FINANCIALS": 14.5,
    "ENERGY": 11.8, "COMMUNICATION SERVICES": 18.4, "CONSUMER CYCLICAL": 22.1,
    "CONSUMER DEFENSIVE": 20.5, "INDUSTRIALS": 21.0, "UTILITIES": 17.5,
    "REAL ESTATE": 35.0
}

# ==========================================
# 2. GOOGLE SHEETS CONNECTION & LOGIC
# ==========================================
# Create the connection object
conn = st.connection("gsheets", type=GSheetsConnection)

def load_watchlist():
    """Reads the Google Sheet and returns a list of tickers."""
    try:
        # Read Column A (Ticker) from Sheet1
        df = conn.read(worksheet="Sheet1", usecols=[0], ttl=0)
        # Clean the data: Drop empty rows and remove duplicates
        clean_list = df.iloc[:, 0].dropna().unique().tolist()
        return clean_list
    except Exception:
        # Default if sheet is empty or connection fails
        return ["AAPL", "PLTR"]

def save_watchlist(ticker_list):
    """Overwrites the Google Sheet with the new list."""
    try:
        # Convert list to DataFrame
        df = pd.DataFrame(ticker_list, columns=["Ticker"])
        # Update the sheet
        conn.update(worksheet="Sheet1", data=df)
        # Clear cache so the app sees the update immediately
        st.cache_data.clear()
    except Exception as e:
        st.error(f"Could not save to cloud: {e}")

# ==========================================
# 3. DATA ENGINE
# ==========================================
def safe_float(val):
    try:
        if val is None or val == "None" or val == "-": return None
        return float(val)
    except: return None

@st.cache_data(ttl=300)
def get_alpha_data(ticker, api_key):
    base = "https://www.alphavantage.co/query"
    try: r_ov = requests.get(f"{base}?function=OVERVIEW&symbol={ticker}&apikey={api_key}").json()
    except: r_ov = {}

    try: r_cf = requests.get(f"{base}?function=CASH_FLOW&symbol={ticker}&apikey={api_key}").json()
    except: r_cf = {}
    
    # Fetch Price History
    try:
        r_price = requests.get(f"{base}?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={api_key}").json()
        ts = r_price.get('Time Series (Daily)', {})
        if ts:
            df = pd.DataFrame.from_dict(ts, orient='index')
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df.rename(columns={'4. close': 'close'})
        else: df = pd.DataFrame()
    except: df = pd.DataFrame()
        
    return df, r_ov, r_cf

@st.cache_data(ttl=3600)
def get_historical_pe(ticker, api_key, price_df):
    """Calculates Daily P/E Ratio History."""
    if price_df.empty: return pd.DataFrame()
    
    base = "https://www.alphavantage.co/query"
    try:
        # Fetch Quarterly Earnings
        url = f"{base}?function=EARNINGS&symbol={ticker}&apikey={api_key}"
        data = requests.get(url).json()
        q_earnings = data.get('quarterlyEarnings', [])
        
        if not q_earnings: return pd.DataFrame()
        
        # Process Earnings into DataFrame
        eps_df = pd.DataFrame(q_earnings)
        eps_df['fiscalDateEnding'] = pd.to_datetime(eps_df['fiscalDateEnding'])
        eps_df['reportedEPS'] = pd.to_numeric(eps_df['reportedEPS'], errors='coerce')
        eps_df = eps_df.set_index('fiscalDateEnding').sort_index()
        
        # Calculate TTM EPS
        eps_df['ttm_eps'] = eps_df['reportedEPS'].rolling(window=4).sum()
        
        # Merge Price and EPS
        merged = pd.merge_asof(price_df, eps_df['ttm_eps'], left_index=True, right_index=True, direction='backward')
        merged['pe_ratio'] = merged['close'] / merged['ttm_eps']
        merged = merged[(merged['pe_ratio'] > 0) & (merged['pe_ratio'] < 200)]
        
        return merged[['pe_ratio']]
        
    except: return pd.DataFrame()

# ==========================================
# 4. SCORING ENGINE
# ==========================================
def get_points(val, best, worst, max_pts, high_is_good=False):
    if val is None: return 0
    if high_is_good:
        if val >= best: return max_pts
        if val <= worst: return 0
        return round(((val - worst)/(best - worst)) * max_pts, 1)
    else:
        if val <= best: return max_pts
        if val >= worst: return 0
        return round(((worst - val)/(worst - best)) * max_pts, 1)

def calculate_dynamic_score(overview, cash_flow):
    earned, possible = 0, 0
    log = {}
    
    # 1. PE (10pts)
    pe = safe_float(overview.get('ForwardPE'))
    if pe:
        pts = get_points(pe, 15, 40, 10)
        earned += pts; possible += 10
        log["Forward P/E"] = f"{pe:.2f} ({pts}/10)"
    else: log["Fwd P/E"] = "N/A"

    # 2. EV/EBITDA (15pts)
    ev = safe_float(overview.get('EVToEBITDA'))
    if ev:
        pts = get_points(ev, 10, 30, 15)
        earned += pts; possible += 15
        log["EV/EBITDA"] = f"{ev:.2f} ({pts}/15)"
    else: log["EV/EBITDA"] = "N/A"

    # 3. PEG (25pts)
    peg = safe_float(overview.get('PEGRatio'))
    if peg:
        pts = get_points(peg, 1.0, 3.0, 25)
        earned += pts; possible += 25
        log["PEG Ratio"] = f"{peg:.2f} ({pts}/25)"
    else: log["PEG"] = "N/A"

    # 4. ROE (30pts)
    roe = safe_float(overview.get('ReturnOnEquityTTM'))
    if roe:
        if roe < 5: roe = roe * 100
        pts = get_points(roe, 30, 5, 30, True)
        earned += pts; possible += 30
        log["ROE"] = f"{roe:.1f}% ({pts}/30)"
    else: log["ROE"] = "N/A"

    # 5. FCF Yield (20pts)
    fcf_val = None
    try:
        rep = cash_flow.get('annualReports', [])[0]
        ocf = safe_float(rep.get('operatingCashflow'))
        cap = safe_float(rep.get('capitalExpenditures'))
        mc = safe_float(overview.get('MarketCapitalization'))
        if ocf and cap and mc:
            fcf_val = ((ocf - cap)/mc)*100
    except: pass
    
    if fcf_val is not None:
        pts = get_points(fcf_val, 6, 1, 20, True)
        earned += pts; possible += 20
        log["FCF Yield"] = f"{fcf_val:.1f}% ({pts}/20)"
    else: log["FCF Yield"] = "N/A"

    score = int((earned/possible)*100) if possible > 0 else 0
    return score, log

# ==========================================
# 5. UI HELPERS
# ==========================================
def tf_selector(key_suffix):
    c_tf = st.columns(4)
    tf_map = {"1M": 30, "3M": 90, "1Y": 365, "5Y": 1825}
    choice = st.radio("Range", list(tf_map.keys()), index=2, horizontal=True, key=f"tf_{key_suffix}")
    return tf_map[choice]

def plot_dual_axis(price_df, pe_df, title, days):
    cutoff = price_df.index[-1] - timedelta(days=days)
    p_sub = price_df[price_df.index >= cutoff]
    
    fig = go.Figure()
    # 1. Price Line
    fig.add_trace(go.Scatter(
        x=p_sub.index, y=p_sub['close'], 
        name="Price ($)", line=dict(color='#00CC96', width=2)
    ))
    # 2. PE Line
    if not pe_df.empty:
        pe_sub = pe_df[pe_df.index >= cutoff]
        fig.add_trace(go.Scatter(
            x=pe_sub.index, y=pe_sub['pe_ratio'], 
            name="P/E Ratio", line=dict(color='#636EFA', width=2, dash='dot'),
            yaxis="y2"
        ))
    
    fig.update_layout(
        title=title,
        yaxis=dict(title="Price ($)"),
        yaxis2=dict(title="P/E Ratio", overlaying="y", side="right"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. MAIN APP
# ==========================================
st.title("🦅 Alpha Pro v10.0 (Cloud Edition)")

with st.sidebar:
    st.header("Settings")
    key = st.text_input("API Key", value=DEFAULT_KEY, type="password")
    
    # Initialize Watchlist from Cloud on Startup
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = load_watchlist()
    
    st.info(f"Watchlist Loaded: {len(st.session_state.watchlist)} Tickers")

t1, t2, t3 = st.tabs(["🔍 Analysis", "📈 Manage Watchlist", "📊 Sectors"])

# --- TAB 1: ANALYSIS ---
with t1:
    c1, c2 = st.columns([3, 1])
    tick = c1.text_input("Analyze Ticker", "AAPL").upper()
    
    if tick and key:
        with st.spinner("Fetching Fundamentals..."):
            hist, ov, cf = get_alpha_data(tick, key)
            
        if not hist.empty and ov:
            score, log = calculate_dynamic_score(ov, cf)
            
            with st.spinner("Calculating Historical P/E..."):
                pe_hist = get_historical_pe(tick, key, hist)
            
            # Header Metrics
            pe_now = safe_float(ov.get('ForwardPE', 0))
            sec = ov.get('Sector', 'Unknown')
            sec_avg = SECTOR_PE_BENCHMARKS.get(sec.upper(), 20.0)
            
            st.markdown(f"## {tick} - {ov.get('Name')}")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Market Cap", f"${safe_float(ov.get('MarketCapitalization',0))/1e9:.1f} B")
            # Fixed the Division Error logic here
            k2.metric("Div Yield", f"{(safe_float(ov.get('DividendYield', 0)) or 0) * 100:.2f}%")
            k3.metric("Fwd P/E", f"{pe_now:.2f}")
            k4.metric("Sector Avg P/E", sec_avg, delta=f"{sec_avg - pe_now:.1f}")
            
            col_metrics, col_chart = st.columns([1, 2])
            
            with col_metrics:
                st.metric("Quality Score", f"{score}/100")
                st.table(pd.DataFrame(list(log.items()), columns=["Metric", "Value"]))
                
                # --- GOOGLE SHEETS ADD BUTTON ---
                if st.button("⭐ Add to Cloud Watchlist"):
                    if tick not in st.session_state.watchlist:
                        st.session_state.watchlist.append(tick)
                        save_watchlist(st.session_state.watchlist)
                        st.success(f"Saved {tick} to Google Sheets!")
                    else:
                        st.warning(f"{tick} is already in your watchlist.")

            with col_chart:
                days = tf_selector("ind")
                plot_dual_axis(hist, pe_hist, f"{tick}: Price vs Valuation (P/E)", days)

        else:
            st.error("Data Unavailable.")

# --- TAB 2: WATCHLIST (CONNECTED TO SHEETS) ---
with t2:
    st.header("My Cloud Watchlist")
    st.caption("Data is synced with your Google Sheet.")
    
    if not st.session_state.watchlist:
        st.info("Your watchlist is empty. Go to the Analysis tab to add stocks!")
    else:
        # Loop through watchlist items
        for ticker in st.session_state.watchlist:
            col_txt, col_btn = st.columns([4, 1])
            with col_txt:
                st.subheader(ticker)
            with col_btn:
                # Unique key needed for every button
                if st.button(f"Remove {ticker}", key=f"del_{ticker}"):
                    st.session_state.watchlist.remove(ticker)
                    save_watchlist(st.session_state.watchlist)
                    st.rerun()
            st.divider()

# --- TAB 3: SECTORS ---
with t3:
    st.header("Sectors")
    SECTORS = {"Tech": "XLK", "Healthcare": "XLV", "Financials": "XLF", "Energy": "XLE"}
    s = st.selectbox("Sector", list(SECTORS.keys()))
    days = tf_selector("sec")
    if st.button("Analyze Sector"):
        h, _, _ = get_alpha_data(SECTORS[s], key)
        if not h.empty:
            st.line_chart(h.tail(days)['close'])
