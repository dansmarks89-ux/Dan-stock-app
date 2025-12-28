import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
st.set_page_config(page_title="Alpha Pro v11.0", layout="wide")

DEFAULT_KEY = "GLN6L0BRQIEN59OL"

# Top holdings to simulate "Sector Average" without fetching 500 stocks
SECTOR_HOLDINGS = {
    "Technology": ["AAPL", "MSFT", "NVDA"],
    "Healthcare": ["LLY", "UNH", "JNJ"],
    "Financials": ["JPM", "V", "MA"],
    "Energy": ["XOM", "CVX", "EOG"],
    "Communication": ["GOOGL", "META", "NFLX"],
    "Consumer Disc": ["AMZN", "TSLA", "HD"],
    "Consumer Stap": ["PG", "COST", "WMT"],
    "Industrials": ["GE", "CAT", "UNP"],
    "Utilities": ["NEE", "SO", "DUK"],
    "Real Estate": ["PLD", "AMT", "EQIX"]
}

SECTOR_ETFS = {
    "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF",
    "Energy": "XLE", "Communication": "XLC", "Consumer Disc": "XLY",
    "Consumer Stap": "XLP", "Industrials": "XLI", "Utilities": "XLU",
    "Real Estate": "XLRE"
}

SECTOR_PE_BENCHMARKS = {
    "TECHNOLOGY": 28.5, "HEALTHCARE": 19.2, "FINANCIALS": 14.5,
    "ENERGY": 11.8, "COMMUNICATION": 18.4, "CONSUMER DISC": 22.1,
    "CONSUMER STAP": 20.5, "INDUSTRIALS": 21.0, "UTILITIES": 17.5,
    "REAL ESTATE": 35.0
}

# ==========================================
# 2. GOOGLE SHEETS DATABASE (TICKER, DATE, SCORE)
# ==========================================
conn = st.connection("gsheets", type=GSheetsConnection)

def get_watchlist_data():
    """Reads the full history from Google Sheets."""
    try:
        # Read columns A, B, C (Ticker, Date, Score)
        df = conn.read(worksheet="Sheet1", usecols=[0, 1, 2], ttl=0)
        df = df.dropna(subset=['Ticker'])
        # Ensure correct data types
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
        return df
    except Exception:
        # Return empty DF structure if sheet is new/empty
        return pd.DataFrame(columns=["Ticker", "Date", "Score"])

def add_log_to_sheet(ticker, score):
    """Appends a new entry to the sheet."""
    try:
        existing_df = get_watchlist_data()
        
        # Create new row
        new_row = pd.DataFrame([{
            "Ticker": ticker,
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Score": score
        }])
        
        # Append and save
        updated_df = pd.concat([existing_df, new_row], ignore_index=True)
        conn.update(worksheet="Sheet1", data=updated_df)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False

def remove_ticker_from_sheet(ticker):
    """Removes ALL history for a specific ticker."""
    try:
        df = get_watchlist_data()
        # Filter out the ticker
        df = df[df['Ticker'] != ticker]
        conn.update(worksheet="Sheet1", data=df)
        st.cache_data.clear()
    except Exception as e:
        st.error(f"Remove failed: {e}")

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
        url = f"{base}?function=EARNINGS&symbol={ticker}&apikey={api_key}"
        data = requests.get(url).json()
        q_earnings = data.get('quarterlyEarnings', [])
        
        if not q_earnings: return pd.DataFrame()
        
        eps_df = pd.DataFrame(q_earnings)
        eps_df['fiscalDateEnding'] = pd.to_datetime(eps_df['fiscalDateEnding'])
        eps_df['reportedEPS'] = pd.to_numeric(eps_df['reportedEPS'], errors='coerce')
        eps_df = eps_df.set_index('fiscalDateEnding').sort_index()
        
        # Calculate TTM EPS
        eps_df['ttm_eps'] = eps_df['reportedEPS'].rolling(window=4).sum()
        
        # Merge Price and EPS
        merged = pd.merge_asof(price_df, eps_df['ttm_eps'], left_index=True, right_index=True, direction='backward')
        merged['pe_ratio'] = merged['close'] / merged['ttm_eps']
        
        # Filter Logic: Clean out negative PE or extreme outliers > 300 for graph readability
        merged = merged[(merged['pe_ratio'] > 0) & (merged['pe_ratio'] < 300)]
        
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
# 5. UI HELPERS & PLOTTING
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
st.title("🦅 Alpha Pro v11.0 (Cloud Edition)")

with st.sidebar:
    st.header("Settings")
    key = st.text_input("API Key", value=DEFAULT_KEY, type="password")
    
    # Init watchlist state
    if 'watchlist_df' not in st.session_state:
        st.session_state.watchlist_df = get_watchlist_data()
    
    unique_tickers = st.session_state.watchlist_df['Ticker'].unique().tolist() if not st.session_state.watchlist_df.empty else []
    st.info(f"Tracking {len(unique_tickers)} Stocks")

t1, t2, t3 = st.tabs(["🔍 Analysis", "📈 Watchlist & Trends", "📊 Sectors"])

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
            k2.metric("Div Yield", f"{(safe_float(ov.get('DividendYield', 0)) or 0) * 100:.2f}%")
            k3.metric("Fwd P/E", f"{pe_now:.2f}")
            k4.metric("Sector Avg P/E", sec_avg, delta=f"{sec_avg - pe_now:.1f}")
            
            col_metrics, col_chart = st.columns([1, 2])
            
            with col_metrics:
                st.metric("Quality Score", f"{score}/100")
                st.table(pd.DataFrame(list(log.items()), columns=["Metric", "Value"]))
                
                # --- GOOGLE SHEETS ADD BUTTON ---
                if st.button("⭐ Log to Cloud Watchlist"):
                    success = add_log_to_sheet(tick, score)
                    if success:
                        st.success(f"Logged {tick} (Score: {score}) to History!")
                        st.session_state.watchlist_df = get_watchlist_data() # Refresh local
            
            with col_chart:
                days = tf_selector("ind")
                plot_dual_axis(hist, pe_hist, f"{tick}: Price vs Valuation (P/E)", days)

        else:
            st.error("Data Unavailable.")

# --- TAB 2: WATCHLIST TRENDS ---
with t2:
    st.header("My Watchlist Trends")
    st.caption("Tracking Quality Scores over time from your Google Sheet.")
    
    df_wl = st.session_state.watchlist_df
    
    if df_wl.empty:
        st.info("Watchlist is empty. Analyze a stock to add it.")
    else:
        # Get Unique Tickers
        unique_list = df_wl['Ticker'].unique()
        
        for t in unique_list:
            # Get history for this ticker
            history = df_wl[df_wl['Ticker'] == t].sort_values("Date")
            latest = history.iloc[-1]
            
            c_info, c_plot, c_del = st.columns([1, 2, 0.5])
            
            with c_info:
                st.subheader(t)
                st.metric("Latest Score", f"{latest['Score']}/100", f"Date: {latest['Date']}")
            
            with c_plot:
                # Plot Score Trend if more than 1 point
                if len(history) > 1:
                    fig = px.line(history, x='Date', y='Score', title=f"{t} Score Trend", markers=True)
                    fig.update_layout(height=150, margin=dict(l=0,r=0,t=30,b=0))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("Not enough history for trend graph.")
            
            with c_del:
                st.write("") # Spacer
                if st.button("🗑️", key=f"del_{t}"):
                    remove_ticker_from_sheet(t)
                    st.session_state.watchlist_df = get_watchlist_data()
                    st.rerun()
            st.divider()

# --- TAB 3: SECTOR ROTATION & FLOWS ---
with t3:
    st.header("Sector Rotation & Money Flows")
    st.markdown("""
    **How to read this:**
    * **Money Flow:** We compare each sector against the S&P 500 (SPY).
    * **Rising Line:** Money is rotating **INTO** this sector (Outperforming).
    * **Falling Line:** Money is rotating **OUT** of this sector (Underperforming).
    """)

    # 1. Configuration
    SECTOR_ETFS = {
        "Technology (XLK)": "XLK", 
        "Healthcare (XLV)": "XLV", 
        "Financials (XLF)": "XLF",
        "Energy (XLE)": "XLE", 
        "Communication (XLC)": "XLC", 
        "Consumer Disc (XLY)": "XLY",
        "Consumer Stap (XLP)": "XLP", 
        "Industrials (XLI)": "XLI", 
        "Utilities (XLU)": "XLU",
        "Real Estate (XLRE)": "XLRE",
        "S&P 500 (SPY)": "SPY"
    }

    # 2. Controls
    c_sel, c_tf = st.columns([3, 1])
    with c_sel:
        # Default to comparing Tech vs Energy vs Market
        selected_sectors = st.multiselect(
            "Select Sectors to Compare", 
            list(SECTOR_ETFS.keys()), 
            default=["Technology (XLK)", "Energy (XLE)", "Financials (XLF)"]
        )
    with c_tf:
        days = tf_selector("rot")

    # 3. The Analysis Engine
    if st.button("Analyze Flows", type="primary"):
        if not selected_sectors:
            st.error("Please select at least one sector.")
        else:
            # A. Fetch Benchmark (SPY) first
            with st.spinner("Analyzing Market Flows..."):
                spy_hist, _, _ = get_alpha_data("SPY", key)
                if spy_hist.empty:
                    st.error("Could not fetch S&P 500 data. Check API Key.")
                    st.stop()
                
                # Filter SPY by date
                cutoff = spy_hist.index[-1] - timedelta(days=days)
                spy_sub = spy_hist[spy_hist.index >= cutoff]['close']
                
                # Data container
                df_rel = pd.DataFrame() # For Relative Performance (The "Flows")
                df_abs = pd.DataFrame() # For Absolute Price (The "Trend")
                metrics = []

                # B. Fetch Each Selected Sector
                progress = st.progress(0)
                
                for i, sec_name in enumerate(selected_sectors):
                    ticker = SECTOR_ETFS[sec_name]
                    
                    # Rate limit pause (AlphaVantage free tier is 5 calls/min)
                    if i > 0: time.sleep(1.5) 
                    
                    hist, _, _ = get_alpha_data(ticker, key)
                    
                    if not hist.empty:
                        # Align dates with SPY
                        sec_sub = hist[hist.index >= cutoff]['close']
                        
                        # 1. Calculate Relative Strength (Sector / SPY)
                        # We reindex to match SPY dates to handle any missing days
                        combined = pd.concat([sec_sub, spy_sub], axis=1).dropna()
                        combined.columns = ['Sector', 'SPY']
                        
                        # "Relative Ratio": If > 1, beating market. If rising, gaining momentum.
                        # We normalize start to 0% for easier comparison
                        rel_perf = (combined['Sector'] / combined['SPY'])
                        rel_perf = ((rel_perf / rel_perf.iloc[0]) - 1) * 100
                        
                        df_rel[sec_name] = rel_perf
                        df_abs[sec_name] = sec_sub

                        # 2. Calculate "Overvalued" Metric (Distance from 200MA)
                        # Current Price vs 200 Day Moving Average
                        ma_200 = hist['close'].rolling(window=200).mean().iloc[-1]
                        curr_price = hist['close'].iloc[-1]
                        
                        if pd.notna(ma_200):
                            dist = ((curr_price / ma_200) - 1) * 100
                            status = "🔥 Overheated" if dist > 15 else "❄️ Oversold" if dist < -5 else "Normal"
                        else:
                            dist = 0
                            status = "N/A"

                        metrics.append({
                            "Sector": sec_name,
                            "Price": f"${curr_price:.2f}",
                            "vs 200-Day Avg": f"{dist:+.1f}%",
                            "Status": status
                        })
                    
                    progress.progress((i + 1) / len(selected_sectors))
                
                # C. Visualize
                st.divider()
                
                # 1. The "Flows" Chart
                st.subheader(f"🔄 Money Flows (Relative to S&P 500)")
                st.caption("A rising line means the sector is attracting money (Beating the Market).")
                if not df_rel.empty:
                    fig_rel = px.line(df_rel, title=f"Relative Performance vs SPY (Last {days} Days)")
                    fig_rel.update_layout(yaxis_title="Outperformance %")
                    # Add a zero line
                    fig_rel.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.5)
                    st.plotly_chart(fig_rel, use_container_width=True)

                # 2. The "Overvalued" Metrics
                st.subheader("⚠️ Valuation Check (Technical)")
                st.caption("Sectors >15% above their 200-Day average are often considered 'Overextended'.")
                st.table(pd.DataFrame(metrics).set_index("Sector"))
