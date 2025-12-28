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
st.set_page_config(page_title="Alpha Pro v13.0", layout="wide")

# (Note: Hardcoded API Key removed for security)

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

SECTOR_PE_BENCHMARKS = {
    "TECHNOLOGY": 28.5, "HEALTHCARE": 19.2, "FINANCIALS": 14.5,
    "ENERGY": 11.8, "COMMUNICATION": 18.4, "CONSUMER DISC": 22.1,
    "CONSUMER STAP": 20.5, "INDUSTRIALS": 21.0, "UTILITIES": 17.5,
    "REAL ESTATE": 35.0
}

# ==========================================
# 2. GOOGLE SHEETS DATABASE
# ==========================================
conn = st.connection("gsheets", type=GSheetsConnection)

def get_watchlist_data():
    """Reads the full history from Google Sheets."""
    try:
        df = conn.read(worksheet="Sheet1", usecols=[0, 1, 2], ttl=0)
        df = df.dropna(subset=['Ticker'])
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
        return df
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Date", "Score"])

def add_log_to_sheet(ticker, score):
    """Appends a new entry to the sheet."""
    try:
        existing_df = get_watchlist_data()
        new_row = pd.DataFrame([{
            "Ticker": ticker,
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Score": score
        }])
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
        
        eps_df['ttm_eps'] = eps_df['reportedEPS'].rolling(window=4).sum()
        merged = pd.merge_asof(price_df, eps_df['ttm_eps'], left_index=True, right_index=True, direction='backward')
        merged['pe_ratio'] = merged['close'] / merged['ttm_eps']
        merged = merged[(merged['pe_ratio'] > 0) & (merged['pe_ratio'] < 300)]
        return merged[['pe_ratio']]
    except: return pd.DataFrame()

# ==========================================
# 4. SCORING ENGINE (V2.1 - DYNAMIC WEIGHTS)
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

def calculate_dynamic_score(overview, cash_flow, price_df, weights):
    """
    Alpha Score v2.1: Dynamic Weighting based on Market Regime
    """
    earned, possible = 0, 0
    log = {}
    
    # Helper to clean up the logic
    def process_metric(label, raw_val, weight_key, base_score):
        nonlocal earned, possible
        w = weights[weight_key]
        if raw_val is not None:
            weighted_points = (base_score / 20) * w
            earned += weighted_points
            possible += w
            return f"{raw_val} ({weighted_points:.1f}/{w})"
        else:
            return "N/A"

    # 1. GROWTH
    rev_growth = safe_float(overview.get('QuarterlyRevenueGrowthYOY'))
    base_pts = get_points(rev_growth * 100, 20, 0, 20, True) if rev_growth else 0
    log["Revenue Growth"] = process_metric("Rev Growth", f"{rev_growth*100:.1f}%" if rev_growth else None, 'growth', base_pts)

    # 2. PROFITABILITY
    margin = safe_float(overview.get('ProfitMargin'))
    base_pts = get_points(margin * 100, 25, 5, 20, True) if margin else 0
    log["Profit Margin"] = process_metric("Margin", f"{margin*100:.1f}%" if margin else None, 'margins', base_pts)

    # 3. QUALITY
    roe = safe_float(overview.get('ReturnOnEquityTTM'))
    if roe and roe < 5: roe = roe * 100 
    base_pts = get_points(roe, 25, 5, 20, True) if roe else 0
    log["ROE"] = process_metric("ROE", f"{roe:.1f}%" if roe else None, 'roe', base_pts)

    # 4. VALUE
    peg = safe_float(overview.get('PEGRatio'))
    base_pts = get_points(peg, 1.0, 2.5, 20) if peg else 0
    log["PEG Ratio"] = process_metric("PEG", f"{peg:.2f}" if peg else None, 'value', base_pts)

    # 5. MOMENTUM (With Parabolic Penalty)
    pct_diff = None
    base_pts = 0
    if not price_df.empty and len(price_df) > 200:
        curr_price = price_df['close'].iloc[-1]
        ma_200 = price_df['close'].rolling(window=200).mean().iloc[-1]
        pct_diff = ((curr_price / ma_200) - 1) * 100
        
        if pct_diff < 0:
            base_pts = 0
            note = "Downtrend"
        elif 0 <= pct_diff <= 25:
            base_pts = 20
            note = "Healthy Trend"
        else:
            # PARABOLIC PENALTY
            penalty = (pct_diff - 25) * 0.5 
            base_pts = max(0, 20 - penalty)
            note = "⚠️ Overheated"
    else:
        note = "No Data"

    val_str = f"{pct_diff:+.1f}% ({note})" if pct_diff is not None else None
    log["vs 200-Day MA"] = process_metric("Momentum", val_str, 'momentum', base_pts)

    # Final Score
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
    fig.add_trace(go.Scatter(x=p_sub.index, y=p_sub['close'], name="Price ($)", line=dict(color='#00CC96', width=2)))
    if not pe_df.empty:
        pe_sub = pe_df[pe_df.index >= cutoff]
        fig.add_trace(go.Scatter(x=pe_sub.index, y=pe_sub['pe_ratio'], name="P/E Ratio", line=dict(color='#636EFA', width=2, dash='dot'), yaxis="y2"))
    
    fig.update_layout(title=title, yaxis=dict(title="Price ($)"), yaxis2=dict(title="P/E Ratio", overlaying="y", side="right"), hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. MAIN APP
# ==========================================
st.title("🦅 Alpha Pro v13.0 (Regime-Based)")

with st.sidebar:
    st.header("Settings")
    
    # --- AUTOMATIC KEY LOADING (NO TEXT BOX) ---
    if "AV_KEY" in st.secrets:
        key = st.secrets["AV_KEY"]
    else:
        key = ""
        st.warning("⚠️ AV_KEY missing in Secrets")
    # ------------------------------------------
    
    st.subheader("🧠 Strategy Mode")
    strategy = st.radio("Market Phase", ["Balanced (Default)", "Aggressive Growth", "Defensive Value"])
    
    # Define Weights based on selection
    if "Growth" in strategy:
        active_weights = {'growth': 35, 'momentum': 30, 'margins': 15, 'roe': 10, 'value': 10}
        st.caption("🚀 Focus: High Growth, Uptrends.")
    elif "Value" in strategy:
        active_weights = {'growth': 10, 'momentum': 10, 'margins': 25, 'roe': 25, 'value': 30}
        st.caption("🛡️ Focus: Cash Flow, Cheap Price.")
    else:
        active_weights = {'growth': 20, 'momentum': 20, 'margins': 20, 'roe': 20, 'value': 20}
        st.caption("⚖️ Focus: All-Weather Blend.")

    # Init watchlist state
    if 'watchlist_df' not in st.session_state:
        st.session_state.watchlist_df = get_watchlist_data()
    
    st.markdown("---")
    unique_tickers = st.session_state.watchlist_df['Ticker'].unique().tolist() if not st.session_state.watchlist_df.empty else []
    st.info(f"Tracking {len(unique_tickers)} Stocks")

t1, t2, t3 = st.tabs(["🔍 Analysis", "📈 Watchlist & Trends", "📊 Sector Flows"])

# --- TAB 1: ANALYSIS ---
with t1:
    c1, c2 = st.columns([3, 1])
    tick = c1.text_input("Analyze Ticker", "AAPL").upper()
    
    if tick and key:
        with st.spinner("Fetching Fundamentals..."):
            hist, ov, cf = get_alpha_data(tick, key)
            
        if not hist.empty and ov:
            score, log = calculate_dynamic_score(ov, cf, hist, active_weights)
            
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
            
            # Safe P/E Logic
            if pe_now is not None:
                k3.metric("Fwd P/E", f"{pe_now:.2f}")
                k4.metric("Sector Avg P/E", sec_avg, delta=f"{sec_avg - pe_now:.1f}")
            else:
                k3.metric("Fwd P/E", "N/A")
                k4.metric("Sector Avg P/E", sec_avg, delta=None)
            
            col_metrics, col_chart = st.columns([1, 2])
            
            with col_metrics:
                st.metric("Alpha Score v2.1", f"{score}/100", help="Score changes based on selected Strategy Mode")
                st.table(pd.DataFrame(list(log.items()), columns=["Metric", "Value / Weight"]))
                
                if st.button("⭐ Log to Cloud Watchlist"):
                    success = add_log_to_sheet(tick, score)
                    if success:
                        st.success(f"Logged {tick} (Score: {score}) to History!")
                        st.session_state.watchlist_df = get_watchlist_data()
            
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
        unique_list = df_wl['Ticker'].unique()
        for t in unique_list:
            history = df_wl[df_wl['Ticker'] == t].sort_values("Date")
            latest = history.iloc[-1]
            c_info, c_plot, c_del = st.columns([1, 2, 0.5])
            with c_info:
                st.subheader(t)
                st.metric("Latest Score", f"{latest['Score']}/100", f"Date: {latest['Date']}")
            with c_plot:
                if len(history) > 1:
                    fig = px.line(history, x='Date', y='Score', title=f"{t} Score Trend", markers=True)
                    fig.update_layout(height=150, margin=dict(l=0,r=0,t=30,b=0))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("Not enough history for trend graph.")
            with c_del:
                st.write("") 
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

    c_sel, c_tf = st.columns([3, 1])
    with c_sel:
        selectable = [k for k in SECTOR_ETFS.keys() if "SPY" not in k]
        selected_sectors = st.multiselect("Select Sectors to Compare", selectable, default=["Technology (XLK)", "Energy (XLE)", "Financials (XLF)"])
    with c_tf:
        days = tf_selector("rot")

    if st.button("Analyze Flows", type="primary"):
        if not selected_sectors:
            st.error("Please select at least one sector.")
        else:
            with st.spinner("Analyzing Market Flows..."):
                spy_hist, _, _ = get_alpha_data("SPY", key)
                if spy_hist.empty:
                    st.error("Could not fetch S&P 500 data. Check API Key.")
                    st.stop()
                
                cutoff = spy_hist.index[-1] - timedelta(days=days)
                spy_sub = spy_hist[spy_hist.index >= cutoff]['close']
                
                df_rel = pd.DataFrame()
                metrics = []
                progress = st.progress(0)
                
                for i, sec_name in enumerate(selected_sectors):
                    ticker = SECTOR_ETFS[sec_name]
                    if i > 0: time.sleep(1.5) 
                    hist, _, _ = get_alpha_data(ticker, key)
                    
                    if not hist.empty:
                        sec_sub = hist[hist.index >= cutoff]['close']
                        combined = pd.concat([sec_sub, spy_sub], axis=1).dropna()
                        combined.columns = ['Sector', 'SPY']
                        
                        rel_perf = (combined['Sector'] / combined['SPY'])
                        rel_perf = ((rel_perf / rel_perf.iloc[0]) - 1) * 100
                        df_rel[sec_name] = rel_perf

                        ma_200 = hist['close'].rolling(window=200).mean().iloc[-1]
                        curr_price = hist['close'].iloc[-1]
                        
                        if pd.notna(ma_200):
                            dist = ((curr_price / ma_200) - 1) * 100
                            status = "🔥 Overheated" if dist > 15 else "❄️ Oversold" if dist < -5 else "Normal"
                        else:
                            dist = 0
                            status = "N/A"

                        metrics.append({"Sector": sec_name, "Price": f"${curr_price:.2f}", "vs 200-Day Avg": f"{dist:+.1f}%", "Status": status})
                    
                    progress.progress((i + 1) / len(selected_sectors))
                
                st.divider()
                st.subheader(f"🔄 Money Flows (Relative to S&P 500)")
                if not df_rel.empty:
                    fig_rel = px.line(df_rel, title=f"Relative Performance vs SPY (Last {days} Days)")
                    fig_rel.update_layout(yaxis_title="Outperformance %")
                    fig_rel.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.5)
                    st.plotly_chart(fig_rel, use_container_width=True)

                st.subheader("⚠️ Valuation Check")
                if metrics:
                    st.table(pd.DataFrame(metrics).set_index("Sector"))
