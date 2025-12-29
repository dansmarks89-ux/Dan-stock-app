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
st.set_page_config(page_title="Alpha Pro v15.0", layout="wide")

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

# Define Weights Globally for multi-score calculation
WEIGHTS_BALANCED = {'growth': 20, 'momentum': 20, 'margins': 20, 'roe': 20, 'value': 20}
WEIGHTS_AGGRESSIVE = {'growth': 35, 'momentum': 30, 'margins': 15, 'roe': 10, 'value': 10}
WEIGHTS_DEFENSIVE = {'growth': 10, 'momentum': 10, 'margins': 25, 'roe': 25, 'value': 30}

# ==========================================
# 2. GOOGLE SHEETS DATABASE
# ==========================================
conn = st.connection("gsheets", type=GSheetsConnection)

def get_watchlist_data():
    """Reads the full history from Google Sheets."""
    try:
        # Read ALL columns to preserve data structure
        df = conn.read(worksheet="Sheet1", ttl=0)
        df = df.dropna(subset=['Ticker'])
        # Handle empty/new sheet case
        if 'Ticker' not in df.columns:
            return pd.DataFrame(columns=["Ticker", "Date", "Score"])
        return df
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Date", "Score"])

def add_log_to_sheet(ticker, active_score, raw_metrics, scores_dict):
    """
    Appends or Updates a stock entry in the sheet.
    Ensures 1 line per ticker.
    """
    try:
        existing_df = get_watchlist_data()
        
        # Construct the new row with all variables
        new_data = {
            "Ticker": ticker,
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Score": active_score, # The score based on current strategy
            "Rev Growth": raw_metrics.get('Rev Growth'),
            "Profit Margin": raw_metrics.get('Profit Margin'),
            "ROE": raw_metrics.get('ROE'),
            "PEG": raw_metrics.get('PEG'),
            "Mom Position %": raw_metrics.get('Mom Position'),
            "Mom Slope %": raw_metrics.get('Mom Slope'),
            "Score (Balanced)": scores_dict.get('Balanced'),
            "Score (Aggressive)": scores_dict.get('Aggressive'),
            "Score (Defensive)": scores_dict.get('Defensive')
        }
        
        new_row = pd.DataFrame([new_data])
        
        # Combine and Deduplicate (Keep Last) to ensure 1 line per stock
        if not existing_df.empty:
            # Drop old entry for this ticker if it exists
            existing_df = existing_df[existing_df['Ticker'] != ticker]
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
        else:
            updated_df = new_row
            
        conn.update(worksheet="Sheet1", data=updated_df)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False

def remove_ticker_from_sheet(ticker):
    """Removes a specific ticker."""
    try:
        df = get_watchlist_data()
        if not df.empty:
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

@st.cache_data(ttl=3600)
def get_alpha_data(ticker, api_key):
    base = "https://www.alphavantage.co/query"
    try: r_ov = requests.get(f"{base}?function=OVERVIEW&symbol={ticker}&apikey={api_key}").json()
    except: r_ov = {}

    try: r_cf = requests.get(f"{base}?function=CASH_FLOW&symbol={ticker}&apikey={api_key}").json()
    except: r_cf = {}
    
    try:
        r_price = requests.get(f"{base}?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={api_key}").json()
        ts = r_price.get('Time Series (Daily)', {})
        if ts:
            df = pd.DataFrame.from_dict(ts, orient='index')
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            if '5. adjusted close' in df.columns:
                df = df.rename(columns={'5. adjusted close': 'close'})
            else:
                df = df.rename(columns={'4. close': 'close'})
        else: df = pd.DataFrame()
    except: df = pd.DataFrame()
        
    return df, r_ov, r_cf

@st.cache_data(ttl=3600)
def get_historical_pe(ticker, api_key, price_df):
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

def calculate_dynamic_score(overview, cash_flow, price_df, weights):
    """
    Returns: Final Score (int), Log (dict), Raw Metrics (dict), Base Scores (dict)
    """
    earned, possible = 0, 0
    log = {}
    raw_metrics = {}
    base_scores = {} # Stores the 0-20 score for each category
    
    # Helper to clean up logic
    def process_metric(label, raw_val, weight_key, base_score):
        nonlocal earned, possible
        # Store for external use
        base_scores[weight_key] = base_score
        if raw_val is not None:
            # Strip string chars if needed for raw storage? No, keep logic simple.
            pass 
            
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
    raw_metrics['Rev Growth'] = rev_growth * 100 if rev_growth else None
    base_pts = get_points(rev_growth * 100, 20, 0, 20, True) if rev_growth else 0
    log["Revenue Growth"] = process_metric("Rev Growth", f"{rev_growth*100:.1f}%" if rev_growth else None, 'growth', base_pts)

    # 2. PROFITABILITY
    margin = safe_float(overview.get('ProfitMargin'))
    raw_metrics['Profit Margin'] = margin * 100 if margin else None
    base_pts = get_points(margin * 100, 25, 5, 20, True) if margin else 0
    log["Profit Margin"] = process_metric("Margin", f"{margin*100:.1f}%" if margin else None, 'margins', base_pts)

    # 3. QUALITY
    roe = safe_float(overview.get('ReturnOnEquityTTM'))
    if roe and roe < 5: roe = roe * 100 
    raw_metrics['ROE'] = roe
    base_pts = get_points(roe, 25, 5, 20, True) if roe else 0
    log["ROE"] = process_metric("ROE", f"{roe:.1f}%" if roe else None, 'roe', base_pts)

    # 4. VALUE
    peg = safe_float(overview.get('PEGRatio'))
    raw_metrics['PEG'] = peg
    base_pts = get_points(peg, 1.0, 2.5, 20) if peg else 0
    log["PEG Ratio"] = process_metric("PEG", f"{peg:.2f}" if peg else None, 'value', base_pts)

    # 5. MOMENTUM
    pct_diff = None
    slope_pct = None
    base_pts = 0
    
    if not price_df.empty and len(price_df) > 265:
        # A. Position
        curr_price = price_df['close'].iloc[-1]
        ma_200_now = price_df['close'].rolling(window=200).mean().iloc[-1]
        
        pos_score = 0
        if pd.notna(ma_200_now):
            pct_diff = ((curr_price / ma_200_now) - 1) * 100
            if pct_diff < 0: pos_score = 0
            elif 0 <= pct_diff <= 25: pos_score = 10
            else: 
                penalty = (pct_diff - 25) * 0.5
                pos_score = max(0, 10 - penalty)
        
        # B. Velocity
        ma_200_old = price_df['close'].rolling(window=200).mean().iloc[-63]
        slope_score = 0
        if pd.notna(ma_200_old) and ma_200_old > 0:
            slope_pct = ((ma_200_now - ma_200_old) / ma_200_old) * 100
            if slope_pct <= 0: slope_score = 0
            else:
                slope_score = min(10, (slope_pct / 5) * 10)
        
        base_pts = pos_score + slope_score
        
        # For Log Label
        trend_status = "Falling" if (slope_pct or 0) < 0 else "Flat" if (slope_pct or 0) < 1 else "Rising"
        val_str = f"Pos: {pct_diff:+.1f}% / Slope: {slope_pct:+.1f}% ({trend_status})"
    else:
        val_str = "Insufficient History"
        base_pts = 0

    raw_metrics['Mom Position'] = pct_diff
    raw_metrics['Mom Slope'] = slope_pct
    
    log["vs 200-Day MA"] = process_metric("Momentum", val_str, 'momentum', base_pts)

    score = int((earned/possible)*100) if possible > 0 else 0
    return score, log, raw_metrics, base_scores

# Helper to compute any score from base points
def compute_weighted_score(base_scores, weights):
    earned = 0
    possible = 0
    for key, weight in weights.items():
        if key in base_scores:
            # base score is 0-20. Scale to weight.
            pts = (base_scores[key] / 20) * weight
            earned += pts
            possible += weight
    return int((earned/possible)*100) if possible > 0 else 0

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
st.title("🦅 Alpha Pro v15.0 (Cloud Database)")

with st.sidebar:
    st.header("Settings")
    if "AV_KEY" in st.secrets:
        key = st.secrets["AV_KEY"]
    else:
        key = ""
        st.warning("⚠️ AV_KEY missing in Secrets")
    
    st.subheader("🧠 Strategy Mode")
    strategy = st.radio("Market Phase", ["Balanced (Default)", "Aggressive Growth", "Defensive Value"])
    
    if "Growth" in strategy:
        active_weights = WEIGHTS_AGGRESSIVE
        st.caption("🚀 Focus: High Growth, Uptrends.")
    elif "Value" in strategy:
        active_weights = WEIGHTS_DEFENSIVE
        st.caption("🛡️ Focus: Cash Flow, Cheap Price.")
    else:
        active_weights = WEIGHTS_BALANCED
        st.caption("⚖️ Focus: All-Weather Blend.")

    if 'watchlist_df' not in st.session_state:
        st.session_state.watchlist_df = get_watchlist_data()
    
    st.markdown("---")
    # Safe check for unique tickers
    if not st.session_state.watchlist_df.empty and 'Ticker' in st.session_state.watchlist_df.columns:
        unique_tickers = st.session_state.watchlist_df['Ticker'].unique().tolist()
    else:
        unique_tickers = []
        
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
            # Calculate metrics
            score, log, raw_metrics, base_scores = calculate_dynamic_score(ov, cf, hist, active_weights)
            
            # Calculate ALL scores for database
            scores_db = {
                'Balanced': compute_weighted_score(base_scores, WEIGHTS_BALANCED),
                'Aggressive': compute_weighted_score(base_scores, WEIGHTS_AGGRESSIVE),
                'Defensive': compute_weighted_score(base_scores, WEIGHTS_DEFENSIVE)
            }
            
            with st.spinner("Calculating Historical P/E..."):
                pe_hist = get_historical_pe(tick, key, hist)
            
            # Latest Price
            curr_price = hist['close'].iloc[-1]
            if len(hist) > 1:
                day_delta = curr_price - hist['close'].iloc[-2]
            else: day_delta = 0
            
            pe_now = safe_float(ov.get('ForwardPE', 0))
            sec = ov.get('Sector', 'Unknown')
            sec_avg = SECTOR_PE_BENCHMARKS.get(sec.upper(), 20.0)
            
            st.markdown(f"## {tick} - {ov.get('Name')}")
            k0, k1, k2, k3, k4 = st.columns(5)
            k0.metric("Price", f"${curr_price:.2f}", f"{day_delta:+.2f}")
            k1.metric("Market Cap", f"${safe_float(ov.get('MarketCapitalization',0))/1e9:.1f} B")
            k2.metric("Div Yield", f"{(safe_float(ov.get('DividendYield', 0)) or 0) * 100:.2f}%")
            if pe_now:
                k3.metric("Fwd P/E", f"{pe_now:.2f}")
                k4.metric("Sector Avg P/E", sec_avg, delta=f"{sec_avg - pe_now:.1f}")
            else:
                k3.metric("Fwd P/E", "N/A")
                k4.metric("Sector Avg P/E", sec_avg, delta=None)
            
            col_metrics, col_chart = st.columns([1, 2])
            with col_metrics:
                st.metric("Alpha Score v3.0", f"{score}/100", help="Score changes based on selected Strategy Mode")
                st.table(pd.DataFrame(list(log.items()), columns=["Metric", "Value / Weight"]))
                
                if st.button("⭐ Log to Cloud Watchlist"):
                    success = add_log_to_sheet(tick, score, raw_metrics, scores_db)
                    if success:
                        st.success(f"Logged {tick} to Cloud!")
                        st.session_state.watchlist_df = get_watchlist_data()
            
            with col_chart:
                days = tf_selector("ind")
                plot_dual_axis(hist, pe_hist, f"{tick}: Price vs Valuation (P/E)", days)
        else:
            st.error("Data Unavailable.")

# --- TAB 2: WATCHLIST TRENDS ---
with t2:
    st.header("My Watchlist Trends")
    df_wl = st.session_state.watchlist_df
    
    if df_wl.empty or 'Ticker' not in df_wl.columns:
        st.info("Watchlist is empty. Analyze a stock to add it.")
    else:
        unique_list = df_wl['Ticker'].unique()
        for t in unique_list:
            # Filter history
            history = df_wl[df_wl['Ticker'] == t].sort_values("Date")
            latest = history.iloc[-1]
            
            c_info, c_plot, c_del = st.columns([1, 2, 0.5])
            with c_info:
                st.subheader(t)
                # Show the score logged at that time
                st.metric("Logged Score", f"{latest['Score']}/100", f"Date: {latest['Date']}")
            with c_plot:
                if len(history) > 1:
                    fig = px.line(history, x='Date', y='Score', title=f"{t} Score Trend", markers=True)
                    fig.update_layout(height=150, margin=dict(l=0,r=0,t=30,b=0))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("Not enough history for trend.")
            with c_del:
                st.write("") 
                if st.button("🗑️", key=f"del_{t}"):
                    remove_ticker_from_sheet(t)
                    st.session_state.watchlist_df = get_watchlist_data()
                    st.rerun()
            st.divider()

# --- TAB 3: SECTORS ---
with t3:
    st.header("Sector Rotation & Money Flows")
    c_sel, c_tf = st.columns([3, 1])
    with c_sel:
        selectable = [k for k in SECTOR_ETFS.keys() if "SPY" not in k]
        selected_sectors = st.multiselect("Select Sectors", selectable, default=["Technology (XLK)", "Energy (XLE)", "Financials (XLF)"])
    with c_tf:
        days = tf_selector("rot")

    if st.button("Analyze Flows", type="primary"):
        if not selected_sectors:
            st.error("Please select at least one sector.")
        else:
            with st.spinner("Analyzing Market Flows..."):
                spy_hist, _, _ = get_alpha_data("SPY", key)
                if spy_hist.empty:
                    st.error("Check API Key.")
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
                        rel_perf = ((combined['Sector'] / combined['SPY']) / (combined['Sector'].iloc[0] / combined['SPY'].iloc[0]) - 1) * 100
                        df_rel[sec_name] = rel_perf
                        
                        ma_200 = hist['close'].rolling(200).mean().iloc[-1]
                        curr = hist['close'].iloc[-1]
                        dist = ((curr/ma_200)-1)*100 if pd.notna(ma_200) else 0
                        status = "🔥 Overheated" if dist > 15 else "❄️ Oversold" if dist < -5 else "Normal"
                        metrics.append({"Sector": sec_name, "Price": f"${curr:.2f}", "vs 200MA": f"{dist:+.1f}%", "Status": status})
                    progress.progress((i+1)/len(selected_sectors))
                
                if not df_rel.empty:
                    fig = px.line(df_rel, title=f"Relative Performance vs SPY")
                    fig.add_hline(y=0, line_dash="dot", line_color="white")
                    st.plotly_chart(fig, use_container_width=True)
                if metrics: st.table(pd.DataFrame(metrics).set_index("Sector"))
