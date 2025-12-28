import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION & DATABASE
# ==========================================
st.set_page_config(page_title="Alpha Pro v9.0", layout="wide")

DB_FILE = "market_data.json"
DEFAULT_KEY = "GLN6L0BRQIEN59OL" # Your Premium Key

# Static Sector Averages for Context (Approximate Current Market Values)
SECTOR_PE_BENCHMARKS = {
    "TECHNOLOGY": 28.5, "HEALTHCARE": 19.2, "FINANCIALS": 14.5,
    "ENERGY": 11.8, "COMMUNICATION SERVICES": 18.4, "CONSUMER CYCLICAL": 22.1,
    "CONSUMER DEFENSIVE": 20.5, "INDUSTRIALS": 21.0, "UTILITIES": 17.5,
    "REAL ESTATE": 35.0, "BASIC MATERIALS": 16.5
}

def load_db():
    if not os.path.exists(DB_FILE): return {}
    with open(DB_FILE, 'r') as f: return json.load(f)

def save_db(data):
    with open(DB_FILE, 'w') as f: json.dump(data, f)

def add_to_watchlist(ticker, score, price, sector):
    db = load_db()
    today = datetime.now().strftime("%Y-%m-%d")
    
    if ticker not in db:
        # Create new entry
        db[ticker] = {
            "sector": sector,
            "history": [{"date": today, "score": score, "price": price}]
        }
    else:
        # Update existing
        history = db[ticker]["history"]
        # Only append if today's date isn't already there
        if not history or history[-1]['date'] != today:
            history.append({"date": today, "score": score, "price": price})
            db[ticker]["history"] = history
            # Update sector just in case
            db[ticker]["sector"] = sector
            
    save_db(db)

# ==========================================
# 2. SMART REFRESHER (The "15-Day" Logic)
# ==========================================
def check_and_refresh_watchlist(api_key):
    """
    Runs on startup. Checks if any watchlist stock is >15 days old.
    If so, updates it automatically.
    """
    db = load_db()
    today = datetime.now()
    updates_made = False
    
    status_placeholder = st.empty()
    
    for ticker, data in db.items():
        history = data.get("history", [])
        if not history: continue
            
        last_date_str = history[-1]['date']
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
        
        # Check if 15 days have passed
        if (today - last_date).days >= 15:
            status_placeholder.info(f"🔄 Auto-Updating {ticker} (expired)...")
            
            # Fetch new data
            _, ov, cf = get_alpha_data(ticker, api_key)
            if ov:
                new_score, _ = calculate_dynamic_score(ov, cf)
                price = safe_float(ov.get('AnalystTargetPrice', 0)) # Fallback if history fetch skipped
                
                # Append new data point
                new_entry = {
                    "date": today.strftime("%Y-%m-%d"),
                    "score": new_score,
                    "price": price if price else 0
                }
                db[ticker]["history"].append(new_entry)
                updates_made = True
    
    if updates_made:
        save_db(db)
        status_placeholder.success("✅ Watchlist Updated Successfully!")
        status_placeholder.empty()

# ==========================================
# 3. DATA ENGINE
# ==========================================
@st.cache_data(ttl=300)
def get_alpha_data(ticker, api_key):
    base = "https://www.alphavantage.co/query"
    try:
        r_ov = requests.get(f"{base}?function=OVERVIEW&symbol={ticker}&apikey={api_key}").json()
    except: r_ov = {}

    try:
        r_cf = requests.get(f"{base}?function=CASH_FLOW&symbol={ticker}&apikey={api_key}").json()
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

def safe_float(val):
    try:
        if val is None or val == "None" or val == "-": return None
        return float(val)
    except: return None

# ==========================================
# 4. SCORING (Dynamic Gradient)
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
# 5. UI COMPONENTS
# ==========================================
def plot_price(df, title, tf_days=365):
    if df.empty: return
    # Filter by date
    cutoff = df.index[-1] - timedelta(days=tf_days)
    mask = df.index >= cutoff
    subset = df.loc[mask]
    
    fig = px.area(subset, x=subset.index, y='close', title=title)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Price ($)")
    st.plotly_chart(fig, use_container_width=True)

def tf_selector(key_suffix):
    # Returns number of days
    col_tf = st.columns(4)
    tf_map = {"1M": 30, "3M": 90, "1Y": 365, "5Y": 1825}
    # Default to 1Y
    choice = st.radio("Time Frame", list(tf_map.keys()), index=2, horizontal=True, key=f"tf_{key_suffix}")
    return tf_map[choice]

# ==========================================
# 6. MAIN APP
# ==========================================
st.title("🦅 Alpha Pro v9.0 (Portfolio Tracker)")

with st.sidebar:
    st.header("Settings")
    key = st.text_input("API Key", value=DEFAULT_KEY, type="password")
    if st.button("Force Refresh Watchlist"):
        check_and_refresh_watchlist(key)

# Check for updates on load
if 'updates_checked' not in st.session_state:
    check_and_refresh_watchlist(key)
    st.session_state['updates_checked'] = True

t1, t2, t3 = st.tabs(["🔍 Analysis", "📈 My Watchlist", "📊 Sectors"])

# --- TAB 1: INDIVIDUAL ANALYSIS ---
with t1:
    c1, c2 = st.columns([3, 1])
    tick = c1.text_input("Analyze Ticker", "AAPL").upper()
    
    if tick and key:
        with st.spinner("Analyzing..."):
            hist, ov, cf = get_alpha_data(tick, key)
        
        if not hist.empty and ov:
            score, log = calculate_dynamic_score(ov, cf)
            curr_price = hist['close'].iloc[-1]
            sector = ov.get('Sector', 'Unknown')
            
            # --- HEADER STATS ---
            mcap = safe_float(ov.get('MarketCapitalization', 0))
            div = safe_float(ov.get('DividendYield', 0))
            pe_curr = safe_float(ov.get('ForwardPE', 0))
            
            # Formatting Market Cap
            if mcap > 1e12: mcap_s = f"${mcap/1e12:.2f} T"
            elif mcap > 1e9: mcap_s = f"${mcap/1e9:.2f} B"
            else: mcap_s = f"${mcap/1e6:.2f} M"
            
            # Formatting Div
            div_s = f"{div*100:.2f}%" if div else "0.00%"
            
            # Sector Context
            sec_bench = SECTOR_PE_BENCHMARKS.get(sector.upper(), 20.0)
            
            st.markdown(f"## {tick} - {ov.get('Name', '')}")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Market Cap", mcap_s)
            k2.metric("Div Yield", div_s)
            k3.metric("Fwd P/E", f"{pe_curr:.2f}")
            k4.metric("Sector Avg P/E", f"{sec_bench}", delta=f"{sec_bench - pe_curr:.1f} vs Avg")
            
            # --- MAIN BODY ---
            col_s, col_g = st.columns([1, 2])
            
            with col_s:
                st.metric("Quality Score", f"{score}/100")
                st.table(pd.DataFrame(list(log.items()), columns=["Metric", "Value"]))
                
                # TRACKING BUTTON
                if st.button(f"⭐ Add {tick} to Watchlist"):
                    add_to_watchlist(tick, score, curr_price, sector)
                    st.success(f"Tracked!")
            
            with col_g:
                days = tf_selector("ind")
                plot_price(hist, f"{tick} Price History", days)
        else:
            st.error("Data Unavailable.")

# --- TAB 2: WATCHLIST ---
with t2:
    db = load_db()
    if db:
        st.header("Comparative Analysis")
        
        # 1. Multi-Select for Comparison
        all_tickers = list(db.keys())
        # Default to first 3
        defaults = all_tickers[:3] if len(all_tickers) >= 3 else all_tickers
        selected = st.multiselect("Compare Quality Scores (Max 10)", all_tickers, default=defaults)
        
        if len(selected) > 10: st.warning("Please select max 10.")
        
        if selected:
            # Build Dataframe for Plotly
            comp_data = []
            for t in selected:
                history = db[t].get("history", [])
                for h in history:
                    comp_data.append({"Ticker": t, "Date": h['date'], "Score": h['score']})
            
            df_comp = pd.DataFrame(comp_data)
            if not df_comp.empty:
                fig = px.line(df_comp, x="Date", y="Score", color="Ticker", markers=True, title="Quality Score Trends")
                fig.update_yaxes(range=[0, 100])
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.header("Portfolio Table")
        
        # Summary Table
        rows = []
        for t, d in db.items():
            h = d.get("history", [])
            if h:
                last = h[-1]
                rows.append({
                    "Ticker": t,
                    "Sector": d.get("sector", "-"),
                    "Last Update": last['date'],
                    "Score": last['score'],
                    "Price": f"${last['price']:.2f}"
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    else:
        st.info("Watchlist Empty. Go analyze some stocks!")

# --- TAB 3: SECTORS ---
with t3:
    st.header("Sector Dashboard")
    SECTORS = {"Tech": "XLK", "Healthcare": "XLV", "Financials": "XLF", "Energy": "XLE", "Real Estate": "XLRE"}
    
    c_sel, c_tf = st.columns([1, 1])
    with c_sel:
        sec = st.selectbox("Select Sector", list(SECTORS.keys()))
    with c_tf:
        days = tf_selector("sec")
        
    if st.button("Analyze Sector"):
        with st.spinner("Fetching..."):
            hist, _, _ = get_alpha_data(SECTORS[sec], key)
        
        if not hist.empty:
            plot_price(hist, f"{sec} ({SECTORS[sec]}) Performance", days)
