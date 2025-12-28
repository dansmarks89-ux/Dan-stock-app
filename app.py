import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Alpha Pro v10.0", layout="wide")

DB_FILE = "market_data.json"
DEFAULT_KEY = "GLN6L0BRQIEN59OL"

SECTOR_PE_BENCHMARKS = {
    "TECHNOLOGY": 28.5, "HEALTHCARE": 19.2, "FINANCIALS": 14.5,
    "ENERGY": 11.8, "COMMUNICATION SERVICES": 18.4, "CONSUMER CYCLICAL": 22.1,
    "CONSUMER DEFENSIVE": 20.5, "INDUSTRIALS": 21.0, "UTILITIES": 17.5,
    "REAL ESTATE": 35.0
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
        db[ticker] = {"sector": sector, "history": [{"date": today, "score": score, "price": price}]}
    else:
        hist = db[ticker]["history"]
        if not hist or hist[-1]['date'] != today:
            hist.append({"date": today, "score": score, "price": price})
            db[ticker]["history"] = hist
            db[ticker]["sector"] = sector
    save_db(db)

# ==========================================
# 2. DATA ENGINE
# ==========================================
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
    """
    Calculates Daily P/E Ratio History.
    Logic: Daily Price / Rolling 4-Quarter Sum of EPS.
    """
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
        
        # Calculate TTM EPS (Sum of last 4 quarters)
        # We need at least 4 quarters of data to start calculating
        eps_df['ttm_eps'] = eps_df['reportedEPS'].rolling(window=4).sum()
        
        # Merge Price and EPS
        # "merge_asof" matches each Daily Price to the most recent TTM EPS available
        merged = pd.merge_asof(price_df, eps_df['ttm_eps'], left_index=True, right_index=True, direction='backward')
        
        # Calculate PE
        merged['pe_ratio'] = merged['close'] / merged['ttm_eps']
        
        # Clean extremes (remove negative PE or crazy outliers for cleaner chart)
        merged = merged[(merged['pe_ratio'] > 0) & (merged['pe_ratio'] < 200)]
        
        return merged[['pe_ratio']]
        
    except: return pd.DataFrame()

def safe_float(val):
    try:
        if val is None or val == "None" or val == "-": return None
        return float(val)
    except: return None

# ==========================================
# 3. SCORING
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
# 4. UI HELPERS
# ==========================================
def tf_selector(key_suffix):
    c_tf = st.columns(4)
    tf_map = {"1M": 30, "3M": 90, "1Y": 365, "5Y": 1825}
    choice = st.radio("Range", list(tf_map.keys()), index=2, horizontal=True, key=f"tf_{key_suffix}")
    return tf_map[choice]

def plot_dual_axis(price_df, pe_df, title, days):
    # Filter Date Range
    cutoff = price_df.index[-1] - timedelta(days=days)
    p_sub = price_df[price_df.index >= cutoff]
    
    fig = go.Figure()
    
    # 1. Price Line
    fig.add_trace(go.Scatter(
        x=p_sub.index, y=p_sub['close'], 
        name="Price ($)", line=dict(color='#00CC96', width=2)
    ))
    
    # 2. PE Line (Secondary Axis) - Only if we have data
    if not pe_df.empty:
        pe_sub = pe_df[pe_df.index >= cutoff]
        fig.add_trace(go.Scatter(
            x=pe_sub.index, y=pe_sub['pe_ratio'], 
            name="P/E Ratio", line=dict(color='#636EFA', width=2, dash='dot'),
            yaxis="y2"
        ))
    
    # Layout
    fig.update_layout(
        title=title,
        yaxis=dict(title="Price ($)"),
        yaxis2=dict(title="P/E Ratio", overlaying="y", side="right"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. MAIN APP
# ==========================================
st.title("🦅 Alpha Pro v10.0 (Deep Dive)")

with st.sidebar:
    st.header("Settings")
    key = st.text_input("API Key", value=DEFAULT_KEY, type="password")

t1, t2, t3 = st.tabs(["🔍 Analysis", "📈 Watchlist", "📊 Sectors"])

# --- TAB 1 ---
with t1:
    c1, c2 = st.columns([3, 1])
    tick = c1.text_input("Analyze Ticker", "AAPL").upper()
    
    if tick and key:
        with st.spinner("Fetching Fundamentals & History..."):
            hist, ov, cf = get_alpha_data(tick, key)
            
        if not hist.empty and ov:
            # 1. Score
            score, log = calculate_dynamic_score(ov, cf)
            
            # 2. Historical PE Calculation
            with st.spinner("Calculating Historical P/E..."):
                pe_hist = get_historical_pe(tick, key, hist)
            
            # --- DISPLAY HEADER ---
            curr_price = hist['close'].iloc[-1]
            sec = ov.get('Sector', 'Unknown')
            pe_now = safe_float(ov.get('ForwardPE', 0))
            sec_avg = SECTOR_PE_BENCHMARKS.get(sec.upper(), 20.0)
            
            st.markdown(f"## {tick} - {ov.get('Name')}")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Market Cap", f"${safe_float(ov.get('MarketCapitalization',0))/1e9:.1f} B")
            k2.metric("Div Yield", f"{safe_float(ov.get('DividendYield',0))*100:.2f}%" if ov.get('DividendYield') else "0%")
            k3.metric("Fwd P/E", f"{pe_now:.2f}")
            k4.metric("Sector Avg P/E", sec_avg, delta=f"{sec_avg - pe_now:.1f}")
            
            # --- MAIN ---
            col_metrics, col_chart = st.columns([1, 2])
            
            with col_metrics:
                st.metric("Quality Score", f"{score}/100")
                st.table(pd.DataFrame(list(log.items()), columns=["Metric", "Value"]))
                if st.button("⭐ Add to Watchlist"):
                    add_to_watchlist(tick, score, curr_price, sec)
                    st.success("Added!")

            with col_chart:
                days = tf_selector("ind")
                plot_dual_axis(hist, pe_hist, f"{tick}: Price vs Valuation (P/E)", days)
                
                # Median PE Stat
                if not pe_hist.empty:
                    med_pe = pe_hist['pe_ratio'].tail(365*5).median()
                    st.caption(f"📈 5-Year Median P/E: **{med_pe:.1f}x** (Current: {pe_now:.1f}x)")
        else:
            st.error("Data Unavailable.")

# --- TAB 2 & 3 (Simplified for brevity, same logic as before) ---
with t2:
    db = load_db()
    if db:
        st.header("Comparative Portfolio")
        sel = st.multiselect("Compare (Max 10)", list(db.keys()), default=list(db.keys())[:3])
        if sel:
            data = []
            for t in sel:
                for h in db[t]['history']: data.append({'Ticker':t, 'Date':h['date'], 'Score':h['score']})
            st.plotly_chart(px.line(pd.DataFrame(data), x='Date', y='Score', color='Ticker', markers=True), use_container_width=True)
        
        # Portfolio Table
        rows = []
        for t, d in db.items():
            last = d['history'][-1]
            rows.append({"Ticker": t, "Sector": d.get('sector','-'), "Score": last['score'], "Price": f"${last['price']:.2f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else: st.info("Watchlist Empty")

with t3:
    st.header("Sectors")
    SECTORS = {"Tech": "XLK", "Healthcare": "XLV", "Financials": "XLF", "Energy": "XLE"}
    s = st.selectbox("Sector", list(SECTORS.keys()))
    days = tf_selector("sec")
    if st.button("Analyze"):
        h, _, _ = get_alpha_data(SECTORS[s], key)
        if not h.empty:
            st.line_chart(h.tail(days)['close'])
