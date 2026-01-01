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
st.set_page_config(page_title="Alpha Pro v20.0", layout="wide")

# ETF Map for Sector Analysis
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

# --- STRATEGY WEIGHTS ---
WEIGHTS_BALANCED = {'growth': 20, 'momentum': 20, 'profitability': 20, 'roe': 20, 'value': 20}
WEIGHTS_SPECULATIVE = {'growth': 40, 'momentum': 60, 'profitability': 0, 'roe': 0, 'value': 0}
WEIGHTS_DEFENSIVE = {'value': 25, 'roe': 15, 'profitability': 25, 'momentum': 10, 'growth': 25}
WEIGHTS_AGGRESSIVE = {'growth': 35, 'momentum': 20, 'profitability': 20, 'value': 25, 'roe': 0}

# Add after WEIGHTS definitions in Configuration section
SECTOR_BENCHMARKS = {
    "Technology": {"margin_median": 20, "growth_median": 15, "pe_median": 25, "fcf_yield_median": 3.5, "roe_median": 18},
    "Healthcare": {"margin_median": 15, "growth_median": 8, "pe_median": 22, "fcf_yield_median": 4.0, "roe_median": 15},
    "Financials": {"margin_median": 25, "growth_median": 6, "pe_median": 12, "fcf_yield_median": 5.0, "roe_median": 12},
    "Energy": {"margin_median": 8, "growth_median": 5, "pe_median": 10, "fcf_yield_median": 6.0, "roe_median": 10},
    "Consumer Cyclical": {"margin_median": 8, "growth_median": 10, "pe_median": 18, "fcf_yield_median": 4.0, "roe_median": 14},
    "Consumer Defensive": {"margin_median": 6, "growth_median": 4, "pe_median": 20, "fcf_yield_median": 4.5, "roe_median": 18},
    "Industrials": {"margin_median": 10, "growth_median": 7, "pe_median": 18, "fcf_yield_median": 4.0, "roe_median": 13},
    "Utilities": {"margin_median": 12, "growth_median": 2, "pe_median": 16, "fcf_yield_median": 5.5, "roe_median": 9},
    "Real Estate": {"margin_median": 25, "growth_median": 3, "pe_median": 30, "fcf_yield_median": 4.0, "roe_median": 8},
    "Communication Services": {"margin_median": 18, "growth_median": 8, "pe_median": 20, "fcf_yield_median": 4.0, "roe_median": 15},
    "Materials": {"margin_median": 10, "growth_median": 5, "pe_median": 15, "fcf_yield_median": 5.0, "roe_median": 12},
    "Default": {"margin_median": 12, "growth_median": 8, "pe_median": 18, "fcf_yield_median": 4.0, "roe_median": 14}
}

def get_sector_context(overview):
    """Extract sector and return benchmarks"""
    sector = overview.get('Sector', 'Default')
    # Normalize sector names
    sector_map = {
        "TECHNOLOGY": "Technology",
        "HEALTH CARE": "Healthcare", 
        "FINANCIAL SERVICES": "Financials",
        "ENERGY": "Energy",
        "CONSUMER CYCLICAL": "Consumer Cyclical",
        "CONSUMER DEFENSIVE": "Consumer Defensive",
        "INDUSTRIALS": "Industrials",
        "UTILITIES": "Utilities",
        "REAL ESTATE": "Real Estate",
        "COMMUNICATION SERVICES": "Communication Services",
        "BASIC MATERIALS": "Materials"
    }
    normalized = sector_map.get(sector.upper(), "Default")
    return normalized, SECTOR_BENCHMARKS.get(normalized, SECTOR_BENCHMARKS["Default"])
    
# ==========================================
# 2. GOOGLE SHEETS DATABASE
# ==========================================
conn = st.connection("gsheets", type=GSheetsConnection)

def get_watchlist_data():
    try:
        df = conn.read(worksheet="Sheet1", ttl=0)
        if df.empty or 'Ticker' not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=['Ticker'])
        return df
    except Exception:
        return pd.DataFrame()

def add_log_to_sheet(ticker, curr_price, raw_metrics, scores_dict):
    try:
        existing_df = get_watchlist_data()
        log_date = (datetime.now() - timedelta(hours=6)).strftime("%Y-%m-%d")

        new_data = {
            "Ticker": ticker,
            "Date": log_date,
            "Price": curr_price,
            "Rev Growth": raw_metrics.get('Rev Growth'),
            "Profit Margin": raw_metrics.get('Profit Margin'),
            "FCF Yield": raw_metrics.get('FCF Yield'),
            "ROE": raw_metrics.get('ROE'),
            "PEG": raw_metrics.get('PEG'),
            "Mom Position %": raw_metrics.get('Mom Position'),
            "Mom Slope %": raw_metrics.get('Mom Slope'),
            "RVOL": raw_metrics.get('RVOL'),
            "Score (Balanced)": scores_dict.get('Balanced'),
            "Score (Aggressive)": scores_dict.get('Aggressive'),
            "Score (Defensive)": scores_dict.get('Defensive'),
            "Score (Speculative)": scores_dict.get('Speculative')
        }
        
        new_row = pd.DataFrame([new_data])
        
        if not existing_df.empty:
            mask = (existing_df['Ticker'] == ticker) & (existing_df['Date'] == log_date)
            if not existing_df[mask].empty:
                existing_df = existing_df[~mask]
        
        updated_df = pd.concat([existing_df, new_row], ignore_index=True)
        conn.update(worksheet="Sheet1", data=updated_df)
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False

def remove_ticker_from_sheet(ticker):
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
    try: r_bs = requests.get(f"{base}?function=BALANCE_SHEET&symbol={ticker}&apikey={api_key}").json()
    except: r_bs = {}
    try:
        r_price = requests.get(f"{base}?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={api_key}").json()
        ts = r_price.get('Time Series (Daily)', {})
        if ts:
            df = pd.DataFrame.from_dict(ts, orient='index')
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            if '5. adjusted close' in df.columns: df = df.rename(columns={'5. adjusted close': 'close'})
            else: df = df.rename(columns={'4. close': 'close'})
            if '6. volume' in df.columns: df = df.rename(columns={'6. volume': 'volume'})
            elif '5. volume' in df.columns: df = df.rename(columns={'5. volume': 'volume'})
        else: df = pd.DataFrame()
    except: df = pd.DataFrame()
    return df, r_ov, r_cf, r_bs

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
# 4. SCORING ENGINE (V20.0 - ULTIMATE)
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

# --- NEW HELPER: ADVANCED INCOME METRICS ---
def get_advanced_income_metrics(cash_flow_data, mcap_int):
    metrics = {"payout_ratio": None, "shareholder_yield": None}
    try:
        reports = cash_flow_data.get('annualReports', [])
        if len(reports) >= 1:
            latest = reports[0]
            # 1. Shareholder Yield (Divs + Buybacks)
            divs_paid = safe_float(latest.get('dividendPayout')) or 0
            buybacks = abs(safe_float(latest.get('paymentsForRepurchaseOfCommonStock')) or 0)
            if mcap_int and mcap_int > 0:
                metrics["shareholder_yield"] = ((divs_paid + buybacks) / mcap_int) * 100
            # 2. Payout Ratio (Divs / Net Income)
            net_income = safe_float(latest.get('netIncome'))
            if net_income and net_income > 0:
                metrics["payout_ratio"] = (divs_paid / net_income) * 100
    except: pass
    return metrics

# Add after WEIGHTS definitions in Configuration section
SECTOR_BENCHMARKS = {
    "Technology": {"margin_median": 20, "growth_median": 15, "pe_median": 25, "fcf_yield_median": 3.5, "roe_median": 18},
    "Healthcare": {"margin_median": 15, "growth_median": 8, "pe_median": 22, "fcf_yield_median": 4.0, "roe_median": 15},
    "Financials": {"margin_median": 25, "growth_median": 6, "pe_median": 12, "fcf_yield_median": 5.0, "roe_median": 12},
    "Energy": {"margin_median": 8, "growth_median": 5, "pe_median": 10, "fcf_yield_median": 6.0, "roe_median": 10},
    "Consumer Cyclical": {"margin_median": 8, "growth_median": 10, "pe_median": 18, "fcf_yield_median": 4.0, "roe_median": 14},
    "Consumer Defensive": {"margin_median": 6, "growth_median": 4, "pe_median": 20, "fcf_yield_median": 4.5, "roe_median": 18},
    "Industrials": {"margin_median": 10, "growth_median": 7, "pe_median": 18, "fcf_yield_median": 4.0, "roe_median": 13},
    "Utilities": {"margin_median": 12, "growth_median": 2, "pe_median": 16, "fcf_yield_median": 5.5, "roe_median": 9},
    "Real Estate": {"margin_median": 25, "growth_median": 3, "pe_median": 30, "fcf_yield_median": 4.0, "roe_median": 8},
    "Communication Services": {"margin_median": 18, "growth_median": 8, "pe_median": 20, "fcf_yield_median": 4.0, "roe_median": 15},
    "Materials": {"margin_median": 10, "growth_median": 5, "pe_median": 15, "fcf_yield_median": 5.0, "roe_median": 12},
    "Default": {"margin_median": 12, "growth_median": 8, "pe_median": 18, "fcf_yield_median": 4.0, "roe_median": 14}
}

def get_sector_context(overview):
    """Extract sector and return benchmarks"""
    sector = overview.get('Sector', 'Default')
    # Normalize sector names
    sector_map = {
        "TECHNOLOGY": "Technology",
        "HEALTH CARE": "Healthcare", 
        "FINANCIAL SERVICES": "Financials",
        "ENERGY": "Energy",
        "CONSUMER CYCLICAL": "Consumer Cyclical",
        "CONSUMER DEFENSIVE": "Consumer Defensive",
        "INDUSTRIALS": "Industrials",
        "UTILITIES": "Utilities",
        "REAL ESTATE": "Real Estate",
        "COMMUNICATION SERVICES": "Communication Services",
        "BASIC MATERIALS": "Materials"
    }
    normalized = sector_map.get(sector.upper(), "Default")
    return normalized, SECTOR_BENCHMARKS.get(normalized, SECTOR_BENCHMARKS["Default"])

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
        if not pe_sub.empty:
            fig.add_trace(go.Scatter(x=pe_sub.index, y=pe_sub['pe_ratio'], name="P/E Ratio", line=dict(color='#636EFA', width=2, dash='dot'), yaxis="y2"))
    fig.update_layout(title=title, yaxis=dict(title="Price ($)"), yaxis2=dict(title="P/E Ratio", overlaying="y", side="right"), hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 6. MAIN APP UI
# ==========================================
st.title("🦅 Alpha Pro v20.0 (Ultimate)")

with st.sidebar:
    st.header("Settings")
    if "AV_KEY" in st.secrets: key = st.secrets["AV_KEY"]
    else: 
        key = ""
        st.warning("⚠️ AV_KEY missing")
    
    is_premium = st.checkbox("🔑 I have a Premium API Key")
    
    st.subheader("🧠 Strategy Mode")
    strategy = st.radio("Market Phase", ["Balanced", "Aggressive Growth", "Defensive / Cyclical", "Speculative / Hype"])
    
    is_speculative = False
    mode_name = "Balanced"
    
    if "Aggressive" in strategy:
        active_weights = WEIGHTS_AGGRESSIVE
        mode_name = "Aggressive"
    elif "Defensive" in strategy:
        active_weights = WEIGHTS_DEFENSIVE
        mode_name = "Defensive"
    elif "Speculative" in strategy:
        active_weights = WEIGHTS_SPECULATIVE
        mode_name = "Speculative"
        is_speculative = True
    else:
        active_weights = WEIGHTS_BALANCED
        mode_name = "Balanced"

    # Initialize Watchlist State
    if 'watchlist_df' not in st.session_state:
        st.session_state.watchlist_df = get_watchlist_data()
    
    st.markdown("---")
    
    # --- BATCH ADD FEATURE ---
    with st.expander("➕ Batch Add Stocks"):
        batch_input = st.text_area("Enter tickers (comma separated)", placeholder="AAPL, MSFT, GOOGL")
        if st.button("Process Batch"):
            if not key:
                st.error("Need API Key")
            elif batch_input:
                tickers = [t.strip().upper() for t in batch_input.split(",") if t.strip()]
                progress_bar = st.progress(0)
                status_txt = st.empty()
                
                for i, t in enumerate(tickers):
                    status_txt.text(f"Analyzing {t} ({i+1}/{len(tickers)})...")
                    
                    # 1. Fetch Data
                    hist, ov, cf, bs = get_alpha_data(t, key)
                    pe_hist = get_historical_pe(t, key, hist)
                    
                    if not hist.empty and ov:
                        # 2. Calculate All Scores
                        s_bal, _, raw_metrics, _ = calculate_dynamic_score(ov, cf, bs, hist, WEIGHTS_BALANCED, use_50ma=False, mode="Balanced", historical_pe_df=pe_hist)
                        s_agg, _, _, _ = calculate_dynamic_score(ov, cf, bs, hist, WEIGHTS_AGGRESSIVE, use_50ma=False, mode="Aggressive", historical_pe_df=pe_hist)
                        s_def, _, _, _ = calculate_dynamic_score(ov, cf, bs, hist, WEIGHTS_DEFENSIVE, use_50ma=False, mode="Defensive", historical_pe_df=pe_hist)
                        s_spec, _, _, _ = calculate_dynamic_score(ov, cf, bs, hist, WEIGHTS_SPECULATIVE, use_50ma=True, mode="Speculative")
                        
                        scores_db = {'Balanced': s_bal, 'Aggressive': s_agg, 'Defensive': s_def, 'Speculative': s_spec}
                        curr_price = hist['close'].iloc[-1]
                        
                        # 3. Log to Sheet
                        add_log_to_sheet(t, curr_price, raw_metrics, scores_db)
                    
                    # 4. Rate Limit (4.5s delay for Premium)
                    time.sleep(4.5) 
                    progress_bar.progress((i + 1) / len(tickers))
                
                st.success("Batch Complete!")
                st.session_state.watchlist_df = get_watchlist_data()
                st.rerun()

    if not st.session_state.watchlist_df.empty and 'Ticker' in st.session_state.watchlist_df.columns:
        unique_tickers = st.session_state.watchlist_df['Ticker'].unique().tolist()
    else: unique_tickers = []
    st.info(f"Tracking {len(unique_tickers)} Stocks")

t1, t2, t3 = st.tabs(["🔍 Analysis", "📈 Watchlist & Trends", "📊 Sector Flows"])

with t1:
    c1, c2 = st.columns([3, 1])
    tick = c1.text_input("Analyze Ticker", "AAPL").upper()
    
    if tick and key:
        with st.spinner("Fetching Data..."):
            hist, ov, cf, bs = get_alpha_data(tick, key)
            pe_hist = get_historical_pe(tick, key, hist)
            
        if not hist.empty and ov:
            score, log, raw_metrics, base_scores = calculate_dynamic_score(
                ov, cf, bs, hist, active_weights, 
                use_50ma=is_speculative, 
                mode=mode_name,
                historical_pe_df=pe_hist
            )
            
            # DB Prep
            s_bal, _, _, _ = calculate_dynamic_score(ov, cf, bs, hist, WEIGHTS_BALANCED, use_50ma=False, mode="Balanced", historical_pe_df=pe_hist)
            s_agg, _, _, _ = calculate_dynamic_score(ov, cf, bs, hist, WEIGHTS_AGGRESSIVE, use_50ma=False, mode="Aggressive", historical_pe_df=pe_hist)
            s_def, _, _, _ = calculate_dynamic_score(ov, cf, bs, hist, WEIGHTS_DEFENSIVE, use_50ma=False, mode="Defensive", historical_pe_df=pe_hist)
            s_spec, _, _, _ = calculate_dynamic_score(ov, cf, bs, hist, WEIGHTS_SPECULATIVE, use_50ma=True, mode="Speculative")
            scores_db = {'Balanced': s_bal, 'Aggressive': s_agg, 'Defensive': s_def, 'Speculative': s_spec}
            
            curr_price = hist['close'].iloc[-1]
            day_delta = curr_price - hist['close'].iloc[-2] if len(hist)>1 else 0
            
            pe_now = safe_float(ov.get('PERatio', 0))
            short_float = safe_float(ov.get('ShortPercentFloat'))
            
            st.markdown(f"## {tick} - {ov.get('Name')}")
            k0, k1, k2, k3, k4 = st.columns(5)
            k0.metric("Price", f"${curr_price:.2f}", f"{day_delta:+.2f}")
            k1.metric("Market Cap", f"${safe_float(ov.get('MarketCapitalization',0))/1e9:.1f} B")
            k2.metric("Short Int", f"{short_float*100:.2f}%" if short_float else "N/A")
            k3.metric("P/E (TTM)", f"{pe_now:.2f}" if pe_now else "N/A")
            fcf_val = raw_metrics.get('FCF Yield')
            k4.metric("FCF Yield", f"{fcf_val:.2f}%" if fcf_val else "N/A")
            
            col_metrics, col_chart = st.columns([1, 2])
            with col_metrics:
                st.metric(f"Score ({mode_name})", f"{score}/100")
                df_log = pd.DataFrame(list(log.items()), columns=["Metric", "Details"])
                st.dataframe(df_log, hide_index=True, use_container_width=True)
                if st.button("⭐ Log to Cloud Watchlist"):
                    add_log_to_sheet(tick, curr_price, raw_metrics, scores_db)
                    st.success(f"Logged {tick}!")
                    st.session_state.watchlist_df = get_watchlist_data()
            
            with col_chart:
                days = tf_selector("ind")
                plot_dual_axis(hist, pe_hist, f"{tick}: Price vs Valuation (P/E)", days)
        else:
            st.error("Data Unavailable.")

with t2:
    st.header("Watchlist Trends")
    df_wl = st.session_state.watchlist_df
    
    # --- TOP PICKS DASHBOARD (ELITE MOMENTUM) ---
    if not df_wl.empty and 'Ticker' in df_wl.columns:
        st.markdown("### 🏆 Daily Top Picks (Elite > 75)")
        
        # 1. Prepare Data
        latest_entries = []
        for t in df_wl['Ticker'].unique():
            row = df_wl[df_wl['Ticker'] == t].sort_values("Date").iloc[-1]
            latest_entries.append(row)
        df_latest = pd.DataFrame(latest_entries)
        
        # 2. Helper to filter & sort
        def get_top_picks(df, score_col, min_score=75):
            # Filter for elite scores first
            elite = df[df[score_col] >= min_score]
            # If no elite stocks, fall back to top 2 raw scores
            if elite.empty:
                candidates = df
            else:
                candidates = elite
            
            # Sort by Score (Primary) and Momentum Slope (Secondary)
            sorted_df = candidates.sort_values(by=[score_col, 'Mom Slope %'], ascending=[False, False])
            return sorted_df.head(2)

        # 3. Get Leaders
        top_bal = get_top_picks(df_latest, 'Score (Balanced)')
        top_agg = get_top_picks(df_latest, 'Score (Aggressive)')
        top_def = get_top_picks(df_latest, 'Score (Defensive)')
        
        # 4. Display Cards
        c1, c2, c3 = st.columns(3)
        
        def display_card(container, title, rows, score_col, color_func):
            container.markdown(f"**{title}**")
            if rows.empty:
                container.caption("No data.")
                return

            for _, row in rows.iterrows():
                score = int(row[score_col])
                slope = float(row.get('Mom Slope %', 0))
                rvol = float(row.get('RVOL', 0))
                
                # Dynamic Badges
                badges = []
                if score >= 75: badges.append("⭐ Elite")
                if rvol > 1.5: badges.append("🔥 High Vol")
                elif rvol > 1.2: badges.append("⚡ Vol")
                
                badge_str = " ".join(badges) if badges else ""
                
                with container.container(border=True):
                    st.subheader(f"{row['Ticker']} {badge_str}")
                    st.metric("Score", f"{score}/100", f"Slope: {slope:+.1f}%")
                    st.caption(f"RVOL: {rvol:.2f}x Normal")
                    st.progress(score)

        with c1:
            display_card(st, "🏆 Balanced (Best Overall)", top_bal, 'Score (Balanced)', lambda x: "blue")
        
        with c2:
            display_card(st, "🚀 Aggressive (Growth+Mom)", top_agg, 'Score (Aggressive)', lambda x: "red")
                
        with c3:
            display_card(st, "🛡️ Defensive (Value+Mom)", top_def, 'Score (Defensive)', lambda x: "green")
        
        st.divider()
    
    # --- TURNAROUND SCANNER ---
    st.markdown("### 🦅 Turnaround Scanner")
    st.caption("Looking for: High Quality (Score > 60) + Broken Trend (Slope < -5%) + Cheap (PEG < 2 or High FCF)")
    
    if not df_wl.empty and 'Ticker' in df_wl.columns:
        unique_list = df_wl['Ticker'].unique()
        candidates = []
        for t in unique_list:
            # Get latest row for this ticker
            row = df_wl[df_wl['Ticker'] == t].sort_values("Date").iloc[-1]
            
            # Check 1: Quality
            score_ok = False
            if 'Score (Balanced)' in row and row['Score (Balanced)'] >= 60: score_ok = True
            
            # Check 2: Trend
            trend_broken = False
            slope = safe_float(row.get('Mom Slope %'))
            if slope and slope < -5: trend_broken = True
                
            # Check 3: Value
            value_ok = False
            peg = safe_float(row.get('PEG'))
            fcf = safe_float(row.get('FCF Yield'))
            if (peg and peg < 2.0) or (fcf and fcf > 4.0): value_ok = True
            
            if score_ok and trend_broken and value_ok: candidates.append(row)
        
        if candidates:
            st.success(f"Found {len(candidates)} Turnaround Candidates!")
            cols = st.columns(3)
            for i, row in enumerate(candidates):
                with cols[i % 3]:
                    with st.container(border=True):
                        st.subheader(f"{row['Ticker']}")
                        st.metric("Balanced Score", f"{int(row['Score (Balanced)'])}/100")
                        st.metric("Trend Slope", f"{float(row['Mom Slope %']):.1f}%", delta_color="inverse")
                        val_metric = f"PEG: {row['PEG']}" if row['PEG'] else f"FCF: {row['FCF Yield']}%"
                        st.caption(f"Valuation: {val_metric}")
        else:
            st.info("No turnaround candidates found.")
    else:
        st.info("Watchlist empty.")
    
    st.divider()

    # --- MAIN WATCHLIST DISPLAY ---
    if not df_wl.empty and 'Ticker' in df_wl.columns:
        unique_list = df_wl['Ticker'].unique()
        for t in unique_list:
            history = df_wl[df_wl['Ticker'] == t].sort_values("Date")
            latest = history.iloc[-1]
            st.subheader(f"{t} Analysis")
            
            # Plot Scores Over Time
            score_cols = [c for c in history.columns if 'Score (' in c]
            if score_cols and len(history) > 1:
                fig = px.line(history, x='Date', y=score_cols, markers=True)
                fig.update_layout(height=300, yaxis_title="Score")
                st.plotly_chart(fig, use_container_width=True)
            
            c_metrics, c_actions = st.columns([4, 1])
            with c_metrics:
                cols = st.columns(4)
                if 'Score (Balanced)' in latest: cols[0].metric("Bal", int(latest['Score (Balanced)']))
                if 'Score (Aggressive)' in latest: cols[1].metric("Agg", int(latest['Score (Aggressive)']))
                if 'Score (Defensive)' in latest: cols[2].metric("Def", int(latest['Score (Defensive)']))
                if 'Score (Speculative)' in latest: cols[3].metric("Spec", int(latest['Score (Speculative)']))
                st.caption(f"Price: ${latest.get('Price', 'N/A')} | Logged: {latest['Date']}")
            
            with c_actions:
                # Individual Update Button (Fixed Logic)
                if st.button(f"🔄 Update", key=f"upd_{t}"):
                    with st.spinner(f"Fetching {t}..."):
                        hist, ov, cf, bs = get_alpha_data(t, key)
                        pe_hist = get_historical_pe(t, key, hist)
                        if not hist.empty and ov:
                            s_bal, _, raw_metrics, _ = calculate_dynamic_score(ov, cf, bs, hist, WEIGHTS_BALANCED, use_50ma=False, mode="Balanced", historical_pe_df=pe_hist)
                            s_agg, _, _, _ = calculate_dynamic_score(ov, cf, bs, hist, WEIGHTS_AGGRESSIVE, use_50ma=False, mode="Aggressive", historical_pe_df=pe_hist)
                            s_def, _, _, _ = calculate_dynamic_score(ov, cf, bs, hist, WEIGHTS_DEFENSIVE, use_50ma=False, mode="Defensive", historical_pe_df=pe_hist)
                            s_spec, _, _, _ = calculate_dynamic_score(ov, cf, bs, hist, WEIGHTS_SPECULATIVE, use_50ma=True, mode="Speculative")
                            scores_db = {'Balanced': s_bal, 'Aggressive': s_agg, 'Defensive': s_def, 'Speculative': s_spec}
                            curr_price = hist['close'].iloc[-1]
                            add_log_to_sheet(t, curr_price, raw_metrics, scores_db)
                            st.success(f"Updated {t}!")
                            st.session_state.watchlist_df = get_watchlist_data()
                            st.rerun()
                        else: st.error("API Error.")
                
                # Delete Button
                if st.button("🗑️ Delete", key=f"del_{t}"):
                    remove_ticker_from_sheet(t)
                    st.session_state.watchlist_df = get_watchlist_data()
                    st.rerun()
            st.divider()

with t3:
    st.header("Sector Flows")
    c_sel, c_tf = st.columns([3, 1])
    with c_sel:
        all_sectors = [k for k in SECTOR_ETFS.keys() if "SPY" not in k]
        selected_sectors = st.multiselect("Select Sectors", all_sectors, default=all_sectors)
    with c_tf: days = tf_selector("rot")
    if st.button("Analyze All Sectors", type="primary"):
        if not selected_sectors: st.error("Select sectors.")
        else:
            spy_hist, _, _, _ = get_alpha_data("SPY", key)
            if spy_hist.empty: st.stop()
            cutoff = spy_hist.index[-1] - timedelta(days=days)
            spy_sub = spy_hist[spy_hist.index >= cutoff]['close']
            df_rel = pd.DataFrame()
            metrics_list = []
            prog_bar = st.progress(0)
            status_text = st.empty()
            
            for i, sec_name in enumerate(selected_sectors):
                ticker = SECTOR_ETFS[sec_name]
                status_text.text(f"Fetching {ticker}...")
                hist, _, _, _ = get_alpha_data(ticker, key)
                if not hist.empty:
                    sec_sub = hist[hist.index >= cutoff]['close']
                    combined = pd.concat([sec_sub, spy_sub], axis=1).dropna()
                    combined.columns = ['Sector', 'SPY']
                    rel_series = (combined['Sector'] / combined['SPY'])
                    rel_perf = (rel_series / rel_series.iloc[0] - 1) * 100
                    df_rel[sec_name] = rel_perf
                    
                    ma_200 = hist['close'].rolling(200).mean().iloc[-1]
                    curr_price = hist['close'].iloc[-1]
                    dist_200 = ((curr_price/ma_200)-1)*100 if pd.notna(ma_200) else 0
                    if len(rel_series) > 20:
                        recent_trend = rel_series.iloc[-20:]
                        slope_proxy = (recent_trend.iloc[-1] - recent_trend.iloc[0]) / recent_trend.iloc[0] * 100
                    else: slope_proxy = 0
                    
                    trend_score = 0
                    if dist_200 > 0: trend_score += 40
                    if rel_perf > 0: trend_score += 30
                    if slope_proxy > 0: trend_score += 30
                    status_icon = "🟢" if trend_score >= 70 else "🟡" if trend_score >= 40 else "🔴"
                    
                    metrics_list.append({
                        "Sector": sec_name, "Price": f"${curr_price:.2f}",
                        "vs 200MA": f"{dist_200:+.1f}%", "Rel Perf": f"{rel_perf:+.1f}%",
                        "Score": f"{trend_score}/100", "Status": status_icon
                    })
                prog_bar.progress((i + 1) / len(selected_sectors))
            status_text.text("Done!")
            
            if metrics_list:
                df_metrics = pd.DataFrame(metrics_list).sort_values("Score", ascending=False).reset_index(drop=True)
                st.subheader("🏆 Sector Leaderboard")
                st.dataframe(df_metrics, use_container_width=True)
                st.subheader("📈 Relative Strength vs SPY")
                if not df_rel.empty:
                    fig = px.line(df_rel, title="Sector Relative Performance")
                    fig.add_hline(y=0, line_dash="dot", line_color="white")
                    st.plotly_chart(fig, use_container_width=True)
