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
st.set_page_config(page_title="Alpha Pro v19.3", layout="wide")

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
WEIGHTS_AGGRESSIVE = {'growth': 40, 'momentum': 40, 'profitability': 10, 'roe': 5, 'value': 5}
# Defensive 2.0: Value (35), Quality/Debt (30), Profit/Stability (20), Momentum (15)
WEIGHTS_DEFENSIVE = {'value': 35, 'roe': 30, 'profitability': 20, 'momentum': 15, 'growth': 0}
WEIGHTS_SPECULATIVE = {'growth': 40, 'momentum': 60, 'profitability': 0, 'roe': 0, 'value': 0}

# ==========================================
# 2. GOOGLE SHEETS DATABASE
# ==========================================
conn = st.connection("gsheets", type=GSheetsConnection)

def get_watchlist_data():
    try:
        df = conn.read(worksheet="Sheet1", ttl=0)
        df = df.dropna(subset=['Ticker'])
        if 'Ticker' not in df.columns: return pd.DataFrame()
        return df
    except: return pd.DataFrame()

def add_log_to_sheet(ticker, curr_price, raw_metrics, scores_dict):
    try:
        existing_df = get_watchlist_data()
        log_date = (datetime.now() - timedelta(hours=6)).strftime("%Y-%m-%d") # Timezone fix

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
        st.cache_data.clear()
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
    except Exception as e: st.error(f"Remove failed: {e}")

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
# 4. SCORING ENGINE (V19.3 - HYBRID VALUE)
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

def calculate_dynamic_score(overview, cash_flow, balance_sheet, price_df, weights, use_50ma=False, mode="Balanced", historical_pe_df=None):
    earned, possible = 0, 0
    log = {}
    raw_metrics = {}
    base_scores = {} 
    
    def process_metric(label, raw_val, weight_key, base_score):
        nonlocal earned, possible
        base_scores[weight_key] = base_score
        w = weights.get(weight_key, 0)
        
        if raw_val is not None:
            weighted_points = (base_score / 20) * w
            earned += weighted_points
            possible += w
            return f"{raw_val} ({weighted_points:.1f}/{w})"
        else:
            earned += 0
            possible += w
            return f"N/A (0.0/{w})"

    # --- 1. GROWTH ---
    rev_growth = safe_float(overview.get('QuarterlyRevenueGrowthYOY'))
    raw_metrics['Rev Growth'] = rev_growth * 100 if rev_growth else None
    base_pts = get_points(rev_growth * 100, 20, 0, 20, True) if rev_growth else 0
    log["Revenue Growth"] = process_metric("Rev Growth", f"{rev_growth*100:.1f}%" if rev_growth else None, 'growth', base_pts)

    # --- 2. PROFITABILITY ---
    if mode == "Defensive":
        eps = safe_float(overview.get('EPS'))
        stability_score = 20 if (eps and eps > 0) else 0
        
        fcf_yield = None
        fcf_score = 0
        try:
            reports = cash_flow.get('annualReports', [])
            if reports:
                latest = reports[0]
                ocf = safe_float(latest.get('operatingCashflow'))
                capex = safe_float(latest.get('capitalExpenditures'))
                mcap = safe_float(overview.get('MarketCapitalization'))
                if ocf and capex and mcap:
                    fcf = ocf - capex
                    fcf_yield = (fcf / mcap) * 100
                    fcf_score = get_points(fcf_yield, 5.0, 0.0, 20, True)
        except: pass
        
        combined_prof = (stability_score + fcf_score) / 2
        raw_metrics['FCF Yield'] = fcf_yield
        log["Profit Stability"] = process_metric("EPS+FCF", f"EPS: {eps} / FCF: {fcf_yield:.1f}%" if fcf_yield else "N/A", 'profitability', combined_prof)
    
    else:
        margin = safe_float(overview.get('ProfitMargin'))
        base_margin = get_points(margin * 100, 25, 5, 20, True) if margin else 0
        raw_metrics['Profit Margin'] = margin * 100 if margin else None
        log["Profitability"] = process_metric("Net Margin", f"{margin*100:.1f}%" if margin else None, 'profitability', base_margin)

    # --- 3. QUALITY / ROE ---
    if mode == "Defensive":
        de_ratio = None
        base_solvency = 0
        try:
            reports = balance_sheet.get('annualReports', [])
            if reports:
                latest = reports[0]
                equity = safe_float(latest.get('totalShareholderEquity'))
                # SMART DEBT CHECK (Financial Debt preferred)
                short_debt = safe_float(latest.get('shortTermDebt')) or 0
                long_debt = safe_float(latest.get('longTermDebt')) or 0
                total_financial_debt = short_debt + long_debt
                
                if total_financial_debt > 0 and equity and equity > 0:
                    de_ratio = total_financial_debt / equity
                else:
                    liab = safe_float(latest.get('totalLiabilities'))
                    if liab and equity and equity > 0:
                        de_ratio = liab / equity
                        
                if de_ratio is not None:
                    base_solvency = get_points(de_ratio, 0.5, 2.0, 20, False)
        except: pass
        
        raw_metrics['ROE'] = de_ratio 
        log["Solvency (Debt)"] = process_metric("Debt/Eq", f"{de_ratio:.2f}" if de_ratio else None, 'roe', base_solvency)
    else:
        roe = safe_float(overview.get('ReturnOnEquityTTM'))
        if roe and roe < 5: roe = roe * 100 
        raw_metrics['ROE'] = roe
        base_pts = get_points(roe, 25, 5, 20, True) if roe else 0
        log["ROE"] = process_metric("ROE", f"{roe:.1f}%" if roe else None, 'roe', base_pts)

    # --- 4. VALUE (HYBRID MODEL) ---
    val_label = "PEG"
    val_raw = None
    base_val_pts = 0

    if mode == "Defensive":
        val_label = "Hybrid Val"
        
        # A. RELATIVE SCORE (50%)
        rel_score = 0
        current_pe = safe_float(overview.get('PERatio'))
        avg_pe = None
        rel_str = "N/A"
        
        if historical_pe_df is not None and not historical_pe_df.empty:
            avg_pe = historical_pe_df['pe_ratio'].median() # Median robust to outliers
            
        if current_pe and avg_pe:
            pe_discount = ((avg_pe - current_pe) / avg_pe) * 100
            rel_score = get_points(pe_discount, 5.0, -50.0, 20, True)
            rel_str = f"Prem: {pe_discount:+.0f}% (Med: {avg_pe:.1f})"
        
        # B. ABSOLUTE SCORE (50%)
        abs_score = 0
        ev_ebitda = safe_float(overview.get('EVToEBITDA'))
        abs_str = "N/A"
        
        if ev_ebitda:
            # Scale: <8x is Cheap, >20x is Expensive
            abs_score = get_points(ev_ebitda, 8.0, 20.0, 20, False)
            abs_str = f"{ev_ebitda:.1f}x EBITDA"
            val_raw = ev_ebitda 
            
        # C. COMBINE
        if current_pe and ev_ebitda:
            base_val_pts = (rel_score + abs_score) / 2
            val_str = f"{rel_str} | {abs_str}"
        elif ev_ebitda:
            base_val_pts = abs_score
            val_str = f"{abs_str} (Abs Only)"
        else:
            base_val_pts = 0
            val_str = "Data Missing"
            
        raw_metrics['PEG'] = val_raw
        log["Valuation"] = process_metric(val_label, val_str, 'value', base_val_pts)

    else:
        # Standard Logic
        if mode == "Aggressive":
            val_label = "EV/Sales"
            val_raw = safe_float(overview.get('EVToRevenue'))
            base_val_pts = get_points(val_raw, 3.0, 12.0, 20, False) if val_raw else 0
        else:
            val_label = "PEG"
            val_raw = safe_float(overview.get('PEGRatio'))
            base_val_pts = get_points(val_raw, 1.0, 2.5, 20, False) if val_raw else 0
        
        raw_metrics['PEG'] = val_raw 
        log["Valuation"] = process_metric(val_label, f"{val_raw:.2f}" if val_raw else None, 'value', base_val_pts)

    # --- 5. MOMENTUM ---
    pct_diff, slope_pct, rvol = None, None, None
    base_pts = 0
    ma_window = 50 if use_50ma else 200
    slope_lookback = 22 if use_50ma else 63
    required_history = ma_window + slope_lookback + 5
    val_str = "N/A"
    
    if not price_df.empty and len(price_df) > required_history:
        curr_price = price_df['close'].iloc[-1]
        ma_now = price_df['close'].rolling(window=ma_window).mean().iloc[-1]
        pos_score = 0
        if pd.notna(ma_now):
            pct_diff = ((curr_price / ma_now) - 1) * 100
            if pct_diff < 0: pos_score = 0
            elif 0 <= pct_diff <= 25: pos_score = 10
            else: 
                penalty = (pct_diff - 25) * 0.5
                pos_score = max(5, 10 - penalty)
        
        ma_old = price_df['close'].rolling(window=ma_window).mean().iloc[-slope_lookback]
        slope_score = 0
        if pd.notna(ma_old) and ma_old > 0:
            slope_pct = ((ma_now - ma_old) / ma_old) * 100
            target_slope = 10 if use_50ma else 5
            slope_score = min(10, (slope_pct / target_slope) * 10) if slope_pct > 0 else 0
        
        try:
            vol_5 = price_df['volume'].iloc[-5:].mean()
            vol_20 = price_df['volume'].iloc[-20:].mean()
            rvol = vol_5 / vol_20 if vol_20 > 0 else 1.0
        except: rvol = 1.0

        base_pts = pos_score + slope_score
        rvol_msg = ""
        if rvol > 1.2 and slope_pct > 0: 
            base_pts = min(20, base_pts + 2) 
            rvol_msg = " + Vol Bonus"
        elif rvol < 0.6 and slope_pct > 0:
            base_pts = max(0, base_pts - 2)
            rvol_msg = " - Low Vol"

        trend_status = "Falling" if (slope_pct or 0) < 0 else "Flat" if (slope_pct or 0) < 1 else "Rising"
        val_str = f"Pos: {pct_diff:+.1f}% / Slope: {slope_pct:+.1f}% / RVOL: {rvol:.2f}{rvol_msg}"
    else: val_str = None

    raw_metrics['Mom Position'] = pct_diff
    raw_metrics['Mom Slope'] = slope_pct
    raw_metrics['RVOL'] = rvol
    
    ma_label = "50-Day" if use_50ma else "200-Day"
    log[f"Trend ({ma_label})"] = process_metric("Momentum", val_str, 'momentum', base_pts)

    # 6. DEFENSIVE PENALTY
    if mode == "Defensive":
        mcap = safe_float(overview.get('MarketCapitalization'))
        if mcap and mcap < 2e9:
            score = int((earned/possible)*100) if possible > 0 else 0
            score = int(score * 0.50)
            log["Stability Check"] = "Small Cap (Penalty: -50%)"
            return score, log, raw_metrics, base_scores

    score = int((earned/possible)*100) if possible > 0 else 0
    return score, log, raw_metrics, base_scores

def compute_weighted_score(base_scores, weights):
    earned = 0
    possible = 0
    for key, weight in weights.items():
        if key in base_scores:
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
        if not pe_sub.empty:
            fig.add_trace(go.Scatter(x=pe_sub.index, y=pe_sub['pe_ratio'], name="P/E Ratio", line=dict(color='#636EFA', width=2, dash='dot'), yaxis="y2"))
    fig.update_layout(title=title, yaxis=dict(title="Price ($)"), yaxis2=dict(title="P/E Ratio", overlaying="y", side="right"), hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. MAIN APP
# ==========================================
st.title("🦅 Alpha Pro v19.3 (Hybrid Valuation)")

with st.sidebar:
    st.header("Settings")
    if "AV_KEY" in st.secrets: key = st.secrets["AV_KEY"]
    else: 
        key = ""
        st.warning("⚠️ AV_KEY missing")
    
    is_premium = st.checkbox("🔑 I have a Premium API Key")
    if is_premium: st.success("Premium Mode Active")
    else: st.info("Free Mode: Delays Active")
    
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

    if 'watchlist_df' not in st.session_state:
        st.session_state.watchlist_df = get_watchlist_data()
    
    st.markdown("---")
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
            # Main Score
            score, log, raw_metrics, base_scores = calculate_dynamic_score(
                ov, cf, bs, hist, active_weights, 
                use_50ma=is_speculative, 
                mode=mode_name,
                historical_pe_df=pe_hist
            )
            
            # DB Prep
            _, _, _, base_margin_scores = calculate_dynamic_score(ov, cf, bs, hist, {}, use_50ma=False, mode="Aggressive")
            s_bal = compute_weighted_score(base_margin_scores, WEIGHTS_BALANCED)
            s_agg = compute_weighted_score(base_margin_scores, WEIGHTS_AGGRESSIVE)
            s_def, _, _, _ = calculate_dynamic_score(ov, cf, bs, hist, WEIGHTS_DEFENSIVE, use_50ma=False, mode="Defensive", historical_pe_df=pe_hist)
            s_spec, _, _, _ = calculate_dynamic_score(ov, cf, bs, hist, WEIGHTS_SPECULATIVE, use_50ma=True, mode="Speculative")
            scores_db = {'Balanced': s_bal, 'Aggressive': s_agg, 'Defensive': s_def, 'Speculative': s_spec}
            
            curr_price = hist['close'].iloc[-1]
            if len(hist) > 1: day_delta = curr_price - hist['close'].iloc[-2]
            else: day_delta = 0
            
            pe_now = safe_float(ov.get('PERatio', 0))
            short_float = safe_float(ov.get('ShortPercentFloat'))
            short_ratio = safe_float(ov.get('ShortRatio'))
            if not short_float:
                shares_short = safe_float(ov.get('SharesShort'))
                shares_out = safe_float(ov.get('SharesOutstanding'))
                if shares_short and shares_out and shares_out > 0:
                    short_float = shares_short / shares_out
            
            st.markdown(f"## {tick} - {ov.get('Name')}")
            k0, k1, k2, k3, k4 = st.columns(5)
            k0.metric("Price", f"${curr_price:.2f}", f"{day_delta:+.2f}")
            k1.metric("Market Cap", f"${safe_float(ov.get('MarketCapitalization',0))/1e9:.1f} B")
            if short_float: k2.metric("Short Int", f"{short_float*100:.2f}%")
            elif short_ratio: k2.metric("Short Ratio", f"{short_ratio:.1f} Days")
            else: k2.metric("Short Info", "N/A")
            if pe_now: k3.metric("P/E (TTM)", f"{pe_now:.2f}")
            else: k3.metric("P/E (TTM)", "N/A")
            fcf_val = raw_metrics.get('FCF Yield')
            k4.metric("FCF Yield", f"{fcf_val:.2f}%" if fcf_val else "N/A")
            
            col_metrics, col_chart = st.columns([1, 2])
            with col_metrics:
                st.metric(f"Score ({mode_name})", f"{score}/100")
                df_log = pd.DataFrame(list(log.items()), columns=["Metric", "Details"])
                st.dataframe(df_log, hide_index=True, use_container_width=True)
                if st.button("⭐ Log to Cloud Watchlist"):
                    success = add_log_to_sheet(tick, curr_price, raw_metrics, scores_db)
                    if success:
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
    if df_wl.empty or 'Ticker' not in df_wl.columns:
        st.info("Watchlist is empty.")
    else:
        unique_list = df_wl['Ticker'].unique()
        for t in unique_list:
            history = df_wl[df_wl['Ticker'] == t].sort_values("Date")
            latest = history.iloc[-1]
            st.subheader(f"{t} Analysis")
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
                if st.button(f"🔄 Update", key=f"upd_{t}"):
                    with st.spinner(f"Fetching {t}..."):
                        hist, ov, cf, bs = get_alpha_data(t, key)
                        pe_hist = get_historical_pe(t, key, hist)
                        if not hist.empty and ov:
                            _, _, raw_metrics, base_margin_scores = calculate_dynamic_score(ov, cf, bs, hist, {}, use_50ma=False, mode="Aggressive")
                            s_bal = compute_weighted_score(base_margin_scores, WEIGHTS_BALANCED)
                            s_agg = compute_weighted_score(base_margin_scores, WEIGHTS_AGGRESSIVE)
                            s_def, _, _, _ = calculate_dynamic_score(ov, cf, bs, hist, WEIGHTS_DEFENSIVE, use_50ma=False, mode="Defensive", historical_pe_df=pe_hist)
                            s_spec, _, _, _ = calculate_dynamic_score(ov, cf, bs, hist, WEIGHTS_SPECULATIVE, use_50ma=True, mode="Speculative")
                            scores_db = {'Balanced': s_bal, 'Aggressive': s_agg, 'Defensive': s_def, 'Speculative': s_spec}
                            curr_price = hist['close'].iloc[-1]
                            success = add_log_to_sheet(t, curr_price, raw_metrics, scores_db)
                            if success:
                                st.success(f"Updated {t}!")
                                st.session_state.watchlist_df = get_watchlist_data()
                                time.sleep(1)
                                st.rerun()
                        else: st.error("API Error.")
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
        if not selected_sectors:
            st.error("Select sectors.")
        else:
            if not is_premium:
                st.info(f"⏳ Free Mode: Scanning {len(selected_sectors)} sectors (~{len(selected_sectors) * 12}s). Check 'Premium' in sidebar to speed up.")
            
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
                
                if i > 0 and not is_premium: time.sleep(12) 
                
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
                    
                    if len(rel_series) > 126:
                        rel_6m_ago = rel_series.iloc[-126]
                        rel_now = rel_series.iloc[-1]
                        rel_6m_change = ((rel_now / rel_6m_ago) - 1) * 100
                        recent_trend = rel_series.iloc[-20:]
                        slope_proxy = (recent_trend.iloc[-1] - recent_trend.iloc[0]) / recent_trend.iloc[0] * 100
                    else:
                        rel_6m_change = 0
                        slope_proxy = 0
                    
                    trend_score = 0
                    if dist_200 > 0: trend_score += 40
                    if rel_6m_change > 0: trend_score += 30
                    if slope_proxy > 0: trend_score += 30
                    status_icon = "🟢 Bullish" if trend_score >= 70 else "🟡 Neutral" if trend_score >= 40 else "🔴 Bearish"
                    
                    metrics_list.append({
                        "Sector": sec_name,
                        "Price": f"${curr_price:.2f}",
                        "vs 200MA": f"{dist_200:+.1f}%",
                        "6M Rel Perf": f"{rel_6m_change:+.1f}%",
                        "1M Rel Slope": f"{slope_proxy:+.2f}%",
                        "Score": f"{trend_score}/100",
                        "Status": status_icon
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
