import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION & API SETUP
# ==========================================
st.set_page_config(page_title="Market Sector & Stock Analyzer", layout="wide")

# SECURITY: Get API Key from Streamlit Secrets
try:
    API_KEY = st.secrets["FMP_KEY"]
except:
    st.error("API Key not found. Please set FMP_KEY in Streamlit Secrets.")
    st.stop()

BASE_URL = "https://financialmodelingprep.com/api/v3"

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
# 2. DATA FETCHING FUNCTIONS (UPDATED)
# ==========================================

@st.cache_data(ttl=3600*24)
def get_daily_price_history(ticker):
    # UPDATED: Using 'historical-chart/1day' instead of 'historical-price-full'
    url = f"{BASE_URL}/historical-chart/1day/{ticker}?apikey={API_KEY}"
    try:
        data = requests.get(url).json()
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            # Ensure we have date and close
            if 'date' in df.columns and 'close' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').set_index('date')
                return df['close'] # Use close instead of adjClose for this endpoint
        return None
    except: return None

@st.cache_data(ttl=3600*24)
def get_key_metrics(ticker):
    url = f"{BASE_URL}/key-metrics/{ticker}?period=quarter&limit=80&apikey={API_KEY}"
    try:
        data = requests.get(url).json()
        if isinstance(data, list):
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').set_index('date')
            return df
        return pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(ttl=3600*24)
def get_ratios(ticker):
    url = f"{BASE_URL}/ratios/{ticker}?period=quarter&limit=80&apikey={API_KEY}"
    try:
        data = requests.get(url).json()
        if isinstance(data, list):
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').set_index('date')
            return df
        return pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(ttl=3600*24)
def get_full_data_merged(ticker):
    prices = get_daily_price_history(ticker)
    if prices is None: return None
    ratios = get_ratios(ticker)
    metrics = get_key_metrics(ticker)
    df = pd.DataFrame(index=prices.index)
    df['price'] = prices
    if not ratios.empty:
        cols = ['priceEarningsRatio', 'pegRatio']
        ratios_subset = ratios[cols] if set(cols).issubset(ratios.columns) else pd.DataFrame()
        df = pd.merge_asof(df, ratios_subset, left_index=True, right_index=True, direction='backward')
    if not metrics.empty:
        cols_map = {'enterpriseValueOverEBITDA': 'ev_ebitda', 'roic': 'roic', 'freeCashFlowYield': 'fcf_yield'}
        available_cols = [c for c in cols_map.keys() if c in metrics.columns]
        metrics_subset = metrics[available_cols].rename(columns=cols_map)
        df = pd.merge_asof(df, metrics_subset, left_index=True, right_index=True, direction='backward')
    df['sma_50'] = df['price'].rolling(window=50).mean()
    df['sma_200'] = df['price'].rolling(window=200).mean()
    return df

# ==========================================
# 3. SCORING LOGIC (UNCHANGED)
# ==========================================

def calculate_score(row):
    score = 0
    pe = row.get('priceEarningsRatio', 35)
    if pd.isna(pe): pe = 35
    if pe < 20: score += 10
    elif 20 <= pe <= 30: score += 5
    
    ev = row.get('ev_ebitda', 25)
    if pd.isna(ev): ev = 25
    if ev < 12: score += 15
    elif 12 <= ev <= 20: score += 7.5
    
    peg = row.get('pegRatio', 3)
    if pd.isna(peg): peg = 3
    if peg < 1: score += 25
    elif 1 <= peg <= 1.5: score += 12.5
    
    roic = row.get('roic', 0)
    if pd.isna(roic): roic = 0
    if roic < 1.0: roic = roic * 100 
    if roic > 20: score += 30
    elif 10 <= roic <= 20: score += 15
    
    fcf = row.get('fcf_yield', 0)
    if pd.isna(fcf): fcf = 0
    if fcf < 1.0: fcf = fcf * 100 
    if fcf > 5: score += 20
    elif 3 <= fcf <= 5: score += 10
    
    return score

# ==========================================
# 4. APP UI (UNCHANGED)
# ==========================================

st.title("📈 Strategic Market & Stock Analyzer")
st.markdown("---")
page = st.sidebar.radio("Navigate", ["Overall Market Trends (Sectors)", "Individual Stock Analyzer", "Trade Tester"])

if page == "Overall Market Trends (Sectors)":
    st.header("Overall Market Trends")
    with st.expander("⚙️ Manage Sectors (Add/Remove Stocks)"):
        sector_names = list(st.session_state['sectors'].keys())
        selected_edit_sector = st.selectbox("Select Sector to Edit", sector_names)
        current_tickers = st.session_state['sectors'][selected_edit_sector]
        st.write(f"Current: {', '.join(current_tickers)}")
        new_ticker_str = st.text_input("Update Tickers (comma separated)", value=", ".join(current_tickers))
        if st.button("Update Sector"):
            st.session_state['sectors'][selected_edit_sector] = [x.strip().upper() for x in new_ticker_str.split(",")]
            st.success("Sector Updated!")
            st.rerun()
        new_sector_name = st.text_input("Add New Sector Name (e.g., Robotics)")
        if st.button("Add New Sector") and new_sector_name:
            st.session_state['sectors'][new_sector_name] = []
            st.success(f"Added {new_sector_name}")
            st.rerun()

    st.write("Fetching Market Data... (This may take 30-60s on first load)")
    spy_data = get_daily_price_history("SPY")
    
    st.subheader("Comparison: All Sectors")
    metric_view = st.selectbox("Select Metric to Compare", ["Relative Strength (Ratio vs SPY)", "Breadth (% > 50 SMA)"])
    comparison_fig = go.Figure()
    
    for sec_name, tickers in st.session_state['sectors'].items():
        if not tickers: continue
        etf_ticker = tickers[0] 
        if metric_view == "Relative Strength (Ratio vs SPY)":
            etf_price = get_daily_price_history(etf_ticker)
            if etf_price is not None and spy_data is not None:
                common_idx = etf_price.index.intersection(spy_data.index)
                ratio = etf_price.loc[common_idx] / spy_data.loc[common_idx]
                ratio_weekly = ratio.resample('W-FRI').last()
                comparison_fig.add_trace(go.Scatter(x=ratio_weekly.index, y=ratio_weekly.values, name=sec_name))
        
        elif metric_view == "Breadth (% > 50 SMA)":
            above_count = pd.Series(0, index=spy_data.index) 
            valid_stock_count = 0
            for t in tickers:
                p = get_daily_price_history(t)
                if p is not None:
                    sma50 = p.rolling(50).mean()
                    common_idx = p.index.intersection(above_count.index)
                    is_above = (p.loc[common_idx] > sma50.loc[common_idx]).astype(int)
                    temp_series = pd.Series(0, index=above_count.index)
                    temp_series.loc[common_idx] = is_above
                    above_count += temp_series
                    valid_stock_count += 1
            if valid_stock_count > 0:
                breadth = (above_count / valid_stock_count) * 100
                breadth_weekly = breadth.resample('W-FRI').last()
                comparison_fig.add_trace(go.Scatter(x=breadth_weekly.index, y=breadth_weekly.values, name=sec_name))

    st.plotly_chart(comparison_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Deep Dive: Single Sector")
    selected_sector = st.selectbox("Choose Sector to Inspect", list(st.session_state['sectors'].keys()))
    sector_tickers = st.session_state['sectors'][selected_sector]
    inspect_metric = st.radio("Select Metric for Deep Dive", ["Price History"])
    
    if inspect_metric == "Price History":
        fig_sec = go.Figure()
        for t in sector_tickers:
            p = get_daily_price_history(t)
            if p is not None:
                p_norm = (p / p.iloc[0] - 1) * 100
                fig_sec.add_trace(go.Scatter(x=p.index, y=p_norm, name=t))
        fig_sec.update_layout(title="Percentage Return Comparison (Normalized)", yaxis_title="% Return")
        st.plotly_chart(fig_sec, use_container_width=True)

elif page == "Individual Stock Analyzer":
    st.header("Individual Stock Analysis & Scoring")
    ticker_input = st.text_input("Enter Stock Ticker (e.g. NVDA)", "NVDA").upper()
    if ticker_input:
        df = get_full_data_merged(ticker_input)
        if df is not None and not df.empty:
            df_weekly = df.resample('W-FRI').last().ffill()
            df_weekly['Score'] = df_weekly.apply(calculate_score, axis=1)
            latest = df_weekly.iloc[-1]
            st.subheader(f"{ticker_input} - Current Score: {latest['Score']}/100")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Price", f"${latest['price']:.2f}")
            col2.metric("P/E Ratio", f"{latest.get('priceEarningsRatio', 0):.2f}")
            col3.metric("PEG Ratio", f"{latest.get('pegRatio', 0):.2f}")
            col4.metric("ROIC", f"{latest.get('roic', 0)*100:.2f}%")
            col5.metric("EV/EBITDA", f"{latest.get('ev_ebitda', 0):.2f}x")
            
            st.subheader("Historical Quality Score (0-100)")
            fig_score = px.line(df_weekly, x=df_weekly.index, y="Score", title="Weighted Score Over Time")
            fig_score.add_hrect(y0=80, y1=100, line_width=0, fillcolor="green", opacity=0.1)
            fig_score.add_hrect(y0=0, y1=50, line_width=0, fillcolor="red", opacity=0.1)
            st.plotly_chart(fig_score, use_container_width=True)
            
            st.subheader("Variable Trends")
            vars_to_plot = st.multiselect("Select Variables to Plot", 
                                          ['priceEarningsRatio', 'pegRatio', 'roic', 'ev_ebitda', 'fcf_yield'],
                                          default=['priceEarningsRatio', 'ev_ebitda'])
            fig_vars = go.Figure()
            for v in vars_to_plot:
                fig_vars.add_trace(go.Scatter(x=df_weekly.index, y=df_weekly[v], name=v))
            st.plotly_chart(fig_vars, use_container_width=True)
        else: st.error("Data not found.")

elif page == "Trade Tester":
    st.header("🧪 Trade Lab (Backtest)")
    col1, col2, col3 = st.columns(3)
    t_ticker = col1.text_input("Ticker", "MSFT").upper()
    t_date = col2.date_input("Buy Date", datetime.today() - timedelta(days=365))
    t_shares = col3.number_input("Shares", value=100)
    if st.button("Simulate Trade"):
        df = get_full_data_merged(t_ticker)
        if df is not None:
            t_date_ts = pd.Timestamp(t_date)
            try:
                idx = df.index.get_indexer([t_date_ts], method='nearest')[0]
                row_buy = df.iloc[idx]
                row_now = df.iloc[-1]
                buy_score = calculate_score(row_buy)
                buy_price = row_buy['price']
                curr_price = row_now['price']
                total_return = ((curr_price - buy_price) / buy_price) * 100
                profit = (curr_price - buy_price) * t_shares
                st.success("Trade Simulated!")
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Score at Purchase", f"{buy_score}/100")
                res_col2.metric("Buy Price", f"${buy_price:.2f}")
                res_col3.metric("Total Return", f"{total_return:.2f}%", f"${profit:.2f}")
                st.session_state['trade_log'].append({"Ticker": t_ticker, "Date": t_date, "Score": buy_score, "Buy Price": buy_price, "Return %": total_return})
            except Exception as e: st.error(f"Error: {e}")
        else: st.error("Could not fetch data.")
    if st.session_state['trade_log']:
        st.subheader("Trade History Log")
        st.dataframe(pd.DataFrame(st.session_state['trade_log']))
