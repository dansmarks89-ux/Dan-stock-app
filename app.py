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
st.set_page_config(page_title="Alpha Pro v19.5", layout="wide")

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

# --- STRATEGY WEIGHTS (UPDATED v19.5) ---
WEIGHTS_BALANCED = {'growth': 20, 'momentum': 20, 'profitability': 20, 'roe': 20, 'value': 20}
WEIGHTS_SPECULATIVE = {'growth': 40, 'momentum': 60, 'profitability': 0, 'roe': 0, 'value': 0}

# DEFENSIVE 2.1 (Recession Ready)
WEIGHTS_DEFENSIVE = {'value': 25, 'roe': 15, 'profitability': 25, 'momentum': 10, 'growth': 25}

# AGGRESSIVE 2.0 (GARP Mode)
# Growth (35), Mom (25), Prof (20 - Rule of 40), Value (20 - Hybrid)
WEIGHTS_AGGRESSIVE = {'growth': 35, 'momentum': 25, 'profitability': 20, 'value': 20, 'roe': 0}

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
# 4. SCORING ENGINE (V19.5 - AGGRESSIVE GARP)
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
    if mode == "Defensive":
        # BETA Logic for Defensive
        beta = safe_float(overview.get('Beta'))
        raw_metrics['Rev Growth'] = beta 
        if beta:
            base_pts = get_points(beta, 0.8, 1.3, 20, False)
            log["Volatility (Beta)"] = process_metric("Beta", f"{beta:.2f}", 'growth', base_pts)
        else:
            log["Volatility (Beta)"] = process_metric("Beta", "N/A", 'growth', 0)
    else:
        # REVENUE GROWTH (Critical for Aggressive)
        rev_growth = safe_float(overview.get('QuarterlyRevenueGrowthYOY'))
        raw_metrics['Rev Growth'] = rev_growth * 100 if rev_growth else None
        
        # Scoring: Aggressive targets >30%, Balanced >20%
        target_growth = 30 if mode == "Aggressive" else 20
        base_pts = get_points(rev_growth * 100, target_growth, 0, 20, True) if rev_growth else 0
        log["Revenue Growth"] = process_metric("Rev Growth", f"{rev_growth*100:.1f}%" if rev_growth else None, 'growth', base_pts)

    # --- 2. PROFITABILITY ---
    if mode == "Defensive":
        # Recession Logic: Div + FCF
        div_yield = safe_float(overview.get('DividendYield'))
        div_yield = div_yield * 100 if div_yield else 0
        div_score = get_points(div_yield, 3.5, 0.5, 20, True)
        
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
                    fcf_score = get_points(fcf_yield, 6.0, 0.0, 20, True)
        except: pass
        
        combined_prof = (div_score + fcf_score) / 2
        raw_metrics['FCF Yield'] = fcf_yield
        log["Income (Div+FCF)"] = process_metric("Income", f"Div: {div_yield:.1f}% / FCF: {fcf_yield:.1f}%" if fcf_yield else "N/A", 'profitability', combined_prof)
    
    elif mode == "Aggressive":
        # RULE OF 40 (Growth + FCF Margin)
        r_growth = safe_float(overview.get('QuarterlyRevenueGrowthYOY'))
        r_growth = r_growth * 100 if r_growth else 0
        
        fcf_margin = 0
        try:
            reports = cash_flow.get('annualReports', [])
            if reports:
                latest = reports[0]
                ocf = safe_float(latest.get('operatingCashflow'))
                capex = safe_float(latest.get('capitalExpenditures'))
                rev_annual = safe_float(overview.get('RevenueTTM')) 
                
                if ocf and capex and rev_annual and rev_annual > 0:
                    fcf = ocf - capex
                    fcf_margin = (fcf / rev_annual) * 100
        except: pass
        
        rule_40 = r_growth + fcf_margin
        base_rule_pts = get_points(rule_40, 40.0, 0.0, 20, True)
        
        raw_metrics['Profit Margin'] = fcf_margin 
        log["Rule of 40"] = process_metric("Rev+FCF%", f"{rule_40:.1f}", 'profitability', base_rule_pts)

    else:
        # Standard Net Margin
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
                    base_solvency = get_points(de_ratio, 0.5, 3.0, 20, False)
        except: pass
        raw_metrics['ROE'] = de_ratio 
        log["Solvency (Debt)"] = process_metric("Debt/Eq", f"{de_ratio:.2f}" if de_ratio else None, 'roe', base_solvency)
        
    else:
        # Standard ROE (Aggressive usually ignores this via weights)
        roe = safe_float(overview.get('ReturnOnEquityTTM'))
        if roe and roe < 5: roe = roe * 100 
        raw_metrics['ROE'] = roe
        base_pts = get_points(roe, 25, 5, 20, True) if roe else 0
        log["ROE"] = process_metric("ROE", f"{roe:.1f}%" if roe else None, 'roe', base_pts)

    # --- 4. VALUE ---
    val_label = "PEG"
    val_raw = None
    base_val_pts = 0

    if mode == "Defensive":
        # Hybrid (Rel P/E + EBITDA)
        val_label = "Hybrid Val"
        rel_score = 0
        current_pe = safe_float(overview.get('PERatio'))
        avg_pe = None
        rel_str = "N/A"
        
        if historical_pe_df is not None and not historical_pe_df.empty:
            cutoff_date = historical_pe_df.index[-1] - timedelta(days=1825)
            recent_pe = historical_pe_df[historical_pe_df.index >= cutoff_date]['pe_ratio']
            if not recent_pe.empty:
                q_low = recent_pe.quantile(0.10)
                q_high = recent_pe.quantile(0.90)
                trimmed_pe = recent_pe[(recent_pe >= q_low) & (recent_pe <= q_high)]
                if not trimmed_pe.empty: avg_pe = trimmed_pe.mean()
                else: avg_pe = recent_pe.mean()
                avg_pe = max(avg_pe, 15.0) 
            
        if current_pe and avg_pe:
            pe_discount = ((avg_pe - current_pe) / avg_pe) * 100
            rel_score = get_points(pe_discount, 5.0, -50.0, 20, True)
            rel_str = f"Prem: {pe_discount:+.0f}% (Avg: {avg_pe:.1f})"
        
        abs_score = 0
        ev_ebitda = safe_float(overview.get('EVToEBITDA'))
        abs_str = "N/A"
        if ev_ebitda:
            abs_score = get_points(ev_ebitda, 8.0, 20.0, 20, False)
            abs_str = f"{ev_ebitda:.1f}x EBITDA"
            val_raw = ev_ebitda 
            
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
        
    elif mode == "Aggressive":
        # --- NEW AGGRESSIVE VALUE (1/3 Rel P/E + 2/3 EV/Sales) ---
        val_label = "Hybrid Growth"
        
        # 1. Rel P/E (1/3 Weight)
        rel_pe_score = 0
        current_pe = safe_float(overview.get('PERatio'))
        avg_pe = None
        rel_str = "N/A"
        
        if historical_pe_df is not None and not historical_pe_df.empty:
            # Use same logic as Defensive for extracting Mean P/E
            cutoff_date = historical_pe_df.index[-1] - timedelta(days=1825)
            recent_pe = historical_pe_df[historical_pe_df.index >= cutoff_date]['pe_ratio']
            if not recent_pe.empty:
                q_low = recent_pe.quantile(0.10)
                q_high = recent_pe.quantile(0.90)
                trimmed_pe = recent_pe[(recent_pe >= q_low) & (recent_pe <= q_high)]
                if not trimmed_pe.empty: avg_pe = trimmed_pe.mean()
                else: avg_pe = recent_pe.mean()
        
        if current_pe and avg_pe:
            pe_discount = ((avg_pe - current_pe) / avg_pe) * 100
            rel_pe_score = get_points(pe_discount, 5.0, -50.0, 20, True)
            rel_str = f"PE Prem: {pe_discount:+.0f}%"

        # 2. EV/Sales (2/3 Weight)
        ev_sales = safe_float(overview.get('EVToRevenue'))
        ev_score = 0
        ev_str = "N/A"
        
        if ev_sales:
            # Scale: <5x is Cheap (20), >15x is Expensive (0)
            ev_score = get_points(ev_sales, 5.0, 15.0, 20, False)
            ev_str = f"{ev_sales:.1f}x Sales"
        
        # 3. Combine (0.33 vs 0.67)
        if current_pe and avg_pe and ev_sales:
            base_val_pts = (rel_pe_score * 0.33) + (ev_score * 0.67)
            val_str = f"{rel_str} | {ev_str}"
            val_raw = ev_sales
        elif ev_sales:
            # Fallback if no P/E history (common for growth)
            base_val_pts = ev_score
            val_str = f"{ev_str} (Sales Only)"
            val_raw = ev_sales
        else:
            base_val_pts = 0
            val_str = "Data Missing"
        
        raw_metrics['PEG'] = val_raw 
        log["Valuation"] = process_metric(val_label, val_str, 'value', base_val_pts)

    else:
        # Balanced/Speculative Logic (Standard PEG)
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
    
    if mode == "Defensive":
        # Volatility (Beta) is stored in the "Growth" slot for Defensive, so Mom is standard Trend here
        pass # Logic handled below is standard trend
        
    # STANDARD MOMENTUM (Used by All Modes)
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
