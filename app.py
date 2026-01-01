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

def calculate_sector_relative_score(overview, cash_flow, balance_sheet, price_df, weights, use_50ma=False, mode="Balanced", historical_pe_df=None):
    earned, possible = 0, 0
    log = {}
    raw_metrics = {}
    base_scores = {} 
    
    # Get sector context
    sector_name, sector_bench = get_sector_context(overview)
    log["Sector"] = sector_name
    
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
            
    # --- HELPER: GET FCF (Annual with Quarterly Fallback) ---
    def get_fcf_ttm(cash_flow_data):
        try:
            reports = cash_flow_data.get('annualReports', [])
            if reports:
                latest = reports[0]
                ocf = safe_float(latest.get('operatingCashflow'))
                capex = safe_float(latest.get('capitalExpenditures'))
                if ocf is not None and capex is not None: return ocf - capex
        except: pass
        try:
            q_reports = cash_flow_data.get('quarterlyReports', [])
            if len(q_reports) >= 4:
                ttm_ocf = 0
                ttm_capex = 0
                for i in range(4):
                    ttm_ocf += safe_float(q_reports[i].get('operatingCashflow')) or 0
                    ttm_capex += safe_float(q_reports[i].get('capitalExpenditures')) or 0
                return ttm_ocf - ttm_capex
        except: pass
        return None

    # --- 1. GROWTH ---
    rev_growth = safe_float(overview.get('QuarterlyRevenueGrowthYOY'))
    raw_metrics['Rev Growth'] = rev_growth * 100 if rev_growth else None

    if mode == "Defensive":
        beta = safe_float(overview.get('Beta'))
        if beta:
            base_pts = get_points(beta, 0.8, 1.3, 20, False)
            log["Volatility (Beta)"] = process_metric("Beta", f"{beta:.2f}", 'growth', base_pts)
        else:
            log["Volatility (Beta)"] = process_metric("Beta", "N/A", 'growth', 0)
            
    elif mode == "Balanced":
        eps_growth = safe_float(overview.get('QuarterlyEarningsGrowthYOY'))
        rev_score = get_points(rev_growth * 100, 20, 0, 20, True) if rev_growth else 0
        eps_score = get_points(eps_growth * 100, 20, 0, 20, True) if eps_growth else 0
        avg_score = (rev_score + eps_score) / 2
        val_str = f"Rev: {rev_growth*100:.1f}% / EPS: {eps_growth*100:.1f}%" if eps_growth else f"Rev: {rev_growth*100:.1f}%"
        log["Growth (Top+Bottom)"] = process_metric("Growth Blend", val_str, 'growth', avg_score)
        
    else:
        target_growth = 30 if mode == "Aggressive" else 20
        base_pts = get_points(rev_growth * 100, target_growth, 0, 20, True) if rev_growth else 0
        log["Revenue Growth"] = process_metric("Rev Growth", f"{rev_growth*100:.1f}%" if rev_growth else None, 'growth', base_pts)

    # --- 2. PROFITABILITY ---
    margin = safe_float(overview.get('ProfitMargin'))
    raw_metrics['Profit Margin'] = margin * 100 if margin else None

    fcf_val = get_fcf_ttm(cash_flow)
    mcap = safe_float(overview.get('MarketCapitalization'))
    fcf_yield = None
    if fcf_val is not None and mcap and mcap > 0:
        fcf_yield = (fcf_val / mcap) * 100
    
    raw_metrics['FCF Yield'] = fcf_yield 

    if mode == "Defensive":
        # ADVANCED INCOME METRICS (Shareholder Yield + Payout Safety)
        adv_metrics = get_advanced_income_metrics(cash_flow, mcap)
        
        div_yield = safe_float(overview.get('DividendYield'))
        div_yield = div_yield * 100 if div_yield else 0
        
        # Shareholder Yield
        shareholder_yield = adv_metrics.get("shareholder_yield")
        sy_score = get_points(shareholder_yield, 5.0, 0.0, 20, True) if shareholder_yield else 0
        
        # Payout Penalty
        payout = adv_metrics.get("payout_ratio")
        safety_penalty = 1.0
        if payout and payout > 90:
            safety_penalty = 0.5
            log["Safety Warning"] = f"Payout Ratio {payout:.0f}% (>90%)"
            
        fcf_score = get_points(fcf_yield, 6.0, 0.0, 20, True) if fcf_yield is not None else 0
        div_score = get_points(div_yield, 3.5, 0.5, 20, True)
        
        # Max Logic
        best_income_score = max(fcf_score, div_score, sy_score) * safety_penalty
        
        # Log Logic
        if shareholder_yield and shareholder_yield > div_yield + 1:
             msg = f"Yld: {div_yield:.1f}% | Total: {shareholder_yield:.1f}% (Buybacks!)"
        else:
             fcf_str = f"{fcf_yield:.1f}%" if fcf_yield is not None else "N/A"
             msg = f"Yld: {div_yield:.1f}% / FCF: {fcf_str}"
             
        log["Income (Smart)"] = process_metric("Total Yield", msg, 'profitability', best_income_score)
    
    elif mode == "Balanced":
        fcf_score = get_points(fcf_yield, 5.0, 1.0, 20, True) if fcf_yield is not None else 0
        margin_score = get_points(margin * 100, 25, 5, 20, True) if margin else 0
        avg_prof = (margin_score * 0.3) + (fcf_score * 0.7)
        fcf_str = f"{fcf_yield:.1f}%" if fcf_yield is not None else "N/A"
        val_str = f"N.Marg: {margin*100:.1f}% / FCF: {fcf_str}"
        log["Profit (Cash Heavy)"] = process_metric("Profit Blend", val_str, 'profitability', avg_prof)
        
    elif mode == "Aggressive":
        r_growth = safe_float(overview.get('QuarterlyRevenueGrowthYOY'))
        r_growth = r_growth * 100 if r_growth else 0
        fcf_margin = 0
        rev_annual = safe_float(overview.get('RevenueTTM')) 
        if fcf_val is not None and rev_annual and rev_annual > 0:
            fcf_margin = (fcf_val / rev_annual) * 100
        else:
            try:
                net_income = safe_float(cash_flow.get('annualReports', [])[0].get('netIncome'))
                if net_income and rev_annual: fcf_margin = (net_income / rev_annual) * 100
            except: pass
        rule_40 = r_growth + fcf_margin
        base_rule_pts = get_points(rule_40, 40.0, 0.0, 20, True)
        raw_metrics['Profit Margin'] = fcf_margin 
        log["Rule of 40"] = process_metric("Rev+FCF%", f"{rule_40:.1f} (Gr:{r_growth:.1f}+M:{fcf_margin:.1f})", 'profitability', base_rule_pts)

    else:
        base_margin = get_points(margin * 100, 25, 5, 20, True) if margin else 0
        log["Profitability"] = process_metric("Net Margin", f"{margin*100:.1f}%" if margin else None, 'profitability', base_margin)

    # --- 3. QUALITY / ROE ---
    roe = safe_float(overview.get('ReturnOnEquityTTM'))
    if roe and roe < 5: roe = roe * 100 
    raw_metrics['ROE'] = roe
    
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
        log["Solvency (Debt)"] = process_metric("Debt/Eq", f"{de_ratio:.2f}" if de_ratio else None, 'roe', base_solvency)

    elif mode == "Balanced":
        roa = safe_float(overview.get('ReturnOnAssetsTTM'))
        roa = roa * 100 if roa else 0
        roa_score = get_points(roa, 10.0, 2.0, 20, True)
        log["Quality (ROA)"] = process_metric("ROA", f"{roa:.1f}% (ROE: {roe:.1f}%)", 'roe', roa_score)

    else:
        base_pts = get_points(roe, 25, 5, 20, True) if roe else 0
        log["ROE"] = process_metric("ROE", f"{roe:.1f}%" if roe else None, 'roe', base_pts)

    # --- 4. VALUE ---
    val_label = "PEG"
    val_raw = None
    base_val_pts = 0

    if mode == "Defensive":
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
            rel_str = f"Disc: {pe_discount:+.0f}%"
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
        val_label = "Hybrid Growth"
        rel_pe_score = 0
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
        if current_pe and avg_pe:
            pe_discount = ((avg_pe - current_pe) / avg_pe) * 100
            rel_pe_score = get_points(pe_discount, 5.0, -50.0, 20, True)
            rel_str = f"PE Disc: {pe_discount:+.0f}%"
        ev_sales = safe_float(overview.get('EVToRevenue'))
        ev_score = 0
        ev_str = "N/A"
        if ev_sales:
            ev_score = get_points(ev_sales, 5.0, 15.0, 20, False)
            ev_str = f"{ev_sales:.1f}x Sales"
        if current_pe and avg_pe and ev_sales:
            base_val_pts = (rel_pe_score * 0.33) + (ev_score * 0.67)
            val_str = f"{rel_str} | {ev_str}"
            val_raw = ev_sales
        elif ev_sales:
            base_val_pts = ev_score
            val_str = f"{ev_str} (Sales Only)"
            val_raw = ev_sales
        else:
            base_val_pts = 0
            val_str = "Data Missing"
        raw_metrics['PEG'] = val_raw 
        log["Valuation"] = process_metric(val_label, val_str, 'value', base_val_pts)
    
    elif mode == "Balanced":
        val_label = "Hybrid PEG"
        peg = safe_float(overview.get('PEGRatio'))
        peg_score = get_points(peg, 1.0, 2.5, 20, False) if peg else 0
        rel_score = 0
        current_pe = safe_float(overview.get('PERatio'))
        avg_pe = None
        pe_discount = None 
        if historical_pe_df is not None and not historical_pe_df.empty:
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
            rel_score = get_points(pe_discount, 5.0, -50.0, 20, True)
        
        avg_val_pts = (peg_score + rel_score) / 2
        
        if peg is not None: peg_str = f"{peg:.2f}"
        else: peg_str = "N/A"
        if peg and current_pe and avg_pe and pe_discount is not None:
            val_str = f"PEG: {peg_str} | Disc: {pe_discount:+.0f}%"
        else:
            val_str = f"PEG: {peg_str}"
        raw_metrics['PEG'] = peg
        log["Valuation (PEG+Rel)"] = process_metric(val_label, val_str, 'value', avg_val_pts)

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
    
    if mode == "Defensive":
        pass 
    
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
                        s_bal, _, raw_metrics, _ = calculate_sector_relative_score(ov, cf, bs, hist, WEIGHTS_BALANCED, use_50ma=False, mode="Balanced", historical_pe_df=pe_hist)
                        s_agg, _, _, _ = calculate_sector_relative_score(ov, cf, bs, hist, WEIGHTS_AGGRESSIVE, use_50ma=False, mode="Aggressive", historical_pe_df=pe_hist)
                        s_def, _, _, _ = calculate_sector_relative_score(ov, cf, bs, hist, WEIGHTS_DEFENSIVE, use_50ma=False, mode="Defensive", historical_pe_df=pe_hist)
                        s_spec, _, _, _ = calculate_sector_relative_score(ov, cf, bs, hist, WEIGHTS_SPECULATIVE, use_50ma=True, mode="Speculative")
                        
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
            score, log, raw_metrics, base_scores = calculate_sector_relative_score(
                ov, cf, bs, hist, active_weights, 
                use_50ma=is_speculative, 
                mode=mode_name,
                historical_pe_df=pe_hist
            )
            
            # DB Prep
            s_bal, _, _, _ = calculate_sector_relative_score(ov, cf, bs, hist, WEIGHTS_BALANCED, use_50ma=False, mode="Balanced", historical_pe_df=pe_hist)
            s_agg, _, _, _ = calculate_sector_relative_score(ov, cf, bs, hist, WEIGHTS_AGGRESSIVE, use_50ma=False, mode="Aggressive", historical_pe_df=pe_hist)
            s_def, _, _, _ = calculate_sector_relative_score(ov, cf, bs, hist, WEIGHTS_DEFENSIVE, use_50ma=False, mode="Defensive", historical_pe_df=pe_hist)
            s_spec, _, _, _ = calculate_sector_relative_score(ov, cf, bs, hist, WEIGHTS_SPECULATIVE, use_50ma=True, mode="Speculative")
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
                st.caption(f"📊 **Sector:** {log.get('Sector', 'Unknown')} | Scores are sector-relative (50 = average)")
                st.info("💡 Scores compare this stock to others in its sector. 70+ = excellent, 50 = average, 30- = weak")
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
                            s_bal, _, raw_metrics, _ = calculate_sector_relative_score(ov, cf, bs, hist, WEIGHTS_BALANCED, use_50ma=False, mode="Balanced", historical_pe_df=pe_hist)
                            s_agg, _, _, _ = calculate_sector_relative_score(ov, cf, bs, hist, WEIGHTS_AGGRESSIVE, use_50ma=False, mode="Aggressive", historical_pe_df=pe_hist)
                            s_def, _, _, _ = calculate_sector_relative_score(ov, cf, bs, hist, WEIGHTS_DEFENSIVE, use_50ma=False, mode="Defensive", historical_pe_df=pe_hist)
                            s_spec, _, _, _ = calculate_sector_relative_score(ov, cf, bs, hist, WEIGHTS_SPECULATIVE, use_50ma=True, mode="Speculative")
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
