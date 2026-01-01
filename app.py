import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import numpy as np
is_premium = True

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
st.set_page_config(page_title="Alpha Pro v22.0", layout="wide")

SECTOR_ETFS = {
    "Technology (XLK)": "XLK", "Healthcare (XLV)": "XLV", "Financials (XLF)": "XLF",
    "Energy (XLE)": "XLE", "Communication (XLC)": "XLC", "Consumer Disc (XLY)": "XLY",
    "Consumer Stap (XLP)": "XLP", "Industrials (XLI)": "XLI", "Utilities (XLU)": "XLU",
    "Real Estate (XLRE)": "XLRE", "S&P 500 (SPY)": "SPY"
}

# Revised Weights per request
WEIGHTS_BALANCED = {'growth': 20, 'momentum': 20, 'profitability': 20, 'roe': 20, 'value': 20}
WEIGHTS_DEFENSIVE = {'value': 25, 'roe': 15, 'profitability': 25, 'momentum': 10, 'growth': 25}
WEIGHTS_AGGRESSIVE = {'growth': 35, 'momentum': 20, 'profitability': 20, 'value': 25, 'roe': 0}
# Speculative: Growth 40, Trend 35, Insider 25
WEIGHTS_SPECULATIVE = {'growth': 40, 'momentum': 35, 'insider': 25}

SECTOR_BENCHMARKS = {
    "Technology": {"margin_median": 20, "growth_median": 15, "pe_median": 25, "fcf_yield_median": 3.5, "roe_median": 18},
    "Healthcare": {"margin_median": 15, "growth_median": 8, "pe_median": 22, "fcf_yield_median": 4.0, "roe_median": 15},
    "Default": {"margin_median": 12, "growth_median": 8, "pe_median": 18, "fcf_yield_median": 4.0, "roe_median": 14}
}

def get_sector_context(overview):
    sector = overview.get('Sector', 'Default')
    return sector, SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS["Default"])

# ==========================================
# 2. GOOGLE SHEETS DATABASE (STRICT FORMATTING)
# ==========================================
conn = st.connection("gsheets", type=GSheetsConnection)

def get_watchlist_data():
    try:
        df = conn.read(worksheet="Sheet1", ttl=0)
        if df.empty or 'Ticker' not in df.columns: return pd.DataFrame()
        df = df.dropna(subset=['Ticker'])
        return df
    except: return pd.DataFrame()

def batch_save_to_sheet(new_rows_list):
    if not new_rows_list: return
    try:
        existing_df = get_watchlist_data()
        new_batch_df = pd.DataFrame(new_rows_list)
        
        if not existing_df.empty:
            # Prevent same-day duplicates (One update per day per ticker allowed)
            # We create a composite key to identify today's existing entries
            existing_df['_key'] = existing_df['Ticker'] + "_" + existing_df['Date']
            new_batch_df['_key'] = new_batch_df['Ticker'] + "_" + new_batch_df['Date']
            
            # Remove rows in existing that match the new batch (overwrite today's data)
            existing_df = existing_df[~existing_df['_key'].isin(new_batch_df['_key'])]
            
            # Clean up keys
            existing_df = existing_df.drop(columns=['_key'], errors='ignore')
            new_batch_df = new_batch_df.drop(columns=['_key'], errors='ignore')
            
            updated_df = pd.concat([existing_df, new_batch_df], ignore_index=True)
        else:
            updated_df = new_batch_df
            
        conn.update(worksheet="Sheet1", data=updated_df)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Batch Save Failed: {e}")
        return False

def add_log_to_sheet(ticker, curr_price, raw_metrics, scores_dict, sector, mcap):
    """Single entry save with strict 2 decimal rounding"""
    try:
        existing_df = get_watchlist_data()
        log_date = (datetime.now() - timedelta(hours=6)).strftime("%Y-%m-%d")

        # Helper to round safely
        def r2(val):
            try: return round(float(val), 2)
            except: return None

        new_data = {
            "Ticker": ticker, "Date": log_date, "Sector": sector, "Price": r2(curr_price), "Market Cap": r2(mcap),
            "Rev Growth": r2(raw_metrics.get('Rev Growth')), "Profit Margin": r2(raw_metrics.get('Profit Margin')),
            "FCF Yield": r2(raw_metrics.get('FCF Yield')), "ROE": r2(raw_metrics.get('ROE')),
            "PEG": r2(raw_metrics.get('PEG')), "Mom Position %": r2(raw_metrics.get('Mom Position')),
            "Mom Slope %": r2(raw_metrics.get('Mom Slope')), "RVOL": r2(raw_metrics.get('RVOL')),
            "RSI": r2(raw_metrics.get('RSI')), "Insider %": r2(raw_metrics.get('Insider %')),
            "Score (Balanced)": int(scores_dict.get('Balanced')), "Score (Aggressive)": int(scores_dict.get('Aggressive')),
            "Score (Defensive)": int(scores_dict.get('Defensive')), "Score (Speculative)": int(scores_dict.get('Speculative'))
        }
        
        new_row = pd.DataFrame([new_data])
        
        if not existing_df.empty:
            # Overwrite only if ticker exists for TODAY
            mask = (existing_df['Ticker'] == ticker) & (existing_df['Date'] == log_date)
            existing_df = existing_df[~mask]
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
    try:
        df = get_watchlist_data()
        if not df.empty:
            df = df[df['Ticker'] != ticker]
            conn.update(worksheet="Sheet1", data=df)
            st.cache_data.clear()
    except: st.error("Remove failed")

# ==========================================
# 3. DATA ENGINE
# ==========================================
def safe_float(val):
    try:
        if val is None or val == "None" or val == "-": return None
        return float(val)
    except: return None

def round_metric(val, decimals=2):
    """Safely round numeric values"""
    try:
        return round(float(val), decimals)
    except:
        return None

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
# 4. SCORING ENGINE (UPDATED)
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

def calculate_rsi(prices, period=14):
    try:
        deltas = prices.diff()
        gain = (deltas.where(deltas > 0, 0)).rolling(window=period).mean()
        loss = (-deltas.where(deltas < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    except: return 50.0

def get_earnings_quality(cash_flow_data, income_statement):
    try:
        reports = cash_flow_data.get('annualReports', [])
        if reports:
            latest = reports[0]
            ocf = safe_float(latest.get('operatingCashflow'))
            net_income = safe_float(latest.get('netIncome'))
            if ocf and net_income and net_income > 0:
                cash_conversion = (ocf / net_income) * 100
                if cash_conversion >= 100: return 20, cash_conversion
                elif cash_conversion >= 80: return 15, cash_conversion
                elif cash_conversion >= 60: return 10, cash_conversion
                else: return 5, cash_conversion
    except: pass
    return 10, None

def get_insider_ownership(overview):
    insider_pct = safe_float(overview.get('PercentInsiders'))
    if insider_pct:
        # Smooth Scale: 0-25% is the sweet spot ramp up. 
        insider_pct = insider_pct * 100 if insider_pct < 1 else insider_pct
        
        # Evidence based scoring: 
        # > 30%: 20 pts (Max - high skin in game)
        # 10-30%: 15-20 pts (Ideal)
        # 1-10%: 5-15 pts (Ok)
        # 0%: 0 pts
        
        if insider_pct >= 30: return 20, insider_pct
        if insider_pct >= 10: return 18, insider_pct # Sweet spot
        if insider_pct > 0: return (insider_pct / 10) * 15, insider_pct # Linear ramp to 15
        
    return 0, 0

def get_advanced_income_metrics(cash_flow_data, mcap_int):
    metrics = {"payout_ratio": None, "shareholder_yield": None}
    try:
        reports = cash_flow_data.get('annualReports', [])
        if len(reports) >= 1:
            latest = reports[0]
            divs_paid = safe_float(latest.get('dividendPayout')) or 0
            buybacks = abs(safe_float(latest.get('paymentsForRepurchaseOfCommonStock')) or 0)
            if mcap_int and mcap_int > 0: metrics["shareholder_yield"] = ((divs_paid + buybacks) / mcap_int) * 100
            net_income = safe_float(latest.get('netIncome'))
            if net_income and net_income > 0: metrics["payout_ratio"] = (divs_paid / net_income) * 100
    except: pass
    return metrics

def calculate_sector_relative_score(overview, cash_flow, balance_sheet, price_df, weights, use_50ma=False, mode="Balanced", historical_pe_df=None):
    earned, possible = 0, 0
    log = {}
    raw_metrics = {}
    base_scores = {} 
    
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
                ttm_ocf = 0; ttm_capex = 0
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
        # Aggressive & Speculative
        target_growth = 30 if mode == "Aggressive" else 40 # 40% for Spec
        base_pts = get_points(rev_growth * 100, target_growth, 0, 20, True) if rev_growth else 0
        log["Revenue Growth"] = process_metric("Rev Growth", f"{rev_growth*100:.1f}%" if rev_growth else None, 'growth', base_pts)

    # --- 2. PROFITABILITY ---
    margin = safe_float(overview.get('ProfitMargin'))
    raw_metrics['Profit Margin'] = margin * 100 if margin else None
    fcf_val = get_fcf_ttm(cash_flow)
    mcap = safe_float(overview.get('MarketCapitalization'))
    fcf_yield = None
    if fcf_val is not None and mcap and mcap > 0: fcf_yield = (fcf_val / mcap) * 100
    raw_metrics['FCF Yield'] = fcf_yield 

    if mode == "Balanced":
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
        if fcf_val is not None and rev_annual and rev_annual > 0: fcf_margin = (fcf_val / rev_annual) * 100
        else:
            try:
                net_income = safe_float(cash_flow.get('annualReports', [])[0].get('netIncome'))
                if net_income and rev_annual: fcf_margin = (net_income / rev_annual) * 100
            except: pass
        rule_40 = r_growth + fcf_margin
        base_rule_pts = get_points(rule_40, 40.0, 0.0, 20, True)
        raw_metrics['Profit Margin'] = fcf_margin 
        log["Rule of 40"] = process_metric("Rev+FCF%", f"{rule_40:.1f}", 'profitability', base_rule_pts)
    elif mode == "Defensive":
        adv_metrics = get_advanced_income_metrics(cash_flow, mcap)
        div_yield = safe_float(overview.get('DividendYield'))
        div_yield = div_yield * 100 if div_yield else 0
        shareholder_yield = adv_metrics.get("shareholder_yield")
        sy_score = get_points(shareholder_yield, 5.0, 0.0, 20, True) if shareholder_yield else 0
        payout = adv_metrics.get("payout_ratio")
        safety_penalty = 1.0
        if payout and payout > 90:
            safety_penalty = 0.5
            log["Safety Warning"] = f"Payout Ratio {payout:.0f}% (>90%)"
        fcf_score = get_points(fcf_yield, 6.0, 0.0, 20, True) if fcf_yield is not None else 0
        div_score = get_points(div_yield, 3.5, 0.5, 20, True)
        best_income_score = max(fcf_score, div_score, sy_score) * safety_penalty
        
        fcf_str = f"{fcf_yield:.1f}%" if fcf_yield is not None else "N/A"
        if shareholder_yield and shareholder_yield > div_yield + 1: msg = f"Yld: {div_yield:.1f}% | Tot: {shareholder_yield:.1f}%"
        else: msg = f"Yld: {div_yield:.1f}% / FCF: {fcf_str}"
        log["Income (Smart)"] = process_metric("Total Yield", msg, 'profitability', best_income_score)
    elif mode == "Speculative":
        # Speculative ignores Profitability in favor of Insider
        pass
    else:
        base_margin = get_points(margin * 100, 25, 5, 20, True) if margin else 0
        log["Profitability"] = process_metric("Net Margin", f"{margin*100:.1f}%" if margin else None, 'profitability', base_margin)

    # --- 3. QUALITY / ROE ---
    roe = safe_float(overview.get('ReturnOnEquityTTM'))
    if roe and roe < 5: roe = roe * 100 
    raw_metrics['ROE'] = roe
    eq_score, cash_conv = get_earnings_quality(cash_flow, overview)
    insider_score, insider_pct = get_insider_ownership(overview)
    raw_metrics['Insider %'] = insider_pct
    extra_quality = (eq_score + insider_score) / 2
    
    if mode == "Balanced":
        roa = safe_float(overview.get('ReturnOnAssetsTTM'))
        roa = roa * 100 if roa else 0
        roa_score = get_points(roa, 10.0, 2.0, 20, True)
        final_quality = (roa_score * 0.7) + (extra_quality * 0.3)
        log["Quality (ROA+Ins)"] = process_metric("Quality", f"ROA {roa:.1f}% | Cash Conv {cash_conv if cash_conv else 0:.0f}%", 'roe', final_quality)
    elif mode == "Defensive":
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
                if total_financial_debt > 0 and equity and equity > 0: de_ratio = total_financial_debt / equity
                if de_ratio is not None: base_solvency = get_points(de_ratio, 0.5, 3.0, 20, False)
        except: pass
        log["Solvency (Debt)"] = process_metric("Debt/Eq", f"{de_ratio:.2f}" if de_ratio else None, 'roe', base_solvency)
    elif mode == "Speculative":
        # --- NEW SPECULATIVE INSIDER METRIC ---
        # "Insider %" replaces ROE in Speculative Mode weights
        log["Insider Align"] = process_metric("Insider Own", f"{insider_pct:.1f}%", 'insider', insider_score)
    else:
        log["ROE"] = process_metric("ROE", f"{roe:.1f}%" if roe else None, 'roe', get_points(roe, 25, 5, 20, True) if roe else 0)

    # --- 4. VALUE ---
    val_label = "PEG"; val_raw = None; base_val_pts = 0
    if mode == "Defensive":
        val_label = "Hybrid Val"
        rel_score = 0; current_pe = safe_float(overview.get('PERatio')); avg_pe = None
        if historical_pe_df is not None and not historical_pe_df.empty:
            cutoff_date = historical_pe_df.index[-1] - timedelta(days=1825)
            recent_pe = historical_pe_df[historical_pe_df.index >= cutoff_date]['pe_ratio']
            if not recent_pe.empty: avg_pe = recent_pe.mean(); avg_pe = max(avg_pe, 15.0) 
        if current_pe and avg_pe:
            pe_discount = ((avg_pe - current_pe) / avg_pe) * 100
            rel_score = get_points(pe_discount, 5.0, -50.0, 20, True)
        abs_score = 0; ev_ebitda = safe_float(overview.get('EVToEBITDA'))
        if ev_ebitda: abs_score = get_points(ev_ebitda, 8.0, 20.0, 20, False); val_raw = ev_ebitda 
        if current_pe and ev_ebitda: base_val_pts = (rel_score + abs_score) / 2
        elif ev_ebitda: base_val_pts = abs_score
        raw_metrics['PEG'] = val_raw
        # Fix: Handle None ev_ebitda
        if ev_ebitda is not None:
            log["Valuation"] = process_metric(val_label, f"{ev_ebitda:.1f}x EBITDA", 'value', base_val_pts)
        else:
            log["Valuation"] = process_metric(val_label, None, 'value', 0)
        
    elif mode == "Aggressive":
        val_label = "Hybrid Growth"; rel_pe_score = 0; current_pe = safe_float(overview.get('PERatio')); avg_pe = None
        if historical_pe_df is not None and not historical_pe_df.empty:
            cutoff_date = historical_pe_df.index[-1] - timedelta(days=1825)
            recent_pe = historical_pe_df[historical_pe_df.index >= cutoff_date]['pe_ratio']
            if not recent_pe.empty: avg_pe = recent_pe.mean()
        if current_pe and avg_pe:
            pe_discount = ((avg_pe - current_pe) / avg_pe) * 100
            rel_pe_score = get_points(pe_discount, 5.0, -50.0, 20, True)
        ev_sales = safe_float(overview.get('EVToRevenue')); ev_score = 0
        if ev_sales: ev_score = get_points(ev_sales, 5.0, 15.0, 20, False)
        if current_pe and avg_pe and ev_sales: base_val_pts = (rel_pe_score * 0.33) + (ev_score * 0.67)
        elif ev_sales: base_val_pts = ev_score
        raw_metrics['PEG'] = ev_sales 
        log["Valuation"] = process_metric(val_label, f"{ev_sales:.1f}x Sales" if ev_sales else "N/A", 'value', base_val_pts)
    
    elif mode == "Balanced":
        val_label = "Hybrid PEG"; peg = safe_float(overview.get('PEGRatio')); peg_score = get_points(peg, 1.0, 2.5, 20, False) if peg else 0
        rel_score = 0; current_pe = safe_float(overview.get('PERatio')); avg_pe = None; pe_discount = None 
        if historical_pe_df is not None and not historical_pe_df.empty:
            cutoff_date = historical_pe_df.index[-1] - timedelta(days=1825)
            recent_pe = historical_pe_df[historical_pe_df.index >= cutoff_date]['pe_ratio']
            if not recent_pe.empty: avg_pe = recent_pe.mean()
        if current_pe and avg_pe:
            pe_discount = ((avg_pe - current_pe) / avg_pe) * 100
            rel_score = get_points(pe_discount, 5.0, -50.0, 20, True)
        avg_val_pts = (peg_score + rel_score) / 2
        peg_str = f"{peg:.2f}" if peg else "N/A"
        val_str = f"PEG: {peg_str} | Disc: {pe_discount:+.0f}%" if (peg and current_pe and avg_pe and pe_discount is not None) else f"PEG: {peg_str}"
        raw_metrics['PEG'] = peg
        log["Valuation (PEG+Rel)"] = process_metric(val_label, val_str, 'value', avg_val_pts)
    elif mode == "Speculative":
        pass # Speculative ignores valuation
    else:
        val_label = "PEG"; val_raw = safe_float(overview.get('PEGRatio'))
        base_val_pts = get_points(val_raw, 1.0, 2.5, 20, False) if val_raw else 0
        raw_metrics['PEG'] = val_raw 
        log["Valuation"] = process_metric(val_label, f"{val_raw:.2f}" if val_raw else None, 'value', base_val_pts)

    # --- 5. MOMENTUM ---
    pct_diff, slope_pct, rvol = None, None, None; rsi_val = 50; base_pts = 0
    ma_window = 50 if use_50ma else 200; slope_lookback = 22 if use_50ma else 63; required_history = ma_window + slope_lookback + 5
    val_str = "N/A"
    
    if not price_df.empty and len(price_df) > required_history:
        curr_price = price_df['close'].iloc[-1]
        ma_now = price_df['close'].rolling(window=ma_window).mean().iloc[-1]
        pos_score = 0
        if pd.notna(ma_now):
            pct_diff = ((curr_price / ma_now) - 1) * 100
            if pct_diff < 0: pos_score = 0
            elif 0 <= pct_diff <= 25: pos_score = 10
            else: pos_score = max(5, 10 - ((pct_diff - 25) * 0.5))
        
        ma_old = price_df['close'].rolling(window=ma_window).mean().iloc[-slope_lookback]
        slope_score = 0
        if pd.notna(ma_old) and ma_old > 0:
            slope_pct = ((ma_now - ma_old) / ma_old) * 100
            target_slope = 10 if use_50ma else 5
            slope_score = min(10, (slope_pct / target_slope) * 10) if slope_pct > 0 else 0
        
        try: rvol = (price_df['volume'].iloc[-5:].mean()) / (price_df['volume'].iloc[-20:].mean())
        except: rvol = 1.0
        
        rsi_val = calculate_rsi(price_df['close'])
        rsi_score = 0
        if 40 <= rsi_val <= 60: rsi_score = 0
        elif 30 <= rsi_val < 40: rsi_score = 2
        elif rsi_val < 30: rsi_score = 3
        elif rsi_val > 70: rsi_score = -2

        base_pts = pos_score + slope_score + rsi_score
        if rvol > 1.2 and slope_pct > 0: base_pts = min(20, base_pts + 2) 
        elif rvol < 0.6 and slope_pct > 0: base_pts = max(0, base_pts - 2)
        val_str = f"Pos: {pct_diff:+.1f}% | RSI: {rsi_val:.0f}"
    else: val_str = None
    
    raw_metrics['Mom Position'] = pct_diff; raw_metrics['Mom Slope'] = slope_pct
    raw_metrics['RVOL'] = rvol; raw_metrics['RSI'] = rsi_val
    ma_label = "50-Day" if use_50ma else "200-Day"
    log[f"Trend ({ma_label})"] = process_metric("Momentum", val_str, 'momentum', base_pts)

    score = int((earned/possible)*100) if possible > 0 else 0
    return score, log, raw_metrics, base_scores

# ==========================================
# 5. UI HELPERS & PLOTTING
# ==========================================
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

# --- GLOBAL HELPER: PROCESS TICKER & LOG ---
def process_and_log_ticker(ticker, api_key):
    hist, ov, cf, bs = get_alpha_data(ticker, api_key)
    pe_hist = get_historical_pe(ticker, api_key, hist)
    if not hist.empty and ov:
        s_bal, _, raw_metrics, _ = calculate_sector_relative_score(ov, cf, bs, hist, WEIGHTS_BALANCED, use_50ma=False, mode="Balanced", historical_pe_df=pe_hist)
        s_agg, _, _, _ = calculate_sector_relative_score(ov, cf, bs, hist, WEIGHTS_AGGRESSIVE, use_50ma=False, mode="Aggressive", historical_pe_df=pe_hist)
        s_def, _, _, _ = calculate_sector_relative_score(ov, cf, bs, hist, WEIGHTS_DEFENSIVE, use_50ma=False, mode="Defensive", historical_pe_df=pe_hist)
        s_spec, _, _, _ = calculate_sector_relative_score(ov, cf, bs, hist, WEIGHTS_SPECULATIVE, use_50ma=True, mode="Speculative")
        scores_db = {'Balanced': s_bal, 'Aggressive': s_agg, 'Defensive': s_def, 'Speculative': s_spec}
        
        sector = ov.get('Sector', 'Unknown')
        mcap = safe_float(ov.get('MarketCapitalization'))
        
        success = add_log_to_sheet(ticker, hist['close'].iloc[-1], raw_metrics, scores_db, sector, mcap)
        return True
    return False

def plot_score_breakdown(base_scores, mode_name):
    categories = list(base_scores.keys()); values = [base_scores[k] for k in categories]
    values_pct = [(v/20)*100 for v in values]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values_pct, theta=[k.title() for k in categories], fill='toself', name=mode_name))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=300, title=f"{mode_name} DNA")
    return fig

def generate_signal(row, score_col):
    score = row.get(score_col, 0); mom_slope = row.get('Mom Slope %', 0); peg = row.get('PEG')
    if score >= 90 and mom_slope > 5: return "🟢 Strong Buy"
    elif score >= 80 and (mom_slope > 0 or (peg and peg < 1.5)): return "🟢 Buy"
    elif score < 40: return "🔴 Sell"
    elif mom_slope < -10 and (peg and peg > 2.5): return "🔴 Sell"
    else: return "🟡 Hold"

def tf_selector(key_suffix):
    """Time frame selector for charts"""
    tf_map = {"1M": 30, "3M": 90, "1Y": 365, "5Y": 1825}
    choice = st.radio("Range", list(tf_map.keys()), index=2, horizontal=True, key=f"tf_{key_suffix}")
    return tf_map[choice]

# ==========================================
# 6. MAIN APP UI
# ==========================================
st.title("🦅 Alpha Pro v22.0 (Strict)")

with st.sidebar:
    st.header("Settings")
    if "AV_KEY" in st.secrets: key = st.secrets["AV_KEY"]
    else: key = ""; st.warning("⚠️ AV_KEY missing")
    
    st.subheader("🧠 Strategy Mode")
    strategy = st.radio("Analysis Mode", ["Balanced", "Aggressive Growth", "Defensive", "Speculative"])
    is_speculative = False; mode_name = "Balanced"
    if "Aggressive" in strategy: active_weights = WEIGHTS_AGGRESSIVE; mode_name = "Aggressive"
    elif "Defensive" in strategy: active_weights = WEIGHTS_DEFENSIVE; mode_name = "Defensive"
    elif "Speculative" in strategy: active_weights = WEIGHTS_SPECULATIVE; mode_name = "Speculative"; is_speculative = True
    else: active_weights = WEIGHTS_BALANCED; mode_name = "Balanced"

    if 'watchlist_df' not in st.session_state:
        st.session_state.watchlist_df = get_watchlist_data()
    
    st.markdown("---")
    
    # --- BATCH ADD ---
    with st.expander("➕ Batch Add Stocks"):
        batch_input = st.text_area("Tickers (comma separated)")
        if st.button("Process Batch"):
            if not key: st.error("Need API Key")
            elif batch_input:
                tickers = [t.strip().upper() for t in batch_input.split(",") if t.strip()]
                prog = st.progress(0); stat = st.empty()
                new_data_buffer = [] 
                
                # Helper to round
                def r2(val):
                    try: return round(float(val), 2)
                    except: return None

                for i, t in enumerate(tickers):
                    stat.text(f"Processing {t} ({i+1}/{len(tickers)})...")
                    
                    hist, ov, cf, bs = get_alpha_data(t, key)
                    pe_hist = get_historical_pe(t, key, hist)
                    if not hist.empty and ov:
                        s_bal, _, raw_metrics, _ = calculate_sector_relative_score(ov, cf, bs, hist, WEIGHTS_BALANCED, use_50ma=False, mode="Balanced", historical_pe_df=pe_hist)
                        s_agg, _, _, _ = calculate_sector_relative_score(ov, cf, bs, hist, WEIGHTS_AGGRESSIVE, use_50ma=False, mode="Aggressive", historical_pe_df=pe_hist)
                        s_def, _, _, _ = calculate_sector_relative_score(ov, cf, bs, hist, WEIGHTS_DEFENSIVE, use_50ma=False, mode="Defensive", historical_pe_df=pe_hist)
                        s_spec, _, _, _ = calculate_sector_relative_score(ov, cf, bs, hist, WEIGHTS_SPECULATIVE, use_50ma=True, mode="Speculative")
                        
                        curr_price = hist['close'].iloc[-1]
                        log_date = (datetime.now() - timedelta(hours=6)).strftime("%Y-%m-%d")
                        sector = ov.get('Sector', 'Unknown')
                        mcap = safe_float(ov.get('MarketCapitalization'))
                        
                        row = {
                            "Ticker": t, "Date": log_date, "Sector": sector, "Price": r2(curr_price), "Market Cap": r2(mcap),
                            "Rev Growth": r2(raw_metrics.get('Rev Growth')), "Profit Margin": r2(raw_metrics.get('Profit Margin')),
                            "FCF Yield": r2(raw_metrics.get('FCF Yield')), "ROE": r2(raw_metrics.get('ROE')),
            "PEG": r2(raw_metrics.get('PEG')), "Mom Position %": r2(raw_metrics.get('Mom Position')),
                            "Mom Slope %": r2(raw_metrics.get('Mom Slope')), "RVOL": r2(raw_metrics.get('RVOL')),
                            "RSI": r2(raw_metrics.get('RSI')), "Insider %": r2(raw_metrics.get('Insider %')),
                            "Score (Balanced)": s_bal, "Score (Aggressive)": s_agg,
                            "Score (Defensive)": s_def, "Score (Speculative)": s_spec
                        }
                        new_data_buffer.append(row)
                    
                    time.sleep(0.8) # Premium Speed (75/min)
                    prog.progress((i+1)/len(tickers))
                
                if new_data_buffer:
                    stat.text("Saving to Database...")
                    batch_save_to_sheet(new_data_buffer)
                    st.success(f"Added {len(new_data_buffer)} tickers!"); st.session_state.watchlist_df = get_watchlist_data(); st.rerun()

t1, t2, t3, t4 = st.tabs(["🔍 Analysis", "📈 Watchlist", "📊 Sectors", "🔎 Screener"])

with t1:
    c1, c2 = st.columns([3, 1])
    tick = c1.text_input("Analyze Ticker", "AAPL").upper()
    if tick and key:
        with st.spinner("Fetching Data..."):
            hist, ov, cf, bs = get_alpha_data(tick, key)
            pe_hist = get_historical_pe(tick, key, hist)
        if not hist.empty and ov:
            score, log, raw_metrics, base_scores = calculate_sector_relative_score(ov, cf, bs, hist, active_weights, use_50ma=is_speculative, mode=mode_name, historical_pe_df=pe_hist)
            curr_price = hist['close'].iloc[-1]
            
            st.markdown(f"## {tick} - {ov.get('Name')}")
            k0, k1, k2, k3, k4 = st.columns(5)
            k0.metric("Price", f"${curr_price:.2f}")
            k1.metric("Market Cap", f"${safe_float(ov.get('MarketCapitalization',0))/1e9:.1f} B")
            k2.metric("RSI", f"{raw_metrics.get('RSI', 50):.0f}")
            
            # Handle None P/E Ratio
            pe_ratio = safe_float(ov.get('PERatio', 0))
            k3.metric("P/E", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
            k4.metric("Insider %", f"{raw_metrics.get('Insider %', 0):.1f}%")
            
            col_metrics, col_chart = st.columns([1, 2])
            with col_metrics:
                score_color = "🟢" if score >= 75 else "🟡" if score >= 50 else "🔴"
                st.metric(f"Score ({mode_name})", f"{score_color} {score}/100")
                st.plotly_chart(plot_score_breakdown(base_scores, mode_name), use_container_width=True)
                st.dataframe(pd.DataFrame(list(log.items()), columns=["Metric", "Details"]), hide_index=True, use_container_width=True)
                if st.button("⭐ Log to Cloud Watchlist"):
                    process_and_log_ticker(tick, key)
                    st.success(f"Logged {tick}!")
                    st.session_state.watchlist_df = get_watchlist_data()
            with col_chart:
                plot_dual_axis(hist, pe_hist, f"{tick}: Price vs Valuation (P/E)", tf_selector("ind"))
        else: st.error("Data Unavailable.")

with t2:
    st.header("Watchlist")
    
    df_wl = st.session_state.watchlist_df
    
    # 1. UPDATE ALL BUTTON
    if st.button("🔄 Update All Stocks Now", type="primary"):
        if not key:
            st.error("⚠️ API Key required")
        elif df_wl.empty:
            st.info("Watchlist is empty")
        else:
            tickers = df_wl['Ticker'].unique().tolist()
            est_time = len(tickers) * 0.8  # Premium speed
            
            st.info(f"⏱️ Updating {len(tickers)} stocks (est. {est_time/60:.1f} minutes)")
            
            prog = st.progress(0)
            stat = st.empty()
            new_data_buffer = []
            
            def r2(val):
                try: 
                    return round(float(val), 2)
                except: 
                    return None
            
            for i, t in enumerate(tickers):
                stat.text(f"Updating {t} ({i+1}/{len(tickers)})...")
                
                hist, ov, cf, bs = get_alpha_data(t, key)
                pe_hist = get_historical_pe(t, key, hist)
                
                if not hist.empty and ov:
                    s_bal, _, raw_metrics, _ = calculate_sector_relative_score(
                        ov, cf, bs, hist, WEIGHTS_BALANCED, 
                        use_50ma=False, mode="Balanced", historical_pe_df=pe_hist
                    )
                    s_agg, _, _, _ = calculate_sector_relative_score(
                        ov, cf, bs, hist, WEIGHTS_AGGRESSIVE, 
                        use_50ma=False, mode="Aggressive", historical_pe_df=pe_hist
                    )
                    s_def, _, _, _ = calculate_sector_relative_score(
                        ov, cf, bs, hist, WEIGHTS_DEFENSIVE, 
                        use_50ma=False, mode="Defensive", historical_pe_df=pe_hist
                    )
                    s_spec, _, _, _ = calculate_sector_relative_score(
                        ov, cf, bs, hist, WEIGHTS_SPECULATIVE, 
                        use_50ma=True, mode="Speculative"
                    )
                    
                    curr_price = hist['close'].iloc[-1]
                    log_date = (datetime.now() - timedelta(hours=6)).strftime("%Y-%m-%d")
                    sector = ov.get('Sector', 'Unknown')
                    mcap = safe_float(ov.get('MarketCapitalization'))
                    
                    row = {
                        "Ticker": t, "Date": log_date, "Sector": sector, 
                        "Price": r2(curr_price), "Market Cap": r2(mcap),
                        "Rev Growth": r2(raw_metrics.get('Rev Growth')), 
                        "Profit Margin": r2(raw_metrics.get('Profit Margin')),
                        "FCF Yield": r2(raw_metrics.get('FCF Yield')), 
                        "ROE": r2(raw_metrics.get('ROE')),
                        "PEG": r2(raw_metrics.get('PEG')), 
                        "Mom Position %": r2(raw_metrics.get('Mom Position')),
                        "Mom Slope %": r2(raw_metrics.get('Mom Slope')), 
                        "RVOL": r2(raw_metrics.get('RVOL')),
                        "RSI": r2(raw_metrics.get('RSI')), 
                        "Insider %": r2(raw_metrics.get('Insider %')),
                        "Score (Balanced)": s_bal, "Score (Aggressive)": s_agg,
                        "Score (Defensive)": s_def, "Score (Speculative)": s_spec
                    }
                    new_data_buffer.append(row)
                
                time.sleep(0.8)  # Premium speed
                prog.progress((i + 1) / len(tickers))
            
            if new_data_buffer:
                stat.text("Saving to database...")
                batch_save_to_sheet(new_data_buffer)
                st.success(f"✅ Updated {len(new_data_buffer)} stocks!")
                st.session_state.watchlist_df = get_watchlist_data()
                st.rerun()
            else:
                st.error("No data retrieved")

    # 2. DISPLAY LOGIC WITH TOP PICKS
    if not df_wl.empty and 'Ticker' in df_wl.columns:
        # User Selection
        view_mode = st.radio(
            "View Score Type", 
            ["Balanced", "Aggressive", "Defensive", "Speculative"], 
            horizontal=True
        )
        score_col = f"Score ({view_mode})"
        
        # Get latest entries
        latest_entries = []
        for t in df_wl['Ticker'].unique():
            row = df_wl[df_wl['Ticker'] == t].sort_values("Date").iloc[-1]
            latest_entries.append(row)
        df_latest = pd.DataFrame(latest_entries)
        
        # Ensure Market Cap exists
        if 'Market Cap' not in df_latest.columns:
            df_latest['Market Cap'] = 0.0
        
        # SPECULATIVE FILTER
        SPECULATIVE_MAX_MCAP = 50_000_000_000  # $50B
        if view_mode == "Speculative":
            original_len = len(df_latest)
            df_latest = df_latest[df_latest['Market Cap'] < SPECULATIVE_MAX_MCAP]
            filtered_count = original_len - len(df_latest)
            if filtered_count > 0:
                st.caption(f"🔍 Filtered {filtered_count} large-cap stocks (> $50B)")

        # Generate Signals
        df_latest['Signal'] = df_latest.apply(
            lambda row: generate_signal(row, score_col), axis=1
        )
        
        # ============================================
        # TOP PICKS IDENTIFICATION (Score > 80)
        # ============================================
        def get_top_2_picks(df, score_column, secondary_metric, category_name):
            """Get top 2 stocks with score > 80 based on secondary metric"""
            qualified = df[df[score_column] > 80].copy()
            
            if len(qualified) == 0:
                return []
            
            # Handle missing values in secondary metric
            qualified = qualified.dropna(subset=[secondary_metric])
            
            if len(qualified) == 0:
                return []
            
            # For metrics where LOWER is better (PEG)
            if secondary_metric in ['PEG']:
                top_picks = qualified.nsmallest(2, secondary_metric)
            else:
                # For metrics where HIGHER is better
                top_picks = qualified.nlargest(2, secondary_metric)
            
            return top_picks['Ticker'].tolist()
        
        # Identify top picks for EACH strategy
        balanced_tops = get_top_2_picks(
            df_latest, 
            'Score (Balanced)', 
            'FCF Yield',  # Best metric for balanced (cash generation)
            'Balanced'
        )
        
        aggressive_tops = get_top_2_picks(
            df_latest, 
            'Score (Aggressive)', 
            'Rev Growth',  # Best metric for growth
            'Aggressive'
        )
        
        defensive_tops = get_top_2_picks(
            df_latest, 
            'Score (Defensive)', 
            'PEG',  # Best metric for value (lower is better)
            'Defensive'
        )
        
        # Rename for display
        df_latest = df_latest.rename(columns={score_col: "Score"})
        
        # Display with highlighting (only buy/sell signals)
        display_cols = ['Ticker', 'Signal', 'Score']
        
        st.dataframe(
            df_latest[display_cols].sort_values("Score", ascending=False).style.applymap(
                lambda x: "background-color: #d4edda; color: black" if "Buy" in str(x) 
                else ("background-color: #f8d7da; color: black" if "Sell" in str(x) else ""), 
                subset=['Signal']
            ), 
            use_container_width=True, 
            hide_index=True
        )
        
        # Legend for top picks
        st.markdown("""
        **🌟 TOP PICKS (Score > 80):**
        - 🟢 **Green** = Top 2 Balanced Stocks (Best FCF Yield)
        - 🔵 **Blue** = Top 2 Growth Stocks (Best Revenue Growth)
        - 🟡 **Gold** = Top 2 Value Stocks (Best PEG Ratio)
        """)
        
        # Show top picks summary
        if balanced_tops or aggressive_tops or defensive_tops:
            st.divider()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.success("**🟢 Balanced Leaders**")
                if balanced_tops:
                    for ticker in balanced_tops:
                        row = df_latest[df_latest['Ticker'] == ticker].iloc[0]
                        fcf = row.get('FCF Yield', 0)
                        st.write(f"• **{ticker}** (FCF: {fcf:.1f}%)")
                else:
                    st.caption("None qualify (Score > 80)")
            
            with col2:
                st.info("**🔵 Growth Leaders**")
                if aggressive_tops:
                    for ticker in aggressive_tops:
                        row = df_latest[df_latest['Ticker'] == ticker].iloc[0]
                        growth = row.get('Rev Growth', 0)
                        st.write(f"• **{ticker}** (Growth: {growth:.1f}%)")
                else:
                    st.caption("None qualify (Score > 80)")
            
            with col3:
                st.warning("**🟡 Value Leaders**")
                if defensive_tops:
                    for ticker in defensive_tops:
                        row = df_latest[df_latest['Ticker'] == ticker].iloc[0]
                        peg = row.get('PEG', 0)
                        st.write(f"• **{ticker}** (PEG: {peg:.2f})")
                else:
                    st.caption("None qualify (Score > 80)")
        
        # 3. INDIVIDUAL TRENDS
        st.divider()
        t_sel = st.selectbox("Select Stock for History", df_wl['Ticker'].unique())
        if t_sel:
            hist_data = df_wl[df_wl['Ticker'] == t_sel].sort_values("Date")
            display_score_col = score_col if score_col in hist_data.columns else 'Score (Balanced)'
            
            st.plotly_chart(
                px.line(
                    hist_data, x='Date', y=display_score_col, 
                    title=f"{t_sel} Historical Score"
                ), 
                use_container_width=True
            )
            
            if st.button(f"🗑️ Delete {t_sel}"):
                remove_ticker_from_sheet(t_sel)
                st.session_state.watchlist_df = get_watchlist_data()
                st.rerun()
    else:
        st.info("📭 Watchlist is empty. Add stocks using the sidebar.")

with t3:
    st.header("🌊 Sector Flow Intelligence")
    
    # --- CONFIGURATION ---
    col_config = st.columns([2, 2, 1])
    with col_config[0]:
        selected_sectors = st.multiselect(
            "Select Sectors", 
            list(SECTOR_ETFS.keys())[:-1], 
            default=list(SECTOR_ETFS.keys())[:-1]
        )
    with col_config[1]:
        timeframes = st.multiselect(
            "Analysis Periods",
            ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"],
            default=["1 Month", "3 Months", "6 Months"]
        )
    with col_config[2]:
        include_volume = st.checkbox("Volume-Weighted", value=True)
    
    if st.button("🔄 Analyze Market Rotation", type="primary"):
        # Timeframe mapping
        tf_map = {
            "1 Week": 5, "1 Month": 22, "3 Months": 63,
            "6 Months": 126, "1 Year": 252
        }
        
        with st.spinner("Fetching sector data..."):
            # Get SPY baseline
            spy_hist, _, _, _ = get_alpha_data("SPY", key)
            
            if not spy_hist.empty:
                # --- DATA COLLECTION ---
                sector_data = {}
                prog = st.progress(0)
                status = st.empty()
                
                for i, sec in enumerate(selected_sectors):
                    ticker = SECTOR_ETFS[sec]
                    status.text(f"Loading {sec}...")
                    hist, _, _, _ = get_alpha_data(ticker, key)
                    
                    if not hist.empty:
                        sector_data[sec] = hist
                    
                    prog.progress((i + 1) / len(selected_sectors))
                    time.sleep(0.8 if is_premium else 12)
                
                status.empty()
                prog.empty()
                
                # --- ANALYSIS TABS ---
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Performance Heatmap",
                    "🎯 Rotation Matrix",
                    "📈 Relative Strength",
                    "💹 Momentum Signals"
                ])
                
                # ============================================
                # TAB 1: MULTI-TIMEFRAME HEATMAP
                # ============================================
                with tab1:
                    st.subheader("Multi-Timeframe Performance vs SPY")
                    
                    heatmap_data = []
                    for sec in selected_sectors:
                        if sec not in sector_data:
                            continue
                        
                        row_data = {"Sector": sec}
                        hist = sector_data[sec]
                        
                        for tf_name in timeframes:
                            days = tf_map[tf_name]
                            
                            if len(hist) < days:
                                row_data[tf_name] = None
                                continue
                            
                            # Calculate relative return
                            sec_start = hist['close'].iloc[-days]
                            sec_end = hist['close'].iloc[-1]
                            sec_ret = ((sec_end / sec_start) - 1) * 100
                            
                            spy_start = spy_hist['close'].iloc[-days]
                            spy_end = spy_hist['close'].iloc[-1]
                            spy_ret = ((spy_end / spy_start) - 1) * 100
                            
                            rel_perf = sec_ret - spy_ret
                            
                            # Volume-weighted if enabled
                            if include_volume and 'volume' in hist.columns:
                                try:
                                    recent_vol = hist['volume'].iloc[-5:].mean()
                                    avg_vol = hist['volume'].iloc[-days:].mean()
                                    vol_factor = recent_vol / avg_vol if avg_vol > 0 else 1
                                    rel_perf = rel_perf * vol_factor
                                except:
                                    pass
                            
                            row_data[tf_name] = rel_perf
                        
                        heatmap_data.append(row_data)
                    
                    df_heatmap = pd.DataFrame(heatmap_data)
                    
                    if not df_heatmap.empty:
                        # Create heatmap
                        fig_heat = go.Figure(data=go.Heatmap(
                            z=df_heatmap[timeframes].values,
                            x=timeframes,
                            y=df_heatmap['Sector'],
                            colorscale='RdYlGn',
                            zmid=0,
                            text=df_heatmap[timeframes].round(1).values,
                            texttemplate='%{text}%',
                            textfont={"size": 10},
                            colorbar=dict(title="Relative<br>Performance")
                        ))
                        
                        fig_heat.update_layout(
                            title="Sector Outperformance vs SPY (%)",
                            height=400,
                            xaxis_title="Time Period",
                            yaxis_title="Sector"
                        )
                        
                        st.plotly_chart(fig_heat, use_container_width=True)
                        
                        # Rankings
                        st.markdown("### 📊 Current Rankings")
                        if "1 Month" in df_heatmap.columns:
                            ranked = df_heatmap.sort_values("1 Month", ascending=False)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "🥇 Top Performer",
                                    ranked.iloc[0]['Sector'],
                                    f"+{ranked.iloc[0]['1 Month']:.1f}% vs SPY"
                                )
                            with col2:
                                st.metric(
                                    "⚖️ Middle Pack",
                                    ranked.iloc[len(ranked)//2]['Sector'],
                                    f"{ranked.iloc[len(ranked)//2]['1 Month']:+.1f}% vs SPY"
                                )
                            with col3:
                                st.metric(
                                    "🥉 Laggard",
                                    ranked.iloc[-1]['Sector'],
                                    f"{ranked.iloc[-1]['1 Month']:.1f}% vs SPY"
                                )
                
                # ============================================
                # TAB 2: ROTATION MATRIX (QUADRANTS)
                # ============================================
                with tab2:
                    st.subheader("Sector Rotation Matrix")
                    st.caption("Identifies sectors transitioning between leadership phases")
                    
                    matrix_data = []
                    for sec in selected_sectors:
                        if sec not in sector_data:
                            continue
                        
                        hist = sector_data[sec]
                        
                        if len(hist) < 63:
                            continue
                        
                        # X-axis: 3-month momentum TREND (slope)
                        prices_3m = hist['close'].iloc[-63:]
                        spy_3m = spy_hist['close'].iloc[-63:]
                        
                        rel_ratio = prices_3m / spy_3m
                        rel_ratio_norm = rel_ratio / rel_ratio.iloc[0]
                        
                        # Calculate linear regression slope
                        x = np.arange(len(rel_ratio_norm))
                        y = rel_ratio_norm.values
                        slope = np.polyfit(x, y, 1)[0] * 100  # Trend direction
                        
                        # Y-axis: 1-month momentum (recent strength)
                        sec_ret_1m = ((hist['close'].iloc[-1] / hist['close'].iloc[-22]) - 1) * 100
                        spy_ret_1m = ((spy_hist['close'].iloc[-1] / spy_hist['close'].iloc[-22]) - 1) * 100
                        recent_momentum = sec_ret_1m - spy_ret_1m
                        
                        matrix_data.append({
                            "Sector": sec,
                            "Trend": slope,
                            "Momentum": recent_momentum
                        })
                    
                    df_matrix = pd.DataFrame(matrix_data)
                    
                    if not df_matrix.empty:
                        # Quadrant classification
                        def classify_quadrant(row):
                            if row['Momentum'] > 0 and row['Trend'] > 0:
                                return "🟢 Leading"
                            elif row['Momentum'] > 0 and row['Trend'] <= 0:
                                return "🟡 Improving"
                            elif row['Momentum'] <= 0 and row['Trend'] > 0:
                                return "🟠 Weakening"
                            else:
                                return "🔴 Lagging"
                        
                        df_matrix['Quadrant'] = df_matrix.apply(classify_quadrant, axis=1)
                        
                        # Scatter plot
                        fig_matrix = px.scatter(
                            df_matrix,
                            x='Trend',
                            y='Momentum',
                            text='Sector',
                            color='Quadrant',
                            size=[20]*len(df_matrix),
                            color_discrete_map={
                                "🟢 Leading": "#00CC66",
                                "🟡 Improving": "#FFD700",
                                "🟠 Weakening": "#FF8C00",
                                "🔴 Lagging": "#FF4444"
                            },
                            title="Sector Rotation Quadrants"
                        )
                        
                        fig_matrix.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig_matrix.add_vline(x=0, line_dash="dash", line_color="gray")
                        
                        fig_matrix.update_traces(textposition='top center')
                        fig_matrix.update_layout(height=500, showlegend=True)
                        
                        st.plotly_chart(fig_matrix, use_container_width=True)
                        
                        # Actionable insights
                        leading = df_matrix[df_matrix['Quadrant'] == "🟢 Leading"]
                        improving = df_matrix[df_matrix['Quadrant'] == "🟡 Improving"]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.success("**🎯 BUY SIGNALS (Leading Sectors)**")
                            if not leading.empty:
                                for _, row in leading.iterrows():
                                    st.write(f"• {row['Sector']}")
                            else:
                                st.caption("None currently")
                        
                        with col2:
                            st.info("**👀 WATCH (Improving Sectors)**")
                            if not improving.empty:
                                for _, row in improving.iterrows():
                                    st.write(f"• {row['Sector']}")
                            else:
                                st.caption("None currently")
                
                # ============================================
                # TAB 3: RELATIVE STRENGTH CHART
                # ============================================
                with tab3:
                    st.subheader("Relative Strength vs SPY (3 Months)")
                    
                    df_rel = pd.DataFrame()
                    cutoff = spy_hist.index[-1] - timedelta(days=63)
                    spy_sub = spy_hist[spy_hist.index >= cutoff]['close']
                    
                    for sec in selected_sectors:
                        if sec not in sector_data:
                            continue
                        
                        sec_sub = sector_data[sec][sector_data[sec].index >= cutoff]['close']
                        comb = pd.concat([sec_sub, spy_sub], axis=1).dropna()
                        
                        if len(comb) > 0:
                            rel = comb.iloc[:, 0] / comb.iloc[:, 1]
                            df_rel[sec] = (rel / rel.iloc[0] - 1) * 100
                    
                    if not df_rel.empty:
                        fig_rel = px.line(
                            df_rel,
                            title="Relative Performance vs SPY (%)",
                            labels={"value": "Relative Return (%)", "index": "Date"}
                        )
                        fig_rel.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
                        fig_rel.update_layout(height=500, hovermode='x unified')
                        
                        st.plotly_chart(fig_rel, use_container_width=True)
                
                # ============================================
                # TAB 4: MOMENTUM SIGNALS
                # ============================================
                with tab4:
                    st.subheader("Technical Momentum Signals")
                    
                    signal_data = []
                    for sec in selected_sectors:
                        if sec not in sector_data:
                            continue
                        
                        hist = sector_data[sec]
                        
                        if len(hist) < 200:
                            continue
                        
                        curr_price = hist['close'].iloc[-1]
                        ma50 = hist['close'].rolling(50).mean().iloc[-1]
                        ma200 = hist['close'].rolling(200).mean().iloc[-1]
                        
                        # RSI
                        rsi = calculate_rsi(hist['close'])
                        
                        # Relative Volume
                        try:
                            rvol = hist['volume'].iloc[-5:].mean() / hist['volume'].iloc[-20:].mean()
                        except:
                            rvol = 1.0
                        
                        # Signal generation
                        signals = []
                        if curr_price > ma50 > ma200:
                            signals.append("🟢 Bullish Trend")
                        elif curr_price < ma50 < ma200:
                            signals.append("🔴 Bearish Trend")
                        
                        if rsi < 30:
                            signals.append("⚡ Oversold")
                        elif rsi > 70:
                            signals.append("⚠️ Overbought")
                        
                        if rvol > 1.5:
                            signals.append("📢 High Volume")
                        
                        signal_data.append({
                            "Sector": sec,
                            "Price vs MA50": f"{((curr_price/ma50 - 1)*100):+.1f}%",
                            "Price vs MA200": f"{((curr_price/ma200 - 1)*100):+.1f}%",
                            "RSI": f"{rsi:.0f}",
                            "RVOL": f"{rvol:.2f}",
                            "Signals": " | ".join(signals) if signals else "Neutral"
                        })
                    
                    df_signals = pd.DataFrame(signal_data)
                    
                    if not df_signals.empty:
                        st.dataframe(
                            df_signals,
                            use_container_width=True,
                            hide_index=True
                        )
            else:
                st.error("Unable to fetch SPY data for baseline comparison")

with t4:
    st.header("🔎 Dynamic Screener")
    
    df_wl = st.session_state.watchlist_df
    
    if df_wl.empty or 'Ticker' not in df_wl.columns:
        st.info("📭 Watchlist is empty. Add stocks using the sidebar.")
    elif not df_wl.empty:
        # 1. Select Score Type
        screen_mode = st.selectbox("Score Type", ["Balanced", "Aggressive", "Defensive", "Speculative"])
        target_score_col = f"Score ({screen_mode})"
        
        # 2. Select Metrics
        available_metrics = ['Rev Growth', 'Profit Margin', 'FCF Yield', 'ROE', 'PEG', 'Mom Slope %', 'RVOL', 'RSI', 'Insider %']
        selected_metrics = st.multiselect("Filter By Metrics", available_metrics, default=['Rev Growth', 'PEG'])
        
        # 3. Dynamic Sliders
        filters = {}
        filters['min_score'] = st.slider(f"Min {screen_mode} Score", 0, 100, 60)
        
        cols = st.columns(len(selected_metrics)) if selected_metrics else []
        for i, metric in enumerate(selected_metrics):
            with cols[i]:
                # Heuristic ranges for sliders based on metric name
                if 'PEG' in metric:
                    filters[metric] = st.slider(f"Max {metric}", 0.0, 5.0, 2.0)
                elif 'RSI' in metric:
                    filters[metric] = st.slider(f"Max {metric}", 0, 100, 70)
                else:
                    filters[metric] = st.slider(f"Min {metric}", -50, 100, 10)
        
        # Logic
        latest_entries = []
        for t in df_wl['Ticker'].unique():
            latest_entries.append(df_wl[df_wl['Ticker'] == t].sort_values("Date").iloc[-1])
        df_screen = pd.DataFrame(latest_entries)
        
        # Apply Score Filter
        mask = (df_screen[target_score_col] >= filters['min_score'])
        
        # Apply Metric Filters
        for metric in selected_metrics:
            if metric in df_screen.columns:
                if 'PEG' in metric or 'RSI' in metric: # Max filters
                    mask = mask & (df_screen[metric] <= filters[metric])
                else: # Min filters
                    mask = mask & (df_screen[metric] >= filters[metric])
        
        filtered = df_screen[mask]
        
        # Clean Output
        if not filtered.empty:
            # Calculate signal for this view
            filtered['Signal'] = filtered.apply(lambda row: generate_signal(row, target_score_col), axis=1)
            filtered = filtered.rename(columns={target_score_col: "Score"})
            
            out_cols = ['Ticker', 'Signal', 'Score'] + selected_metrics
            st.dataframe(filtered[out_cols].style.background_gradient(subset=['Score']), use_container_width=True, hide_index=True)
        else: st.info("No matches found.")
    else: st.info("Watchlist empty.")
