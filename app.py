import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Market Analyzer v3.0", layout="wide")

# HARDCODED KEY (Safety Net)
API_KEY = "HO1Gg4eZ38sEt6MhH0SKI7XrhmGjjrX8"
BASE_URL = "https://financialmodelingprep.com/api/v3"

st.title("🚀 Market Analyzer v3.0 (Cache Buster)")
st.caption("If you see this title, the code update SUCCESSFUL.")

# ==========================================
# 2. NEW DATA ENGINE (Renamed Functions)
# ==========================================

# I renamed this function to 'get_data_v3' so Streamlit CANNOT use the old cache
@st.cache_data(ttl=60) 
def get_data_v3(ticker):
    # We use the 'historical-chart/1day' endpoint which is NOT legacy.
    target_url = f"{BASE_URL}/historical-chart/1day/{ticker}?apikey={API_KEY}"
    
    # DEBUG: Print the URL we are actually calling (to prove it's the new one)
    st.text(f"Pinging: .../historical-chart/1day/{ticker}...")
    
    try:
        response = requests.get(target_url)
        data = response.json()
        
        # Check for the specific "Legacy" error
        if isinstance(data, dict) and "Error Message" in data:
            return None, data['Error Message']
            
        # Success Case
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').set_index('date')
            return df, None
            
        return None, "Empty List Returned"
        
    except Exception as e:
        return None, str(e)

# ==========================================
# 3. APP UI
# ==========================================

ticker = st.text_input("Enter Ticker", "NVDA").upper()

if st.button("Get Data"):
    # Clear cache button inside the app logic
    st.cache_data.clear()
    
    df, error = get_data_v3(ticker)
    
    if df is not None:
        st.success(f"✅ Success! Found {len(df)} days of data.")
        
        # Metric
        curr_price = df.iloc[-1]['close']
        st.metric("Current Price", f"${curr_price:.2f}")
        
        # Chart
        st.line_chart(df['close'])
        
    else:
        st.error("❌ FAILED.")
        st.error(f"Reason: {error}")
        st.info("If the reason is 'Legacy Endpoint', then FMP has blocked the Chart endpoint too.")
