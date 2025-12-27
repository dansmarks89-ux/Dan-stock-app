import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Market Analyzer (Diagnostic Mode)", layout="wide")

# Get API Key
try:
    API_KEY = st.secrets["FMP_KEY"]
except:
    st.error("⚠️ API Key not found. Please set FMP_KEY in Streamlit Secrets.")
    st.stop()

BASE_URL = "https://financialmodelingprep.com/api/v3"

# ==========================================
# 2. DIAGNOSTIC CONNECTION TEST
# ==========================================
st.title("🛠️ Diagnostic Mode: Connection Test")

st.info(f"Testing connection with Key ending in: ...{str(API_KEY)[-4:]}")

# TEST 1: Basic Profile (Cheapest endpoint)
st.write("**Test 1: Fetching Company Profile (AAPL)...**")
try:
    test_url = f"{BASE_URL}/profile/AAPL?apikey={API_KEY}"
    test_response = requests.get(test_url)
    st.write(f"Status Code: {test_response.status_code}")
    if test_response.status_code == 200:
        data = test_response.json()
        if isinstance(data, list) and len(data) > 0:
            st.success(f"✅ Connection Success! Found: {data[0].get('companyName')}")
        elif "Error Message" in data:
            st.error(f"❌ API Error: {data['Error Message']}")
        else:
            st.warning(f"⚠️ Unexpected Response: {data}")
    else:
        st.error("❌ Failed to connect.")
except Exception as e:
    st.error(f"❌ System Error: {e}")

st.markdown("---")

# ==========================================
# 3. ROBUST DATA FETCHING (The Fix)
# ==========================================

@st.cache_data(ttl=3600)
def get_price_data(ticker):
    # STRATEGY A: Standard Full History
    url_a = f"{BASE_URL}/historical-price-full/{ticker}?apikey={API_KEY}"
    
    # STRATEGY B: Intraday 4Hour (Fallback if A fails)
    url_b = f"{BASE_URL}/historical-chart/4hour/{ticker}?apikey={API_KEY}"
    
    try:
        # Try Strategy A
        resp = requests.get(url_a)
        data = resp.json()
        
        # Check for FMP specific errors
        if "Error Message" in data:
            st.warning(f"Strategy A failed for {ticker}: {data['Error Message']}")
            # Try Strategy B
            resp = requests.get(url_b)
            data = resp.json()
            
        if "historical" in data: # Format A
            df = pd.DataFrame(data['historical'])
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date').set_index('date')['adjClose']
            
        elif isinstance(data, list) and len(data) > 0 and 'date' in data[0]: # Format B
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date').set_index('date')['close']
            
        return None
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return None

# ==========================================
# 4. APP INTERFACE
# ==========================================

st.subheader("Market Trend Check")
ticker = st.text_input("Enter Ticker to Test Chart", "NVDA").upper()

if st.button("Get Chart"):
    with st.spinner("Fetching data..."):
        prices = get_price_data(ticker)
        
        if prices is not None and not prices.empty:
            st.success(f"✅ Found {len(prices)} data points for {ticker}")
            
            # Simple Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prices.index, y=prices, mode='lines', name=ticker))
            fig.update_layout(title=f"{ticker} Price History", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show Raw Data Head (For confirmation)
            with st.expander("View Raw Data"):
                st.dataframe(prices.head())
        else:
            st.error("❌ No data found. See error messages above.")
