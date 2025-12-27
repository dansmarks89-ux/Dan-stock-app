import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

# ==========================================
# LITE MODE (Free Tier Compatible)
# ==========================================
st.set_page_config(page_title="Market Analyzer (Lite)", layout="wide")

# Hardcoded for reliability during testing
API_KEY = "HO1Gg4eZ38sEt6MhH0SKI7XrhmGjjrX8"
BASE_URL = "https://financialmodelingprep.com/api/v3"

st.title("📉 Market Analyzer (Lite Mode)")

# 1. Input Ticker
ticker = st.text_input("Enter Ticker", "NVDA").upper()

if st.button("Analyze Stock"):
    # 2. Fetch Data (Using the FREE '4hour' endpoint)
    # This gives us ~2 months of data, which is allowed on free plans.
    url = f"{BASE_URL}/historical-chart/4hour/{ticker}?apikey={API_KEY}"
    
    st.write(f"Fetching data for {ticker}...")
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # Check for errors
        if isinstance(data, dict) and "Error Message" in data:
            st.error(f"FMP Error: {data['Error Message']}")
        
        elif isinstance(data, list) and len(data) > 0:
            # Process Data
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Get Current Price
            current_price = df.iloc[-1]['close']
            
            # Display Metric
            st.metric(label=f"{ticker} Price", value=f"${current_price:.2f}")
            
            # Plot Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name=ticker))
            fig.update_layout(title=f"{ticker} (Recent History - 4hr)", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ Success! Data loaded.")
            
        else:
            st.warning("No data found. Ticker might be wrong.")
            
    except Exception as e:
        st.error(f"System Error: {e}")
