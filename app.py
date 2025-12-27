import streamlit as st
import requests

# HARDCODED KEY (To eliminate all doubt)
API_KEY = "HO1Gg4eZ38sEt6MhH0SKI7XrhmGjjrX8"

st.title("🔌 Bare Metal Connection Test")

# 1. Define the URL (Using the Safe V3 Endpoint)
url = f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={API_KEY}"

st.write(f"**Target:** {url.replace(API_KEY, 'HIDDEN_KEY')}")

if st.button("PING SERVER"):
    try:
        # 2. Fire the Request
        response = requests.get(url)
        
        # 3. Print Results
        st.write(f"**Status Code:** {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            st.success("✅ CONNECTION SUCCESSFUL")
            st.json(data) # This will show the raw data from FMP
        else:
            st.error(f"❌ CONNECTION FAILED: {response.status_code}")
            st.write(response.text)
            
    except Exception as e:
        st.error(f"System Error: {e}")
