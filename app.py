import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# HARDCODED KEY TEST (Temporary)
# ==========================================
# We are putting the key directly here to rule out "Secrets" issues.
API_KEY = "HO1Gg4eZ38sEt6MhH0SKI7XrhmGjjrX8" 
BASE_URL = "https://financialmodelingprep.com/api/v3"

st.title("🔓 Hardcoded Key Test")

# 1. Print the Key being used (Masked)
st.write(f"Testing Key: {API_KEY[:4]}...{API_KEY[-4:]}")

# 2. Try the simplest possible request (Company Profile)
url = f"{BASE_URL}/profile/AAPL?apikey={API_KEY}"

if st.button("Run Connection Test"):
    try:
        st.write(f"Pinging: {BASE_URL}/profile/AAPL...")
        response = requests.get(url)
        
        # PRINT THE RAW STATUS (Crucial)
        st.write(f"**Status Code:** {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            st.success("✅ SUCCESS! We have data.")
            st.write(f"Company Found: {data[0].get('companyName')}")
            st.write(f"Price: ${data[0].get('price')}")
        else:
            st.error("❌ Connection Rejected.")
            st.write(f"Reason: {response.text}")
            
    except Exception as e:
        st.error(f"❌ System Error: {e}")
