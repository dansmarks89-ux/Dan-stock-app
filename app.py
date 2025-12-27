import streamlit as st
import requests

API_KEY = "HO1Gg4eZ38sEt6MhH0SKI7XrhmGjjrX8"
BASE_URL = "https://financialmodelingprep.com/api/v3"

st.title("🔍 Endpoint Auditor")

def check_endpoint(name, url):
    try:
        response = requests.get(url)
        data = response.json()
        
        # Check for Legacy Error
        if isinstance(data, dict) and "Error Message" in data:
            st.error(f"❌ {name}: BLOCKED ({data['Error Message'][:50]}...)")
            return False
            
        # Check for Success (List or Dict)
        if isinstance(data, list) and len(data) > 0:
            st.success(f"✅ {name}: WORKING (Found {len(data)} records)")
            return True
        elif isinstance(data, dict) and 'symbol' in data:
             st.success(f"✅ {name}: WORKING")
             return True
             
        st.warning(f"⚠️ {name}: Empty Response (Not Blocked)")
        return True
    except Exception as e:
        st.error(f"❌ {name}: System Error {e}")
        return False

if st.button("RUN AUDIT"):
    # 1. Check Current Price (Quote)
    check_endpoint("Current Price (Quote)", f"{BASE_URL}/quote/NVDA?apikey={API_KEY}")
    
    # 2. Check Fundamentals (Ratios) - NEEDED FOR SCORING
    check_endpoint("Ratios (P/E, PEG)", f"{BASE_URL}/ratios/NVDA?limit=10&apikey={API_KEY}")
    
    # 3. Check Metrics (ROIC, EV/EBITDA) - NEEDED FOR SCORING
    check_endpoint("Key Metrics", f"{BASE_URL}/key-metrics/NVDA?limit=10&apikey={API_KEY}")
    
    # 4. Check V4 Price (New Alternative?)
    # Sometimes V4 works when V3 fails
    check_endpoint("V4 Price History", f"https://financialmodelingprep.com/api/v4/historical-price-adjusted/NVDA/1/day?apikey={API_KEY}")
