import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Yahoo Diagnostic", layout="wide")

st.title("🚑 Yahoo Finance Connection Test")

# 1. Input
ticker = st.text_input("Enter Ticker", "MSFT").upper()

if st.button("Test Connection"):
    st.write(f"Testing connection to Yahoo Finance for: **{ticker}**...")
    
    # TEST 1: Price History (The Basics)
    st.markdown("### Test 1: Price History")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        
        if not hist.empty:
            st.success(f"✅ Price History Found! ({len(hist)} days)")
            st.line_chart(hist['Close'])
        else:
            st.error("❌ Price History is EMPTY. (Yahoo might be blocking IP)")
            st.write("Raw History Output:", hist)
            
    except Exception as e:
        st.error(f"❌ Crash during Price History fetch: {e}")

    # TEST 2: Fundamental Info (The Complex Part)
    st.markdown("### Test 2: Fundamentals (Info)")
    try:
        # Some versions of yfinance fail here if Yahoo changes the API
        info = stock.info
        
        # Check a specific key to verify
        if info and 'forwardPE' in info:
            st.success(f"✅ Fundamentals Found! P/E: {info.get('forwardPE')}")
        elif info:
            st.warning("⚠️ Info returned, but keys might be missing.")
            st.write(info)
        else:
            st.error("❌ Info dictionary is Empty.")
            
    except Exception as e:
        st.error(f"❌ Crash during Info fetch: {e}")
        st.info("💡 Note: If Test 1 works but Test 2 fails, we can switch to 'Fast Info' mode.")
