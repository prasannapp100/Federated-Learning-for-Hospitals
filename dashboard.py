import streamlit as st
import requests
import subprocess
import sys
import os
from PIL import Image
import io

st.set_page_config(page_title="Federated Learning Dashboard", layout="wide")
st.title("🏥 Federated Learning Monitor")

# Sidebar
st.sidebar.header("Control Panel")
server_url = st.sidebar.text_input("Server URL", "http://127.0.0.1:8000")

if st.sidebar.button("🚀 Run Sequential Training"):
    hospitals = ["Hospital_A", "Hospital_B"]
    
    for h_id in hospitals:
        st.sidebar.info(f"Training {h_id}...")
        # subprocess.run WAITS for the script to finish
        subprocess.run(
            [sys.executable, "hospital_client.py"],
            env={**dict(os.environ), "HOSPITAL_ID": h_id}
        )
        st.sidebar.success(f"{h_id} Uploaded Successfully!")
    
    st.sidebar.balloons()
    st.rerun()

# --- Section 1: Stats ---
def fetch_stats():
    try:
        return requests.get(f"{server_url}/stats", timeout=2).json()
    except: return None

stats = fetch_stats()
if stats:
    col1, col2 = st.columns(2)
    col1.metric("Round", stats['current_round'])
    col2.metric("Connected", f"{stats['hospitals_connected']}/{stats['threshold']}")
    st.progress(stats['hospitals_connected'] / stats['threshold'])
else:
    st.error("Server Offline")

if st.button("🔄 Refresh"): st.rerun()

st.divider()

# --- Section 2: Inference ---
st.header("🔍 Global Model Diagnosis")
file = st.file_uploader("Upload X-Ray", type=["jpg", "png", "jpeg"])
if file:
    st.image(file, width=300)
    if st.button("Predict"):
        res = requests.post(f"{server_url}/predict", files={"file": file.getvalue()}).json()
        st.write(f"**Result:** {res['prediction']} | **Confidence:** {res['confidence']}")