import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import requests
import base64
from datetime import datetime

# --- 1. DIAGNOSTICS (HELP US FIND THE ERROR) ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide")
st.title("🏦 Wealth Terminal Pro")

with st.expander("🔍 System Diagnostics (Check here for errors)"):
    # Check if Secrets exist
    secrets_found = "GITHUB_TOKEN" in st.secrets and "GITHUB_REPO" in st.secrets
    st.write(f"Secrets Found in Streamlit: **{'✅ Yes' if secrets_found else '❌ No'}**")
    
    if secrets_found:
        st.write(f"Repo Path: `{st.secrets['GITHUB_REPO']}`")
        # Test GitHub Connection
        headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
        test_url = f"https://api.github.com/repos/{st.secrets['GITHUB_REPO']}"
        res = requests.get(test_url, headers=headers)
        if res.status_code == 200:
            st.write("GitHub Connection: **✅ Success**")
        else:
            st.write(f"GitHub Connection: **❌ Failed (Error {res.status_code})**")
            st.info("Tip: If Error 404, check your Repo name or Token 'repo' scope.")

# --- 2. CONFIG & GITHUB ENGINE ---
if not secrets_found:
    st.warning("Please configure GITHUB_TOKEN and GITHUB_REPO in Streamlit Secrets to proceed.")
    st.stop()

GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_REPO = st.secrets["GITHUB_REPO"]
FILE_PATH = "data/master_portfolio.csv"

def push_to_github(df):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    res = requests.get(url, headers=headers)
    sha = res.json().get('sha') if res.status_code == 200 else None
    
    encoded = base64.b64encode(df.to_csv(index=False).encode()).decode()
    payload = {
        "message": f"Sync: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "content": encoded, "branch": "main"
    }
    if sha: payload["sha"] = sha
    
    put_res = requests.put(url, headers=headers, json=payload)
    return put_res.status_code in [200, 201]

# --- 3. APP FLOW ---
# Load from GitHub (The 'Source of Truth')
if 'master_data' not in st.session_state:
    raw_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{FILE_PATH}?v={datetime.now().timestamp()}"
    try:
        st.session_state['master_data'] = pd.read_csv(raw_url)
    except:
        st.session_state['master_data'] = None

# Sidebar Sync
with st.sidebar:
    st.header("⚙️ Data Sync")
    if st.button("🚀 Push GSheets ➔ GitHub"):
        try:
            conn = st.connection("gsheets", type=GSheetsConnection)
            # (Insert previous harmonizing logic here...)
            # For testing, let's just try to push a simple dataframe
            test_df = pd.DataFrame({"Sync_Date": [datetime.now()]})
            if push_to_github(test_df):
                st.success("Successfully pushed to GitHub!")
                st.rerun()
        except Exception as e:
            st.error(f"Sync failed: {e}")

# Render Dashboard if data exists
if st.session_state.get('master_data') is not None:
    st.write("### ✅ Master Data Loaded")
    st.dataframe(st.session_state['master_data'].head())
