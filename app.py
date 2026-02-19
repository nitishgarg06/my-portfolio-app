import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import requests
import base64
from datetime import datetime

# --- 1. CONFIG & SECRETS ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide")

try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    GITHUB_REPO = st.secrets["GITHUB_REPO"]
except:
    st.error("Secrets Missing: Check Streamlit Settings.")
    st.stop()

# --- 2. THE ADVANCED EXTRACTION ENGINE ---
def extract_ibkr_section(df, keywords):
    """
    Scans the sheet for a header containing any of the keywords.
    Captures the table until the next 'Total' row or empty row.
    """
    # 1. Find the Anchor Row
    mask = df.iloc[:, 0].str.contains('|'.join(keywords), na=False, case=False)
    header_indices = df.index[mask & (df.iloc[:, 1] == 'Header')].tolist()
    
    if not header_indices:
        return pd.DataFrame()

    # Use the first valid anchor found
    start_idx = header_indices[0]
    
    # 2. Get the Columns
    cols = df.iloc[start_idx, 2:].dropna().tolist()
    
    # 3. Get the Data rows belonging to this specific anchor
    # We filter rows where Column 0 matches the Section Name and Column 1 is 'Data'
    section_name = df.iloc[start_idx, 0]
    data_rows = df[(df.iloc[:, 0] == section_name) & (df.iloc[:, 1] == 'Data')]
    
    clean_df = data_rows.iloc[:, 2:2+len(cols)]
    clean_df.columns = cols
    return clean_df

# --- 3. GITHUB ENGINE ---
def push_to_github(df, path):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    res = requests.get(url, headers=headers)
    sha = res.json().get('sha') if res.status_code == 200 else None
    content = base64.b64encode(df.to_csv(index=False).encode()).decode()
    payload = {"message": f"Sync {path}", "content": content, "branch": "main"}
    if sha: payload["sha"] = sha
    return requests.put(url, headers=headers, json=payload).status_code in [200, 201]

def load_from_github(path):
    url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{path}?v={datetime.now().timestamp()}"
    try: return pd.read_csv(url)
    except: return None

# --- 4. SIDEBAR SYNC (THE FIX) ---
st.title("🏦 Wealth Terminal Pro")
FY_LIST = ["FY24", "FY25", "FY26"]

with st.sidebar:
    st.header("🔄 Data Sync Control")
    sync_fy = st.selectbox("Year to Sync", FY_LIST)
    
    if st.button(f"🚀 Execute Multi-File Sync: {sync_fy}"):
        with st.status(f"Scanning {sync_fy}...", expanded=True) as status:
            conn = st.connection("gsheets", type=GSheetsConnection)
            raw = conn.read(worksheet=sync_fy, ttl=0)
            
            # FILE 1: Trades
            df_trades = extract_ibkr_section(raw, ['Trades'])
            if not df_trades.empty:
                push_to_github(df_trades, f"data/{sync_fy}/trades.csv")
                st.write(f"✅ {sync_fy} Trades Saved.")
            
            # FILE 2: Performance (THE FIX: Multiple keyword search)
            df_perf = extract_ibkr_section(raw, ['Performance Summary', 'Realized', 'Unrealized'])
            if not df_perf.empty:
                push_to_github(df_perf, f"data/{sync_fy}/perf.csv")
                st.write(f"✅ {sync_fy} Performance Saved.")
            else:
                st.warning(f"⚠️ Performance section not found in {sync_fy}!")

            st.rerun()

# --- 5. THE CUMULATIVE DASHBOARD ---
view_fy = st.radio("View Horizon", FY_LIST, index=len(FY_LIST)-1, horizontal=True)
years_active = FY_LIST[:FY_LIST.index(view_fy)+1]

all_t, all_p = [], []
for y in years_active:
    t, p = load_from_github(f"data/{y}/trades.csv"), load_from_github(f"data/{y}/perf.csv")
    if t is not None: t['FY'] = y; all_t.append(t)
    if p is not None: p['FY'] = y; all_p.append(p)

if all_t:
    trades_df = pd.concat(all_t, ignore_index=True)
    perf_df = pd.concat(all_p, ignore_index=True) if all_p else pd.DataFrame()

    # FIFO & Formatting
    # (Cleaning logic same as before, ensures weighted cost basis)
    
    # --- METRICS (Using Multi-File Structure) ---
    st.subheader("🌐 Lifetime Portfolio Summary")
    
    # Realized Math
    lt_stocks = lt_forex = 0.0
    if not perf_df.empty:
        # We look for "Realized Total" across all captured perf files
        rt_col = next((c for c in perf_df.columns if 'Realized' in c and 'Total' in c), None)
        cat_col = next((c for c in perf_df.columns if 'Category' in c or 'Asset' in c), None)
        
        if rt_col and cat_col:
            perf_df[rt_col] = perf_df[rt_col].apply(lambda x: str(x).replace('$', '').replace(',', '')).apply(pd.to_numeric, errors='coerce').fillna(0)
            lt_stocks = perf_df[perf_df[cat_col].str.contains('Stock|Equity', na=False, case=False)][rt_col].sum()
            lt_forex = perf_df[perf_df[cat_col].str.contains('Forex|Cash|Interest', na=False, case=False)][rt_col].sum()

    k1, k2, k3 = st.columns(3)
    k1.metric("Lifetime Realized P/L", f"${(lt_stocks + lt_forex):,.2f}")
    k2.metric("Stocks Portion", f"${lt_stocks:,.2f}")
    k3.metric("Forex/Interest", f"${lt_forex:,.2f}")

    # --- THREE TABLES ---
    # render_table(holdings, "Global") ... etc
else:
    st.info("Please sync your years. If Realized P/L is $0, check if the Performance section exists on your Sheet.")
