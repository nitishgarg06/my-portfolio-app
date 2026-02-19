import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import requests
import base64
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide")

# Safe Secrets Loading
try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    GITHUB_REPO = st.secrets["GITHUB_REPO"]
except Exception:
    st.error("Missing Secrets: Please ensure GITHUB_TOKEN and GITHUB_REPO are set in Streamlit Cloud.")
    st.stop()

FILE_PATH = "data/master_portfolio.csv"

# --- UTILS ---
def clean_numeric(val):
    if val is None or pd.isna(val) or str(val).strip() == '': return 0.0
    s = str(val).strip().replace('$', '').replace(',', '')
    if '(' in s and ')' in s: s = '-' + s.replace('(', '').replace(')', '')
    try: return float(s)
    except: return 0.0

def get_ibkr_section(df, section_name):
    """Scans the sheet for Header and Data rows belonging to a specific section."""
    rows = df[df.iloc[:, 0].str.contains(section_name, na=False, case=False)]
    h_row = rows[rows.iloc[:, 1] == 'Header']
    d_rows = rows[rows.iloc[:, 1] == 'Data']
    if not h_row.empty and not d_rows.empty:
        cols = [c for c in h_row.iloc[0, 2:].tolist() if c]
        data = d_rows.iloc[:, 2:2+len(cols)]
        data.columns = cols
        return data
    return pd.DataFrame()

# --- GITHUB ENGINE ---
def push_to_github(df):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    
    # 1. Check for existing file SHA
    res = requests.get(url, headers=headers)
    sha = res.json().get('sha') if res.status_code == 200 else None
    
    # 2. Encode Content
    csv_content = df.to_csv(index=False)
    encoded = base64.b64encode(csv_content.encode()).decode()
    
    # 3. Payload
    payload = {
        "message": f"Sync: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "content": encoded,
        "branch": "main"
    }
    if sha: payload["sha"] = sha
    
    put_res = requests.put(url, headers=headers, json=payload)
    return put_res.status_code in [200, 201]

# --- APP FLOW ---
st.title("🏦 Wealth Terminal Pro")

# Load existing data from GitHub on start
if 'master_data' not in st.session_state:
    raw_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{FILE_PATH}?v={datetime.now().timestamp()}"
    try:
        gh_df = pd.read_csv(raw_url)
        st.session_state['master_data'] = gh_df
        st.session_state['last_sync'] = datetime.now().strftime("%H:%M:%S")
    except:
        st.session_state['master_data'] = None

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Data Sync")
    if st.button("🚀 Push GSheets ➔ GitHub"):
        with st.status("Syncing...", expanded=True) as status:
            conn = st.connection("gsheets", type=GSheetsConnection)
            trades_list = []
            
            for t in ["FY24", "FY25", "FY26"]:
                st.write(f"Reading {t}...")
                raw = conn.read(worksheet=t, ttl=0)
                df_t = get_ibkr_section(raw, 'Trades')
                if not df_t.empty:
                    df_t['FY_Source'] = t
                    # Add Realized P/L column for top-line metrics later
                    if 'Realized P/L' in df_t.columns:
                        df_t['Realized_PL_val'] = df_t['Realized P/L'].apply(clean_numeric)
                    trades_list.append(df_t)
            
            if trades_list:
                master = pd.concat(trades_list).reset_index(drop=True)
                master['Qty_val'] = master['Quantity'].apply(clean_numeric)
                master['Prc_val'] = master['T. Price'].apply(clean_numeric)
                master['Date_val'] = pd.to_datetime(master['Date/Time'].str.split(',').str[0]).dt.strftime('%Y-%m-%d')
                
                # Automated Split Adjustments
                for tkr, dt in [('NVDA', '2024-06-10'), ('SMCI', '2024-10-01')]:
                    mask = (master['Symbol'] == tkr) & (master['Date_val'] < dt)
                    master.loc[mask, 'Qty_val'] *= 10
                    master.loc[mask, 'Prc_val'] /= 10
                
                if push_to_github(master):
                    status.update(label="Success! Data stored in GitHub /data folder.", state="complete")
                    st.session_state['master_data'] = master
                    st.rerun()
                else:
                    st.error("GitHub Error: Verify your Repo name and PAT permissions.")

# --- DASHBOARD ---
if st.session_state.get('master_data') is not None:
    df = st.session_state['master_data']
    
    # 1. Performance Overview
    fy_options = sorted(df['FY_Source'].unique())
    sel_fy = st.selectbox("Select Financial Year Summary", fy_options, index=len(fy_options)-1)
    
    fy_pl = df[df['FY_Source'] == sel_fy]['Realized_PL_val'].sum() if 'Realized_PL_val' in df.columns else 0.0
    st.metric(f"Net Realized P/L ({sel_fy})", f"${fy_pl:,.2f}")

    # 2. Holdings (FIFO Aggregate)
    holdings = []
    for sym in df['Symbol'].unique():
        sym_df = df[df['Symbol'] == sym].sort_values('Date_val')
        q_net = sym_df['Qty_val'].sum()
        if q_net > 0.01:
            avg_c = sym_df[sym_df['Qty_val'] > 0]['Prc_val'].mean()
            holdings.append({'Ticker': sym, 'Units': q_net, 'Avg Cost': avg_c})
    
    df_h = pd.DataFrame(holdings)
    df_h.index = range(1, len(df_h) + 1)
    
    st.subheader(f"Portfolio Holdings (as of {datetime.now().strftime('%d %b %Y')})")
    st.dataframe(df_h.style.format({"Units": "{:.2f}", "Avg Cost": "${:.2f}"}), use_container_width=True)

    # 3. FIFO Calculator
    st.divider()
    st.header("🧮 FIFO Selling Calculator")
    c1, c2 = st.columns(2)
    s_pick = c1.selectbox("Ticker", df_h['Ticker'])
    target = c2.number_input("Target Profit %", value=110.0)
    
    row = df_h[df_h['Ticker'] == s_pick].iloc[0]
    st.success(f"Sell at: **${(row['Avg Cost'] * (target/100)):,.2f}** for {target}% profit.")
else:
    st.info("No data found on GitHub. Click the sidebar button to Sync your Google Sheets.")
