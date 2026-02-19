import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import requests
import base64
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Wealth Terminal: Folder-Sync", layout="wide")

# Fetch Secrets
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_REPO = st.secrets["GITHUB_REPO"]
# Updated path to store in the 'data' folder
FILE_PATH = "data/master_portfolio.csv" 

# --- UTILS ---
def clean_numeric(val):
    if val is None or pd.isna(val) or str(val).strip() == '': return 0.0
    s = str(val).strip().replace('$', '').replace(',', '')
    if '(' in s and ')' in s: s = '-' + s.replace('(', '').replace(')', '')
    try: return float(s)
    except: return 0.0

def get_ibkr_section(df, section_name):
    rows = df[df.iloc[:, 0].str.contains(section_name, na=False, case=False)]
    h_row = rows[rows.iloc[:, 1] == 'Header']
    d_rows = rows[rows.iloc[:, 1] == 'Data']
    if not h_row.empty and not d_rows.empty:
        cols = [c for c in h_row.iloc[0, 2:].tolist() if c]
        data = d_rows.iloc[:, 2:2+len(cols)]
        data.columns = cols
        return data
    return pd.DataFrame()

# --- GITHUB API HELPERS ---
def push_to_github(df):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    
    res = requests.get(url, headers=headers)
    sha = res.json().get('sha') if res.status_code == 200 else None
    
    content = base64.b64encode(df.to_csv(index=False).encode()).decode()
    data = {
        "message": f"Sync Portfolio to data folder: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "content": content,
        "branch": "main"
    }
    if sha: data["sha"] = sha
    
    put_res = requests.put(url, headers=headers, json=data)
    return put_res.status_code in [200, 201]

# --- THE APP ---
st.title("🏦 Wealth Terminal: Data Folder Edition")

# 1. LOAD FROM GITHUB
if 'master_data' not in st.session_state:
    raw_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{FILE_PATH}?v={datetime.now().timestamp()}"
    try:
        gh_df = pd.read_csv(raw_url)
        st.session_state['master_data'] = gh_df
        st.session_state['last_sync'] = "GitHub Cache"
    except:
        st.session_state['master_data'] = None

# 2. SIDEBAR SYNC
with st.sidebar:
    st.header("⚙️ Data Lifecycle")
    if st.button("🚀 Sync GSheets ➔ /data/ Folder"):
        with st.status("Harmonizing Data...", expanded=True) as status:
            conn = st.connection("gsheets", type=GSheetsConnection)
            trades_list = []
            
            for t in ["FY24", "FY25", "FY26"]:
                st.write(f"Scanning {t}...")
                raw = conn.read(worksheet=t, ttl=0)
                
                # Extract Trades
                df_t = get_ibkr_section(raw, 'Trades')
                if not df_t.empty:
                    # Capture the Realized P/L and Category to handle the top-line split
                    df_t['FY_Source'] = t
                    trades_list.append(df_t)
            
            if trades_list:
                master = pd.concat(trades_list).reset_index(drop=True)
                
                # Standard Cleaning
                master['Qty_v'] = master['Quantity'].apply(clean_numeric)
                master['Prc_v'] = master['T. Price'].apply(clean_numeric)
                # Ensure realized P/L is captured for the top-line dashboard
                if 'Realized P/L' in master.columns:
                    master['Realized_PL'] = master['Realized P/L'].apply(clean_numeric)
                
                master['Date_v'] = pd.to_datetime(master['Date/Time'].str.split(',').str[0]).dt.strftime('%Y-%m-%d')
                
                # Stock Split Adjustment (Automated for master file)
                for tkr, dt in [('NVDA', '2024-06-10'), ('SMCI', '2024-10-01')]:
                    mask = (master['Symbol'] == tkr) & (master['Date_v'] < dt)
                    master.loc[mask, 'Qty_v'] *= 10
                    master.loc[mask, 'Prc_v'] /= 10
                
                if push_to_github(master):
                    status.update(label="GitHub /data/ Folder Updated!", state="complete")
                    st.session_state['master_data'] = master
                    st.rerun()
            else:
                st.error("No Trades found!")

# 3. DASHBOARD (Stage 2)
if st.session_state.get('master_data') is not None:
    df = st.session_state['master_data']
    
    # CALCULATE METRICS
    # Group by FY for the selector
    fy_options = sorted(df['FY_Source'].unique())
    sel_fy = st.selectbox("View Financial Year Summary", fy_options, index=len(fy_options)-1)
    
    fy_total_pl = df[df['FY_Source'] == sel_fy]['Realized_PL'].sum() if 'Realized_PL' in df.columns else 0.0
    
    m1, m2 = st.columns(2)
    m1.metric(f"Net Realized P/L ({sel_fy})", f"${fy_total_pl:,.2f}")
    m2.metric("Data Status", "Synced with GitHub")

    # FIFO HOLDINGS 
    holdings = []
    for sym in df['Symbol'].unique():
        sym_df = df[df['Symbol'] == sym].sort_values('Date_v')
        q_net = sym_df['Qty_v'].sum()
        if q_net > 0.01:
            avg_c = sym_df[sym_df['Qty_v'] > 0]['Prc_v'].mean()
            holdings.append({'Ticker': sym, 'Units': q_net, 'Avg Cost': avg_c})
    
    df_h = pd.DataFrame(holdings)
    df_h.index = range(1, len(df_h) + 1)

    st.subheader(f"Portfolio Holdings (as of {datetime.now().strftime('%d %b %Y')})")
    st.dataframe(df_h.style.format({"Units": "{:.2f}", "Avg Cost": "${:.2f}"}), use_container_width=True)

    # FIFO CALCULATOR
    st.divider()
    st.header("🧮 FIFO Selling Calculator")
    ca, cb = st.columns([1, 2])
    s_pick = ca.selectbox("Select Ticker", df_h['Ticker'])
    row = df_h[df_h['Ticker'] == s_pick].iloc[0]
    
    mode = ca.radio("Amount Type", ["Units", "Percentage"])
    q_sell = cb.slider("Amount", 0.0, float(row['Units'])) if mode == "Units" else row['Units'] * (cb.slider("%", 0, 100, 25)/100)
    target = cb.number_input("Target Profit %", value=110.0)
    
    target_price = row['Avg Cost'] * (target/100)
    st.success(f"To hit {target}% profit: Sell at **${target_price:,.2f}**")
    st.info(f"Residual Position: {row['Units'] - q_sell:.2f} units at ${row['Avg Cost']:,.2f}")

else:
    st.info("👋 Welcome! Press the Sync button to create the /data/ folder and Master CSV in GitHub.")
