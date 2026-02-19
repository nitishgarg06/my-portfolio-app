import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import requests
import base64
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide")

GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_REPO = st.secrets["GITHUB_REPO"]
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

def push_to_github(df):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    res = requests.get(url, headers=headers)
    sha = res.json().get('sha') if res.status_code == 200 else None
    
    encoded = base64.b64encode(df.to_csv(index=False).encode()).decode()
    payload = {
        "message": f"Portfolio Sync: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "content": encoded, "branch": "main"
    }
    if sha: payload["sha"] = sha
    
    put_res = requests.put(url, headers=headers, json=payload)
    return put_res.status_code in [200, 201]

# --- APP FLOW ---
st.title("🏦 Wealth Terminal Pro")

# 1. ATTEMPT TO LOAD FROM GITHUB
if 'master_data' not in st.session_state:
    raw_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{FILE_PATH}?v={datetime.now().timestamp()}"
    try:
        st.session_state['master_data'] = pd.read_csv(raw_url)
    except:
        st.session_state['master_data'] = None

# 2. SIDEBAR SYNC
with st.sidebar:
    st.header("⚙️ Data Sync")
    if st.button("🚀 Push GSheets ➔ GitHub Master"):
        with st.status("Harmonizing Data...", expanded=True) as status:
            conn = st.connection("gsheets", type=GSheetsConnection)
            all_trades = []
            
            for t in ["FY24", "FY25", "FY26"]:
                st.write(f"Processing {t}...")
                raw = conn.read(worksheet=t, ttl=0)
                
                # Trades Section
                df_t = get_ibkr_section(raw, 'Trades')
                if not df_t.empty:
                    df_t['FY_Source'] = t
                    # Capture Realized PL & Category
                    for col in ['Realized P/L', 'Asset Category']:
                        if col in df_t.columns:
                            df_t[f'clean_{col}'] = df_t[col]
                    all_trades.append(df_t)
            
            if all_trades:
                master = pd.concat(all_trades).reset_index(drop=True)
                # Final Formatting
                master['Qty_v'] = master['Quantity'].apply(clean_numeric)
                master['Prc_v'] = master['T. Price'].apply(clean_numeric)
                master['Date_v'] = pd.to_datetime(master['Date/Time'].str.split(',').str[0]).dt.strftime('%Y-%m-%d')
                
                # Split Adjustments
                for tkr, dt in [('NVDA', '2024-06-10'), ('SMCI', '2024-10-01')]:
                    mask = (master['Symbol'] == tkr) & (master['Date_v'] < dt)
                    master.loc[mask, 'Qty_v'] *= 10
                    master.loc[mask, 'Prc_v'] /= 10

                if push_to_github(master):
                    status.update(label="GitHub Updated!", state="complete")
                    st.session_state['master_data'] = master
                    st.rerun()
            else:
                st.error("No trades found in Google Sheets.")

# 3. DASHBOARD
if st.session_state.get('master_data') is not None:
    df = st.session_state['master_data']
    
    # Metrics Split
    fy_options = sorted(df['FY_Source'].unique())
    sel_fy = st.selectbox("Financial Year View", fy_options, index=len(fy_options)-1)
    
    fy_df = df[df['FY_Source'] == sel_fy]
    
    # Logic to split Forex vs Stock Realized P/L
    stocks_pl = fy_df[fy_df['clean_Asset Category'].str.contains('Stock', na=False, case=False)]['clean_Realized P/L'].apply(clean_numeric).sum()
    forex_pl = fy_df[fy_df['clean_Asset Category'].str.contains('Forex|Cash', na=False, case=False)]['clean_Realized P/L'].apply(clean_numeric).sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("Net Realized P/L", f"${(stocks_pl + forex_pl):,.2f}")
    m2.metric("Stocks Realized", f"${stocks_pl:,.2f}")
    m3.metric("Forex/Cash Realized", f"${forex_pl:,.2f}")

    # Holdings Table (FIFO)
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

    # Calculator
    st.divider()
    st.header("🧮 FIFO Selling Calculator")
    c1, c2 = st.columns(2)
    s_pick = c1.selectbox("Ticker", df_h['Ticker'])
    row = df_h[df_h['Ticker'] == s_pick].iloc[0]
    target = c2.number_input("Target Profit %", value=110.0)
    st.success(f"Sell at: **${(row['Avg Cost'] * (target/100)):,.2f}** for {target}% profit.")
else:
    st.info("No Master Data on GitHub. Click the sidebar button to Sync.")
