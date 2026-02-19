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

def fuzzy_find(df, keywords):
    """Finds a column name that contains any of the keywords."""
    for col in df.columns:
        if any(k.lower() in str(col).lower() for k in keywords):
            return col
    return None

# --- GITHUB ENGINE ---
def push_to_github(df):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    res = requests.get(url, headers=headers)
    sha = res.json().get('sha') if res.status_code == 200 else None
    encoded = base64.b64encode(df.to_csv(index=False).encode()).decode()
    payload = {"message": f"Sync: {datetime.now()}", "content": encoded, "branch": "main"}
    if sha: payload["sha"] = sha
    return requests.put(url, headers=headers, json=payload).status_code in [200, 201]

# --- THE APP ---
st.title("🏦 Wealth Terminal Pro")

# 1. LOAD MASTER DATA
if 'master_data' not in st.session_state:
    raw_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{FILE_PATH}?v={datetime.now().timestamp()}"
    try:
        st.session_state['master_data'] = pd.read_csv(raw_url)
    except:
        st.session_state['master_data'] = None

# 2. SIDEBAR SYNC (Kept for updates)
with st.sidebar:
    if st.button("🚀 Re-Sync GSheets ➔ GitHub"):
        conn = st.connection("gsheets", type=GSheetsConnection)
        all_trades = []
        for t in ["FY24", "FY25", "FY26"]:
            raw = conn.read(worksheet=t, ttl=0)
            rows = raw[raw.iloc[:, 0].str.contains('Trades', na=False, case=False)]
            h_row = rows[rows.iloc[:, 1] == 'Header']
            d_rows = rows[rows.iloc[:, 1] == 'Data']
            if not h_row.empty:
                cols = [c for c in h_row.iloc[0, 2:].tolist() if c]
                data = d_rows.iloc[:, 2:2+len(cols)]
                data.columns = cols
                data['FY_Source'] = t
                all_trades.append(data)
        if all_trades:
            master = pd.concat(all_trades).reset_index(drop=True)
            if push_to_github(master):
                st.session_state['master_data'] = master
                st.rerun()

# 3. DASHBOARD LOGIC
if st.session_state.get('master_data') is not None:
    df = st.session_state['master_data']
    
    # Fuzzy Column Mapping
    col_qty = fuzzy_find(df, ['Quantity', 'Qty'])
    col_prc = fuzzy_find(df, ['T. Price', 'Price'])
    col_sym = fuzzy_find(df, ['Symbol', 'Ticker'])
    col_pl  = fuzzy_find(df, ['Realized P/L', 'Realized Total'])
    col_cat = fuzzy_find(df, ['Asset Category', 'Category'])
    col_dt  = fuzzy_find(df, ['Date/Time', 'Date'])

    if all([col_qty, col_prc, col_sym]):
        # Data Cleaning
        df['Q_clean'] = df[col_qty].apply(clean_numeric)
        df['P_clean'] = df[col_prc].apply(clean_numeric)
        df['PL_clean'] = df[col_pl].apply(clean_numeric) if col_pl else 0.0
        
        # Metrics Split
        fy_list = sorted(df['FY_Source'].unique()) if 'FY_Source' in df.columns else []
        if fy_list:
            sel_fy = st.selectbox("Financial Year", fy_list, index=len(fy_list)-1)
            fy_df = df[df['FY_Source'] == sel_fy]
            
            # Stock vs Forex Logic
            s_pl = fy_df[fy_df[col_cat].str.contains('Stock', na=False, case=False)]['PL_clean'].sum() if col_cat else 0.0
            f_pl = fy_df[fy_df[col_cat].str.contains('Forex|Cash', na=False, case=False)]['PL_clean'].sum() if col_cat else 0.0
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Net Realized P/L", f"${(s_pl + f_pl):,.2f}")
            m2.metric("Stocks", f"${s_pl:,.2f}")
            m3.metric("Forex/Impact", f"${f_pl:,.2f}")

        # FIFO Holdings 
        holdings = []
        for sym in df[col_sym].unique():
            sym_df = df[df[col_sym] == sym]
            q_net = sym_df['Q_clean'].sum()
            if q_net > 0.01:
                avg_c = sym_df[sym_df['Q_clean'] > 0]['P_clean'].mean()
                holdings.append({'Ticker': sym, 'Units': q_net, 'Avg Cost': avg_c})
        
        if holdings:
            df_h = pd.DataFrame(holdings)
            st.subheader(f"Current Holdings (as of {datetime.now().strftime('%d %b %Y')})")
            st.dataframe(df_h.style.format({"Units": "{:.2f}", "Avg Cost": "${:.2f}"}), use_container_width=True)
        else:
            st.warning("No active holdings found in the Master File.")

    # --- DEBUGGER (Visible if info is missing) ---
    with st.expander("🛠️ Master File Inspector"):
        st.write("Current Columns in GitHub File:", df.columns.tolist())
        st.write("Preview of Data:", df.head())
else:
    st.info("No Master Data found. Please use the sidebar to Sync.")
