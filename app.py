import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.set_page_config(layout="wide", page_title="Wealth Terminal Pro")
st.title("🏦 Wealth Terminal Pro")

# --- UTILITIES ---
def cl(x): return pd.to_numeric(str(x).replace('$','').replace(',','').replace('(','-').replace(')',''), errors='coerce').fillna(0)

# --- DATA EXTRACTION ---
# We use a direct read for validation to ensure the numbers appear immediately
FY_LIST = ["FY24", "FY25", "FY26"]
view_fy = st.sidebar.selectbox("Select Financial Year", FY_LIST, index=2)

@st.cache_data(ttl=0)
def fetch_topline_data(fy):
    conn = st.connection("gsheets", type=GSheetsConnection)
    df = conn.read(worksheet=fy, ttl=0)
    
    # Extract Trades
    t_rows = df[df.iloc[:, 0].str.contains('Trades', na=False, case=False)]
    th = t_rows[t_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
    td = t_rows[t_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(th)]
    td.columns = th
    
    # Extract Performance
    p_rows = df[df.iloc[:, 0].str.contains('Performance Summary|Realized', na=False, case=False)]
    ph = p_rows[p_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
    pd_data = p_rows[p_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(ph)]
    pd_data.columns = ph
    
    return td, pd_data

try:
    # Load CURRENT FY for FY Metrics
    curr_trades, curr_perf = fetch_topline_data(view_fy)
    
    # Load ALL for Lifetime Metrics
    all_t = []
    all_p = []
    for y in FY_LIST[:FY_LIST.index(view_fy)+1]:
        t, p = fetch_topline_data(y)
        all_t.append(t)
        all_p.append(p)
    df_t_all = pd.concat(all_t)
    df_p_all = pd.concat(all_p)

    # --- TOPLINE CALCULATION ---
    # 1. FY INVESTMENT & P/L
    fy_inv = (curr_trades[curr_trades['Qty'].apply(cl) > 0]['Qty'].apply(cl) * curr_trades[curr_trades['Qty'].apply(cl) > 0]['Price'].apply(cl)).sum()
    fy_s_pnl = curr_perf[curr_perf['Category'].str.contains('Stock|Equity', na=False)]['Realized Total'].apply(cl).sum()
    fy_f_pnl = curr_perf[curr_perf['Category'].str.contains('Forex|Cash|Interest', na=False)]['Realized Total'].apply(cl).sum()
    
    # 2. FY COMMISSIONS (Split by Symbol Length)
    fy_s_comm = curr_trades[curr_trades['Symbol'].str.len() <= 5]['Comm/Fee'].apply(cl).sum()
    fy_f_comm = curr_trades[curr_trades['Symbol'].str.len() > 5]['Comm/Fee'].apply(cl).sum()

    # 3. LIFETIME TOTALS
    lt_inv = (df_t_all[df_t_all['Qty'].apply(cl) > 0]['Qty'].apply(cl) * df_t_all[df_t_all['Qty'].apply(cl) > 0]['Price'].apply(cl)).sum()
    lt_s_pnl = df_p_all[df_p_all['Category'].str.contains('Stock|Equity', na=False)]['Realized Total'].apply(cl).sum()
    lt_f_pnl = df_p_all[df_p_all['Category'].str.contains('Forex|Cash|Interest', na=False)]['Realized Total'].apply(cl).sum()
    lt_s_comm = df_t_all[df_t_all['Symbol'].str.len() <= 5]['Comm/Fee'].apply(cl).sum()
    lt_f_comm = df_t_all[df_t_all['Symbol'].str.len() > 5]['Comm/Fee'].apply(cl).sum()

    # --- DISPLAY ---
    

    st.subheader(f"📊 {view_fy} Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Total Investment ({view_fy})", f"${fy_inv:,.2f}")
    c2.metric("FY Realized (Stock / FX)", f"${fy_s_pnl:,.2f} / ${fy_f_pnl:,.2f}")
    c3.metric("FY Commission (Stock / FX)", f"${abs(fy_s_comm):,.2f} / ${abs(fy_f_comm):,.2f}")

    st.subheader("🌐 Lifetime Totals")
    l1, l2, l3 = st.columns(3)
    l1.metric("Lifetime Investment", f"${lt_inv:,.2f}")
    l2.metric("Total Realized (Stock / FX)", f"${lt_s_pnl:,.2f} / ${lt_f_pnl:,.2f}")
    l3.metric("Total Commission (Stock / FX)", f"${abs(lt_s_comm):,.2f} / ${abs(lt_f_comm):,.2f}")

except Exception as e:
    st.error(f"Waiting for Data Sync... (Technical Error: {e})")
    st.info("Please ensure your Google Sheet sections are named 'Trades' and 'Performance Summary'.")
