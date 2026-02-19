import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from datetime import datetime

st.set_page_config(layout="wide", page_title="Wealth Terminal Pro")
st.title("🏦 Wealth Terminal Pro")

# --- 1. CORE UTILITIES ---
def cl(x): 
    return pd.to_numeric(str(x).replace('$','').replace(',','').replace('(','-').replace(')',''), errors='coerce').fillna(0)

# --- 2. DATA EXTRACTION ENGINE ---
def get_data_from_sheet(sheet_name):
    conn = st.connection("gsheets", type=GSheetsConnection)
    df = conn.read(worksheet=sheet_name, ttl=0)
    
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

# --- 3. THE TOPLINE CALCULATOR ---
FY_LIST = ["FY24", "FY25", "FY26"]
view_fy = st.sidebar.selectbox("Select View", FY_LIST, index=2)

# Load data for ALL years up to selected for Lifetime metrics
all_trades = []
all_perf = []

with st.spinner("Calculating Lifetime Metrics..."):
    for y in FY_LIST[:FY_LIST.index(view_fy)+1]:
        try:
            t, p = get_data_from_sheet(y)
            t['FY_Source'] = y
            p['FY_Source'] = y
            all_trades.append(t)
            all_perf.append(p)
        except:
            st.sidebar.warning(f"No data found for {y}")

if all_trades:
    df_t = pd.concat(all_trades)
    df_p = pd.concat(all_perf)
    
    # --- METRIC LOGIC ---
    # FY Specific
    fy_t = df_t[df_t['FY_Source'] == view_fy]
    fy_p = df_p[df_p['FY_Source'] == view_fy]
    
    # Commissions (Stock vs Forex split by Ticker length)
    def split_comm(df):
        s = df[df['Symbol'].str.len() <= 5]['Comm/Fee'].apply(cl).sum()
        f = df[df['Symbol'].str.len() > 5]['Comm/Fee'].apply(cl).sum()
        return abs(s), abs(f)

    fy_s_comm, fy_f_comm = split_comm(fy_t)
    lt_s_comm, lt_f_comm = split_comm(df_t)

    # Realized P/L
    def split_pnl(df):
        s = df[df['Category'].str.contains('Stock|Equity', na=False)]['Realized Total'].apply(cl).sum()
        f = df[df['Category'].str.contains('Forex|Cash|Interest', na=False)]['Realized Total'].apply(cl).sum()
        return s, f

    fy_s_pnl, fy_f_pnl = split_pnl(fy_p)
    lt_s_pnl, lt_f_pnl = split_pnl(df_p)

    # Investment
    fy_inv = (fy_t[fy_t['Qty'].apply(cl) > 0]['Qty'].apply(cl) * fy_t[fy_t['Qty'].apply(cl) > 0]['Price'].apply(cl)).sum()
    lt_inv = (df_t[df_t['Qty'].apply(cl) > 0]['Qty'].apply(cl) * df_t[df_t['Qty'].apply(cl) > 0]['Price'].apply(cl)).sum()

    # --- DISPLAY TOPLINE ---
    
    
    st.subheader(f"📊 {view_fy} Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"FY Investment ({view_fy})", f"${fy_inv:,.2f}")
    c2.metric("FY Realized (Stock / FX)", f"${fy_s_pnl:,.2f} / ${fy_f_pnl:,.2f}")
    c3.metric("FY Comm (Stock / FX)", f"${fy_s_comm:,.2f} / ${fy_f_comm:,.2f}")

    st.subheader("🌐 Lifetime Totals")
    l1, l2, l3 = st.columns(3)
    l1.metric("Lifetime Investment", f"${lt_inv:,.2f}")
    l2.metric("Total Realized (Stock / FX)", f"${lt_s_pnl:,.2f} / ${lt_f_pnl:,.2f}")
    l3.metric("Total Comm (Stock / FX)", f"${lt_s_comm:,.2f} / ${lt_f_comm:,.2f}")
    
    st.info("💡 *Realized P/L values are net of the commissions shown above.*")

    # (FIFO Holdings Table and Calculator would continue here)
else:
    st.error("No data could be extracted. Please check your Google Sheet tab names.")
