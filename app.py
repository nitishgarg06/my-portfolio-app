import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

# --- UTILITY: FORCED NUMERIC CLEANING ---
def n(x): return pd.to_numeric(str(x).replace('$','').replace(',','').replace('(','-').replace(')',''), errors='coerce').fillna(0)

st.title("🏦 Wealth Terminal: Topline Validation")

# --- DATA EXTRACTION ---
FY_LIST = ["FY24", "FY25", "FY26"]
view_fy = st.sidebar.selectbox("Select View", FY_LIST, index=2)

conn = st.connection("gsheets", type=GSheetsConnection)

def get_raw_fy_data(fy_name):
    df = conn.read(worksheet=fy_name, ttl=0)
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

# --- AGGREGATION & CALCULATION ---
try:
    all_t, all_p = [], []
    # Load all years up to the selected one for Lifetime stats
    for y in FY_LIST[:FY_LIST.index(view_fy)+1]:
        td, pd_data = get_raw_fy_data(y)
        td['FY_Ref'] = y
        pd_data['FY_Ref'] = y
        all_t.append(td); all_p.append(pd_data)

    master_t = pd.concat(all_t); master_p = pd.concat(all_p)
    
    # Filter for Current FY only
    curr_t = master_t[master_t['FY_Ref'] == view_fy]
    curr_p = master_p[master_p['FY_Ref'] == view_fy]

    # 1. Commission Splits (Stock Tickers are typically <= 5 chars)
    fy_s_com = curr_t[curr_t['Symbol'].str.len() <= 5]['Comm/Fee'].apply(n).sum()
    fy_f_com = curr_t[curr_t['Symbol'].str.len() > 5]['Comm/Fee'].apply(n).sum()
    lt_s_com = master_t[master_t['Symbol'].str.len() <= 5]['Comm/Fee'].apply(n).sum()
    lt_f_com = master_t[master_t['Symbol'].str.len() > 5]['Comm/Fee'].apply(n).sum()

    # 2. Realized P/L Splits
    fy_s_pnl = curr_p[curr_p['Category'].str.contains('Stock|Equity', na=False)]['Realized Total'].apply(n).sum()
    fy_f_pnl = curr_p[curr_p['Category'].str.contains('Forex|Cash|Interest', na=False)]['Realized Total'].apply(n).sum()
    lt_s_pnl = master_p[master_p['Category'].str.contains('Stock|Equity', na=False)]['Realized Total'].apply(n).sum()
    lt_f_pnl = master_p[master_p['Category'].str.contains('Forex|Cash|Interest', na=False)]['Realized Total'].apply(n).sum()

    # 3. Investment Totals
    fy_inv = (curr_t[curr_t['Qty'].apply(n) > 0]['Qty'].apply(n) * curr_t[curr_t['Qty'].apply(n) > 0]['Price'].apply(n)).sum()
    lt_inv = (master_t[master_t['Qty'].apply(n) > 0]['Qty'].apply(n) * master_t[master_t['Qty'].apply(n) > 0]['Price'].apply(n)).sum()

    # --- DISPLAY INTERFACE ---
    st.subheader(f"📊 {view_fy} Performance Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"FY Investment ({view_fy})", f"${fy_inv:,.2f}")
    c2.metric("FY Realized (Stock / FX)", f"${fy_s_pnl:,.2f} / ${fy_f_pnl:,.2f}")
    c3.metric("FY Comm (Stock / FX)", f"${abs(fy_s_com):,.2f} / ${abs(fy_f_com):,.2f}")

    st.subheader("🌐 Lifetime Master Totals")
    l1, l2, l3 = st.columns(3)
    l1.metric("Lifetime Investment", f"${lt_inv:,.2f}")
    l2.metric("Total Realized (Stock / FX)", f"${lt_s_pnl:,.2f} / ${lt_f_pnl:,.2f}")
    l3.metric("Total Comm (Stock / FX)", f"${abs(lt_s_com):,.2f} / ${abs(lt_f_com):,.2f}")

except Exception as e:
    st.error(f"Syncing Error: {e}")
