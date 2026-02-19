import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

# --- UTILITIES ---
def cl(x): return pd.to_numeric(str(x).replace('$','').replace(',','').replace('(','-').replace(')',''), errors='coerce').fillna(0)

st.title("🏦 Wealth Terminal: Topline Validation")

# --- DATA FETCH ---
FY_LIST = ["FY24", "FY25", "FY26"]
view_fy = st.sidebar.selectbox("Select View", FY_LIST, index=2)

conn = st.connection("gsheets", type=GSheetsConnection)

def get_raw_metrics(fy_name):
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

# --- CALCULATIONS ---
try:
    all_t, all_p = [], []
    for y in FY_LIST[:FY_LIST.index(view_fy)+1]:
        td, pd_data = get_raw_metrics(y)
        td['FY_Ref'] = y
        pd_data['FY_Ref'] = y
        all_t.append(td); all_p.append(pd_data)

    df_t = pd.concat(all_t); df_p = pd.concat(all_p)
    fy_t = df_t[df_t['FY_Ref'] == view_fy]; fy_p = df_p[df_p['FY_Ref'] == view_fy]

    # Commission Splits (Stocks vs Forex)
    f_s_com = fy_t[fy_t['Symbol'].str.len() <= 5]['Comm/Fee'].apply(cl).sum()
    f_f_com = fy_t[fy_t['Symbol'].str.len() > 5]['Comm/Fee'].apply(cl).sum()
    l_s_com = df_t[df_t['Symbol'].str.len() <= 5]['Comm/Fee'].apply(cl).sum()
    l_f_com = df_t[df_t['Symbol'].str.len() > 5]['Comm/Fee'].apply(cl).sum()

    # Realized Splits
    f_s_pnl = fy_p[fy_p['Category'].str.contains('Stock|Equity', na=False)]['Realized Total'].apply(cl).sum()
    f_f_pnl = fy_p[fy_p['Category'].str.contains('Forex|Cash|Interest', na=False)]['Realized Total'].apply(cl).sum()
    l_s_pnl = df_p[df_p['Category'].str.contains('Stock|Equity', na=False)]['Realized Total'].apply(cl).sum()
    l_f_pnl = df_p[df_p['Category'].str.contains('Forex|Cash|Interest', na=False)]['Realized Total'].apply(cl).sum()

    # Investment
    f_inv = (fy_t[fy_t['Qty'].apply(cl) > 0]['Qty'].apply(cl) * fy_t[fy_t['Qty'].apply(cl) > 0]['Price'].apply(cl)).sum()
    l_inv = (df_t[df_t['Qty'].apply(cl) > 0]['Qty'].apply(cl) * df_t[df_t['Qty'].apply(cl) > 0]['Price'].apply(cl)).sum()

    # --- DISPLAY ---
    st.subheader(f"📊 {view_fy} Performance Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("FY Investment", f"${f_inv:,.2f}")
    c2.metric("FY Realized (Stock / FX)", f"${f_s_pnl:,.2f} / ${f_f_pnl:,.2f}")
    c3.metric("FY Comm (Stock / FX)", f"${abs(f_s_com):,.2f} / ${abs(f_f_com):,.2f}")

    st.subheader("🌐 Lifetime Master Totals")
    l1, l2, l3 = st.columns(3)
    l1.metric("Lifetime Investment", f"${l_inv:,.2f}")
    l2.metric("Total Realized (Stock / FX)", f"${l_s_pnl:,.2f} / ${l_f_pnl:,.2f}")
    l3.metric("Total Comm (Stock / FX)", f"${abs(l_s_com):,.2f} / ${abs(l_f_com):,.2f}")

except Exception as e:
    st.error(f"Waiting for Data Sync... (Last Error: {e})")
