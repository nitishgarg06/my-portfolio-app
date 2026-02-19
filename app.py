import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.title("🏦 Wealth Terminal: Topline Validation")

# Utility to convert currency strings to numbers safely
def clean(val):
    return pd.to_numeric(str(val).replace('$','').replace(',','').replace('(','-').replace(')',''), errors='coerce').fillna(0)

# --- 1. DATA PULL ---
FY_LIST = ["FY24", "FY25", "FY26"]
view_fy = st.sidebar.selectbox("Select Financial Year", FY_LIST, index=2)

conn = st.connection("gsheets", type=GSheetsConnection)

# Function to get cleaned Trades and Performance for any year
def get_fy_data(year):
    df = conn.read(worksheet=year, ttl=0)
    t_rows = df[df.iloc[:, 0].str.contains('Trades', na=False, case=False)]
    th = t_rows[t_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
    td = t_rows[t_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(th)]
    td.columns = th
    
    p_rows = df[df.iloc[:, 0].str.contains('Performance Summary|Realized', na=False, case=False)]
    ph = p_rows[p_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
    pd_data = p_rows[p_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(ph)]
    pd_data.columns = ph
    return td, pd_data

# --- 2. AGGREGATION ---
try:
    # Get Current FY
    t_fy, p_fy = get_fy_data(view_fy)
    
    # Get Cumulative (Lifetime)
    all_t = []
    all_p = []
    for y in FY_LIST[:FY_LIST.index(view_fy)+1]:
        t, p = get_fy_data(y)
        all_t.append(t)
        all_p.append(p)
    df_t_lt = pd.concat(all_t)
    df_p_lt = pd.concat(all_p)

    # --- 3. CALCULATIONS ---
    # Commissions Split (Simplified via Ticker length)
    fy_s_comm = t_fy[t_fy['Symbol'].str.len() <= 5]['Comm/Fee'].apply(clean).sum()
    fy_f_comm = t_fy[t_fy['Symbol'].str.len() > 5]['Comm/Fee'].apply(clean).sum()
    lt_s_comm = df_t_lt[df_t_lt['Symbol'].str.len() <= 5]['Comm/Fee'].apply(clean).sum()
    lt_f_comm = df_t_lt[df_t_lt['Symbol'].str.len() > 5]['Comm/Fee'].apply(clean).sum()

    # Realized Split
    fy_s_pnl = p_fy[p_fy['Category'].str.contains('Stock', na=False)]['Realized Total'].apply(clean).sum()
    fy_f_pnl = p_fy[p_fy['Category'].str.contains('Forex|Cash', na=False)]['Realized Total'].apply(clean).sum()
    lt_s_pnl = df_p_lt[df_p_lt['Category'].str.contains('Stock', na=False)]['Realized Total'].apply(clean).sum()
    lt_f_pnl = df_p_lt[df_p_lt['Category'].str.contains('Forex|Cash', na=False)]['Realized Total'].apply(clean).sum()

    # Investment
    fy_inv = (t_fy[t_fy['Qty'].apply(clean) > 0]['Qty'].apply(clean) * t_fy[t_fy['Qty'].apply(clean) > 0]['Price'].apply(clean)).sum()
    lt_inv = (df_t_lt[df_t_lt['Qty'].apply(clean) > 0]['Qty'].apply(clean) * df_t_lt[df_t_lt['Qty'].apply(clean) > 0]['Price'].apply(clean)).sum()

    # --- 4. DISPLAY ---
    st.subheader(f"📊 {view_fy} Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Investment ({view_fy})", f"${fy_inv:,.2f}")
    c2.metric(f"Realized (Stock/FX)", f"${fy_s_pnl:,.2f} / ${fy_f_pnl:,.2f}")
    c3.metric(f"Comm (Stock/FX)", f"${abs(fy_s_comm):,.2f} / ${abs(fy_f_comm):,.2f}")

    st.subheader("🌐 Lifetime Master Totals")
    l1, l2, l3 = st.columns(3)
    l1.metric("Lifetime Investment", f"${lt_inv:,.2f}")
    l2.metric("Total Realized (Stock/FX)", f"${lt_s_pnl:,.2f} / ${lt_f_pnl:,.2f}")
    l3.metric("Total Comm (Stock/FX)", f"${abs(lt_s_comm):,.2f} / ${abs(lt_f_comm):,.2f}")

except Exception as e:
    st.error(f"Error parsing data: {e}. Check if 'Trades' and 'Performance Summary' headers exist.")
