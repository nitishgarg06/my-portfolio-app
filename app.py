import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.set_page_config(layout="wide")
st.title("🏦 FY26 Topline Validation")

# Numeric cleaner to prevent math errors
def n(x): return pd.to_numeric(str(x).replace('$','').replace(',','').replace('(','-').replace(')',''), errors='coerce').fillna(0)

# --- DATA LOAD ---
FY_LIST = ["FY24", "FY25", "FY26"]
view_fy = st.sidebar.selectbox("Select View", FY_LIST, index=2)

try:
    conn = st.connection("gsheets", type=GSheetsConnection)
    df = conn.read(worksheet=view_fy, ttl=0)
    st.success(f"Connected to {view_fy}")

    # --- 1. SEARCH FOR TRADES ---
    t_rows = df[df.iloc[:, 0].str.contains('Trades', na=False, case=False)]
    if not t_rows.empty:
        th = t_rows[t_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
        td = t_rows[t_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(th)]
        td.columns = th
        
        # Calculate FY Investment & Commissions
        fy_inv = (td[td['Qty'].apply(n) > 0]['Qty'].apply(n) * td[td['Qty'].apply(n) > 0]['Price'].apply(n)).sum()
        s_comm = abs(td[td['Symbol'].str.len() <= 5]['Comm/Fee'].apply(n).sum())
        f_comm = abs(td[td['Symbol'].str.len() > 5]['Comm/Fee'].apply(n).sum())
        
        st.subheader(f"📊 {view_fy} Investment & Fees")
        m1, m2, m3 = st.columns(3)
        m1.metric("FY Investment", f"${fy_inv:,.2f}")
        m2.metric("Stock Comm", f"${s_comm:,.2f}")
        m3.metric("Forex/Cash Comm", f"${f_comm:,.2f}")
    else:
        st.warning("Could not find 'Trades' section in this sheet.")

    # --- 2. SEARCH FOR PERFORMANCE ---
    p_rows = df[df.iloc[:, 0].str.contains('Performance Summary|Realized', na=False, case=False)]
    if not p_rows.empty:
        ph = p_rows[p_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
        pd_data = p_rows[p_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(ph)]
        pd_data.columns = ph
        
        # Calculate P/L Splits
        s_pnl = pd_data[pd_data['Category'].str.contains('Stock|Equity', na=False)]['Realized Total'].apply(n).sum()
        f_pnl = pd_data[pd_data['Category'].str.contains('Forex|Cash|Interest', na=False)]['Realized Total'].apply(n).sum()
        
        st.subheader(f"📈 {view_fy} Realized P/L")
        p1, p2, p3 = st.columns(3)
        p1.metric("Stock Realized", f"${s_pnl:,.2f}")
        p2.metric("Forex/Interest", f"${f_pnl:,.2f}")
        p3.metric("Total Realized", f"${(s_pnl + f_pnl):,.2f}")
    else:
        st.warning("Could not find 'Performance' section in this sheet.")

except Exception as e:
    st.error(f"Logic Error: {e}")
