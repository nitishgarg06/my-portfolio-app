import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.set_page_config(layout="wide")
st.title("🏦 Wealth Terminal: Topline Validation")

# Utility for safe math
def n(x):
    try:
        if pd.isna(x) or str(x).strip() in ['', '--']: return 0.0
        return float(str(x).replace('$','').replace(',','').replace('(','-').replace(')','').strip())
    except: return 0.0

# --- 1. DATA PULL ---
conn = st.connection("gsheets", type=GSheetsConnection)

def get_fy_data(sheet_name):
    df = conn.read(worksheet=sheet_name, ttl=0)
    # Target Trades
    t_data = df[df.iloc[:, 0].str.contains('Trades', na=False, case=False) & (df.iloc[:, 1] == 'Data')]
    # Target Performance
    p_data = df[df.iloc[:, 0].str.contains('Performance Summary|Realized', na=False, case=False) & (df.iloc[:, 1] == 'Data')]
    return t_data, p_data

try:
    # We load FY26 directly for validation
    t_26, p_26 = get_fy_data("FY26")
    
    # --- 2. CALCULATE COMMISSIONS (Split Stock vs FX) ---
    # In IBKR CSVs: Col 5 is Symbol, Col 11 is Comm/Fee
    s_comm = t_26[t_26.iloc[:, 4].str.len() <= 5].iloc[:, 10].apply(n).sum()
    f_comm = t_26[t_26.iloc[:, 4].str.len() > 5].iloc[:, 10].apply(n).sum()

    # --- 3. CALCULATE REALIZED (Split Stock vs FX) ---
    # Col 3 is Category, Col 8 is Realized Total
    s_pnl = p_26[p_26.iloc[:, 2].str.contains('Stock|Equity', na=False)].iloc[:, 7].apply(n).sum()
    f_pnl = p_26[p_26.iloc[:, 2].str.contains('Forex|Cash|Interest', na=False)].iloc[:, 7].apply(n).sum()

    # --- 4. CALCULATE INVESTMENT ---
    # Col 6 is Qty, Col 8 is Price
    buys = t_26[t_26.iloc[:, 5].apply(n) > 0]
    fy_inv = (buys.iloc[:, 5].apply(n) * buys.iloc[:, 7].apply(n)).sum() + abs(s_comm + f_comm)

    # --- DISPLAY ---
    st.subheader("📊 FY26 Topline Validation")
    c1, c2, c3 = st.columns(3)
    c1.metric("FY Investment (FY26)", f"${fy_inv:,.2f}")
    c2.metric("Realized Stocks", f"${s_pnl:,.2f}")
    c3.metric("Realized Forex/Cash", f"${f_pnl:,.2f}")

    st.subheader("💸 Commission Splits")
    k1, k2 = st.columns(2)
    k1.metric("Stock Commissions", f"${abs(s_comm):,.2f}")
    k2.metric("Forex Commissions", f"${abs(f_comm):,.2f}")
    
    # Show the "missing" $102.88 explicitly if found
    if abs(s_comm + f_comm) > 0:
        st.info(f"✅ Total Commissions Found: ${abs(s_comm + f_comm):,.2f}")

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Check if your FY26 sheet has sections named 'Trades' and 'Performance Summary'.")
