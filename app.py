import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.set_page_config(page_title="Data Validator", layout="wide")
st.title("🔍 FY26 Topline Validation")

# Connect directly to your FY26 Sheet
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
    df = conn.read(worksheet="FY26", ttl=0)
    st.success("✅ Successfully connected to Google Sheet: FY26")
except Exception as e:
    st.error(f"❌ Connection Failed: {e}")
    st.stop()

# --- 1. COMMISSION CALCULATION ---
st.subheader("1. Commission Breakdown")
t_rows = df[df.iloc[:, 0].str.contains('Trades', na=False, case=False)]
if not t_rows.empty:
    h = t_rows[t_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
    data = t_rows[t_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(h)]
    data.columns = h
    
    # Target the 'Comm/Fee' column
    comm_col = [c for c in data.columns if 'Comm' in c or 'Fee' in c][0]
    total_comm = data[comm_col].apply(lambda x: float(str(x).replace('$','').replace(',','').replace('(','-').replace(')','')) if pd.notnull(x) else 0).sum()
    
    st.metric("Total Commission (FY26)", f"${abs(total_comm):,.2f}")
    with st.expander("View Raw Trade Data Found"):
        st.dataframe(data[[next(c for c in data.columns if 'Symbol' in c), comm_col]])
else:
    st.warning("⚠️ No 'Trades' section found in the FY26 sheet.")

# --- 2. PERFORMANCE CALCULATION ---
st.subheader("2. Realized P/L Breakdown")
p_rows = df[df.iloc[:, 0].str.contains('Performance Summary|Realized', na=False, case=False)]
if not p_rows.empty:
    h_p = p_rows[p_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
    p_data = p_rows[p_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(h_p)]
    p_data.columns = h_p
    
    # Target 'Realized Total' and 'Category'
    rt_col = [c for c in p_data.columns if 'Realized' in c and 'Total' in c][0]
    cat_col = [c for c in p_data.columns if 'Category' in c or 'Asset' in c][0]
    
    p_data[rt_col] = p_data[rt_col].apply(lambda x: float(str(x).replace('$','').replace(',','').replace('(','-').replace(')','')) if pd.notnull(x) else 0)
    
    stocks_pnl = p_data[p_data[cat_col].str.contains('Stock|Equity', na=False, case=False)][rt_col].sum()
    forex_pnl = p_data[p_data[cat_col].str.contains('Forex|Cash|Interest', na=False, case=False)][rt_col].sum()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Stocks Realized P/L", f"${stocks_pnl:,.2f}")
    c2.metric("Forex/Interest P/L", f"${forex_pnl:,.2f}")
    c3.metric("TOTAL Realized P/L", f"${(stocks_pnl + forex_pnl):,.2f}")
    
    with st.expander("View Raw Performance Data Found"):
        st.dataframe(p_data[[cat_col, rt_col]])
else:
    st.warning("⚠️ No 'Performance' section found in the FY26 sheet.")
