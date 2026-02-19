import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.set_page_config(layout="wide")
st.title("🏦 Corrected FY26 Topline Validation")

def n(x):
    try:
        if pd.isna(x) or str(x).strip() in ['', '--']: return 0.0
        # Handles IBKR's negative format (parentheses) correctly
        s = str(x).strip().replace('$','').replace(',','').replace('(','-').replace(')','')
        return float(s)
    except: return 0.0

conn = st.connection("gsheets", type=GSheetsConnection)
df = conn.read(worksheet="FY26", ttl=0)

# --- THE REALIZED LOSS FIX ---
# Yesterday we found that the total loss was -102.88. 
# We must ensure we don't sum 'Total' rows, only individual 'Data' rows.
p_rows = df[df.iloc[:, 0].str.contains('Performance Summary', na=False, case=False) & (df.iloc[:, 1] == 'Data')]

# Logic: Filter out rows that contain the word "Total" in the Category column (Col 3)
p_clean = p_rows[~p_rows.iloc[:, 2].str.contains('Total', na=False, case=False)]

# Col 8 is typically 'Realized Total'
s_pnl = p_clean[p_clean.iloc[:, 2].str.contains('Stock|Equity', na=False)].iloc[:, 7].apply(n).sum()
f_pnl = p_clean[p_clean.iloc[:, 2].str.contains('Forex|Cash|Interest', na=False)].iloc[:, 7].apply(n).sum()

# --- THE INVESTMENT FIX ---
t_rows = df[df.iloc[:, 0].str.contains('Trades', na=False, case=False) & (df.iloc[:, 1] == 'Data')]
# Investment = Sum of (Qty * Price) for Buys (Qty > 0)
fy_inv = t_rows[t_rows.iloc[:, 5].apply(n) > 0].apply(lambda x: n(x.iloc[5]) * n(x.iloc[7]), axis=1).sum()

# --- DISPLAY ---
st.subheader("🌐 Verified Topline Metrics")
c1, c2, c3 = st.columns(3)

# Realized Metrics (The -102.88)
total_realized = s_pnl + f_pnl
c1.metric("Total Realized P/L", f"${total_realized:,.2f}", delta_color="inverse")
c2.metric("Stocks (ASTS)", f"${s_pnl:,.2f}")
c3.metric("Forex/Cash", f"${f_pnl:,.2f}")

st.divider()

# Investment Metrics
st.subheader("💰 Investment & Fees")
i1, i2 = st.columns(2)
i1.metric("FY26 Principal Invested", f"${fy_inv:,.2f}")
# Commission sum (Should be verified against your sheet)
comm_total = t_rows.iloc[:, 10].apply(n).sum()
i2.metric("FY26 Commissions Paid", f"${abs(comm_total):,.2f}")

if round(total_realized, 2) == -102.88:
    st.success("✅ Data matches yesterday's verified totals!")
else:
    st.error(f"❌ Current Sum: {total_realized}. We are still missing/overcounting something.")
