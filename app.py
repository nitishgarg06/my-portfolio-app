import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.set_page_config(layout="wide")
st.title("🛡️ Data Integrity Check: FY26")

def n(x):
    try:
        if pd.isna(x) or str(x).strip() in ['', '--']: return 0.0
        return float(str(x).replace('$','').replace(',','').replace('(','-').replace(')','').strip())
    except: return 0.0

conn = st.connection("gsheets", type=GSheetsConnection)
df = conn.read(worksheet="FY26", ttl=0)

# --- 1. THE TRADES AUDIT (Investment & Commissions) ---
st.subheader("1. Trades Section Audit")
t_all = df[df.iloc[:, 0].str.contains('Trades', na=False, case=False)]
t_data = t_all[t_all.iloc[:, 1] == 'Data']

# IBKR Columns: 4=Symbol, 5=Date/Time, 6=Qty, 7=Price, 10=Comm/Fee, 11=Basis (usually)
# We will use explicit column indexing to avoid header name issues
trades_df = t_data.iloc[:, [4, 6, 7, 10]].copy()
trades_df.columns = ['Symbol', 'Qty', 'Price', 'Comm']
for col in ['Qty', 'Price', 'Comm']: trades_df[col] = trades_df[col].apply(n)

# Split logic
stock_trades = trades_df[trades_df['Symbol'].str.len() <= 5]
forex_trades = trades_df[trades_df['Symbol'].str.len() > 5]

fy_inv = (stock_trades[stock_trades['Qty'] > 0]['Qty'] * stock_trades[stock_trades['Qty'] > 0]['Price']).sum()
s_comm = stock_trades['Comm'].sum()
f_comm = forex_trades['Comm'].sum()

c1, c2, c3 = st.columns(3)
c1.metric("FY26 Raw Investment", f"${fy_inv:,.2f}")
c2.metric("Stock Comm (Total)", f"${abs(s_comm):,.2f}")
c3.metric("Forex/Cash Comm (Total)", f"${abs(f_comm):,.2f}")

with st.expander("🔍 View Raw Trade Row Calculations"):
    st.write(trades_df)

# --- 2. THE PERFORMANCE AUDIT (Realized P/L) ---
st.subheader("2. Performance Section Audit")
p_all = df[df.iloc[:, 0].str.contains('Performance Summary|Realized', na=False, case=False)]
p_data = p_all[p_all.iloc[:, 1] == 'Data']

# Columns: 2=Category, 7=Realized Total
perf_df = p_data.iloc[:, [2, 7]].copy()
perf_df.columns = ['Category', 'RealizedTotal']
perf_df['RealizedTotal'] = perf_df['RealizedTotal'].apply(n)

s_pnl = perf_df[perf_df['Category'].str.contains('Stock|Equity', na=False)]['RealizedTotal'].sum()
f_pnl = perf_df[perf_df['Category'].str.contains('Forex|Cash|Interest', na=False)]['RealizedTotal'].sum()

p1, p2, p3 = st.columns(3)
p1.metric("Realized Stocks", f"${s_pnl:,.2f}")
p2.metric("Realized Forex/Cash", f"${f_pnl:,.2f}")
p3.metric("Combined Realized", f"${(s_pnl + f_pnl):,.2f}")

with st.expander("🔍 View Raw Performance Row Calculations"):
    st.write(perf_df)
