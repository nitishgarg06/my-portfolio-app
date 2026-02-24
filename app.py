import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.title("🛡️ Step 1: Topline Data Verification")

def n(x):
    try:
        s = str(x).strip().replace('$','').replace(',','').replace('(','-').replace(')','')
        return float(s) if s not in ['', '--'] else 0.0
    except: return 0.0

conn = st.connection("gsheets", type=GSheetsConnection)

# 1. Fetch all years for Lifetime metrics
all_t = []
all_p = []
for yr in ["FY24", "FY25", "FY26"]:
    df = conn.read(worksheet=yr, ttl=0)
    t = df[df.iloc[:, 0].str.contains('Trades', na=False) & (df.iloc[:, 1] == 'Data')]
    p = df[df.iloc[:, 0].str.contains('Performance Summary', na=False) & (df.iloc[:, 1] == 'Data')]
    # Filter out 'Total' rows to prevent double-counting
    p = p[~p.iloc[:, 2].str.contains('Total', na=False, case=False)]
    t['YR'] = yr
    p['YR'] = yr
    all_t.append(t); all_p.append(p)

df_t = pd.concat(all_t)
df_p = pd.concat(all_p)

# --- (a) & (b) LIFETIME ---
# Lifetime Invested = Total Cost of Buys - Total Principal of Sells
lt_inv = (df_t[df_t.iloc[:, 5].apply(n) > 0].apply(lambda x: n(x.iloc[5]) * n(x.iloc[7]), axis=1).sum())
lt_pl = df_p.iloc[:, 7].apply(n).sum()

# --- (c) & (d) FY26 ONLY ---
fy_t = df_t[df_t['YR'] == "FY26"]
fy_p = df_p[df_p['YR'] == "FY26"]

fy_inv = (fy_t[fy_t.iloc[:, 5].apply(n) > 0].apply(lambda x: n(x.iloc[5]) * n(x.iloc[7]), axis=1).sum())
fy_pl = fy_p.iloc[:, 7].apply(n).sum()

# --- (e) COMMISSIONS (FY26 SPLIT) ---
# Ticker <= 5 chars is Stock; > 5 chars is Forex
stock_comm = abs(fy_t[fy_t.iloc[:, 4].str.len() <= 5].iloc[:, 10].apply(n).sum())
forex_comm = abs(fy_t[fy_t.iloc[:, 4].str.len() > 5].iloc[:, 10].apply(n).sum())

# --- OUTPUT ---
st.header("FY26 Verification Results")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total FY26 Investment (c)", f"${fy_inv:,.2f}")
    st.metric("Total FY26 P/L (d)", f"${fy_pl:,.2f}")
with col2:
    st.metric("Lifetime Investment (a)", f"${lt_inv:,.2f}")
    st.metric("Lifetime P/L (b)", f"${lt_pl:,.2f}")

st.subheader("Total Commissions (e)")
st.write(f"**Stock Commissions:** ${stock_comm:,.2f}")
st.write(f"**Forex Commissions:** ${forex_comm:,.2f}")
st.write(f"**Total FY26 Fees:** ${stock_comm + forex_comm:,.2f}")
