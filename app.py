import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.title("🛡️ Direct Data Verification (No-Guess Mode)")

# 1. THE CLEANER
def n(x):
    try:
        s = str(x).strip().replace('$','').replace(',','').replace('(','-').replace(')','')
        return float(s) if s not in ['', '--', 'None'] else 0.0
    except: return 0.0

# 2. THE EXTRACTOR
def get_clean_section(df, section_name):
    # Find the section and the header row
    section_rows = df[df.iloc[:, 0].str.contains(section_name, na=False, case=False)]
    if section_rows.empty: return pd.DataFrame()
    
    # Get headers and data
    headers = section_rows[section_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
    data = section_rows[section_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(headers)]
    data.columns = headers
    return data

conn = st.connection("gsheets", type=GSheetsConnection)

# 3. THE CALCULATION
FY_TABS = ["FY24", "FY25", "FY26"]
results = {"fy_inv": 0, "fy_pl": 0, "lt_inv": 0, "lt_pl": 0, "s_comm": 0, "f_comm": 0}

for tab in FY_TABS:
    df_raw = conn.read(worksheet=tab, ttl=0)
    
    # Get Sections
    trades = get_clean_section(df_raw, "Trades")
    perf = get_clean_section(df_raw, "Performance Summary")
    
    # Standardize column search
    q_col = next((c for c in trades.columns if 'Qty' in c), None)
    p_col = next((c for c in trades.columns if 'Price' in c), None)
    c_col = next((c for c in trades.columns if 'Comm' in c), None)
    sym_col = next((c for c in trades.columns if 'Symbol' in c), None)
    pl_col = next((c for c in perf.columns if 'Realized' in c and 'Total' in c), None)
    cat_col = next((c for c in perf.columns if 'Category' in c), None)

    # LIFETIME ACCUMULATION
    if q_col and p_col:
        # Lifetime Investment: Only sum 'Buys' (Qty > 0)
        results["lt_inv"] += trades[trades[q_col].apply(n) > 0].apply(lambda x: n(x[q_col]) * n(x[p_col]), axis=1).sum()
    if pl_col:
        # Avoid 'Total' rows in Performance
        results["lt_pl"] += perf[~perf[cat_col].str.contains('Total', na=False, case=False)][pl_col].apply(n).sum()

    # FY26 SPECIFIC (FOR THE SPLITS)
    if tab == "FY26":
        results["fy_inv"] = trades[trades[q_col].apply(n) > 0].apply(lambda x: n(x[q_col]) * n(x[p_col]), axis=1).sum()
        results["fy_pl"] = perf[~perf[cat_col].str.contains('Total', na=False, case=False)][pl_col].apply(n).sum()
        
        # Commission Split: Stock (short tickers) vs Forex (long/contains '.')
        results["s_comm"] = abs(trades[trades[sym_col].str.len() <= 5][c_col].apply(n).sum())
        results["f_comm"] = abs(trades[trades[sym_col].str.len() > 5][c_col].apply(n).sum())

# --- THE OUTPUT ---
st.header("Validated Topline Metrics")
a, b = st.columns(2)
a.metric("(a) Lifetime Investment", f"${results['lt_inv']:,.2f}")
b.metric("(b) Lifetime P/L", f"${results['lt_pl']:,.2f}")

c, d = st.columns(2)
c.metric("(c) Total FY26 Investment", f"${results['fy_inv']:,.2f}")
d.metric("(d) Total FY26 P/L", f"${results['fy_pl']:,.2f}")

st.subheader("(e) FY26 Commission Split")
st.write(f"**Stock Commissions:** ${results['s_comm']:,.2f}")
st.write(f"**Forex Commissions:** ${results['f_comm']:,.2f}")
st.write(f"**Total FY26 Fees:** ${results['s_comm'] + results['f_comm']:,.2f}")
