import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="IBKR Portfolio Pro", layout="wide", page_icon="📈")

# --- CUSTOM CSS FOR MODERN UI ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; color: #1e293b; }
    .stDataFrame { border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }
    .reportview-container .main .block-container { padding-top: 2rem; }
    .section-header { font-size: 20px; font-weight: 600; color: #334155; margin-bottom: 1rem; border-left: 5px solid #3b82f6; padding-left: 10px; }
    </style>
    """, unsafe_allow_stdio=True)

# --- 1. CONNECTION & PARSING ---
conn = st.connection("gsheets", type=GSheetsConnection)

def parse_ibkr_sheet(df):
    sections = {}
    if df.empty: return sections
    df = df.astype(str).apply(lambda x: x.str.strip())
    for name in df.iloc[:, 0].unique():
        if name == 'nan' or not name: continue
        sec_df = df[df.iloc[:, 0] == name]
        header_row = sec_df[sec_df.iloc[:, 1] == 'Header']
        if not header_row.empty:
            cols = [c for c in header_row.iloc[0, 2:].tolist() if c and c != 'nan']
            data = sec_df[sec_df.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(cols)]
            data.columns = cols
            sections[name] = data
    return sections

# --- 2. DATA INGESTION ---
tabs = ["FY24", "FY25", "FY26"]
fy_data_map = {}
all_trades_list = []

for tab in tabs:
    try:
        raw_df = conn.read(worksheet=tab, ttl=0)
        parsed = parse_ibkr_sheet(raw_df)
        fy_data_map[tab] = parsed
        if "Trades" in parsed: all_trades_list.append(parsed["Trades"])
    except: continue

# --- 3. THE ENGINE (FIFO & SPLITS) ---
if all_trades_list:
    trades = pd.concat(all_trades_list, ignore_index=True)
    trades['Quantity'] = pd.to_numeric(trades['Quantity'], errors='coerce')
    trades['T. Price'] = pd.to_numeric(trades['T. Price'], errors='coerce')
    trades['Date/Time'] = pd.to_datetime(trades['Date/Time'])
    trades = trades.sort_values('Date/Time')

    # SPLIT ADJUSTMENTS
    trades.loc[(trades['Symbol'] == 'NVDA') & (trades['Date/Time'] < '2024-06-10'), 'Quantity'] *= 10
    trades.loc[(trades['Symbol'] == 'NVDA') & (trades['Date/Time'] < '2024-06-10'), 'T. Price'] /= 10
    trades.loc[(trades['Symbol'] == 'SMCI') & (trades['Date/Time'] < '2024-10-01'), 'Quantity'] *= 10
    trades.loc[(trades['Symbol'] == 'SMCI') & (trades['Date/Time'] < '2024-10-01'), 'T. Price'] /= 10

    today = pd.Timestamp.now()
    holdings = []
    for sym in trades['Symbol'].unique():
        lots = []
        for _, row in trades[trades['Symbol'] == sym].iterrows():
            q = row['Quantity']
            if q > 0: lots.append({'date': row['Date/Time'], 'qty': q, 'price': row['T. Price']})
            elif q < 0:
                sq = abs(q)
                while sq > 0 and lots:
                    if lots[0]['qty'] <= sq: sq -= lots[0].pop('qty'); lots.pop(0)
                    else: lots[0]['qty'] -= sq; sq = 0
        for lot in lots:
            lot['Symbol'] = sym
            lot['Type'] = "Long-Term" if (today - lot['date']).days > 365 else "Short-Term"
            holdings.append(lot)
    df_lots = pd.DataFrame(holdings)

# --- 4. TOP ROW: PERFORMANCE OVERVIEW ---
st.title("🏦 Wealth Terminal")

# Sidebar for FY Selection (Isolated to Performance section)
with st.sidebar:
    st.header("Settings")
    sel_fy = st.selectbox("Financial Year Context", tabs, index=len(tabs)-1)
    st.divider()
    st.caption("Data refreshed from Google Sheets (Live)")

with st.container():
    col1, col2, col3 = st.columns(3)
    
    data_fy = fy_data_map.get(sel_fy, {})
    
    # Funds Invested Metric
    invested = 0
    if "Deposits & Withdrawals" in data_fy:
        dw = data_fy["Deposits & Withdrawals"]
        dw['Amount'] = pd.to_numeric(dw['Amount'], errors='coerce').fillna(0)
        invested = dw[~dw.apply(lambda r: r.astype(str).str.contains('Total', case=False).any(), axis=1)]['Amount'].sum()
    col1.metric("Capital Injected", f"${invested:,.2f}", help=f"Net deposits for {sel_fy}")

    # Realized Profit Metric
    realized = 0
    if "Realized & Unrealized Performance Summary" in data_fy:
        perf = data_fy["Realized & Unrealized Performance Summary"]
        realized = pd.to_numeric(perf
