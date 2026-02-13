import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="IBKR Pro Dashboard", layout="wide", page_icon="📈")

# --- 1. CONNECTION ---
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
            cols = [c for c in header_row.iloc[0, 2:].tolist() if c]
            data = sec_df[sec_df.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(cols)]
            data.columns = cols
            sections[name] = data
    return sections

# --- 2. DATA LOADING (Global) ---
tabs_to_read = ["FY24", "FY25", "FY26"]
fy_data = {}
all_trades, all_realized = [], []

for tab in tabs_to_read:
    try:
        raw_df = conn.read(worksheet=tab, ttl=0)
        parsed = parse_ibkr_sheet(raw_df)
        fy_data[tab] = parsed
        if "Trades" in parsed: all_trades.append(parsed["Trades"])
        if "Realized & Unrealized Performance Summary" in parsed: 
            all_realized.append(parsed["Realized & Unrealized Performance Summary"])
    except: continue

st.title("🚀 Portfolio Pro-Analyst")

# --- 3. REQUIREMENT (B): INVESTMENT SUMMARY (FY SPECIFIC) ---
st.header("💰 Investment Capital")
selected_fy = st.selectbox("Select Financial Year to view Funds Injected", tabs_to_read, index=len(tabs_to_read)-1)

data_inv = fy_data.get(selected_fy, {})
if "Deposits & Withdrawals" in data_inv:
    dw = data_inv["Deposits & Withdrawals"]
    dw['Amount'] = pd.to_numeric(dw['Amount'], errors='coerce').fillna(0)
    # Filter out 'Total' rows
    actual_funds = dw[~dw['Currency'].astype(str).str.contains('Total', case=False, na=False)]
    total_inv = actual_funds['Amount'].sum()
    st.metric(f"Net Funds Injected ({selected_fy})", f"${total_inv:,.2f}")
else:
    st.info(f"No deposit data found for {selected_fy}.")

# --- 4. REQUIREMENT (A): HOLDINGS & CGT BREAKDOWN ---
st.divider()
st.header("📊 Stock Summary & CGT Status")

if all_trades:
    df_trades = pd.concat(all_trades)
    df_trades['Quantity'] = pd.to_numeric(df_trades['Quantity'], errors='coerce')
    df_trades['T. Price'] = pd.to_numeric(df_trades['T. Price'], errors='coerce')
    df_trades['Date/Time'] = pd.to_datetime(df_trades['Date/Time'])
    df_trades = df_trades.sort_values('Date/Time')

    # Get Realized Profit Data
    df_realized = pd.concat(all_realized) if all_realized else pd.DataFrame()
    if not df_realized.empty:
        df_realized['Realized Total'] = pd.to_numeric(df_realized['Realized Total'], errors='coerce').fillna(0)
        realized_map = df_realized.groupby('Symbol')['Realized Total'].sum().to_dict()
    else:
        realized_map = {}

    symbols = df_trades['Symbol'].unique().tolist()
    today = pd.Timestamp.now()
    summary_list = []

    for sym in symbols:
        # FIFO Logic for LT/ST
        lots = [] # List of (date, qty, price)
        stock_trades = df_trades[df_trades['Symbol'] == sym]
        
        for _, row in stock_trades.iterrows():
            q = row['Quantity']
            if q > 0: # Buy
                lots.append({'date': row['Date/Time'], 'qty': q, 'price': row['T. Price']})
            elif q < 0: # Sell
                sell_q = abs(q)
                while sell_q > 0 and lots:
                    if lots[0]['qty'] <= sell_q:
                        sell_q -= lots[0]['qty']
                        lots.pop(0)
                    else:
                        lots[0]['qty'] -= sell_q
                        sell_q = 0
        
        # Categorize remaining lots
        lt_qty, st_qty, total_cost = 0, 0, 0
        for lot in lots:
            total_cost += lot['qty'] *
