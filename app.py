import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="IBKR Pro Manager", layout="wide")

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
            cols = header_row.iloc[0, 2:].tolist()
            data = sec_df[sec_df.iloc[:, 1] == 'Data'].iloc[:, 2:]
            data.columns = cols[:len(data.columns)]
            sections[name] = data
    return sections

# --- 2. DATA INGESTION ---
tabs = ["FY24", "FY25", "FY26"]
fy_data = {}
for tab in tabs:
    try:
        raw = conn.read(worksheet=tab, ttl=0)
        fy_data[tab] = parse_ibkr_sheet(raw)
    except: continue

# Sidebar FY Selector
selected_fy = st.sidebar.selectbox("Select Financial Year View", tabs, index=len(tabs)-1)
data_current = fy_data.get(selected_fy, {})

# --- 3. REQUIREMENT 1: FUNDS INVESTED ---
st.header(f"💰 Financial Summary: {selected_fy}")
if "Deposits & Withdrawals" in data_current:
    dw = data_current["Deposits & Withdrawals"]
    dw['Amount'] = pd.to_numeric(dw['Amount'], errors='coerce').fillna(0)
    total_invested = dw['Amount'].sum()
    st.metric("Total Funds Injected (Net)", f"${total_invested:,.2f}")
else:
    st.info("No 'Deposits & Withdrawals' section found for this FY.")

# --- 4. REQUIREMENT 4: LONG-TERM VS SHORT-TERM ---
st.divider()
st.header("⏳ Holding Period Analysis (CGT Status)")
all_trades_list = [v["Trades"] for k, v in fy_data.items() if "Trades" in v]
if all_trades_list:
    trades_master = pd.concat(all_trades_list)
    trades_master['Quantity'] = pd.to_numeric(trades_master['Quantity'], errors='coerce')
    trades_master['T. Price'] = pd.to_numeric(trades_master['T. Price'], errors='coerce')
    trades_master['Date/Time'] = pd.to_datetime(trades_master['Date/Time'])
    
    # Calculate holding period
    today = pd.Timestamp.now()
    buys = trades_master[trades_master['Quantity'] > 0].copy()
    buys['Age_Days'] = (today - buys['Date/Time']).dt.days
    buys['Category'] = buys['Age_Days'].apply(lambda x: "Long-Term (>1yr)" if x > 365 else "Short-Term (<1yr)")
    
    lt_sum = buys[buys['Category'] == "Long-Term (>1yr)"]['Quantity'].sum()
    st_sum = buys[buys['Category'] == "Short-Term (<1yr)"]['Quantity'].sum()
    
    c1, c2 = st.columns(2)
    c1.info(f"**Long-Term Holdings:** {lt_sum:.2f} units")
    c2.warning(f"**Short-Term Holdings:** {st_sum:.2f} units")

# --- 5. REQUIREMENT 2 & 3: ADVANCED FIFO CALCULATOR ---
st.divider()
st.header("🧮 Smart FIFO & Residual Calculator")
symbols = trades_master['Symbol'].unique().tolist()
sel_stock = st.selectbox("Select Stock to Analyze", symbols)

# Get specific stock data
stock_buys = buys[buys['Symbol'] == sel_stock].sort_values('Date/Time')
total_qty = stock_buys['Quantity'].sum()

# Dynamic Progress Bar/Slider (Requirement 2)
calc_mode = st.radio("Calculation Mode", ["Percentage", "Specific Units"])
if calc_mode == "Percentage":
    sell_amt = st.slider("Select % to Sell", 0, 100, 25) / 100
    qty_to_sell = total_qty * sell_amt
else:
    qty_to_sell = st.slider("Select Units to Sell", 0.0, float(total_qty), float(total_qty * 0.25))

target_p = st.number_input("Target Profit %", value=105.0)

if total_qty > 0:
    # FIFO Math
    temp_qty, sold_cost = qty_to_sell, 0
    shares_consumed = 0
    
    for _, row in stock_buys.iterrows():
        if temp_qty <= 0: break
        take = min(row['Quantity'], temp_qty)
        sold_cost += take * row['T. Price']
        temp_qty -= take
    
    target_sell_price = (sold_cost * (1 + target_p/100)) / qty_to_sell
    st.success(f"Target Sell Price: **${target_sell_price:.2f}**")

    # REQUIREMENT 3: RESIDUAL ANALYSIS
    remaining_qty = total_qty - qty_to_sell
    if remaining_qty > 0:
        total_portfolio_cost = (stock_buys['Quantity'] * stock_buys['T. Price']).sum()
        remaining_cost_basis = total_portfolio_cost - sold_cost
        avg_remaining_price = remaining_cost_basis / remaining_qty
        
        st.info(f"**Residual Portfolio Advice:**")
        st.write(f"After selling, you will have **{remaining_qty:.2f}** units left.")
        st.write(f"New Avg Price for remaining units: **${avg_remaining_price:.2f}**")
        
        # Current state of remaining
        with st.spinner("Pinging Yahoo Finance..."):
            last_p = yf.Ticker(sel_stock).fast_info['last_price']
            rem_profit = (last_p - avg_remaining_price) * remaining_qty
            color = "green" if rem_profit > 0 else "red"
            st.markdown(f"Status of Remaining Units: :{color}[**${rem_profit:.2f} ({((last_p/avg_remaining_price)-1)*100:.2f}%)**]")
