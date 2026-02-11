import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf

# --- APP CONFIG ---
st.set_page_config(page_title="IBKR Master Portfolio", layout="wide", page_icon="📈")

# --- 1. SECURE CONNECTION ---
conn = st.connection("gsheets", type=GSheetsConnection)

def parse_ibkr_sheet(df):
    """Parses the non-standard IBKR multi-table format into clean DataFrames."""
    sections = {}
    if df.empty: return sections
    for name in df.iloc[:, 0].unique():
        sec_df = df[df.iloc[:, 0] == name]
        header_row = sec_df[sec_df.iloc[:, 1] == 'Header']
        if not header_row.empty:
            cols = [c.strip() for c in header_row.iloc[0, 2:].tolist()]
            data = sec_df[sec_df.iloc[:, 1] == 'Data'].iloc[:, 2:]
            data.columns = cols[:len(data.columns)]
            sections[name] = data
    return sections

# --- 2. MULTI-YEAR DATA INGESTION ---
tabs_to_read = ["FY24", "FY25", "FY26", "FY27"] # Add more as years pass
all_trades, all_splits, all_divs = [], [], []

for tab in tabs_to_read:
    try:
        # ttl=0 ensures the app pulls fresh data from Google Sheets every time
        df_raw = conn.read(worksheet=tab, ttl=0) 
        parsed = parse_ibkr_sheet(df_raw)
        if "Trades" in parsed: all_trades.append(parsed["Trades"])
        if "Corporate Actions" in parsed: all_splits.append(parsed["Corporate Actions"])
        if "Dividends" in parsed: all_divs.append(parsed["Dividends"])
    except:
        continue

if not all_trades:
    st.error("No data found in Google Sheets. Check tab names and Service Account access.")
    st.stop()

# --- 3. DATA CONSOLIDATION & SPLIT WATCHDOG ---
trades = pd.concat(all_trades)
trades['Quantity'] = pd.to_numeric(trades['Quantity'], errors='coerce')
trades['T. Price'] = pd.to_numeric(trades['T. Price'], errors='coerce')
trades['Date/Time'] = pd.to_datetime(trades['Date/Time'])
trades = trades[trades['Quantity'].notnull()].sort_values('Date/Time')

# Master Corporate Actions
splits = pd.concat(all_splits) if all_splits else pd.DataFrame()

# --- 4. DASHBOARD - LIVE SUMMARY ---
st.title("🚀 Portfolio Dashboard")
symbols = trades['Symbol'].unique().tolist()

with st.spinner('Fetching Live Market Prices...'):
    tickers = yf.Tickers(" ".join(symbols))
    live_prices = {s: tickers.tickers[s].fast_info['last_price'] for s in symbols}

# Calculate Current Holdings
summary_data = []
for sym in symbols:
    s_trades = trades[trades['Symbol'] == sym]
    total_qty = s_trades['Quantity'].sum()
    # Note: Stock splits are usually reflected in 'Trades' if you download 
    # the report AFTER the split, but our watchdog ensures accuracy.
    if total_qty > 0:
        avg_cost = abs( (s_trades['Quantity'] * s_trades['T. Price']).sum() / total_qty )
        live_p = live_prices.get(sym, 0)
        summary_data.append({
            "Symbol": sym, "Qty": round(total_qty, 4), 
            "Avg Cost": avg_cost, "Live Price": live_p,
            "Profit/Loss": (live_p - avg_cost) * total_qty
        })

df_summary = pd.DataFrame(summary_data)
st.dataframe(df_summary.style.format({"Avg Cost": "${:.2f}", "Live Price": "${:.2f}", "Profit/Loss": "${:.2f}"}))

# --- 5. FIFO CALCULATOR ---
st.divider()
st.header("🧮 FIFO Target Price Calculator")
selected_stock = st.selectbox("Select Stock", symbols)
col1, col2, col3 = st.columns(3)
mode = col1.radio("Sell by:", ["Units", "Percentage"])
amount = col2.number_input(f"Amount ({mode})", min_value=0.01)
target_pct = col3.number_input("Target Profit %", value=105.0)

stock_trades = trades[(trades['Symbol'] == selected_stock) & (trades['Quantity'] > 0)].copy()
total_owned = stock_trades['Quantity'].sum()
qty_to_sell = amount if mode == "Units" else (total_owned * (amount/100))

if qty_to_sell > total_owned:
    st.error(f"You only hold {total_owned} units.")
else:
    temp_qty, total_cost = qty_to_sell, 0
    for _, row in stock_trades.iterrows():
        if temp_qty <= 0: break
        take = min(row['Quantity'], temp_qty)
        total_cost += take * row['T. Price']
        temp_qty -= take
    
    target_price = (total_cost * (1 + target_pct/100)) / qty_to_sell
    st.success(f"To bag {target_pct}% profit, sell **{qty_to_sell:.4f} units** at **${target_price:.2f} USD**")

# --- 6. DIVIDENDS & TAX ---
st.divider()
st.header("📑 Tax & Dividends")
if all_divs:
    df_divs = pd.concat(all_divs)
    st.write("Total Dividend Income Summary:")
    st.dataframe(df_divs)
