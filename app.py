import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="IBKR Portfolio Pro", layout="wide", page_icon="📈")

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
            cols = [c for c in header_row.iloc[0, 2:].tolist() if c and c != 'nan']
            data = sec_df[sec_df.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(cols)]
            data.columns = cols
            sections[name] = data
    return sections

# --- 2. GLOBAL DATA LOADING ---
tabs_to_read = ["FY24", "FY25", "FY26"]
fy_data_map = {}
all_trades_list = []
all_perf_list = []

for tab in tabs_to_read:
    try:
        raw_df = conn.read(worksheet=tab, ttl=0)
        parsed = parse_ibkr_sheet(raw_df)
        fy_data_map[tab] = parsed
        if "Trades" in parsed: all_trades_list.append(parsed["Trades"])
        if "Realized & Unrealized Performance Summary" in parsed: 
            all_perf_list.append(parsed["Realized & Unrealized Performance Summary"])
    except Exception:
        continue # Skip missing tabs or empty sheets

st.title("🚀 IBKR Portfolio Pro-Analyst")

# --- 3. INVESTMENT SUMMARY (FY SPECIFIC) ---
st.header("💰 Investment Capital")
selected_fy = st.selectbox("Select Financial Year for Funds Injected", tabs_to_read, index=len(tabs_to_read)-1)

data_inv = fy_data_map.get(selected_fy, {})
if "Deposits & Withdrawals" in data_inv:
    dw = data_inv["Deposits & Withdrawals"]
    # Identify the correct column name for Amount (sometimes it varies)
    amt_col = 'Amount' if 'Amount' in dw.columns else dw.columns[-1]
    dw[amt_col] = pd.to_numeric(dw[amt_col], errors='coerce').fillna(0)
    
    # Filter out rows that are 'Total' summaries
    actual_funds = dw[~dw.apply(lambda row: row.astype(str).str.contains('Total', case=False).any(), axis=1)]
    total_inv = actual_funds[amt_col].sum()
    st.metric(f"Net Funds Injected ({selected_fy})", f"${total_inv:,.2f}")
else:
    st.info(f"No deposit data found for {selected_fy}.")

# --- 4. HOLDINGS & CGT BREAKDOWN ---
st.divider()
st.header("📊 Stock Summary & CGT Status")

if all_trades_list:
    df_trades = pd.concat(all_trades_list, ignore_index=True)
    df_trades['Quantity'] = pd.to_numeric(df_trades['Quantity'], errors='coerce')
    df_trades['T. Price'] = pd.to_numeric(df_trades['T. Price'], errors='coerce')
    df_trades['Date/Time'] = pd.to_datetime(df_trades['Date/Time'])
    df_trades = df_trades.sort_values('Date/Time')

    # Realized Profit Logic
    realized_map = {}
    if all_perf_list:
        df_perf = pd.concat(all_perf_list, ignore_index=True)
        if 'Realized Total' in df_perf.columns:
            df_perf['Realized Total'] = pd.to_numeric(df_perf['Realized Total'], errors='coerce').fillna(0)
            realized_map = df_perf.groupby('Symbol')['Realized Total'].sum().to_dict()

    symbols = df_trades['Symbol'].unique().tolist()
    today = pd.Timestamp.now()
    summary_list = []

    for sym in symbols:
        lots = [] 
        stock_history = df_trades[df_trades['Symbol'] == sym]
        for _, row in stock_history.iterrows():
            q = row['Quantity']
            if q > 0: lots.append({'date': row['Date/Time'], 'qty': q, 'price': row['T. Price']})
            elif q < 0:
                sell_q = abs(q)
                while sell_q > 0 and lots:
                    if lots[0]['qty'] <= sell_q:
                        sell_q -= lots[0]['qty']
                        lots.pop(0)
                    else:
                        lots[0]['qty'] -= sell_q
                        sell_q = 0
        
        lt_qty, st_qty, total_cost = 0, 0, 0
        for lot in lots:
            total_cost += (lot['qty'] * lot['price'])
            if (today - lot['date']).days > 365: lt_qty += lot['qty']
            else: st_qty += lot['qty']
        
        total_qty = lt_qty + st_qty
        if total_qty > 0.001:
            avg_cost = total_cost / total_qty
            summary_list.append({
                "Symbol": sym, "Total Qty": round(total_qty, 4),
                "Long-Term (LT)": round(lt_qty, 4), "Short-Term (ST)": round(st_qty, 4),
                "Avg Cost": avg_cost, "Realized Profit": realized_map.get(sym, 0)
            })

    if summary_list:
        df_final = pd.DataFrame(summary_list)
        st.dataframe(df_final.style.format({"Avg Cost": "${:.2f}", "Realized Profit": "${:.2f}"}))
        st.info(f"**CGT Summary:** {df_final['Long-Term (LT)'].sum():.2f} units are Long-Term (50% CGT Discount) and {df_final['Short-Term (ST)'].sum():.2f} units are Short-Term.")
    else:
        st.warning("No open positions found.")
