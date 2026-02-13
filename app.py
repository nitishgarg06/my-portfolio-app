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
        realized = pd.to_numeric(perf['Realized Total'], errors='coerce').sum()
    col2.metric("Realized Profit", f"${realized:,.2f}", delta_color="normal")

    # Live Price logic for Total Value
    if not df_lots.empty:
        unique_syms = df_lots['Symbol'].unique().tolist()
        prices = yf.download(unique_syms, period="1d")['Close'].iloc[-1].to_dict()
        if len(unique_syms) == 1: prices = {unique_syms[0]: prices}
        
        df_lots['Value'] = df_lots['qty'] * df_lots['Symbol'].map(prices)
        total_val = df_lots['Value'].sum()
        col3.metric("Current Portfolio Value", f"${total_val:,.2f}")

# --- 5. TABS FOR STOCK BREAKDOWN ---
st.markdown('<p class="section-header">Portfolio Breakdown</p>', unsafe_allow_stdio=True)

tab1, tab2, tab3 = st.tabs(["🌎 All Holdings", "⏳ Short-Term", "💎 Long-Term"])

def style_df(df_in):
    if df_in.empty: return "No holdings."
    df_agg = df_in.groupby('Symbol').agg({'qty': 'sum', 'qty': 'sum', 'price': 'mean'}).reset_index() # Simplified for display
    # (Re-calculate the full metrics here as per previous logic)
    df_agg['Cost'] = df_in.groupby('Symbol').apply(lambda x: (x['qty']*x['price']).sum()).values
    df_agg['Avg Price'] = df_agg['Cost'] / df_agg['qty']
    df_agg['Live Price'] = df_agg['Symbol'].map(prices)
    df_agg['Value'] = df_agg['qty'] * df_agg['Live Price']
    df_agg['P/L'] = df_agg['Value'] - df_agg['Cost']
    df_agg['P/L %'] = (df_agg['P/L'] / df_agg['Cost']) * 100
    
    return df_agg[['Symbol', 'qty', 'Avg Price', 'Live Price', 'Value', 'P/L', 'P/L %']].style.format({
        "Avg Price": "${:.2f}", "Live Price": "${:.2f}", "Value": "${:.2f}", "P/L": "${:.2f}", "P/L %": "{:.2f}%"
    }).applymap(lambda x: 'color: #10b981' if x > 0 else 'color: #ef4444', subset=['P/L', 'P/L %'])

with tab1: st.dataframe(style_df(df_lots.copy()), use_container_width=True)
with tab2: st.dataframe(style_df(df_lots[df_lots['Type'] == "Short-Term"].copy()), use_container_width=True)
with tab3: st.dataframe(style_df(df_lots[df_lots['Type'] == "Long-Term"].copy()), use_container_width=True)

# --- 6. FIFO CALCULATOR WITH NEW DESIGN ---
st.divider()
with st.expander("🧮 Open FIFO Calculator Tool", expanded=True):
    c1, c2 = st.columns([1, 2])
    sel_stock = c1.selectbox("Stock", unique_syms)
    stock_lots = df_lots[df_lots['Symbol'] == sel_stock].sort_values('date')
    
    total_owned = stock_lots['qty'].sum()
    amt = c2.slider(f"Units of {sel_stock} to sell", 0.0, float(total_owned), float(total_owned*0.25))
    
    target_pct = st.number_input("Target Profit %", value=105.0)
    
    # Calculation Logic
    tmp_q, s_cost = amt, 0
    for _, l in stock_lots.iterrows():
        if tmp_q <= 0: break
        take = min(l['qty'], tmp_q)
        s_cost += take * l['price']
        tmp_q -= take
    
    if amt > 0:
        target_p = (s_cost * (1 + target_pct/100)) / amt
        st.success(f"To hit {target_pct}% profit: Sell at **${target_p:.2f}**")
