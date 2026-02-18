import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

# --- 1. CONNECTION ---
conn = st.connection("gsheets", type=GSheetsConnection)

def safe_parse(df):
    sections = {}
    if df is None or df.empty: return sections
    df = df.astype(str).replace('nan', '').apply(lambda x: x.str.strip())
    for name in df.iloc[:, 0].unique():
        if name in ['', 'Statement', 'Field Name']: continue
        sec_df = df[df.iloc[:, 0] == name]
        h_row = sec_df[sec_df.iloc[:, 1] == 'Header']
        d_rows = sec_df[sec_df.iloc[:, 1] == 'Data']
        if not h_row.empty and not d_rows.empty:
            cols = [c for c in h_row.iloc[0, 2:].tolist() if c]
            data = d_rows.iloc[:, 2:2+len(cols)]
            data.columns = cols
            sections[name] = data
    return sections

# --- 2. DATA INGESTION ---
tabs = ["FY24", "FY25", "FY26"]
fy_data_map = {}
all_trades_list = []

for tab in tabs:
    try:
        raw = conn.read(worksheet=tab, ttl=0)
        parsed = safe_parse(raw)
        if parsed:
            fy_data_map[tab] = parsed
            if "Trades" in parsed: 
                t_df = parsed["Trades"]
                if 'DataDiscriminator' in t_df.columns:
                    t_df = t_df[t_df['DataDiscriminator'] == 'Order']
                all_trades_list.append(t_df)
    except: continue

# --- 3. FIFO ENGINE ---
df_lots = pd.DataFrame()
if all_trades_list:
    try:
        trades = pd.concat(all_trades_list, ignore_index=True)
        trades['Quantity'] = pd.to_numeric(trades['Quantity'], errors='coerce').fillna(0)
        trades['T. Price'] = pd.to_numeric(trades['T. Price'], errors='coerce').fillna(0)
        # Fix for commissions: Clean strings like '($1.00)' or '-1.00'
        trades['Comm'] = pd.to_numeric(trades['Comm in AUD'].astype(str).str.replace(r'[()$,]', '', regex=True), errors='coerce').fillna(0).abs()
        trades['Date/Time'] = pd.to_datetime(trades['Date/Time'].str.split(',').str[0], errors='coerce')
        trades = trades.dropna(subset=['Date/Time']).sort_values('Date/Time')

        # Split adjustments
        for tkr, dt in [('NVDA', '2024-06-10'), ('SMCI', '2024-10-01')]:
            trades.loc[(trades['Symbol'] == tkr) & (trades['Date/Time'] < dt), 'Quantity'] *= 10
            trades.loc[(trades['Symbol'] == tkr) & (trades['Date/Time'] < dt), 'T. Price'] /= 10

        today = pd.Timestamp.now()
        holdings = []
        for sym in trades['Symbol'].unique():
            lots = []
            for _, row in trades[trades['Symbol'] == sym].iterrows():
                if row['Quantity'] > 0: 
                    # Add commission to cost basis for buys
                    lots.append({'date': row['Date/Time'], 'qty': row['Quantity'], 'price': row['T. Price'], 'buy_comm': row['Comm']})
                elif row['Quantity'] < 0:
                    sq = abs(row['Quantity'])
                    while sq > 0 and lots:
                        if lots[0]['qty'] <= sq: sq -= lots[0].pop('qty'); lots.pop(0)
                        else: lots[0]['qty'] -= sq; sq = 0
            for l in lots:
                l['Symbol'] = sym
                l['Type'] = "Long-Term" if (today - l['date']).days > 365 else "Short-Term"
                holdings.append(l)
        df_lots = pd.DataFrame(holdings)
    except: pass

# --- 4. TOP PERFORMANCE BAR ---
st.title("🏦 Wealth Terminal Pro")
sel_fy = st.selectbox("Financial Year Performance", tabs, index=len(tabs)-1)
data_fy = fy_data_map.get(sel_fy, {})

def get_metric(section, keys, df_dict):
    if section not in df_dict: return 0.0
    df = df_dict[section]
    df = df[~df.apply(lambda r: r.astype(str).str.contains('Total|Subtotal', case=False).any(), axis=1)]
    target = next((c for c in df.columns if any(k.lower() in c.lower() for k in keys)), None)
    return pd.to_numeric(df[target], errors='coerce').sum() if target else 0.0

# Calculate FX vs Stock
stocks_pl = 0.0
forex_pl = 0.0
total_comm = 0.0

if 'Realized & Unrealized Performance Summary' in data_fy:
    perf = data_fy['Realized & Unrealized Performance Summary']
    perf = perf[~perf['Symbol'].str.contains('Total|Asset', case=False, na=False)]
    perf['Realized Total'] = pd.to_numeric(perf['Realized Total'], errors='coerce').fillna(0)
    if 'Asset Category' in perf.columns:
        stocks_pl = perf[perf['Asset Category'].str.contains('Stock|Equity', case=False, na=False)]['Realized Total'].sum()
        forex_pl = perf[perf['Asset Category'].str.contains('Forex|Cash', case=False, na=False)]['Realized Total'].sum()

if 'Trades' in data_fy:
    tr = data_fy['Trades']
    total_comm = pd.to_numeric(tr['Comm in AUD'].astype(str).str.replace(r'[()$,]', '', regex=True), errors='coerce').fillna(0).abs().sum()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Funds Injected", f"${get_metric('Deposits & Withdrawals', ['Amount'], data_fy):,.2f}")
m2.metric("Total Realized P/L", f"${(stocks_pl + forex_pl):,.2f}")
m3.metric("Total Commissions", f"${total_comm:,.2f}")
m4.metric("Dividends", f"${get_metric('Dividends', ['Amount'], data_fy):,.2f}")

with st.expander("📊 View FY Breakdown (Stocks / Forex / Net)"):
    ca, cb, cc = st.columns(3)
    ca.write("**Stock Realized P/L**")
    ca.subheader(f"${stocks_pl:,.2f}")
    cb.write("**Forex Realized P/L**")
    cb.subheader(f"${forex_pl:,.2f}")
    cc.write("**Net P/L (After Fees)**")
    cc.subheader(f"${(stocks_pl + forex_pl - total_comm):,.2f}")

# --- 5. STOCK BREAKDOWNS ---
st.divider()
cur_date = datetime.now().strftime('%d %b %Y')

if not df_lots.empty:
    unique_syms = sorted(df_lots['Symbol'].unique().tolist())
    prices = yf.download(unique_syms, period="1d")['Close'].iloc[-1].to_dict()
    if len(unique_syms) == 1: prices = {unique_syms[0]: prices}

    def show_sec(subset, label):
        st.markdown(f"### {label} (as of {cur_date})")
        if subset.empty: return st.info("No holdings found.")
        
        # Calculate Cost including buy commissions
        subset['Cost_No_Comm'] = subset['qty'] * subset['price']
        subset['Comm_Paid'] = subset.get('buy_comm', 0)
        
        agg = subset.groupby('Symbol').agg({'qty': 'sum', 'Cost_No_Comm': 'sum', 'Comm_Paid': 'sum'}).reset_index()
        agg['Avg Buy Price'] = agg['Cost_No_Comm'] / agg['qty']
        agg['Current Price'] = agg['Symbol'].map(prices).fillna(0)
        agg['Current Value'] = agg['qty'] * agg['Current Price']
        
        # P/L Logic
        agg['Gross P/L $'] = agg['Current Value'] - agg['Cost_No_Comm']
        agg['Net P/L $'] = agg['Gross P/L $'] - agg['Comm_Paid']
        agg['Net P/L %'] = (agg['Net P/L $'] / (agg['Cost_No_Comm'] + agg['Comm_Paid'])) * 100
        
        agg.index = range(1, len(agg) + 1)
        
        st.dataframe(agg.style.format({
            "Avg Buy Price": "${:.2f}", "Current Price": "${:.2f}", "Current Value": "${:.2f}", 
            "Comm_Paid": "${:.2f}", "Gross P/L $": "${:.2f}", "Net P/L $": "${:.2f}", "Net P/L %": "{:.2f}%"
        }).map(lambda x: 'color: green' if x > 0 else 'color: red', subset=['Net P/L $', 'Net P/L %']), use_container_width=True)
        st.write(f"**Total Net Value:** `${agg['Current Value'].sum():,.2f}` | **Total Commissions in this View:** `${agg['Comm_Paid'].sum():,.2f}`")

    show_sec(df_lots.copy(), "1. Current Global Holdings")
    show_sec(df_lots[df_lots['Type'] == "Short-Term"].copy(), "2. Short-Term Holdings")
    show_sec(df_lots[df_lots['Type'] == "Long-Term"].copy(), "3. Long-Term Holdings")
