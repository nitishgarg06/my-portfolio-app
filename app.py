import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

# Minimal CSS to avoid TypeErrors seen in logs
st.markdown("<style>.metric-label { font-size: 16px; font-weight: bold; }</style>", unsafe_allow_stdio=True)

# --- 1. CONNECTION ---
conn = st.connection("gsheets", type=GSheetsConnection)

def clean_val(series):
    """Deep cleans currency strings and handles parentheses for negatives."""
    if series is None: return pd.Series(0.0)
    # Remove $, commas, and handle '(1.00)' as -1.00
    cleaned = series.astype(str).str.replace(r'[$,]', '', regex=True)
    cleaned = cleaned.str.replace(r'\(', '-', regex=True).str.replace(r'\)', '', regex=True)
    return pd.to_numeric(cleaned, errors='coerce').fillna(0.0)

def find_col(df, keywords):
    """Finds a column name in a dataframe based on keywords (case-insensitive)."""
    for col in df.columns:
        if any(key.lower() in col.lower() for key in keywords):
            return col
    return None

def safe_parse(df):
    """Robust parser for IBKR grid structure."""
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
    except Exception:
        continue

# --- 3. FIFO ENGINE ---
df_lots = pd.DataFrame()
if all_trades_list:
    try:
        trades = pd.concat(all_trades_list, ignore_index=True)
        # Dynamic Column Finding
        qty_col = find_col(trades, ['Quantity'])
        prc_col = find_col(trades, ['Price', 'T. Price'])
        cmm_col = find_col(trades, ['Comm', 'Commission'])
        dt_col = find_col(trades, ['Date/Time', 'Date'])

        trades['Quantity'] = clean_val(trades.get(qty_col))
        trades['T. Price'] = clean_val(trades.get(prc_col))
        trades['Comm'] = clean_val(trades.get(cmm_col)).abs()
        
        trades['Date/Time'] = pd.to_datetime(trades[dt_col].str.split(',').str[0], errors='coerce')
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
                    lots.append({'date': row['Date/Time'], 'qty': row['Quantity'], 'price': row['T. Price'], 'buy_comm': row['Comm']})
                elif row['Quantity'] < 0:
                    sq = abs(row['Quantity'])
                    while sq > 0 and lots:
                        if lots[0]['qty'] <= sq:
                            sq -= lots[0].pop('qty')
                            lots.pop(0)
                        else:
                            lots[0]['qty'] -= sq
                            sq = 0
            for l in lots:
                l['Symbol'] = sym
                l['Type'] = "Long-Term" if (today - l['date']).days > 365 else "Short-Term"
                holdings.append(l)
        df_lots = pd.DataFrame(holdings)
    except Exception as e:
        st.error(f"FIFO Engine Error: {e}")

# --- 4. TOP PERFORMANCE BAR ---
st.title("🏦 Wealth Terminal Pro")
sel_fy = st.selectbox("Financial Year Performance", tabs, index=len(tabs)-1)
data_fy = fy_data_map.get(sel_fy, {})

def get_fy_metric(section, keywords, df_dict):
    if section not in df_dict: return 0.0
    df = df_dict[section]
    col = find_col(df, keywords)
    if not col: return 0.0
    # Filter out total rows
    df = df[~df.apply(lambda r: r.astype(str).str.contains('Total|Subtotal', case=False).any(), axis=1)]
    return clean_val(df[col]).sum()

funds = get_fy_metric('Deposits & Withdrawals', ['Amount'], data_fy)
stocks_pl = get_fy_metric('Realized & Unrealized Performance Summary', ['Realized Total'], data_fy)
# Commissions are usually in 'Trades' section
comms = get_fy_metric('Trades', ['Comm', 'Commission'], data_fy).abs()
divs = get_fy_metric('Dividends', ['Amount'], data_fy)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Funds Injected", f"${funds:,.2f}")
m2.metric("Total Realized P/L", f"${stocks_pl:,.2f}")
m3.metric("Total Commissions", f"${comms:,.2f}")
m4.metric("Dividends", f"${divs:,.2f}")

with st.expander("📊 View Net P/L Breakdown"):
    st.write(f"**Net Profit/Loss (After Fees):** `${(stocks_pl - comms):,.2f}`")

# --- 5. STOCK BREAKDOWNS ---
st.divider()
cur_date = datetime.now().strftime('%d %b %Y')

if not df_lots.empty:
    unique_syms = sorted(df_lots['Symbol'].unique().tolist())
    try:
        prices = yf.download(unique_syms, period="1d")['Close'].iloc[-1].to_dict()
        if len(unique_syms) == 1: prices = {unique_syms[0]: prices}
    except: prices = {s: 0.0 for s in unique_syms}

    def show_sec(subset, label):
        st.markdown(f"### {label} (as of {cur_date})")
        if subset.empty: return st.info("No holdings.")
        
        subset['Cost_No_Comm'] = subset['qty'] * subset['price']
        subset['Comm_Paid'] = subset.get('buy_comm', 0.0)
        
        agg = subset.groupby('Symbol').agg({'qty': 'sum', 'Cost_No_Comm': 'sum', 'Comm_Paid': 'sum'}).reset_index()
        agg['Avg Buy Price'] = agg['Cost_No_Comm'] / agg['qty']
        agg['Current Price'] = agg['Symbol'].map(prices).fillna(0.0)
        agg['Current Value'] = agg['qty'] * agg['Current Price']
        agg['Net P/L $'] = (agg['Current Value'] - agg['Cost_No_Comm']) - agg['Comm_Paid']
        agg['Net P/L %'] = (agg['Net P/L $'] / (agg['Cost_No_Comm'] + agg['Comm_Paid'])) * 100
        
        agg.index = range(1, len(agg) + 1)
        st.dataframe(agg.style.format({
            "Avg Buy Price": "${:.2f}", "Current Price": "${:.2f}", "Current Value": "${:.2f}", 
            "Comm_Paid": "${:.2f}", "Net P/L $": "${:.2f}", "Net P/L %": "{:.2f}%"
        }).map(lambda x: 'color: green' if x > 0 else 'color: red', subset=['Net P/L $', 'Net P/L %']), use_container_width=True)

    show_sec(df_lots.copy(), "1. Current Global Holdings")
