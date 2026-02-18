import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

# --- 1. CONNECTION ---
conn = st.connection("gsheets", type=GSheetsConnection)

def universal_clean(val):
    """Extremely aggressive cleaner for any numeric or date string."""
    if val is None or pd.isna(val): return 0.0
    s = str(val).strip().replace('$', '').replace(',', '')
    # Handle IBKR's (1.00) notation for negative numbers
    if '(' in s and ')' in s:
        s = '-' + s.replace('(', '').replace(')', '')
    try:
        return float(s)
    except:
        return 0.0

def safe_find_col(df, keywords):
    """Finds column names even with extra spaces or different casing."""
    for col in df.columns:
        if any(key.lower() in str(col).lower() for key in keywords):
            return col
    return None

def parse_ibkr_grid(df):
    """Parses the raw IBKR sheet into distinct sections."""
    sections = {}
    if df is None or df.empty: return sections
    df = df.astype(str).replace('nan', '')
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
        parsed = parse_ibkr_grid(raw)
        if parsed:
            fy_data_map[tab] = parsed
            if "Trades" in parsed:
                t_df = parsed["Trades"]
                if 'DataDiscriminator' in t_df.columns:
                    t_df = t_df[t_df['DataDiscriminator'] == 'Order']
                all_trades_list.append(t_df)
    except:
        continue

# --- 3. FIFO ENGINE ---
df_lots = pd.DataFrame()
if all_trades_list:
    try:
        trades = pd.concat(all_trades_list, ignore_index=True)
        # Dynamic Column Discovery
        q_col = safe_find_col(trades, ['Quantity'])
        p_col = safe_find_col(trades, ['Price', 'T. Price'])
        c_col = safe_find_col(trades, ['Comm', 'Commission'])
        d_col = safe_find_col(trades, ['Date/Time', 'Date'])

        trades['qty'] = trades[q_col].apply(universal_clean)
        trades['prc'] = trades[p_col].apply(universal_clean)
        trades['cmm'] = trades[c_col].apply(universal_clean).abs()
        
        # Clean Date Strings
        trades['dt'] = pd.to_datetime(trades[d_col].str.split(',').str[0], errors='coerce')
        trades = trades.dropna(subset=['dt']).sort_values('dt')

        # Split adjustments
        for tkr, split_dt in [('NVDA', '2024-06-10'), ('SMCI', '2024-10-01')]:
            trades.loc[(trades['Symbol'] == tkr) & (trades['dt'] < split_dt), 'qty'] *= 10
            trades.loc[(trades['Symbol'] == tkr) & (trades['dt'] < split_dt), 'prc'] /= 10

        holdings = []
        for sym in trades['Symbol'].unique():
            lots = []
            for _, row in trades[trades['Symbol'] == sym].iterrows():
                if row['qty'] > 0: 
                    lots.append({'date': row['dt'], 'qty': row['qty'], 'price': row['prc'], 'comm': row['cmm']})
                elif row['qty'] < 0:
                    sq = abs(row['qty'])
                    while sq > 0 and lots:
                        if lots[0]['qty'] <= sq:
                            sq -= lots[0].pop('qty')
                            lots.pop(0)
                        else:
                            lots[0]['qty'] -= sq
                            sq = 0
            for l in lots:
                l['Symbol'] = sym
                l['Type'] = "Long-Term" if (pd.Timestamp.now() - l['date']).days > 365 else "Short-Term"
                holdings.append(l)
        df_lots = pd.DataFrame(holdings)
    except:
        pass

# --- 4. DASHBOARD ---
st.title("🏦 Wealth Terminal Pro")
sel_fy = st.selectbox("Financial Year View", tabs, index=len(tabs)-1)
data_fy = fy_data_map.get(sel_fy, {})

def get_metric(section, keys, df_dict):
    if section not in df_dict: return 0.0
    df = df_dict[section]
    col = safe_find_col(df, keys)
    if not col: return 0.0
    df = df[~df.apply(lambda r: r.astype(str).str.contains('Total', case=False).any(), axis=1)]
    return df[col].apply(universal_clean).sum()

# Metrics
funds = get_metric('Deposits & Withdrawals', ['Amount'], data_fy)
pl = get_metric('Realized & Unrealized Performance Summary', ['Realized Total'], data_fy)
comms = get_metric('Trades', ['Comm', 'Commission'], data_fy)
divs = get_metric('Dividends', ['Amount'], data_fy)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Funds Injected", f"${funds:,.2f}")
m2.metric("Total Realized P/L", f"${pl:,.2f}")
m3.metric("Commissions", f"${abs(comms):,.2f}")
m4.metric("Dividends", f"${divs:,.2f}")

# --- 5. HOLDINGS TABLES ---
st.divider()
if not df_lots.empty:
    unique_syms = sorted(df_lots['Symbol'].unique().tolist())
    try:
        prices = yf.download(unique_syms, period="1d")['Close'].iloc[-1].to_dict()
        if len(unique_syms) == 1: prices = {unique_syms[0]: prices}
    except: prices = {s: 0.0 for s in unique_syms}

    def render_table(subset, label):
        st.subheader(f"{label} (as of {datetime.now().strftime('%d %b %Y')})")
        if subset.empty: return st.info("No positions.")
        
        subset['Cost'] = subset['qty'] * subset['price']
        agg = subset.groupby('Symbol').agg({'qty': 'sum', 'Cost': 'sum', 'comm': 'sum'}).reset_index()
        agg['Avg Buy'] = agg['Cost'] / agg['qty']
        agg['Price'] = agg['Symbol'].map(prices).fillna(0.0)
        agg['Value'] = agg['qty'] * agg['Price']
        agg['Net P/L $'] = (agg['Value'] - agg['Cost']) - agg['comm']
        agg['Net P/L %'] = (agg['Net P/L $'] / (agg['Cost'] + agg['comm'])) * 100
        
        agg.index = range(1, len(agg) + 1)
        st.dataframe(agg.style.format({
            "Avg Buy": "${:.2f}", "Price": "${:.2f}", "Value": "${:.2f}", 
            "comm": "${:.2f}", "Net P/L $": "${:.2f}", "Net P/L %": "{:.2f}%"
        }).map(lambda x: 'color: green' if x > 0 else 'color: red', subset=['Net P/L $', 'Net P/L %']), use_container_width=True)

    render_table(df_lots.copy(), "1. Current Global Holdings")
