import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

# --- 1. CONNECTION & UTILS ---
conn = st.connection("gsheets", type=GSheetsConnection)

def safe_num(val):
    if val is None or pd.isna(val) or val == '': return 0.0
    s = str(val).strip().replace('$', '').replace(',', '')
    if '(' in s and ')' in s: s = '-' + s.replace('(', '').replace(')', '')
    try: return float(s)
    except: return 0.0

def safe_find(df, keywords):
    for col in df.columns:
        if any(k.lower() in str(col).lower() for k in keywords): return col
    return None

def parse_ibkr_grid(df):
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

# --- 2. DATA LOADING ---
tabs = ["FY24", "FY25", "FY26"]
fy_map = {}
all_trades = []

for t in tabs:
    try:
        raw = conn.read(worksheet=t, ttl=0)
        parsed = parse_ibkr_grid(raw)
        if parsed:
            fy_map[t] = parsed
            if "Trades" in parsed: all_trades.append(parsed["Trades"])
    except: continue

# --- 3. FIFO ENGINE ---
df_lots = pd.DataFrame()
if all_trades:
    try:
        trades = pd.concat(all_trades, ignore_index=True)
        trades.columns = trades.columns.str.strip()
        q_c, p_c, c_c, d_c = safe_find(trades, ['Qty']), safe_find(trades, ['Price']), safe_find(trades, ['Comm']), safe_find(trades, ['Date'])
        trades['qty_v'] = trades[q_c].apply(safe_num)
        trades['prc_v'] = trades[p_c].apply(safe_num)
        trades['cmm_v'] = trades[c_c].apply(safe_num).abs()
        trades['dt_v'] = pd.to_datetime(trades[d_c].str.split(',').str[0], errors='coerce')
        trades = trades.dropna(subset=['dt_v']).sort_values('dt_v')

        # Fallback Splits (NVDA/SMCI)
        for tkr, dt in [('NVDA', '2024-06-10'), ('SMCI', '2024-10-01')]:
            mask = (trades['Symbol'] == tkr) & (trades['dt_v'] < pd.to_datetime(dt))
            if not trades[mask].empty:
                trades.loc[mask, 'qty_v'] *= 10; trades.loc[mask, 'prc_v'] /= 10

        holdings = []
        for sym in trades['Symbol'].unique():
            lots = []
            sym_trades = trades[trades['Symbol'] == sym]
            for _, row in sym_trades.iterrows():
                if row['qty_v'] > 0: 
                    lots.append({'date': row['dt_v'], 'qty': row['qty_v'], 'price': row['prc_v'], 'comm': row['cmm_v']})
                elif row['qty_v'] < 0:
                    sq = abs(row['qty_v'])
                    while sq > 0 and lots:
                        if lots[0]['qty'] <= sq: sq -= lots[0].pop('qty'); lots.pop(0)
                        else: lots[0]['qty'] -= sq; sq = 0
            for l in lots:
                l['Symbol'] = sym
                l['Type'] = "Long-Term" if (pd.Timestamp.now() - l['date']).days > 365 else "Short-Term"
                holdings.append(l)
        df_lots = pd.DataFrame(holdings)
    except: pass

# --- 4. START RENDERING ---
st.title("🏦 Wealth Terminal Pro")
curr_date = datetime.now().strftime('%d %b %Y')

# --- 5. SAFE LIVE PRICE FETCHING ---
prices = {}
if not df_lots.empty:
    unique_tickers = sorted(df_lots['Symbol'].unique())
    # Default to cost basis in case fetching fails
    for tkr in unique_tickers:
        prices[tkr] = df_lots[df_lots['Symbol'] == tkr]['price'].mean()
        
    try:
        # Request data with a timeout
        data = yf.download(unique_tickers, period="1d", timeout=5)['Close']
        if not data.empty:
            for tkr in unique_tickers:
                # Handle single vs multiple ticker returns
                val = data[tkr].iloc[-1] if len(unique_tickers) > 1 else data.iloc[-1]
                if not pd.isna(val):
                    prices[tkr] = val
    except Exception as e:
        st.sidebar.warning(f"Live Price Server Busy. Using Cost Basis for P/L.")

# --- 6. TOP METRICS ---
if fy_map:
    sel_fy = st.selectbox("Financial Year View", tabs, index=len(tabs)-1)
    data_fy = fy_map.get(sel_fy, {})
    stocks_pl, forex_pl = 0.0, 0.0
    perf_s = 'Realized & Unrealized Performance Summary'
    if perf_s in data_fy:
        pdf = data_fy[perf_s]
        rt_c, ct_c = safe_find(pdf, ['Realized Total']), safe_find(pdf, ['Asset Category'])
        if rt_c and ct_c:
            clean_p = pdf[~pdf['Symbol'].str.contains('Total|Asset', case=False, na=False)]
            stocks_pl = clean_p[clean_p[ct_c].str.contains('Stock', case=False)][rt_c].apply(safe_num).sum()
            forex_pl = clean_p[clean_p[ct_c].str.contains('Forex|Cash', case=False)][rt_c].apply(safe_num).sum()

    total_inv = (df_lots['qty'] * df_lots['price']).sum() + df_lots['comm'].sum() if not df_lots.empty else 0.0
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Investment", f"${total_inv:,.2f}")
    m2.metric("Net Realized P/L", f"${(stocks_pl + forex_pl):,.2f}")
    m3.metric("Forex/Cash Impact", f"${forex_pl:,.2f}")

# --- 7. HOLDINGS TABLES ---
st.divider()

def render_table(subset, label):
    st.subheader(f"{label} (as of {curr_date})")
    if subset.empty: 
        st.info(f"No {label} positions.")
        return
    
    agg = subset.groupby('Symbol').agg({'qty': 'sum', 'price': 'mean', 'comm': 'sum'}).reset_index()
    agg['Live Price'] = agg['Symbol'].map(prices)
    agg['Total Basis'] = (agg['qty'] * agg['price']) + agg['comm']
    agg['Market Value'] = agg['qty'] * agg['Live Price']
    agg['P/L $'] = agg['Market Value'] - agg['Total Basis']
    agg['P/L %'] = (agg['P/L $'] / agg['Total Basis']) * 100
    
    agg.columns = ['Ticker', 'Units', 'Avg Cost', 'Comms', 'Live Price', 'Total Basis', 'Market Value', 'P/L $', 'P/L %']
    agg.index = range(1, len(agg) + 1)
    
    st.dataframe(agg.style.format({
        "Units": "{:.2f}", "Avg Cost": "${:.2f}", "Comms": "${:.2f}", "Live Price": "${:.2f}",
        "Total Basis": "${:.2f}", "Market Value": "${:.2f}", "P/L $": "${:.2f}", "P/L %": "{:.2f}%"
    }).map(lambda x: 'color: green' if x > 0 else 'color: red', subset=['P/L $', 'P/L %']), use_container_width=True)

if not df_lots.empty:
    render_table(df_lots.copy(), "1. Current Global Holdings")
    render_table(df_lots[df_lots['Type'] == "Short-Term"].copy(), "2. Short-Term Holdings")
    render_table(df_lots[df_lots['Type'] == "Long-Term"].copy(), "3. Long-Term Holdings")
