import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

# --- 1. CONNECTION & CLEANERS ---
conn = st.connection("gsheets", type=GSheetsConnection)

def safe_num(val):
    if val is None or pd.isna(val) or val == '': return 0.0
    s = str(val).strip().replace('$', '').replace(',', '')
    if '(' in s and ')' in s: s = '-' + s.replace('(', '').replace(')', '')
    try: return float(s)
    except: return 0.0

def find_col(df, keywords):
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

# --- 2. DATA INGESTION ---
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

# --- 3. DYNAMIC FIFO ENGINE (WITH AUTOMATIC SPLITS) ---
df_lots = pd.DataFrame()
if all_trades:
    try:
        trades = pd.concat(all_trades, ignore_index=True)
        trades.columns = trades.columns.str.strip()
        
        q_c, p_c, c_c, d_c = find_col(trades, ['Qty']), find_col(trades, ['Price']), find_col(trades, ['Comm']), find_col(trades, ['Date'])

        trades['qty_v'] = trades[q_c].apply(safe_num)
        trades['prc_v'] = trades[p_c].apply(safe_num)
        trades['cmm_v'] = trades[c_c].apply(safe_num).abs()
        trades['dt_v'] = pd.to_datetime(trades[d_c].str.split(',').str[0], errors='coerce')
        trades = trades.dropna(subset=['dt_v']).sort_values('dt_v')

        # DYNAMIC SPLIT ADJUSTMENT
        unique_tickers = trades['Symbol'].unique()
        for ticker in unique_tickers:
            try:
                stock_obj = yf.Ticker(ticker)
                splits = stock_obj.splits
                if not splits.empty:
                    for split_date, ratio in splits.items():
                        # Adjust trades that happened BEFORE the split date
                        trades.loc[(trades['Symbol'] == ticker) & (trades['dt_v'].dt.tz_localize(None) < split_date.tz_localize(None)), 'qty_v'] *= ratio
                        trades.loc[(trades['Symbol'] == ticker) & (trades['dt_v'].dt.tz_localize(None) < split_date.tz_localize(None)), 'prc_v'] /= ratio
            except: continue

        holdings = []
        for sym in trades['Symbol'].unique():
            lots = []
            for _, row in trades[trades['Symbol'] == sym].iterrows():
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

# --- 4. TOP BAR (CORRECTED NET LOGIC) ---
st.title("🏦 Wealth Terminal Pro")
sel_fy = st.selectbox("Financial Year View", tabs, index=len(tabs)-1)
data_fy = fy_map.get(sel_fy, {})

stocks_pl, forex_pl, fy_comms = 0.0, 0.0, 0.0
perf_s = 'Realized & Unrealized Performance Summary'

if perf_s in data_fy:
    pdf = data_fy[perf_s]
    rt_c, ct_c = find_col(pdf, ['Realized Total']), find_col(pdf, ['Asset Category'])
    if rt_c and ct_c:
        pdf = pdf[~pdf['Symbol'].str.contains('Total|Asset', case=False, na=False)]
        stocks_pl = pdf[pdf[ct_c].str.contains('Stock', case=False)] [rt_c].apply(safe_num).sum()
        forex_pl = pdf[pdf[ct_c].str.contains('Forex|Cash', case=False)] [rt_c].apply(safe_num).sum()

if 'Trades' in data_fy:
    fy_comms = data_fy['Trades'][find_col(data_fy['Trades'], ['Comm'])].apply(safe_num).abs().sum()

# Total Investment Calculation
total_inv = (df_lots['qty'] * df_lots['price']).sum() + df_lots['comm'].sum() if not df_lots.empty else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Investment", f"${total_inv:,.2f}")
m2.metric("Net Realized P/L", f"${(stocks_pl + forex_pl):,.2f}") # a) Already net of comms
m3.metric("FY Commissions Paid", f"${fy_comms:,.2f}")
m4.metric("FY Dividends", f"${get_metric('Dividends', ['Amount'], data_fy):,.2f}" if 'Dividends' in data_fy else "$0.00")

# --- 5. HOLDINGS TABLES & CALCULATOR ---
# (Existing table and calculator code follows here, already utilizing the dynamic splits)
