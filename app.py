import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

# --- 1. CONNECTION & ROBUST TOOLS ---
conn = st.connection("gsheets", type=GSheetsConnection)

def safe_num(val):
    """Deep cleans currency strings and handles parentheses for negatives."""
    if val is None or pd.isna(val) or val == '': return 0.0
    s = str(val).strip().replace('$', '').replace(',', '')
    if '(' in s and ')' in s: 
        s = '-' + s.replace('(', '').replace(')', '')
    try: return float(s)
    except: return 0.0

def safe_find(df, keywords):
    """Finds a column name regardless of extra spaces or case."""
    for col in df.columns:
        if any(k.lower() in str(col).lower() for k in keywords):
            return col
    return None

def parse_ibkr_grid(df):
    """Standardizes the IBKR table grid for the app."""
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
corporate_actions = []

for t in tabs:
    try:
        raw = conn.read(worksheet=t, ttl=0)
        parsed = parse_ibkr_grid(raw)
        if parsed:
            fy_map[t] = parsed
            if "Trades" in parsed: all_trades.append(parsed["Trades"])
            if "Corporate Actions" in parsed: corporate_actions.append(parsed["Corporate Actions"])
    except: continue

# --- 3. FIFO ENGINE ---
df_lots = pd.DataFrame()
if all_trades:
    try:
        trades = pd.concat(all_trades, ignore_index=True)
        trades.columns = trades.columns.str.strip()
        
        q_c = safe_find(trades, ['Qty', 'Quantity'])
        p_c = safe_find(trades, ['Price', 'T. Price'])
        c_c = safe_find(trades, ['Comm', 'Commission'])
        d_c = safe_find(trades, ['Date', 'Time'])

        trades['qty_v'] = trades[q_c].apply(safe_num)
        trades['prc_v'] = trades[p_c].apply(safe_num)
        trades['cmm_v'] = trades[c_c].apply(safe_num).abs()
        trades['dt_v'] = pd.to_datetime(trades[d_c].str.split(',').str[0], errors='coerce')
        trades = trades.dropna(subset=['dt_v']).sort_values('dt_v')

        # Fallback for Splits
        for tkr, dt in [('NVDA', '2024-06-10'), ('SMCI', '2024-10-01')]:
            mask = (trades['Symbol'] == tkr) & (trades['dt_v'] < pd.to_datetime(dt))
            if not trades[mask].empty:
                trades.loc[mask, 'qty_v'] *= 10
                trades.loc[mask, 'prc_v'] /= 10

        holdings = []
        for sym in trades['Symbol'].unique():
            lots = []
            sym_trades = trades[trades['Symbol'] == sym]
            for _, row in sym_trades.iterrows():
                if row['qty_v'] > 0: 
                    lots.append({'date': row['dt_v'], 'qty': row['qty_v'], 'price':
