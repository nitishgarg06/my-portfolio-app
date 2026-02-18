import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
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

# --- 3. FIFO ENGINE (INTERNAL SPLIT LOGIC) ---
df_lots = pd.DataFrame()
if all_trades:
    try:
        trades = pd.concat(all_trades, ignore_index=True)
        trades.columns = trades.columns.str.strip()
        
        # Identify Columns
        q_c = find_col(trades, ['Qty'])
        p_c = find_col(trades, ['Price'])
        c_c = find_col(trades, ['Comm'])
        d_c = find_col(trades, ['Date'])

        trades['qty_v'] = trades[q_c].apply(safe_num)
        trades['prc_v'] = trades[p_c].apply(safe_num)
        trades['cmm_v'] = trades[c_c].apply(safe_num).abs()
        trades['dt_v'] = pd.to_datetime(trades[d_c].str.split(',').str[0], errors='coerce')
        trades = trades.dropna(subset=['dt_v']).sort_values('dt_v')

        # --- INTERNAL SPLIT ADJUSTMENT ---
        if corporate_actions:
            ca_df = pd.concat(corporate_actions, ignore_index=True)
            # Filter for Forward/Reverse Splits
            splits = ca_df[ca_df['Description'].str.contains('Split', case=False, na=False)]
            
            for _, split in splits.iterrows():
                ticker = split['Symbol']
                # Parse ratio from description like "10 for 1"
                match = re.search(r'(\d+)\s+for\s+(\d+)', split['Description'])
                if match:
                    ratio = float(match.group(1)) / float(match.group(2))
                    split_dt = pd.to_datetime(split['Date/Time'].split(',')[0])
                    
                    # Adjust historical trades BEFORE this split
                    mask = (trades['Symbol'] == ticker) & (trades['dt_v'] < split_dt)
                    trades.loc[mask, 'qty_v'] *= ratio
                    trades.loc[mask, 'prc_v'] /= ratio

        # --- FIFO Calculation ---
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

# --- 4. TOP BAR (FIXED NET LOGIC) ---
st.title("🏦 Wealth Terminal Pro")
sel_fy = st.selectbox("Financial Year", tabs, index=len(tabs)-1)
data_fy = fy_map.get(sel_fy, {})

# Use the 'Net Realized P/L' directly from the performance summary
stocks_pl, forex_pl = 0.0, 0.0
perf_s = 'Realized & Unrealized Performance Summary'

if perf_s in data_fy:
    pdf = data_fy[perf_s]
    rt_c, ct_c = find_col(pdf, ['Realized Total']), find_col(pdf, ['Asset Category'])
    if rt_c and ct_c:
        pdf = pdf[~pdf['Symbol'].str.contains('Total|Asset', case=False, na=False)]
        stocks_pl = pdf[pdf[ct_c].str.contains('Stock', case=False)] [rt_c].apply(safe_num).sum()
        forex_pl = pdf[pdf[ct_c].str.contains('Forex|Cash', case=False)] [rt_c].apply(safe_num).sum()

m1, m2, m3 = st.columns(3)
# a) Confirmed: Realized Total is already NET of commissions
m1.metric("Net Realized P/L", f"${(stocks_pl + forex_pl):,.2f}")
m2.metric("Stocks Portion", f"${stocks_pl:,.2f}")
m3.metric("Forex/Cash Portion", f"${forex_pl:,.2f}")

# --- 5. HOLDINGS TABLES & CALCULATOR ---
# (Rest of existing logic for rendering tables and FIFO calculator)
