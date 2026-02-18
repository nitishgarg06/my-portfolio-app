import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import re
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

# --- 1. CONNECTION & ROBUST UTILS ---
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
    """Deep scan for IBKR Header/Data rows."""
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

# --- 3. FIFO ENGINE (WITH FALLBACK SPLITS) ---
df_lots = pd.DataFrame()
if all_trades:
    try:
        trades = pd.concat(all_trades, ignore_index=True)
        trades.columns = trades.columns.str.strip()
        
        q_c = find_col(trades, ['Qty'])
        p_c = find_col(trades, ['Price'])
        c_c = find_col(trades, ['Comm'])
        d_c = find_col(trades, ['Date'])

        trades['qty_v'] = trades[q_c].apply(safe_num)
        trades['prc_v'] = trades[p_c].apply(safe_num)
        trades['cmm_v'] = trades[c_c].apply(safe_num).abs()
        trades['dt_v'] = pd.to_datetime(trades[d_c].str.split(',').str[0], errors='coerce')
        trades = trades.dropna(subset=['dt_v']).sort_values('dt_v')

        # --- DYNAMIC SPLIT LOGIC ---
        applied_splits = False
        if corporate_actions:
            ca_df = pd.concat(corporate_actions, ignore_index=True)
            splits = ca_df[ca_df['Description'].str.contains('Split', case=False, na=False)]
            for _, split in splits.iterrows():
                ticker = split['Symbol']
                match = re.search(r'(\d+)\s+for\s+(\d+)', split['Description'])
                if match:
                    ratio = float(match.group(1)) / float(match.group(2))
                    split_dt = pd.to_datetime(split['Date/Time'].split(',')[0])
                    mask = (trades['Symbol'] == ticker) & (trades['dt_v'] < split_dt)
                    trades.loc[mask, 'qty_v'] *= ratio
                    trades.loc[mask, 'prc_v'] /= ratio
                    applied_splits = True

        # --- FALLBACK: If no splits found in Corporate Actions, use hardcoded ones ---
        if not applied_splits:
            for tkr, dt in [('NVDA', '2024-06-10'), ('SMCI', '2024-10-01')]:
                mask = (trades['Symbol'] == tkr) & (trades['dt_v'] < pd.to_datetime(dt))
                trades.loc[mask, 'qty_v'] *= 10
                trades.loc[mask, 'prc_v'] /= 10

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
                        if lots[0]['qty'] <= sq:
                            sq -= lots[0].pop('qty'); lots.pop(0)
                        else:
                            lots[0]['qty'] -= sq; sq = 0
            for l in lots:
                l['Symbol'] = sym
                l['Type'] = "Long-Term" if (pd.Timestamp.now() - l['date']).days > 365 else "Short-Term"
                holdings.append(l)
        df_lots = pd.DataFrame(holdings)
    except Exception as e:
        st.error(f"FIFO Calculation Error: {e}")

# --- 4. DASHBOARD RENDER ---
st.title("🏦 Wealth Terminal Pro")

if not fy_map:
    st.warning("No data found in GSheets. Please check tab names (FY24, FY25, FY26).")
else:
    sel_fy = st.selectbox("Financial Year", tabs, index=len(tabs)-1)
    data_fy = fy_map.get(sel_fy, {})

    # Top Metrics
    stocks_pl, forex_pl = 0.0, 0.0
    perf_s = 'Realized & Unrealized Performance Summary'
    if perf_s in data_fy:
        pdf = data_fy[perf_s]
        rt_c, ct_c = find_col(pdf, ['Realized Total']), find_col(pdf, ['Asset Category'])
        if rt_c and ct_c:
            clean_p = pdf[~pdf['Symbol'].str.contains('Total|Asset', case=False, na=False)]
            stocks_pl = clean_p[clean_p[ct_c].str.contains('Stock', case=False)][rt_c].apply(safe_num).sum()
            forex_pl = clean_p[clean_p[ct_c].str.contains('Forex|Cash', case=False)][rt_c].apply(safe_num).sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("Net Realized P/L", f"${(stocks_pl + forex_pl):,.2f}")
    m2.metric("Stocks Portion", f"${stocks_pl:,.2f}")
    m3.metric("Forex/Cash Portion", f"${forex_pl:,.2f}")

    # --- 5. HOLDINGS TABLES ---
    st.divider()
    if not df_lots.empty:
        # We need a small hack to get "Live" values since yfinance is off
        # For now, we show Cost and Qty. We can add price manual inputs if needed.
        def render_table(subset, label):
            st.subheader(label)
            if subset.empty: return st.info(f"No {label} positions.")
            agg = subset.groupby('Symbol').agg({'qty': 'sum', 'price': 'mean', 'comm': 'sum'}).reset_index()
            agg['Total Cost'] = (agg['qty'] * agg['price']) + agg['comm']
            agg.index = range(1, len(agg) + 1)
            st.dataframe(agg.style.format({"price": "${:.2f}", "Total Cost": "${:.2f}", "comm": "${:.2f}"}), use_container_width=True)

        render_table(df_lots.copy(), "Current Holdings")
        
        # --- 6. CALCULATOR ---
        st.divider()
        st.header("🧮 FIFO Selling Calculator")
        s_pick = st.selectbox("Ticker", sorted(df_lots['Symbol'].unique()))
        s_lots = df_lots[df_lots['Symbol'] == s_pick].sort_values('date')
        q_sell = st.slider("Units", 0.0, float(s_lots['qty'].sum()), float(s_lots['qty'].sum()*0.25))
        t_prof = st.number_input("Target Profit %", value=105.0)
        
        if q_sell > 0:
            curr_q, curr_c = q_sell, 0
            for _, lot in s_lots.iterrows():
                if curr_q <= 0: break
                take = min(lot['qty'], curr_q)
                curr_
