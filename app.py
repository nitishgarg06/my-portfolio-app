import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

# --- 1. CONNECTION & CLEANING ---
conn = st.connection("gsheets", type=GSheetsConnection)

def universal_clean(val):
    if val is None or pd.isna(val) or val == '': return 0.0
    s = str(val).strip().replace('$', '').replace(',', '')
    if '(' in s and ')' in s:
        s = '-' + s.replace('(', '').replace(')', '')
    try:
        return float(s)
    except:
        return 0.0

def safe_find_col(df, keywords):
    for col in df.columns:
        if any(key.lower() in str(col).lower() for key in keywords):
            return col
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
                all_trades_list.append(t_df)
    except: continue

# --- 3. FIFO ENGINE ---
df_lots = pd.DataFrame()
if all_trades_list:
    try:
        trades = pd.concat(all_trades_list, ignore_index=True)
        trades.columns = trades.columns.str.strip()
        
        q_col = safe_find_col(trades, ['Quantity'])
        p_col = safe_find_col(trades, ['Price', 'T. Price'])
        c_col = safe_find_col(trades, ['Comm', 'Commission'])
        d_col = safe_find_col(trades, ['Date/Time', 'Date'])

        trades['qty'] = trades[q_col].apply(universal_clean)
        trades['prc'] = trades[p_col].apply(universal_clean)
        trades['cmm'] = trades[c_col].apply(universal_clean).abs()
        trades['dt'] = pd.to_datetime(trades[d_col].str.split(',').str[0], errors='coerce')
        
        trades = trades.dropna(subset=['dt', 'qty']).sort_values('dt')

        # Split adjustments
        for tkr, split_dt in [('NVDA', '2024-06-10'), ('SMCI', '2024-10-01')]:
            trades.loc[(trades['Symbol'] == tkr) & (trades['dt'] < split_dt), 'qty'] *= 10
            trades.loc[(trades['Symbol'] == tkr) & (trades['dt'] < split_dt), 'prc'] /= 10

        holdings = []
        for sym in trades['Symbol'].unique():
            lots = []
            sym_trades = trades[trades['Symbol'] == sym]
            for _, row in sym_trades.iterrows():
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
    except Exception as e:
        st.error(f"FIFO Engine failed: {e}")

# --- 4. DASHBOARD & TOP LINE SPLIT ---
st.title("🏦 Wealth Terminal Pro")
sel_fy = st.selectbox("Financial Year View", tabs, index=len(tabs)-1)
data_fy = fy_data_map.get(sel_fy, {})

# Stocks vs Forex Realized P/L Split
perf_sec = 'Realized & Unrealized Performance Summary'
stocks_pl, forex_pl = 0.0, 0.0

if perf_sec in data_fy:
    perf_df = data_fy[perf_sec]
    rt_col = safe_find_col(perf_df, ['Realized Total'])
    cat_col = safe_find_col(perf_df, ['Asset Category'])
    
    if rt_col and cat_col:
        perf_df = perf_df[~perf_df['Symbol'].str.contains('Total|Asset', case=False, na=False)]
        stocks_pl = perf_df[perf_df[cat_col].str.contains('Stock|Equity', case=False, na=False)][rt_col].apply(universal_clean).sum()
        forex_pl = perf_df[perf_df[cat_col].str.contains('Forex|Cash', case=False, na=False)][rt_col].apply(universal_clean).sum()

def get_metric(section, keys, df_dict):
    if section not in df_dict: return 0.0
    df = df_dict[section]
    col = safe_find_col(df, keys)
    if not col: return 0.0
    return df[~df.apply(lambda r: r.astype(str).str.contains('Total', case=False).any(), axis=1)][col].apply(universal_clean).sum()

funds = get_metric('Deposits & Withdrawals', ['Amount'], data_fy)
comms = get_metric('Trades', ['Comm', 'Commission'], data_fy).abs()
divs = get_metric('Dividends', ['Amount'], data_fy)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Funds Injected", f"${funds:,.2f}")
m2.metric("Total Realized P/L", f"${(stocks_pl + forex_pl):,.2f}")
m3.metric("Commissions Paid", f"${comms:,.2f}")
m4.metric("Dividends", f"${divs:,.2f}")

with st.expander("📊 Top-Line Realized P/L Split"):
    c1, c2 = st.columns(2)
    c1.metric("Stocks P/L", f"${stocks_pl:,.2f}")
    c2.metric("Forex/Cash P/L", f"${forex_pl:,.2f}")

# --- 5. HOLDINGS TABLES ---
st.divider()

def render_table(subset, label):
    st.subheader(f"{label} (as of {datetime.now().strftime('%d %b %Y')})")
    if subset is None or subset.empty: 
        st.info(f"No {label} positions identified.")
        return
    
    subset['Cost'] = subset['qty'] * subset['price']
    agg = subset.groupby('Symbol').agg({'qty': 'sum', 'Cost': 'sum', 'comm': 'sum'}).reset_index()
    agg['Avg Buy'] = agg['Cost'] / agg['qty']
    
    try:
        p_data = yf.download(agg['Symbol'].tolist(), period="1d")['Close'].iloc[-1]
        prices = p_data.to_dict() if isinstance(p_data, pd.Series) else {agg['Symbol'].iloc[0]: p_data}
    except: prices = {}
    
    agg['Price'] = agg['Symbol'].map(prices).fillna(0.0)
    agg['Value'] = agg['qty'] * agg['Price']
    agg['Net P/L $'] = (agg['Value'] - agg['Cost']) - agg['comm']
    agg['Net P/L %'] = (agg['Net P/L $'] / (agg['Cost'] + agg['comm'])) * 100
    
    agg.index = range(1, len(agg) + 1)
    st.dataframe(agg.style.format({
        "Avg Buy": "${:.2f}", "Price": "${:.2f}", "Value": "${:.2f}", 
        "comm": "${:.2f}", "Net P/L $": "${:.2f}", "Net P/L %": "{:.2f}%"
    }).map(lambda x: 'color: green' if x > 0 else 'color: red', subset=['Net P/L $', 'Net P/L %']), use_container_width=True)

if not df_lots.empty:
    render_table(df_lots.copy(), "1. Current Global Holdings")
    render_table(df_lots[df_lots['Type'] == "Short-Term"].copy(), "2. Short-Term Holdings")
    render_table(df_lots[df_lots['Type'] == "Long-Term"].copy(), "3. Long-Term Holdings")

    # --- 6. FIFO CALCULATOR ---
    st.divider()
    st.header("🧮 FIFO Selling Calculator")
    c_a, c_b = st.columns([1, 2])
    stock_pick = c_a.selectbox("Pick Ticker", sorted(df_lots['Symbol'].unique()))
    s_lots = df_lots[df_lots['Symbol'] == stock_pick].sort_values('date')
    total_q = s_lots['qty'].sum()
    
    mode = c_a.radio("Sale Mode", ["Units", "Percentage"])
    sell_qty = c_b.slider("Qty to Sell", 0.0, float(total_q), float(total_q*0.25)) if mode == "Units" else total_q * (c_b.slider("% to Sell", 0, 100, 25) / 100)
    target_profit = c_b.number_input("Target Profit %", value=105.0)
    
    if sell_qty > 0:
        t_q, s_c = sell_qty, 0
        for _, lot in s_lots.iterrows():
            if t_q <= 0: break
            take = min(lot['qty'], t_q)
            s_c += take * lot['price']
            t_q -= take
        target_price = (s_c * (1 + target_profit/100)) / sell_qty
        st.success(f"To hit {target_profit}% profit: Sell **{sell_qty:.4f} units** at **${target_price:.2f}**")
