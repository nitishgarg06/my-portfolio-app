import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

# --- 1. CONNECTION & SAFE CLEANER ---
conn = st.connection("gsheets", type=GSheetsConnection)

def safe_num(val):
    """Aggressively converts any string to float, handling ($1.00) and empty cells."""
    if val is None or pd.isna(val) or val == '': return 0.0
    s = str(val).strip().replace('$', '').replace(',', '')
    if '(' in s and ')' in s: s = '-' + s.replace('(', '').replace(')', '')
    try:
        return float(s)
    except:
        return 0.0

def find_col(df, keywords):
    """Dynamic column finder to prevent KeyErrors."""
    for col in df.columns:
        if any(k.lower() in str(col).lower() for k in keywords):
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
        
        q_c = find_col(trades, ['Quantity'])
        p_c = find_col(trades, ['Price', 'T. Price'])
        c_c = find_col(trades, ['Comm', 'Commission'])
        d_c = find_col(trades, ['Date/Time', 'Date'])

        trades['qty_v'] = trades[q_c].apply(safe_num)
        trades['prc_v'] = trades[p_c].apply(safe_num)
        trades['cmm_v'] = trades[c_c].apply(safe_num).abs()
        trades['dt_v'] = pd.to_datetime(trades[d_c].str.split(',').str[0], errors='coerce')
        
        trades = trades.dropna(subset=['dt_v']).sort_values('dt_v')

        # Split adjustments
        for tkr, split_dt in [('NVDA', '2024-06-10'), ('SMCI', '2024-10-01')]:
            trades.loc[(trades['Symbol'] == tkr) & (trades['dt_v'] < split_dt), 'qty_v'] *= 10
            trades.loc[(trades['Symbol'] == tkr) & (trades['dt_v'] < split_dt), 'prc_v'] /= 10

        holdings = []
        for sym in trades['Symbol'].unique():
            lots = []
            for _, row in trades[trades['Symbol'] == sym].iterrows():
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
    except: pass

# --- 4. TOP BAR & P/L BREAKDOWN ---
st.title("🏦 Wealth Terminal Pro")
sel_fy = st.selectbox("Financial Year View", tabs, index=len(tabs)-1)
data_fy = fy_map.get(sel_fy, {})

# a) Realized P/L and Investment Calcs
stocks_pl, forex_pl, comms = 0.0, 0.0, 0.0
perf_s = 'Realized & Unrealized Performance Summary'

if perf_s in data_fy:
    pdf = data_fy[perf_s]
    rt_c = find_col(pdf, ['Realized Total'])
    ct_c = find_col(pdf, ['Asset Category'])
    if rt_c and ct_c:
        pdf = pdf[~pdf['Symbol'].str.contains('Total|Asset', case=False, na=False)]
        stocks_pl = pdf[pdf[ct_c].str.contains('Stock|Equity', case=False, na=False)][rt_c].apply(safe_num).sum()
        forex_pl = pdf[pdf[ct_c].str.contains('Forex|Cash', case=False, na=False)][rt_c].apply(safe_num).sum()

if 'Trades' in data_fy:
    comms = data_fy['Trades'][find_col(data_fy['Trades'], ['Comm', 'Commission'])].apply(safe_num).abs().sum()

# Total Investment (Cost Basis of current holdings)
total_inv = (df_lots['qty'] * df_lots['price']).sum() + df_lots['comm'].sum() if not df_lots.empty else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Investment", f"${total_inv:,.2f}") # a) Missing top line info
m2.metric("Total Realized P/L", f"${(stocks_pl + forex_pl):,.2f}")
m3.metric("Total Commissions", f"${comms:,.2f}")
m4.metric("Net FY P/L", f"${(stocks_pl + forex_pl - comms):,.2f}")

with st.expander("📊 Stocks vs Forex Breakdown"):
    st.write(f"**Stocks Realized:** `${stocks_pl:,.2f}` | **Forex/Cash Realized:** `${forex_pl:,.2f}`")

# --- 5. HOLDINGS TABLES ---
st.divider()

def show_agg(subset, label):
    st.subheader(label)
    if subset is None or subset.empty: return st.info(f"No {label} identified.")
    subset['Cost'] = subset['qty'] * subset['price']
    agg = subset.groupby('Symbol').agg({'qty': 'sum', 'Cost': 'sum', 'comm': 'sum'}).reset_index()
    
    try:
        p_fetch = yf.download(agg['Symbol'].tolist(), period="1d")['Close'].iloc[-1]
        prices = p_fetch.to_dict() if isinstance(p_fetch, pd.Series) else {agg['Symbol'].iloc[0]: p_fetch}
    except: prices = {}
    
    agg['Price'] = agg['Symbol'].map(prices).fillna(0.0)
    agg['Value'] = agg['qty'] * agg['Price']
    agg['Net P/L $'] = (agg['Value'] - agg['Cost']) - agg['comm']
    agg['Net P/L %'] = (agg['Net P/L $'] / (agg['Cost'] + agg['comm'])) * 100
    agg.index = range(1, len(agg) + 1)
    
    st.dataframe(agg.style.format({
        "Cost": "${:.2f}", "Price": "${:.2f}", "Value": "${:.2f}", 
        "comm": "${:.2f}", "Net P/L $": "${:.2f}", "Net P/L %": "{:.2f}%"
    }).map(lambda x: 'color: green' if x > 0 else 'color: red', subset=['Net P/L $', 'Net P/L %']), use_container_width=True)
    return prices

prices = {}
if not df_lots.empty:
    prices = show_agg(df_lots.copy(), "1. Current Global Holdings")
    show_agg(df_lots[df_lots['Type'] == "Short-Term"].copy(), "2. Short-Term Holdings")
    show_agg(df_lots[df_lots['Type'] == "Long-Term"].copy(), "3. Long-Term Holdings")

    # --- 6. FIFO CALCULATOR ---
    st.divider()
    st.header("🧮 FIFO Selling Calculator")
    ca, cb = st.columns([1, 2])
    s_pick = ca.selectbox("Analyze Ticker", sorted(df_lots['Symbol'].unique()))
    s_lots = df_lots[df_lots['Symbol'] == s_pick].sort_values('date')
    tot_q = s_lots['qty'].sum()
    
    # b) Radio button for percentage
    mode = ca.radio("Amount Mode", ["Units", "Percentage"])
    if mode == "Units":
        q_sell = cb.slider("Units to Sell", 0.0, float(tot_q), float(tot_q*0.25))
    else:
        pct = cb.slider("Percentage of Holding (%)", 0, 100, 25)
        q_sell = tot_q * (pct / 100)
        
    t_prof = cb.number_input("Target Profit %", value=105.0)
    
    if q_sell > 0:
        curr_q, curr_c = q_sell, 0
        for _, lot in s_lots.iterrows():
            if curr_q <= 0: break
            take = min(lot['qty'], curr_q)
            curr_c += take * lot['price']
            curr_q -= take
        t_price = (curr_c * (1 + t_prof/100)) / q_sell
        st.success(f"To hit {t_prof}% profit: Sell **{q_sell:.4f} units** at **${t_price:.2f}**")
        
        # c) Remaining stock details
        rem_q = tot_q - q_sell
        if rem_q > 0:
            st.markdown("---")
            st.write("#### 💎 Residual Portfolio Status")
            rem_cost = (s_lots['qty'] * s_lots['price']).sum() - curr_c
            rem_avg = rem_cost / rem_q
            live = prices.get(s_pick, 0.0)
            rem_pl = (live - rem_avg) * rem_q
            r1, r2, r3 = st.columns(3)
            r1.metric("Units Left", f"{rem_q:.2f}")
            r2.metric("New Avg Cost", f"${rem_avg:.2f}")
            r3.metric("Projected P/L", f"${rem_pl:.2f}", f"{((live/rem_avg)-1)*100:.2f}%")
