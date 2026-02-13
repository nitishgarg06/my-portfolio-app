import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- 1. SETUP & THEME ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

st.markdown("""
    <style>
    .section-header { font-size: 1.4rem; font-weight: 700; color: #3b82f6; margin-top: 30px; border-bottom: 2px solid #334155; padding-bottom: 5px; }
    [data-testid="stMetricValue"] { font-size: 28px !important; }
    </style>
    """, unsafe_allow_stdio=True)

# --- 2. THE CONNECTION (RAW GRID MODE) ---
conn = st.connection("gsheets", type=GSheetsConnection)

def parse_raw_grid(df):
    """Manually maps IBKR sections from a raw text grid."""
    sections = {}
    if df is None or df.empty: return sections
    df = df.astype(str).replace('nan', '').apply(lambda x: x.str.strip())
    
    # IBKR files use the first column for section names
    for name in df.iloc[:, 0].unique():
        if not name or name in ['Statement', 'Field Name', '']: continue
        
        # Isolate rows for this specific section
        sec_rows = df[df.iloc[:, 0] == name]
        h_row = sec_rows[sec_rows.iloc[:, 1] == 'Header']
        d_rows = sec_rows[sec_rows.iloc[:, 1] == 'Data']
        
        if not h_row.empty and not d_rows.empty:
            # Skip first 2 helper columns (Section Name, Header/Data Tag)
            raw_cols = h_row.iloc[0, 2:].tolist()
            # Handle empty/shifted headers
            cols = [c if (c and c.strip()) else f"Col_{i}" for i, c in enumerate(raw_cols)]
            data = d_rows.iloc[:, 2:2+len(cols)]
            data.columns = cols
            sections[name] = data.loc[:, ~data.columns.str.startswith('Col_')]
    return sections

# --- 3. DATA LOADING ---
tabs = ["FY24", "FY25", "FY26"]
fy_map = {}
all_trades = []

for tab in tabs:
    try:
        # header=None is the 'Magic Fix' to prevent the initial GSheets crash
        grid = conn.read(worksheet=tab, ttl=0, header=None)
        parsed = parse_raw_grid(grid)
        if parsed:
            fy_map[tab] = parsed
            if "Trades" in parsed:
                t = parsed["Trades"]
                if 'DataDiscriminator' in t.columns:
                    t = t[t['DataDiscriminator'] == 'Order']
                all_trades.append(t)
    except: continue

# --- 4. FIFO ENGINE ---
df_lots = pd.DataFrame()
if all_trades:
    try:
        trades = pd.concat(all_trades, ignore_index=True)
        trades['Quantity'] = pd.to_numeric(trades['Quantity'], errors='coerce').fillna(0)
        trades['T. Price'] = pd.to_numeric(trades['T. Price'], errors='coerce').fillna(0)
        trades['dt'] = pd.to_datetime(trades['Date/Time'].str.split(',').str[0], errors='coerce')
        trades = trades.dropna(subset=['dt']).sort_values('dt')

        # Splits
        for tkr, split_dt in [('NVDA', '2024-06-10'), ('SMCI', '2024-10-01')]:
            trades.loc[(trades['Symbol'] == tkr) & (trades['dt'] < split_dt), 'Quantity'] *= 10
            trades.loc[(trades['Symbol'] == tkr) & (trades['dt'] < split_dt), 'T. Price'] /= 10

        today = pd.Timestamp.now()
        holdings = []
        for sym in trades['Symbol'].unique():
            lots = []
            for _, row in trades[trades['Symbol'] == sym].iterrows():
                if row['Quantity'] > 0: 
                    lots.append({'date': row['dt'], 'qty': row['Quantity'], 'price': row['T. Price']})
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

# --- 5. TOP METRICS (FY ONLY) ---
st.title("🏦 Wealth Terminal Pro")
sel_fy = st.selectbox("Select Financial Year Context", tabs, index=len(tabs)-1)
data_fy = fy_map.get(sel_fy, {})

m1, m2, m3 = st.columns(3)

def get_total(section, keys, df_dict):
    if section not in df_dict: return 0.0
    df = df_dict[section]
    df = df[~df.apply(lambda r: r.astype(str).str.contains('Total|Subtotal', case=False).any(), axis=1)]
    # Dynamic header matching for 'Amount' or 'Realized Total'
    target = next((c for c in df.columns if any(k.lower() in c.lower() for k in keys)), None)
    return pd.to_numeric(df[target], errors='coerce').sum() if target else 0.0

m1.metric("Funds Injected", f"${get_total('Deposits & Withdrawals', ['Amount'], data_fy):,.2f}")
m2.metric("Realized Profit", f"${get_total('Realized & Unrealized Performance Summary', ['Realized Total'], data_fy):,.2f}")
m3.metric("Dividends", f"${get_total('Dividends', ['Amount'], data_fy):,.2f}")

# --- 6. BREAKDOWNS (CENTERED & COLOR-CODED) ---
st.divider()
cur_dt = datetime.now().strftime('%d %b %Y')

if not df_lots.empty:
    unique_syms = sorted(df_lots['Symbol'].unique().tolist())
    try:
        prices = yf.download(unique_syms, period="1d")['Close'].iloc[-1].to_dict()
        if len(unique_syms) == 1: prices = {unique_syms[0]: prices}
    except: prices = {s: 0.0 for s in unique_syms}

    def render_table(subset, label):
        st.markdown(f'<div class="section-header">{label} (as of {cur_dt})</div>', unsafe_allow_stdio=True)
        if subset.empty: return st.info("No holdings found.")
        
        subset['Cost'] = subset['qty'] * subset['price']
        agg = subset.groupby('Symbol').agg({'qty': 'sum', 'Cost': 'sum'}).reset_index()
        agg['Avg Buy'] = agg['Cost'] / agg['qty']
        agg['Price'] = agg['Symbol'].map(prices).fillna(0)
        agg['Value'] = agg['qty'] * agg['Price']
        agg['P/L $'] = agg['Value'] - agg['Cost']
        agg['P/L %'] = (agg['P/L $'] / agg['Cost']) * 100
        agg.insert(0, 'Sr.', range(1, len(agg) + 1)) # Start from 1

        _, table_col, _ = st.columns([1, 10, 1]) # CENTERED WIDTH
        with table_col:
            st.dataframe(agg.style.format({
                "Avg Buy": "${:.2f}", "Price": "${:.2f}", "Value": "${:.2f}", "P/L $": "${:.2f}", "P/L %": "{:.2f}%"
            }).map(lambda x: 'color: #10b981' if x > 0 else 'color: #ef4444', subset=['P/L $', 'P/L %']), 
            use_container_width=True, hide_index=True)
            st.write(f"**Total {label} Value:** `${agg['Value'].sum():,.2f}`")

    render_table(df_lots.copy(), "1. Current Global Holdings")
    render_table(df_lots[df_lots['Type'] == "Short-Term"].copy(), "2. Short-Term Holdings")
    render_table(df_lots[df_lots['Type'] == "Long-Term"].copy(), "3. Long-Term Holdings")

# --- 7. FIFO CALCULATOR & RESIDUALS ---
st.divider()
st.header("🧮 FIFO Selling Calculator")
if not df_lots.empty:
    ca, cb = st.columns([1, 2])
    stock_pick = ca.selectbox("Pick Stock", unique_syms)
    s_lots = df_lots[df_lots['Symbol'] == stock_pick].sort_values('date')
    tot = s_lots['qty'].sum()
    
    mode = ca.radio("Sale Mode", ["Units", "Percentage"])
    qty_s = cb.slider("Amount", 0.0, float(tot), float(tot*0.25)) if mode == "Units" else tot * (cb.slider("Percent", 0, 100, 25) / 100)
    target_p = cb.number_input("Target Profit %", value=105.0)
    
    if qty_s > 0:
        t_qty, s_cost = qty_s, 0
        for _, l in s_lots.iterrows():
            if t_qty <= 0: break
            take = min(l['qty'], t_qty)
            s_cost += take * l['price']
            t_qty -= take
        t_price = (s_cost * (1 + target_p/100)) / qty_s
        st.success(f"**Action:** Sell {qty_s:.4f} units at **${t_price:.2f}** for {target_p}% profit.")
        
        # Residuals
        rem_q = tot - qty_s
        if rem_q > 0:
            st.write("---")
            st.write("#### 💎 Residual Portfolio Status")
            rcost = (s_lots['qty'] * s_lots['price']).sum() - s_cost
            ravg = rcost / rem_q
            rpl = (prices[stock_pick] - ravg) * rem_q
            r1, r2, r3 = st.columns(3)
            r1.metric("Remaining Units", f"{rem_q:.2f}")
            r2.metric("New Avg Cost", f"${ravg:.2f}")
            r3.metric("Leftover P/L", f"${rpl:.2f}", f"{((prices[stock_pick]/ravg)-1)*100:.2f}%")
