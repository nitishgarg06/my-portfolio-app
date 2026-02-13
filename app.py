import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

# --- 1. THE CONNECTION & RAW PARSER ---
conn = st.connection("gsheets", type=GSheetsConnection)

def parse_ibkr_grid(df):
    """Bypasses auto-formatting to handle IBKR's unique CSV structure."""
    sections = {}
    if df is None or df.empty: return sections
    df = df.astype(str).replace('nan', '')
    
    for name in df.iloc[:, 0].unique():
        if not name or name in ['Statement', 'Field Name']: continue
        sec_df = df[df.iloc[:, 0] == name]
        h_row = sec_df[sec_df.iloc[:, 1] == 'Header']
        d_rows = sec_df[sec_df.iloc[:, 1] == 'Data']
        
        if not h_row.empty and not d_rows.empty:
            # Keep empty headers as unique names to prevent data shifting
            raw_cols = h_row.iloc[0, 2:].tolist()
            cols = [c if (c and c.strip()) else f"Blank_{i}" for i, c in enumerate(raw_cols)]
            data = d_rows.iloc[:, 2:2+len(cols)]
            data.columns = cols
            # Drop the blank helper columns
            sections[name] = data.loc[:, ~data.columns.str.startswith('Blank_')]
    return sections

# --- 2. DATA INGESTION ---
tabs = ["FY24", "FY25", "FY26"]
fy_data_map = {}
all_trades = []

for tab in tabs:
    try:
        raw_grid = conn.read(worksheet=tab, ttl=0, header=None)
        parsed = parse_ibkr_grid(raw_grid)
        if parsed:
            fy_data_map[tab] = parsed
            if "Trades" in parsed:
                t = parsed["Trades"]
                # Filter for 'Order' to prevent double-counting totals
                if 'DataDiscriminator' in t.columns:
                    t = t[t['DataDiscriminator'] == 'Order']
                all_trades.append(t)
    except: continue

# --- 3. FIFO ENGINE ---
df_lots = pd.DataFrame()
if all_trades:
    try:
        trades = pd.concat(all_trades, ignore_index=True)
        trades['Quantity'] = pd.to_numeric(trades['Quantity'], errors='coerce').fillna(0)
        trades['T. Price'] = pd.to_numeric(trades['T. Price'], errors='coerce').fillna(0)
        # Handle IBKR's Date format
        trades['Date/Time'] = pd.to_datetime(trades['Date/Time'].str.split(',').str[0], errors='coerce')
        trades = trades.dropna(subset=['Date/Time']).sort_values('Date/Time')

        # Split Adjustments (NVDA/SMCI)
        for tkr, dt in [('NVDA', '2024-06-10'), ('SMCI', '2024-10-01')]:
            trades.loc[(trades['Symbol'] == tkr) & (trades['Date/Time'] < dt), 'Quantity'] *= 10
            trades.loc[(trades['Symbol'] == tkr) & (trades['Date/Time'] < dt), 'T. Price'] /= 10

        today = pd.Timestamp.now()
        holdings = []
        for sym in trades['Symbol'].unique():
            lots = []
            for _, row in trades[trades['Symbol'] == sym].iterrows():
                if row['Quantity'] > 0: 
                    lots.append({'date': row['Date/Time'], 'qty': row['Quantity'], 'price': row['T. Price']})
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

# --- 4. TOP PERFORMANCE BAR (FY SPECIFIC) ---
st.title("🏦 Wealth Terminal Pro")
sel_fy = st.selectbox("Select Financial Year Context", tabs, index=len(tabs)-1)
data_fy = fy_data_map.get(sel_fy, {})

c1, c2, c3 = st.columns(3)
# Capital
cap = 0.0
if "Deposits & Withdrawals" in data_fy:
    dw = data_fy["Deposits & Withdrawals"]
    amt_col = 'Amount' if 'Amount' in dw.columns else dw.columns[-1]
    dw_clean = dw[~dw.apply(lambda r: r.astype(str).str.contains('Total', case=False).any(), axis=1)]
    cap = pd.to_numeric(dw_clean[amt_col], errors='coerce').sum()
c1.metric(f"Funds Injected ({sel_fy})", f"${cap:,.2f}")

# Realized Profit
real = 0.0
if "Realized & Unrealized Performance Summary" in data_fy:
    perf = data_fy["Realized & Unrealized Performance Summary"]
    if 'Realized Total' in perf.columns:
        real = pd.to_numeric(perf['Realized Total'], errors='coerce').sum()
c2.metric(f"Realized Profit ({sel_fy})", f"${real:,.2f}")

# Dividends
div = 0.0
if "Dividends" in data_fy:
    dv = data_fy["Dividends"]
    div = pd.to_numeric(dv['Amount'], errors='coerce').sum()
c3.metric(f"Dividends ({sel_fy})", f"${div:,.2f}")

# --- 5. STOCK BREAKDOWNS ---
st.divider()
cur_dt = datetime.now().strftime('%d %b %Y')

if not df_lots.empty:
    unique_syms = sorted(df_lots['Symbol'].unique().tolist())
    try:
        prices = yf.download(unique_syms, period="1d")['Close'].iloc[-1].to_dict()
        if len(unique_syms) == 1: prices = {unique_syms[0]: prices}
    except: prices = {s: 0.0 for s in unique_syms}

    def render_sec(subset, label):
        st.subheader(f"{label} (as of {cur_dt})")
        if subset.empty: return st.info("No holdings.")
        subset['Cost'] = subset['qty'] * subset['price']
        agg = subset.groupby('Symbol').agg({'qty': 'sum', 'Cost': 'sum'}).reset_index()
        agg['Avg Buy'] = agg['Cost'] / agg['qty']
        agg['Price'] = agg['Symbol'].map(prices).fillna(0)
        agg['Value'] = agg['qty'] * agg['Price']
        agg['P/L $'] = agg['Value'] - agg['Cost']
        agg['P/L %'] = (agg['P/L $'] / agg['Cost']) * 100
        
        # Sr No. logic
        agg.insert(0, 'Sr.', range(1, len(agg) + 1))
        
        # Styled Table (Native Method)
        st.dataframe(agg.style.format({
            "Avg Buy": "${:.2f}", "Price": "${:.2f}", "Value": "${:.2f}", "P/L $": "${:.2f}", "P/L %": "{:.2f}%"
        }).map(lambda x: 'color: #10b981' if x > 0 else 'color: #ef4444', subset=['P/L $', 'P/L %']), use_container_width=True, hide_index=True)
        st.write(f"**Total {label} Value:** `${agg['Value'].sum():,.2f}`")

    render_sec(df_lots.copy(), "1. Current Global Holdings")
    render_sec(df_lots[df_lots['Type'] == "Short-Term"].copy(), "2. Short-Term Holdings")
    render_sec(df_lots[df_lots['Type'] == "Long-Term"].copy(), "3. Long-Term Holdings")

    # --- 6. FIFO CALCULATOR ---
    st.divider()
    st.header("🧮 FIFO Selling Calculator")
    ca, cb = st.columns([1, 2])
    ss = ca.selectbox("Analyze Stock", unique_syms)
    s_lots = df_lots[df_lots['Symbol'] == ss].sort_values('date')
    tot = s_lots['qty'].sum()
    
    mode = ca.radio("Sale Mode", ["Specific Units", "Percentage"])
    qty = cb.slider("Amount to Sell", 0.0, float(tot), float(tot*0.25)) if mode == "Specific Units" else tot * (cb.slider("Percentage", 0, 100, 25) / 100)
    target_pct = cb.number_input("Target Profit %", value=105.0)
    
    if qty > 0:
        tq, sc = qty, 0
        for _, l in s_lots.iterrows():
            if tq <= 0: break
            take = min(l['qty'], tq)
            sc += take * l['price']
            tq -= take
        t_price = (sc * (1 + target_pct/100)) / qty
        st.success(f"To bag {target_pct}% profit: Sell **{qty:.4f} units** at **${t_price:.2f}**")
        
        # CLEAN RESIDUAL INFO
        rem_q = tot - qty
        if rem_q > 0:
            st.markdown("---")
            st.write("#### 💎 Residual Portfolio Status")
            rem_cost = (s_lots['qty'] * s_lots['price']).sum() - sc
            rem_avg = rem_cost / rem_q
            live_now = prices[ss]
            rem_pl = (live_now - rem_avg) * rem_q
            
            r1, r2, r3 = st.columns(3)
            r1.metric("Remaining Units", f"{rem_q:.2f}")
            r2.metric("New Avg Cost", f"${rem_avg:.2f}")
            r3.metric("Leftover P/L", f"${rem_pl:.2f}", f"{((live_now/rem_avg)-1)*100:.2f}%")
