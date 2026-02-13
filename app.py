import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- 1. SETUP & ROBUST STYLE ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

st.markdown("""
    <style>
    .section-header {
        font-size: 1.4rem; font-weight: 700; color: #3b82f6;
        margin-top: 30px; border-bottom: 1px solid #334155;
        padding-bottom: 5px;
    }
    div[data-testid="stMetricValue"] { font-size: 24px !important; }
    </style>
    """, unsafe_allow_stdio=True)

# --- 2. THE CONNECTION & DEEP PARSER ---
conn = st.connection("gsheets", type=GSheetsConnection)

def robust_ibkr_parse(df):
    """Scans the grid for Header/Data rows and ignores everything else."""
    sections = {}
    if df is None or df.empty: return sections
    df = df.astype(str).replace('nan', '')
    
    # Identify all section names present in the first column
    potential_sections = df.iloc[:, 0].unique()
    
    for sec_name in potential_sections:
        if not sec_name or sec_name in ['Statement', 'Field Name', '']: continue
        
        sec_rows = df[df.iloc[:, 0] == sec_name]
        h_row = sec_rows[sec_rows.iloc[:, 1] == 'Header']
        d_rows = sec_rows[sec_rows.iloc[:, 1] == 'Data']
        
        if not h_row.empty and not d_rows.empty:
            # Map columns and handle empty headers to prevent data shifting
            raw_cols = h_row.iloc[0, 2:].tolist()
            clean_cols = [c if (c and c.strip()) else f"Col_{i}" for i, c in enumerate(raw_cols)]
            
            data = d_rows.iloc[:, 2:2+len(clean_cols)]
            data.columns = clean_cols
            # Remove helper columns used for spacing
            sections[sec_name] = data.loc[:, ~data.columns.str.startswith('Col_')]
            
    return sections

# --- 3. DATA LOADING (SCRUB & SKIP) ---
tabs = ["FY24", "FY25", "FY26"]
fy_data_map = {}
all_trades_list = []

for tab in tabs:
    try:
        # Read raw grid without assuming headers
        raw_grid = conn.read(worksheet=tab, ttl=0, header=None)
        parsed = robust_ibkr_parse(raw_grid)
        if parsed:
            fy_data_map[tab] = parsed
            if "Trades" in parsed:
                t = parsed["Trades"]
                # Only keep 'Order' rows to prevent double counting
                if 'DataDiscriminator' in t.columns:
                    t = t[t['DataDiscriminator'] == 'Order']
                all_trades_list.append(t)
    except Exception:
        continue # Silently skip missing/corrupt tabs

# --- 4. FIFO ENGINE ---
df_lots = pd.DataFrame()
if all_trades_list:
    try:
        trades = pd.concat(all_trades_list, ignore_index=True)
        trades['Quantity'] = pd.to_numeric(trades['Quantity'], errors='coerce').fillna(0)
        trades['T. Price'] = pd.to_numeric(trades['T. Price'], errors='coerce').fillna(0)
        # Handle Date/Time (stripping AEDT/Time info)
        trades['Date/Time'] = pd.to_datetime(trades['Date/Time'].str.split(',').str[0], errors='coerce')
        trades = trades.dropna(subset=['Date/Time']).sort_values('Date/Time')

        # Split Adjustments
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

# --- 5. TOP METRICS ---
st.title("🏦 Wealth Terminal Pro")
sel_fy = st.selectbox("Select Financial Year View", tabs, index=len(tabs)-1)
data_fy = fy_data_map.get(sel_fy, {})

c1, c2, c3 = st.columns(3)

def get_metric_val(section, keys, df_dict):
    if section not in df_dict: return 0.0
    df = df_dict[section]
    # Filter for 'Amount' or 'Realized Total' and ignore 'Total' summary rows
    df = df[~df.apply(lambda r: r.astype(str).str.contains('Total|Subtotal', case=False).any(), axis=1)]
    target = next((c for c in df.columns if any(k.lower() in c.lower() for k in keys)), None)
    return pd.to_numeric(df[target], errors='coerce').sum() if target else 0.0

c1.metric("Funds Injected", f"${get_metric_val('Deposits & Withdrawals', ['Amount'], data_fy):,.2f}")
c2.metric("Realized Profit", f"${get_metric_val('Realized & Unrealized Performance Summary', ['Realized Total'], data_fy):,.2f}")
c3.metric("Dividends", f"${get_metric_val('Dividends', ['Amount'], data_fy):,.2f}")

# --- 6. BREAKDOWNS (CENTERED & COMPACT) ---
st.divider()
cur_dt = datetime.now().strftime('%d %b %Y')

if not df_lots.empty:
    unique_syms = sorted(df_lots['Symbol'].unique().tolist())
    try:
        # Fetch prices with fallback to 0
        prices = yf.download(unique_syms, period="1d")['Close'].iloc[-1].to_dict()
        if len(unique_syms) == 1: prices = {unique_syms[0]: prices}
    except: prices = {s: 0.0 for s in unique_syms}

    def render_centered_table(subset, label):
        st.markdown(f'<div class="section-header">{label} (as of {cur_dt})</div>', unsafe_allow_stdio=True)
        if subset.empty: return st.info(f"No {label} positions.")
        
        subset['Cost'] = subset['qty'] * subset['price']
        agg = subset.groupby('Symbol').agg({'qty': 'sum', 'Cost': 'sum'}).reset_index()
        agg['Avg Buy'] = agg['Cost'] / agg['qty']
        agg['Price'] = agg['Symbol'].map(prices).fillna(0)
        agg['Value'] = agg['qty'] * agg['Price']
        agg['P/L $'] = agg['Value'] - agg['Cost']
        agg['P/L %'] = (agg['P/L $'] / agg['Cost']) * 100
        agg.insert(0, 'Sr.', range(1, len(agg) + 1))

        # --- COMPACT WIDTH CONTROL ---
        _, table_col, _ = st.columns([1, 10, 1]) 
        with table_col:
            st.dataframe(agg.style.format({
                "Avg Buy": "${:.2f}", "Price": "${:.2f}", "Value": "${:.2f}", "P/L $": "${:.2f}", "P/L %": "{:.2f}%"
            }).map(lambda x: 'color: #10b981' if x > 0 else 'color: #ef4444', subset=['P/L $', 'P/L %']), use_container_width=True, hide_index=True)
            st.write(f"**Total {label} Value:** `${agg['Value'].sum():,.2f}`")

    render_centered_table(df_lots.copy(), "1. Current Global Holdings")
    render_centered_table(df_lots[df_lots['Type'] == "Short-Term"].copy(), "2. Short-Term Holdings")
    render_centered_table(df_lots[df_lots['Type'] == "Long-Term"].copy(), "3. Long-Term Holdings")

    # --- 7. FIFO CALCULATOR ---
    st.divider()
    st.header("🧮 FIFO Selling Calculator")
    ca, cb = st.columns([1, 2])
    stock_pick = ca.selectbox("Pick Stock", unique_syms)
    stock_lots = df_lots[df_lots['Symbol'] == stock_pick].sort_values('date')
    total_owned = stock_lots['qty'].sum()
    
    sale_mode = ca.radio("Select Sale Mode", ["Units", "Percentage"])
    units_to_sell = cb.slider("Amount", 0.0, float(total_owned), float(total_owned*0.25)) if sale_mode == "Units" else total_owned * (cb.slider("Percent", 0, 100, 25) / 100)
    target_profit = cb.number_input("Target Profit %", value=105.0)
    
    if units_to_sell > 0:
        t_qty, s_cost = units_to_sell, 0
        for _, l in stock_lots.iterrows():
            if t_qty <= 0: break
            taken = min(l['qty'], t_qty)
            s_cost += taken * l['price']
            t_qty -= taken
        target_price = (s_cost * (1 + target_profit/100)) / units_to_sell
        st.success(f"**Action:** Sell {units_to_sell:.4f} units at **${target_price:.2f}**")
        
        # RESIDUAL INFO
        rem_units = total_owned - units_to_sell
        if rem_units > 0:
            rem_cost = (stock_lots['qty'] * stock_lots['price']).sum() - s_cost
            rem_avg_cost = rem_cost / rem_units
            current_live = prices[stock_pick]
            rem_total_pl = (current_live - rem_avg_cost) * rem_units
            
            st.write("---")
            r1, r2, r3 = st.columns(3)
            r1.metric("Remaining Units", f"{rem_units:.2f}")
            r2.metric("New Avg Cost", f"${rem_avg_cost:.2f}")
            r3.metric("Leftover P/L Status", f"${rem_total_pl:.2f}", f"{((current_live/rem_avg_cost)-1)*100:.2f}%")
