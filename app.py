import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf

# --- PAGE CONFIG ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

# --- 1. THE CONNECTION (BACK TO BASICS) ---
conn = st.connection("gsheets", type=GSheetsConnection)

def parse_ibkr_raw(df):
    """The original parser logic that worked, but with string safety."""
    sections = {}
    if df is None or df.empty: return sections
    
    # Convert everything to string immediately to prevent 'float' errors during parsing
    df = df.astype(str).apply(lambda x: x.str.strip())
    
    for name in df.iloc[:, 0].unique():
        if name in ['nan', '', 'Statement']: continue
        sec_df = df[df.iloc[:, 0] == name]
        h_row = sec_df[sec_df.iloc[:, 1] == 'Header']
        d_rows = sec_df[sec_df.iloc[:, 1] == 'Data']
        
        if not h_row.empty and not d_rows.empty:
            # We take the actual number of columns present in the data
            cols = [c for c in h_row.iloc[0, 2:].tolist() if c and c != 'nan']
            data = d_rows.iloc[:, 2:2+len(cols)]
            data.columns = cols
            sections[name] = data
    return sections

# --- 2. DATA INGESTION ---
tabs = ["FY24", "FY25", "FY26"]
fy_data_map = {}
all_trades = []

for tab in tabs:
    try:
        # We use the simplest read method possible
        raw = conn.read(worksheet=tab, ttl=0)
        parsed = parse_ibkr_raw(raw)
        if parsed:
            fy_data_map[tab] = parsed
            if "Trades" in parsed: all_trades.append(parsed["Trades"])
    except: continue

# --- 3. FIFO ENGINE ---
df_lots = pd.DataFrame()
if all_trades:
    try:
        trades = pd.concat(all_trades, ignore_index=True)
        trades['Quantity'] = pd.to_numeric(trades['Quantity'], errors='coerce').fillna(0)
        trades['T. Price'] = pd.to_numeric(trades['T. Price'], errors='coerce').fillna(0)
        trades['Date/Time'] = pd.to_datetime(trades['Date/Time'].str.split(',').str[0], errors='coerce')
        trades = trades.dropna(subset=['Date/Time']).sort_values('Date/Time')

        # Split adjustments (NVDA/SMCI)
        trades.loc[(trades['Symbol'] == 'NVDA') & (trades['Date/Time'] < '2024-06-10'), 'Quantity'] *= 10
        trades.loc[(trades['Symbol'] == 'NVDA') & (trades['Date/Time'] < '2024-06-10'), 'T. Price'] /= 10
        trades.loc[(trades['Symbol'] == 'SMCI') & (trades['Date/Time'] < '2024-10-01'), 'Quantity'] *= 10
        trades.loc[(trades['Symbol'] == 'SMCI') & (trades['Date/Time'] < '2024-10-01'), 'T. Price'] /= 10

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
    except: st.error("Error calculating FIFO lots.")

# --- 4. UI: FY PERFORMANCE ---
st.title("🏦 Wealth Terminal")
sel_fy = st.selectbox("Financial Year", tabs, index=len(tabs)-1)
data_fy = fy_data_map.get(sel_fy, {})

# Styling for Metrics
st.markdown("<style>div[data-testid='stMetricValue'] { color: #00ffcc; font-weight: bold; }</style>", unsafe_allow_stdio=True)

c1, c2, c3 = st.columns(3)

# Capital (Safe Calculation)
cap = 0.0
if "Deposits & Withdrawals" in data_fy:
    df_dw = data_fy["Deposits & Withdrawals"]
    amt_col = 'Amount' if 'Amount' in df_dw.columns else df_dw.columns[-1]
    clean_dw = df_dw[~df_dw.apply(lambda r: r.astype(str).str.contains('Total', case=False).any(), axis=1)]
    cap = pd.to_numeric(clean_dw[amt_col], errors='coerce').sum()
c1.metric(f"Funds Injected", f"${cap:,.2f}")

# Profit (Safe Calculation)
realized = 0.0
if "Realized & Unrealized Performance Summary" in data_fy:
    df_perf = data_fy["Realized & Unrealized Performance Summary"]
    if 'Realized Total' in df_perf.columns:
        realized = pd.to_numeric(df_perf['Realized Total'], errors='coerce').sum()
c2.metric(f"Realized Profit", f"${realized:,.2f}")

# Dividends (Safe Calculation)
divs = 0.0
if "Dividends" in data_fy:
    divs = pd.to_numeric(data_fy["Dividends"]['Amount'], errors='coerce').sum()
c3.metric(f"Dividends", f"${divs:,.2f}")

# --- 5. STOCK TABLES ---
if not df_lots.empty:
    with st.spinner('Syncing Live Prices...'):
        unique_syms = df_lots['Symbol'].unique().tolist()
        try:
            prices = yf.download(unique_syms, period="1d")['Close'].iloc[-1].to_dict()
            if len(unique_syms) == 1: prices = {unique_syms[0]: prices}
        except: prices = {s: 0.0 for s in unique_syms}

    def render_section(subset, title):
        st.subheader(title)
        if subset.empty: 
            st.info("No stocks in this category.")
            return
        subset['Cost'] = subset['qty'] * subset['price']
        agg = subset.groupby('Symbol').agg({'qty': 'sum', 'Cost': 'sum'}).reset_index()
        agg['Avg Buy'] = agg['Cost'] / agg['qty']
        agg['Live'] = agg['Symbol'].map(prices).fillna(0)
        agg['Value'] = agg['qty'] * agg['Live']
        agg['P/L $'] = agg['Value'] - agg['Cost']
        agg['P/L %'] = (agg['P/L $'] / agg['Cost']) * 100
        
        st.dataframe(agg.style.format({
            "Avg Buy": "${:.2f}", "Live": "${:.2f}", "Value": "${:.2f}", "P/L $": "${:.2f}", "P/L %": "{:.2f}%"
        }), use_container_width=True)
        st.write(f"**{title} Total:** ${agg['Value'].sum():,.2f}")

    render_section(df_lots.copy(), "1. Current Global Holdings")
    render_section(df_lots[df_lots['Type'] == "Short-Term"].copy(), "2. Short-Term Holdings")
    render_section(df_lots[df_lots['Type'] == "Long-Term"].copy(), "3. Long-Term Holdings")

# --- 6. FIFO CALCULATOR ---
st.divider()
st.header("🧮 FIFO Calculator")
if not df_lots.empty:
    ca, cb = st.columns([1, 2])
    ss = ca.selectbox("Stock", df_lots['Symbol'].unique())
    s_lots = df_lots[df_lots['Symbol'] == ss].sort_values('date')
    tot = s_lots['qty'].sum()
    
    qty = cb.slider("Units", 0.0, float(tot), float(tot*0.25))
    t_pct = cb.number_input("Target Profit %", value=105.0)
    
    if qty > 0:
        tq, sc = qty, 0
        for _, l in s_lots.iterrows():
            if tq <= 0: break
            take = min(l['qty'], tq)
            sc += take * l['price']
            tq -= take
        tp = (sc * (1 + t_pct/100)) / qty
        st.success(f"Sell at **${tp:.2f}**")
