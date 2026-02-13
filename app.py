import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 24px; font-weight: 700; color: #00ffcc; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] { height: 45px; background-color: #f1f5f9; border-radius: 8px; padding: 10px 20px; font-weight: 500; }
    .stTabs [aria-selected="true"] { background-color: #2563eb; color: white; }
    .section-header { font-size: 1.2rem; font-weight: 700; color: #1e293b; margin: 1.5rem 0 1rem 0; border-left: 4px solid #2563eb; padding-left: 10px; }
    </style>
    """, unsafe_allow_stdio=True)

# --- 1. CONNECTION ---
conn = st.connection("gsheets", type=GSheetsConnection)

def parse_ibkr_sheet(df):
    """Deep scrubber to handle messy IBKR imports."""
    sections = {}
    if df is None or df.empty: return sections
    
    # Ensure all data is string and stripped
    df = df.astype(str).apply(lambda x: x.str.strip())
    
    # Find unique section names in the first column
    section_names = df.iloc[:, 0].unique()
    
    for name in section_names:
        if name in ['nan', '', 'Statement']: continue
        
        # Get all rows for this section
        sec_df = df[df.iloc[:, 0] == name]
        
        # Identify Header and Data rows
        header_row = sec_df[sec_df.iloc[:, 1] == 'Header']
        data_rows = sec_df[sec_df.iloc[:, 1] == 'Data']
        
        if not header_row.empty and not data_rows.empty:
            # Extract column names from the header row (skipping first 2 columns)
            cols = [c for c in header_row.iloc[0, 2:].tolist() if c and c != 'nan']
            # Extract data (skipping first 2 columns)
            data = data_rows.iloc[:, 2:2+len(cols)]
            data.columns = cols
            sections[name] = data
    return sections

# --- 2. GLOBAL DATA LOADING ---
tabs = ["FY24", "FY25", "FY26"]
fy_data_map = {}
all_trades_list = []

for tab in tabs:
    try:
        # ttl=0 forces fresh data
        raw_df = conn.read(worksheet=tab, ttl=0)
        parsed = parse_ibkr_sheet(raw_df)
        if parsed:
            fy_data_map[tab] = parsed
            if "Trades" in parsed: all_trades_list.append(parsed["Trades"])
    except Exception as e:
        continue 

# --- 3. FIFO ENGINE ---
df_lots = pd.DataFrame()
if all_trades_list:
    try:
        trades = pd.concat(all_trades_list, ignore_index=True)
        trades['Quantity'] = pd.to_numeric(trades['Quantity'], errors='coerce').fillna(0)
        trades['T. Price'] = pd.to_numeric(trades['T. Price'], errors='coerce').fillna(0)
        trades['Date/Time'] = pd.to_datetime(trades['Date/Time'], errors='coerce')
        trades = trades.dropna(subset=['Date/Time']).sort_values('Date/Time')

        # Split Adjustments
        trades.loc[(trades['Symbol'] == 'NVDA') & (trades['Date/Time'] < '2024-06-10'), 'Quantity'] *= 10
        trades.loc[(trades['Symbol'] == 'NVDA') & (trades['Date/Time'] < '2024-06-10'), 'T. Price'] /= 10
        trades.loc[(trades['Symbol'] == 'SMCI') & (trades['Date/Time'] < '2024-10-01'), 'Quantity'] *= 10
        trades.loc[(trades['Symbol'] == 'SMCI') & (trades['Date/Time'] < '2024-10-01'), 'T. Price'] /= 10

        today = pd.Timestamp.now()
        holdings = []
        for sym in trades['Symbol'].unique():
            lots = []
            for _, row in trades[trades['Symbol'] == sym].iterrows():
                q = row['Quantity']
                if q > 0: lots.append({'date': row['Date/Time'], 'qty': q, 'price': row['T. Price']})
                elif q < 0:
                    sq = abs(q)
                    while sq > 0 and lots:
                        if lots[0]['qty'] <= sq: sq -= lots[0].pop('qty'); lots.pop(0)
                        else: lots[0]['qty'] -= sq; sq = 0
            for l in lots:
                l['Symbol'] = sym
                l['Type'] = "Long-Term" if (today - l['date']).days > 365 else "Short-Term"
                holdings.append(l)
        df_lots = pd.DataFrame(holdings)
    except Exception as e:
        st.error(f"Engine Error: {e}")

# --- 4. TOP PERFORMANCE BAR (FY SPECIFIC) ---
st.title("🏦 Wealth Terminal")
sel_fy = st.selectbox("Financial Year Selection", tabs, index=len(tabs)-1)

with st.container():
    c1, c2, c3 = st.columns(3)
    data_fy = fy_data_map.get(sel_fy, {})
    
    # Capital
    invested = 0.0
    if "Deposits & Withdrawals" in data_fy:
        dw = data_fy["Deposits & Withdrawals"]
        amt_col = 'Amount' if 'Amount' in dw.columns else dw.columns[-1]
        vals = pd.to_numeric(dw[amt_col], errors='coerce').fillna(0)
        # Filter totals
        clean_dw = dw[~dw.apply(lambda r: r.astype(str).str.contains('Total', case=False).any(), axis=1)]
        invested = pd.to_numeric(clean_dw[amt_col], errors='coerce').sum()
    c1.metric("Capital Injected", f"${invested:,.2f}")

    # Profit
    realized = 0.0
    if "Realized & Unrealized Performance Summary" in data_fy:
        perf = data_fy["Realized & Unrealized Performance Summary"]
        if 'Realized Total' in perf.columns:
            realized = pd.to_numeric(perf['Realized Total'], errors='coerce').sum()
    c2.metric("Realized Profit", f"${realized:,.2f}")

    # Dividends
    div_sum = 0.0
    if "Dividends" in data_fy:
        dv = data_fy["Dividends"]
        div_sum = pd.to_numeric(dv['Amount'], errors='coerce').sum()
    c3.metric("Dividends", f"${div_sum:,.2f}")

# --- 5. BREAKDOWNS ---
if not df_lots.empty:
    unique_syms = df_lots['Symbol'].unique().tolist()
    with st.spinner('Updating Market Prices...'):
        try:
            prices = yf.download(unique_syms, period="1d")['Close'].iloc[-1].to_dict()
            if len(unique_syms) == 1: prices = {unique_syms[0]: prices}
        except: prices = {s: 0.0 for s in unique_syms}

    def render_table(df_subset):
        if df_subset.empty: return st.info("No holdings in this category.")
        df_subset['Cost'] = df_subset['qty'] * df_subset['price']
        agg = df_subset.groupby('Symbol').agg({'qty': 'sum', 'Cost': 'sum'}).reset_index()
        agg['Avg Price'] = agg['Cost'] / agg['qty']
        agg['Live Price'] = agg['Symbol'].map(prices).fillna(0)
        agg['Value'] = agg['qty'] * agg['Live Price']
        agg['P/L $'] = agg['Value'] - agg['Cost']
        agg['P/L %'] = (agg['P/L $'] / agg['Cost']) * 100
        st.dataframe(agg.style.format({
            "Avg Price": "${:.2f}", "Live Price": "${:.2f}", "Value": "${:.2f}", "P/L $": "${:.2f}", "P/L %": "{:.2f}%"
        }), use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🌎 Global", "⏳ Short-Term", "💎 Long-Term", "📑 Div History"])
    with tab1: render_table(df_lots.copy())
    with tab2: render_table(df_lots[df_lots['Type'] == "Short-Term"].copy())
    with tab3: render_table(df_lots[df_lots['Type'] == "Long-Term"].copy())
    with tab4: 
        if "Dividends" in data_fy: st.dataframe(data_fy["Dividends"], use_container_width=True)

    # --- 6. CALCULATOR ---
    st.divider()
    st.header("🧮 FIFO Calculator")
    ca, cb = st.columns([1, 2])
    sel_s = ca.selectbox("Stock", unique_syms)
    s_lots = df_lots[df_lots['Symbol'] == sel_s].sort_values('date')
    tot = s_lots['qty'].sum()
    
    qty_s = cb.slider("Quantity", 0.0, float(tot), float(tot*0.25))
    trgt = cb.number_input("Target Profit %", value=105.0)
    
    if qty_s > 0:
        t_q, s_c = qty_s, 0
        for _, l in s_lots.iterrows():
            if t_q <= 0: break
            take = min(l['qty'], t_q)
            s_c += take * l['price']
            t_q -= take
        t_p = (s_c * (1 + trgt/100)) / qty_s
        st.success(f"Sell at: **${t_p:.2f}**")
