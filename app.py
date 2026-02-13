import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- 1. MODERN THEME & CSS ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

st.markdown("""
    <style>
    /* Main Background and Font */
    .main { background-color: #0e1117; }
    
    /* Custom Card Design for Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 32px !important;
        font-weight: 700 !important;
    }
    
    /* Header Styling */
    .section-header {
        font-size: 1.5rem;
        font-weight: 800;
        color: #ffffff;
        padding-bottom: 10px;
        border-bottom: 2px solid #3b82f6;
        margin-bottom: 20px;
        margin-top: 30px;
    }

    /* Target Profit Success Message */
    .stAlert {
        border-radius: 10px;
        border: none;
        background-color: #1e293b;
    }

    /* DataFrame Styling */
    .stDataFrame {
        border: 1px solid #334155;
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_stdio=True)

# --- 2. CONNECTION & PARSING (STABLE ENGINE) ---
conn = st.connection("gsheets", type=GSheetsConnection)

def safe_parse(df):
    sections = {}
    if df is None or df.empty: return sections
    df = df.astype(str).replace('nan', '').apply(lambda x: x.str.strip())
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

# --- 3. DATA INGESTION ---
tabs = ["FY24", "FY25", "FY26"]
fy_data_map = {}
all_trades = []

for tab in tabs:
    try:
        raw = conn.read(worksheet=tab, ttl=0)
        parsed = safe_parse(raw)
        if parsed:
            fy_data_map[tab] = parsed
            if "Trades" in parsed: 
                t_df = parsed["Trades"]
                # Ensure we only get Order rows to maintain value accuracy
                if 'DataDiscriminator' in t_df.columns:
                    t_df = t_df[t_df['DataDiscriminator'] == 'Order']
                else:
                    t_df = t_df[~t_df.apply(lambda r: r.astype(str).str.contains('Total', case=False).any(), axis=1)]
                all_trades.append(t_df)
    except: continue

# --- 4. FIFO ENGINE ---
df_lots = pd.DataFrame()
if all_trades:
    try:
        trades = pd.concat(all_trades, ignore_index=True)
        trades['Quantity'] = pd.to_numeric(trades['Quantity'], errors='coerce').fillna(0)
        trades['T. Price'] = pd.to_numeric(trades['T. Price'], errors='coerce').fillna(0)
        trades['Date/Time'] = pd.to_datetime(trades['Date/Time'].str.split(',').str[0], errors='coerce')
        trades = trades.dropna(subset=['Date/Time']).sort_values('Date/Time')

        # Split adjustments
        for ticker in ['NVDA', 'SMCI']:
            split_date = '2024-06-10' if ticker == 'NVDA' else '2024-10-01'
            trades.loc[(trades['Symbol'] == ticker) & (trades['Date/Time'] < split_date), 'Quantity'] *= 10
            trades.loc[(trades['Symbol'] == ticker) & (trades['Date/Time'] < split_date), 'T. Price'] /= 10

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

# --- 5. TOP METRICS BAR ---
st.title("💰 Wealth Terminal Pro")
sel_fy = st.sidebar.selectbox("Financial Year Selection", tabs, index=len(tabs)-1)
data_fy = fy_data_map.get(sel_fy, {})

def get_metric(section, keys, df_dict):
    if section not in df_dict: return 0.0
    df = df_dict[section]
    df = df[~df.apply(lambda r: r.astype(str).str.contains('Total|Subtotal', case=False).any(), axis=1)]
    target = next((c for c in df.columns if any(k.lower() in c.lower() for k in keys)), None)
    return pd.to_numeric(df[target], errors='coerce').sum() if target else 0.0

m1, m2, m3 = st.columns(3)
m1.metric("Funds Injected", f"${get_metric('Deposits & Withdrawals', ['Amount'], data_fy):,.2f}")
m2.metric("Realized Profit", f"${get_metric('Realized & Unrealized Performance Summary', ['Realized Total'], data_fy):,.2f}")
m3.metric("Dividends", f"${get_metric('Dividends', ['Amount'], data_fy):,.2f}")

# --- 6. GLOBAL BREAKDOWNS WITH COLOR CODING ---
cur_date = datetime.now().strftime('%d %b %Y')

if not df_lots.empty:
    unique_syms = sorted(df_lots['Symbol'].unique().tolist())
    prices = yf.download(unique_syms, period="1d")['Close'].iloc[-1].to_dict()
    if len(unique_syms) == 1: prices = {unique_syms[0]: prices}

    def render_styled_table(subset, label):
        st.markdown(f'<div class="section-header">{label} (as of {cur_date})</div>', unsafe_allow_stdio=True)
        if subset.empty: return st.info("No holdings.")
        subset['Cost'] = subset['qty'] * subset['price']
        agg = subset.groupby('Symbol').agg({'qty': 'sum', 'Cost': 'sum'}).reset_index()
        agg['Avg Buy Price'] = agg['Cost'] / agg['qty']
        agg['Current Price'] = agg['Symbol'].map(prices).fillna(0)
        agg['Current Value'] = agg['qty'] * agg['Current Price']
        agg['P/L $'] = agg['Current Value'] - agg['Cost']
        agg['P/L %'] = (agg['P/L $'] / agg['Cost']) * 100
        agg.index = range(1, len(agg) + 1)

        # Apply Color Logic (Green for Profit, Red for Loss)
        def color_pl(val):
            color = '#10b981' if val > 0 else '#ef4444' # Tailwind Emerald-500 and Rose-500
            return f'color: {color}; font-weight: bold;'

        st.dataframe(
            agg.style.format({
                "Avg Buy Price": "${:.2f}", "Current Price": "${:.2f}", 
                "Current Value": "${:.2f}", "P/L $": "${:.2f}", "P/L %": "{:.2f}%"
            }).applymap(color_pl, subset=['P/L $', 'P/L %']), 
            use_container_width=True
        )

    render_styled_table(df_lots.copy(), "1. Current Global Holdings")
    render_styled_table(df_lots[df_lots['Type'] == "Short-Term"].copy(), "2. Short-Term Holdings")
    render_styled_table(df_lots[df_lots['Type'] == "Long-Term"].copy(), "3. Long-Term Holdings")

    # --- 7. FIFO CALCULATOR WITH NEW UI ---
    st.markdown('<div class="section-header">🧮 FIFO Selling Calculator</div>', unsafe_allow_stdio=True)
    with st.container():
        ca, cb = st.columns([1, 2])
        stock = ca.selectbox("Select Ticker", unique_syms)
        s_lots = df_lots[df_lots['Symbol'] == stock].sort_values('date')
        tot = s_lots['qty'].sum()
        
        calc_mode = ca.radio("Sale Mode", ["Specific Units", "Percentage of Holding"])
        amt = cb.slider("Units to Sell", 0.0, float(tot), float(tot*0.25)) if calc_mode == "Specific Units" else tot * (cb.slider("Percentage (%)", 0, 100, 25) / 100)
        t_p_pct = cb.number_input("Target Profit %", value=105.0)
        
        if amt > 0:
            tq, sc = amt, 0
            for _, l in s_lots.iterrows():
                if tq <= 0: break
                take = min(l['qty'], tq)
                sc += take * l['price']
                tq -= take
            res_price = (sc * (1 + t_p_pct/100)) / amt
            
            st.success(f"**Target Sell Price:** Sell {amt:.4f} units at **${res_price:.2f}**")
            
            # RESIDUAL METRICS IN CARDS
            rem_q = tot - amt
            if rem_q > 0:
                rem_cost = (s_lots['qty'] * s_lots['price']).sum() - sc
                rem_avg = rem_cost / rem_q
                live_now = prices[stock]
                rem_pl = (live_now - rem_avg) * rem_q
                
                st.write("---")
                r1, r2, r3 = st.columns(3)
                r1.metric("Remaining Units", f"{rem_q:.2f}")
                r2.metric("New Avg Cost", f"${rem_avg:.2f}")
                r3.metric("Remaining Status", f"${rem_pl:.2f}", delta=f"{((live_now/rem_avg)-1)*100:.2f}%")
