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
    sections = {}
    if df.empty: return sections
    df = df.astype(str).apply(lambda x: x.str.strip())
    for name in df.iloc[:, 0].unique():
        if name == 'nan' or not name: continue
        sec_df = df[df.iloc[:, 0] == name]
        header_row = sec_df[sec_df.iloc[:, 1] == 'Header']
        if not header_row.empty:
            cols = [c for c in header_row.iloc[0, 2:].tolist() if c and c != 'nan']
            data = sec_df[sec_df.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(cols)]
            data.columns = cols
            sections[name] = data
    return sections

# --- 2. GLOBAL DATA LOADING ---
tabs = ["FY24", "FY25", "FY26"]
fy_data_map = {}
all_trades_list = []

for tab in tabs:
    try:
        raw_df = conn.read(worksheet=tab, ttl=0)
        parsed = parse_ibkr_sheet(raw_df)
        fy_data_map[tab] = parsed
        if "Trades" in parsed: all_trades_list.append(parsed["Trades"])
    except: continue

# --- 3. FIFO ENGINE (GLOBAL) ---
if all_trades_list:
    trades = pd.concat(all_trades_list, ignore_index=True)
    trades['Quantity'] = pd.to_numeric(trades['Quantity'], errors='coerce')
    trades['T. Price'] = pd.to_numeric(trades['T. Price'], errors='coerce')
    trades['Date/Time'] = pd.to_datetime(trades['Date/Time'])
    trades = trades.sort_values('Date/Time')

    # SPLIT ADJUSTMENTS (NVDA & SMCI)
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
                    if lots[0]['qty'] <= sq: sq -= lots[0]['qty']; lots.pop(0)
                    else: lots[0]['qty'] -= sq; sq = 0
        for l in lots:
            l['Symbol'] = sym
            l['Type'] = "Long-Term" if (today - l['date']).days > 365 else "Short-Term"
            holdings.append(l)
    df_lots = pd.DataFrame(holdings)

# --- 4. TOP PERFORMANCE BAR (FY SPECIFIC) ---
st.title("🏦 Wealth Terminal")
sel_fy = st.selectbox("Financial Year Analysis", tabs, index=len(tabs)-1)

with st.container():
    c1, c2, c3 = st.columns(3)
    data_fy = fy_data_map.get(sel_fy, {})
    
    # Capital Injected
    invested = 0
    if "Deposits & Withdrawals" in data_fy:
        dw = data_fy["Deposits & Withdrawals"]
        dw['Amount'] = pd.to_numeric(dw['Amount'], errors='coerce').fillna(0)
        # Filter 'Total' rows
        actual_dw = dw[~dw.apply(lambda r: r.astype(str).str.contains('Total', case=False).any(), axis=1)]
        invested = actual_dw['Amount'].sum()
    c1.metric(f"Funds Injected ({sel_fy})", f"${invested:,.2f}")

    # Realized Profit (FY Level)
    realized = 0
    if "Realized & Unrealized Performance Summary" in data_fy:
        perf = data_fy["Realized & Unrealized Performance Summary"]
        if 'Realized Total' in perf.columns:
            realized = pd.to_numeric(perf['Realized Total'], errors='coerce').sum()
    c2.metric(f"Realized Profit ({sel_fy})", f"${realized:,.2f}")

    # Dividends (FY Level)
    div_sum = 0
    if "Dividends" in data_fy:
        divs = data_fy["Dividends"]
        divs['Amount'] = pd.to_numeric(divs['Amount'], errors='coerce').fillna(0)
        div_sum = divs['Amount'].sum()
    c3.metric(f"Dividends Received ({sel_fy})", f"${div_sum:,.2f}")

# --- 5. STOCK BREAKDOWN SECTIONS ---
if not df_lots.empty:
    unique_syms = df_lots['Symbol'].unique().tolist()
    with st.spinner('Fetching Live Market Prices...'):
        prices = yf.download(unique_syms, period="1d")['Close'].iloc[-1].to_dict()
        if len(unique_syms) == 1: prices = {unique_syms[0]: prices}

def render_table(df_subset):
    if df_subset.empty: return st.info("No holdings found in this category.")
    df_subset['Cost'] = df_subset['qty'] * df_subset['price']
    df_agg = df_subset.groupby('Symbol').agg({'qty': 'sum', 'Cost': 'sum'}).reset_index()
    df_agg['Avg Price'] = df_agg['Cost'] / df_agg['qty']
    df_agg['Live Price'] = df_agg['Symbol'].map(prices)
    df_agg['Current Value'] = df_agg['qty'] * df_agg['Live Price']
    df_agg['P/L $'] = df_agg['Current Value'] - df_agg['Cost']
    df_agg['P/L %'] = (df_agg['P/L $'] / df_agg['Cost']) * 100
    
    st.dataframe(df_agg.style.format({
        "Avg Price": "${:.2f}", "Live Price": "${:.2f}", "Current Value": "${:.2f}", "P/L $": "${:.2f}", "P/L %": "{:.2f}%"
    }), use_container_width=True)
    st.markdown(f"**Total Value:** `${df_agg['Current Value'].sum():,.2f}` | **Total P/L:** `${df_agg['P/L $'].sum():,.2f}`")

st.markdown('<p class="section-header">Portfolio Breakdowns</p>', unsafe_allow_stdio=True)
tab1, tab2, tab3, tab4 = st.tabs(["🌎 Current Holdings", "⏳ Short-Term (ST)", "💎 Long-Term (LT)", "📑 Dividend History"])

with tab1: render_table(df_lots.copy())
with tab2: render_table(df_lots[df_lots['Type'] == "Short-Term"].copy())
with tab3: render_table(df_lots[df_lots['Type'] == "Long-Term"].copy())
with tab4:
    if "Dividends" in data_fy:
        st.dataframe(data_fy["Dividends"], use_container_width=True)
    else:
        st.info(f"No dividend records found for {sel_fy}.")

# --- 6. FIFO CALCULATOR ---
st.divider()
st.header("🧮 FIFO Selling Tool")
c_a, c_b = st.columns([1, 2])
sel_stock = c_a.selectbox("Analyze Stock", unique_syms)
stock_lots = df_lots[df_lots['Symbol'] == sel_stock].sort_values('date')
total_owned = stock_lots['qty'].sum()

mode = c_a.radio("Mode", ["Units", "Percentage"])
amt = c_b.slider("Quantity", 0.0, float(total_owned) if mode=="Units" else 100.0, float(total_owned*0.25) if mode=="Units" else 25.0)
target = c_b.number_input("Target Profit %", value=105.0)

q_sell = amt if mode == "Units" else (total_owned * (amt/100))

if q_sell > 0:
    tmp_q, s_cost = q_sell, 0
    for _, l in stock_lots.iterrows():
        if tmp_q <= 0: break
        take = min(l['qty'], tmp_q)
        s_cost += take * l['price']
        tmp_q -= take
    
    target_p = (s_cost * (1 + target/100)) / q_sell
    st.success(f"To bag {target}% profit: Sell **{q_sell:.4f} units** at **${target_p:.2f}**")
    
    rem_q = total_owned - q_sell
    if rem_q > 0:
        rem_cost = (stock_lots['qty'] * stock_lots['price']).sum() - s_cost
        rem_avg = rem_cost / rem_q
        rem_pl = (prices[sel_stock] - rem_avg) * rem_q
        st.info(f"**Residual Advice:** {rem_q:.2f} units | **New Avg:** ${rem_avg:.2f} | **Current Status:** ${rem_pl:.2f}")
