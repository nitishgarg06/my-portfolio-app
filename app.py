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

# --- 1. CONNECTION & ROBUST PARSER ---
conn = st.connection("gsheets", type=GSheetsConnection)

def parse_ibkr_sheet(df):
    """Robust parser that handles empty columns and shifts."""
    sections = {}
    if df is None or df.empty: return sections
    df = df.astype(str).apply(lambda x: x.str.strip())
    
    for name in df.iloc[:, 0].unique():
        if name in ['nan', '', 'Statement']: continue
        sec_df = df[df.iloc[:, 0] == name]
        header_row = sec_df[sec_df.iloc[:, 1] == 'Header']
        data_rows = sec_df[sec_df.iloc[:, 1] == 'Data']
        
        if not header_row.empty and not data_rows.empty:
            raw_cols = header_row.iloc[0, 2:].tolist()
            # Clean headers: handle empty/nan and ensure unique names
            cols = []
            for i, c in enumerate(raw_cols):
                clean_c = c if (c and c != 'nan') else f"Col_{i}"
                cols.append(clean_c)
            
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
        raw_df = conn.read(worksheet=tab, ttl=0)
        parsed = parse_ibkr_sheet(raw_df)
        if parsed:
            fy_data_map[tab] = parsed
            if "Trades" in parsed: all_trades_list.append(parsed["Trades"])
    except: continue

# --- 3. FIFO ENGINE (GLOBAL) ---
df_lots = pd.DataFrame()
if all_trades_list:
    try:
        trades = pd.concat(all_trades_list, ignore_index=True)
        trades['Quantity'] = pd.to_numeric(trades['Quantity'], errors='coerce').fillna(0)
        trades['T. Price'] = pd.to_numeric(trades['T. Price'], errors='coerce').fillna(0)
        # Handle AEDT timezones and messy date strings
        trades['Date/Time'] = pd.to_datetime(trades['Date/Time'].str.split(',').str[0], errors='coerce')
        trades = trades.dropna(subset=['Date/Time']).sort_values('Date/Time')

        # MANUAL SPLIT ADJUSTMENTS
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
    except: pass

# --- 4. TOP PERFORMANCE BAR (FY SPECIFIC) ---
st.title("🏦 Wealth Terminal")
sel_fy = st.selectbox("Select Financial Year to view Performance", tabs, index=len(tabs)-1)

with st.container():
    c1, c2, c3 = st.columns(3)
    data_fy = fy_data_map.get(sel_fy, {})
    
    # 1. Capital Injected
    invested = 0.0
    if "Deposits & Withdrawals" in data_fy:
        dw = data_fy["Deposits & Withdrawals"]
        amt_col = 'Amount' if 'Amount' in dw.columns else dw.columns[-1]
        clean_dw = dw[~dw.apply(lambda r: r.astype(str).str.contains('Total', case=False).any(), axis=1)]
        invested = pd.to_numeric(clean_dw[amt_col], errors='coerce').sum()
    c1.metric(f"Funds Injected ({sel_fy})", f"${invested:,.2f}")

    # 2. Realized Profit (FY specific)
    realized = 0.0
    if "Realized & Unrealized Performance Summary" in data_fy:
        perf = data_fy["Realized & Unrealized Performance Summary"]
        if 'Realized Total' in perf.columns:
            realized = pd.to_numeric(perf['Realized Total'], errors='coerce').sum()
    c2.metric(f"Realized Profit ({sel_fy})", f"${realized:,.2f}")

    # 3. Dividends (FY specific)
    div_sum = 0.0
    if "Dividends" in data_fy:
        dv = data_fy["Dividends"]
        div_sum = pd.to_numeric(dv['Amount'], errors='coerce').sum()
    c3.metric(f"Dividends Received ({sel_fy})", f"${div_sum:,.2f}")

# --- 5. GLOBAL HOLDINGS & BREAKDOWNS ---
if not df_lots.empty:
    unique_syms = df_lots['Symbol'].unique().tolist()
    with st.spinner('Syncing Market Prices...'):
        try:
            tickers = yf.Tickers(" ".join(unique_syms))
            prices = {s: tickers.tickers[s].fast_info['last_price'] for s in unique_syms}
        except: prices = {s: 0.0 for s in unique_syms}

    def render_breakdown_table(df_subset, header_text):
        st.markdown(f'<p class="section-header">{header_text}</p>', unsafe_allow_stdio=True)
        if df_subset.empty: 
            st.info("No stocks found in this category.")
            return
        
        df_subset['Cost'] = df_subset['qty'] * df_subset['price']
        agg = df_subset.groupby('Symbol').agg({'qty': 'sum', 'Cost': 'sum'}).reset_index()
        agg['Avg Buy Price'] = agg['Cost'] / agg['qty']
        agg['Live Price'] = agg['Symbol'].map(prices).fillna(0)
        agg['Current Value'] = agg['qty'] * agg['Live Price']
        agg['P/L $'] = agg['Current Value'] - agg['Cost']
        agg['P/L %'] = (agg['P/L $'] / agg['Cost']) * 100
        
        st.dataframe(agg.style.format({
            "Avg Buy Price": "${:.2f}", "Live Price": "${:.2f}", "Current Value": "${:.2f}", "P/L $": "${:.2f}", "P/L %": "{:.2f}%"
        }), use_container_width=True)
        
        val, pl = agg['Current Value'].sum(), agg['P/L $'].sum()
        st.write(f"**Final Summary:** Total Value: `${val:,.2f}` | Total P/L: `${pl:,.2f}`")

    # The 3 Sections as requested
    render_breakdown_table(df_lots.copy(), "1. Current Global Holdings")
    render_breakdown_table(df_lots[df_lots['Type'] == "Short-Term"].copy(), "2. Short-Term Holdings (< 1 Year)")
    render_breakdown_table(df_lots[df_lots['Type'] == "Long-Term"].copy(), "3. Long-Term Holdings (> 1 Year)")

    # --- 6. RESTORED FIFO CALCULATOR ---
    st.divider()
    st.header("🧮 FIFO Calculator & Residual Analysis")
    ca, cb = st.columns([1, 2])
    sel_s = ca.selectbox("Analyze Stock", unique_syms)
    s_lots = df_lots[df_lots['Symbol'] == sel_s].sort_values('date')
    tot = s_lots['qty'].sum()
    
    mode = ca.radio("Mode", ["Units", "Percentage"])
    qty_s = cb.slider("Select Amount", 0.0, float(tot) if mode=="Units" else 100.0, float(tot*0.25) if mode=="Units" else 25.0)
    trgt = cb.number_input("Target Profit %", value=105.0)
    
    q_sell = qty_s if mode == "Units" else (tot * (qty_s/100))
    if q_sell > 0:
        t_q, s_c = q_sell, 0
        for _, l in s_lots.iterrows():
            if t_q <= 0: break
            take = min(l['qty'], t_q)
            s_c += take * l['price']
            t_q -= take
        t_p = (s_c * (1 + trgt/100)) / q_sell
        st.success(f"To bag {trgt}% profit: Sell at **${t_p:.2f}**")
        
        rem_q = tot - q_sell
        if rem_q > 0:
            rem_avg = ((s_lots['qty'] * s_lots['price']).sum() - s_c) / rem_q
            rem_pl = (prices[sel_s] - rem_avg) * rem_q
            st.info(f"**Residual Advice:** Remaining: {rem_q:.2f} units | New Avg: ${rem_avg:.2f} | Current Status: ${rem_pl:.2f}")

    # --- 7. DIVIDEND HISTORY ---
    st.divider()
    st.header("📑 Dividend History Detail")
    if "Dividends" in data_fy: st.dataframe(data_fy["Dividends"], use_container_width=True)
    else: st.info(f"No dividend records found for {sel_fy}.")
else:
    st.warning("Ensure tabs are named FY24, FY25, FY26 and contain 'Trades' section.")
