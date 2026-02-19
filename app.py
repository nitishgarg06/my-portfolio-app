import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

# --- 1. CONNECTION & ROBUST TOOLS ---
conn = st.connection("gsheets", type=GSheetsConnection)

def safe_num(val):
    if val is None or pd.isna(val) or val == '': return 0.0
    s = str(val).strip().replace('$', '').replace(',', '')
    if '(' in s and ')' in s: s = '-' + s.replace('(', '').replace(')', '')
    try: return float(s)
    except: return 0.0

def safe_find(df, keywords):
    for col in df.columns:
        if any(k.lower() in str(col).lower() for k in keywords): return col
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

# --- 2. DATA LOADING ---
tabs = ["FY24", "FY25", "FY26"]
fy_map = {}
all_trades_raw = []

for t in tabs:
    try:
        raw = conn.read(worksheet=t, ttl=0)
        parsed = parse_ibkr_grid(raw)
        if parsed:
            fy_map[t] = parsed
            if "Trades" in parsed: all_trades_raw.append(parsed["Trades"])
    except: continue

# --- 3. FIFO ENGINE ---
df_lots = pd.DataFrame()
if all_trades_raw:
    try:
        trades = pd.concat(all_trades_raw, ignore_index=True)
        trades.columns = trades.columns.str.strip()
        
        # Discover Columns
        q_c = safe_find(trades, ['Qty', 'Quantity'])
        p_c = safe_find(trades, ['Price', 'T. Price'])
        c_c = safe_find(trades, ['Comm', 'Commission'])
        d_c = safe_find(trades, ['Date', 'Time'])

        trades['qty_v'] = trades[q_c].apply(safe_num)
        trades['prc_v'] = trades[p_c].apply(safe_num)
        trades['cmm_v'] = trades[c_c].apply(safe_num).abs()
        trades['dt_v'] = pd.to_datetime(trades[d_c].str.split(',').str[0], errors='coerce')
        trades = trades.dropna(subset=['dt_v']).sort_values('dt_v')

        # Split Fallbacks
        for tkr, dt in [('NVDA', '2024-06-10'), ('SMCI', '2024-10-01')]:
            mask = (trades['Symbol'] == tkr) & (trades['dt_v'] < pd.to_datetime(dt))
            if not trades[mask].empty:
                trades.loc[mask, 'qty_v'] *= 10
                trades.loc[mask, 'prc_v'] /= 10

        holdings = []
        for sym in trades['Symbol'].unique():
            lots = []
            sym_trades = trades[trades['Symbol'] == sym]
            for _, row in sym_trades.iterrows():
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

# --- 4. DASHBOARD RENDER ---
st.title("🏦 Wealth Terminal Pro")
curr_date = datetime.now().strftime('%d %b %Y')

if not fy_map:
    st.error("No data tabs found.")
else:
    sel_fy = st.selectbox("Financial Year", tabs, index=len(tabs)-1)
    data_fy = fy_map.get(sel_fy, {})

    # Top Metrics
    stocks_pl, forex_pl = 0.0, 0.0
    perf_s = 'Realized & Unrealized Performance Summary'
    if perf_s in data_fy:
        pdf = data_fy[perf_s]
        rt_c, ct_c = safe_find(pdf, ['Realized Total']), safe_find(pdf, ['Asset Category'])
        if rt_c and ct_c:
            clean_p = pdf[~pdf['Symbol'].str.contains('Total|Asset', case=False, na=False)]
            stocks_pl = clean_p[clean_p[ct_c].str.contains('Stock', case=False)][rt_c].apply(safe_num).sum()
            forex_pl = clean_p[clean_p[ct_c].str.contains('Forex|Cash', case=False)][rt_c].apply(safe_num).sum()

    # a) Total Investment Logic
    total_inv = (df_lots['qty'] * df_lots['price']).sum() + df_lots['comm'].sum() if not df_lots.empty else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Investment", f"${total_inv:,.2f}")
    m2.metric("Net Realized P/L", f"${(stocks_pl + forex_pl):,.2f}")
    m3.metric("Forex/Cash Impact", f"${forex_pl:,.2f}")

    # --- 5. HOLDINGS WITH P/L CALCULATION ---
    st.divider()
    
    # We add a small manual price input area to enable P/L without blowing out the sidebar
    if not df_lots.empty:
        with st.expander("📝 Set Current Prices for P/L Calculation"):
            st.caption("Update these prices to see live P/L in the tables below.")
            price_cols = st.columns(4)
            prices = {}
            for i, ticker in enumerate(sorted(df_lots['Symbol'].unique())):
                with price_cols[i % 4]:
                    prices[ticker] = st.number_input(f"{ticker}", value=float(df_lots[df_lots['Symbol']==ticker]['price'].mean()))

        def render_table(subset, label):
            st.subheader(f"{label} (as of {curr_date})")
            if subset.empty: 
                st.info(f"No {label} positions.")
                return
            
            agg = subset.groupby('Symbol').agg({'qty': 'sum', 'price': 'mean', 'comm': 'sum'}).reset_index()
            agg['Current Price'] = agg['Symbol'].map(prices).fillna(0.0)
            agg['Total Basis'] = (agg['qty'] * agg['price']) + agg['comm']
            agg['Market Value'] = agg['qty'] * agg['Current Price']
            agg['P/L $'] = agg['Market Value'] - agg['Total Basis']
            agg['P/L %'] = (agg['P/L $'] / agg['Total Basis']) * 100
            
            agg.columns = ['Ticker', 'Units', 'Avg Cost', 'Comms', 'Live Price', 'Total Basis', 'Value', 'P/L $', 'P/L %']
            agg.index = range(1, len(agg) + 1)
            
            st.dataframe(agg.style.format({
                "Units": "{:.2f}", "Avg Cost": "${:.2f}", "Comms": "${:.2f}", "Live Price": "${:.2f}",
                "Total Basis": "${:.2f}", "Value": "${:.2f}", "P/L $": "${:.2f}", "P/L %": "{:.2f}%"
            }).map(lambda x: 'color: green' if x > 0 else 'color: red', subset=['P/L $', 'P/L %']), use_container_width=True)

        render_table(df_lots.copy(), "1. Current Global Holdings")
        render_table(df_lots[df_lots['Type'] == "Short-Term"].copy(), "2. Short-Term Holdings")
        render_table(df_lots[df_lots['Type'] == "Long-Term"].copy(), "3. Long-Term Holdings")

        # --- 6. CALCULATOR ---
        st.divider()
        st.header("🧮 FIFO Selling Calculator")
        ca, cb = st.columns([1, 2])
        s_pick = ca.selectbox("Analyze Ticker", sorted(df_lots['Symbol'].unique()))
        s_lots = df_lots[df_lots['Symbol'] == s_pick].sort_values('date')
        tot_q = s_lots['qty'].sum()
        
        mode = ca.radio("Amount Mode", ["Units", "Percentage"])
        q_sell = cb.slider("Amount to Sell", 0.0, float(tot_q), float(tot_q*0.25)) if mode == "Units" else tot_q * (cb.slider("% to Sell", 0, 100, 25) / 100)
        t_prof = cb.number_input("Target Profit %", value=105.0)
        
        if q_sell > 0:
            curr_q, curr_c = q_sell, 0
            for _, lot in s_lots.iterrows():
                if curr_q <= 0: break
                take = min(lot['qty'], curr_q)
                curr_c += take * lot['price']
                curr_q -= take
            target_price = (curr_c * (1 + t_prof/100)) / q_sell
            st.success(f"Target Price for {t_prof}% Profit: **${(curr_c*(1+t_prof/100))/q_sell:.2f}**")
            
            rem_q = tot_q - q_sell
            if rem_q > 0:
                st.write("#### 💎 Residual Portfolio Status")
                rem_cost = (s_lots['qty'] * s_lots['price']).sum() - curr_c
                r1, r2 = st.columns(2)
                r1.metric("Units Left", f"{rem_q:.2f}")
                r2.metric("New Avg Cost", f"${(rem_cost/rem_q):,.2f}")
    else:
        st.info("Calculating holdings... If this is empty, check that your 'Trades' section is present.")
