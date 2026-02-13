import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

# --- 1. CONNECTION ---
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

# --- 2. DATA INGESTION ---
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
                # CRITICAL FIX for (c): Filter out IBKR's summary rows to prevent double counting
                t_df = parsed["Trades"]
                t_df = t_df[~t_df.apply(lambda r: r.astype(str).str.contains('Total', case=False).any(), axis=1)]
                all_trades.append(t_df)
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

# --- 4. TOP PERFORMANCE BAR ---
st.title("🏦 Wealth Terminal")
sel_fy = st.selectbox("Financial Year Performance", tabs, index=len(tabs)-1)
data_fy = fy_data_map.get(sel_fy, {})

c1, c2, c3 = st.columns(3)
def get_metric(section, keys, df_dict):
    if section not in df_dict: return 0.0
    df = df_dict[section]
    df = df[~df.apply(lambda r: r.astype(str).str.contains('Total|Subtotal', case=False).any(), axis=1)]
    target = next((c for c in df.columns if any(k.lower() in c.lower() for k in keys)), None)
    return pd.to_numeric(df[target], errors='coerce').sum() if target else 0.0

c1.metric("Funds Injected", f"${get_metric('Deposits & Withdrawals', ['Amount'], data_fy):,.2f}")
c2.metric("Realized Profit", f"${get_metric('Realized & Unrealized Performance Summary', ['Realized Total'], data_fy):,.2f}")
c3.metric("Dividends", f"${get_metric('Dividends', ['Amount'], data_fy):,.2f}")

# --- 5. STOCK BREAKDOWNS ---
st.divider()
cur_date = datetime.now().strftime('%d %b %Y')

if not df_lots.empty:
    unique_syms = df_lots['Symbol'].unique().tolist()
    prices = yf.download(unique_syms, period="1d")['Close'].iloc[-1].to_dict()
    if len(unique_syms) == 1: prices = {unique_syms[0]: prices}

    def show_sec(subset, label):
        st.markdown(f"### {label} (as of {cur_date})") # FIX (b)
        if subset.empty: return st.info("No holdings.")
        subset['Cost'] = subset['qty'] * subset['price']
        agg = subset.groupby('Symbol').agg({'qty': 'sum', 'Cost': 'sum'}).reset_index()
        agg['Avg Buy'] = agg['Cost'] / agg['qty']
        agg['Price'] = agg['Symbol'].map(prices).fillna(0)
        agg['Value'] = agg['qty'] * agg['Price']
        agg['P/L $'] = agg['Value'] - agg['Cost']
        agg['P/L %'] = (agg['P/L $'] / agg['Cost']) * 100
        
        # FIX (a): Change index to start from 1
        agg.index = agg.index + 1
        
        st.dataframe(agg.style.format({"Avg Buy": "${:.2f}", "Price": "${:.2f}", "Value": "${:.2f}", "P/L $": "${:.2f}", "P/L %": "{:.2f}%"}), use_container_width=True)
        st.write(f"**Total {label} Value:** `${agg['Value'].sum():,.2f}`")

    show_sec(df_lots.copy(), "1. Current Global Holdings")
    show_sec(df_lots[df_lots['Type'] == "Short-Term"].copy(), "2. Short-Term Holdings")
    show_sec(df_lots[df_lots['Type'] == "Long-Term"].copy(), "3. Long-Term Holdings")

    # --- 6. FIFO CALCULATOR FIX (d) ---
    st.divider()
    st.header("🧮 FIFO Calculator & Residual Analysis")
    ca, cb = st.columns([1, 2])
    stock = ca.selectbox("Pick Stock", unique_syms)
    s_lots = df_lots[df_lots['Symbol'] == stock].sort_values('date')
    tot = s_lots['qty'].sum()
    
    # Restored radio button
    calc_mode = ca.radio("Select Sale Mode", ["Units", "Percentage"])
    
    if calc_mode == "Units":
        amt = cb.slider("Units to Sell", 0.0, float(tot), float(tot*0.25))
    else:
        pct = cb.slider("Percentage to Sell", 0, 100, 25)
        amt = tot * (pct / 100)
        
    t_p_pct = cb.number_input("Target Profit %", value=105.0)
    
    if amt > 0:
        tq, sc = amt, 0
        for _, l in s_lots.iterrows():
            if tq <= 0: break
            take = min(l['qty'], tq)
            sc += take * l['price']
            tq -= take
        res = (sc * (1 + t_p_pct/100)) / amt
        st.success(f"To achieve {t_p_pct}% profit, sell **{amt:.4f} units** at **${res:.2f}**")
        
        # Restored Residual Info
        rem_q = tot - amt
        if rem_q > 0:
            rem_cost = (s_lots['qty'] * s_lots['price']).sum() - sc
            rem_avg = rem_cost / rem_q
            live_now = prices[stock]
            rem_pl = (live_now - rem_avg) * rem_q
            st.info(f"**Residual Portfolio Advice:** Remaining: **{rem_q:.2f}** units | New Avg Price: **${rem_avg:.2f}** | Status: **${rem_pl:.2f}**")
