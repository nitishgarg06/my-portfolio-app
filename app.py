import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
from datetime import datetime
import re

# --- CONFIG ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

# FIXED CSS: Use valid markdown/HTML or skip if causing issues
st.markdown("""
<style>
    /* target metrics for styling without causing TypeErrors */
    [data-testid="stMetricValue"] {
        color: #00ffcc;
        font-weight: bold;
    }
</style>
""", unsafe_allow_stdio=True)

# --- 1. CONNECTION ---
conn = st.connection("gsheets", type=GSheetsConnection)

def safe_parse(df):
    """Robust parser to handle IBKR's specific grid structure."""
    sections = {}
    if df is None or df.empty: return sections
    
    # Standardize all text
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

def clean_currency(series):
    """Safely converts financial strings (e.g., '($1.00)') to float."""
    if series is None or series.empty:
        return pd.Series(0.0)
    # Remove currency symbols, commas, and handle parentheses for negatives
    cleaned = series.astype(str).str.replace(r'[$,]', '', regex=True)
    # Convert '(value)' to '-value'
    cleaned = cleaned.str.replace(r'\((.*)\)', r'-\1', regex=True)
    return pd.to_numeric(cleaned, errors='coerce').fillna(0.0)

# --- 2. DATA INGESTION ---
tabs = ["FY24", "FY25", "FY26"]
fy_data_map = {}
all_trades_list = []

for tab in tabs:
    try:
        raw = conn.read(worksheet=tab, ttl=0)
        parsed = safe_parse(raw)
        if parsed:
            fy_data_map[tab] = parsed
            if "Trades" in parsed: 
                t_df = parsed["Trades"]
                # Standardize columns to avoid KeyErrors
                t_df.columns = t_df.columns.str.strip()
                if 'DataDiscriminator' in t_df.columns:
                    t_df = t_df[t_df['DataDiscriminator'] == 'Order']
                all_trades_list.append(t_df)
    except Exception as e:
        st.sidebar.warning(f"Note: Could not load tab {tab}.")
        continue

# --- 3. FIFO ENGINE ---
df_lots = pd.DataFrame()
if all_trades_list:
    try:
        trades = pd.concat(all_trades_list, ignore_index=True)
        # Safe numeric conversion
        trades['Quantity'] = clean_currency(trades.get('Quantity'))
        trades['T. Price'] = clean_currency(trades.get('T. Price'))
        
        # Safe Commission Loading
        comm_col = next((c for c in trades.columns if 'Comm' in c and 'AUD' in c), None)
        trades['Comm'] = clean_currency(trades[comm_col]) if comm_col else 0.0

        trades['Date/Time'] = pd.to_datetime(trades['Date/Time'].str.split(',').str[0], errors='coerce')
        trades = trades.dropna(subset=['Date/Time']).sort_values('Date/Time')

        # Split adjustments
        for tkr, dt in [('NVDA', '2024-06-10'), ('SMCI', '2024-10-01')]:
            if tkr in trades['Symbol'].values:
                trades.loc[(trades['Symbol'] == tkr) & (trades['Date/Time'] < dt), 'Quantity'] *= 10
                trades.loc[(trades['Symbol'] == tkr) & (trades['Date/Time'] < dt), 'T. Price'] /= 10

        today = pd.Timestamp.now()
        holdings = []
        for sym in trades['Symbol'].unique():
            lots = []
            for _, row in trades[trades['Symbol'] == sym].iterrows():
                if row['Quantity'] > 0: 
                    lots.append({'date': row['Date/Time'], 'qty': row['Quantity'], 'price': row['T. Price'], 'buy_comm': row['Comm']})
                elif row['Quantity'] < 0:
                    sq = abs(row['Quantity'])
                    while sq > 0 and lots:
                        if lots[0]['qty'] <= sq:
                            sq -= lots[0].pop('qty')
                            lots.pop(0)
                        else:
                            lots[0]['qty'] -= sq
                            sq = 0
            for l in lots:
                l['Symbol'] = sym
                l['Type'] = "Long-Term" if (today - l['date']).days > 365 else "Short-Term"
                holdings.append(l)
        df_lots = pd.DataFrame(holdings)
    except Exception as e:
        st.error(f"Error in FIFO Engine: {e}")

# --- 4. TOP PERFORMANCE BAR ---
st.title("🏦 Wealth Terminal Pro")
sel_fy = st.selectbox("Financial Year Performance", tabs, index=len(tabs)-1)
data_fy = fy_data_map.get(sel_fy, {})

def get_safe_metric(section, keys, df_dict):
    """Safely finds a column by key and sums it."""
    if section not in df_dict: return 0.0
    df = df_dict[section]
    df.columns = df.columns.str.strip()
    df = df[~df.apply(lambda r: r.astype(str).str.contains('Total|Subtotal', case=False).any(), axis=1)]
    target = next((c for c in df.columns if any(k.lower() in c.lower() for k in keys)), None)
    return clean_currency(df[target]).sum() if target else 0.0

# Calculate FX vs Stock
stocks_pl = get_safe_metric('Realized & Unrealized Performance Summary', ['Realized Total'], data_fy)
# Simplified check for Forex if category missing
forex_pl = 0.0 # Standard reports separate these, but we'll sum all for now

# Safe Commission Summing
total_comm = 0.0
if 'Trades' in data_fy:
    tr = data_fy['Trades']
    comm_col = next((c for c in tr.columns if 'Comm' in c and 'AUD' in c), None)
    total_comm = clean_currency(tr[comm_col]).abs().sum() if comm_col else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Funds Injected", f"${get_safe_metric('Deposits & Withdrawals', ['Amount'], data_fy):,.2f}")
m2.metric("Total Realized P/L", f"${(stocks_pl + forex_pl):,.2f}")
m3.metric("Total Commissions", f"${total_comm:,.2f}")
m4.metric("Dividends", f"${get_safe_metric('Dividends', ['Amount'], data_fy):,.2f}")

with st.expander("📊 View FY Breakdown (Stocks / Forex / Net)"):
    ca, cb, cc = st.columns(3)
    ca.write("**Stock Realized P/L**")
    ca.subheader(f"${stocks_pl:,.2f}")
    cb.write("**Forex Realized P/L**")
    cb.subheader(f"${forex_pl:,.2f}")
    cc.write("**Net P/L (After Fees)**")
    cc.subheader(f"${(stocks_pl + forex_pl - total_comm):,.2f}")

# --- 5. STOCK BREAKDOWNS ---
st.divider()
cur_date = datetime.now().strftime('%d %b %Y')

if not df_lots.empty:
    unique_syms = sorted(df_lots['Symbol'].unique().tolist())
    try:
        prices = yf.download(unique_syms, period="1d")['Close'].iloc[-1].to_dict()
        if len(unique_syms) == 1: prices = {unique_syms[0]: prices}
    except: prices = {s: 0.0 for s in unique_syms}

    def show_sec(subset, label):
        st.markdown(f"### {label} (as of {cur_date})")
        if subset.empty: return st.info("No holdings found.")
        
        subset['Cost_No_Comm'] = subset['qty'] * subset['price']
        subset['Comm_Paid'] = subset.get('buy_comm', 0.0)
        
        agg = subset.groupby('Symbol').agg({'qty': 'sum', 'Cost_No_Comm': 'sum', 'Comm_Paid': 'sum'}).reset_index()
        agg['Avg Buy Price'] = agg['Cost_No_Comm'] / agg['qty']
        agg['Current Price'] = agg['Symbol'].map(prices).fillna(0.0)
        agg['Current Value'] = agg['qty'] * agg['Current Price']
        
        agg['Gross P/L $'] = agg['Current Value'] - agg['Cost_No_Comm']
        agg['Net P/L $'] = agg['Gross P/L $'] - agg['Comm_Paid']
        agg['Net P/L %'] = (agg['Net P/L $'] / (agg['Cost_No_Comm'] + agg['Comm_Paid'])) * 100
        
        agg.index = range(1, len(agg) + 1)
        
        st.dataframe(agg.style.format({
            "Avg Buy Price": "${:.2f}", "Current Price": "${:.2f}", "Current Value": "${:.2f}", 
            "Comm_Paid": "${:.2f}", "Gross P/L $": "${:.2f}", "Net P/L $": "${:.2f}", "Net P/L %": "{:.2f}%"
        }).map(lambda x: 'color: green' if x > 0 else 'color: red', subset=['Net P/L $', 'Net P/L %']), use_container_width=True)

    show_sec(df_lots.copy(), "1. Current Global Holdings")
    show_sec(df_lots[df_lots['Type'] == "Short-Term"].copy(), "2. Short-Term Holdings")
    show_sec(df_lots[df_lots['Type'] == "Long-Term"].copy(), "3. Long-Term Holdings")

# --- 6. FIFO CALCULATOR ---
st.divider()
st.header("🧮 FIFO Selling Calculator")
if not df_lots.empty:
    ca, cb = st.columns([1, 2])
    unique_syms = sorted(df_lots['Symbol'].unique().tolist())
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
        st.success(f"To achieve {t_p_pct}% profit, sell **{amt:.4f} units** at **${res_price:.2f}**")
        
        rem_q = tot - amt
        if rem_q > 0:
            st.markdown("---")
            st.write("#### 💎 Residual Portfolio Strategy")
            rem_cost = (s_lots['qty'] * s_lots['price']).sum() - sc
            rem_avg = rem_cost / rem_q
            live_now = prices.get(stock, 0)
            rem_pl = (live_now - rem_avg) * rem_q
            rem_pct = ((live_now / rem_avg) - 1) * 100 if rem_avg != 0 else 0
            
            cr1, cr2, cr3 = st.columns(3)
            cr1.metric("Remaining Quantity", f"{rem_q:.2f} units")
            cr2.metric("New Avg Price", f"${rem_avg:.2f}")
            cr3.metric("Remaining P/L Status", f"${rem_pl:.2f}", f"{rem_pct:.2f}%")
