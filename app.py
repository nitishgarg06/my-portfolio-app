import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

# --- 1. CONNECTION & ROBUST UTILS ---
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

@st.cache_data(ttl=600)
def get_google_price(ticker):
    """Robust fallback: Scrapes Google Finance for the current price."""
    try:
        # We assume NASDAQ or NYSE for simplicity, can be adjusted
        url = f"https://www.google.com/search?q={ticker}+stock+price"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        # This selector often finds the large price text on Google Search
        price_text = soup.find('span', {'class': 'I67upf'}).text or soup.find('span', {'jsname': 'vW79of'}).text
        return safe_num(price_text)
    except:
        return None

# --- 2. DATA LOADING ---
tabs = ["FY24", "FY25", "FY26"]
fy_map = {}
all_trades = []

for t in tabs:
    try:
        raw = conn.read(worksheet=t, ttl=0)
        parsed = parse_ibkr_grid(raw)
        if parsed:
            fy_map[t] = parsed
            if "Trades" in parsed: all_trades.append(parsed["Trades"])
    except: continue

# --- 3. FIFO ENGINE ---
df_lots = pd.DataFrame()
if all_trades:
    try:
        trades = pd.concat(all_trades, ignore_index=True)
        trades.columns = trades.columns.str.strip()
        q_c, p_c, c_c, d_c = safe_find(trades, ['Qty']), safe_find(trades, ['Price']), safe_find(trades, ['Comm']), safe_find(trades, ['Date'])
        trades['qty_v'] = trades[q_c].apply(safe_num)
        trades['prc_v'] = trades[p_c].apply(safe_num)
        trades['cmm_v'] = trades[c_c].apply(safe_num).abs()
        trades['dt_v'] = pd.to_datetime(trades[d_c].str.split(',').str[0], errors='coerce')
        trades = trades.dropna(subset=['dt_v']).sort_values('dt_v')

        # Fallback Splits (NVDA/SMCI)
        for tkr, dt in [('NVDA', '2024-06-10'), ('SMCI', '2024-10-01')]:
            mask = (trades['Symbol'] == tkr) & (trades['dt_v'] < pd.to_datetime(dt))
            if not trades[mask].empty:
                trades.loc[mask, 'qty_v'] *= 10; trades.loc[mask, 'prc_v'] /= 10

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
                        if lots[0]['qty'] <= sq: sq -= lots[0].pop('qty'); lots.pop(0)
                        else: lots[0]['qty'] -= sq; sq = 0
            for l in lots:
                l['Symbol'] = sym
                l['Type'] = "Long-Term" if (pd.Timestamp.now() - l['date']).days > 365 else "Short-Term"
                holdings.append(l)
        df_lots = pd.DataFrame(holdings)
    except: pass

# --- 4. RENDER DASHBOARD ---
st.title("🏦 Wealth Terminal Pro")
curr_date = datetime.now().strftime('%d %b %Y')

if not fy_map:
    st.error("GSheet Connection Failed.")
else:
    # --- METRICS & PRICE LOGIC ---
    sel_fy = st.selectbox("Financial Year View", tabs, index=len(tabs)-1)
    data_fy = fy_map.get(sel_fy, {})
    
    # Calculate Live Prices
    live_prices = {}
    if not df_lots.empty:
        for tkr in df_lots['Symbol'].unique():
            p = get_google_price(tkr)
            live_prices[tkr] = p if p else df_lots[df_lots['Symbol'] == tkr]['price'].mean()

    stocks_pl, forex_pl = 0.0, 0.0
    perf_s = 'Realized & Unrealized Performance Summary'
    if perf_s in data_fy:
        pdf = data_fy[perf_s]
        rt_c, ct_c = safe_find(pdf, ['Realized Total']), safe_find(pdf, ['Asset Category'])
        if rt_c and ct_c:
            clean_p = pdf[~pdf['Symbol'].str.contains('Total|Asset', case=False, na=False)]
            stocks_pl = clean_p[clean_p[ct_c].str.contains('Stock', case=False)][rt_c].apply(safe_num).sum()
            forex_pl = clean_p[clean_p[ct_c].str.contains('Forex|Cash', case=False)][rt_c].apply(safe_num).sum()
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Net Realized P/L", f"${(stocks_pl + forex_pl):,.2f}")
    m2.metric("Stocks Portion", f"${stocks_pl:,.2f}")
    m3.metric("Forex/Cash Impact", f"${forex_pl:,.2f}")

# --- 5. HOLDINGS TABLES ---
st.divider()

def render_table(subset, label):
    st.subheader(f"{label} (as of {curr_date})")
    if subset.empty: return st.info(f"No {label} positions identified.")
    
    agg = subset.groupby('Symbol').agg({'qty': 'sum', 'price': 'mean', 'comm': 'sum'}).reset_index()
    agg['Live Price'] = agg['Symbol'].map(live_prices)
    agg['Total Basis'] = (agg['qty'] * agg['price']) + agg['comm']
    agg['Market Value'] = agg['qty'] * agg['Live Price']
    agg['P/L $'] = agg['Market Value'] - agg['Total Basis']
    agg['P/L %'] = (agg['P/L $'] / agg['Total Basis']) * 100
    
    agg.columns = ['Ticker', 'Units', 'Avg Cost', 'Comms', 'Live Price', 'Total Basis', 'Market Value', 'P/L $', 'P/L %']
    agg.index = range(1, len(agg) + 1)
    
    st.dataframe(agg.style.format({
        "Units": "{:.2f}", "Avg Cost": "${:.2f}", "Comms": "${:.2f}", "Live Price": "${:.2f}",
        "Total Basis": "${:.2f}", "Market Value": "${:.2f}", "P/L $": "${:.2f}", "P/L %": "{:.2f}%"
    }).map(lambda x: 'color: green' if x > 0 else 'color: red', subset=['P/L $', 'P/L %']), use_container_width=True)

if not df_lots.empty:
    render_table(df_lots.copy(), "1. Current Global Holdings")
    render_table(df_lots[df_lots['Type'] == "Short-Term"].copy(), "2. Short-Term Holdings")
    render_table(df_lots[df_lots['Type'] == "Long-Term"].copy(), "3. Long-Term Holdings")
