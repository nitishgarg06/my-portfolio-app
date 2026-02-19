import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

# --- 1. ROBUST UTILS ---
def safe_num(val):
    if val is None or pd.isna(val) or str(val).strip() == '': return 0.0
    s = str(val).strip().replace('$', '').replace(',', '')
    if '(' in s and ')' in s: s = '-' + s.replace('(', '').replace(')', '')
    try: return float(s)
    except: return 0.0

def safe_find(df, keywords):
    """Fuzzy column finder."""
    for col in df.columns:
        if any(k.lower() in str(col).lower() for k in keywords): return col
    return None

@st.cache_data(ttl=600)
def get_google_price(ticker):
    try:
        url = f"https://www.google.com/search?q={ticker}+stock+price"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=3)
        soup = BeautifulSoup(response.text, 'html.parser')
        price_tag = soup.find('span', {'class': 'I67upf'}) or soup.find('span', {'jsname': 'vW79of'})
        return safe_num(price_tag.text) if price_tag else 0.0
    except: return 0.0

def parse_ibkr_grid(df):
    sections = {}
    if df is None or df.empty: return sections
    df = df.astype(str).replace('nan', '')
    for name in df.iloc[:, 0].unique():
        if name in ['', 'Statement']: continue
        sec_df = df[df.iloc[:, 0] == name]
        h_row = sec_df[sec_df.iloc[:, 1] == 'Header']
        d_rows = sec_df[sec_df.iloc[:, 1] == 'Data']
        if not h_row.empty and not d_rows.empty:
            cols = [c for c in h_row.iloc[0, 2:].tolist() if c]
            data = d_rows.iloc[:, 2:2+len(cols)]
            data.columns = cols
            sections[name] = data
    return sections

# --- 2. THE APP BODY ---
st.title("🏦 Wealth Terminal Pro")
curr_date = datetime.now().strftime('%d %b %Y')

try:
    conn = st.connection("gsheets", type=GSheetsConnection)
    tabs = ["FY24", "FY25", "FY26"]
    fy_map = {}
    all_trades_list = []

    for t in tabs:
        raw = conn.read(worksheet=t, ttl=0)
        if raw is not None:
            parsed = parse_ibkr_grid(raw)
            fy_map[t] = parsed
            if "Trades" in parsed: all_trades_list.append(parsed["Trades"])

    # --- 3. FIFO ENGINE ---
    df_lots = pd.DataFrame()
    if all_trades_list:
        trades = pd.concat(all_trades_list, ignore_index=True)
        trades.columns = trades.columns.str.strip()
        
        # Fuzzy column discovery
        sym_c = safe_find(trades, ['Symbol'])
        qty_c = safe_find(trades, ['Qty', 'Quantity'])
        prc_c = safe_find(trades, ['Price'])
        dt_c = safe_find(trades, ['Date'])
        comm_c = safe_find(trades, ['Comm'])

        if all([sym_c, qty_c, prc_c, dt_c]):
            trades['q_v'] = trades[qty_c].apply(safe_num)
            trades['p_v'] = trades[prc_c].apply(safe_num)
            trades['c_v'] = trades[comm_c].apply(safe_num).abs() if comm_c else 0.0
            trades['d_v'] = pd.to_datetime(trades[dt_c].str.split(',').str[0], errors='coerce')
            trades = trades.dropna(subset=['d_v']).sort_values('d_v')

            holdings = []
            for sym in trades[sym_c].unique():
                lots = []
                sym_trades = trades[trades[sym_c] == sym]
                for _, row in sym_trades.iterrows():
                    if row['q_v'] > 0: 
                        lots.append({'date': row['d_v'], 'qty': row['q_v'], 'price': row['p_v'], 'comm': row['c_v']})
                    elif row['q_v'] < 0:
                        sq = abs(row['q_v'])
                        while sq > 0 and lots:
                            if lots[0]['qty'] <= sq: sq -= lots[0].pop('qty'); lots.pop(0)
                            else: lots[0]['qty'] -= sq; sq = 0
                for l in lots:
                    l['Symbol'] = sym
                    l['Type'] = "Long-Term" if (pd.Timestamp.now() - l['date']).days > 365 else "Short-Term"
                    holdings.append(l)
            df_lots = pd.DataFrame(holdings)

    # --- 4. TOP METRICS ---
    sel_fy = st.selectbox("Financial Year View", tabs, index=len(tabs)-1)
    data_fy = fy_map.get(sel_fy, {})
    
    stocks_pl, forex_pl = 0.0, 0.0
    if 'Realized & Unrealized Performance Summary' in data_fy:
        pdf = data_fy['Realized & Unrealized Performance Summary']
        rt_c = safe_find(pdf, ['Realized Total'])
        ct_c = safe_find(pdf, ['Asset Category'])
        if rt_c and ct_c:
            pdf = pdf[~pdf['Symbol'].str.contains('Total|Asset', case=False, na=False)]
            stocks_pl = pdf[pdf[ct_c].str.contains('Stock', case=False)][rt_c].apply(safe_num).sum()
            forex_pl = pdf[pdf[ct_c].str.contains('Forex|Cash', case=False)][rt_c].apply(safe_num).sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("Net Realized P/L", f"${(stocks_pl + forex_pl):,.2f}")
    m2.metric("Stocks Portion", f"${stocks_pl:,.2f}")
    m3.metric("Forex/Cash Impact", f"${forex_pl:,.2f}")

    # --- 5. HOLDINGS TABLE ---
    st.divider()
    if not df_lots.empty:
        agg = df_lots.groupby('Symbol').agg({'qty': 'sum', 'price': 'mean', 'comm': 'sum'}).reset_index()
        
        # Safe Price Map
        live_prices = {}
        for tkr in agg['Symbol']:
            p = get_google_price(tkr)
            live_prices[tkr] = p if p > 0 else agg[agg['Symbol'] == tkr]['price'].iloc[0]

        agg['Live Price'] = agg['Symbol'].map(live_prices)
        agg['Basis'] = (agg['qty'] * agg['price']) + agg['comm']
        agg['Value'] = agg['qty'] * agg['Live Price']
        agg['P/L $'] = agg['Value'] - agg['Basis']
        agg['P/L %'] = (agg['P/L $'] / agg['Basis']) * 100
        
        agg.columns = ['Ticker', 'Units', 'Avg Cost', 'Total Comm', 'Live Price', 'Basis', 'Value', 'P/L $', 'P/L %']
        agg.index = range(1, len(agg) + 1)
        
        st.subheader(f"1. Current Global Holdings (as of {curr_date})")
        st.dataframe(agg.style.format({
            "Avg Cost": "${:.2f}", "Live Price": "${:.2f}", "Basis": "${:.2f}", 
            "Value": "${:.2f}", "P/L $": "${:.2f}", "P/L %": "{:.2f}%"
        }).map(lambda x: 'color: green' if x > 0 else 'color: red', subset=['P/L $', 'P/L %']), use_container_width=True)
    else:
        st.info("No current holdings identified. Check the Debugger below.")
        with st.expander("🛠️ Debugger: Raw Trade Data Found"):
            if all_trades_list:
                st.write(pd.concat(all_trades_list, ignore_index=True))
            else:
                st.write("No 'Trades' section found in any tab.")

except Exception as e:
    st.error(f"Critical Error: {e}")
