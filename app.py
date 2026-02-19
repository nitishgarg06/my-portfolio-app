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
    for col in df.columns:
        if any(k.lower() in str(col).lower() for k in keywords): return col
    return None

@st.cache_data(ttl=600)
def get_google_price(ticker):
    """Scrapes Google Finance for a price. Returns 0.0 on failure."""
    try:
        url = f"https://www.google.com/search?q={ticker}+stock+price"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=3)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Common Google Finance Selectors
        price_tag = soup.find('span', {'class': 'I67upf'}) or soup.find('span', {'jsname': 'vW79of'})
        return safe_num(price_tag.text) if price_tag else 0.0
    except:
        return 0.0

# --- 2. THE APP BODY (ALWAYS RENDERS) ---
st.title("🏦 Wealth Terminal Pro")
curr_date = datetime.now().strftime('%d %b %Y')

try:
    conn = st.connection("gsheets", type=GSheetsConnection)
    tabs = ["FY24", "FY25", "FY26"]
    fy_map = {}
    all_trades = []

    # 3. DATA LOADING WITH FEEDBACK
    for t in tabs:
        try:
            raw = conn.read(worksheet=t, ttl=0)
            if raw is not None and not raw.empty:
                # Basic IBKR Parsing
                df = raw.astype(str).replace('nan', '')
                sections = {}
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
                fy_map[t] = sections
                if "Trades" in sections: all_trades.append(sections["Trades"])
        except Exception as e:
            st.sidebar.warning(f"Could not load {t}: {e}")

    # 4. FIFO ENGINE (INSIDE A SAFETY BOX)
    df_lots = pd.DataFrame()
    if all_trades:
        try:
            trades = pd.concat(all_trades, ignore_index=True)
            trades.columns = trades.columns.str.strip()
            q_c = safe_find(trades, ['Qty'])
            p_c = safe_find(trades, ['Price'])
            
            trades['qty_v'] = trades[q_c].apply(safe_num)
            trades['prc_v'] = trades[p_c].apply(safe_num)
            trades['dt_v'] = pd.to_datetime(trades[safe_find(trades, ['Date'])].str.split(',').str[0], errors='coerce')
            trades = trades.dropna(subset=['dt_v']).sort_values('dt_v')

            holdings = []
            for sym in trades['Symbol'].unique():
                lots = []
                for _, row in trades[trades['Symbol'] == sym].iterrows():
                    if row['qty_v'] > 0: 
                        lots.append({'date': row['dt_v'], 'qty': row['qty_v'], 'price': row['prc_v']})
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
        except Exception as e:
            st.error(f"FIFO Calculation Error: {e}")

    # 5. METRICS RENDERING
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

    # 6. HOLDINGS TABLE
    st.divider()
    if not df_lots.empty:
        agg = df_lots.groupby('Symbol').agg({'qty': 'sum', 'price': 'mean'}).reset_index()
        # Fetch prices ONE BY ONE and check for 0.0
        live_prices = {}
        for tkr in agg['Symbol']:
            p = get_google_price(tkr)
            live_prices[tkr] = p if p > 0 else agg[agg['Symbol'] == tkr]['price'].iloc[0]

        agg['Live Price'] = agg['Symbol'].map(live_prices)
        agg['Total Basis'] = agg['qty'] * agg['price']
        agg['Market Value'] = agg['qty'] * agg['Live Price']
        agg['P/L $'] = agg['Market Value'] - agg['Total Basis']
        agg.columns = ['Ticker', 'Units', 'Avg Cost', 'Live Price', 'Basis', 'Value', 'P/L $']
        
        st.subheader(f"1. Current Global Holdings (as of {curr_date})")
        st.dataframe(agg.style.format("${:.2f}", subset=['Avg Cost', 'Live Price', 'Basis', 'Value', 'P/L $'])
                     .map(lambda x: 'color: green' if x > 0 else 'color: red', subset=['P/L $']), use_container_width=True)
    else:
        st.info("No current holdings found. Your Realized P/L is displayed above.")

except Exception as global_e:
    st.error(f"Critical App Error: {global_e}")
    st.write("Please check your GSheet tab names and column headers.")
