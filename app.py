import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import requests
import base64
from datetime import datetime

# --- 1. CONFIG ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide")

try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    GITHUB_REPO = st.secrets["GITHUB_REPO"]
except:
    st.error("Secrets Missing: Ensure GITHUB_TOKEN and GITHUB_REPO are in Streamlit Secrets.")
    st.stop()

# --- 2. HELPERS ---
def clean_numeric(val):
    if val is None or pd.isna(val) or str(val).strip() == '': return 0.0
    s = str(val).strip().replace('$', '').replace(',', '')
    if '(' in s and ')' in s: s = '-' + s.replace('(', '').replace(')', '')
    try: return float(s)
    except: return 0.0

def fuzzy_find(df, keywords):
    for col in df.columns:
        if any(k.lower() in str(col).lower() for k in keywords): return col
    return None

def fetch_static_prices(tickers):
    price_data = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    for t in tickers:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t}"
            res = requests.get(url, headers=headers, timeout=5).json()
            p = res['chart']['result'][0]['meta']['regularMarketPrice']
            price_data.append({"Symbol": t, "StaticPrice": p, "CacheDate": datetime.now().strftime("%Y-%m-%d %H:%M")})
        except:
            price_data.append({"Symbol": t, "StaticPrice": 0.0, "CacheDate": "Error"})
    return pd.DataFrame(price_data)

# --- 3. GITHUB ENGINE ---
def push_to_github(df, path):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    res = requests.get(url, headers=headers)
    sha = res.json().get('sha') if res.status_code == 200 else None
    content = base64.b64encode(df.to_csv(index=False).encode()).decode()
    payload = {"message": f"Sync {path}", "content": content, "branch": "main"}
    if sha: payload["sha"] = sha
    return requests.put(url, headers=headers, json=payload).status_code in [200, 201]

def load_from_github(path):
    url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{path}?v={datetime.now().timestamp()}"
    try:
        df = pd.read_csv(url)
        return df if not df.empty else None
    except: return None

# --- 4. SIDEBAR SYNC ---
st.title("🏦 Wealth Terminal Pro")
FY_LIST = ["FY24", "FY25", "FY26"]
curr_date = datetime.now().strftime('%d %b %Y')

with st.sidebar:
    st.header("🔄 On-Demand Sync")
    sync_fy = st.selectbox("Year to Sync", FY_LIST)
    if st.button(f"🚀 Sync {sync_fy}"):
        with st.status(f"Processing {sync_fy}...") as status:
            conn = st.connection("gsheets", type=GSheetsConnection)
            raw = conn.read(worksheet=sync_fy, ttl=0)
            
            # Extract Sections
            t_rows = raw[raw.iloc[:, 0].str.contains('Trades', na=False, case=False)]
            if not t_rows.empty:
                h = t_rows[t_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
                d = t_rows[t_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(h)]
                d.columns = h
                push_to_github(d, f"data/{sync_fy}/trades.csv")
                
                # Update Cache
                price_df = fetch_static_prices(d['Symbol'].unique().tolist())
                push_to_github(price_df, "data/price_cache.csv")
            
            p_rows = raw[raw.iloc[:, 0].str.contains('Performance Summary', na=False, case=False)]
            if not p_rows.empty:
                h = p_rows[p_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
                d = p_rows[p_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(h)]
                d.columns = h
                push_to_github(d, f"data/{sync_fy}/perf.csv")
            
            st.success("Sync Done!")
            st.rerun()

# --- 5. CUMULATIVE LOAD ---
st.divider()
view_fy = st.radio("Cumulative View", FY_LIST, index=len(FY_LIST)-1, horizontal=True)
years_to_load = FY_LIST[:FY_LIST.index(view_fy)+1]

all_t, all_p = [], []
with st.expander("🛠️ System Health Check (GitHub Data Status)"):
    for y in years_to_load:
        t = load_from_github(f"data/{y}/trades.csv")
        p = load_from_github(f"data/{y}/perf.csv")
        st.write(f"**{y}:** Trades {'✅' if t is not None else '❌'} | Performance {'✅' if p is not None else '❌'}")
        if t is not None: t['Source_FY'] = y; all_t.append(t)
        if p is not None: p['Source_FY'] = y; all_p.append(p)
    
    prices_cache = load_from_github("data/price_cache.csv")
    st.write(f"**Price Cache:** {'✅' if prices_cache is not None else '❌'}")

# --- 6. DASHBOARD LOGIC ---
if all_t:
    df_trades = pd.concat(all_t, ignore_index=True)
    df_perf = pd.concat(all_p, ignore_index=True) if all_p else pd.DataFrame()

    # Column Mapping
    c_q = fuzzy_find(df_trades, ['Qty', 'Quantity'])
    c_p = fuzzy_find(df_trades, ['Price', 'T. Price'])
    c_s = fuzzy_find(df_trades, ['Symbol', 'Ticker'])
    c_d = fuzzy_find(df_trades, ['Date'])
    c_c = fuzzy_find(df_trades, ['Comm'])

    df_trades['Q_v'] = df_trades[c_q].apply(clean_numeric)
    df_trades['P_v'] = df_trades[c_p].apply(clean_numeric)
    df_trades['C_v'] = df_trades[c_c].apply(clean_numeric).abs() if c_c else 0.0
    df_trades['DT_v'] = pd.to_datetime(df_trades[c_d].str.split(',').str[0], errors='coerce')

    # FIFO Engine
    open_lots = []
    for ticker in df_trades[c_s].unique():
        sym_df = df_trades[df_trades[c_s] == ticker].sort_values('DT_v')
        lots = []
        for _, row in sym_df.iterrows():
            if row['Q_v'] > 0: # BUY
                lots.append({'dt': row['DT_v'], 'q': row['Q_v'], 'p': row['P_v'], 'c': row['C_v']})
            elif row['Q_v'] < 0: # SELL
                sq = abs(row['Q_v'])
                while sq > 0 and lots:
                    if lots[0]['q'] <= sq: sq -= lots.pop(0)['q']
                    else: lots[0]['q'] -= sq; sq = 0
        for l in lots:
            l['Symbol'] = ticker
            l['Status'] = "Long-Term" if (pd.Timestamp.now() - l['dt']).days > 365 else "Short-Term"
            open_lots.append(l)
    
    df_h = pd.DataFrame(open_lots)

    # Lifetime Metrics
    total_inv = (df_h['q'] * df_h['p']).sum() + df_h['c'].sum() if not df_h.empty else 0.0
    
    # Performance Metrics
    cat_c, rt_c = fuzzy_find(df_perf, ['Category']), fuzzy_find(df_perf, ['Realized Total', 'Realized P/L'])
    lt_stocks = lt_forex = 0.0
    if not df_perf.empty and cat_c and rt_c:
        lt_stocks = df_perf[df_perf[cat_c].str.contains('Stock', na=False, case=False)][rt_c].apply(clean_numeric).sum()
        lt_forex = df_perf[df_perf[cat_c].str.contains('Forex|Cash', na=False, case=False)][rt_c].apply(clean_numeric).sum()

    st.subheader("🌐 Lifetime Overview")
    k1, k2, k3 = st.columns(3)
    k1.metric("Lifetime Investment", f"${total_inv:,.2f}")
    k2.metric("Lifetime Realized P/L", f"${(lt_stocks + lt_forex):,.2f}")
    k3.metric("Lifetime Forex Impact", f"${lt_forex:,.2f}")
    st.caption("ℹ️ *Disclaimer: Total Realized P/L is net of commissions.*")

    # Tables
    
    
    def render_table(data, title):
        st.subheader(f"{title} (as of {curr_date})")
        if data.empty: return st.info("No active holdings found.")
        
        agg = data.groupby('Symbol').agg({'q': 'sum', 'p': 'mean', 'c': 'sum'}).reset_index()
        
        # Merge Price Cache
        if prices_cache is not None:
            agg = agg.merge(prices_cache[['Symbol', 'StaticPrice']], on='Symbol', how='left').fillna(0)
        else:
            agg['StaticPrice'] = 0.0
            
        agg['Total Basis'] = (agg['q'] * agg['p']) + agg['c']
        agg['Market Value'] = agg['q'] * agg['StaticPrice']
        agg['P/L $'] = agg['Market Value'] - agg['Total Basis']
        agg['P/L %'] = (agg['P/L $'] / agg['Total Basis'] * 100) if not agg.empty else 0.0
        
        agg.columns = ['Ticker', 'Units', 'Avg Cost', 'Comms', 'Live Price', 'Total Basis', 'Market Value', 'P/L $', 'P/L %']
        st.dataframe(agg.style.format({"Units": "{:.2f}", "Avg Cost": "${:.2f}", "Comms": "${:.2f}", "Live Price": "${:.2f}", "Total Basis": "${:.2f}", "Market Value": "${:.2f}", "P/L $": "${:.2f}", "P/L %": "{:.2f}%"}), use_container_width=True)

    st.divider()
    render_table(df_h, "1. Current Global Holdings")
    c1, c2 = st.columns(2)
    with c1: render_table(df_h[df_h['Status'] == "Short-Term"], "2. Short-Term Holdings")
    with c2: render_table(df_h[df_h['Status'] == "Long-Term"], "3. Long-Term Holdings")
else:
    st.info("No data found on GitHub. Please sync your years in the sidebar.")
