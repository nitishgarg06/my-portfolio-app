import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import requests
import base64
from datetime import datetime

# --- 1. CONFIG & SECRETS ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    GITHUB_REPO = st.secrets["GITHUB_REPO"]
except:
    st.error("Secrets Missing: Check GITHUB_TOKEN and GITHUB_REPO in Streamlit Settings.")
    st.stop()

# --- 2. CORE UTILITIES ---
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
    """Fetches prices once to save into the Static Cache CSV."""
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

# --- 3. GITHUB PERSISTENCE ---
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
    try: return pd.read_csv(url)
    except: return None

# --- 4. SIDEBAR SYNC ENGINE ---
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
            trades_rows = raw[raw.iloc[:, 0].str.contains('Trades', na=False, case=False)]
            if not trades_rows.empty:
                h = trades_rows[trades_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
                d = trades_rows[trades_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(h)]
                d.columns = h
                push_to_github(d, f"data/{sync_fy}/trades.csv")
                
                # Update Price Cache for all current tickers
                price_df = fetch_static_prices(d['Symbol'].unique().tolist())
                push_to_github(price_df, "data/price_cache.csv")
            
            perf_rows = raw[raw.iloc[:, 0].str.contains('Performance Summary', na=False, case=False)]
            if not perf_rows.empty:
                h = perf_rows[perf_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
                d = perf_rows[perf_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(h)]
                d.columns = h
                push_to_github(d, f"data/{sync_fy}/perf.csv")
            
            st.success("Sync Complete!")
            st.rerun()

# --- 5. CUMULATIVE DASHBOARD ---
st.divider()
view_fy = st.radio("Select Dashboard View (Cumulative)", FY_LIST, index=len(FY_LIST)-1, horizontal=True)
years_active = FY_LIST[:FY_LIST.index(view_fy)+1]

# Load Waterfall
all_t, all_p = [], []
for y in years_active:
    t, p = load_from_github(f"data/{y}/trades.csv"), load_from_github(f"data/{y}/perf.csv")
    if t is not None: t['Src_FY'] = y; all_t.append(t)
    if p is not None: p['Src_FY'] = y; all_p.append(p)

prices_df = load_from_github("data/price_cache.csv")

if all_t:
    df_trades = pd.concat(all_t)
    df_perf = pd.concat(all_p) if all_p else pd.DataFrame()

    # FIFO Engine
    c_q, c_p, c_s, c_d, c_c = fuzzy_find(df_trades, ['Qty']), fuzzy_find(df_trades, ['Price']), fuzzy_find(df_trades, ['Symbol']), fuzzy_find(df_trades, ['Date']), fuzzy_find(df_trades, ['Comm'])
    df_trades['Q_v'] = df_trades[c_q].apply(clean_numeric)
    df_trades['P_v'] = df_trades[c_p].apply(clean_numeric)
    df_trades['C_v'] = df_trades[c_c].apply(clean_numeric).abs() if c_c else 0.0
    df_trades['DT_v'] = pd.to_datetime(df_trades[c_d].str.split(',').str[0], errors='coerce')

    open_lots = []
    for ticker in df_trades[c_s].unique():
        sym_df = df_trades[df_trades[c_s] == ticker].sort_values('DT_v')
        lots = []
        for _, row in sym_df.iterrows():
            if row['Q_v'] > 0:
                lots.append({'dt': row['DT_v'], 'q': row['Q_v'], 'p': row['P_v'], 'c': row['C_v']})
            elif row['Q_v'] < 0:
                sell_q = abs(row['Qty_v'])
                while sell_q > 0 and lots:
                    if lots[0]['q'] <= sell_q: sell_q -= lots.pop(0)['q']
                    else: lots[0]['q'] -= sell_q; sell_q = 0
        for l in lots:
            l['Symbol'] = ticker
            l['Status'] = "Long-Term" if (pd.Timestamp.now() - l['dt']).days > 365 else "Short-Term"
            open_lots.append(l)
    
    df_h = pd.DataFrame(open_lots)

    # --- TOP LINE KPIs ---
    
    
    total_inv = (df_h['q'] * df_h['p']).sum() + df_h['c'].sum() if not df_h.empty else 0.0
    
    cat_c, rt_c = fuzzy_find(df_perf, ['Category']), fuzzy_find(df_perf, ['Realized Total'])
    lt_s = df_perf[df_perf[cat_c].str.contains('Stock', na=False, case=False)][rt_c].apply(clean_numeric).sum() if not df_perf.empty else 0.0
    lt_f = df_perf[df_perf[cat_c].str.contains('Forex|Cash', na=False, case=False)][rt_c].apply(clean_numeric).sum() if not df_perf.empty else 0.0

    st.subheader("🌐 Lifetime Overview")
    k1, k2, k3 = st.columns(3)
    k1.metric("Lifetime Investment", f"${total_inv:,.2f}")
    k2.metric("Lifetime Realized P/L", f"${(lt_s + lt_f):,.2f}")
    k3.metric("Lifetime Forex Impact", f"${lt_f:,.2f}")
    st.caption("ℹ️ *Disclaimer: Total Realized P/L is net of commissions.*")

    # --- TABLES ---
    def render_table(data, title):
        st.subheader(f"{title} (as of {curr_date})")
        if data.empty: return st.info("No active holdings.")
        
        agg = data.groupby('Symbol').agg({'q': 'sum', 'p': 'mean', 'c': 'sum'}).reset_index()
        
        # Merge with Static Price Cache
        if prices_df is not None:
            agg = agg.merge(prices_df[['Symbol', 'StaticPrice']], on='Symbol', how='left').fillna(0)
        else:
            agg['StaticPrice'] = 0.0

        agg['Total Basis'] = (agg['q'] * agg['p']) + agg['c']
        agg['Value'] = agg['q'] * agg['StaticPrice']
        agg['P/L $'] = agg['Value'] - agg['Total Basis']
        agg['P/L %'] = (agg['P/L $'] / agg['Total Basis'] * 100) if not agg.empty else 0.0
        
        agg.columns = ['Ticker', 'Units', 'Avg Cost', 'Comms', 'Live Price', 'Total Basis', 'Market Value', 'P/L $', 'P/L %']
        st.dataframe(agg.style.format({"Units": "{:.2f}", "Avg Cost": "${:.2f}", "Comms": "${:.2f}", "Live Price": "${:.2f}", "Total Basis": "${:.2f}", "Market Value": "${:.2f}", "P/L $": "${:.2f}", "P/L %": "{:.2f}%"}), use_container_width=True)

    st.divider()
    render_table(df_h, "1. Current Global Holdings")
    c1, c2 = st.columns(2)
    with c1: render_table(df_h[df_h['Status'] == "Short-Term"], "2. Short-Term Holdings")
    with c2: render_table(df_h[df_h['Status'] == "Long-Term"], "3. Long-Term Holdings")

else:
    st.info("No data found. Use the sidebar to sync your Financial Years from Google Sheets.")
