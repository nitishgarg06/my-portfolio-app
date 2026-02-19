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
    st.error("Secrets Missing: Add GITHUB_TOKEN and GITHUB_REPO to Streamlit Secrets.")
    st.stop()

# --- 2. THE ULTIMATE EXTRACTION ENGINE ---
def clean_numeric(val):
    if val is None or pd.isna(val) or str(val).strip() == '': return 0.0
    s = str(val).strip().replace('$', '').replace(',', '')
    if '(' in s and ')' in s: s = '-' + s.replace('(', '').replace(')', '')
    try: return float(s)
    except: return 0.0

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

# --- 3. SIDEBAR: THE SYNC CONTROL ---
st.title("🏦 Wealth Terminal Pro")
FY_LIST = ["FY24", "FY25", "FY26"]

with st.sidebar:
    st.header("🔄 Control Center")
    sync_fy = st.selectbox("Select Year to Sync", FY_LIST)
    
    if st.button(f"🚀 Execute Full Sync for {sync_fy}"):
        with st.status(f"Scanning GSheet: {sync_fy}...") as status:
            conn = st.connection("gsheets", type=GSheetsConnection)
            raw = conn.read(worksheet=sync_fy, ttl=0)
            
            # --- TRADES ---
            t_rows = raw[raw.iloc[:, 0].str.contains('Trades', na=False, case=False)]
            if not t_rows.empty:
                h = t_rows[t_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
                d = t_rows[t_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(h)]
                d.columns = h
                push_to_github(d, f"data/{sync_fy}/trades.csv")
                
                # Update Price Cache
                tickers = d[next(c for c in d.columns if 'Symbol' in c)].unique().tolist()
                p_data = []
                for t in tickers:
                    try:
                        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t}"
                        p = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).json()['chart']['result'][0]['meta']['regularMarketPrice']
                        p_data.append({"Symbol": t, "CurrentPrice": p})
                    except: p_data.append({"Symbol": t, "CurrentPrice": 0.0})
                push_to_github(pd.DataFrame(p_data), "data/price_cache.csv")

            # --- PERFORMANCE (FIXED) ---
            p_rows = raw[raw.iloc[:, 0].str.contains('Performance Summary|Realized', na=False, case=False)]
            if not p_rows.empty:
                h = p_rows[p_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
                d = p_rows[p_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(h)]
                d.columns = h
                push_to_github(d, f"data/{sync_fy}/perf.csv")
            
            st.session_state[f'sync_{sync_fy}'] = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.rerun()

    if f'sync_{sync_fy}' in st.session_state:
        st.caption(f"📅 Last Sync: {st.session_state[f'sync_{sync_fy}']}")

# --- 4. THE DASHBOARD ---
st.divider()
view_fy = st.radio("Dashboard View", FY_LIST, index=len(FY_LIST)-1, horizontal=True)
years = FY_LIST[:FY_LIST.index(view_fy)+1]

all_t, all_p = [], []
with st.expander("🛠️ Data Health Inspector"):
    for y in years:
        t = load_from_github(f"data/{y}/trades.csv")
        p = load_from_github(f"data/{y}/perf.csv")
        st.write(f"**{y}:** Trades {'✅' if t is not None else '❌'} | Performance {'✅' if p is not None else '❌'}")
        if t is not None: t['FY'] = y; all_t.append(t)
        if p is not None: p['FY'] = y; all_p.append(p)
    
    price_cache = load_from_github("data/price_cache.csv")
    st.write(f"**Price Cache:** {'✅' if price_cache is not None else '❌'}")

if all_t:
    df_trades = pd.concat(all_t, ignore_index=True)
    df_perf = pd.concat(all_p, ignore_index=True) if all_p else pd.DataFrame()

    # --- FIFO ENGINE ---
    # Fuzzy column mapping to handle naming variations
    c_q = next(c for c in df_trades.columns if 'Qty' in c or 'Quantity' in c)
    c_p = next(c for c in df_trades.columns if 'Price' in c)
    c_s = next(c for c in df_trades.columns if 'Symbol' in c)
    c_d = next(c for c in df_trades.columns if 'Date' in c)
    c_c = next((c for c in df_trades.columns if 'Comm' in c), None)

    df_trades['Q_v'] = df_trades[c_q].apply(clean_numeric)
    df_trades['P_v'] = df_trades[c_p].apply(clean_numeric)
    df_trades['C_v'] = df_trades[c_c].apply(clean_numeric).abs() if c_c else 0.0
    df_trades['DT_v'] = pd.to_datetime(df_trades[c_d].str.split(',').str[0])

    open_lots = []
    for ticker in df_trades[c_s].unique():
        sym_df = df_trades[df_trades[c_s] == ticker].sort_values('DT_v')
        lots = []
        for _, row in sym_df.iterrows():
            if row['Q_v'] > 0: lots.append({'dt': row['DT_v'], 'q': row['Q_v'], 'p': row['P_v'], 'c': row['C_v']})
            elif row['Q_v'] < 0:
                sq = abs(row['Q_v'])
                while sq > 0 and lots:
                    if lots[0]['q'] <= sq: sq -= lots.pop(0)['q']
                    else: lots[0]['q'] -= sq; sq = 0
        for l in lots:
            l['Symbol'] = ticker
            l['Status'] = "Long-Term" if (pd.Timestamp.now() - l['dt']).days > 365 else "Short-Term"
            open_lots.append(l)
    
    df_h = pd.DataFrame(open_lots)

    # --- KPI CALCULATIONS ---
    
    
    total_inv = (df_h['q'] * df_h['p']).sum() if not df_h.empty else 0.0
    total_comm = df_trades['C_v'].sum()
    
    lt_s = lt_f = 0.0
    if not df_perf.empty:
        rt_col = next((c for c in df_perf.columns if 'Realized' in c and 'Total' in c), None)
        cat_col = next((c for c in df_perf.columns if 'Category' in c or 'Asset' in c), None)
        if rt_col and cat_col:
            df_perf[rt_col] = df_perf[rt_col].apply(clean_numeric)
            lt_s = df_perf[df_perf[cat_col].str.contains('Stock|Equity', na=False, case=False)][rt_col].sum()
            lt_f = df_perf[df_perf[cat_col].str.contains('Forex|Cash|Interest', na=False, case=False)][rt_col].sum()

    st.subheader("🌐 Lifetime Overview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Lifetime Investment", f"${total_inv:,.2f}")
    k2.metric("Total Realized P/L", f"${(lt_s + lt_f):,.2f}")
    k3.metric("Stocks Portion", f"${lt_s:,.2f}")
    k4.metric("Forex/Interest", f"${lt_f:,.2f}")
    st.info(f"💰 **Lifetime Commissions:** ${total_comm:,.2f} | Realized P/L is net of these fees.")

    # --- TABLES ---
    def render(data, title):
        st.subheader(f"{title} (as of {datetime.now().strftime('%d %b %Y')})")
        if data.empty: return st.info("No active holdings.")
        agg = data.groupby('Symbol').agg({'q': 'sum', 'p': 'mean', 'c': 'sum'}).reset_index()
        if price_cache is not None:
            agg = agg.merge(price_cache[['Symbol', 'CurrentPrice']], on='Symbol', how='left').fillna(0)
        else: agg['CurrentPrice'] = 0.0
        
        agg['Total Basis'] = (agg['q'] * agg['p']) + agg['c']
        agg['Market Value'] = agg['q'] * agg['CurrentPrice']
        agg['P/L $'] = agg['Market Value'] - agg['Total Basis']
        agg['P/L %'] = (agg['P/L $'] / agg['Total Basis'] * 100) if not agg.empty else 0.0
        
        agg.columns = ['Ticker', 'Units', 'Avg Cost', 'Comms', 'Current Price', 'Total Basis', 'Market Value', 'P/L $', 'P/L %']
        agg.index = range(1, len(agg) + 1)
        st.dataframe(agg.style.format({
            "Units": "{:.2f}", "Avg Cost": "${:.2f}", "Comms": "${:.2f}", "Current Price": "${:.2f}",
            "Total Basis": "${:.2f}", "Market Value": "${:.2f}", "P/L $": "${:.2f}", "P/L %": "{:.2f}%"
        }), use_container_width=True)

    st.divider()
    render(df_h, "1. Current Global Holdings")
    c1, c2 = st.columns(2)
    with c1: render(df_h[df_h['Status'] == "Short-Term"], "2. Short-Term Holdings")
    with c2: render(df_h[df_h['Status'] == "Long-Term"], "3. Long-Term Holdings")

    # --- CALCULATOR ---
    
    st.divider()
    st.header("🧮 FIFO Selling Calculator")
    sel = st.selectbox("Stock to Simulate", df_h['Symbol'].unique())
    u_tot = df_h[df_h['Symbol'] == sel]['q'].sum()
    a_cost = df_h[df_h['Symbol'] == sel]['p'].mean()
    
    col_a, col_b = st.columns(2)
    mode = col_a.radio("Input Mode", ["Units", "Percentage (%)"])
    qty = col_b.slider("Qty to Sell", 0.0, float(u_tot)) if mode == "Units" else u_tot * (col_b.slider("% to Sell", 0, 100, 25)/100)
    target = col_b.number_input("Target Profit %", value=110.0)
    
    st.success(f"**Target Price:** ${(a_cost * (target/100)):,.2f} | **Residual:** {u_tot - qty:.2f} units.")

else:
    st.warning("⚠️ No data found on GitHub. You must Sync EACH year starting with FY24.")
