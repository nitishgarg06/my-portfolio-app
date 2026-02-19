import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import requests
import base64
from datetime import datetime

# --- 1. CORE CONFIG ---
st.set_page_config(layout="wide", page_title="Wealth Terminal Pro")
st.title("🏦 Wealth Terminal Pro")

try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    GITHUB_REPO = st.secrets["GITHUB_REPO"]
except:
    st.error("Secrets Missing! Add GITHUB_TOKEN and GITHUB_REPO to Streamlit.")
    st.stop()

# --- 2. THE WEALTH ENGINE ---
class WealthEngine:
    @staticmethod
    def clean(val):
        """Forcefully converts IBKR strings to numbers."""
        if pd.isna(val) or str(val).strip() in ['', '--']: return 0.0
        s = str(val).strip().replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
        try: return float(s)
        except: return 0.0

    @staticmethod
    def push(df, path):
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
        headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
        res = requests.get(url, headers=headers)
        sha = res.json().get('sha') if res.status_code == 200 else None
        content = base64.b64encode(df.to_csv(index=False).encode()).decode()
        payload = {"message": f"Sync {path}", "content": content, "branch": "main"}
        if sha: payload["sha"] = sha
        return requests.put(url, headers=headers, json=payload).status_code in [200, 201]

    @staticmethod
    def load(path):
        url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{path}?v={datetime.now().timestamp()}"
        try:
            df = pd.read_csv(url)
            return df if not df.empty else None
        except: return None

# --- 3. SIDEBAR: DATA LIFECYCLE ---
FY_LIST = ["FY24", "FY25", "FY26"]
with st.sidebar:
    st.header("🔄 Control Center")
    sync_fy = st.selectbox("Year to Sync", FY_LIST)
    
    if st.button(f"🚀 Execute Full Sync {sync_fy}"):
        with st.status(f"Harvesting {sync_fy}...") as status:
            conn = st.connection("gsheets", type=GSheetsConnection)
            raw = conn.read(worksheet=sync_fy, ttl=0)
            
            # --- EXTRACT TRADES ---
            t_rows = raw[raw.iloc[:, 0].str.contains('Trades', na=False, case=False)]
            if not t_rows.empty:
                h = t_rows[t_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
                d = t_rows[t_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(h)]
                d.columns = h
                WealthEngine.push(d, f"data/{sync_fy}/trades.csv")
                
                # Update Cache
                tickers = d[next(c for c in d.columns if 'Symbol' in c)].unique().tolist()
                p_data = []
                for t in tickers:
                    try:
                        p = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{t}", headers={'User-Agent': 'Mozilla/5.0'}).json()['chart']['result'][0]['meta']['regularMarketPrice']
                        p_data.append({"Symbol": t, "CurrentPrice": p})
                    except: p_data.append({"Symbol": t, "CurrentPrice": 0.0})
                WealthEngine.push(pd.DataFrame(p_data), "data/price_cache.csv")
            
            # --- EXTRACT PERFORMANCE ---
            p_rows = raw[raw.iloc[:, 0].str.contains('Performance Summary|Realized', na=False, case=False)]
            if not p_rows.empty:
                h = p_rows[p_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
                d = p_rows[p_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(h)]
                d.columns = h
                WealthEngine.push(d, f"data/{sync_fy}/perf.csv")
            
            st.session_state[f'sync_{sync_fy}'] = datetime.now().strftime("%d %b %Y, %H:%M")
            st.rerun()

    if f'sync_{sync_fy}' in st.session_state:
        st.caption(f"📅 **Last Synced:** {st.session_state[f'sync_{sync_fy}']}")

# --- 4. DASHBOARD: CUMULATIVE ENGINE ---
st.divider()
view_fy = st.radio("Display Horizon", FY_LIST, index=len(FY_LIST)-1, horizontal=True)
years = FY_LIST[:FY_LIST.index(view_fy)+1]

all_t, all_p = [], []
for y in years:
    t, p = WealthEngine.load(f"data/{y}/trades.csv"), WealthEngine.load(f"data/{y}/perf.csv")
    if t is not None: t['FY'] = y; all_t.append(t)
    if p is not None: p['FY'] = y; all_p.append(p)
price_cache = WealthEngine.load("data/price_cache.csv")

if all_t:
    df_raw = pd.concat(all_t, ignore_index=True)
    df_perf = pd.concat(all_p, ignore_index=True) if all_p else pd.DataFrame()

    # Column Mapping
    c_q = next(c for c in df_raw.columns if 'Qty' in c or 'Quantity' in c)
    c_p = next(c for c in df_raw.columns if 'Price' in c)
    c_s = next(c for c in df_raw.columns if 'Symbol' in c)
    c_d = next(c for c in df_raw.columns if 'Date' in c)
    c_c = next((c for c in df_raw.columns if 'Comm' in c), None)

    df_raw['Q_v'] = df_raw[c_q].apply(WealthEngine.clean)
    df_raw['P_v'] = df_raw[c_p].apply(WealthEngine.clean)
    df_raw['C_v'] = df_raw[c_c].apply(WealthEngine.clean).abs() if c_c else 0.0
    df_raw['DT_v'] = pd.to_datetime(df_raw[c_d].str.split(',').str[0], errors='coerce')

    # FIFO Calculation
    lots = []
    for ticker in df_raw[c_s].unique():
        sym_df = df_raw[df_raw[c_s] == ticker].sort_values('DT_v')
        open_lots = []
        for _, row in sym_df.iterrows():
            if row['Q_v'] > 0: # Buy
                open_lots.append({'dt': row['DT_v'], 'q': row['Q_v'], 'p': row['P_v'], 'c': row['C_v']})
            elif row['Q_v'] < 0: # Sell
                sq = abs(row['Q_v'])
                while sq > 0 and open_lots:
                    if open_lots[0]['q'] <= sq: sq -= open_lots.pop(0)['q']
                    else: open_lots[0]['q'] -= sq; sq = 0
        for l in open_lots:
            l['Symbol'] = ticker
            l['Age'] = (pd.Timestamp.now() - l['dt']).days
            l['Status'] = "Long-Term" if l['Age'] > 365 else "Short-Term"
            lots.append(l)
    
    df_h = pd.DataFrame(lots)

    # --- TOP LINE METRICS (LIFETIME) ---
    
    
    total_inv = (df_h['q'] * df_h['p']).sum() + df_h['c'].sum() if not df_h.empty else 0.0
    total_comm = df_raw['C_v'].sum()
    
    lt_s = lt_f = 0.0
    if not df_perf.empty:
        rt_col = next((c for c in df_perf.columns if 'Realized' in c and 'Total' in c), None)
        cat_col = next((c for c in df_perf.columns if 'Category' in c or 'Asset' in c), None)
        if rt_col and cat_col:
            df_perf[rt_col] = df_perf[rt_col].apply(WealthEngine.clean)
            lt_s = df_perf[df_perf[cat_col].str.contains('Stock|Equity', na=False, case=False)][rt_col].sum()
            lt_f = df_perf[df_perf[cat_col].str.contains('Forex|Cash|Interest', na=False, case=False)][rt_col].sum()

    st.subheader("🌐 Lifetime Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lifetime Investment", f"${total_inv:,.2f}")
    m2.metric("Total Realized P/L", f"${(lt_s + lt_f):,.2f}")
    m3.metric("Stocks Realized", f"${lt_s:,.2f}")
    m4.metric("Forex/Interest", f"${lt_f:,.2f}")
    st.info(f"💰 **Total Lifetime Commissions:** ${total_comm:,.2f}")

    # --- TABLES ---
    def render_table(data, title):
        st.subheader(f"{title} (as of {datetime.now().strftime('%d %b %Y')})")
        if data.empty: return st.info("No active holdings found.")
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
            "Units":"{:.2f}", "Avg Cost":"${:.2f}", "Comms":"${:.2f}", "Current Price":"${:.2f}",
            "Total Basis":"${:.2f}", "Market Value":"${:.2f}", "P/L $":"${:.2f}", "P/L %":"{:.2f}%"
        }), use_container_width=True)

    st.divider()
    render_table(df_h, "1. Current Global Holdings")

    # --- FIFO CALCULATOR ---
    
    st.divider()
    st.header("🧮 FIFO Selling Calculator")
    sel_ticker = st.selectbox("Select Stock to Simulate", df_h['Symbol'].unique())
    h_data = df_h[df_h['Symbol'] == sel_ticker]
    u_total, a_cost = h_data['q'].sum(), h_data['p'].mean()
    
    c1, c2 = st.columns(2)
    mode = c1.radio("Input Style", ["Units", "Percentage (%)"])
    s_qty = c2.slider("Quantity to Sell", 0.0, float(u_total)) if mode == "Units" else u_total * (c2.slider("% to Sell", 0, 100, 25)/100)
    target = c2.number_input("Target Profit %", value=110.0)
    
    st.success(f"**Target Price:** ${(a_cost * (target/100)):,.2f} | **Residual:** {u_total - s_qty:.2f} units.")
else:
    st.info("👋 No data found. Please run the sync in the sidebar (starting with FY24) to populate the terminal.")
