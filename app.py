import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import requests
import base64
from datetime import datetime

# --- 1. SETTINGS & SECRETS ---
st.set_page_config(layout="wide", page_title="Wealth Terminal Pro")
st.title("🏦 Wealth Terminal Pro")

try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    GITHUB_REPO = st.secrets["GITHUB_REPO"]
except:
    st.error("Missing Secrets: Add GITHUB_TOKEN and GITHUB_REPO to your Streamlit Cloud secrets.")
    st.stop()

# --- 2. THE DATA PIPELINE ENGINE ---
class WealthEngine:
    @staticmethod
    def clean(val):
        """Standardizes messy IBKR numbers."""
        if val is None or pd.isna(val) or str(val).strip() == '': return 0.0
        s = str(val).strip().replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
        try: return float(s)
        except: return 0.0

    @staticmethod
    def push_to_github(df, path):
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
        headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
        res = requests.get(url, headers=headers)
        sha = res.json().get('sha') if res.status_code == 200 else None
        content = base64.b64encode(df.to_csv(index=False).encode()).decode()
        payload = {"message": f"Sync {path}", "content": content, "branch": "main"}
        if sha: payload["sha"] = sha
        return requests.put(url, headers=headers, json=payload).status_code in [200, 201]

    @staticmethod
    def load_from_github(path):
        url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{path}?v={datetime.now().timestamp()}"
        try:
            df = pd.read_csv(url)
            return df if not df.empty else None
        except: return None

# --- 3. SIDEBAR: THE SYNC TRIGGER ---
FY_LIST = ["FY24", "FY25", "FY26"]
with st.sidebar:
    st.header("🔄 On-Demand Sync")
    sync_fy = st.selectbox("Year to Sync", FY_LIST)
    
    if st.button(f"🚀 Full Sync {sync_fy}"):
        with st.status(f"Harvesting {sync_fy}...") as status:
            conn = st.connection("gsheets", type=GSheetsConnection)
            raw = conn.read(worksheet=sync_fy, ttl=0)
            
            # --- EXTRACT TRADES ---
            t_rows = raw[raw.iloc[:, 0].str.contains('Trades', na=False, case=False)]
            if not t_rows.empty:
                h = t_rows[t_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
                d = t_rows[t_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(h)]
                d.columns = h
                WealthEngine.push_to_github(d, f"data/{sync_fy}/trades.csv")
                
                # --- UPDATE PRICE CACHE ---
                tickers = d[next(c for c in d.columns if 'Symbol' in c)].unique().tolist()
                prices = []
                for t in tickers:
                    try:
                        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t}"
                        p = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).json()['chart']['result'][0]['meta']['regularMarketPrice']
                        prices.append({"Symbol": t, "CurrentPrice": p})
                    except: prices.append({"Symbol": t, "CurrentPrice": 0.0})
                WealthEngine.push_to_github(pd.DataFrame(prices), "data/price_cache.csv")
            
            # --- EXTRACT PERFORMANCE ---
            p_rows = raw[raw.iloc[:, 0].str.contains('Performance Summary|Realized', na=False, case=False)]
            if not p_rows.empty:
                h = p_rows[p_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
                d = p_rows[p_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(h)]
                d.columns = h
                WealthEngine.push_to_github(d, f"data/{sync_fy}/perf.csv")
            
            st.session_state[f'sync_{sync_fy}'] = datetime.now().strftime("%H:%M")
            st.rerun()

    if f'sync_{sync_fy}' in st.session_state:
        st.caption(f"📅 **Last Synced:** {st.session_state[f'sync_{sync_fy}']}")

# --- 4. THE DASHBOARD ENGINE ---
view_fy = st.radio("Dashboard Horizon", FY_LIST, index=len(FY_LIST)-1, horizontal=True)
years_to_load = FY_LIST[:FY_LIST.index(view_fy)+1]

all_trades, all_perf = [], []
with st.expander("🛠️ System Monitor: GitHub Connectivity"):
    for y in years_to_load:
        t = WealthEngine.load_from_github(f"data/{y}/trades.csv")
        p = WealthEngine.load_from_github(f"data/{y}/perf.csv")
        st.write(f"**{y}:** Trades {'✅' if t is not None else '❌'} | Performance {'✅' if p is not None else '❌'}")
        if t is not None: t['FY'] = y; all_trades.append(t)
        if p is not None: p['FY'] = y; all_perf.append(p)
    
    price_cache = WealthEngine.load_from_github("data/price_cache.csv")
    st.write(f"**Price Cache:** {'✅' if price_cache is not None else '❌'}")

# --- 5. CALCULATIONS ---
if all_trades:
    df_raw = pd.concat(all_trades, ignore_index=True)
    df_perf = pd.concat(all_perf, ignore_index=True) if all_perf else pd.DataFrame()

    # FIFO Processing
    c_q, c_p, c_s, c_d = next(c for c in df_raw.columns if 'Qty' in c), next(c for c in df_raw.columns if 'Price' in c), next(c for c in df_raw.columns if 'Symbol' in c), next(c for c in df_raw.columns if 'Date' in c)
    c_c = next((c for c in df_raw.columns if 'Comm' in c), None)

    df_raw['Q'] = df_raw[c_q].apply(WealthEngine.clean)
    df_raw['P'] = df_raw[c_p].apply(WealthEngine.clean)
    df_raw['C'] = df_raw[c_c].apply(WealthEngine.clean).abs() if c_c else 0.0
    df_raw['DT'] = pd.to_datetime(df_raw[c_d].str.split(',').str[0], errors='coerce')

    lots = []
    for ticker in df_raw[c_s].unique():
        sym_df = df_raw[df_raw[c_s] == ticker].sort_values('DT')
        open_lots = []
        for _, row in sym_df.iterrows():
            if row['Q'] > 0: open_lots.append({'dt': row['DT'], 'q': row['Q'], 'p': row['P'], 'c': row['C']})
            elif row['Q'] < 0:
                sq = abs(row['Q'])
                while sq > 0 and open_lots:
                    if open_lots[0]['q'] <= sq: sq -= open_lots.pop(0)['q']
                    else: open_lots[0]['q'] -= sq; sq = 0
        for l in open_lots:
            l['Symbol'] = ticker
            l['Type'] = "Long-Term" if (pd.Timestamp.now() - l['dt']).days > 365 else "Short-Term"
            lots.append(l)
    
    df_h = pd.DataFrame(lots)

    # --- TOP METRICS ---
    
    
    lt_invest = (df_h['q'] * df_h['p']).sum() + df_h['c'].sum() if not df_h.empty else 0.0
    lt_comm = df_raw['C'].sum()
    
    lt_s = lt_f = 0.0
    if not df_perf.empty:
        rt_col = next((c for c in df_perf.columns if 'Realized' in c and 'Total' in c), None)
        cat_col = next((c for c in df_perf.columns if 'Category' in c or 'Asset' in c), None)
        if rt_col and cat_col:
            df_perf[rt_col] = df_perf[rt_col].apply(WealthEngine.clean)
            lt_s = df_perf[df_perf[cat_col].str.contains('Stock|Equity', na=False, case=False)][rt_col].sum()
            lt_f = df_perf[df_perf[cat_col].str.contains('Forex|Cash|Interest', na=False, case=False)][rt_col].sum()

    st.subheader("🌐 Lifetime Overview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Lifetime Investment", f"${lt_invest:,.2f}")
    k2.metric("Total Realized P/L", f"${(lt_s + lt_f):,.2f}")
    k3.metric("Stocks Realized", f"${lt_s:,.2f}")
    k4.metric("Forex/Interest", f"${lt_f:,.2f}")
    st.info(f"💰 **Lifetime Commission:** ${lt_comm:,.2f} | *Realized P/L is net of these fees.*")

    # --- HOLDINGS ---
    def render_table(data, title):
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
    render_table(df_h, "1. Current Global Holdings")
    
    # --- FIFO CALCULATOR ---
    
    st.divider()
    st.header("🧮 FIFO Selling Calculator")
    sel_ticker = st.selectbox("Stock to Simulate", df_h['Symbol'].unique())
    h_row = df_h[df_h['Symbol'] == sel_ticker]
    u_tot, a_cost = h_row['q'].sum(), h_row['p'].mean()
    
    col_a, col_b = st.columns(2)
    mode = col_a.radio("Input Mode", ["Units", "Percentage (%)"])
    qty = col_b.slider("Qty to Sell", 0.0, float(u_tot)) if mode == "Units" else u_tot * (col_b.slider("% to Sell", 0, 100, 25)/100)
    target = col_b.number_input("Target Profit %", value=110.0)
    
    st.success(f"**Target Price:** ${(a_cost * (target/100)):,.2f} | **Residual:** {u_tot - qty:.2f} units.")
else:
    st.warning("⚠️ No trades found. Check the 'System Monitor' above to see if GitHub files are missing.")
