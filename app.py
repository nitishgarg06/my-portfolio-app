import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import requests
import base64
from datetime import datetime

# --- 1. SETTINGS & SECRETS ---
st.set_page_config(layout="wide", page_title="Wealth Terminal Pro")
st.title("🏦 Wealth Terminal Pro")

def clean_num(x):
    if pd.isna(x) or str(x).strip() in ['', '--']: return 0.0
    s = str(x).strip().replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
    try: return float(s)
    except: return 0.0

# --- 2. THE DATA PIPELINE ENGINE ---
class WealthEngine:
    @staticmethod
    def push_to_github(df, path):
        url = f"https://api.github.com/repos/{st.secrets['GITHUB_REPO']}/contents/{path}"
        headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}", "Accept": "application/vnd.github.v3+json"}
        res = requests.get(url, headers=headers)
        sha = res.json().get('sha') if res.status_code == 200 else None
        content = base64.b64encode(df.to_csv(index=False).encode()).decode()
        payload = {"message": f"Update {path}", "content": content, "branch": "main"}
        if sha: payload["sha"] = sha
        return requests.put(url, headers=headers, json=payload).status_code in [200, 201]

    @staticmethod
    def load_from_github(path):
        url = f"https://raw.githubusercontent.com/{st.secrets['GITHUB_REPO']}/main/{path}?v={datetime.now().timestamp()}"
        try: return pd.read_csv(url)
        except: return None

# --- 3. SIDEBAR: THE DIAGNOSTIC SYNC ---
FY_LIST = ["FY24", "FY25", "FY26"]
with st.sidebar:
    st.header("🔄 Diagnostic Sync")
    sync_fy = st.selectbox("Year to Sync", FY_LIST)
    
    if st.button(f"🚀 Run Full Diagnostic Sync: {sync_fy}"):
        conn = st.connection("gsheets", type=GSheetsConnection)
        raw = conn.read(worksheet=sync_fy, ttl=0)
        
        st.write("### 🔎 Diagnostic Scan Results")
        
        # --- TRADES SCAN ---
        t_rows = raw[raw.iloc[:, 0].str.contains('Trades', na=False, case=False)]
        if not t_rows.empty:
            h = t_rows[t_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
            d = t_rows[t_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(h)]
            d.columns = h
            st.write(f"✅ Found {len(d)} Trade rows for {sync_fy}")
            WealthEngine.push_to_github(d, f"data/{sync_fy}/trades.csv")
            
            # Update Prices
            tickers = d[next(c for c in d.columns if 'Symbol' in c)].unique().tolist()
            prices = []
            for t in tickers:
                try:
                    p = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{t}", headers={'User-Agent': 'Mozilla/5.0'}).json()['chart']['result'][0]['meta']['regularMarketPrice']
                    prices.append({"Symbol": t, "CurrentPrice": p})
                except: prices.append({"Symbol": t, "CurrentPrice": 0.0})
            WealthEngine.push_to_github(pd.DataFrame(prices), "data/price_cache.csv")
        else:
            st.error("❌ 'Trades' section not detected! Check Column A in your Sheet.")

        # --- PERFORMANCE SCAN ---
        p_rows = raw[raw.iloc[:, 0].str.contains('Performance Summary|Realized', na=False, case=False)]
        if not p_rows.empty:
            h = p_rows[p_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
            d = p_rows[p_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(h)]
            d.columns = h
            st.write(f"✅ Found {len(d)} Performance rows for {sync_fy}")
            WealthEngine.push_to_github(d, f"data/{sync_fy}/perf.csv")
        else:
            st.error("❌ 'Performance' section not detected!")

        st.caption(f"Last Sync: {datetime.now().strftime('%H:%M:%S')}")

# --- 4. THE DASHBOARD ---
view_fy = st.radio("Display Horizon", FY_LIST, index=len(FY_LIST)-1, horizontal=True)
years = FY_LIST[:FY_LIST.index(view_fy)+1]

all_t, all_p = [], []
for y in years:
    t, p = WealthEngine.load_from_github(f"data/{y}/trades.csv"), WealthEngine.load_from_github(f"data/{y}/perf.csv")
    if t is not None: t['FY'] = y; all_t.append(t)
    if p is not None: p['FY'] = y; all_p.append(p)
price_cache = WealthEngine.load_from_github("data/price_cache.csv")

if all_t:
    df_raw = pd.concat(all_t, ignore_index=True)
    df_perf = pd.concat(all_p, ignore_index=True) if all_p else pd.DataFrame()

    # Column Mapping
    c_q = next(c for c in df_raw.columns if 'Qty' in c or 'Quantity' in c)
    c_p = next(c for c in df_raw.columns if 'Price' in c)
    c_s = next(c for c in df_raw.columns if 'Symbol' in c)
    c_d = next(c for c in df_raw.columns if 'Date' in c)
    c_c = next((c for c in df_raw.columns if 'Comm' in c), None)

    df_raw['Q'] = df_raw[c_q].apply(clean_num)
    df_raw['P'] = df_raw[c_p].apply(clean_num)
    df_raw['C'] = df_raw[c_c].apply(clean_num).abs() if c_c else 0.0
    df_raw['DT'] = pd.to_datetime(df_raw[c_d].str.split(',').str[0], errors='coerce')

    # FIFO Lot Logic
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
            l['Status'] = "Long-Term" if (pd.Timestamp.now() - l['dt']).days > 365 else "Short-Term"
            lots.append(l)
    
    df_h = pd.DataFrame(lots)

    # --- TOP METRICS ---
    inv = (df_h['q'] * df_h['p']).sum() + df_h['c'].sum() if not df_h.empty else 0.0
    comm = df_raw['C'].sum()
    lt_s = lt_f = 0.0
    if not df_perf.empty:
        rt_c = next((c for c in df_perf.columns if 'Realized' in c and 'Total' in c), None)
        cat_c = next((c for c in df_perf.columns if 'Category' in c or 'Asset' in c), None)
        if rt_c and cat_c:
            df_perf[rt_c] = df_perf[rt_c].apply(clean_num)
            lt_s = df_perf[df_perf[cat_c].str.contains('Stock|Equity', na=False, case=False)][rt_c].sum()
            lt_f = df_perf[df_perf[cat_c].str.contains('Forex|Cash|Interest', na=False, case=False)][rt_c].sum()

    st.subheader("🌐 Lifetime Overview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Lifetime Investment", f"${inv:,.2f}")
    k2.metric("Total Realized P/L", f"${(lt_s + lt_f):,.2f}")
    k3.metric("Stocks Portion", f"${lt_s:,.2f}")
    k4.metric("Forex/Interest", f"${lt_f:,.2f}")
    st.info(f"💰 **Total Commissions:** ${comm:,.2f}")

    # --- HOLDINGS TABLE ---
    st.subheader(f"Holdings (as of {datetime.now().strftime('%d %b %Y')})")
    if not df_h.empty:
        agg = df_h.groupby('Symbol').agg({'q':'sum', 'p':'mean', 'c':'sum'}).reset_index()
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

    # --- FIFO CALCULATOR ---
    
    st.divider()
    st.header("🧮 FIFO Selling Calculator")
    sel = st.selectbox("Stock", df_h['Symbol'].unique())
    u_tot, a_cost = df_h[df_h['Symbol'] == sel]['q'].sum(), df_h[df_h['Symbol'] == sel]['p'].mean()
    
    col_a, col_b = st.columns(2)
    mode = col_a.radio("Input Mode", ["Units", "Percentage (%)"])
    qty = col_b.slider("Qty to Sell", 0.0, float(u_tot)) if mode == "Units" else u_tot * (col_b.slider("% to Sell", 0, 100, 25)/100)
    target = col_b.number_input("Target Profit %", value=110.0)
    st.success(f"**Target Price:** ${(a_cost * (target/100)):,.2f} | **Residual:** {u_tot - qty:.2f} units.")

else:
    st.warning("⚠️ No data. Run 'Diagnostic Sync' in sidebar (FY24 first).")
