import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import requests
import base64
from datetime import datetime

# --- 1. CORE CONFIG ---
st.set_page_config(layout="wide", page_title="Wealth Terminal Pro")
st.title("🏦 Wealth Terminal Pro")

# --- 2. THE PERMANENT DATA ENGINE ---
class MasterDataEngine:
    @staticmethod
    def clean(val):
        if pd.isna(val) or str(val).strip() in ['', '--']: return 0.0
        s = str(val).strip().replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
        try: return float(s)
        except: return 0.0

    @staticmethod
    def push_to_master(df, path):
        """Final Gate: Only pushes to GitHub if data is valid (non-empty)."""
        if df.empty:
            return False
        url = f"https://api.github.com/repos/{st.secrets['GITHUB_REPO']}/contents/{path}"
        headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}", "Accept": "application/vnd.github.v3+json"}
        res = requests.get(url, headers=headers)
        sha = res.json().get('sha') if res.status_code == 200 else None
        content = base64.b64encode(df.to_csv(index=False).encode()).decode()
        payload = {"message": f"Master Sync {path}", "content": content, "branch": "main"}
        if sha: payload["sha"] = sha
        return requests.put(url, headers=headers, json=payload).status_code in [200, 201]

    @staticmethod
    def load_master(path):
        url = f"https://raw.githubusercontent.com/{st.secrets['GITHUB_REPO']}/main/{path}?v={datetime.now().timestamp()}"
        try: return pd.read_csv(url)
        except: return None

# --- 3. SIDEBAR: THE MASTER GATEKEEPER ---
FY_LIST = ["FY24", "FY25", "FY26"]
with st.sidebar:
    st.header("🔄 Master Data Sync")
    sync_fy = st.selectbox("Select Year to Finalize", FY_LIST)
    
    if st.button(f"🚀 Standardize & Save {sync_fy}"):
        conn = st.connection("gsheets", type=GSheetsConnection)
        raw = conn.read(worksheet=sync_fy, ttl=0)
        
        # 1. Standardize Trades
        t_rows = raw[raw.iloc[:, 0].str.contains('Trades', na=False, case=False)]
        if not t_rows.empty:
            h = t_rows[t_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
            d = t_rows[t_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(h)]
            d.columns = h
            if MasterDataEngine.push_to_master(d, f"data/{sync_fy}/trades.csv"):
                st.success(f"Trades for {sync_fy} standardized on GitHub.")
        
        # 2. Standardize Performance (Lifetime P/L Source)
        p_rows = raw[raw.iloc[:, 0].str.contains('Performance Summary|Realized', na=False, case=False)]
        if not p_rows.empty:
            h = p_rows[p_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
            d = p_rows[p_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(h)]
            d.columns = h
            if MasterDataEngine.push_to_master(d, f"data/{sync_fy}/perf.csv"):
                st.success(f"Performance for {sync_fy} standardized on GitHub.")
        
        # 3. Update Current Prices
        tickers = d[next(c for c in d.columns if 'Symbol' in c)].unique().tolist() if 'd' in locals() else []
        prices = []
        for t in tickers:
            try:
                p = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{t}", headers={'User-Agent': 'Mozilla/5.0'}).json()['chart']['result'][0]['meta']['regularMarketPrice']
                prices.append({"Symbol": t, "CurrentPrice": p})
            except: prices.append({"Symbol": t, "CurrentPrice": 0.0})
        MasterDataEngine.push_to_master(pd.DataFrame(prices), "data/price_cache.csv")
        
        st.caption(f"Last Master Sync: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# --- 4. THE CUMULATIVE DASHBOARD (Uses GitHub 'Source of Truth') ---
view_fy = st.radio("Dashboard View", FY_LIST, index=len(FY_LIST)-1, horizontal=True)
years = FY_LIST[:FY_LIST.index(view_fy)+1]

all_trades = []
all_perf = []
for y in years:
    t = MasterDataEngine.load_master(f"data/{y}/trades.csv")
    p = MasterDataEngine.load_master(f"data/{y}/perf.csv")
    if t is not None: all_trades.append(t)
    if p is not None: all_perf.append(p)
price_cache = MasterDataEngine.load_master("data/price_cache.csv")

if all_trades:
    df_raw = pd.concat(all_trades, ignore_index=True)
    df_perf = pd.concat(all_perf, ignore_index=True) if all_perf else pd.DataFrame()

    # FIFO Logic (Fixed Weighted Calculation)
    c_q = next(c for c in df_raw.columns if 'Qty' in c or 'Quantity' in c)
    c_p = next(c for c in df_raw.columns if 'Price' in c)
    c_s = next(c for c in df_raw.columns if 'Symbol' in c)
    c_d = next(c for c in df_raw.columns if 'Date' in c)
    c_c = next((c for c in df_raw.columns if 'Comm' in c), None)

    df_raw['Q_clean'] = df_raw[c_q].apply(MasterDataEngine.clean)
    df_raw['P_clean'] = df_raw[c_p].apply(MasterDataEngine.clean)
    df_raw['C_clean'] = df_raw[c_c].apply(MasterDataEngine.clean).abs() if c_c else 0.0
    df_raw['DT_clean'] = pd.to_datetime(df_raw[c_d].str.split(',').str[0], errors='coerce')

    lots = []
    for ticker in df_raw[c_s].unique():
        sym_df = df_raw[df_raw[c_s] == ticker].sort_values('DT_clean')
        open_lots = []
        for _, row in sym_df.iterrows():
            if row['Q_clean'] > 0:
                open_lots.append({'dt': row['DT_clean'], 'q': row['Q_clean'], 'p': row['P_clean'], 'c': row['C_clean']})
            elif row['Q_clean'] < 0:
                sq = abs(row['Q_clean'])
                while sq > 0 and open_lots:
                    if open_lots[0]['q'] <= sq: sq -= open_lots.pop(0)['q']
                    else: open_lots[0]['q'] -= sq; sq = 0
        for l in open_lots:
            l['Symbol'] = ticker
            l['Status'] = "Long-Term" if (pd.Timestamp.now() - l['dt']).days > 365 else "Short-Term"
            lots.append(l)
    
    df_h = pd.DataFrame(lots)

    # --- TOP LINE METRICS (LIFETIME) ---
    
    
    lt_invest = (df_h['q'] * df_h['p']).sum() + df_h['c'].sum() if not df_h.empty else 0.0
    lt_comm = df_raw['C_clean'].sum()
    
    lt_s = lt_f = 0.0
    if not df_perf.empty:
        rt_col = next((c for c in df_perf.columns if 'Realized' in c and 'Total' in c), None)
        cat_col = next((c for c in df_perf.columns if 'Category' in c or 'Asset' in c), None)
        if rt_col and cat_col:
            df_perf[rt_col] = df_perf[rt_col].apply(MasterDataEngine.clean)
            lt_s = df_perf[df_perf[cat_col].str.contains('Stock|Equity', na=False, case=False)][rt_col].sum()
            lt_f = df_perf[df_perf[cat_col].str.contains('Forex|Cash|Interest', na=False, case=False)][rt_col].sum()

    st.subheader("🌐 Lifetime Overview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Lifetime Investment", f"${lt_invest:,.2f}")
    k2.metric("Total Realized P/L", f"${(lt_s + lt_f):,.2f}")
    k3.metric("Stocks Realized", f"${lt_s:,.2f}")
    k4.metric("Forex/Interest", f"${lt_f:,.2f}")
    st.info(f"💰 **Lifetime Commission:** ${lt_comm:,.2f}")

    # --- HOLDINGS TABLE ---
    st.subheader(f"Current Holdings (as of {datetime.now().strftime('%d %b %Y')})")
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
        st.dataframe(agg.style.format({"Units": "{:.2f}", "Avg Cost": "${:.2f}", "Comms": "${:.2f}", "Current Price": "${:.2f}", "Total Basis": "${:.2f}", "Market Value": "${:.2f}", "P/L $": "${:.2f}", "P/L %": "{:.2f}%"}), use_container_width=True)

    # --- FIFO CALCULATOR ---
    
    st.divider()
    st.header("🧮 FIFO Selling Calculator")
    sel_ticker = st.selectbox("Stock to Simulate", df_h['Symbol'].unique())
    h_row = df_h[df_h['Symbol'] == sel_ticker]
    u_tot, a_cost = h_row['q'].sum(), h_row['p'].mean()
    
    col_a, col_b = st.columns(2)
    mode = col_a.radio("Input Mode", ["Units", "Percentage (%)"])
    qty_to_sell = col_b.slider("Quantity", 0.0, float(u_tot)) if mode == "Units" else u_tot * (col_b.slider("%", 0, 100, 25)/100)
    target_pct = col_b.number_input("Target Profit %", value=110.0)
    
    st.success(f"**Target Exit Price:** ${(a_cost * (target_pct/100)):,.2f} | **Residual:** {u_tot - qty_to_sell:.2f} units.")
else:
    st.warning("⚠️ No standardized data found on GitHub. Please select a year in the sidebar and click 'Standardize & Save' to build your master copy.")
