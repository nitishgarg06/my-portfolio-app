import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import requests
import base64
from datetime import datetime

# --- 1. DATA ENGINE (MODULAR CLASSES) ---
class DataEngine:
    @staticmethod
    def push_to_github(df, path):
        url = f"https://api.github.com/repos/{st.secrets['GITHUB_REPO']}/contents/{path}"
        headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}", "Accept": "application/vnd.github.v3+json"}
        res = requests.get(url, headers=headers)
        sha = res.json().get('sha') if res.status_code == 200 else None
        content = base64.b64encode(df.to_csv(index=False).encode()).decode()
        payload = {"message": f"Sync {path}", "content": content, "branch": "main"}
        if sha: payload["sha"] = sha
        return requests.put(url, headers=headers, json=payload).status_code in [200, 201]

    @staticmethod
    def load_from_github(path):
        url = f"https://raw.githubusercontent.com/{st.secrets['GITHUB_REPO']}/main/{path}?v={datetime.now().timestamp()}"
        try:
            df = pd.read_csv(url)
            return df if not df.empty else None
        except: return None

# --- 2. THE PRICE ENGINE ---
class PriceEngine:
    @staticmethod
    def refresh_cache(tickers):
        price_data = []
        headers = {'User-Agent': 'Mozilla/5.0'}
        for t in tickers:
            try:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t}"
                res = requests.get(url, headers=headers, timeout=5).json()
                price = res['chart']['result'][0]['meta']['regularMarketPrice']
                price_data.append({"Symbol": t, "CurrentPrice": price, "Date": datetime.now().strftime("%Y-%m-%d %H:%M")})
            except: price_data.append({"Symbol": t, "CurrentPrice": 0.0, "Date": "Error"})
        df = pd.DataFrame(price_data)
        DataEngine.push_to_github(df, "data/price_cache.csv")
        return df

# --- 3. MAIN INTERFACE ---
st.set_page_config(layout="wide", page_title="Wealth Terminal Pro")
st.title("🏦 Wealth Terminal Pro")
FY_LIST = ["FY24", "FY25", "FY26"]

# SIDEBAR: SYNC & FOOTER
with st.sidebar:
    st.header("🔄 Data Lifecycle")
    sync_fy = st.selectbox("Year to Sync", FY_LIST)
    if st.button(f"🚀 Sync {sync_fy} & Prices"):
        with st.status(f"Pushing {sync_fy} to GitHub...") as s:
            conn = st.connection("gsheets", type=GSheetsConnection)
            raw = conn.read(worksheet=sync_fy, ttl=0)
            
            # Save Trades
            t_rows = raw[raw.iloc[:, 0].str.contains('Trades', na=False)]
            th = t_rows[t_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
            td = t_rows[t_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(th)]
            td.columns = th
            DataEngine.push_to_github(td, f"data/{sync_fy}/trades.csv")
            PriceEngine.refresh_cache(td['Symbol'].unique().tolist())
            
            # Save Performance (Fixed keyword search for non-zero Realized P/L)
            p_rows = raw[raw.iloc[:, 0].str.contains('Performance Summary|Realized', na=False, case=False)]
            if not p_rows.empty:
                ph = p_rows[p_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
                pd_data = p_rows[p_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(ph)]
                pd_data.columns = ph
                DataEngine.push_to_github(pd_data, f"data/{sync_fy}/perf.csv")
            
            st.session_state[f'last_sync_{sync_fy}'] = datetime.now().strftime("%d %b %Y, %H:%M")
            st.rerun()
    
    # FOOTER UNDER SYNC
    if f'last_sync_{sync_fy}' in st.session_state:
        st.caption(f"📅 **Last Sync ({sync_fy}):** {st.session_state[f'last_sync_{sync_fy}']}")
    else:
        st.caption("📅 **Last Sync:** Never (this session)")

# DASHBOARD: CUMULATIVE ENGINE
view_fy = st.radio("Cumulative View", FY_LIST, index=len(FY_LIST)-1, horizontal=True)
load_years = FY_LIST[:FY_LIST.index(view_fy)+1]

all_trades, all_perf = [], []
for y in load_years:
    t, p = DataEngine.load_from_github(f"data/{y}/trades.csv"), DataEngine.load_from_github(f"data/{y}/perf.csv")
    if t is not None: all_trades.append(t)
    if p is not None: all_perf.append(p)

prices_cache = DataEngine.load_from_github("data/price_cache.csv")

if all_trades:
    # --- CALCULATE OPEN LOTS (FIFO) ---
    df_raw = pd.concat(all_trades)
    
    def clean(x): return pd.to_numeric(str(x).replace('$','').replace(',','').replace('(','-').replace(')',''), errors='coerce').fillna(0)
    
    # Map Columns
    c_q, c_p, c_s, c_d = next(c for c in df_raw.columns if 'Qty' in c), next(c for c in df_raw.columns if 'Price' in c), next(c for c in df_raw.columns if 'Symbol' in c), next(c for c in df_raw.columns if 'Date' in c)
    c_c = next((c for c in df_raw.columns if 'Comm' in c), None)

    df_raw['Q'] = df_raw[c_q].apply(clean)
    df_raw['P'] = df_raw[c_p].apply(clean)
    df_raw['C'] = df_raw[c_c].apply(clean).abs() if c_c else 0.0
    df_raw['DT'] = pd.to_datetime(df_raw[c_d].str.split(',').str[0])

    open_lots = []
    for ticker in df_raw[c_s].unique():
        sym_df = df_raw[df_raw[c_s] == ticker].sort_values('DT')
        lots = []
        for _, row in sym_df.iterrows():
            if row['Q'] > 0: lots.append({'dt': row['DT'], 'q': row['Q'], 'p': row['P'], 'c': row['C']})
            elif row['Q'] < 0:
                sq = abs(row['Q'])
                while sq > 0 and lots:
                    if lots[0]['q'] <= sq: sq -= lots.pop(0)['q']
                    else: lots[0]['q'] -= sq; sq = 0
        for l in lots:
            l['Symbol'] = ticker
            l['Status'] = "Long-Term" if (pd.Timestamp.now() - l['dt']).days > 365 else "Short-Term"
            open_lots.append(l)
    
    df_h = pd.DataFrame(open_lots)

    # --- TOP LINE METRICS ---
    
    
    total_inv = (df_h['q'] * df_h['p']).sum() if not df_h.empty else 0.0
    total_comm = df_raw['C'].sum()
    
    lt_s = lt_f = 0.0
    if all_perf:
        df_p = pd.concat(all_perf)
        rt_col = next((c for c in df_p.columns if 'Realized' in c and 'Total' in c), None)
        cat_col = next((c for c in df_p.columns if 'Category' in c or 'Asset' in c), None)
        if rt_col and cat_col:
            df_p[rt_col] = df_p[rt_col].apply(clean)
            lt_s = df_p[df_p[cat_col].str.contains('Stock|Equity', na=False, case=False)][rt_col].sum()
            lt_f = df_p[df_p[cat_col].str.contains('Forex|Cash|Interest', na=False, case=False)][rt_col].sum()

    st.subheader("🌐 Lifetime Overview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Lifetime Investment", f"${total_inv:,.2f}")
    k2.metric("Lifetime Realized P/L", f"${(lt_s + lt_f):,.2f}")
    k3.metric("Stocks Realized", f"${lt_s:,.2f}")
    k4.metric("Forex/Impact", f"${lt_f:,.2f}")
    
    st.info(f"💰 **Lifetime Commission Paid:** ${total_comm:,.2f} | *Realized P/L is net of commissions.*")

    # --- TABLES ---
    def render_table(data, title):
        st.subheader(f"{title} (as of {datetime.now().strftime('%d %b %Y')})")
        if data.empty: return st.info("No active holdings.")
        
        agg = data.groupby('Symbol').agg({'q': 'sum', 'p': 'mean', 'c': 'sum'}).reset_index()
        if prices_cache is not None:
            agg = agg.merge(prices_cache[['Symbol', 'CurrentPrice']], on='Symbol', how='left').fillna(0)
        else: agg['CurrentPrice'] = 0.0
        
        agg['Total Basis'] = (agg['q'] * agg['p']) + agg['c']
        agg['Market Value'] = agg['q'] * agg['CurrentPrice']
        agg['P/L $'] = agg['Market Value'] - agg['Total Basis']
        agg['P/L %'] = (agg['P/L $'] / agg['Total Basis'] * 100) if not agg.empty else 0.0
        
        agg.columns = ['Ticker', 'Units', 'Avg Cost', 'Comms', 'Current Price', 'Total Basis', 'Market Value', 'P/L $', 'P/L %']
        agg.index = range(1, len(agg) + 1) # Fix S.No starting from 1
        st.dataframe(agg.style.format({"Units": "{:.2f}", "Avg Cost": "${:.2f}", "Comms": "${:.2f}", "Current Price": "${:.2f}", "Total Basis": "${:.2f}", "Market Value": "${:.2f}", "P/L $": "${:.2f}", "P/L %": "{:.2f}%"}), use_container_width=True)

    st.divider()
    render_table(df_h, "1. Current Global Holdings")
    c_a, c_b = st.columns(2)
    with c_a: render_table(df_h[df_h['Status'] == "Short-Term"], "2. Short-Term Holdings")
    with c_b: render_table(df_h[df_h['Status'] == "Long-Term"], "3. Long-Term Holdings")

    # --- FIFO CALCULATOR ---
    
    st.divider()
    st.header("🧮 FIFO Selling Calculator")
    sel_ticker = st.selectbox("Stock to Simulate", df_h['Symbol'].unique())
    h_row = df_h[df_h['Symbol'] == sel_ticker]
    u_total, c_avg = h_row['q'].sum(), h_row['p'].mean()
    
    calc_a, calc_b = st.columns(2)
    calc_mode = calc_a.radio("Input Mode", ["Units", "Percentage (%)"])
    
    if calc_mode == "Units":
        s_qty = calc_b.slider("Quantity to Sell", 0.0, float(u_total), float(u_total*0.25))
    else:
        s_pct = calc_b.slider("Percentage (%) to Sell", 0, 100, 25)
        s_qty = u_total * (s_pct/100)
    
    target_pct = calc_b.number_input("Target Profit %", value=110.0)
    target_price = c_avg * (target_pct/100)
    
    st.success(f"**Target Exit Price:** ${target_price:,.2f} for {target_pct}% Profit.")
    st.info(f"📉 **Residual:** {u_total - s_qty:.2f} units of {sel_ticker} remaining at ${c_avg:,.2f} cost.")

else:
    st.warning("⚠️ No trade data found on GitHub. Please sync your years in the sidebar.")
