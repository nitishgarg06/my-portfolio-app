import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import requests
import base64
from datetime import datetime

# --- 1. DATA ENGINE: GITHUB & STORAGE ---
class DataEngine:
    @staticmethod
    def push_to_github(df, path):
        """Pushes a dataframe as a CSV to your private GitHub repository."""
        url = f"https://api.github.com/repos/{st.secrets['GITHUB_REPO']}/contents/{path}"
        headers = {
            "Authorization": f"token {st.secrets['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github.v3+json"
        }
        res = requests.get(url, headers=headers)
        sha = res.json().get('sha') if res.status_code == 200 else None
        
        content = base64.b64encode(df.to_csv(index=False).encode()).decode()
        payload = {
            "message": f"Data Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "content": content,
            "branch": "main"
        }
        if sha: payload["sha"] = sha
        return requests.put(url, headers=headers, json=payload).status_code in [200, 201]

    @staticmethod
    def load_from_github(path):
        """Loads a CSV from GitHub with a cache-buster for the latest data."""
        url = f"https://raw.githubusercontent.com/{st.secrets['GITHUB_REPO']}/main/{path}?v={datetime.now().timestamp()}"
        try:
            df = pd.read_csv(url)
            return df if not df.empty else None
        except:
            return None

# --- 2. PRICE ENGINE: STATIC CACHE ---
class PriceEngine:
    @staticmethod
    def refresh_cache(tickers):
        """Fetches current prices and saves them to a static CSV on GitHub."""
        price_data = []
        headers = {'User-Agent': 'Mozilla/5.0'}
        for t in tickers:
            try:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t}"
                res = requests.get(url, headers=headers, timeout=5).json()
                price = res['chart']['result'][0]['meta']['regularMarketPrice']
                price_data.append({"Symbol": t, "StaticPrice": price, "Date": datetime.now().strftime("%Y-%m-%d %H:%M")})
            except:
                price_data.append({"Symbol": t, "StaticPrice": 0.0, "Date": "Error"})
        
        df = pd.DataFrame(price_data)
        DataEngine.push_to_github(df, "data/price_cache.csv")
        return df

# --- 3. ANALYTICS ENGINE: FIFO & METRICS ---
class AnalyticsEngine:
    @staticmethod
    def run_fifo(df_trades):
        """Processes cumulative trades to find open lots and ST/LT status."""
        # Weighted data cleaning
        def clean(x): return pd.to_numeric(str(x).replace('$','').replace(',','').replace('(','-').replace(')',''), errors='coerce')
        
        # Fuzzy map columns
        c_q = next(c for c in df_trades.columns if 'Qty' in c or 'Quantity' in c)
        c_p = next(c for c in df_trades.columns if 'Price' in c)
        c_s = next(c for c in df_trades.columns if 'Symbol' in c or 'Ticker' in c)
        c_d = next(c for c in df_trades.columns if 'Date' in c)
        c_c = next((c for c in df_trades.columns if 'Comm' in c), None)

        df_trades['Q'] = df_trades[c_q].apply(clean).fillna(0)
        df_trades['P'] = df_trades[c_p].apply(clean).fillna(0)
        df_trades['C'] = df_trades[c_c].apply(clean).abs().fillna(0) if c_c else 0.0
        df_trades['DT'] = pd.to_datetime(df_trades[c_d].str.split(',').str[0])

        open_lots = []
        for ticker in df_trades[c_s].unique():
            sym_df = df_trades[df_trades[c_s] == ticker].sort_values('DT')
            lots = []
            for _, row in sym_df.iterrows():
                if row['Q'] > 0: # Buy
                    lots.append({'dt': row['DT'], 'q': row['Q'], 'p': row['P'], 'c': row['C']})
                elif row['Q'] < 0: # Sell
                    sell_q = abs(row['Q'])
                    while sell_q > 0 and lots:
                        if lots[0]['q'] <= sell_q: sell_q -= lots.pop(0)['q']
                        else: lots[0]['q'] -= sell_q; sell_q = 0
            for l in lots:
                l['Symbol'] = ticker
                l['Status'] = "Long-Term" if (pd.Timestamp.now() - l['dt']).days > 365 else "Short-Term"
                open_lots.append(l)
        return pd.DataFrame(open_lots)

# --- 4. MAIN INTERFACE & DASHBOARD ---
st.set_page_config(layout="wide", page_title="Wealth Terminal Pro")
st.title("🏦 Wealth Terminal Pro")

# SIDEBAR: DATA SYNC
with st.sidebar:
    st.header("🔄 Multi-File Sync")
    sync_fy = st.selectbox("Sync Year", ["FY24", "FY25", "FY26"])
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
            
            # Update Prices
            PriceEngine.refresh_cache(td['Symbol'].unique().tolist())
            
            # Save Performance Summary
            p_rows = raw[raw.iloc[:, 0].str.contains('Performance Summary|Realized', na=False)]
            if not p_rows.empty:
                ph = p_rows[p_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
                pd_data = p_rows[p_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(ph)]
                pd_data.columns = ph
                DataEngine.push_to_github(pd_data, f"data/{sync_fy}/perf.csv")
            
            st.rerun()

# DASHBOARD: CUMULATIVE VIEW
view_fy = st.radio("Display Horizon", ["FY24", "FY25", "FY26"], index=2, horizontal=True)
load_years = ["FY24", "FY25", "FY26"][:["FY24", "FY25", "FY26"].index(view_fy)+1]

all_trades, all_perf = [], []
for y in load_years:
    t, p = DataEngine.load_from_github(f"data/{y}/trades.csv"), DataEngine.load_from_github(f"data/{y}/perf.csv")
    if t is not None: all_trades.append(t)
    if p is not None: all_perf.append(p)

prices_cache = DataEngine.load_from_github("data/price_cache.csv")

if all_trades:
    # 1. Process FIFO & Lifetime Metrics
    df_lots = AnalyticsEngine.run_fifo(pd.concat(all_trades))
    df_p_all = pd.concat(all_perf) if all_perf else pd.DataFrame()

    st.subheader("🌐 Lifetime Overview")
    
    # Calculate Realized
    lt_s = lt_f = 0.0
    if not df_p_all.empty:
        rt_col = next((c for c in df_p_all.columns if 'Realized' in c and 'Total' in c), None)
        cat_col = next((c for c in df_p_all.columns if 'Category' in c or 'Asset' in c), None)
        if rt_col and cat_col:
            df_p_all[rt_col] = pd.to_numeric(df_p_all[rt_col].astype(str).str.replace('$','').str.replace(',',''), errors='coerce').fillna(0)
            lt_s = df_p_all[df_p_all[cat_col].str.contains('Stock|Equity', na=False, case=False)][rt_col].sum()
            lt_f = df_p_all[df_p_all[cat_col].str.contains('Forex|Cash|Interest', na=False, case=False)][rt_col].sum()

    total_inv = (df_lots['q'] * df_lots['p']).sum() + df_lots['c'].sum() if not df_lots.empty else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Lifetime Investment", f"${total_inv:,.2f}")
    k2.metric("Total Realized P/L", f"${(lt_s + lt_f):,.2f}")
    k3.metric("Stocks Portion", f"${lt_s:,.2f}")
    k4.metric("Forex/Impact", f"${lt_f:,.2f}")
    st.caption("ℹ️ *Realized P/L is net of commissions.*")

    # 2. Render Holdings Tables
    
    
    def display_holdings(data, title):
        st.subheader(f"{title} (as of {datetime.now().strftime('%d %b %Y')})")
        if data.empty: return st.info("No holdings in this category.")
        
        agg = data.groupby('Symbol').agg({'q':'sum', 'p':'mean', 'c':'sum'}).reset_index()
        if prices_cache is not None:
            agg = agg.merge(prices_cache[['Symbol', 'StaticPrice']], on='Symbol', how='left').fillna(0)
        else: agg['StaticPrice'] = 0.0
        
        agg['Total Basis'] = (agg['q'] * agg['p']) + agg['c']
        agg['Market Value'] = agg['q'] * agg['StaticPrice']
        agg['P/L $'] = agg['Market Value'] - agg['Total Basis']
        agg['P/L %'] = (agg['P/L $'] / agg['Total Basis'] * 100) if not agg.empty else 0.0
        
        agg.columns = ['Ticker', 'Units', 'Avg Cost', 'Comms', 'Static Price', 'Total Basis', 'Market Value', 'P/L $', 'P/L %']
        agg.index = range(1, len(agg) + 1)
        st.dataframe(agg.style.format({
            "Units":"{:.2f}", "Avg Cost":"${:.2f}", "Comms":"${:.2f}", "Static Price":"${:.2f}",
            "Total Basis":"${:.2f}", "Market Value":"${:.2f}", "P/L $":"${:.2f}", "P/L %":"{:.2f}%"
        }), use_container_width=True)

    st.divider()
    display_holdings(df_lots, "1. Current Global Holdings")
    c1, c2 = st.columns(2)
    with c1: display_holdings(df_lots[df_lots['Status']=="Short-Term"], "2. Short-Term Holdings")
    with c2: display_holdings(df_lots[df_lots['Status']=="Long-Term"], "3. Long-Term Holdings")

    # 3. FIFO Calculator
    
    st.divider()
    st.header("🧮 FIFO Selling Calculator")
    sel_stock = st.selectbox("Select Ticker", df_lots['Symbol'].unique())
    u_total = df_lots[df_lots['Symbol']==sel_stock]['q'].sum()
    c_avg = df_lots[df_lots['Symbol']==sel_stock]['p'].mean()
    
    col_a, col_b = st.columns(2)
    sell_amt = col_a.slider("Quantity to Sell", 0.0, float(u_total), float(u_total*0.25))
    target_pct = col_b.number_input("Target Profit %", value=110.0)
    
    exit_p = c_avg * (target_pct/100)
    st.success(f"To achieve {target_pct}% Profit, sell at **${exit_p:,.2f}**")
    st.info(f"**Residual Holding:** {u_total - sell_amt:.2f} units of {sel_stock} remaining at ${c_avg:,.2f} cost basis.")

else:
    st.warning("⚠️ No data detected on GitHub. Use the sidebar to Sync your Financial Years.")
