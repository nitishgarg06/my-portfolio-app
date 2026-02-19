import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import requests
from bs4 import BeautifulSoup
import base64
from datetime import datetime

# --- 1. CONFIG & SECRETS ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    GITHUB_REPO = st.secrets["GITHUB_REPO"]
except Exception:
    st.error("Secrets Error: Check GITHUB_TOKEN and GITHUB_REPO in Streamlit Secrets.")
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

def get_ibkr_section(df, section_name):
    rows = df[df.iloc[:, 0].str.contains(section_name, na=False, case=False)]
    h_row = rows[rows.iloc[:, 1] == 'Header']
    d_rows = rows[rows.iloc[:, 1] == 'Data']
    if not h_row.empty and not d_rows.empty:
        cols = [c for c in h_row.iloc[0, 2:].tolist() if c]
        data = d_rows.iloc[:, 2:2+len(cols)]
        data.columns = cols
        return data
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_live_price(ticker):
    try:
        url = f"https://www.google.com/search?q={ticker}+stock+price"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        price_tag = soup.find('span', {'class': 'I67upf'}) or soup.find('span', {'jsname': 'vW79of'})
        return clean_numeric(price_tag.text) if price_tag else 0.0
    except: return 0.0

# --- 3. GITHUB ENGINE ---
def push_to_github(df, fy, filename):
    path = f"data/{fy}/{filename}.csv"
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    res = requests.get(url, headers=headers)
    sha = res.json().get('sha') if res.status_code == 200 else None
    content = base64.b64encode(df.to_csv(index=False).encode()).decode()
    payload = {"message": f"Sync {fy}: {datetime.now()}", "content": content, "branch": "main"}
    if sha: payload["sha"] = sha
    return requests.put(url, headers=headers, json=payload).status_code in [200, 201]

def load_from_github(fy, filename):
    url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/data/{fy}/{filename}.csv?v={datetime.now().timestamp()}"
    try: return pd.read_csv(url)
    except: return None

# --- 4. SIDEBAR: ON-DEMAND SYNC ---
st.title("🏦 Wealth Terminal Pro")
curr_date = datetime.now().strftime('%d %b %Y')
FY_LIST = ["FY24", "FY25", "FY26"]

with st.sidebar:
    st.header("🔄 On-Demand Sync")
    sync_fy = st.selectbox("Sync specific year", FY_LIST)
    if st.button(f"🚀 Sync {sync_fy}"):
        with st.status(f"Processing {sync_fy}...", expanded=True) as status:
            conn = st.connection("gsheets", type=GSheetsConnection)
            raw = conn.read(worksheet=sync_fy, ttl=0)
            t_df = get_ibkr_section(raw, 'Trades')
            p_df = get_ibkr_section(raw, 'Realized & Unrealized Performance Summary')
            if not t_df.empty: push_to_github(t_df, sync_fy, "trades")
            if not p_df.empty: push_to_github(p_df, sync_fy, "perf")
            st.session_state[f'last_sync_{sync_fy}'] = datetime.now().strftime("%H:%M:%S")
            st.rerun()

# --- 5. CUMULATIVE LOAD ENGINE ---
st.divider()
view_fy = st.radio("Select View Horizon (Cumulative)", FY_LIST, index=len(FY_LIST)-1, horizontal=True)

# Determine which years to load based on selection (e.g., if FY26, load 24, 25, 26)
years_to_load = FY_LIST[:FY_LIST.index(view_fy)+1]

all_trades = []
all_perf = []

for y in years_to_load:
    t = load_from_github(y, "trades")
    p = load_from_github(y, "perf")
    if t is not None: 
        t['Source_FY'] = y
        all_trades.append(t)
    if p is not None: 
        p['Source_FY'] = y
        all_perf.append(p)

if all_trades:
    df_trades = pd.concat(all_trades)
    df_perf = pd.concat(all_perf) if all_perf else pd.DataFrame()

    # Harmonize Data
    c_qty = fuzzy_find(df_trades, ['Qty'])
    c_prc = fuzzy_find(df_trades, ['Price'])
    c_dt = fuzzy_find(df_trades, ['Date'])
    c_sym = fuzzy_find(df_trades, ['Symbol'])
    c_cm = fuzzy_find(df_trades, ['Comm'])

    df_trades['Qty_v'] = df_trades[c_qty].apply(clean_numeric)
    df_trades['Prc_v'] = df_trades[c_prc].apply(clean_numeric)
    df_trades['CM_v'] = df_trades[c_cm].apply(clean_numeric).abs() if c_cm else 0.0
    df_trades['DT_v'] = pd.to_datetime(df_trades[c_dt].str.split(',').str[0])

    # FIFO Engine (Cumulative)
    open_lots = []
    for sym in df_trades[c_sym].unique():
        sym_df = df_trades[df_trades[c_sym] == sym].sort_values('DT_v')
        lots = []
        for _, row in sym_df.iterrows():
            if row['Qty_v'] > 0: lots.append({'dt': row['DT_v'], 'q': row['Qty_v'], 'p': row['Prc_v'], 'c': row['CM_v']})
            elif row['Qty_v'] < 0:
                sq = abs(row['Qty_v'])
                while sq > 0 and lots:
                    if lots[0]['q'] <= sq: sq -= lots.pop(0)['q']
                    else: lots[0]['q'] -= sq; sq = 0
        for l in lots:
            l['Symbol'] = sym
            l['Type'] = "Long-Term" if (pd.Timestamp.now() - l['dt']).days > 365 else "Short-Term"
            open_lots.append(l)
    
    df_h = pd.DataFrame(open_lots)

    # --- TOP LINE KPIs (Lifetime & FY) ---
    
    
    # Lifetime Realized
    lt_stocks = lt_forex = 0.0
    if not df_perf.empty:
        cat_c = fuzzy_find(df_perf, ['Category'])
        rt_c = fuzzy_find(df_perf, ['Realized Total'])
        lt_stocks = df_perf[df_perf[cat_c].str.contains('Stock', na=False, case=False)][rt_c].apply(clean_numeric).sum()
        lt_forex = df_perf[df_perf[cat_c].str.contains('Forex|Cash', na=False, case=False)][rt_c].apply(clean_numeric).sum()

    # Lifetime Investment
    lt_invest = (df_h['q'] * df_h['p']).sum() + df_h['c'].sum() if not df_h.empty else 0.0

    st.subheader("🌐 Lifetime Overview")
    k1, k2, k3 = st.columns(3)
    k1.metric("Lifetime Investment", f"${lt_invest:,.2f}")
    k2.metric("Lifetime Realized P/L", f"${(lt_stocks + lt_forex):,.2f}")
    k3.metric("Lifetime Forex Impact", f"${lt_forex:,.2f}")
    
    st.subheader(f"📅 {view_fy} Performance")
    # FY Specific Realized
    fy_perf = df_perf[df_perf['Source_FY'] == view_fy]
    fy_s = fy_perf[fy_perf[cat_c].str.contains('Stock', na=False, case=False)][rt_c].apply(clean_numeric).sum()
    fy_f = fy_perf[fy_perf[cat_c].str.contains('Forex|Cash', na=False, case=False)][rt_c].apply(clean_numeric).sum()
    
    m1, m2 = st.columns(2)
    m1.metric(f"{view_fy} Realized P/L", f"${(fy_s + fy_f):,.2f}")
    m2.metric(f"{view_fy} Stocks", f"${fy_s:,.2f}")
    st.caption("ℹ️ *Disclaimer: Realized P/L is net of commissions.*")

    # --- TABLES ---
    def render_table(data, title):
        st.subheader(f"{title} (as of {curr_date})")
        if data.empty: return st.info("No holdings.")
        agg = data.groupby('Symbol').agg({'q': 'sum', 'p': 'mean', 'c': 'sum'}).reset_index()
        agg['Live Price'] = agg['Symbol'].apply(get_live_price)
        agg['Total Basis'] = (agg['q'] * agg['p']) + agg['c']
        agg['Market Value'] = agg['q'] * agg['Live Price']
        agg['P/L $'] = agg['Market Value'] - agg['Total Basis']
        agg['P/L %'] = (agg['P/L $'] / agg['Total Basis']) * 100
        st.dataframe(agg.style.format({"q": "{:.2f}", "p": "${:.2f}", "Live Price": "${:.2f}", "Total Basis": "${:.2f}", "Market Value": "${:.2f}", "P/L $": "${:.2f}", "P/L %": "{:.2f}%"}), use_container_width=True)

    st.divider()
    render_table(df_h, "1. Current Global Holdings")
    c_st, c_lt = st.columns(2)
    with c_st: render_table(df_h[df_h['Type'] == "Short-Term"], "2. Short-Term Holdings")
    with c_lt: render_table(df_h[df_h['Type'] == "Long-Term"], "3. Long-Term Holdings")

    # --- CALCULATOR ---
    
    st.divider()
    st.header("🧮 FIFO Selling Calculator")
    c_pick, c_calc = st.columns([1, 2])
    ticker = c_pick.selectbox("Select Stock", df_h['Symbol'].unique())
    h_row = df_h[df_h['Symbol'] == ticker]
    t_q, a_c = h_row['q'].sum(), h_row['p'].mean()
    
    mode = c_pick.radio("Input Mode", ["Units", "Percentage"])
    s_q = c_calc.slider("Quantity", 0.0, float(t_q), float(t_q*0.25)) if mode == "Units" else t_q * (c_calc.slider("%", 0, 100, 25)/100)
    target = c_calc.number_input("Target Profit %", value=110.0)
    st.success(f"Sell at **${(a_c * (target/100)):,.2f}** | Residual: {t_q - s_q:.2f} units")

else:
    st.info("No data found on GitHub. Please sync your years in the sidebar.")
