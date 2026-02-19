import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import requests
from bs4 import BeautifulSoup
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
    """Returns the first column name containing any of the keywords."""
    for col in df.columns:
        if any(k.lower() in str(col).lower() for k in keywords): return col
    return None

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

# --- 3. DATA PERSISTENCE ---
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
    try:
        df = pd.read_csv(url)
        return df if not df.empty else None
    except: return None

# --- 4. SIDEBAR SYNC ---
st.title("🏦 Wealth Terminal Pro")
FY_LIST = ["FY24", "FY25", "FY26"]

with st.sidebar:
    st.header("🔄 Selective Sync")
    sync_fy = st.selectbox("Year to Sync", FY_LIST)
    if st.button(f"Sync {sync_fy}"):
        with st.status(f"Processing {sync_fy}...") as status:
            conn = st.connection("gsheets", type=GSheetsConnection)
            raw = conn.read(worksheet=sync_fy, ttl=0)
            
            # Extract Trade section using raw slicing to avoid column index errors
            rows = raw[raw.iloc[:, 0].str.contains('Trades', na=False, case=False)]
            if not rows.empty:
                h = rows[rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
                d = rows[rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(h)]
                d.columns = h
                push_to_github(d, sync_fy, "trades")
            
            # Performance section
            p_rows = raw[raw.iloc[:, 0].str.contains('Performance Summary', na=False, case=False)]
            if not p_rows.empty:
                h = p_rows[p_rows.iloc[:, 1] == 'Header'].iloc[0, 2:].dropna().tolist()
                d = p_rows[p_rows.iloc[:, 1] == 'Data'].iloc[:, 2:2+len(h)]
                d.columns = h
                push_to_github(d, sync_fy, "perf")
            
            st.session_state[f'last_sync_{sync_fy}'] = datetime.now().strftime("%H:%M:%S")
            st.rerun()

# --- 5. THE DASHBOARD ---
st.divider()
view_fy = st.radio("Cumulative View", FY_LIST, index=len(FY_LIST)-1, horizontal=True)
years_active = FY_LIST[:FY_LIST.index(view_fy)+1]

all_t, all_p = [], []
for y in years_active:
    t = load_from_github(y, "trades")
    p = load_from_github(y, "perf")
    if t is not None: t['FY_Src'] = y; all_t.append(t)
    if p is not None: p['FY_Src'] = y; all_p.append(p)

if all_t:
    trades_df = pd.concat(all_t)
    perf_df = pd.concat(all_p) if all_p else pd.DataFrame()

    # Column Mapping
    c_q = fuzzy_find(trades_df, ['Qty', 'Quantity'])
    c_p = fuzzy_find(trades_df, ['Price', 'T. Price'])
    c_s = fuzzy_find(trades_df, ['Symbol', 'Ticker'])
    c_d = fuzzy_find(trades_df, ['Date'])
    c_c = fuzzy_find(trades_df, ['Comm'])

    trades_df['Q_v'] = trades_df[c_q].apply(clean_numeric)
    trades_df['P_v'] = trades_df[c_p].apply(clean_numeric)
    trades_df['C_v'] = trades_df[c_c].apply(clean_numeric).abs() if c_c else 0.0
    trades_df['DT'] = pd.to_datetime(trades_df[c_d].str.split(',').str[0], errors='coerce')

    # FIFO Engine
    holdings = []
    for ticker in trades_df[c_s].unique():
        sym_df = trades_df[trades_df[c_s] == ticker].sort_values('DT')
        lots = []
        for _, row in sym_df.iterrows():
            if row['Q_v'] > 0: lots.append({'dt': row['DT'], 'q': row['Q_v'], 'p': row['P_v'], 'c': row['C_v']})
            elif row['Q_v'] < 0:
                sq = abs(row['Q_v'])
                while sq > 0 and lots:
                    if lots[0]['q'] <= sq: sq -= lots.pop(0)['q']
                    else: lots[0]['q'] -= sq; sq = 0
        for l in lots:
            l['Symbol'] = ticker
            l['Type'] = "Long-Term" if (pd.Timestamp.now() - l['dt']).days > 365 else "Short-Term"
            holdings.append(l)
    
    df_h = pd.DataFrame(holdings)

    # Lifetime Metrics
    lt_invest = (df_h['q'] * df_h['p']).sum() + df_h['c'].sum() if not df_h.empty else 0.0
    
    # Realized Logic
    lt_s = lt_f = 0.0
    if not perf_df.empty:
        cat_c = fuzzy_find(perf_df, ['Category'])
        rt_c = fuzzy_find(perf_df, ['Realized Total', 'Realized P/L'])
        lt_s = perf_df[perf_df[cat_c].str.contains('Stock', na=False, case=False)][rt_c].apply(clean_numeric).sum()
        lt_f = perf_df[perf_df[cat_c].str.contains('Forex|Cash', na=False, case=False)][rt_c].apply(clean_numeric).sum()

    st.subheader("🌐 Lifetime Overview")
    k1, k2, k3 = st.columns(3)
    k1.metric("Lifetime Investment", f"${lt_invest:,.2f}")
    k2.metric("Lifetime Realized P/L", f"${(lt_s + lt_f):,.2f}")
    k3.metric("Lifetime Forex Impact", f"${lt_f:,.2f}")

    # Tables
    
    st.divider()
    if not df_h.empty:
        agg = df_h.groupby('Symbol').agg({'q': 'sum', 'p': 'mean', 'c': 'sum'}).reset_index()
        agg['Live'] = agg['Symbol'].apply(get_live_price)
        agg['Market Value'] = agg['q'] * agg['Live']
        st.write("### Current Global Holdings")
        st.dataframe(agg, use_container_width=True)
    else:
        st.info("No active holdings identified.")

else:
    st.info("Please sync at least one year in the sidebar to populate the dashboard.")
