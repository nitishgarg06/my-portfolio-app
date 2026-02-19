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
    st.error("Secrets Error: Ensure GITHUB_TOKEN and GITHUB_REPO are in your Streamlit Secrets.")
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

# --- 3. GITHUB PERSISTENCE ENGINE ---
def push_to_github(df, fy, filename):
    path = f"data/{fy}/{filename}.csv"
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    
    res = requests.get(url, headers=headers)
    sha = res.json().get('sha') if res.status_code == 200 else None
    content = base64.b64encode(df.to_csv(index=False).encode()).decode()
    
    payload = {
        "message": f"Sync {fy} {filename}: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "content": content, "branch": "main"
    }
    if sha: payload["sha"] = sha
    return requests.put(url, headers=headers, json=payload).status_code in [200, 201]

def load_from_github(fy, filename):
    url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/data/{fy}/{filename}.csv?v={datetime.now().timestamp()}"
    try:
        return pd.read_csv(url)
    except:
        return None

# --- 4. SIDEBAR: SELECTIVE SYNC ---
st.title("🏦 Wealth Terminal Pro")
curr_date = datetime.now().strftime('%d %b %Y')

with st.sidebar:
    st.header("🔄 On-Demand Sync")
    sync_fy = st.selectbox("Select Year to Sync", ["FY24", "FY25", "FY26"])
    
    if st.button(f"🚀 Sync {sync_fy} GSheets ➔ GitHub"):
        with st.status(f"Processing {sync_fy}...", expanded=True) as status:
            conn = st.connection("gsheets", type=GSheetsConnection)
            raw = conn.read(worksheet=sync_fy, ttl=0)
            
            # Extract and Clean Sections
            t_df = get_ibkr_section(raw, 'Trades')
            p_df = get_ibkr_section(raw, 'Realized & Unrealized Performance Summary')
            f_df = get_ibkr_section(raw, 'Cash Report')
            
            # Push to GitHub
            success = True
            if not t_df.empty: success &= push_to_github(t_df, sync_fy, "trades")
            if not p_df.empty: success &= push_to_github(p_df, sync_fy, "perf")
            if not f_df.empty: success &= push_to_github(f_df, sync_fy, "forex")
            
            if success:
                st.session_state[f'last_sync_{sync_fy}'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                status.update(label=f"{sync_fy} Synced Successfully!", state="complete")
                st.rerun()
            else:
                st.error("Sync failed. Check GitHub settings.")

    if f'last_sync_{sync_fy}' in st.session_state:
        st.caption(f"📅 **Last Sync:** {st.session_state[f'last_sync_{sync_fy}']}")

# --- 5. DASHBOARD: VIEWER ---
st.divider()
view_fy = st.radio("Select Financial Year to View", ["FY24", "FY25", "FY26"], horizontal=True)

# Load data for selected FY
t_df = load_from_github(view_fy, "trades")
p_df = load_from_github(view_fy, "perf")
f_df = load_from_github(view_fy, "forex")

if t_df is not None:
    # --- DATA HARMONIZATION ---
    c_qty = fuzzy_find(t_df, ['Quantity', 'Qty'])
    c_prc = fuzzy_find(t_df, ['Price', 'T. Price'])
    c_dt  = fuzzy_find(t_df, ['Date'])
    c_sym = fuzzy_find(t_df, ['Symbol'])
    c_cm  = fuzzy_find(t_df, ['Comm'])

    t_df['Qty_v'] = t_df[c_qty].apply(clean_numeric)
    t_df['Prc_v'] = t_df[c_prc].apply(clean_numeric)
    t_df['Comm_v'] = t_df[c_cm].apply(clean_numeric).abs() if c_cm else 0.0
    t_df['DT_v'] = pd.to_datetime(t_df[c_dt].str.split(',').str[0], errors='coerce')

    # FIFO Engine
    all_open_lots = []
    for sym in t_df[c_sym].unique():
        sym_df = t_df[t_df[c_sym] == sym].sort_values('DT_v')
        lots = []
        for _, row in sym_df.iterrows():
            if row['Qty_v'] > 0:
                lots.append({'dt': row['DT_v'], 'q': row['Qty_v'], 'p': row['Prc_v'], 'c': row['Comm_v']})
            elif row['Qty_v'] < 0:
                sq = abs(row['Qty_v'])
                while sq > 0 and lots:
                    if lots[0]['q'] <= sq: sq -= lots.pop(0)['q']
                    else: lots[0]['q'] -= sq; sq = 0
        for l in lots:
            l['Symbol'] = sym
            l['Days'] = (pd.Timestamp.now() - l['dt']).days
            l['Type'] = "Long-Term" if l['Days'] > 365 else "Short-Term"
            all_open_lots.append(l)

    df_h = pd.DataFrame(all_open_lots)

    # --- TOP LINE KPIs ---
    
    
    # Realized P/L from Performance file
    stocks_pl, forex_pl = 0.0, 0.0
    if p_df is not None:
        cat_col = fuzzy_find(p_df, ['Asset Category', 'Category'])
        rt_col = fuzzy_find(p_df, ['Realized Total', 'Realized P/L'])
        if cat_col and rt_col:
            stocks_pl = p_df[p_df[cat_col].str.contains('Stock', na=False, case=False)][rt_col].apply(clean_numeric).sum()
            forex_pl = p_df[p_df[cat_col].str.contains('Forex|Cash', na=False, case=False)][rt_col].apply(clean_numeric).sum()

    total_inv = (df_h['q'] * df_h['p']).sum() + df_h['c'].sum() if not df_h.empty else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Investment", f"${total_inv:,.2f}")
    m2.metric("Net Realized P/L", f"${(stocks_pl + forex_pl):,.2f}")
    m3.metric("Stocks Realized", f"${stocks_pl:,.2f}")
    m4.metric("Forex/Impact", f"${forex_pl:,.2f}")
    st.caption("ℹ️ *Disclaimer: Total Realized P/L is net of commissions.*")

    # --- TABLES ---
    def render_h_table(data, title):
        st.subheader(f"{title} (as of {curr_date})")
        if data.empty: return st.info("No holdings.")
        agg = data.groupby('Symbol').agg({'q': 'sum', 'p': 'mean', 'c': 'sum'}).reset_index()
        agg['Live Price'] = agg['Symbol'].apply(get_live_price)
        agg['Total Basis'] = (agg['q'] * agg['p']) + agg['c']
        agg['Market Value'] = agg['q'] * agg['Live Price']
        agg['P/L $'] = agg['Market Value'] - agg['Total Basis']
        agg['P/L %'] = (agg['P/L $'] / agg['Total Basis']) * 100
        st.dataframe(agg.style.format({
            "q": "{:.2f}", "p": "${:.2f}", "c": "${:.2f}", "Live Price": "${:.2f}",
            "Total Basis": "${:.2f}", "Market Value": "${:.2f}", "P/L $": "${:.2f}", "P/L %": "{:.2f}%"
        }), use_container_width=True)

    st.divider()
    render_h_table(df_h, "1. Current Global Holdings")
    
    ca, cb = st.columns(2)
    with ca: render_h_table(df_h[df_h['Type'] == "Short-Term"], "2. Short-Term Holdings")
    with cb: render_h_table(df_h[df_h['Type'] == "Long-Term"], "3. Long-Term Holdings")

    # --- CALCULATOR ---
    
    st.divider()
    st.header("🧮 FIFO Selling Calculator")
    c1, c2 = st.columns([1, 2])
    pick = c1.selectbox("Select Ticker", df_h['Symbol'].unique())
    h_row = df_h[df_h['Symbol'] == pick]
    t_units = h_row['q'].sum()
    a_cost = h_row['p'].mean()
    
    mode = c1.radio("Mode", ["Units", "Percentage"])
    s_qty = c2.slider("Quantity", 0.0, float(t_units), float(t_units*0.25)) if mode == "Units" else t_units * (c2.slider("%", 0, 100, 25)/100)
    t_prof = c2.number_input("Target Profit %", value=110.0)
    
    st.success(f"Sell at **${(a_cost * (t_prof/100)):,.2f}** | Residual: {t_units - s_qty:.2f} units")

else:
    st.info(f"No GitHub data for {view_fy}. Please select it in the sidebar and click Sync.")
