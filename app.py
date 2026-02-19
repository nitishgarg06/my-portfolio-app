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
    FILE_PATH = "data/master_portfolio.csv"
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

@st.cache_data(ttl=3600)
def get_live_price(ticker):
    """Scrapes Google Finance for the current stock price."""
    try:
        url = f"https://www.google.com/search?q={ticker}+stock+price"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Common Google Finance price spans
        price_tag = soup.find('span', {'class': 'I67upf'}) or soup.find('span', {'jsname': 'vW79of'})
        return clean_numeric(price_tag.text) if price_tag else 0.0
    except: return 0.0

# --- 3. STAGE 1: HARMONIZER & GITHUB ENGINE ---
def push_to_github(df):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    res = requests.get(url, headers=headers)
    sha = res.json().get('sha') if res.status_code == 200 else None
    
    encoded = base64.b64encode(df.to_csv(index=False).encode()).decode()
    payload = {
        "message": f"Sync Portfolio: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "content": encoded, "branch": "main"
    }
    if sha: payload["sha"] = sha
    return requests.put(url, headers=headers, json=payload).status_code in [200, 201]

# --- 4. APP LOGIC ---
st.title("🏦 Wealth Terminal Pro")
curr_date = datetime.now().strftime('%d %b %Y')

# Load Persistent Data
if 'master_data' not in st.session_state:
    raw_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{FILE_PATH}?v={datetime.now().timestamp()}"
    try:
        st.session_state['master_data'] = pd.read_csv(raw_url)
    except:
        st.session_state['master_data'] = None

# Sidebar Sync Trigger
with st.sidebar:
    st.header("🔄 Data Pipeline")
    if st.button("🚀 Sync GSheets ➔ GitHub Master"):
        with st.status("Harmonizing Data...", expanded=True) as status:
            conn = st.connection("gsheets", type=GSheetsConnection)
            trades_list = []
            for t in ["FY24", "FY25", "FY26"]:
                try:
                    raw = conn.read(worksheet=t, ttl=0)
                    rows = raw[raw.iloc[:, 0].str.contains('Trades', na=False, case=False)]
                    h_row = rows[rows.iloc[:, 1] == 'Header']
                    d_rows = rows[rows.iloc[:, 1] == 'Data']
                    if not h_row.empty:
                        cols = [c for c in h_row.iloc[0, 2:].tolist() if c]
                        data = d_rows.iloc[:, 2:2+len(cols)]
                        data.columns = cols
                        data['FY_Source'] = t
                        trades_list.append(data)
                except: continue
            
            if trades_list:
                master = pd.concat(trades_list).reset_index(drop=True)
                # Apply numeric cleaning immediately
                c_qty = fuzzy_find(master, ['Quantity', 'Qty'])
                c_prc = fuzzy_find(master, ['T. Price', 'Price'])
                c_dt = fuzzy_find(master, ['Date'])
                
                master['Qty_v'] = master[c_qty].apply(clean_numeric)
                master['Prc_v'] = master[c_prc].apply(clean_numeric)
                master['Date_v'] = pd.to_datetime(master[c_dt].str.split(',').str[0], errors='coerce').dt.strftime('%Y-%m-%d')
                
                # Push to GitHub
                if push_to_github(master):
                    status.update(label="GitHub Updated!", state="complete")
                    st.session_state['master_data'] = master
                    st.rerun()
            else:
                st.error("No trades found in Sheets.")

# --- 5. DASHBOARD RENDERING ---
if st.session_state.get('master_data') is not None:
    df = st.session_state['master_data']
    
    # Fuzzy Map for Calculations
    c_cat = fuzzy_find(df, ['Category', 'Asset Class'])
    c_pl = fuzzy_find(df, ['Realized P/L', 'Realized Total'])
    c_cm = fuzzy_find(df, ['Comm'])
    c_sym = fuzzy_find(df, ['Symbol', 'Ticker'])

    # Cleaned series for math
    df['PL_v'] = df[c_pl].apply(clean_numeric) if c_pl else 0.0
    df['CM_v'] = df[c_cm].apply(clean_numeric).abs() if c_cm else 0.0
    df['Date_v'] = pd.to_datetime(df['Date_v'])

    # FIFO Engine: Current Holdings Logic
    all_open_lots = []
    for sym in df[c_sym].unique():
        sym_df = df[df[c_sym] == sym].sort_values('Date_v')
        lots = []
        for _, row in sym_df.iterrows():
            if row['Qty_v'] > 0: # Buy
                lots.append({'dt': row['Date_v'], 'q': row['Qty_v'], 'p': row['Prc_v'], 'c': row['CM_v']})
            elif row['Qty_v'] < 0: # Sell
                sell_q = abs(row['Qty_v'])
                while sell_q > 0 and lots:
                    if lots[0]['q'] <= sell_q:
                        sell_q -= lots[0]['q']
                        lots.pop(0)
                    else:
                        lots[0]['q'] -= sell_q
                        sell_q = 0
        for l in lots:
            l['Symbol'] = sym
            l['Age'] = (pd.Timestamp.now() - l['dt']).days
            l['Class'] = "Long-Term" if l['Age'] > 365 else "Short-Term"
            all_open_lots.append(l)

    df_h = pd.DataFrame(all_open_lots)

    # --- TOP LINE KPIs ---
    st.header("📈 Portfolio Performance")
    fy_list = sorted(df['FY_Source'].unique())
    sel_fy = st.selectbox("Financial Year View", fy_list, index=len(fy_list)-1)
    
    fy_df = df[df['FY_Source'] == sel_fy]
    s_pl = fy_df[fy_df[c_cat].str.contains('Stock', na=False, case=False)]['PL_v'].sum() if c_cat else 0.0
    f_pl = fy_df[fy_df[c_cat].str.contains('Forex|Cash', na=False, case=False)]['PL_v'].sum() if c_cat else 0.0
    
    # Total Investment = Basis of current lots
    total_inv = (df_h['q'] * df_h['p']).sum() + df_h['c'].sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Investment", f"${total_inv:,.2f}")
    m2.metric("Net Realized P/L", f"${(s_pl + f_pl):,.2f}")
    m3.metric("Stocks Realized", f"${s_pl:,.2f}")
    m4.metric("Forex/Impact", f"${f_pl:,.2f}")
    st.caption("ℹ️ *Disclaimer: Total Realized P/L is net of commissions.*")

    # --- HOLDINGS TABLES ---
    
    
    def render_table(data, title):
        st.subheader(f"{title} (as of {curr_date})")
        if data.empty: return st.info("No holdings in this category.")
        
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
    render_table(df_h, "1. Current Global Holdings")
    
    col_a, col_b = st.columns(2)
    with col_a: render_table(df_h[df_h['Class'] == "Short-Term"], "2. Short-Term Holdings")
    with col_b: render_table(df_h[df_h['Class'] == "Long-Term"], "3. Long-Term Holdings")

    # --- FIFO CALCULATOR ---
    
    st.divider()
    st.header("🧮 FIFO Selling Calculator")
    c1, c2 = st.columns([1, 2])
    
    s_ticker = c1.selectbox("Ticker to Simulate", df_h['Symbol'].unique())
    total_units = df_h[df_h['Symbol'] == s_ticker]['q'].sum()
    avg_cost = df_h[df_h['Symbol'] == s_ticker]['p'].mean()
    
    mode = c1.radio("Input Mode", ["Units", "Percentage"])
    if mode == "Units":
        sell_amt = c2.slider("Units to Sell", 0.0, float(total_units), float(total_units * 0.25))
    else:
        pct = c2.slider("Percentage to Sell", 0, 100, 25)
        sell_amt = total_units * (pct / 100)
    
    target_pct = c2.number_input("Target Profit %", value=110.0)
    target_price = avg_cost * (target_pct / 100)
    
    st.success(f"To hit {target_pct}% Profit: Sell **{sell_amt:.2f} units** at **${target_price:,.2f}**")
    st.info(f"Residual: {total_units - sell_amt:.2f} units remaining at ${avg_cost:,.2f} cost basis.")

else:
    st.info("👋 Welcome! Please click 'Sync GSheets ➔ GitHub Master' in the sidebar to load your data.")
