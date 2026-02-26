import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

# ==========================================
# MODULE 1: THE LOCKED SUMMARY ENGINE (Updated for Safety)
# ==========================================
def get_summary_metrics(df_target):
    def s_if(target_col, a=None, b=None, c=None, d=None, e=None):
        if df_target.empty: return 0.0
        mask = pd.Series([True] * len(df_target), index=df_target.index)
        if a: mask &= (df_target['A'].astype(str).str.strip() == a)
        if b: mask &= (df_target['B'].astype(str).str.strip() == b)
        if c: mask &= (df_target['C'].astype(str).str.strip() == c)
        if d: mask &= (df_target['D'].astype(str).str.strip() == d)
        if e: mask &= (df_target['E'].astype(str).str.strip() == e)
        
        # FIX: Convert to numeric ONLY during the sum to protect text in other rows
        values = pd.to_numeric(
            df_target.loc[mask, target_col].astype(str).replace(r'[$,\s()]', '', regex=True), 
            errors='coerce'
        ).fillna(0.0)
        return float(values.sum())

    def get_realized(scope):
        return [s_if('F', a="Realized & Unrealized Performance Summary", c=scope),
                s_if('G', a="Realized & Unrealized Performance Summary", c=scope),
                s_if('H', a="Realized & Unrealized Performance Summary", c=scope),
                s_if('I', a="Realized & Unrealized Performance Summary", c=scope),
                0.0] # Total calculated in the table UI

    return {
        "inv_usd": s_if('M', a='Trades', b='Total', d='Stocks', e='USD'),
        "inv_aud": s_if('M', a='Trades', b='Total', d='Stocks', e='AUD'),
        "div_usd": s_if('F', a='Dividends', c='Total'),
        "div_aud": s_if('F', a='Dividends', c='Total in AUD'),
        "tax_usd": s_if('F', a='Withholding Tax', c='Total'),
        "depo_aud": s_if('F', a='Deposits & Withdrawals', c='Total'),
        "stocks_realized": get_realized("Stocks"),
        "forex_realized": get_realized("Forex"),
        "total_realized": get_realized("Total (All Assets)")
    }

# ==========================================
# MODULE 2: DATA LOADING (REPAIRED)
# ==========================================
@st.cache_data(ttl=600)
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    def prep(name):
        df = conn.read(worksheet=name)
        if df is not None and not df.empty:
            df = df.iloc[:, :13]
            df.columns = list("ABCDEFGHIJKLM")
            df['YearSource'] = name
            # FIX: We NO LONGER force F, G, H, I to numbers here. 
            # We only force K and M because they are ALWAYS numbers.
            for col in ['K', 'M']:
                df[col] = pd.to_numeric(df[col].astype(str).replace(r'[$,\s()]', '', regex=True), errors='coerce').fillna(0.0)
            return df
        return pd.DataFrame()
    return pd.concat([prep("FY24"), prep("FY25"), prep("FY26")], ignore_index=True)

df_all = load_data()

# ==========================================
# MODULE 3: TABS (FIFO Dropdown Fixed)
# ==========================================
st.sidebar.header("Navigation")
view_choice = st.sidebar.selectbox("Select Period", ["Lifetime", "FY26", "FY25", "FY24"])
df_view = df_all if view_choice == "Lifetime" else df_all[df_all['YearSource'] == view_choice]

tab1, tab2, tab3 = st.tabs(["📊 Summary", "Current Holdings", "🧮 FIFO Calculator"])

with tab1:
    m = get_summary_metrics(df_view)
    st.header(f"Summary: {view_choice}")
    # ... (Standard Summary UI code follows)

with tab3:
    st.header("FIFO Sell Calculator")
    
    # Now Column F will correctly show 'AMD', 'TSLA', etc. instead of '0.0'
    is_trade = df_all['A'].astype(str).str.strip() == "Trades"
    is_data = df_all['B'].astype(str).str.strip() == "Data"
    fifo_source = df_all[is_trade & is_data]
    
    ticker_list = sorted([str(x).strip() for x in fifo_source['F'].unique() 
                         if str(x).strip() not in ['0.0', 'nan', 'None', '']])
    
    if ticker_list:
        sel_t = st.selectbox("Select Stock", ticker_list)
        st.success(f"Selected: {sel_t}")
    else:
        st.error("Dropdown is still empty. Check Column F in your 'Data' rows.")
