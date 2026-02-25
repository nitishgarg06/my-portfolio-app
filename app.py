import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

# ==========================================
# MODULE 1: THE LOCKED SUMMARY ENGINE
# DO NOT EDIT THIS SECTION
# ==========================================
def get_summary_metrics(df_target):
    """
    This function is the 'Locked' Source of Truth for Tab 1.
    It replicates the exact SUMIFS logic from your spreadsheet.
    """
    def s_if(target_col, a=None, b=None, c=None, d=None, e=None):
        if df_target.empty: return 0.0
        mask = pd.Series([True] * len(df_target), index=df_target.index)
        if a: mask &= (df_target['A'].astype(str).str.strip() == a)
        if b: mask &= (df_target['B'].astype(str).str.strip() == b)
        if c: mask &= (df_target['C'].astype(str).str.strip() == c)
        if d: mask &= (df_target['D'].astype(str).str.strip() == d)
        if e: mask &= (df_target['E'].astype(str).str.strip() == e)
        return float(df_target.loc[mask, target_col].sum())

    # Return a dictionary of all required metrics
    return {
        "inv_usd": s_if('M', a='Trades', b='Total', d='Stocks', e='USD'),
        "inv_aud": s_if('M', a='Trades', b='Total', d='Stocks', e='AUD'),
        "depo_aud": s_if('F', a='Deposits & Withdrawals', c='Total'),
        "div_usd": s_if('F', a='Dividends', c='Total'),
        "div_aud": s_if('F', a='Dividends', c='Total in AUD'),
        "tax_usd": s_if('F', a='Withholding Tax', c='Total'),
        # Performance Rows
        "stocks_p": [s_if('F', a="Realized & Unrealized Performance Summary", c="Stocks"),
                     s_if('G', a="Realized & Unrealized Performance Summary", c="Stocks"),
                     s_if('H', a="Realized & Unrealized Performance Summary", c="Stocks"),
                     s_if('I', a="Realized & Unrealized Performance Summary", c="Stocks")]
    }

# ==========================================
# MODULE 2: DATA & UI SETUP
# ==========================================
st.set_page_config(layout="wide", page_title="Portfolio Alpha")

@st.cache_data(ttl=600)
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    def prep(name):
        df = conn.read(worksheet=name)
        if df is not None and not df.empty:
            df = df.iloc[:, :13]
            df.columns = list("ABCDEFGHIJKLM")
            df['YearSource'] = name
            for col in ['F', 'G', 'H', 'I', 'K', 'M']:
                df[col] = pd.to_numeric(df[col].astype(str).replace(r'[$,\s()]', '', regex=True), errors='coerce').fillna(0.0)
            return df
        return pd.DataFrame()
    return pd.concat([prep("FY24"), prep("FY25"), prep("FY26")], ignore_index=True)

df_all = load_data()
view_choice = st.sidebar.selectbox("Select Period", ["Lifetime", "FY26", "FY25", "FY24"])
df_view = df_all if view_choice == "Lifetime" else df_all[df_all['YearSource'] == view_choice]

tab1, tab2, tab3 = st.tabs(["📊 Summary", "Current Holdings", "🧮 FIFO Calculator"])

# ==========================================
# MODULE 3: TAB EXECUTION
# ==========================================

with tab1:
    # CALLING THE LOCKED ENGINE
    m = get_summary_metrics(df_view)
    
    st.header(f"Performance Metrics: {view_choice}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Investment (USD)", f"${m['inv_usd']:,.2f}")
    c2.metric("Total Investment (AUD)", f"${m['inv_aud']:,.2f}")
    c3.metric("Funds Deposited (AUD)", f"${m['depo_aud']:,.2f}")
    
    # Realized Gain Table Logic...
    st.write("Realized gains based on locked logic.")

with tab2:
    st.header("Holdings")
    # Work on this freely without affecting Module 1
    st.write("Developing holdings view...")

with tab3:
    st.header("FIFO Calculator")
    # Work on this freely without affecting Module 1
    st.write("Developing FIFO logic...")
