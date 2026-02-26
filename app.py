import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.set_page_config(layout="wide", page_title="Portfolio Alpha")

# ==========================================
# MODULE 1: DATA LOADING (TWO SEPARATE BUCKETS)
# ==========================================
@st.cache_data(ttl=600)
def load_data_split():
    conn = st.connection("gsheets", type=GSheetsConnection)
    def prep_sheet(name):
        return conn.read(worksheet=name).iloc[:, :13]

    # Load raw data once
    fy24 = prep_sheet("FY24")
    fy25 = prep_sheet("FY25")
    fy26 = prep_sheet("FY26")
    raw_combined = pd.concat([fy24, fy25, fy26], ignore_index=True)
    raw_combined.columns = list("ABCDEFGHIJKLM")
    raw_combined['YearSource'] = "FY" # Placeholder logic

    # --- BUCKET A: FOR SUMMARY (The "Locked" version) ---
    df_summary = raw_combined.copy()
    for col in ['F', 'G', 'H', 'I', 'K', 'M']:
        df_summary[col] = pd.to_numeric(df_summary[col].astype(str).replace(r'[$,\s()]', '', regex=True), errors='coerce').fillna(0.0)

    # --- BUCKET B: FOR FIFO/HOLDINGS (The "Text-Safe" version) ---
    df_fifo = raw_combined.copy()
    for col in ['K', 'M']: # Only force Quantity and Cost to numbers
        df_fifo[col] = pd.to_numeric(df_fifo[col].astype(str).replace(r'[$,\s()]', '', regex=True), errors='coerce').fillna(0.0)
    
    return df_summary, df_fifo

# Pull the two separate instances
df_summary, df_fifo = load_data_split()

# ==========================================
# MODULE 2: THE LOCKED SUMMARY ENGINE
# ==========================================
def s_if(df_target, target_col, a=None, b=None, c=None, d=None, e=None):
    if df_target.empty: return 0.0
    mask = pd.Series([True] * len(df_target), index=df_target.index)
    if a: mask &= (df_target['A'].astype(str).str.strip() == a)
    if b: mask &= (df_target['B'].astype(str).str.strip() == b)
    if c: mask &= (df_target['C'].astype(str).str.strip() == c)
    if d: mask &= (df_target['D'].astype(str).str.strip() == d)
    if e: mask &= (df_target['E'].astype(str).str.strip() == e)
    return float(df_target.loc[mask, target_col].sum())

# ==========================================
# MODULE 3: UI & TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Summary", "Current Holdings", "🧮 FIFO Calculator"])

with tab1:
    st.header("Summary (Locked Instance)")
    c1, c2, c3 = st.columns(3)
    # Using df_summary bucket
    c1.metric("Total Investment (USD)", f"${s_if(df_summary, 'M', a='Trades', b='Total', d='Stocks', e='USD'):,.2f}")
    c2.metric("Total Investment (AUD)", f"${s_if(df_summary, 'M', a='Trades', b='Total', d='Stocks', e='AUD'):,.2f}")
    c3.metric("Funds Deposited (AUD)", f"${s_if(df_summary, 'F', a='Deposits & Withdrawals', c='Total'):,.2f}")
    # ... Rest of your summary metrics using df_summary ...

with tab3:
    st.header("FIFO Calculator (Text-Safe Instance)")
    # Using df_fifo bucket where Column F is still TEXT
    is_trade = df_fifo['A'].astype(str).str.strip() == "Trades"
    is_data = df_fifo['B'].astype(str).str.strip() == "Data"
    fifo_source = df_fifo[is_trade & is_data]
    
    ticker_list = sorted([str(x).strip() for x in fifo_source['F'].unique() 
                         if str(x).strip() not in ['0.0', 'nan', 'None', '']])
    
    if ticker_list:
        sel_t = st.selectbox("Select Stock", ticker_list)
        st.success(f"Successfully identified ticker: {sel_t}")
    else:
        st.error("Dropdown still empty. Check Column F in Google Sheets.")
