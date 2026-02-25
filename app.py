import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.set_page_config(layout="wide", page_title="Portfolio Alpha")

# --- 1. DATA LOADING (NO GLOBAL FILTERING) ---
@st.cache_data(ttl=60)
def load_data():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        def prep(s_name):
            df = conn.read(worksheet=s_name)
            if df is not None and not df.empty:
                df = df.iloc[:, :13] 
                df.columns = list("ABCDEFGHIJKLM")
                df['YearSource'] = s_name
                # Convert columns to string for matching, numbers for math
                for col in ['F', 'G', 'H', 'I', 'K', 'M']:
                    df[col] = pd.to_numeric(df[col].astype(str).replace(r'[$,\s()]', '', regex=True), errors='coerce').fillna(0.0)
                return df
            return pd.DataFrame()
        
        frames = [prep("FY24"), prep("FY25"), prep("FY26")]
        return pd.concat([f for f in frames if not f.empty], ignore_index=True)
    except Exception as e:
        st.error(f"Load Error: {e}")
        return pd.DataFrame()

df_all = load_data()

# --- 2. THE SUMIFS ENGINE (For Summary Tab) ---
def s_if(df_target, target_col, a=None, b=None, c=None, d=None, e=None):
    if df_target.empty: return 0.0
    mask = pd.Series([True] * len(df_target), index=df_target.index)
    # Use case-insensitive, stripped matching
    if a: mask &= (df_target['A'].astype(str).str.strip().str.upper() == a.upper())
    if b: mask &= (df_target['B'].astype(str).str.strip().str.upper() == b.upper())
    if c: mask &= (df_target['C'].astype(str).str.strip().str.upper() == c.upper())
    if d: mask &= (df_target['D'].astype(str).str.strip().str.upper() == d.upper())
    if e: mask &= (df_target['E'].astype(str).str.strip().str.upper() == e.upper())
    
    return float(df_target.loc[mask, target_col].sum())

# --- 3. UI RENDERING ---
st.sidebar.header("Navigation")
view_choice = st.sidebar.selectbox("Select Period", ["Lifetime", "FY26", "FY25", "FY24"])

if not df_all.empty:
    df_view = df_all if view_choice == "Lifetime" else df_all[df_all['YearSource'] == view_choice]
    
    tab_summary, tab_holdings, tab_fifo = st.tabs(["📊 Summary", "Current Holdings", "🧮 FIFO Calculator"])

    with tab_summary:
        st.header(f"Performance Metrics: {view_choice}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Investment (USD)", f"${s_if(df_view, 'M', a='Trades', b='Total', d='Stocks', e='USD'):,.2f}")
        c2.metric("Total Investment (AUD)", f"${s_if(df_view, 'M', a='Trades', b='Total', d='Stocks', e='AUD'):,.2f}")
        c3.metric("Funds Deposited (AUD)", f"${s_if(df_view, 'F', a='Deposits & Withdrawals', c='Total'):,.2f}")
        # (Add other metrics here as needed following the s_if pattern)

    with tab_holdings:
        st.header("Open Positions")
        # Holdings only use individual DATA rows
        h_df = df_all[(df_all['A'].astype(str).str.strip().str.upper() == "TRADES") & 
                      (df_all['B'].astype(str).str.strip().str.upper() == "DATA")]
        if not h_df.empty:
            h = h_df.groupby('F').agg({'K': 'sum', 'M': 'sum'}).reset_index()
            h = h[h['K'] > 0.001]
            st.dataframe(h.style.format({"K": "{:.4f}", "M": "${:,.2f}"}))

    with tab_fifo:
        st.header("FIFO Sell Calculator")
        # Filter for tickers specifically in Column F where Row is Trades + Data
        fifo_data = df_all[(df_all['A'].astype(str).str.strip().str.upper() == "TRADES") & 
                           (df_all['B'].astype(str).str.strip().str.upper() == "DATA")]
        
        # We look specifically at Column F for the ticker names
        ticker_list = sorted([str(x) for x in fifo_data['F'].unique() if str(x) not in ['0.0', '0', 'nan']])
        
        if ticker_list:
            sel_t = st.selectbox("Select Stock", ticker_list)
            # Reconstruct FIFO
            s_df = fifo_data[fifo_data['F'].astype(str) == sel_t].copy()
            # ... FIFO calculation logic (same as before) ...
            st.info(f"Calculator ready for {sel_t}")
        else:
            st.warning("FIFO dropdown is empty. Check if Column F has tickers in your 'Data' rows.")
            # DIAGNOSTIC: Show the user what Column A/B/F look like
            st.write("Current Col A/B/F sample:", df_all[['A', 'B', 'F']].head(10))

else:
    st.error("No data loaded. Check Google Sheets connection.")
