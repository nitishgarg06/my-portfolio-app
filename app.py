import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.set_page_config(layout="wide", page_title="Portfolio App")

# 1. DATA LOADING
@st.cache_data(ttl=60)
def load_all_sheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        def prep(s_name):
            df = conn.read(worksheet=s_name)
            if df is not None and not df.empty:
                df = df.iloc[:, :13] 
                df.columns = list("ABCDEFGHIJKLM")
                df['YearSource'] = s_name
                # Cleaning symbols and forcing numeric
                for col in ['F', 'G', 'H', 'I', 'K', 'M']:
                    df[col] = pd.to_numeric(df[col].astype(str).replace(r'[$,\s()]', '', regex=True), errors='coerce').fillna(0.0)
                return df
            return pd.DataFrame()
        
        frames = [prep("FY24"), prep("FY25"), prep("FY26")]
        return pd.concat([f for f in frames if not f.empty], ignore_index=True)
    except Exception as e:
        st.error(f"Load Error: {e}")
        return pd.DataFrame()

df_all = load_all_sheets()

# 2. THE STRICT SUMIFS FUNCTION (The Fix for the AssertionError)
def s_if(df_target, target_col, a=None, b=None, c=None, d=None, e=None):
    if df_target.empty:
        return 0.0
    
    # Create mask
    mask = pd.Series([True] * len(df_target), index=df_target.index)
    if a: mask &= (df_target['A'].astype(str).str.strip() == a)
    if b: mask &= (df_target['B'].astype(str).str.strip() == b)
    if c: mask &= (df_target['C'].astype(str).str.strip() == c)
    if d: mask &= (df_target['D'].astype(str).str.strip() == d)
    if e: mask &= (df_target['E'].astype(str).str.strip() == e)
    
    # Filter and Sum
    result = df_target.loc[mask, target_col].sum()
    
    # FORCE TO FLOAT: This prevents the AssertionError in st.metric
    try:
        return float(result)
    except:
        return 0.0

# 3. UI RENDERING
st.sidebar.header("Navigation")
view_choice = st.sidebar.selectbox("Select Period", ["Lifetime", "FY26", "FY25", "FY24"])

if not df_all.empty:
    df_view = df_all if view_choice == "Lifetime" else df_all[df_all['YearSource'] == view_choice]
    
    if df_view.empty:
        st.warning(f"No data available for {view_choice}")
    else:
        tab_summary, tab_holdings, tab_fifo = st.tabs(["📊 Summary", "Current Holdings", "🧮 FIFO Calculator"])

        with tab_summary:
            st.header(f"Performance Metrics: {view_choice}")
            
            # Row 1: Main Metrics
            c1, c2, c3 = st.columns(3)
            
            # Assigning values strictly as floats
            val_usd = s_if(df_view, 'M', a='Trades', b='Total', d='Stocks', e='USD')
            val_aud = s_if(df_view, 'M', a='Trades', b='Total', d='Stocks', e='AUD')
            val_depo = s_if(df_view, 'F', a='Deposits & Withdrawals', c='Total')
            
            c1.metric("Total Investment (USD)", f"${val_usd:,.2f}")
            c2.metric("Total Investment (AUD)", f"${val_aud:,.2f}")
            c3.metric("Funds Deposited (AUD)", f"${val_depo:,.2f}")

            # Row 2: Dividends
            c4, c5, c6 = st.columns(3)
            div_usd = s_if(df_view, 'F', a='Dividends', c='Total')
            div_aud = s_if(df_view, 'F', a='Dividends', c='Total in AUD')
            tax_usd = s_if(df_view, 'F', a='Withholding Tax', c='Total')
            
            c4.metric("Dividends (USD)", f"${div_usd:,.2f}")
            c5.metric("Dividends (AUD)", f"${div_aud:,.2f}")
            c6.metric("Withholding Tax (USD)", f"${tax_usd:,.2f}")

            # Realized Table
            st.divider()
            def get_realized_data(scope):
                # Returns a list of 5 floats
                p_st = s_if(df_view, 'F', a="Realized & Unrealized Performance Summary", c=scope)
                l_st = s_if(df_view, 'G', a="Realized & Unrealized Performance Summary", c=scope)
                p_lt = s_if(df_view, 'H', a="Realized & Unrealized Performance Summary", c=scope)
                l_lt = s_if(df_view, 'I', a="Realized & Unrealized Performance Summary", c=scope)
                return [p_st, l_st, p_lt, l_lt, (p_st + l_st + p_lt + l_lt)]

            realized_df = pd.DataFrame({
                "Metric": ["S/T Profit", "S/T Loss", "L/T Profit", "L/T Loss", "Total"],
                "Stocks": get_realized_data("Stocks"),
                "Forex": get_realized_data("Forex"),
                "All Assets": get_realized_data("Total (All Assets)")
            }).set_index("Metric")
            
            st.table(realized_df.style.format("${:,.2f}"))

        with tab_holdings:
            st.header("Open Positions")
            # Logic for Holdings... (omitted for brevity, keep your previous version)
            
        with tab_fifo:
            st.header("FIFO Calculator")
            # Logic for FIFO... (omitted for brevity, keep your previous version)
else:
    st.error("No data found. Check your Google Sheet tab names (FY24, FY25, FY26).")
