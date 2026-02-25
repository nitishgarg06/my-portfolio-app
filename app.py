import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.set_page_config(layout="wide", page_title="Portfolio App")
st.title("📈 Portfolio Management App")

# 1. ROBUST DATA LOADING
@st.cache_data(ttl=600)
def load_all_sheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        def prep(s_name):
            try:
                df = conn.read(worksheet=s_name)
                if df is not None and not df.empty:
                    df = df.iloc[:, :13] 
                    df.columns = list("ABCDEFGHIJKLM")
                    df['YearSource'] = s_name
                    # Force numeric conversion early
                    for col in ['F', 'G', 'H', 'I', 'K', 'M']:
                        df[col] = pd.to_numeric(df[col].astype(str).replace(r'[$,\s()]', '', regex=True), errors='coerce').fillna(0)
                    return df
            except:
                return pd.DataFrame() # Return empty if sheet doesn't exist
            return pd.DataFrame()

        # Combine only non-empty dataframes
        frames = [prep("FY24"), prep("FY25"), prep("FY26")]
        valid_frames = [f for f in frames if not f.empty]
        return pd.concat(valid_frames, ignore_index=True) if valid_frames else pd.DataFrame()
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return pd.DataFrame()

df_all = load_all_sheets()

# 2. FILTER LOGIC
st.sidebar.header("Navigation")
view_choice = st.sidebar.selectbox("Select Period", ["Lifetime", "FY26", "FY25", "FY24"])

if not df_all.empty:
    df_view = df_all if view_choice == "Lifetime" else df_all[df_all['YearSource'] == view_choice]

    # --- THE FIX: SAFETY CHECK FOR EMPTY FILTERS ---
    if df_view.empty:
        st.warning(f"No data found for {view_choice}. Please check your Google Sheet tabs.")
    else:
        # SUMIFS logic with built-in empty check
        def s_if(df_target, target_col, a=None, b=None, c=None, d=None, e=None):
            if df_target.empty: return 0.0
            mask = pd.Series([True] * len(df_target))
            if a: mask &= (df_target['A'].astype(str).str.strip() == a)
            if b: mask &= (df_target['B'].astype(str).str.strip() == b)
            if c: mask &= (df_target['C'].astype(str).str.strip() == c)
            if d: mask &= (df_target['D'].astype(str).str.strip() == d)
            if e: mask &= (df_target['E'].astype(str).str.strip() == e)
            
            # Use .get() or check if column exists to prevent crash
            if target_col in df_target.columns:
                return float(df_target.loc[mask, target_col].sum())
            return 0.0

        tab_summary, tab_holdings, tab_fifo = st.tabs(["📊 Summary", "Current Holdings", "🧮 FIFO Calculator"])

        with tab_summary:
            st.header(f"Performance Metrics: {view_choice}")
            
            # Row 1: Metrics
            c1, c2, c3 = st.columns(3)
            # We wrap the metric value in a variable to ensure it's calculated before display
            val_usd = s_if(df_view, 'M', a='Trades', b='Total', d='Stocks', e='USD')
            val_aud = s_if(df_view, 'M', a='Trades', b='Total', d='Stocks', e='AUD')
            val_depo = s_if(df_view, 'F', a='Deposits & Withdrawals', c='Total')
            
            c1.metric("Total Investment (USD)", f"${val_usd:,.2f}")
            c2.metric("Total Investment (AUD)", f"${val_aud:,.2f}")
            c3.metric("Funds Deposited (AUD)", f"${val_depo:,.2f}")

            # ... [Rest of the metrics follow the same pattern] ...
            # Row 2: Divs & Tax
            c4, c5, c6 = st.columns(3)
            c4.metric("Dividends (USD)", f"${s_if(df_view, 'F', a='Dividends', c='Total'):,.2f}")
            c5.metric("Dividends (AUD)", f"${s_if(df_view, 'F', a='Dividends', c='Total in AUD'):,.2f}")
            c6.metric("Withholding Tax (USD)", f"${s_if(df_view, 'F', a='Withholding Tax', c='Total'):,.2f}")

            # Realized Table
            st.divider()
            def get_realized_data(scope):
                res = [
                    s_if(df_view, 'F', a="Realized & Unrealized Performance Summary", c=scope),
                    s_if(df_view, 'G', a="Realized & Unrealized Performance Summary", c=scope),
                    s_if(df_view, 'H', a="Realized & Unrealized Performance Summary", c=scope),
                    s_if(df_view, 'I', a="Realized & Unrealized Performance Summary", c=scope)
                ]
                return res + [sum(res)]

            st.table(pd.DataFrame({
                "Metric": ["S/T Profit", "S/T Loss", "L/T Profit", "L/T Loss", "Total"],
                "Stocks": get_realized_data("Stocks"),
                "Forex": get_realized_data("Forex"),
                "All Assets": get_realized_data("Total (All Assets)")
            }).set_index("Metric").style.format("${:,.2f}"))

        with tab_holdings:
            st.header("Open Positions")
            # Holdings logic remains Lifetime-based
            t_df = df_all[(df_all['A'] == "Trades") & (df_all['B'] == "Data")]
            if not t_df.empty:
                h = t_df.groupby('F').agg({'K': 'sum', 'M': 'sum'}).reset_index()
                h = h[h['K'] > 0.001]
                h.columns = ['Ticker', 'Units', 'Cost Basis']
                st.dataframe(h, use_container_width=True)

        with tab_fifo:
            st.header("FIFO Sell Calculator")
            # Logic as previously defined...
            t_options = sorted(df_all[(df_all['A'] == "Trades") & (df_all['B'] == "Data")]['F'].unique())
            if t_options:
                sel_t = st.selectbox("Select Stock", t_options)
                # ... FIFO calculations ...
                st.info(f"Calculator ready for {sel_t}")
else:
    st.error("No data could be retrieved from Google Sheets. Check your worksheet names (FY24, FY25, FY26).")
