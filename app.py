import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.set_page_config(layout="wide", page_title="Portfolio App")
st.title("📈 Portfolio Management App")

# 1. DATA LOADING ENGINE
@st.cache_data(ttl=600) # Caches data for 10 mins to prevent constant reloading
def load_all_sheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        def prep(s_name):
            df = conn.read(worksheet=s_name)
            if df is not None and not df.empty:
                df = df.iloc[:, :13] 
                df.columns = list("ABCDEFGHIJKLM")
                df['YearSource'] = s_name
                for col in ['F', 'G', 'H', 'I', 'K', 'M']:
                    df[col] = pd.to_numeric(df[col].astype(str).replace(r'[$,\s()]', '', regex=True), errors='coerce').fillna(0)
                return df
            return pd.DataFrame()

        return pd.concat([prep("FY24"), prep("FY25"), prep("FY26")], ignore_index=True)
    except Exception as e:
        st.error(f"Sheet Connection Error: {e}")
        return pd.DataFrame()

df_all = load_all_sheets()

if not df_all.empty:
    # --- SIDEBAR ---
    st.sidebar.header("Navigation")
    view_choice = st.sidebar.selectbox("Select Period", ["Lifetime", "FY26", "FY25", "FY24"])
    
    df_view = df_all if view_choice == "Lifetime" else df_all[df_all['YearSource'] == view_choice]

    # --- ROBUST SUMIFS LOGIC ---
    def s_if(df_target, target_col, a=None, b=None, c=None, d=None, e=None):
        if df_target.empty: return 0.0
        mask = pd.Series([True] * len(df_target))
        if a: mask &= (df_target['A'].astype(str).str.strip() == a)
        if b: mask &= (df_target['B'].astype(str).str.strip() == b)
        if c: mask &= (df_target['C'].astype(str).str.strip() == c)
        if d: mask &= (df_target['D'].astype(str).str.strip() == d)
        if e: mask &= (df_target['E'].astype(str).str.strip() == e)
        
        val = df_target.loc[mask, target_col].sum()
        return float(val) if not pd.isna(val) else 0.0

    tab_summary, tab_holdings, tab_fifo = st.tabs(["📊 Summary", "Current Holdings", "🧮 FIFO Calculator"])

    with tab_summary:
        st.header(f"Performance Metrics: {view_choice}")
        
        # Row 1: Main Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Investment (USD)", f"${s_if(df_view, 'M', a='Trades', b='Total', d='Stocks', e='USD'):,.2f}")
        c2.metric("Total Investment (AUD)", f"${s_if(df_view, 'M', a='Trades', b='Total', d='Stocks', e='AUD'):,.2f}")
        c3.metric("Funds Deposited (AUD)", f"${s_if(df_view, 'F', a='Deposits & Withdrawals', c='Total'):,.2f}")

        # Row 2: Divs & Tax
        c4, c5, c6 = st.columns(3)
        c4.metric("Dividends (USD)", f"${s_if(df_view, 'F', a='Dividends', c='Total'):,.2f}")
        c5.metric("Dividends (AUD)", f"${s_if(df_view, 'F', a='Dividends', c='Total in AUD'):,.2f}")
        c6.metric("Withholding Tax (USD)", f"${s_if(df_view, 'F', a='Withholding Tax', c='Total'):,.2f}")

        # Row 3: Realized Section
        st.divider()
        st.subheader("Realized Performance Summary")
        
        def get_realized_data(scope):
            res = [
                s_if(df_view, 'F', a="Realized & Unrealized Performance Summary", c=scope), # ST Profit
                s_if(df_view, 'G', a="Realized & Unrealized Performance Summary", c=scope), # ST Loss
                s_if(df_view, 'H', a="Realized & Unrealized Performance Summary", c=scope), # LT Profit
                s_if(df_view, 'I', a="Realized & Unrealized Performance Summary", c=scope)  # LT Loss
            ]
            return res + [sum(res)] # Add total at the end

        # Constructing the table safely
        realized_df = pd.DataFrame({
            "Metric": ["S/T Profit", "S/T Loss", "L/T Profit", "L/T Loss", "Total"],
            "Stocks": get_realized_data("Stocks"),
            "Forex": get_realized_data("Forex"),
            "All Assets": get_realized_data("Total (All Assets)")
        }).set_index("Metric")
        
        st.table(realized_df.style.format("${:,.2f}"))

    with tab_holdings:
        st.header("Open Positions (Lifetime Data)")
        # Holdings logic always uses df_all
        t_df = df_all[(df_all['A'] == "Trades") & (df_all['B'] == "Data")]
        if not t_df.empty:
            h = t_df.groupby('F').agg({'K': 'sum', 'M': 'sum'}).reset_index()
            h = h[h['K'] > 0.001]
            h.columns = ['Ticker', 'Units', 'Cost Basis']
            st.dataframe(h, use_container_width=True)

    with tab_fifo:
        st.header("FIFO Sell Calculator")
        t_options = sorted(df_all[(df_all['A'] == "Trades") & (df_all['B'] == "Data")]['F'].unique())
        if t_options:
            sel_t = st.selectbox("Select Stock", t_options)
            s_df = df_all[(df_all['F'] == sel_t) & (df_all['A'] == "Trades") & (df_all['B'] == "Data")].copy()
            t_units = s_df['K'].sum()
            
            c_in1, c_in2 = st.columns(2)
            mode = c_in1.radio("Sell By:", ["Units", "Percentage"])
            target = c_in2.number_input("Target Profit %", value=15.0)
            
            amt = st.slider("Quantity", 0.0, float(t_units), step=0.01) if mode == "Units" else \
                  (st.slider("Percentage", 0, 100, 20) / 100) * t_units

            if amt > 0:
                # Basic FIFO logic
                queue = []
                for _, r in s_df.iterrows():
                    if r['K'] > 0: queue.append({'q': r['K'], 'b': r['M']})
                    else:
                        rem = abs(r['K'])
                        while rem > 0 and queue:
                            if queue[0]['q'] <= rem: rem -= queue.pop(0)['q']
                            else:
                                queue[0]['q'] -= rem
                                rem = 0
                
                cur_cost, rem_units = 0, amt
                for lot in queue:
                    if rem_units <= 0: break
                    taken = min(lot['q'], rem_units)
                    cur_cost += (taken / lot['q']) * lot['basis'] if lot['q'] > 0 else 0
                    rem_units -= taken
                
                st.success(f"**Target Sell Price:** ${cur_cost * (1 + (target/100)):,.2f}")
                st.info(f"**Remaining Units:** {t_units - amt:,.4f}")
else:
    st.warning("No data found. Please verify Google Sheet tab names are exactly FY24, FY25, FY26.")
