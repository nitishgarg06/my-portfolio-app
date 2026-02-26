import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

# ==========================================
# MODULE 1: THE LOCKED SUMMARY ENGINE
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
        return float(df_target.loc[mask, target_col].sum())

    # Realized Gains Helper
    def get_realized(scope):
        p_st = s_if('F', a="Realized & Unrealized Performance Summary", c=scope)
        l_st = s_if('G', a="Realized & Unrealized Performance Summary", c=scope)
        p_lt = s_if('H', a="Realized & Unrealized Performance Summary", c=scope)
        l_lt = s_if('I', a="Realized & Unrealized Performance Summary", c=scope)
        return [p_st, l_st, p_lt, l_lt, (p_st + l_st + p_lt + l_lt)]

    return {
        "inv_usd": s_if('M', a='Trades', b='Total', d='Stocks', e='USD'),
        "inv_aud": s_if('M', a='Trades', b='Total', d='Stocks', e='AUD'),
        "div_usd": s_if('F', a='Dividends', c='Total'),
        "div_aud": s_if('F', a='Dividends', c='Total in AUD'),
        "tax_usd": s_if('F', a='Withholding Tax', c='Total'),
        "tax_aud": s_if('F', a='Withholding Tax', c='Total in AUD'),
        "depo_aud": s_if('F', a='Deposits & Withdrawals', c='Total'),
        "stocks_realized": get_realized("Stocks"),
        "forex_realized": get_realized("Forex"),
        "total_realized": get_realized("Total (All Assets)")
    }

# ==========================================
# MODULE 2: DATA LOADING
# ==========================================
st.set_page_config(layout="wide", page_title="Portfolio Alpha")

@st.cache_data(ttl=600)
def load_data():
    try:
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
    except Exception as e:
        st.error(f"Load Error: {e}")
        return pd.DataFrame()

df_all = load_data()

# ==========================================
# MODULE 3: UI & TABS
# ==========================================
view_choice = st.sidebar.selectbox("Select Period", ["Lifetime", "FY26", "FY25", "FY24"])
df_view = df_all if view_choice == "Lifetime" else df_all[df_all['YearSource'] == view_choice]

tab1, tab2, tab3 = st.tabs(["📊 Summary", "Current Holdings", "🧮 FIFO Calculator"])

with tab1:
    m = get_summary_metrics(df_view)
    st.header(f"Summary: {view_choice}")
    
    # Row 1: Investment
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Investment (USD)", f"${m['inv_usd']:,.2f}")
    c2.metric("Total Investment (AUD)", f"${m['inv_aud']:,.2f}")
    c3.metric("Funds Deposited (AUD)", f"${m['depo_aud']:,.2f}")

    # Row 2: Divs & Tax
    c4, c5, c6 = st.columns(3)
    c4.metric("Dividends (USD)", f"${m['div_usd']:,.2f}")
    c5.metric("Dividends (AUD)", f"${m['div_aud']:,.2f}")
    c6.metric("Withholding Tax (USD)", f"${m['tax_usd']:,.2f}")

    # Row 3: Realized Table
    st.divider()
    st.subheader("Realized Gains/Losses")
    realized_df = pd.DataFrame({
        "Metric": ["S/T Profit", "S/T Loss", "L/T Profit", "L/T Loss", "Total"],
        "Stocks": m['stocks_realized'],
        "Forex": m['forex_realized'],
        "All Assets": m['total_realized']
    }).set_index("Metric")
    st.table(realized_df.style.format("${:,.2f}"))

with tab2:
    st.header("Current Open Positions")
    # Logic for Holdings using individual 'Data' rows
    h_data = df_all[(df_all['A'].astype(str).str.strip() == "Trades") & (df_all['B'].astype(str).str.strip() == "Data")]
    if not h_data.empty:
        h_table = h_data.groupby('F').agg({'K': 'sum', 'M': 'sum'}).reset_index()
        h_table = h_table[h_table['K'] > 0.001]
        h_table.columns = ['Ticker', 'Units', 'Cost Basis']
        st.dataframe(h_table.style.format({"Units": "{:.4f}", "Cost Basis": "${:,.2f}"}), use_container_width=True)

with tab3:
    st.header("FIFO Sell Calculator")
    
    # 1. FIND THE TRADES
    is_trade_row = df_all['A'].astype(str).str.strip().str.upper() == "TRADES"
    trade_sample = df_all[is_trade_row].copy()

    if not trade_sample.empty:
        st.subheader("🔍 Column Mapping Check")
        st.write("Below are the first 5 'Trades' rows from your sheet. Which column contains the Ticker Symbol?")
        # Display the raw data so you can see where the ticker is
        st.table(trade_sample[['A', 'B', 'C', 'D', 'E', 'F', 'G']].head(5))
        
        # 2. ATTEMPT TO FIND TICKERS
        # We will look in F, but also check if they are in C or E by mistake
        found_tickers = [str(x).strip() for x in trade_sample['F'].unique() if len(str(x).strip()) > 1 and str(x) != '0.0']
        
        if found_tickers:
            sel_t = st.selectbox("Select Stock", sorted(found_tickers))
            st.success(f"Ticker {sel_t} identified in Column F.")
        else:
            st.error("Column F appears to be empty for these rows.")
            st.write("Unique values in Column C:", trade_sample['C'].unique())
            st.write("Unique values in Column E:", trade_sample['E'].unique())
    else:
        st.error("No rows found where Column A = 'Trades'.")


