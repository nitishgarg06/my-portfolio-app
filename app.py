import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.set_page_config(layout="wide", page_title="Portfolio Alpha")
st.title("📈 Portfolio Management App")

# --- 1. DATA LOADING (STRICT COLUMN MAPPING) ---
@st.cache_data(ttl=600)
def load_data():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        def prep(s_name):
            df = conn.read(worksheet=s_name)
            if df is not None and not df.empty:
                df = df.iloc[:, :13] 
                df.columns = list("ABCDEFGHIJKLM")
                df['YearSource'] = s_name
                # Clean numeric columns
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

# --- 2. THE LOCKED SUMIFS ENGINE (For Summary Tab) ---
def s_if(df_target, target_col, a=None, b=None, c=None, d=None, e=None):
    if df_target.empty: return 0.0
    mask = pd.Series([True] * len(df_target), index=df_target.index)
    if a: mask &= (df_target['A'].astype(str).str.strip() == a)
    if b: mask &= (df_target['B'].astype(str).str.strip() == b)
    if c: mask &= (df_target['C'].astype(str).str.strip() == c)
    if d: mask &= (df_target['D'].astype(str).str.strip() == d)
    if e: mask &= (df_target['E'].astype(str).str.strip() == e)
    
    result = df_target.loc[mask, target_col].sum()
    return float(result)

# --- 3. UI RENDERING ---
st.sidebar.header("Navigation")
view_choice = st.sidebar.selectbox("Select Period", ["Lifetime", "FY26", "FY25", "FY24"])

if not df_all.empty:
    df_view = df_all if view_choice == "Lifetime" else df_all[df_all['YearSource'] == view_choice]
    
    tab_summary, tab_holdings, tab_fifo = st.tabs(["📊 Summary", "Current Holdings", "🧮 FIFO Calculator"])

    # --- TAB 1: SUMMARY (RESTORED TO LOCKED VERSION) ---
    with tab_summary:
        st.header(f"Performance Metrics: {view_choice}")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Investment (USD)", f"${s_if(df_view, 'M', a='Trades', b='Total', d='Stocks', e='USD'):,.2f}")
        c2.metric("Total Investment (AUD)", f"${s_if(df_view, 'M', a='Trades', b='Total', d='Stocks', e='AUD'):,.2f}")
        c3.metric("Funds Deposited (AUD)", f"${s_if(df_view, 'F', a='Deposits & Withdrawals', c='Total'):,.2f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Dividends (USD)", f"${s_if(df_view, 'F', a='Dividends', c='Total'):,.2f}")
        c5.metric("Dividends (AUD)", f"${s_if(df_view, 'F', a='Dividends', c='Total in AUD'):,.2f}")
        c6.metric("Withholding Tax (USD)", f"${s_if(df_view, 'F', a='Withholding Tax', c='Total'):,.2f}")

        st.divider()
        st.subheader("Realized Performance Summary")
        
        def get_realized_row(scope):
            p_st = s_if(df_view, 'F', a="Realized & Unrealized Performance Summary", c=scope)
            l_st = s_if(df_view, 'G', a="Realized & Unrealized Performance Summary", c=scope)
            p_lt = s_if(df_view, 'H', a="Realized & Unrealized Performance Summary", c=scope)
            l_lt = s_if(df_view, 'I', a="Realized & Unrealized Performance Summary", c=scope)
            return [p_st, l_st, p_lt, l_lt, (p_st + l_st + p_lt + l_lt)]

        real_df = pd.DataFrame({
            "Metric": ["S/T Profit", "S/T Loss", "L/T Profit", "L/T Loss", "Total"],
            "Stocks": get_realized_row("Stocks"),
            "Forex": get_realized_row("Forex"),
            "All Assets": get_realized_row("Total (All Assets)")
        }).set_index("Metric")
        st.table(real_df.style.format("${:,.2f}"))

    # --- TAB 2: HOLDINGS (FIXED FILTER) ---
    with tab_holdings:
        st.header("Open Positions")
        # Use only individual trade data rows
        h_data = df_all[(df_all['A'].astype(str).str.strip() == "Trades") & 
                        (df_all['B'].astype(str).str.strip() == "Data")]
        if not h_data.empty:
            h_table = h_data.groupby('F').agg({'K': 'sum', 'M': 'sum'}).reset_index()
            h_table = h_table[h_table['K'] > 0.001]
            h_table.columns = ['Ticker', 'Units', 'Cost Basis']
            st.dataframe(h_table.style.format({"Units": "{:.4f}", "Cost Basis": "${:,.2f}"}))
        else:
            st.info("No individual trade rows found.")

    # --- TAB 3: FIFO CALCULATOR (RETROSPECTIVE LOGIC) ---
    with tab_fifo:
        st.header("FIFO Sell Calculator")
        
        # Pull Tickers from Data rows in Column F
        fifo_source = df_all[(df_all['A'].astype(str).str.strip() == "Trades") & 
                             (df_all['B'].astype(str).str.strip() == "Data")]
        
        ticker_list = sorted([str(x) for x in fifo_source['F'].unique() if str(x) not in ['0.0', 'nan', '0']])
        
        if ticker_list:
            sel_t = st.selectbox("Select Stock", ticker_list)
            
            # Filter all years for this stock's history
            s_history = fifo_source[fifo_source['F'].astype(str).str.strip() == sel_t].copy()
            
            # Build FIFO Queue
            queue = []
            for _, r in s_history.iterrows():
                q, b = float(r['K']), float(r['M'])
                if q > 0: queue.append({'q': q, 'b': b})
                elif q < 0:
                    rem = abs(q)
                    while rem > 0 and queue:
                        if queue[0]['q'] <= rem: rem -= queue.pop(0)['q']
                        else:
                            queue[0]['q'] -= rem
                            rem = 0
            
            t_held = sum(i['q'] for i in queue)
            
            c_in1, c_in2 = st.columns(2)
            profit_goal = c_in2.number_input("Target Profit %", value=15.0)
            amt = c_in1.slider("Quantity", 0.0, float(t_held), step=0.01)

            if amt > 0:
                temp_q, cost_sum = amt, 0.0
                for lot in queue:
                    if temp_q <= 0: break
                    take = min(lot['q'], temp_q)
                    cost_sum += (take / lot['q']) * lot['b']
                    temp_q -= take
                
                target_val = cost_sum * (1 + (profit_goal/100))
                st.success(f"### Target Sell Value: ${target_val:,.2f}")
                st.info(f"Remaining: {t_held - amt:,.4f} units")
        else:
            st.warning("No tickers found. Verify Column F contains symbols in 'Data' rows.")

else:
    st.error("Data Load Failed.")
