import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

# --- 1. SETUP & PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Portfolio Alpha")
st.title("📈 Portfolio Management App")

# --- 2. DATA LOADING ENGINE ---
@st.cache_data(ttl=600)
def load_and_standardize_data():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        
        def prep_sheet(name):
            df = conn.read(worksheet=name)
            if df is not None and not df.empty:
                # Force first 13 columns to letters A through M
                df = df.iloc[:, :13]
                df.columns = list("ABCDEFGHIJKLM")
                df['YearSource'] = name
                # Clean numeric columns (remove $, commas, handle parentheses)
                for col in ['F', 'G', 'H', 'I', 'K', 'M']:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).replace(r'[$,\s()]', '', regex=True), 
                        errors='coerce'
                    ).fillna(0.0)
                return df
            return pd.DataFrame()

        # Combine all fiscal years
        sheets = [prep_sheet("FY24"), prep_sheet("FY25"), prep_sheet("FY26")]
        combined = pd.concat([s for s in sheets if not s.empty], ignore_index=True)
        return combined
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return pd.DataFrame()

df_all = load_and_standardize_data()

# --- 3. ROBUST SUMIFS FUNCTION (For Summary Tab) ---
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

# --- 4. NAVIGATION & FILTERING ---
st.sidebar.header("Navigation")
view_choice = st.sidebar.selectbox("Select Period", ["Lifetime", "FY26", "FY25", "FY24"])

if not df_all.empty:
    df_view = df_all if view_choice == "Lifetime" else df_all[df_all['YearSource'] == view_choice]
    
    tab_summary, tab_holdings, tab_fifo = st.tabs(["📊 Summary", "Current Holdings", "🧮 FIFO Calculator"])

    # --- TAB 1: SUMMARY (METHOD B - TOTALS) ---
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

        realized_df = pd.DataFrame({
            "Metric": ["S/T Profit", "S/T Loss", "L/T Profit", "L/T Loss", "Total"],
            "Stocks": get_realized_row("Stocks"),
            "Forex": get_realized_row("Forex"),
            "All Assets": get_realized_row("Total (All Assets)")
        }).set_index("Metric")
        st.table(realized_df.style.format("${:,.2f}"))

    # --- TAB 2: HOLDINGS (METHOD A - DATA ROWS) ---
    with tab_holdings:
        st.header("Open Positions (Historical Data)")
        # Use individual trade data only
        t_df = df_all[(df_all['A'] == "Trades") & (df_all['B'] == "Data")]
        if not t_df.empty:
            h = t_df.groupby('F').agg({'K': 'sum', 'M': 'sum'}).reset_index()
            h = h[h['K'] > 0.001]
            h.columns = ['Ticker', 'Units', 'Cost Basis']
            st.dataframe(h.style.format({"Units": "{:.4f}", "Cost Basis": "${:,.2f}"}), use_container_width=True)
        else:
            st.info("No individual trade data (Column B='Data') found.")

    # --- TAB 3: FIFO CALCULATOR (CHRONOLOGICAL METHOD A) ---
    with tab_fifo:
        st.header("Interactive FIFO Sell Calculator")
        # Identify stocks from Data rows
        ticker_list = sorted(df_all[(df_all['A'] == "Trades") & (df_all['B'] == "Data")]['F'].unique())
        
        if ticker_list:
            sel_t = st.selectbox("Select Stock", ticker_list)
            
            # Reconstruct FIFO Queue from all historical trades
            s_df = df_all[(df_all['F'] == sel_t) & (df_all['A'] == "Trades") & (df_all['B'] == "Data")].copy()
            
            queue = []
            for _, row in s_df.iterrows():
                q, b = float(row['K']), float(row['M'])
                if q > 0: queue.append({'q': q, 'b': b})
                elif q < 0:
                    rem_s = abs(q)
                    while rem_s > 0 and queue:
                        if queue[0]['q'] <= rem_s: rem_s -= queue.pop(0)['q']
                        else:
                            queue[0]['q'] -= rem_s
                            rem_s = 0
            
            total_held = sum(item['q'] for item in queue)
            
            col_in1, col_in2 = st.columns(2)
            calc_mode = col_in1.radio("Sell By:", ["Units", "Percentage"])
            profit_goal = col_in2.number_input("Target Profit %", value=15.0)
            
            sell_amt = st.slider("Quantity", 0.0, float(total_held), step=0.01) if calc_mode == "Units" else \
                      (st.slider("Percentage", 0, 100, 20) / 100) * total_held

            if sell_amt > 0:
                # Calculate cost of the FIFO slice
                temp_qty, cost_slice = sell_amt, 0.0
                for lot in queue:
                    if temp_qty <= 0: break
                    taken = min(lot['q'], temp_qty)
                    cost_slice += (taken / lot['q']) * lot['b']
                    temp_qty -= taken
                
                target_val = cost_slice * (1 + (profit_goal / 100))
                
                st.divider()
                r1, r2 = st.columns(2)
                r1.success(f"### Target Sell Value: ${target_val:,.2f}")
                r1.caption(f"FIFO Cost Basis for these units: ${cost_slice:,.2f}")
                r2.info(f"### Remaining Units: {total_held - sell_amt:,.4f}")
                r2.caption(f"Remaining Portfolio Basis: ${sum(l['q'] for l in queue if l['q']>0)/max(1,sum(l['q'] for l in queue)) * sum(l['b'] for l in queue) - cost_slice:,.2f}")
        else:
            st.warning("No ticker data found. Ensure Column A='Trades' and Column B='Data'.")
else:
    st.error("Connection failed. Check Google Sheet permissions and worksheet names (FY24, FY25, FY26).")
