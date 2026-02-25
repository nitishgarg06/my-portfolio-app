import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.set_page_config(layout="wide", page_title="Portfolio App")
st.title("📈 Portfolio Management App")

# 1. DATA LOADING (Force A-M Column mapping)
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
    def get_and_prep(sheet_name):
        df = conn.read(worksheet=sheet_name)
        if df is not None and not df.empty:
            df = df.iloc[:, :13] 
            df.columns = list("ABCDEFGHIJKLM")
            df['YearSource'] = sheet_name
            for col in ['F', 'G', 'H', 'I', 'K', 'M']:
                df[col] = pd.to_numeric(df[col].astype(str).replace(r'[$,\s()]', '', regex=True), errors='coerce').fillna(0)
            return df
        return pd.DataFrame()

    df_all = pd.concat([get_and_prep("FY24"), get_and_prep("FY25"), get_and_prep("FY26")], ignore_index=True)
except Exception as e:
    st.error(f"Connection Error: {e}")
    st.stop()

if not df_all.empty:
    # --- SIDEBAR FILTER ---
    st.sidebar.header("Filter View")
    view_choice = st.sidebar.selectbox("Select Period", ["Lifetime", "FY26", "FY25", "FY24"])
    
    # Filter the dataframe based on selection
    if view_choice == "Lifetime":
        df_view = df_all
    else:
        df_view = df_all[df_all['YearSource'] == view_choice]

    # --- SUMIFS REPLICATION FUNCTION ---
    def s_if(df_target, target_col, a=None, b=None, c=None, d=None, e=None):
        mask = pd.Series([True] * len(df_target))
        if a: mask &= (df_target['A'].astype(str).str.strip() == a)
        if b: mask &= (df_target['B'].astype(str).str.strip() == b)
        if c: mask &= (df_target['C'].astype(str).str.strip() == c)
        if d: mask &= (df_target['D'].astype(str).str.strip() == d)
        if e: mask &= (df_target['E'].astype(str).str.strip() == e)
        return df_target.loc[mask, target_col].sum()

    # 2. TABS
    tab_summary, tab_holdings, tab_fifo = st.tabs(["📊 Summary", "Current Holdings", "🧮 FIFO Calculator"])

    with tab_summary:
        st.header(f"Performance Metrics: {view_choice}")
        
        # Row 1: Primary Investment Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Investment (USD)", f"${s_if(df_view, 'M', a='Trades', b='Total', d='Stocks', e='USD'):,.2f}")
        c2.metric("Total Investment (AUD)", f"${s_if(df_view, 'M', a='Trades', b='Total', d='Stocks', e='AUD'):,.2f}")
        c3.metric("Funds Deposited (AUD)", f"${s_if(df_view, 'F', a='Deposits & Withdrawals', c='Total'):,.2f}")

        # Row 2: Dividends & Tax
        c4, c5, c6 = st.columns(3)
        c4.metric("Dividends (USD)", f"${s_if(df_view, 'F', a='Dividends', c='Total'):,.2f}")
        c5.metric("Dividends (AUD)", f"${s_if(df_view, 'F', a='Dividends', c='Total in AUD'):,.2f}")
        c6.metric("Withholding Tax (USD)", f"${s_if(df_view, 'F', a='Withholding Tax', c='Total'):,.2f}")

        # Row 3: Realized Logic
        st.divider()
        st.subheader("Realized Performance Summary")
        
        def get_realized_group(scope):
            sp = s_if(df_view, 'F', a="Realized & Unrealized Performance Summary", c=scope)
            sl = s_if(df_view, 'G', a="Realized & Unrealized Performance Summary", c=scope)
            lp = s_if(df_view, 'H', a="Realized & Unrealized Performance Summary", c=scope)
            ll = s_if(df_view, 'I', a="Realized & Unrealized Performance Summary", c=scope)
            return sp, sl, lp, ll

        # Data for Stocks, Forex, and All Assets
        s_vals = get_realized_group("Stocks")
        f_vals = get_realized_group("Forex")
        a_vals = get_realized_group("Total (All Assets)")

        # Create a Comparison Table for Realized Metrics
        realized_data = {
            "Metric": ["S/T Profit", "S/T Loss", "L/T Profit", "L/T Loss", "Total"],
            "Stocks": [s_vals[0], s_vals[1], s_vals[2], s_vals[3], sum(s_vals)],
            "Forex": [f_vals[0], f_vals[1], f_vals[2], f_vals[3], sum(f_vals)],
            "Total Assets": [a_vals[0], a_vals[1], a_vals[2], a_vals[3], sum(a_vals)]
        }
        st.table(pd.DataFrame(realized_data).set_index("Metric").style.format("${:,.2f}"))

    with tab_holdings:
        st.header("Open Positions (All Time)")
        # Holdings MUST look at all historical data to be accurate
        trades_df = df_all[(df_all['A'] == "Trades") & (df_all['B'] == "Data")]
        if not trades_df.empty:
            holdings = trades_df.groupby('F').agg({'K': 'sum', 'M': 'sum'}).reset_index()
            holdings = holdings[holdings['K'] > 0.01] # Filter out sold stocks
            holdings.columns = ['Ticker', 'Current Units', 'Total Cost Basis']
            st.dataframe(holdings, use_container_width=True)
        else:
            st.info("No trade data found in Column A='Trades' and Column B='Data'.")

    with tab_fifo:
        st.header("Interactive FIFO Sell Calculator")
        # Pull unique tickers from Data rows
        ticker_options = df_all[(df_all['A'] == "Trades") & (df_all['B'] == "Data")]['F'].unique()
        
        if len(ticker_options) > 0:
            sel_ticker = st.selectbox("Select Stock", sorted(ticker_options))
            
            # FIFO Processing for selected ticker
            s_data = df_all[(df_all['F'] == sel_ticker) & (df_all['A'] == "Trades") & (df_all['B'] == "Data")].copy()
            total_units = s_data['K'].sum()
            
            # UI Inputs
            col_in1, col_in2 = st.columns(2)
            calc_mode = col_in1.radio("Sell By:", ["Units", "Percentage"])
            target_pct = col_in2.number_input("Target Profit %", value=15.0)
            
            if calc_mode == "Units":
                sell_amt = st.slider("Units to Sell", 0.0, float(total_units), step=0.01)
            else:
                sell_pct = st.slider("Percentage to Sell", 0, 100, 20)
                sell_amt = (sell_pct / 100) * total_units

            # FIFO Calculation
            if sell_amt > 0:
                queue = []
                for _, row in s_data.iterrows():
                    if row['K'] > 0: queue.append({'q': row['K'], 'b': row['M']})
                    else:
                        rem = abs(row['K'])
                        while rem > 0 and queue:
                            if queue[0]['q'] <= rem: rem -= queue.pop(0)['q']
                            else:
                                queue[0]['q'] -= rem
                                rem = 0
                
                # Calculate cost of units sold
                tmp_qty, cost_of_sale = sell_amt, 0
                for lot in queue:
                    if tmp_qty <= 0: break
                    taken = min(lot['q'], tmp_qty)
                    cost_of_sale += (taken / lot['q']) * lot['b']
                    tmp_qty -= taken
                
                target_val = cost_of_sale * (1 + (target_pct / 100))
                
                st.divider()
                st.success(f"### Target Sell Value: ${target_val:,.2f}")
                st.write(f"Calculated Cost Basis for these units: ${cost_of_sale:,.2f}")
                st.info(f"Remaining Units if sold: {total_units - sell_amt:,.4f}")
        else:
            st.warning("Ensure Column A contains 'Trades' and Column B contains 'Data' to use the calculator.")

else:
    st.error("Data could not be loaded. Check your Google Sheet connection and worksheet names.")
