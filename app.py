import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.set_page_config(layout="wide", page_title="Portfolio App")
st.title("📈 Portfolio Management App")

# Initialize df_all
df_all = pd.DataFrame()

# 1. DATA LOADING
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
    
    def get_and_prep(sheet_name):
        df = conn.read(worksheet=sheet_name)
        if df is not None and not df.empty:
            df = df.iloc[:, :13] # Columns A through M
            df.columns = list("ABCDEFGHIJKLM")
            df['YearSource'] = sheet_name
            # Clean numeric columns
            for col in ['F', 'G', 'H', 'I', 'K', 'M']:
                df[col] = pd.to_numeric(df[col].astype(str).replace(r'[$,\s()]', '', regex=True), errors='coerce').fillna(0)
            return df
        return pd.DataFrame()

    df_all = pd.concat([get_and_prep("FY24"), get_and_prep("FY25"), get_and_prep("FY26")], ignore_index=True)
except Exception as e:
    st.error(f"Connection Error: {e}")

if not df_all.empty:
    # --- HELPER: SUMIFS REPLICATION ---
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
        st.header("Lifetime Performance Metrics")
        
        # Calculate totals across all years
        inv_usd = s_if(df_all, 'M', a="Trades", b="Total", d="Stocks", e="USD")
        inv_aud = s_if(df_all, 'M', a="Trades", b="Total", d="Stocks", e="AUD")
        div_usd = s_if(df_all, 'F', a="Dividends", c="Total")
        div_aud = s_if(df_all, 'F', a="Dividends", c="Total in AUD")
        tax_usd = s_if(df_all, 'F', a="Withholding Tax", c="Total")
        tax_aud = s_if(df_all, 'F', a="Withholding Tax", c="Total in AUD")
        depo_aud = s_if(df_all, 'F', a="Deposits & Withdrawals", c="Total")

        # Realized performance
        def get_realized(scope_label):
            sp = s_if(df_all, 'F', a="Realized & Unrealized Performance Summary", c=scope_label)
            sl = s_if(df_all, 'G', a="Realized & Unrealized Performance Summary", c=scope_label)
            lp = s_if(df_all, 'H', a="Realized & Unrealized Performance Summary", c=scope_label)
            ll = s_if(df_all, 'I', a="Realized & Unrealized Performance Summary", c=scope_label)
            return sp, sl, lp, ll

        r_st_p, r_st_l, r_lt_p, r_lt_l = get_realized("Total (All Assets)")
        s_st_p, s_st_l, s_lt_p, s_lt_l = get_realized("Stocks")
        f_st_p, f_st_l, f_lt_p, f_lt_l = get_realized("Forex")

        # Metrics Grid
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Investment (USD)", f"${inv_usd:,.2f}")
        c1.metric("Dividends (USD)", f"${div_usd:,.2f}")
        c1.metric("Withholding Tax (USD)", f"${tax_usd:,.2f}")
        
        c2.metric("Total Investment (AUD)", f"${inv_aud:,.2f}")
        c2.metric("Dividends (AUD)", f"${div_aud:,.2f}")
        c2.metric("Withholding Tax (AUD)", f"${tax_aud:,.2f}")
        
        c3.metric("Funds Deposited (AUD)", f"${depo_aud:,.2f}")
        
        st.divider()
        st.subheader("Realized Gains/Losses")
        col_st, col_lt, col_tot = st.columns(3)
        
        with col_st:
            st.write("**Short Term**")
            st.write(f"Stocks: ${s_st_p + s_st_l:,.2f}")
            st.write(f"Forex: ${f_st_p + f_st_l:,.2f}")
            st.write(f"Total: ${r_st_p + r_st_l:,.2f}")

        with col_lt:
            st.write("**Long Term**")
            st.write(f"Stocks: ${s_lt_p + s_lt_l:,.2f}")
            st.write(f"Forex: ${f_lt_p + f_lt_l:,.2f}")
            st.write(f"Total: ${r_lt_p + r_lt_l:,.2f}")

        with col_tot:
            st.write("**Combined Total**")
            st.write(f"Stocks Realized: ${s_st_p + s_st_l + s_lt_p + s_lt_l:,.2f}")
            st.write(f"Forex Realized: ${f_st_p + f_st_l + f_lt_p + f_lt_l:,.2f}")
            st.write(f"Grand Total: ${s_st_p + s_st_l + s_lt_p + s_lt_l + f_st_p + f_st_l + f_lt_p + f_lt_l:,.2f}")

    with tab_holdings:
        st.header("Current Open Positions")
        # Filter only Data rows for Trades to calculate current units
        trades_df = df_all[(df_all['A'] == "Trades") & (df_all['B'] == "Data")]
        if not trades_df.empty:
            summary = trades_df.groupby('F').agg({'K': 'sum', 'M': 'sum'}).reset_index()
            summary = summary[summary['K'] > 0.001] # Hide closed positions
            summary.columns = ['Ticker', 'Total Units', 'Total Basis']
            st.dataframe(summary, use_container_width=True)
        else:
            st.warning("No individual trade data found to calculate holdings.")

    with tab_fifo:
        st.header("FIFO Sell Calculator")
        # Ensure we are pulling symbols from individual trade rows
        calc_stocks = df_all[(df_all['A'] == "Trades") & (df_all['B'] == "Data")]['F'].unique()
        
        if len(calc_stocks) > 0:
            ticker = st.selectbox("Select Stock", sorted(calc_stocks))
            
            # Get data for this stock
            s_data = df_all[(df_all['A'] == "Trades") & (df_all['B'] == "Data") & (df_all['F'] == ticker)].copy()
            total_held = s_data['K'].sum()
            
            mode = st.radio("Sell by:", ["Units", "Percentage"])
            if mode == "Units":
                amt = st.slider("Select Units", 0.0, float(total_held), step=0.01)
                units_to_sell = amt
            else:
                pct = st.slider("Select Percentage", 0, 100, 25)
                units_to_sell = (pct / 100) * total_held
            
            profit_goal = st.number_input("Target Profit %", value=15.0)
            
            # FIFO Logic calculation
            if units_to_sell > 0:
                # Build queue of buys
                queue = []
                for _, row in s_data.iterrows():
                    if row['K'] > 0: queue.append({'q': row['K'], 'b': row['M']})
                    else: # Subtract sells from queue
                        rem_s = abs(row['K'])
                        while rem_s > 0 and queue:
                            if queue[0]['q'] <= rem_s: rem_s -= queue.pop(0)['q']
                            else:
                                queue[0]['q'] -= rem_s
                                rem_s = 0
                
                # Calculate cost of units to be sold
                temp_u, cost_slice = units_to_sell, 0
                for lot in queue:
                    if temp_u <= 0: break
                    take = min(lot['q'], temp_u)
                    cost_slice += (take / lot['q']) * lot['b']
                    temp_u -= take
                
                target_val = cost_slice * (1 + (profit_goal / 100))
                
                st.success(f"**Target Sell Value:** ${target_val:,.2f}")
                st.info(f"**Remaining Units:** {total_held - units_to_sell:,.4f}")
        else:
            st.error("No valid trade symbols found for the calculator.")
else:
    st.warning("Dataframe is empty. Please check Google Sheet worksheet names and column layout.")
