import streamlit as st
import pandas as pd
import numpy as np

# --- 1. DATA PREPARATION ---
def clean_and_stack_data(raw_data_dict):
    """
    Standardizes columns to A, B, C... M and cleans numeric values.
    raw_data_dict should be: {"FY24": df24, "FY25": df25, "FY26": df26}
    """
    combined_list = []
    col_names = list("ABCDEFGHIJKLM") # Matches your formula logic

    for year, df in raw_data_dict.items():
        if df is None or df.empty:
            continue
            
        # Ensure we only take first 13 columns and rename to letters
        df_clean = df.iloc[:, :13].copy()
        df_clean.columns = col_names
        df_clean['Source'] = year # Track which sheet data came from
        
        # CLEAN NUMERICS: Remove $, commas, and handle parentheses for negative numbers
        for col in ['F', 'G', 'H', 'I', 'K', 'M']:
            if col in df_clean.columns:
                df_clean[col] = (df_clean[col].astype(str)
                                 .str.replace(r'[$,\s]', '', regex=True)
                                 .str.replace(r'\((.*)\)', r'-\1', regex=True) # Handles (100) as -100
                                 )
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
        combined_list.append(df_clean)
    
    return pd.concat(combined_list, ignore_index=True) if combined_list else pd.DataFrame()

# --- 2. THE FORMULA ENGINE (SUMIFS REPLICATOR) ---
def get_metrics_logic(df):
    def s_if(target, conds):
        mask = pd.Series([True] * len(df))
        for col, val in conds.items():
            mask &= (df[col].astype(str).str.strip() == str(val))
        return df.loc[mask, target].sum()

    return {
        "inv_usd": s_if('M', {'A': 'Trades', 'B': 'Total', 'D': 'Stocks', 'E': 'USD'}),
        "inv_aud": s_if('M', {'A': 'Trades', 'B': 'Total', 'D': 'Stocks', 'E': 'AUD'}),
        "div_usd": s_if('F', {'A': 'Dividends', 'C': 'Total'}),
        "realized_stocks": (s_if('F', {'A': 'Realized & Unrealized Performance Summary', 'C': 'Stocks'}) + 
                            s_if('G', {'A': 'Realized & Unrealized Performance Summary', 'C': 'Stocks'}) + 
                            s_if('H', {'A': 'Realized & Unrealized Performance Summary', 'C': 'Stocks'}) + 
                            s_if('I', {'A': 'Realized & Unrealized Performance Summary', 'C': 'Stocks'}))
    }

# --- 3. FIFO ENGINE ---
def run_fifo_engine(df, ticker, target_sell_amount, is_pct, target_profit_pct):
    # Filter for granular trade data (Col B == "Data")
    trades = df[(df['A'] == 'Trades') & (df['B'] == 'Data') & (df['F_Symbol'] == ticker)].copy()
    
    queue = [] # Chronological Buy Lots
    for _, row in trades.sort_index().iterrows():
        qty, basis = float(row['K']), float(row['M'])
        if qty > 0: queue.append({'qty': qty, 'basis': basis})
        else:
            sell_rem = abs(qty)
            while sell_rem > 0 and queue:
                if queue[0]['qty'] <= sell_rem: sell_rem -= queue.pop(0)['qty']
                else:
                    queue[0]['qty'] -= sell_rem
                    sell_rem = 0
    
    total_held = sum(item['qty'] for item in queue)
    units_to_sell = (target_sell_amount / 100) * total_held if is_pct else target_sell_amount
    
    # Calculate Cost of the slice
    temp_units, cost_of_slice = units_to_sell, 0
    for lot in queue:
        if temp_units <= 0: break
        take = min(lot['qty'], temp_units)
        cost_of_slice += (take / lot['qty']) * lot['basis']
        temp_units -= take
        lot['qty'] -= take

    sell_price = cost_of_slice * (1 + (target_profit_pct / 100))
    # Remaining Basis calculation
    rem_basis = sum(l['basis'] * (l['qty'] / (l['qty'] + (take if l['qty']==0 else 0) + 0.000001)) for l in queue)
    
    return {"sell_price": sell_price, "rem_units": total_held - units_to_sell, "rem_basis": rem_basis, "total_held": total_held}

# --- 4. STREAMLIT APP ---
st.set_page_config(layout="wide")
st.title("My Portfolio App")

# --- DATA LOADING (REPLACE WITH YOUR GOOGLE SHEETS CONNECTION) ---
# Assuming you have df_fy24, df_fy25, df_fy26 loaded from your existing setup:
# raw_data = {"FY24": df_fy24, "FY25": df_fy25, "FY26": df_fy26}
# df_all = clean_and_stack_data(raw_data)

# NOTE: For FIFO, we need a helper column because 'F' is used for both Symbols and Profits
# In trades, F is Symbol. In Performance Summary, F is Profit.
if not df_all.empty:
    df_all['F_Symbol'] = df_all['F'] # Create a dedicated Symbol column
    
    tab1, tab2, tab3 = st.tabs(["📊 Metrics", "Portfolio Holdings", "🧮 FIFO Calculator"])

    with tab1:
        st.header("Financial Metrics")
        m26 = get_metrics_logic(df_all[df_all['Source'] == 'FY26'])
        m25 = get_metrics_logic(df_all[df_all['Source'] == 'FY25'])
        
        c1, c2 = st.columns(2)
        c1.metric("Lifetime Investment (USD)", f"${(m26['inv_usd'] + m25['inv_usd']):,.2f}")
        c2.metric("Total Realized Profit", f"${(m26['realized_stocks'] + m25['realized_stocks']):,.2f}")

    with tab2:
        st.header("Current Positions")
        # Logic to show table of non-zero units
        
    with tab3:
        st.header("FIFO Sell Calculator")
        unique_tickers = df_all[(df_all['A'] == 'Trades') & (df_all['B'] == 'Data')]['F'].unique()
        selected_ticker = st.selectbox("Select Stock", unique_tickers)
        
        mode = st.radio("Sell by:", ["Units", "Percentage"])
        state = run_fifo_engine(df_all, selected_ticker, 0, False, 0)
        
        if mode == "Units":
            amt = st.slider("Units", 0.0, state['total_held'], step=1.0)
        else:
            pct = st.slider("Percentage", 0, 100, 25)
            amt = pct
            
        profit_goal = st.number_input("Target Profit %", value=15.0)
        
        if st.button("Calculate"):
            res = run_fifo_engine(df_all, selected_ticker, amt, (mode == "Percentage"), profit_goal)
            st.success(f"### Target Sell Price: ${res['sell_price']:,.2f}")
            st.write(f"Remaining Units: {res['rem_units']:.2f}")
            st.write(f"Remaining Basis: ${res['rem_basis']:,.2f}")
