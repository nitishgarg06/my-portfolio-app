import streamlit as st
import pandas as pd

# --- SECTION 1: THE FORMULA ENGINE (SUMIFS) ---
def get_metrics_from_logic(df):
    """Replicates your specific SUMIFS logic for each sheet."""
    def s_if(target, conds):
        mask = pd.Series([True] * len(df))
        for col, val in conds.items():
            if col in df.columns:
                mask &= (df[col] == val)
        return df.loc[mask, target].sum() if target in df.columns else 0

    return {
        "inv_usd": s_if('M', {'A': 'Trades', 'B': 'Total', 'D': 'Stocks', 'E': 'USD'}),
        "inv_aud": s_if('M', {'A': 'Trades', 'B': 'Total', 'D': 'Stocks', 'E': 'AUD'}),
        "div_usd": s_if('F', {'A': 'Dividends', 'C': 'Total'}),
        "div_aud": s_if('F', {'A': 'Dividends', 'C': 'Total in AUD'}),
        "realized_stocks_total": (
            s_if('F', {'A': 'Realized & Unrealized Performance Summary', 'C': 'Stocks'}) +
            s_if('G', {'A': 'Realized & Unrealized Performance Summary', 'C': 'Stocks'}) +
            s_if('H', {'A': 'Realized & Unrealized Performance Summary', 'C': 'Stocks'}) +
            s_if('I', {'A': 'Realized & Unrealized Performance Summary', 'C': 'Stocks'})
        ),
        "deposits_aud": s_if('F', {'A': 'Deposits & Withdrawals', 'C': 'Total'})
    }

# --- SECTION 2: THE FIFO ENGINE ---
def run_fifo_engine(df, ticker, target_sell_amount, is_pct, target_profit_pct):
    """Calculates Sell Value and Remaining Summary using FIFO logic."""
    # Filter for granular trade data only (Method A)
    trades = df[(df['A'] == 'Trades') & (df['B'] == 'Data') & (df['F'] == ticker)].copy()
    
    # Build Buy Queue
    queue = []
    for _, row in trades.iterrows():
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
    
    # Calculate Cost of the slice to be sold
    temp_units, cost_of_slice = units_to_sell, 0
    for lot in queue:
        if temp_units <= 0: break
        take = min(lot['qty'], temp_units)
        cost_of_slice += (take / lot['qty']) * lot['basis']
        temp_units -= take
        lot['qty'] -= take # Update for remaining summary

    sell_price = cost_of_slice * (1 + (target_profit_pct / 100))
    rem_basis = sum(l['basis'] * (l['qty'] / (l['qty']+0.00001)) for l in queue) # Simplified
    
    return {
        "sell_price": sell_price,
        "rem_units": total_held - units_to_sell,
        "rem_basis": rem_basis,
        "total_held": total_held
    }

# --- SECTION 3: APP UI & TABS ---
st.set_page_config(page_title="My Portfolio App", layout="wide")

# (Data loading logic from your connection goes here - producing df_all)
# Example columns mapping: A=Activity, B=Type, C=Description, D=Category, E=Currency, F=Ticker... M=Basis

tab_metrics, tab_holdings, tab_fifo = st.tabs(["📊 Metrics", "Current Holdings", "🧮 FIFO Calculator"])

with tab_metrics:
    st.header("Portfolio Performance Metrics")
    # Applying formulas to different year filters
    metrics_26 = get_metrics_from_logic(df_all[df_all['Source'] == 'FY26'])
    metrics_25 = get_metrics_from_logic(df_all[df_all['Source'] == 'FY25'])
    
    col1, col2 = st.columns(2)
    col1.metric("Lifetime Investment (USD)", f"${(metrics_26['inv_usd'] + metrics_25['inv_usd']):,.2f}")
    col2.metric("Total Realized Stocks", f"${(metrics_26['realized_stocks_total'] + metrics_25['realized_stocks_total']):,.2f}")

with tab_holdings:
    st.header("Current Portfolio Snapshot")
    # Generate unique list of tickers from 'Trades' in Col F
    tickers = df_all[df_all['A'] == 'Trades']['F'].unique()
    # (Display table logic for units held)

with tab_fifo:
    st.header("Interactive FIFO Sell Calculator")
    
    ticker_choice = st.selectbox("Select Stock", df_all[df_all['A'] == 'Trades']['F'].unique())
    mode = st.radio("Calculation Basis", ["Units", "Percentage"])
    
    # Get current state for this stock
    state = run_fifo_engine(df_all, ticker_choice, 0, False, 0)
    
    if mode == "Units":
        amt = st.slider("Units to Sell", 0.0, state['total_held'], step=1.0)
    else:
        amt = st.slider("Percentage to Sell", 0, 100, 25)
        
    profit_pct = st.number_input("Target Profit %", value=15.0)
    
    if st.button("Calculate Sell Value"):
        res = run_fifo_engine(df_all, ticker_choice, amt, (mode == "Percentage"), profit_pct)
        
        st.success(f"### Target Sell Price: ${res['sell_price']:,.2f}")
        
        st.subheader("Remaining Stock Summary")
        c1, c2 = st.columns(2)
        c1.metric("Remaining Units", f"{res['rem_units']:.2f}")
        c2.metric("Remaining Cost Basis", f"${res['rem_basis']:,.2f}")
