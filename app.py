import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.set_page_config(layout="wide", page_title="Portfolio Alpha")

# ==========================================
# 1. DATA LOADING (THE DUAL-INSTANCE FIX)
# ==========================================
@st.cache_data(ttl=600)
def load_all_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    def fetch(name):
        df = conn.read(worksheet=name)
        if df is not None and not df.empty:
            df = df.iloc[:, :13]
            df.columns = list("ABCDEFGHIJKLM")
            df['YearSource'] = name
            return df
        return pd.DataFrame()

    raw_data = pd.concat([fetch("FY24"), fetch("FY25"), fetch("FY26")], ignore_index=True)

    # --- BUCKET 1: SUMMARY DATA (Forced Numeric for Calculations) ---
    df_s = raw_data.copy()
    for col in ['F', 'G', 'H', 'I', 'K', 'M']:
        df_s[col] = pd.to_numeric(df_s[col].astype(str).replace(r'[$,\s()]', '', regex=True), errors='coerce').fillna(0.0)

    # --- BUCKET 2: FIFO DATA (Protects Column F Text, but needs H for Qty) ---
    df_f = raw_data.copy()
    # We now include 'H' so the calculator can read the Quantity
    for col in ['H', 'K', 'M']: 
        df_f[col] = pd.to_numeric(df_f[col].astype(str).replace(r'[$,\s()]', '', regex=True), errors='coerce').fillna(0.0)
    
    return df_s, df_f

df_summary, df_fifo = load_all_data()

# ==========================================
# 2. THE SUMMARY ENGINE (STRICT LOCKED LOGIC)
# ==========================================
def s_if(df_target, target_col, a=None, b=None, c=None, d=None, e=None):
    if df_target.empty: return 0.0
    mask = pd.Series([True] * len(df_target), index=df_target.index)
    if a: mask &= (df_target['A'].astype(str).str.strip() == a)
    if b: mask &= (df_target['B'].astype(str).str.strip() == b)
    if c: mask &= (df_target['C'].astype(str).str.strip() == c)
    if d: mask &= (df_target['D'].astype(str).str.strip() == d)
    if e: mask &= (df_target['E'].astype(str).str.strip() == e)
    return float(df_target.loc[mask, target_col].sum())

def get_realized_row(df_target, scope):
    p_st = s_if(df_target, 'F', a="Realized & Unrealized Performance Summary", c=scope)
    l_st = s_if(df_target, 'G', a="Realized & Unrealized Performance Summary", c=scope)
    p_lt = s_if(df_target, 'H', a="Realized & Unrealized Performance Summary", c=scope)
    l_lt = s_if(df_target, 'I', a="Realized & Unrealized Performance Summary", c=scope)
    return [p_st, l_st, p_lt, l_lt, (p_st + l_st + p_lt + l_lt)]

# ==========================================
# 3. UI RENDERING
# ==========================================
view_choice = st.sidebar.selectbox("Select Period", ["Lifetime", "FY26", "FY25", "FY24"])
if view_choice == "Lifetime":
    df_s_view, df_f_view = df_summary, df_fifo
else:
    df_s_view = df_summary[df_summary['YearSource'] == view_choice]
    df_f_view = df_fifo[df_fifo['YearSource'] == view_choice]

tab1, tab2, tab3 = st.tabs(["📊 Summary", "Current Holdings", "🧮 FIFO Calculator"])

with tab1:
    st.header(f"Summary: {view_choice}")
    
    # METRICS SECTION
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Investment (USD)", f"${s_if(df_s_view, 'M', a='Trades', b='Total', d='Stocks', e='USD'):,.2f}")
    c2.metric("Total Investment (AUD)", f"${s_if(df_s_view, 'M', a='Trades', b='Total', d='Stocks', e='AUD'):,.2f}")
    c3.metric("Funds Deposited (AUD)", f"${s_if(df_s_view, 'F', a='Deposits & Withdrawals', c='Total'):,.2f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Dividends (USD)", f"${s_if(df_s_view, 'F', a='Dividends', c='Total'):,.2f}")
    c4.metric("Dividends (AUD)", f"${s_if(df_s_view, 'F', a='Dividends', c='Total in AUD'):,.2f}")
    c5.metric("Withholding Tax (USD)", f"${s_if(df_s_view, 'F', a='Withholding Tax', c='Total'):,.2f}")
    c5.metric("Withholding Tax (AUD)", f"${s_if(df_s_view, 'F', a='Withholding Tax', c='Total in AUD'):,.2f}")

    # REALIZED TABLE SECTION
    st.divider()
    st.subheader("Realized Gains/Losses")
    real_data = {
        "Metric": ["S/T Profit", "S/T Loss", "L/T Profit", "L/T Loss", "Total"],
        "Stocks": get_realized_row(df_s_view, "Stocks"),
        "Forex": get_realized_row(df_s_view, "Forex"),
        "Total Assets": get_realized_row(df_s_view, "Total (All Assets)")
    }
    st.table(pd.DataFrame(real_data).set_index("Metric").style.format("${:,.2f}"))

with tab2:
    st.header("Open Positions")
    # Uses df_fifo so ticker names in Col F are visible
    h_data = df_f_view[(df_f_view['A'].astype(str).str.strip() == "Trades") & 
                       (df_f_view['B'].astype(str).str.strip() == "Data")]
    if not h_data.empty:
        h = h_data.groupby('F').agg({'K': 'sum', 'M': 'sum'}).reset_index()
        h = h[h['K'] > 0.001]
        h.columns = ['Ticker', 'Units', 'Cost Basis']
        st.dataframe(h.style.format({"Units": "{:.4f}", "Cost Basis": "${:,.2f}"}))

with tab3:
    st.header("🧮 FIFO Sell Calculator")
    
    # 1. Isolate individual trade data using the df_fifo bucket
    f_data = df_fifo[(df_fifo['A'].astype(str).str.strip().str.upper() == "TRADES") & 
                     (df_fifo['B'].astype(str).str.strip().str.upper() == "DATA")].copy()
    
    # Clean Tickers to ensure matches
    f_data['F_Clean'] = f_data['F'].astype(str).str.strip().str.upper()
    
    # --- FILTER FOR ACTIVE HOLDINGS ONLY ---
    active_tickers = []
    all_tickers_in_data = f_data['F_Clean'].unique()
    
    for t in all_tickers_in_data:
        if t in ['0.0', 'NAN', 'NONE', '', '0']: continue
        t_history = f_data[f_data['F_Clean'] == t]
        current_inv = []
        for _, r in t_history.iterrows():
            q = float(r['H']) # Using Column H for Quantity
            if q > 0: 
                current_inv.append(q)
            elif q < 0:
                rem = abs(q)
                while rem > 0 and current_inv:
                    if current_inv[0] <= rem:
                        rem -= current_inv.pop(0)
                    else:
                        current_inv[0] -= rem
                        rem = 0
        if sum(current_inv) > 0.001:
            active_tickers.append(t)
    
    ticker_list = sorted(active_tickers)
    
    if ticker_list:
        sel_t = st.selectbox("Select Active Stock to Analyze", ticker_list)
        
        # Pull chronological history for this stock
        s_hist = f_data[f_data['F_Clean'] == sel_t].copy()
        
        # RECONSTRUCT INVENTORY
        inventory = []
        s_hist['H'] = pd.to_numeric(s_hist['H'], errors='coerce').fillna(0.0)
        s_hist['M'] = pd.to_numeric(s_hist['M'], errors='coerce').fillna(0.0)

        for _, r in s_hist.iterrows():
            q, b = float(r['H']), float(r['M'])
            if q > 0.00001:  # BUY
                inventory.append({'qty': q, 'basis': b, 'price': b/q if q != 0 else 0})
            elif q < -0.00001:  # SELL
                qty_to_reduce = abs(q)
                while qty_to_reduce > 0.00001 and inventory:
                    if inventory[0]['qty'] <= qty_to_reduce:
                        qty_to_reduce -= inventory.pop(0)['qty']
                    else:
                        inventory[0]['qty'] -= qty_to_reduce
                        qty_to_reduce = 0
        
        total_held = sum(i['qty'] for i in inventory)
        total_basis = sum(i['basis'] for i in inventory)
        
        # --- UI: HOLDING BREAKDOWN ---
        with st.expander("📊 View Open Buy Lots (FIFO Queue)", expanded=True):
            if inventory and total_held > 0.0001:
                st.write(f"Confirmed Holdings: **{total_held:.4f} units**")
                lots_df = pd.DataFrame(inventory)
                lots_df.columns = ['Remaining Units', 'Total Cost Basis', 'Cost Price/Unit']
                st.dataframe(lots_df.style.format({
                    "Remaining Units": "{:.4f}", 
                    "Total Cost Basis": "${:,.2f}", 
                    "Cost Price/Unit": "${:,.2f}"
                }), use_container_width=True)

        # --- UI: CALCULATOR ---
        if total_held > 0:
            st.subheader("💰 Calculate Sell Target")
            sell_mode = st.radio("Select Sell Mode:", ["By Units", "By Percentage"], horizontal=True)
            
            c_in1, c_in2 = st.columns([2, 1])
            max_units = float(total_held)
            
            if sell_mode == "By Units":
                amt_to_sell = c_in1.slider("Units to Sell", 0.0, max_units, value=max_units/2, step=0.0001)
                percent_label = (amt_to_sell / max_units) * 100 if max_units > 0 else 0
            else:
                sell_percent = c_in1.slider("Percentage (%)", 0.0, 100.0, value=50.0, step=1.0)
                amt_to_sell = (sell_percent / 100) * max_units
                percent_label = sell_percent

            profit_goal = c_in2.number_input("Target Profit %", value=15.0)
            
            if amt_to_sell > 0:
                temp_sell_qty = amt_to_sell
                slice_cost_basis = 0.0
                calc_inv = [lot.copy() for lot in inventory]
                
                for lot in calc_inv:
                    if temp_sell_qty <= 0: break
                    take = min(lot['qty'], temp_sell_qty)
                    slice_cost_basis += (take / lot['qty']) * lot['basis']
                    temp_sell_qty -= take
                
                target_total = slice_cost_basis * (1 + (profit_goal/100))
                target_per_share = target_total / amt_to_sell
                
                st.divider()
                r1, r2, r3 = st.columns(3)
                r1.metric("Target Total Sale", f"${target_total:,.2f}")
                r2.metric("Target Price/Unit", f"${target_per_share:,.2f}")
                r3.metric("Profit Lead", f"${target_total - slice_cost_basis:,.2f}", delta=f"{profit_goal}%")
                
                # --- REMAINING STOCK PROFILE ---
                st.subheader("📋 Remaining Position Profile")
                rem_qty = max_units - amt_to_sell
                rem_basis = total_basis - slice_cost_basis
                rem_avg_price = rem_basis / rem_qty if rem_qty > 0 else 0
                
                p1, p2, p3 = st.columns(3)
                p1.metric("Remaining Units", f"{rem_qty:.4f}")
                p2.metric("New Avg. Cost Basis", f"${rem_avg_price:,.2f}")
                p3.metric("Remaining Investment", f"${rem_basis:,.2f}")

                st.caption(f"Selling {amt_to_sell:.4f} units ({percent_label:.1f}%). Cost basis of this slice: ${slice_cost_basis:,.2f}")
        else:
            st.warning(f"Calculated holding for {sel_t} is 0.")
    else:
        st.error("No active holdings found in individual trade data.")










