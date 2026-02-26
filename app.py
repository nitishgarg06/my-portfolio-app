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

    # --- BUCKET 2: FIFO DATA (Protects Column F Text) ---
    df_f = raw_data.copy()
    for col in ['K', 'M']: # Only force Units and Total Basis to numbers
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
    
    # 1. Filter the 'Text-Safe' bucket for individual trade data
    f_data = df_fifo[(df_fifo['A'].astype(str).str.strip() == "Trades") & 
                     (df_fifo['B'].astype(str).str.strip() == "Data")]
    
    ticker_list = sorted([str(x).strip() for x in f_data['F'].unique() if str(x).strip() not in ['0.0', 'nan', 'None', '']])
    
    if ticker_list:
        sel_t = st.selectbox("Select Stock to Analyze", ticker_list)
        
        # Pull chronological history for this stock
        s_hist = f_data[f_data['F'].astype(str).str.strip() == sel_t].copy()
        
        # Build the FIFO Queue
        queue = []
        for _, r in s_hist.iterrows():
            q, b = float(r['K']), float(r['M'])
            if q > 0: 
                queue.append({'qty': q, 'basis': b, 'price': b/q if q != 0 else 0})
            elif q < 0:
                rem = abs(q)
                while rem > 0 and queue:
                    if queue[0]['qty'] <= rem:
                        rem -= queue.pop(0)['qty']
                    else:
                        queue[0]['qty'] -= rem
                        rem = 0
        
        total_held = sum(i['qty'] for i in queue)
        
        # --- UI: HOLDING BREAKDOWN ---
        with st.expander("View Current Open Lots"):
            if queue and total_held > 0:
                lots_df = pd.DataFrame(queue)
                lots_df.columns = ['Remaining Units', 'Total Cost Basis', 'Price per Unit']
                st.table(lots_df.style.format({"Remaining Units": "{:.4f}", "Total Cost Basis": "${:,.2f}", "Price per Unit": "${:,.2f}"}))
            else:
                st.write("No open lots found for this ticker (Position may be closed).")

        # --- UI: CALCULATOR ---
        st.subheader("Simulate a Sale")
        
        if total_held > 0:
            c_in1, c_in2 = st.columns([2, 1])
            
            # THE FIX: Ensure max_value is greater than 0.0 and handle float conversion safely
            max_val = float(total_held)
            amt = c_in1.slider("Units to Sell", 0.0, max_val, step=0.01) if max_val > 0 else 0.0
            profit_goal = c_in2.number_input("Target Profit %", value=15.0)
            
            if amt > 0:
                # Calculate cost of the FIFO slice
                temp_q, cost_sum = amt, 0.0
                for lot in queue:
                    if temp_q <= 0: break
                    take = min(lot['qty'], temp_q)
                    cost_sum += (take / lot['qty']) * lot['basis']
                    temp_q -= take
                
                target_val = cost_sum * (1 + (profit_goal/100))
                price_per_share = target_val / amt
                
                st.divider()
                res1, res2, res3 = st.columns(3)
                res1.metric("Target Total Value", f"${target_val:,.2f}")
                res2.metric("Target Price / Share", f"${price_per_share:,.2f}")
                res3.metric("Est. Net Profit", f"${target_val - cost_sum:,.2f}", delta=f"{profit_goal}%")
                
                st.progress(amt / max_val)
                st.caption(f"Selling { (amt/max_val)*100 :.1f}% of holdings. Cost basis for this slice: ${cost_sum:,.2f}")
        else:
            st.warning(f"You currently hold 0 units of {sel_t} according to the trade history.")

    else:
        st.error("No valid tickers identified in Column F.")

