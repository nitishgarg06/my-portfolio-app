import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.set_page_config(layout="wide", page_title="Portfolio Alpha")

# 1. DATA LOADING
@st.cache_data(ttl=60)
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    def prep(s_name):
        df = conn.read(worksheet=s_name)
        if df is not None and not df.empty:
            df = df.iloc[:, :13] 
            df.columns = list("ABCDEFGHIJKLM")
            df['YearSource'] = s_name
            # Clean numeric cols
            for col in ['F', 'G', 'H', 'I', 'K', 'M']:
                df[col] = pd.to_numeric(df[col].astype(str).replace(r'[$,\s()]', '', regex=True), errors='coerce').fillna(0.0)
            return df
        return pd.DataFrame()
    
    frames = [prep("FY24"), prep("FY25"), prep("FY26")]
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)

df_all = load_data()

# 2. UI TABS
tab_summary, tab_holdings, tab_fifo = st.tabs(["📊 Summary", "Current Holdings", "🧮 FIFO Calculator"])

# (Summary and Holdings tabs remain as previously locked down)

with tab_fifo:
    st.header("FIFO Sell Calculator")

    if not df_all.empty:
        # --- ROBUST FILTERING ---
        # Normalize text to avoid space/case issues
        df_all['A_check'] = df_all['A'].astype(str).str.strip().str.upper()
        df_all['B_check'] = df_all['B'].astype(str).str.strip().str.upper()
        
        # Filter for rows where A is Trades and B is Data
        trade_data_rows = df_all[
            (df_all['A_check'] == "TRADES") & 
            (df_all['B_check'] == "DATA")
        ]

        # Extract symbols from Column F
        # We also filter out 0.0 or empty strings that might be in Col F
        ticker_list = sorted([
            str(x).strip() for x in trade_data_rows['F'].unique() 
            if str(x).strip() not in ['0.0', '0', '', 'nan', 'None']
        ])

        if ticker_list:
            sel_t = st.selectbox("Select Stock", ticker_list)
            
            # FIFO Calculation Logic
            s_df = trade_data_rows[trade_data_rows['F'].astype(str).str.strip() == sel_t].copy()
            
            # Reconstruct FIFO Queue
            queue = []
            for _, row in s_df.iterrows():
                q, b = float(row['K']), float(row['M'])
                if q > 0: queue.append({'q': q, 'b': b})
                elif q < 0:
                    rem = abs(q)
                    while rem > 0 and queue:
                        if queue[0]['q'] <= rem: rem -= queue.pop(0)['q']
                        else:
                            queue[0]['q'] -= rem
                            rem = 0
            
            total_held = sum(item['q'] for item in queue)
            
            # UI Slider
            col1, col2 = st.columns(2)
            profit_goal = col2.number_input("Target Profit %", value=15.0)
            amt = col1.slider("Units to Sell", 0.0, float(total_held), step=0.01)

            if amt > 0:
                temp_q, cost_sum = amt, 0.0
                for lot in queue:
                    if temp_q <= 0: break
                    take = min(lot['q'], temp_q)
                    cost_sum += (take / lot['q']) * lot['b']
                    temp_q -= take
                
                target_val = cost_sum * (1 + (profit_goal/100))
                st.success(f"### Target Sell Value: ${target_val:,.2f}")
                st.info(f"Remaining: {total_held - amt:,.4f} units")
        else:
            # DIAGNOSTIC TOOL (Displays only if list is empty)
            st.warning("Dropdown is empty. Let's find out why:")
            st.write("Found 'Trades' in Column A:", "TRADES" in df_all['A_check'].values)
            st.write("Found 'Data' in Column B:", "DATA" in df_all['B_check'].values)
            st.write("Unique values in Column F (Symbols):", df_all['F'].unique()[:10])
            st.info("Check if your ticker symbols are actually in Column F for the 'Data' rows.")
    else:
        st.error("No data loaded.")
