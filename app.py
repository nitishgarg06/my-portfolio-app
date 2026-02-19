import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

# --- 1. THE ULTIMATE DATA SCRUBBER ---
def clean_numeric(val):
    if val is None or pd.isna(val) or str(val).strip() == '': return 0.0
    s = str(val).strip().replace('$', '').replace(',', '')
    if '(' in s and ')' in s: s = '-' + s.replace('(', '').replace(')', '')
    try: return float(s)
    except: return 0.0

def find_trades_anywhere(df):
    """Scans the entire sheet for rows that look like stock trades."""
    # Look for the 'Trades' marker IBKR uses
    potential_trades = df[df.iloc[:, 0].str.contains('Trades', na=False, case=False)]
    if potential_trades.empty:
        return pd.DataFrame()
    
    # Identify the header row within the 'Trades' section
    header_idx = potential_trades[potential_trades.iloc[:, 1] == 'Header'].index
    data_indices = potential_trades[potential_trades.iloc[:, 1] == 'Data'].index
    
    if not header_idx.empty and not data_indices.empty:
        cols = potential_trades.loc[header_idx[0]].tolist()[2:]
        data = df.loc[data_indices].iloc[:, 2:2+len(cols)]
        data.columns = [c for c in cols if c]
        return data
    return pd.DataFrame()

# --- 2. THE APP BODY ---
st.title("🏦 Wealth Terminal Pro")
curr_date = datetime.now().strftime('%d %b %Y')

try:
    conn = st.connection("gsheets", type=GSheetsConnection)
    tabs = ["FY24", "FY25", "FY26"]
    
    # 3. TOP LEVEL METRICS (Direct from Summary)
    sel_fy = st.selectbox("Financial Year View", tabs, index=len(tabs)-1)
    raw_data = conn.read(worksheet=sel_fy, ttl=0)
    
    if raw_data is not None:
        # Emergency Realized P/L Scan
        perf_rows = raw_data[raw_data.iloc[:, 0].str.contains('Realized & Unrealized Performance Summary', na=False)]
        realized_total = 0.0
        if not perf_rows.empty:
            # Find the 'Realized Total' column and sum the 'Data' rows
            h_row = perf_rows[perf_rows.iloc[:, 1] == 'Header'].iloc[0].tolist()
            try:
                rt_idx = [i for i, x in enumerate(h_row) if 'Realized Total' in str(x)][0]
                d_rows = perf_rows[perf_rows.iloc[:, 1] == 'Data']
                realized_total = d_rows.iloc[:, rt_idx].apply(clean_numeric).sum()
            except: pass

        st.metric("Net Realized P/L", f"${realized_total:,.2f}")
        
        # 4. TRADES & HOLDINGS ENGINE
        trades = find_trades_anywhere(raw_data)
        
        if not trades.empty:
            # Clean Trade Data
            trades.columns = trades.columns.str.strip()
            trades['q_v'] = trades['Quantity'].apply(clean_numeric) if 'Quantity' in trades.columns else 0.0
            trades['p_v'] = trades['T. Price'].apply(clean_numeric) if 'T. Price' in trades.columns else 0.0
            trades['dt_v'] = pd.to_datetime(trades['Date/Time'].str.split(',').str[0], errors='coerce')
            
            # Simple FIFO Aggregate
            holdings = []
            for sym in trades['Symbol'].unique():
                q_sum = trades[trades['Symbol'] == sym]['q_v'].sum()
                if q_sum > 0.01:
                    avg_p = trades[(trades['Symbol'] == sym) & (trades['q_v'] > 0)]['p_v'].mean()
                    holdings.append({'Ticker': sym, 'Units': q_sum, 'Avg Cost': avg_p})
            
            if holdings:
                df_h = pd.DataFrame(holdings)
                df_h['Total Basis'] = df_h['Units'] * df_h['Avg Cost']
                df_h.index = range(1, len(df_h) + 1)
                
                st.subheader(f"Current Holdings (as of {curr_date})")
                st.dataframe(df_h.style.format({"Units": "{:.2f}", "Avg Cost": "${:.2f}", "Total Basis": "${:.2f}"}), use_container_width=True)
            else:
                st.info("No active holdings found in your trade history.")
        else:
            st.error("⚠️ The app cannot find the 'Trades' section in your Google Sheet.")
            st.write("Please ensure your IBKR CSV export was pasted into the sheet correctly.")
            st.download_button("Download Raw Sheet for Inspection", raw_data.to_csv(), "debug_sheet.csv")

except Exception as e:
    st.error(f"Something went wrong: {e}")
