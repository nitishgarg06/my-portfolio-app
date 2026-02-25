import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

# 1. SETUP
st.set_page_config(layout="wide")
st.title("My Portfolio App")

# Initialize df_all as an empty DataFrame to prevent NameError
df_all = pd.DataFrame()

# 2. DATA LOADING ENGINE
try:
    # Establish connection
    conn = st.connection("gsheets", type=GSheetsConnection)
    
    def get_and_prep(sheet_name):
        # Read the specific worksheet
        df = conn.read(worksheet=sheet_name)
        if df is not None and not df.empty:
            # Force first 13 columns to A-M
            df = df.iloc[:, :13]
            df.columns = list("ABCDEFGHIJKLM")
            df['YearSource'] = sheet_name
            # Clean numbers: remove $, commas, and handle ()
            for col in ['F', 'G', 'H', 'I', 'K', 'M']:
                df[col] = pd.to_numeric(df[col].astype(str).replace(r'[$,\s()]', '', regex=True), errors='coerce').fillna(0)
            return df
        return pd.DataFrame()

    # Combine all years
    df_24 = get_and_prep("FY24")
    df_25 = get_and_prep("FY25")
    df_26 = get_and_prep("FY26")
    
    df_all = pd.concat([df_24, df_25, df_26], ignore_index=True)

except Exception as e:
    st.error(f"Data Load Error: {e}")
    st.info("Check if worksheet names (FY24, FY25, FY26) match your Google Sheet exactly.")

# 3. LOGIC ENGINES (Only run if data exists)
if not df_all.empty:
    
    def get_metrics(df):
        def s_if(target, a_val, b_val=None, c_val=None):
            mask = (df['A'].astype(str).str.strip() == a_val)
            if b_val: mask &= (df['B'].astype(str).str.strip() == b_val)
            if c_val: mask &= (df['C'].astype(str).str.strip() == c_val)
            # For Investment: Col D=Stocks, E=USD
            if a_val == "Trades" and b_val == "Total":
                mask &= (df['D'].astype(str).str.strip() == "Stocks")
                mask &= (df['E'].astype(str).str.strip() == "USD")
            return df.loc[mask, target].sum()

        return {
            "inv_usd": s_if('M', "Trades", "Total"),
            "realized": (s_if('F', "Realized & Unrealized Performance Summary", "Stocks") +
                         s_if('G', "Realized & Unrealized Performance Summary", "Stocks") +
                         s_if('H', "Realized & Unrealized Performance Summary", "Stocks") +
                         s_if('I', "Realized & Unrealized Performance Summary", "Stocks"))
        }

    # 4. UI TABS
    tab1, tab2, tab3 = st.tabs(["📊 Metrics", "Portfolio Holdings", "🧮 FIFO Calculator"])

    with tab1:
        m26 = get_metrics(df_all[df_all['YearSource'] == "FY26"])
        m25 = get_metrics(df_all[df_all['YearSource'] == "FY25"])
        st.metric("Lifetime Investment (USD)", f"${(m26['inv_usd'] + m25['inv_usd']):,.2f}")

    with tab3:
        st.header("FIFO Sell Calculator")
        # Filter for unique tickers in Col F where it's a 'Trades' + 'Data' row
        tickers = df_all[(df_all['A'] == "Trades") & (df_all['B'] == "Data")]['F'].unique()
        selected_ticker = st.selectbox("Select Stock", sorted(tickers))
        
        # Add your slider and profit inputs here...
        st.write(f"Analyzing {selected_ticker}...")

else:
    st.warning("No data found in the sheets. Please check your Google Sheet connection.")
