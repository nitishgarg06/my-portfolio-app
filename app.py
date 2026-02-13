import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf

# --- MINIMAL CONFIG ---
st.set_page_config(page_title="IBKR Tracker", layout="wide")

# --- DATA CONNECTION ---
conn = st.connection("gsheets", type=GSheetsConnection)

def parse_ibkr(df):
    """The simplest possible parser to avoid errors."""
    sections = {}
    if df is None: return sections
    df = df.astype(str)
    for name in df.iloc[:, 0].unique():
        if name in ['nan', 'Statement', '']: continue
        sec_df = df[df.iloc[:, 0] == name]
        # Find the row where column 1 is 'Header'
        h_idx = sec_df[sec_df.iloc[:, 1] == 'Header'].index
        d_idx = sec_df[sec_df.iloc[:, 1] == 'Data'].index
        if not h_idx.empty:
            cols = sec_df.loc[h_idx[0]].tolist()[2:]
            data = sec_df.loc[d_idx].iloc[:, 2:]
            # Only keep columns that have names
            valid_cols = [c for c in cols if c and c != 'nan']
            data = data.iloc[:, :len(valid_cols)]
            data.columns = valid_cols
            sections[name] = data
    return sections

# --- MAIN LOGIC ---
st.title("🏦 My IBKR Wealth Terminal")

tabs = ["FY24", "FY25", "FY26"]
all_trades = []

# Try to load each tab one by one
for t in tabs:
    try:
        # header=None is safer for messy IBKR sheets
        raw = conn.read(worksheet=t, ttl=0, header=None)
        parsed = parse_ibkr(raw)
        if "Trades" in parsed:
            trades = parsed["Trades"]
            # Fix column names if IBKR added spaces
            trades.columns = trades.columns.str.strip()
            all_trades.append(trades)
    except Exception as e:
        st.sidebar.warning(f"Could not read {t}: {e}")

if all_trades:
    df = pd.concat(all_trades)
    # Filter for actual trades (Orders)
    if 'DataDiscriminator' in df.columns:
        df = df[df['DataDiscriminator'] == 'Order']
    
    # 1. Row Index Fix (Starting from 1)
    df.index = range(1, len(df) + 1)
    
    # 2. Basic Display
    st.header("1. Current Holdings")
    st.dataframe(df, use_container_width=True)

    # 3. Simple Calculator (Check if this part is the crash point)
    st.divider()
    st.header("🧮 Calculator")
    syms = df['Symbol'].unique()
    target_stock = st.selectbox("Pick a stock", syms)
    
    st.write(f"Showing trades for {target_stock}")
    st.table(df[df['Symbol'] == target_stock])

else:
    st.error("No data found. Please check your Tab Names in Google Sheets (must be FY24, FY25, FY26).")
