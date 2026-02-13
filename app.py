import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- 1. SETUP & STYLE ---
st.set_page_config(page_title="Wealth Terminal Pro", layout="wide", page_icon="🏦")

st.markdown("""
    <style>
    .section-header {
        font-size: 1.4rem; font-weight: 700; color: #3b82f6;
        margin-top: 30px; border-bottom: 1px solid #334155;
        padding-bottom: 5px;
    }
    </style>
    """, unsafe_allow_stdio=True)

# --- 2. CONNECTION & PARSER ---
conn = st.connection("gsheets", type=GSheetsConnection)

def parse_ibkr_grid(df):
    sections = {}
    if df is None or df.empty: return sections
    df = df.astype(str).replace('nan', '')
    for name in df.iloc[:, 0].unique():
        if not name or name in ['Statement', 'Field Name']: continue
        sec_df = df[df.iloc[:, 0] == name]
        h_row = sec_df[sec_df.iloc[:, 1] == 'Header']
        d_rows = sec_df[sec_df.iloc[:, 1] == 'Data']
        if not h_row.empty and not d_rows.empty:
            raw_cols = h_row.iloc[0, 2:].tolist()
            cols = [c if (c and c.strip()) else f"Blank_{i}" for i, c in enumerate(raw_cols)]
            data = d_rows.iloc[:, 2:2+len(cols)]
            data.columns = cols
            sections[name] = data.loc[:, ~data.columns.str.startswith('Blank_')]
    return sections

# --- 3. DATA INGESTION ---
tabs = ["FY24", "FY25", "FY26"]
fy_data_map = {}
all_trades = []

for tab in tabs:
    try:
        raw_grid = conn.read(worksheet=tab, ttl=0, header=None)
        parsed = parse_ibkr_grid(raw_grid)
        if parsed:
            fy_data_map[tab] = parsed
            if "Trades" in parsed:
                t = parsed["Trades"]
                if 'DataDiscriminator' in t.columns:
                    t = t[t['DataDiscriminator'] == 'Order']
                all_trades.append(t)
    except: continue

# --- 4. FIFO ENGINE ---
df_lots = pd.DataFrame()
if all_trades:
    try:
        trades = pd.concat(all_trades, ignore_index=True)
        trades['Quantity'] = pd.to_numeric(trades['Quantity'], errors='coerce').fillna(0)
        trades['T. Price'] = pd.to_numeric(trades['T. Price'], errors='coerce').fillna(0)
        trades['Date/Time'] = pd.to_datetime(trades['Date/Time'].str.split(',').str[0], errors='coerce')
        trades = trades.dropna(subset=['Date/Time']).sort_values('Date/Time')

        for tkr, dt in [('NVDA', '2024-06-10'), ('SMCI', '2024-10-01')]:
            trades.loc[(trades['Symbol'] == tkr) & (trades['Date/Time'] < dt), 'Quantity'] *= 10
            trades.loc[(
