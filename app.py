import streamlit as st
import pandas as pd

# --- TOPLINE CALCULATION ENGINE ---
def get_metrics(df_trades, df_perf, current_fy):
    # 1. Helper to clean data
    def cl(x): return pd.to_numeric(str(x).replace('$','').replace(',','').replace('(','-').replace(')',''), errors='coerce').fillna(0)

    # 2. Filter FY vs Lifetime
    fy_trades = df_trades[df_trades['FY'] == current_fy]
    fy_perf = df_perf[df_perf['FY'] == current_fy] if not df_perf.empty else pd.DataFrame()

    # 3. COMMISSION SPLITS (Stock vs Forex)
    # FY Split
    fy_stock_comm = fy_trades[fy_trades['Symbol'].str.len() <= 5]['Comm/Fee'].apply(cl).sum() # Assumes Stocks are short tickers
    fy_forex_comm = fy_trades[fy_trades['Symbol'].str.len() > 5]['Comm/Fee'].apply(cl).sum()
    
    # Lifetime Split
    lt_stock_comm = df_trades[df_trades['Symbol'].str.len() <= 5]['Comm/Fee'].apply(cl).sum()
    lt_forex_comm = df_trades[df_trades['Symbol'].str.len() > 5]['Comm/Fee'].apply(cl).sum()

    # 4. REALIZED P/L SPLITS
    # FY Split
    fy_s_pnl = fy_perf[fy_perf['Category'].str.contains('Stock|Equity', na=False)]['Realized Total'].apply(cl).sum()
    fy_f_pnl = fy_perf[fy_perf['Category'].str.contains('Forex|Cash|Interest', na=False)]['Realized Total'].apply(cl).sum()
    
    # Lifetime Split
    lt_s_pnl = df_perf[df_perf['Category'].str.contains('Stock|Equity', na=False)]['Realized Total'].apply(cl).sum()
    lt_f_pnl = df_perf[df_perf['Category'].str.contains('Forex|Cash|Interest', na=False)]['Realized Total'].apply(cl).sum()

    # 5. INVESTMENT
    fy_invest = (fy_trades[fy_trades['Qty'] > 0]['Qty'] * fy_trades[fy_trades['Qty'] > 0]['Price']).sum()
    # Lifetime Investment = Total Cost of Current Holdings (from FIFO)
    lt_invest = (df_trades[df_trades['Qty'] > 0]['Qty'] * df_trades[df_trades['Qty'] > 0]['Price']).sum()

    return {
        "fy": {"invest": fy_invest, "s_pnl": fy_s_pnl, "f_pnl": fy_f_pnl, "s_comm": fy_stock_comm, "f_comm": fy_forex_comm},
        "lt": {"invest": lt_invest, "s_pnl": lt_s_pnl, "f_pnl": lt_f_pnl, "s_comm": lt_stock_comm, "f_comm": lt_forex_comm}
    }

# --- DISPLAY INTERFACE ---
m = get_metrics(df_all_trades, df_all_perf, view_fy)

st.subheader(f"📊 {view_fy} Performance")
c1, c2, c3 = st.columns(3)
c1.metric("FY Investment", f"${m['fy']['invest']:,.2f}")
c2.metric("FY Realized (Stock/FX)", f"${m['fy']['s_pnl']:,.2f} / ${m['fy']['f_pnl']:,.2f}")
c3.metric("FY Comm (Stock/FX)", f"${m['fy']['s_comm']:,.2f} / ${m['fy']['f_comm']:,.2f}")

st.subheader("🌐 Lifetime Totals")
l1, l2, l3 = st.columns(3)
l1.metric("Lifetime Investment", f"${m['lt']['invest']:,.2f}")
l2.metric("Total Realized (Stock/FX)", f"${m['lt']['s_pnl']:,.2f} / ${m['lt']['f_pnl']:,.2f}")
l3.metric("Total Comm (Stock/FX)", f"${m['lt']['s_comm']:,.2f} / ${m['lt']['f_comm']:,.2f}")
