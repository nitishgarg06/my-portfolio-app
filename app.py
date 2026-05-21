import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf

st.set_page_config(layout="wide", page_title="IBKR Portfolio Dashboard")

# ==========================================
# 1. DATA LOADING & PROCESSING
# ==========================================
@st.cache_data(ttl=600)
def load_and_process_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    
    years = ["FY24", "FY25", "FY26"]
    all_frames = []

    for yr in years:
        df = conn.read(worksheet=yr)
        if df is not None and not df.empty:
            df = df.iloc[:, :13] 
            df.columns = list("ABCDEFGHIJKLM") 
            df['YearSource'] = yr
            df['Trade_Date'] = pd.to_datetime(df['G'], errors='coerce')
            all_frames.append(df)
    
    if not all_frames:
        return pd.DataFrame()

    full_df = pd.concat(all_frames, ignore_index=True)

    # FIX: Only clean H (Quantity) and M (Amount) here. 
    # We leave F alone so your Ticker symbols aren't deleted!
    for col in ['H', 'M']:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(
                full_df[col].astype(str).replace(r'[$,\s()]', '', regex=True), 
                errors='coerce'
            ).fillna(0.0)
    
    full_df = full_df.sort_values('Trade_Date').reset_index(drop=True)
    return full_df

# Helper function to extract summary metrics
def get_metric(df, col_to_sum, col_A, col_B=None, col_C=None, col_D=None, col_E=None):
    if df.empty: return 0.0
    mask = (df['A'].astype(str).str.strip() == col_A)
    if col_B: mask &= (df['B'].astype(str).str.strip() == col_B)
    if col_C: mask &= (df['C'].astype(str).str.strip() == col_C)
    if col_D: mask &= (df['D'].astype(str).str.strip() == col_D)
    if col_E: mask &= (df['E'].astype(str).str.strip() == col_E)
    
    # FIX: Convert to numbers ON THE FLY just for this calculation
    target_data = df.loc[mask, col_to_sum].astype(str).replace(r'[$,\s()]', '', regex=True)
    numeric_data = pd.to_numeric(target_data, errors='coerce').fillna(0.0)
    
    return float(numeric_data.sum())

def get_realized_row(df, asset_class):
    st_prof = get_metric(df, 'F', "Realized & Unrealized Performance Summary", col_C=asset_class)
    st_loss = get_metric(df, 'G', "Realized & Unrealized Performance Summary", col_C=asset_class)
    lt_prof = get_metric(df, 'H', "Realized & Unrealized Performance Summary", col_C=asset_class)
    lt_loss = get_metric(df, 'I', "Realized & Unrealized Performance Summary", col_C=asset_class)
    return [st_prof, st_loss, lt_prof, lt_loss, (st_prof + st_loss + lt_prof + lt_loss)]

# ==========================================
# 2. FIFO INVENTORY ENGINE
# ==========================================
def get_fifo_inventory(df, ticker=None, asset_category=None):
    """Rebuilds the open lots by replaying trade history."""
    mask = (df['A'].astype(str).str.strip().str.upper() == "TRADES") & \
           (df['B'].astype(str).str.strip().str.upper() == "DATA")
    
    # NEW: Filter by Column D if an asset category is provided
    if asset_category:
        mask &= (df['D'].astype(str).str.strip().str.upper() == asset_category.upper())
        
    trades = df[mask]
    
    if ticker:
        trades = trades[trades['F'].astype(str).str.strip().str.upper() == ticker.upper()]
        
    inventory_map = {}
    
    for _, row in trades.iterrows():
        t = str(row['F']).strip().upper()
        if t not in inventory_map:
            inventory_map[t] = []
            
        qty = round(float(row['H']), 8)
        basis = float(row['M'])
        
        if qty > 0: # BUY
            inventory_map[t].append({'qty': qty, 'basis': basis})
        elif qty < 0: # SELL
            qty_to_close = abs(qty)
            while qty_to_close > 1e-9 and inventory_map[t]:
                if inventory_map[t][0]['qty'] <= qty_to_close + 1e-9:
                    qty_to_close -= inventory_map[t].pop(0)['qty']
                else:
                    fraction = (inventory_map[t][0]['qty'] - qty_to_close) / inventory_map[t][0]['qty']
                    inventory_map[t][0]['qty'] -= qty_to_close
                    inventory_map[t][0]['basis'] *= fraction
                    qty_to_close = 0
                    
    return inventory_map

# ==========================================
# 3. UI RENDERING
# ==========================================
df_master = load_and_process_data()

st.title("📈 IBKR Portfolio Tracker")

# --- GLOBAL FILTER (CUMULATIVE) ---
#view_choice = st.selectbox("Select Financial Period (Cumulative)", ["Lifetime", "FY26", "FY25", "FY24"])
year_order = {"FY24": 1, "FY25": 2, "FY26": 3, "Lifetime": 99}

# Filter data cumulatively
#current_rank = year_order[view_choice]
#df_view = df_master[df_master['YearSource'].map(year_order) <= current_rank]

# Create Tabs
tab1, tab2, tab3 = st.tabs(["📊 Summary", "💼 My Holdings", "🧮 FIFO Calculator"])

# ------------------------------------------
# TAB 1: SUMMARY
# ------------------------------------------
with tab1:
    view_choice_1 = st.selectbox("Select Financial Period (Cumulative)", ["Lifetime", "FY26", "FY25", "FY24"], key="t1_view")
    df_view_1 = df_master[df_master['YearSource'].map(year_order) <= year_order[view_choice_1]]
    
    st.header(f"Portfolio Summary (Up to {view_choice_1})")
    
    # Top Level Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Investment (USD)", f"${get_metric(df_view_1, 'M', 'Trades', 'Total', col_D='Stocks', col_E='USD'):,.2f}")
    c2.metric("Total Investment (AUD)", f"${get_metric(df_view_1, 'M', 'Trades', 'Total', col_D='Stocks', col_E='AUD'):,.2f}")
    c3.metric("Funds Deposited (AUD)", f"${get_metric(df_view_1, 'F', 'Deposits & Withdrawals', 'Total'):,.2f}")

    c4, c5 = st.columns(2)
    c4.metric("Dividends (USD)", f"${get_metric(df_view_1, 'F', 'Dividends', 'Total'):,.2f}")
    c5.metric("Dividends (AUD)", f"${get_metric(df_view_1, 'F', 'Dividends', 'Total in AUD'):,.2f}")

    # Realized Gains Table
    st.divider()
    st.subheader("Realized Gains & Losses")
    realized_data = {
        "Metric": ["Short Term Profit", "Short Term Loss", "Long Term Profit", "Long Term Loss", "Net Total"],
        "Stocks": get_realized_row(df_view_1, "Stocks"),
        "Forex": get_realized_row(df_view_1, "Forex"),
        "Total Assets": get_realized_row(df_view_1, "Total (All Assets)")
    }
    st.dataframe(pd.DataFrame(realized_data).set_index("Metric").style.format("${:,.2f}"), use_container_width=True)

# ------------------------------------------
# TAB 2: MY HOLDINGS
# ------------------------------------------
with tab2:
    view_choice_2 = st.selectbox("Select Financial Period (Cumulative)", ["Lifetime", "FY26", "FY25", "FY24"], key="t2_view")
    df_view_2 = df_master[df_master['YearSource'].map(year_order) <= year_order[view_choice_2]]
    
    st.header("🏢 Current Open Positions")
    
    def render_holdings_table(inventory_data, is_stock=True):
        holdings_data = []
        with st.spinner("Processing Holdings..."):
            for ticker, lots in inventory_data.items():
                total_qty = sum(l['qty'] for l in lots)
                if total_qty > 0.001:
                    total_invested = sum(l['basis'] for l in lots)
                    avg_buy_price = total_invested / total_qty
                    
                    # Only fetch live price for Stocks
                    live_price = 0.0
                    if is_stock:
                        try:
                            live_price = yf.Ticker(ticker).fast_info['last_price']
                        except:
                            live_price = 0.0
                    
                    pl_dollar = (total_qty * live_price) - total_invested if is_stock else 0.0
                    pl_percent = (pl_dollar / total_invested * 100) if (total_invested > 0 and is_stock) else 0.0
                    
                    row_data = {
                        "Symbol": ticker,
                        "Amount Invested": total_invested,
                        "Units": total_qty,
                        "Avg. Buy Price": avg_buy_price,
                    }
                    
                    if is_stock:
                        row_data.update({
                            "Current Price": live_price,
                            "Profit/Loss (%)": pl_percent,
                            "Profit/Loss ($)": pl_dollar
                        })
                        
                    holdings_data.append(row_data)
                    
        if holdings_data:
            h_df = pd.DataFrame(holdings_data)
            
            # Formatting dictionary depending on asset class
            format_dict = {
                "Amount Invested": "${:,.2f}",
                "Units": "{:.4f}",
                "Avg. Buy Price": "${:,.2f}",
            }
            if is_stock:
                format_dict.update({
                    "Current Price": "${:,.2f}",
                    "Profit/Loss (%)": "{:,.2f}%",
                    "Profit/Loss ($)": "${:,.2f}"
                })
                
            styled_df = h_df.style.format(format_dict)
            
            if is_stock:
                styled_df = styled_df.map(
                    lambda x: 'color: #ff4b4b' if x < 0 else 'color: #09ab3b' if x > 0 else '', 
                    subset=['Profit/Loss (%)', 'Profit/Loss ($)']
                )
                
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions found.")

    # --- RENDER STOCKS ---
    st.subheader("📈 Stocks")
    stock_inventory = get_fifo_inventory(df_view_2, asset_category="Stocks")
    render_holdings_table(stock_inventory, is_stock=True)
    
    # --- RENDER FOREX ---
    st.divider()
    st.subheader("💱 Forex")
    forex_inventory = get_fifo_inventory(df_view_2, asset_category="Forex")
    render_holdings_table(forex_inventory, is_stock=False)

# ------------------------------------------
# TAB 3: FIFO CALCULATOR
# ------------------------------------------
with tab3:
    st.header("🧮 FIFO Scenario Calculator")
    
    # Re-calculate active tickers for the dropdown
    active_inventory = get_fifo_inventory(df_master, asset_category="Stocks")
    active_tickers = [t for t, lots in active_inventory.items() if sum(l['qty'] for l in lots) > 0.001]
    
    if not active_tickers:
        st.warning("No active holdings available to sell.")
    else:
        # UI: Selection
        c_sel1, c_sel2 = st.columns([1, 2])
        selected_ticker = c_sel1.selectbox("Select Stock to Sell:", sorted(active_tickers))
        
        # Get lots for selected stock
        lots = active_inventory[selected_ticker]
        total_units = sum(l['qty'] for l in lots)
        total_basis = sum(l['basis'] for l in lots)
        
        c_sel2.info(f"**Current Holding Profile:** \n\n {total_units:.4f} Units | Total Cost Basis: ${total_basis:,.2f} | Avg Cost: ${(total_basis/total_units):,.2f}")
        
        # UI: Sell Settings
        st.subheader("1. Configure Sale Amount")
        sell_mode = st.radio("Define sell amount by:", ["Units", "Percentage (%)"], horizontal=True)
        
        if sell_mode == "Units":
            # Streamlit sliders allow manual typing! Just click the number on the right.
            units_to_sell = st.slider(
                "Units to Sell (Click the number on the right to type an exact amount):", 
                0.0, float(total_units), 
                value=float(total_units)/2, 
                step=0.0001
            )
            percent_to_sell = (units_to_sell / total_units) * 100 if total_units > 0 else 0
        else:
            percent_to_sell = st.slider(
                "Percentage to Sell (%) (Click the number on the right to type an exact amount):", 
                0.0, 100.0, 
                value=50.0, 
                step=1.0
            )
            units_to_sell = (percent_to_sell / 100) * total_units

        # UI: Target Mode
        st.subheader("2. Pricing Strategy")
        use_target_profit = st.checkbox("Calculate based on a Target Profit %", value=True)
        
        if use_target_profit:
            target_profit = st.number_input("Target Profit (%)", value=30.0, step=1.0)
        else:
            try:
                live_price = yf.Ticker(selected_ticker).fast_info['last_price']
                st.success(f"Using Current Market Price: **${live_price:,.2f}**")
            except:
                st.error("Could not fetch live price. Defaulting to $0.")
                live_price = 0.0

        # --- EXECUTE FIFO CALCULATION ---
        if units_to_sell > 0:
            st.divider()
            
            # Slice the FIFO queue
            temp_sell_qty = units_to_sell
            slice_cost_basis = 0.0
            
            for lot in lots:
                if temp_sell_qty <= 0: break
                take = min(lot['qty'], temp_sell_qty)
                slice_cost_basis += (take / lot['qty']) * lot['basis']
                temp_sell_qty -= take

            st.subheader(f"Results for selling {units_to_sell:.4f} units ({percent_to_sell:.1f}%)")
            
            # Case 1: Target Profit Goal
            if use_target_profit:
                target_revenue = slice_cost_basis * (1 + (target_profit/100))
                target_price_per_unit = target_revenue / units_to_sell
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Sale Price Required (Per Unit)", f"${target_price_per_unit:,.2f}")
                m2.metric("Total Revenue Target", f"${target_revenue:,.2f}")
                m3.metric("Projected Profit", f"${target_revenue - slice_cost_basis:,.2f}")
                
            # Case 2: Sell at Market Price
            else:
                projected_revenue = units_to_sell * live_price
                projected_profit = projected_revenue - slice_cost_basis
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Revenue from Sale", f"${projected_revenue:,.2f}")
                m2.metric("Cost Basis of these Units (FIFO)", f"${slice_cost_basis:,.2f}")
                m3.metric(
                    "Total Projected Profit", 
                    f"${projected_profit:,.2f}", 
                    delta=f"{(projected_profit/slice_cost_basis)*100 if slice_cost_basis > 0 else 0:.1f}% Return"
                )

            # --- REMAINING POSITION ---
            st.markdown("### 📋 Remaining Position Profile")
            rem_units = total_units - units_to_sell
            rem_basis = total_basis - slice_cost_basis
            rem_avg_cost = rem_basis / rem_units if rem_units > 0 else 0
            
            r1, r2, r3 = st.columns(3)
            r1.metric("Remaining Units", f"{rem_units:.4f}")
            r2.metric("Remaining Investment (Cost Basis)", f"${rem_basis:,.2f}")
            r3.metric("New Avg. Cost per Unit", f"${rem_avg_cost:,.2f}")
