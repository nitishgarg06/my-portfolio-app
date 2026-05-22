import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import yfinance as yf
import plotly.express as px

# --- SINGLE SOURCE OF TRUTH FOR YEARS ---
PORTFOLIO_YEARS = ["FY24", "FY25", "FY26"]

st.set_page_config(layout="wide", page_title="IBKR Portfolio Dashboard")

# ==========================================
# 1. DATA LOADING & PROCESSING
# ==========================================
@st.cache_data(ttl=600)
def load_and_process_data():
    conn = st.connection("gsheets", type=GSheetsConnection)

    # When FY27 arrives, just change this to: ["FY24", "FY25", "FY26", "FY27"]
    all_frames = []

    for yr in PORTFOLIO_YEARS:
        df = conn.read(worksheet=yr)
        if df is not None and not df.empty:
            df = df.iloc[:, :13] 
            df.columns = list("ABCDEFGHIJKLMNOPQ") 
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
def get_metric(df, target_col, section_name, col_B_value=None, **kwargs):
    """
    Dynamically sums a target column based on the section name in Col A.
    Handles legacy col_B positional arguments and new dynamic kwargs.
    """
    # 1. Base filter: Match the exact section name in Column A
    mask = df['A'].astype(str).str.strip().str.upper() == section_name.upper()
    
    # 2. Legacy fallback: Check Column B if a 4th argument was provided
    if col_B_value:
        mask &= df['B'].astype(str).str.strip().str.upper() == str(col_B_value).upper()
        
    # 3. Dynamic filter: Apply any extra column filters passed via kwargs
    for key, value in kwargs.items():
        if key.startswith('col_'):
            col_letter = key.split('_')[1] # Extracts 'C' from 'col_C'
            mask &= df[col_letter].astype(str).str.strip().str.upper() == str(value).upper()
            
    # 4. Sum the matched rows in the target column
    try:
        return pd.to_numeric(df.loc[mask, target_col], errors='coerce').fillna(0).sum()
    except Exception:
        return 0.0

def get_realized_row(df, asset_class):
    # Added col_B="Data" to strictly prevent double-counting the "Total" rows!
    st_prof = get_metric(df, 'F', "Realized & Unrealized Performance Summary", col_B="Data", col_C=asset_class)
    st_loss = get_metric(df, 'G', "Realized & Unrealized Performance Summary", col_B="Data", col_C=asset_class)
    
    # Using H and I for Long-Term (Change these to 'F' and 'G' if IBKR stacked them in your report)
    lt_prof = get_metric(df, 'H', "Realized & Unrealized Performance Summary", col_B="Data", col_C=asset_class)
    lt_loss = get_metric(df, 'I', "Realized & Unrealized Performance Summary", col_B="Data", col_C=asset_class)
    
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
#year_order = {"FY24": 1, "FY25": 2, "FY26": 3, "Lifetime": 99}

# Filter data cumulatively
#current_rank = year_order[view_choice]
#df_view = df_master[df_master['YearSource'].map(year_order) <= current_rank]

# --- DYNAMIC UI PREP ---
# 1. Automatically generate the dropdown list (e.g., ["Lifetime", "FY26", "FY25", "FY24"])
dropdown_options = ["Lifetime"] + list(reversed(PORTFOLIO_YEARS))

# 2. Automatically assign cumulative math values (FY24=1, FY25=2, FY26=3, Lifetime=99)
year_order = {yr: i+1 for i, yr in enumerate(PORTFOLIO_YEARS)}
year_order["Lifetime"] = 99

#tab1, tab2, tab3 = st.tabs(["📊 Summary", "💼 My Holdings", "🧮 FIFO Calculator"])

# Create Tabs
#year_order = {"FY24": 1, "FY25": 2, "FY26": 3, "Lifetime": 99}

# Helper function to draw the tables for Stocks and Forex
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
        
tab1, tab2, tab3 = st.tabs(["📊 Summary", "💼 My Holdings", "🧮 FIFO Calculator"])

# ------------------------------------------
# TAB 1: SUMMARY & RECENT ACTIVITY
# ------------------------------------------
with tab1:
    view_choice_1 = st.selectbox("Select Financial Period (Cumulative)", dropdown_options, key="t1_view")
    
    # 1. Cumulative Data (For the big numbers)
    df_view_1 = df_master[df_master['YearSource'].map(year_order) <= year_order[view_choice_1]]
    
    # 2. Single Year Data (For the sub-metrics and Realized Gains table)
    if view_choice_1 == "Lifetime":
        df_single_year = df_master.copy()
        single_year_label = "Lifetime"
        gains_title = "Realized Gains & Losses (Lifetime Total)"
    else:
        df_single_year = df_master[df_master['YearSource'] == view_choice_1].copy()
        single_year_label = f"In {view_choice_1}"
        gains_title = f"Realized Gains & Losses ({view_choice_1} Only)"
        
    st.header(f"Portfolio Summary (Up to {view_choice_1})")
    
    # Calculate Cumulative AND Single Year metrics
    cum_inv_usd = get_metric(df_view_1, 'M', 'Trades', 'Total', col_D='Stocks', col_E='USD')
    cur_inv_usd = get_metric(df_single_year, 'M', 'Trades', 'Total', col_D='Stocks', col_E='USD')
    
    cum_inv_aud = get_metric(df_view_1, 'M', 'Trades', 'Total', col_D='Stocks', col_E='AUD')
    cur_inv_aud = get_metric(df_single_year, 'M', 'Trades', 'Total', col_D='Stocks', col_E='AUD')
    
    # Deposits (Summing the raw data rows to avoid Total row double-counting)
    cum_dep_aud = get_metric(df_view_1, 'F', 'Deposits & Withdrawals', col_B='Data', col_C='Total')
    cur_dep_aud = get_metric(df_single_year, 'F', 'Deposits & Withdrawals', col_B='Data', col_C='Total')
    
    # Dividends (Using Data rows for USD, and your dedicated Total row for AUD)
    cum_div_usd = get_metric(df_view_1, 'F', 'Dividends', col_B='Data', col_C='Total')
    cur_div_usd = get_metric(df_single_year, 'F', 'Dividends', col_B='Data', col_C='Total')
    
    cum_div_aud = get_metric(df_view_1, 'F', 'Dividends', col_C='Total in AUD')
    cur_div_aud = get_metric(df_single_year, 'F', 'Dividends', col_C='Total in AUD')
    
    # Top Level Metrics using delta to show the single-year numbers underneath
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Investment (USD)", f"${cum_inv_usd:,.2f}", f"{single_year_label}: ${cur_inv_usd:,.2f}", delta_color="off")
    c2.metric("Total Investment (AUD)", f"${cum_inv_aud:,.2f}", f"{single_year_label}: ${cur_inv_aud:,.2f}", delta_color="off")
    c3.metric("Funds Deposited (AUD)", f"${cum_dep_aud:,.2f}", f"{single_year_label}: ${cur_dep_aud:,.2f}", delta_color="off")

    c4, c5 = st.columns(2)
    c4.metric("Dividends (USD)", f"${cum_div_usd:,.2f}", f"{single_year_label}: ${cur_div_usd:,.2f}", delta_color="off")
    c5.metric("Dividends (AUD)", f"${cum_div_aud:,.2f}", f"{single_year_label}: ${cur_div_aud:,.2f}", delta_color="off")

    # (Your Realized Gains table code stays exactly the same here, it will automatically use the df_single_year we defined above!)

    # Realized Gains Table
    st.divider()
    st.subheader(gains_title)
    
    realized_data = {
        "Metric": ["Short Term Profit", "Short Term Loss", "Long Term Profit", "Long Term Loss", "Net Total"],
        "Stocks": get_realized_row(df_single_year, "Stocks"),
        "Forex": get_realized_row(df_single_year, "Forex"),
        "Total Assets": get_realized_row(df_single_year, "Total (All Assets)")
    }
    
    # 1. Define our traffic light color rules
    def color_gains_losses(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return 'color: #28a745;' # Financial Green
            elif val < 0:
                return 'color: #dc3545;' # Financial Red
            else:
                return 'color: #6c757d;' # Neutral Grey
        return ''

    # 2. Build the table, apply the $ formatting, AND apply the new colors
    df_realized = pd.DataFrame(realized_data).set_index("Metric")
    
    st.dataframe(
        df_realized.style
        .format("${:,.2f}")
        .map(color_gains_losses), # This line paints the cells!
        use_container_width=True
    )

    # --- UNIFIED RECENT ACTIVITY (LAST 5 TRADES) ---
    st.divider()
    st.subheader("📝 Recent Trade Activity")
    
    stock_trades = df_master[
        (df_master['A'].astype(str).str.strip().str.upper() == "TRADES") & 
        (df_master['B'].astype(str).str.strip().str.upper() == "DATA") &
        (df_master['D'].astype(str).str.strip().str.upper() == "STOCKS")
    ].copy()
    
    if not stock_trades.empty:
        stock_trades = stock_trades.sort_values(by='Trade_Date')
        
        # --- THE ULTIMATE FIFO P/L CALCULATOR ---
        def get_fifo_pl_breakdown(target_ticker, target_sell_date, df_all_trades, current_sell_price):
            """Calculates exact units AND exact dollar profit dynamically based on original buy prices."""
            history = df_all_trades[
                (df_all_trades['F'].str.strip() == target_ticker) & 
                (df_all_trades['Trade_Date'] <= target_sell_date)
            ]
            
            buy_queue = []
            st_units, lt_units = 0.0, 0.0
            st_pl, lt_pl = 0.0, 0.0
            
            for _, row in history.iterrows():
                units = float(row['H'])
                trade_date = row['Trade_Date']
                raw_val = abs(float(row['M']))
                
                # Calculate the exact price per unit for this specific row
                price = raw_val / abs(units) if abs(units) > 0 else 0.0
                
                if units > 0:
                    buy_queue.append({'date': trade_date, 'units': units, 'price': price})
                elif units < 0:
                    sell_qty = abs(units)
                    is_target_sell = (trade_date == target_sell_date)
                    
                    while sell_qty > 0 and buy_queue:
                        lot = buy_queue[0]
                        matched_qty = min(lot['units'], sell_qty)
                        
                        if is_target_sell:
                            days_held = (target_sell_date - lot['date']).days
                            # PROFIT MATH: Units * (Sell Price - Original Buy Price)
                            realized = matched_qty * (current_sell_price - lot['price'])
                            
                            if days_held >= 365:
                                lt_units += matched_qty
                                lt_pl += realized
                            else:
                                st_units += matched_qty
                                st_pl += realized
                                
                        sell_qty -= matched_qty
                        lot['units'] -= matched_qty
                        
                        # Remove lot if fully depleted
                        if lot['units'] <= 0:
                            buy_queue.pop(0)
                            
            return st_units, lt_units, st_pl, lt_pl

        # Render the dashboard list
        recent_5 = stock_trades.sort_values(by='Trade_Date', ascending=False).head(5)
        
        for _, row in recent_5.iterrows():
            date_str = row['Trade_Date'].strftime('%d/%b/%Y')
            ticker = str(row['F']).strip()
            
            raw_units = float(row['H'])
            raw_value = float(row['M'])
            
            action = "BOUGHT" if raw_units > 0 else "SOLD"
            units = abs(raw_units)
            total_val = abs(raw_value)
            avg_price = total_val / units if units > 0 else 0.0
            
            if units.is_integer():
                formatted_units = f"{int(units)}"
            else:
                formatted_units = f"{units:.4f}"
            
            icon = "📈" if action == "BOUGHT" else "📉"
            base_text = f"**{icon} {action}:** {formatted_units} units of **{ticker}** on {date_str} for **${total_val:,.2f}** (Avg price: **${avg_price:,.2f}**)"
            
            if action == "SOLD":
                # Trigger our calculator, completely bypassing IBKR's hidden columns!
                st_units, lt_units, st_pl, lt_pl = get_fifo_pl_breakdown(ticker, row['Trade_Date'], stock_trades, avg_price)
                
                total_pl = st_pl + lt_pl
                pl_type = "Profit" if total_pl >= 0 else "Loss"
                
                st_display = f"{int(st_units)}" if st_units.is_integer() else f"{st_units:.4f}"
                lt_display = f"{int(lt_units)}" if lt_units.is_integer() else f"{lt_units:.4f}"
                
                if st_units > 0 and lt_units > 0:
                    st.markdown(
                        f"{base_text}<br>"
                        f"↳ *Realized {pl_type}:* **${abs(total_pl):,.2f}**<br>"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;• *{lt_display} units Long Term (P/L: ${lt_pl:,.2f})*<br>"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;• *{st_display} units Short Term (P/L: ${st_pl:,.2f})*",
                        unsafe_allow_html=True
                    )
                elif lt_units > 0:
                    st.markdown(f"{base_text}<br>↳ *Long Term {pl_type}:* **${abs(total_pl):,.2f}**", unsafe_allow_html=True)
                elif st_units > 0:
                    st.markdown(f"{base_text}<br>↳ *Short Term {pl_type}:* **${abs(total_pl):,.2f}**", unsafe_allow_html=True)
                else:
                    st.markdown(f"{base_text}<br>↳ *Realized {pl_type}:* **${abs(total_pl):,.2f}**", unsafe_allow_html=True)
            else:
                st.markdown(base_text, unsafe_allow_html=True)
                
            st.write("") 
    else:
        st.info("No recent stock trades found.")

    # --- TEMPORARY DEBUG EXPANDER ---
    with st.expander("🕵️ Debug: P/L Columns & FIFO Testing"):
        st.write("**1. Find your Realized P/L Column:**")
        st.write("Scroll to the right in this table. Find the column with your profit numbers, note the letter, and change the 'O' in your code to match!")
        st.dataframe(recent_5, use_container_width=True)
        
        st.write("**2. Test the FIFO Math (Timeline Verification):**")
        st.write("Type a ticker you recently sold to see your exact chronological buy/sell history. You can manually verify if the gap between your buys and the recent sell is > 365 days.")
        
        test_ticker = st.text_input("Enter Ticker to test (e.g., AAPL):").strip().upper()
        if test_ticker:
            fifo_test_df = stock_trades[stock_trades['F'].str.strip().str.upper() == test_ticker]
            st.dataframe(fifo_test_df, use_container_width=True)
        
# ------------------------------------------
# TAB 2: MY HOLDINGS (LIFETIME ONLY)
# ------------------------------------------
with tab2:
    st.header("🏢 Current Open Positions")
    
    # --- RENDER STOCKS ---
    st.subheader("📈 Stocks")
    # Changed from df_view to df_master so it ALWAYS uses lifetime data
    stock_inventory = get_fifo_inventory(df_master, asset_category="Stocks")
    render_holdings_table(stock_inventory, is_stock=True)

    # --- PORTFOLIO ALLOCATION CHARTS (SIDE-BY-SIDE) ---
    st.subheader("📊 Portfolio Allocation")
    
    # Isolate the Open Positions data
    latest_year = df_master['YearSource'].dropna().max()
    df_positions = df_master[
        (df_master['A'].astype(str).str.strip().str.upper() == 'OPEN POSITIONS') & 
        (df_master['B'].astype(str).str.strip().str.upper() == 'DATA') &
        (df_master['D'].astype(str).str.strip().str.upper() == 'STOCKS') &
        (df_master['YearSource'] == latest_year)
    ].copy()
    
    if not df_positions.empty:
        # Pull Ticker (Col F), Invested Value (Col J), and Current Market Value (Col M)
        chart_data = df_positions[['F', 'J', 'L']].copy() 
        chart_data.columns = ['Ticker', 'Invested', 'Current_Value']
        
        # Convert values to numbers and fill missing ones with 0
        chart_data['Invested'] = pd.to_numeric(chart_data['Invested'], errors='coerce').fillna(0)
        chart_data['Current_Value'] = pd.to_numeric(chart_data['Current_Value'], errors='coerce').fillna(0)
        
        # Keep rows that have valid data in either column
        chart_data = chart_data[(chart_data['Invested'] > 0) | (chart_data['Current_Value'] > 0)]
        
        if not chart_data.empty:
            # 1. CREATE SIDE-BY-SIDE COLUMNS
            left_chart_col, right_chart_col = st.columns(2)
            
            # 2. LEFT COLUMN: AMOUNT INVESTED
            with left_chart_col:
                st.markdown("<h4 style='text-align: center;'>By Amount Invested (Cost)</h4>", unsafe_allow_html=True)
                fig_invested = px.pie(
                    chart_data[chart_data['Invested'] > 0], 
                    names='Ticker', 
                    values='Invested', 
                    hole=0.4
                )
                fig_invested.update_traces(textposition='inside', textinfo='percent+label')
                fig_invested.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_invested, use_container_width=True)
                
            # 3. RIGHT COLUMN: CURRENT MARKET VALUE
            with right_chart_col:
                st.markdown("<h4 style='text-align: center;'>By Current Market Value</h4>", unsafe_allow_html=True)
                fig_current = px.pie(
                    chart_data[chart_data['Current_Value'] > 0], 
                    names='Ticker', 
                    values='Current_Value', 
                    hole=0.4
                )
                fig_current.update_traces(textposition='inside', textinfo='percent+label')
                fig_current.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_current, use_container_width=True)
        else:
            st.info("No active positions with values greater than $0 found.")
    else:
        st.info("No 'Open Positions' data found in this report.")    

# ------------------------------------------
# TAB 3: FIFO CALCULATOR (LIFETIME ONLY)
# ------------------------------------------
with tab3:
    st.header("🧮 FIFO Scenario Calculator")
    
    # Using df_master here as well to ensure total accuracy
    active_inventory = get_fifo_inventory(df_master, asset_category="Stocks")
    
    # ... (Keep the rest of your Tab 3 Calculator code exactly as it is) ...
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
