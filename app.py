def render_sec(subset, label):
        st.markdown(f'<div class="section-header">{label} (as of {cur_dt})</div>', unsafe_allow_stdio=True)
        if subset.empty: 
            st.info("No holdings found.")
            return
        
        subset['Cost'] = subset['qty'] * subset['price']
        agg = subset.groupby('Symbol').agg({'qty': 'sum', 'Cost': 'sum'}).reset_index()
        agg['Avg Buy'] = agg['Cost'] / agg['qty']
        agg['Price'] = agg['Symbol'].map(prices).fillna(0)
        agg['Value'] = agg['qty'] * agg['Price']
        agg['P/L $'] = agg['Value'] - agg['Cost']
        agg['P/L %'] = (agg['P/L $'] / agg['Cost']) * 100
        agg.insert(0, 'Sr.', range(1, len(agg) + 1))

        # --- WIDTH CONTROL ---
        # We create 2 columns: one for the table (8 parts) and one for empty space (4 parts)
        # This makes the table take up roughly 66% of the screen width
        table_col, spacer_col = st.columns([8, 4]) 

        with table_col:
            st.dataframe(
                agg.style.format({
                    "Avg Buy": "${:.2f}", "Price": "${:.2f}", "Value": "${:.2f}", "P/L $": "${:.2f}", "P/L %": "{:.2f}%"
                }).map(lambda x: 'color: #10b981' if x > 0 else 'color: #ef4444', subset=['P/L $', 'P/L %']), 
                use_container_width=True, 
                hide_index=True
            )
            st.write(f"**Total {label} Value:** `${agg['Value'].sum():,.2f}`")
