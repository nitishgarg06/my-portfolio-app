import streamlit as st
from streamlit_gsheets import GSheetsConnection

st.title("🔍 Service Account Diagnostic")
conn = st.connection("gsheets", type=GSheetsConnection)

try:
    # This attempt to read the 'first' available tab
    df = conn.read(ttl=0)
    st.success("✅ Success! The app can see your spreadsheet.")
    
    # Let's try to read your specific FY24 tab
    st.write("Attempting to read 'FY24'...")
    df_fy24 = conn.read(worksheet="FY24", ttl=0)
    st.write("Found FY24! Here is a preview:")
    st.dataframe(df_fy24.head())
    
except Exception as e:
    st.error(f"❌ Connection Error: {e}")
    st.info("If you see 'SpreadsheetNotFound', you forgot to click the 'Share' button in Google Sheets.")
