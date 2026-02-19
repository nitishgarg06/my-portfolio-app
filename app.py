import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import requests
import base64
from datetime import datetime

# --- 1. DATA ENGINE: GITHUB & STORAGE ---
class DataEngine:
    @staticmethod
    def push_to_github(df, path):
        """Pushes a dataframe as a CSV to your private GitHub repository."""
        url = f"https://api.github.com/repos/{st.secrets['GITHUB_REPO']}/contents/{path}"
        headers = {
            "Authorization": f"token {st.secrets['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github.v3+json"
        }
        res = requests.get(url, headers=headers)
        sha = res.json().get('sha') if res.status_code == 200 else None
        
        content = base64.b64encode(df.to_csv(index=False).encode()).decode()
        payload = {
            "message": f"Data Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "content": content,
            "branch": "main"
        }
        if sha: payload["sha"] = sha
        return requests.put(url, headers=headers, json=payload).status_code in [200, 201]

    @staticmethod
    def load_from_github(path):
        """Loads a CSV from GitHub with a cache-buster for the latest data."""
        url = f"https://raw.githubusercontent.com/{st.secrets['GITHUB_REPO']}/main/{path}?v={datetime.now().timestamp()}"
        try:
            df = pd.read_csv(url)
            return df if not df.empty else None
        except:
            return None

# --- 2. PRICE ENGINE: STATIC CACHE ---
class PriceEngine:
    @staticmethod
    def refresh_cache(tickers):
        """Fetches current prices and saves them to a static CSV on GitHub."""
        price_data = []
        headers = {'User-Agent': 'Mozilla/5.0'}
        for t in tickers:
            try:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t}"
                res = requests.get(url, headers=headers, timeout=5).json()
                price = res['chart']['result'][0]['meta']['regularMarketPrice']
                price_data.append({"Symbol": t, "StaticPrice": price, "Date": datetime.now().strftime("%Y-%m-%d %H:%M")})
            except:
                price_data.append({"Symbol": t, "StaticPrice": 0.0, "Date": "Error"})
        
        df = pd.DataFrame(price_data)
        DataEngine.push_to_github(df, "data/price_cache.csv")
        return df

# --- 3. ANALYTICS ENGINE: FIFO & METRICS ---
class AnalyticsEngine:
    @staticmethod
    def run_fifo(df_trades):
        """Processes cumulative trades to find open lots and ST/LT status."""
        # Weighted data cleaning
        def clean(x): return pd.to_numeric(str(x).replace('$','').replace(',','').replace('(','-').replace(')',''), errors='coerce')
        
        # Fuzzy map columns
        c_q = next(c for c in df_trades.columns if 'Qty' in c or 'Quantity' in c)
        c_p = next(c for c in df_trades.columns if 'Price' in c)
        c_s = next(c for c in df_trades.columns if 'Symbol' in c or 'Ticker' in c)
        c_d = next(c for c in df_trades.columns if 'Date' in c)
        c_c = next((c for c in df_trades.columns if 'Comm' in c), None)

        df_trades['Q'] = df_trades[c_q].apply(clean).fillna(0)
        df_trades['P'] = df_trades[c_p].apply(clean).fillna(0)
        df_trades['C'] = df_trades[c_c].apply(clean).abs().fillna(0) if c_c else 0.0
        df_trades['DT'] = pd.to_datetime(df_trades[c_d].str.split(',').str[0])

        open_lots = []
        for ticker in df_trades[c_s].unique():
            sym_df = df_trades[df_trades[c_s] == ticker].sort_values('DT')
            lots = []
            for _, row in sym_df.iterrows():
                if row['Q'] > 0: # Buy
                    lots.append({'dt': row['DT'], 'q': row['Q'], 'p': row['P'], 'c': row['C']})
                elif row['Q'] < 0: # Sell
                    sell_q = abs(row['Q'])
                    while sell_q > 0 and lots:
                        if lots[0]['q'] <= sell_q: sell_q -= lots.pop(0)['q']
                        else: lots[0]['q'] -= sell_q; sell_q = 0
            for l in lots:
                l['Symbol'] = ticker
                l['Status'] = "Long-Term" if (pd.Timestamp.now() - l['dt']).days > 365 else "Short-Term"
                open_lots.append(l)
        return pd.DataFrame(
