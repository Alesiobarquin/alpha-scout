import streamlit as st
import pandas as pd
import yfinance as yf
import streamlit.components.v1 as components
import json
import os
import time
from streamlit_autorefresh import st_autorefresh

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Alpha Scout Terminal",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark Theme & UI Polish
st.markdown("""
    <style>
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricValue"] {
        font-size: 26px;
        font-weight: 700;
        color: #FAFAFA;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #A0A0A0;
    }
    /* Table Headers */
    th {
        background-color: #262730 !important;
        color: #FAFAFA !important;
        font-size: 15px !important;
    }
    /* Tab Styling */
    button[data-baseweb="tab"] {
        font-size: 18px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING & MAPPING
# -----------------------------------------------------------------------------
JSON_PATH = "data/latest_report.json"
CSV_PATH = "data/performance_log.csv"

def load_latest_json():
    """Loads the active signal report with your specific JSON keys."""
    if not os.path.exists(JSON_PATH):
        return None
    try:
        with open(JSON_PATH, "r") as f:
            data = json.load(f)
            # Handle list of catalysts, return the top conviction one
            if "catalysts" in data and len(data["catalysts"]) > 0:
                return data["catalysts"][0]
            return None
    except Exception as e:
        st.error(f"Error reading JSON: {e}")
        return None

def load_history_csv():
    """Loads and normalizes the CSV Log."""
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()

    try:
        # Define expected headers from alpha_scout.py
        # ["Date", "Ticker", "Entry_Price", "Conviction", "Market_Cap", "ATR_Value", "Stop_Loss", "Target_Price", "Thesis", "Status"]
        expected_cols = [
            "Date", "Ticker", "Entry_Price", "Conviction", 
            "Market_Cap", "ATR_Value", "Stop_Loss_Target", "Target_Price", 
            "Thesis", "Status"
        ]

        # Load CSV with header=None since we see raw data starting smoothly
        # quotechar='"' helps, but we will also manually strip to be safe.
        df = pd.read_csv(CSV_PATH, header=None, names=expected_cols, quotechar='"', skipinitialspace=True)
        
        # 1. Handle potential header row if file was recreated properly
        if not df.empty and str(df.iloc[0]['Date']).strip() == 'Date':
            df = df.iloc[1:].reset_index(drop=True)

        # 2. Aggressive Cleaning of Tickers (User specifically asked for this)
        if 'Ticker' in df.columns:
            df['Ticker'] = df['Ticker'].astype(str).str.strip().str.replace('"', '').str.replace("'", "")

        # 3. Clean other string columns just in case
        for col in ['Thesis', 'Status', 'Market_Cap']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.replace('"', '')

        # Standardize formatting
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values(by='Date', ascending=False)

        # Ensure numeric columns are actually numeric
        cols_to_clean = ['Entry_Price', 'Target_Price', 'ATR_Value', 'Stop_Loss_Target']
        for c in cols_to_clean:
            if c in df.columns:
                # Remove '$' if present from price columns before converting
                df[c] = df[c].astype(str).str.replace('$', '').str.replace(',', '')
                df[c] = pd.to_numeric(df[c], errors='coerce')

        return df
    except Exception as e:
        # Graceful failure
        # st.info(f"Could not load history: {e}") 
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# 3. CACHED DATA WRAPPER
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_data_bundle():
    """Bundles data loading and provides a sync timestamp."""
    json_data = load_latest_json()
    csv_data = load_history_csv()
    sync_time = time.strftime("%H:%M:%S")
    return json_data, csv_data, sync_time

# -----------------------------------------------------------------------------
# 4. LIVE MARKET DATA (CACHED)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_prices(tickers):
    """
    Fetches real-time prices for a list of tickers.
    Cached for 5 minutes to prevent rate limiting.
    """
    if not tickers:
        return {}
    
    clean_tickers = list(set([str(t).upper().strip() for t in tickers]))
    price_map = {}
    
    try:
        # 1. Single Ticker (Faster method)
        if len(clean_tickers) == 1:
            t = clean_tickers[0]
            ticker_obj = yf.Ticker(t)
            # fast_info provides the latest cached price from exchange
            price = ticker_obj.fast_info.last_price
            price_map[t] = price
            
        # 2. Batch Tickers (Efficient for History Table)
        else:
            ticker_str = " ".join(clean_tickers)
            data = yf.download(ticker_str, period="1d", group_by='ticker', progress=False, threads=True)
            
            for t in clean_tickers:
                try:
                    price = data[t]['Close'].iloc[-1]
                    price_map[t] = price
                except:
                    price_map[t] = None
                    
    except Exception as e:
        print(f"YFinance Error: {e}")
        
    return price_map

# -----------------------------------------------------------------------------
# 4. CHART WIDGET
# -----------------------------------------------------------------------------
def render_chart(ticker):
    html_code = f"""
    <div class="tradingview-widget-container" style="height:600px;width:100%">
      <div id="tradingview_chart" style="height:calc(100% - 32px);width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "height": 600,
        "width": "100%",
        "symbol": "{ticker}",
        "interval": "D",
        "timezone": "America/New_York",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_chart"
      }}
      );
      </script>
    </div>
    """
    components.html(html_code, height=600)

# -----------------------------------------------------------------------------
# 5. MAIN APP
# -----------------------------------------------------------------------------
def main():
    # 1. Auto-Refresh (every 30 mins)
    st_autorefresh(interval=30 * 60 * 1000, key="datarefresh")

    # 2. Sidebar Controls
    with st.sidebar:
        st.header("Terminal Controls")
        if st.button("🔄 Force Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
    # 3. Load Data Bundle
    signal, df, last_sync = get_data_bundle()

    # 4. UI Header
    c_title, c_sync = st.columns([3, 1])
    with c_title:
        st.title("Alpha Scout Terminal")
    with c_sync:
        st.markdown(f"""
            <div style="text-align: right; color: #666; font-size: 0.8em; margin-top: 25px;">
                Last Sync: {last_sync}
            </div>
        """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🚀 Active Signal", "📜 Performance History"])

    # --- TAB 1: ACTIVE SIGNAL (FROM JSON) ---
    with tab1:
        # signal = load_latest_json() # Now using bundled data
        
        if signal:
            # Map JSON keys specifically based on your provided file
            ticker = signal.get("ticker", "UNKNOWN")
            entry = signal.get("current_price", 0.0) # Using AI estimated price as entry reference
            
            # Use 'calculated_target' if available, otherwise 0
            target = signal.get("calculated_target", 0.0)
            
            # Use 'calculated_stop_loss' if available
            stop_val = signal.get("calculated_stop_loss", 0.0)
            
            # Thesis text
            thesis = signal.get("thesis", "No thesis provided.")
            catalyst_txt = signal.get("catalyst_details", "")

            # Header
            c1, c2 = st.columns([3, 1])
            with c1:
                st.subheader(f"High Conviction: ${ticker}")
            with c2:
                st.link_button(
                    "Trade on Robinhood 🏹", 
                    f"https://robinhood.com/us/en/stocks/{ticker}/",
                    type="primary", 
                    use_container_width=True
                )
            
            st.divider()

            # Live Price Check
            prices = fetch_live_prices([ticker])
            live_price = prices.get(ticker, entry)
            
            # Calc Move
            pct_move = 0.0
            if entry > 0 and live_price:
                pct_move = ((live_price - entry) / entry) * 100

            # Metrics - Two Rows
            r1c1, r1c2 = st.columns(2)
            r1c1.metric("Entry Price", f"${entry:.2f}")
            r1c2.metric("Live Price", f"${live_price:.2f}", f"{pct_move:.2f}%")
            
            r2c1, r2c2 = st.columns(2)
            r2c1.metric("Target", f"${target:.2f}")
            r2c2.metric("Stop Loss", f"${stop_val:.2f}")

            st.divider()
            
            # Additional keys from JSON for context box
            risks = signal.get("risk", "N/A")
            absorption = signal.get("absorption_status", "N/A")

            # Content - Styled Box
            st.markdown(f"""
            <div style="background-color: #262730; padding: 20px; border-radius: 10px; border: 1px solid #444;">
                <h4 style="margin-top:0;">📝 Investment Thesis</h4>
                <p>{thesis}</p>
                <hr style="border-top: 1px solid #444;">
                <p><strong>🔥 Catalyst:</strong> {catalyst_txt}</p>
                <p><strong>⚠️ Risks:</strong> {risks}</p>
                <p><strong>🌊 Absorption:</strong> {absorption}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("### Technical Chart")
            render_chart(ticker)

        else:
            st.info("No active signal JSON found. Run the backend script.")

    # --- TAB 2: HISTORY (FROM CSV) ---
    with tab2:
        # df = load_history_csv() # Now using bundled data
        
        if not df.empty and 'Ticker' in df.columns:
            
            # 1. Fetch Live Prices for table
            tickers = df['Ticker'].unique().tolist()
            with st.spinner("Syncing live prices..."):
                live_prices_map = fetch_live_prices(tickers)
            
            # 2. Add Live Columns
            df['Live Price'] = df['Ticker'].map(live_prices_map)
            
            def calc_return(row):
                try:
                    curr = float(row['Live Price'])
                    ent = float(row['Entry_Price'])
                    if ent > 0:
                        return ((curr - ent) / ent) * 100
                    return 0.0
                except:
                    return 0.0

            df['Profit/Loss %'] = df.apply(calc_return, axis=1)

            # 3. Select & Order Columns
            display_cols = [
                "Date", "Ticker", "Entry_Price", "Live Price", 
                "Target_Price", "Stop_Loss_Target", "ATR_Value", "Profit/Loss %"
            ]
            
            # Filter columns that actually exist
            valid_cols = [c for c in display_cols if c in df.columns]
            view_df = df[valid_cols].copy()

            # 4. Display Table
            selection = st.dataframe(
                view_df,
                use_container_width=True,
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun",
                column_config={
                    "Date": st.column_config.DatetimeColumn("Alert Date", format="MM-DD HH:mm"),
                    "Entry_Price": st.column_config.NumberColumn("Entry", format="$%.2f"),
                    "Live Price": st.column_config.NumberColumn("Live", format="$%.2f"),
                    "Target_Price": st.column_config.NumberColumn("Target", format="$%.2f"),
                    "Stop_Loss_Target": st.column_config.TextColumn("Stop Loss"),
                    "ATR_Value": st.column_config.NumberColumn("ATR", format="%.2f"),
                    "Profit/Loss %": st.column_config.NumberColumn(
                        "Profit/Loss %", 
                        format="%.2f%%",
                        help="Unrealized P/L based on Live Price vs Entry"
                    )
                }
            )

            # 5. Chart Selection
            sel_ticker = None
            if len(selection.selection.rows) > 0:
                idx = selection.selection.rows[0]
                sel_ticker = view_df.iloc[idx]['Ticker']
            elif not view_df.empty:
                sel_ticker = view_df.iloc[0]['Ticker']
            
            if sel_ticker:
                st.divider()
                st.caption(f"Analyzing History: {sel_ticker}")
                render_chart(sel_ticker)
        else:
            st.warning("Performance log is empty or could not be loaded.")

if __name__ == "__main__":
    main()