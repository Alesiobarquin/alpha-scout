import os
import json
import time
import requests
import re
import csv
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from google.genai import types
import pytz

# --- CONFIGURATION ---
MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview") 
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DATA_FILE = "data/latest_report.json"
PERFORMANCE_LOG_FILE = "data/performance_log.csv"

# --- DATA MODELS ---
class Catalyst(BaseModel):
    ticker: str = Field(..., description="Stock Ticker (e.g., AAPL)")
    current_price: float = Field(..., description="AI estimated price (will be overwritten by API).")
    market_cap: str = Field(..., description="Market Cap string (e.g., '$450M', '$2.1B').")
    conviction_score: int = Field(..., description="1-10 Score.")
    thesis: str = Field(..., description="1-2 sentence thesis.")
    catalyst_details: str = Field(..., description="Event details.")
    absorption_status: str = Field(..., description="Why market hasn't reacted.")
    earnings_date: str = Field(..., description="Next earnings date.")
    relative_volume: str = Field(..., description="Vol vs 30d avg.")
    stop_loss_trigger: str = Field(..., description="AI suggested stop (will be overwritten by ATR calc).")
    sentiment: str = Field(..., description="Bullish/Bearish.")
    prediction_market: str = Field(..., description="Odds.")
    recency_proof: str = Field(..., description="Link/Timestamp.")
    risk: str = Field(..., description="Risk factor.")
    expected_upside: str = Field(..., description="Upside potential.")
    mispricing_evidence: str = Field(..., description="Evidence.")
    x_sentiment: Optional[str] = Field(None, description="X buzz.")
    
    # New Technical Fields (Optional as they are populated post-AI)
    atr_value: Optional[float] = None
    target_price: Optional[float] = None

class ScoutReport(BaseModel):
    catalysts: List[Catalyst]

# --- HELPER FUNCTIONS ---
def parse_market_cap_to_millions(cap_str: str) -> float:
    clean = cap_str.upper().replace('$', '').replace(',', '').strip()
    try:
        if 'B' in clean: return float(re.search(r"[\d\.]+", clean).group()) * 1000
        elif 'M' in clean: return float(re.search(r"[\d\.]+", clean).group())
        elif 'T' in clean: return float(re.search(r"[\d\.]+", clean).group()) * 1000000
        return 0.0
    except: return 0.0

def parse_upside_percentage(upside_str: str) -> float:
    matches = re.findall(r'(\d+(?:\.\d+)?)%', upside_str)
    if not matches: return 0.0
    values = [float(x) for x in matches]
    return sum(values) / len(values)

# --- QUANTITATIVE ENGINE (YFINANCE + ATR) ---
def enrich_with_technical_data(catalyst: Catalyst) -> Optional[Catalyst]:
    """
    Fetches live data from yfinance, calculates ATR (14), 
    and updates price, stop loss, and target.
    """
    print(f"[*] Fetching technical data for {catalyst.ticker}...")
    try:
        ticker = yf.Ticker(catalyst.ticker)
        
        # 1. Get Historical Data (Need ~1 month to calculate 14-day ATR safely)
        hist = ticker.history(period="1mo")
        
        if hist.empty or len(hist) < 15:
            print(f"[!] Insufficient data for {catalyst.ticker}. Skipping.")
            return None

        # 2. Get Real-Time Price (Last Close or Current)
        # Use the last row of history as the most recent confirmed data point
        current_price = hist['Close'].iloc[-1]
        
        # 3. Calculate ATR (14)
        high_low = hist['High'] - hist['Low']
        high_close = (hist['High'] - hist['Close'].shift()).abs()
        low_close = (hist['Low'] - hist['Close'].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=14).mean().iloc[-1]

        # 4. Calculate Risk Levels
        stop_loss = current_price - (1.5 * atr)
        target_price = current_price + (3.0 * atr)

        # 5. Update Catalyst Object
        catalyst.current_price = round(current_price, 2)
        catalyst.atr_value = round(atr, 2)
        catalyst.target_price = round(target_price, 2)
        
        # Overwrite AI strings with Math-based values
        catalyst.stop_loss_trigger = f"${stop_loss:.2f} (Trailing 1.5 ATR)"
        
        # Append target to upside description
        catalyst.expected_upside = f"{catalyst.expected_upside} | Target: ${target_price:.2f} (3.0 ATR)"

        return catalyst

    except Exception as e:
        print(f"[!] Error processing {catalyst.ticker}: {e}")
        return None

# --- PERFORMANCE LOGGER ---
def log_alert_to_csv(catalyst: Catalyst):
    os.makedirs(os.path.dirname(PERFORMANCE_LOG_FILE), exist_ok=True)
    file_exists = os.path.isfile(PERFORMANCE_LOG_FILE)
    
    ny_tz = pytz.timezone('America/New_York')
    ny_time = datetime.now(ny_tz).strftime('%Y-%m-%d %H:%M:%S')
    
    # Extract numeric stop loss for logging if possible
    stop_val = catalyst.stop_loss_trigger.split(' ')[0]

    row_data = [
        ny_time,
        catalyst.ticker,
        catalyst.current_price, # Verified API Price
        catalyst.conviction_score,
        catalyst.market_cap,
        catalyst.atr_value,     # Volatility Metric
        stop_val,               # Calculated Stop
        catalyst.target_price,  # Calculated Target
        catalyst.thesis,
        catalyst.absorption_status,
        'OPEN'
    ]
    
    with open(PERFORMANCE_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if not file_exists:
            headers = [
                "Timestamp_NY", "Ticker", "Entry_Price", "Conviction", 
                "Market_Cap", "ATR_14", "Stop_Loss", "Target_Price", 
                "Thesis", "Absorption_Status", "Status"
            ]
            writer.writerow(headers)
        writer.writerow(row_data)
    
    print(f"[*] Logged verified alert for {catalyst.ticker}")

# --- AGENT SETUP ---
def get_alpha_scout_response():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    now = datetime.now()
    current_year = now.year
    three_days_ago = (now - timedelta(days=3)).strftime('%Y-%m-%d')
    today_str = now.strftime('%Y-%m-%d')

    system_instruction = f"""
    Role: You are “Alpha Scout,” a Quant-Fundamental Analyst.
    STRATEGY: PEAD, Biotech PDUFA, Insider Aggression.
    FILTERS:
    1. 8% RULE: Discard if stock is already up >8% today.
    2. Liquidity: Market Cap $500M - $10B.
    3. Source: SEC Edgar, FDA.
    FORBIDDEN DATES: Ignore results from {current_year - 1} or earlier. Today is {today_str}.
    """

    prompt = f"""
    Current Date: {today_str}
    Find unpriced bullish catalysts between {three_days_ago} and {today_str}.
    Provide 'current_price' (estimate), 'absorption_status', and 'market_cap'.
    """

    tools = [types.Tool(google_search=types.GoogleSearch())]
    
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools,
            response_mime_type="application/json",
            response_schema=ScoutReport
        )
    )
    return response.parsed

# --- OUTPUT HANDLERS ---
def save_to_json(report: ScoutReport):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w") as f:
        f.write(report.model_dump_json(indent=2))

def format_telegram_message(catalyst: Catalyst) -> str:
    try:
        ny_tz = pytz.timezone("America/New_York")
        now_ny = datetime.now(ny_tz).strftime('%H:%M %Z')
    except:
        now_ny = datetime.now().strftime('%H:%M UTC')

    robinhood_link = f"https://robinhood.com/us/en/stocks/{catalyst.ticker}/"

    return (
        f"🚀 *Alpha Scout Signal: ${catalyst.ticker}*\n"
        f"✅ *API Verified Price:* ${catalyst.current_price:.2f}\n"
        f"📉 *Calculated Stop Loss (ATR-based):* {catalyst.stop_loss_trigger}\n"
        f"🎯 *Target:* ${catalyst.target_price:.2f}\n"
        f"🔗 *[Trade on Robinhood]({robinhood_link})*\n"
        f"━━━━━\n"
        f"*Conviction: {catalyst.conviction_score}/10* | *Mkt Cap:* {catalyst.market_cap}\n"
        f"*ATR (14d):* {catalyst.atr_value}\n"
        f"━━━━━\n"
        f"💡 *Thesis:*\n{catalyst.thesis}\n\n"
        f"⏳ *Absorption Status:*\n{catalyst.absorption_status}\n\n"
        f"📅 *Catalyst:*\n{catalyst.catalyst_details}\n\n"
        f"🔗 *Proof:* {catalyst.recency_proof}\n"
        f"_Generated at {now_ny}_"
    )

def send_telegram_alert(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
    }
    try:
        requests.post(url, json=payload).raise_for_status()
        print(f"[*] Telegram sent.")
    except Exception as e:
        print(f"[!] Failed to send Telegram: {e}")

# --- MAIN EXECUTION ---
def main():
    print("[-] Alpha Scout initializing (Hybrid Quant-Fundamental Mode)...")
    try:
        report = get_alpha_scout_response()
        
        if not report or not report.catalysts:
            print("[!] No catalysts found.")
            return
        
        print(f"[-] Analyzing {len(report.catalysts)} raw candidates...")
        valid_catalysts = []
        
        for c in report.catalysts:
            # 1. Basic Filters
            mcap_millions = parse_market_cap_to_millions(c.market_cap)
            is_liquid = mcap_millions >= 300
            
            if c.sentiment != "Bullish" or not is_liquid:
                continue

            # 2. QUANTITATIVE VERIFICATION (yfinance)
            # This overwrites price and sets ATR stop/targets
            enriched_c = enrich_with_technical_data(c)
            
            if not enriched_c:
                continue # Skip if API fails or data insufficient

            # 3. Final Scoring & Filtering
            # Check 8% Rule using verified data if possible (requires Open price, 
            # but we rely on AI's initial filter + current price check if needed)
            
            if enriched_c.conviction_score >= 7:
                valid_catalysts.append(enriched_c)

        if not valid_catalysts:
            print("[-] No catalysts met the Hybrid criteria.")
            return
        
        # Sort by Conviction
        valid_catalysts.sort(key=lambda c: c.conviction_score, reverse=True)
        top_picks = valid_catalysts[:3]
        
        # Save
        final_report = ScoutReport(catalysts=valid_catalysts)
        save_to_json(final_report)
        
        # Log & Alert
        for item in top_picks:
            log_alert_to_csv(item)
            msg = format_telegram_message(item)
            send_telegram_alert(msg)
            time.sleep(1)
        
        print(f"[*] Processed {len(valid_catalysts)} valid signals. Sent {len(top_picks)} alerts.")

    except Exception as e:
        print(f"[!] Critical Error: {e}")

if __name__ == "__main__":
    main()