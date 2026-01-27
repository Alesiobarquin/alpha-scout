import os
import json
import time
import requests
import re
import csv
import pandas as pd
import yfinance as yf
import subprocess
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from google.genai import types
import pytz
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

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
    market_cap: str = Field(..., description="Market Cap string.")
    conviction_score: int = Field(..., description="1-10 Score.")
    thesis: str = Field(..., description="1-2 sentence thesis.")
    catalyst_details: str = Field(..., description="Event details.")
    absorption_status: str = Field(..., description="Why market hasn't reacted.")
    earnings_date: str = Field(..., description="Next earnings date.")
    relative_volume: str = Field(..., description="Vol vs 30d avg.")
    stop_loss_trigger: str = Field(..., description="AI suggested stop (text description).")
    sentiment: str = Field(..., description="Bullish/Bearish.")
    prediction_market: str = Field(..., description="Odds.")
    recency_proof: str = Field(..., description="Link/Timestamp.")
    risk: str = Field(..., description="Risk factor.")
    expected_upside: str = Field(..., description="Upside potential.")
    mispricing_evidence: str = Field(..., description="Evidence.")
    x_sentiment: Optional[str] = Field(None, description="X buzz.")
    
    # Technical Fields (Populated by yfinance)
    atr_value: Optional[float] = None
    calculated_stop_loss: Optional[float] = None
    calculated_target: Optional[float] = None

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

# --- QUANTITATIVE ENGINE ---
def enrich_with_technical_data(catalyst: Catalyst) -> Optional[Catalyst]:
    """Fetches live data, calculates ATR, updates price/stop/target."""
    print(f"[*] Fetching technical data for {catalyst.ticker}...")
    try:
        ticker = yf.Ticker(catalyst.ticker)
        hist = ticker.history(period="1mo")
        
        if hist.empty or len(hist) < 15:
            print(f"[!] Insufficient data for {catalyst.ticker}. Skipping.")
            return None

        # Real-Time Price
        current_price = hist['Close'].iloc[-1]
        
        # ATR (14) Calculation
        high_low = hist['High'] - hist['Low']
        high_close = (hist['High'] - hist['Close'].shift()).abs()
        low_close = (hist['Low'] - hist['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=14).mean().iloc[-1]

        # Risk Calculation (2:1 Reward/Risk)
        stop_loss = current_price - (1.5 * atr)
        target_price = current_price + (3.0 * atr)

        # Update Object
        catalyst.current_price = round(current_price, 2)
        catalyst.atr_value = round(atr, 2)
        catalyst.calculated_stop_loss = round(stop_loss, 2)
        catalyst.calculated_target = round(target_price, 2)
        
        # Update display strings for Telegram
        catalyst.stop_loss_trigger = f"${stop_loss:.2f} (1.5 ATR)"
        catalyst.expected_upside = f"{catalyst.expected_upside} | Target: ${target_price:.2f}"

        return catalyst
    except Exception as e:
        print(f"[!] Error processing {catalyst.ticker}: {e}")
        return None

# --- PERSISTENT LOGGING LOGIC ---
def log_to_performance_csv(catalyst: Catalyst):
    """
    Logs the signal to CSV with strictly numeric columns for frontend math.
    """
    os.makedirs(os.path.dirname(PERFORMANCE_LOG_FILE), exist_ok=True)
    
    # 1. Duplicate Prevention
    if os.path.exists(PERFORMANCE_LOG_FILE):
        try:
            df = pd.read_csv(PERFORMANCE_LOG_FILE)
            # Check if ticker exists AND status is OPEN
            if not df.empty and 'Ticker' in df.columns and 'Status' in df.columns:
                is_active = df[(df['Ticker'] == catalyst.ticker) & (df['Status'] == 'OPEN')]
                if not is_active.empty:
                    print(f"[-] Skipping Log: {catalyst.ticker} is already OPEN in {PERFORMANCE_LOG_FILE}")
                    return
        except Exception as e:
            print(f"[!] Warning: Could not read existing CSV for duplicate check: {e}")

    # 2. Prepare Data (Strictly Numeric for Math Columns)
    ny_tz = pytz.timezone('America/New_York')
    ny_time = datetime.now(ny_tz).strftime('%Y-%m-%d %H:%M:%S')
    
    row_data = [
        ny_time,
        catalyst.ticker,
        catalyst.current_price,        # Float
        catalyst.conviction_score,     # Int
        catalyst.market_cap,           # String
        catalyst.atr_value,            # Float
        catalyst.calculated_stop_loss, # Float (No text!)
        catalyst.calculated_target,    # Float (No text!)
        catalyst.thesis,
        'OPEN'
    ]
    
    headers = [
        "Date", "Ticker", "Entry_Price", "Conviction", 
        "Market_Cap", "ATR_Value", "Stop_Loss", "Target_Price", 
        "Thesis", "Status"
    ]

    # 3. Write to CSV (Append Mode)
    file_exists = os.path.isfile(PERFORMANCE_LOG_FILE)
    try:
        with open(PERFORMANCE_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row_data)
        print(f"[*] Successfully logged {catalyst.ticker} to {PERFORMANCE_LOG_FILE}")
    except IOError as e:
        print(f"[!] Failed to write to CSV: {e}")



# --- AGENT SETUP ---
def get_alpha_scout_response():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    now = datetime.now()
    today_str = now.strftime('%Y-%m-%d')
    three_days_ago = (now - timedelta(days=3)).strftime('%Y-%m-%d')

    system_instruction = """
    Role: You are “Stock Scout,” a Quant-Fundamental Analyst.
    STRATEGY: PEAD, Biotech PDUFA, Insider Aggression.
    FILTERS:
    1. 8% RULE: Discard if stock is already up >8% today.
    2. Liquidity: Market Cap $500M - $10B.
    FORBIDDEN DATES: Ignore results from previous years.
    """

    prompt = f"""
    Current Date: {today_str}
    Find unpriced bullish catalysts between {three_days_ago} and {today_str}.
    Provide 'current_price' (estimate), 'absorption_status', and 'market_cap'.
    """

    tools = [types.Tool(google_search=types.GoogleSearch())]
    
    max_retries = 5
    base_delay = 5

    for attempt in range(max_retries):
        try:
            print(f"[-] AI Query Attempt {attempt+1}/{max_retries}...")
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
            return response.parsed, False
        except Exception as e:
            error_str = str(e).lower()
            if attempt < max_retries - 1:
                # Retry on: 503 (Overloaded), 429 (Rate Limit), Disconnected, Timeout
                if any(x in error_str for x in ["503", "overloaded", "disconnected", "timeout", "network"]):
                    wait_time = base_delay * (2 ** attempt) # Exponential backoff
                    print(f"[!] Network/API Error ({error_str[:50]}...). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif "429" in error_str:
                    wait_time = 30
                    print(f"[!] Rate limit (429). Cooling down for {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            
            # If we're out of retries for the primary model, try the BACKUP
            if attempt == max_retries - 1:
                print(f"[!] Primary model {MODEL_ID} failed after {max_retries} attempts.")
                print(f"[*] Switching to Fallback Model: gemini-3.0-flash...")
                try:
                    response = client.models.generate_content(
                        model="gemini-3.0-flash-preview",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            tools=tools,
                            response_mime_type="application/json",
                            response_schema=ScoutReport
                        )
                    )
                    return response.parsed, True
                except Exception as backup_e:
                    print(f"[!] Critical: Backup model also failed: {backup_e}")
                    raise backup_e
            
            # If it's a non-retryable error (not caught above) or logic fails
            print(f"[!] API call failed: {e}")
            raise e

# --- OUTPUT HANDLERS ---
def save_to_json(report: ScoutReport):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w") as f:
        f.write(report.model_dump_json(indent=2))

def format_telegram_message(catalyst: Catalyst, is_fallback: bool = False) -> str:
    robinhood_link = f"https://robinhood.com/us/en/stocks/{catalyst.ticker}/"
    try:
        ny_time = datetime.now(pytz.timezone("America/New_York")).strftime('%H:%M %Z')
    except:
        ny_time = datetime.now().strftime('%H:%M UTC')

    model_note = " (Backup Model)" if is_fallback else ""

    return (
        f"🚀 *Stock Scout Signal: ${catalyst.ticker}*{model_note}\n"
        f"✅ *API Verified Price:* ${catalyst.current_price:.2f}\n"
        f"📉 *Calculated Stop Loss (ATR):* ${catalyst.calculated_stop_loss:.2f}\n"
        f"🎯 *Target:* ${catalyst.calculated_target:.2f}\n"
        f"🔗 *[Trade on Robinhood]({robinhood_link})*\n"
        f"━━━━━\n"
        f"*Conviction: {catalyst.conviction_score}/10* | *Mkt Cap:* {catalyst.market_cap}\n"
        f"*ATR (14d):* {catalyst.atr_value}\n"
        f"━━━━━\n"
        f"💡 *Thesis:*\n{catalyst.thesis}\n\n"
        f"⏳ *Absorption Status:*\n{catalyst.absorption_status}\n\n"
        f"📅 *Catalyst:*\n{catalyst.catalyst_details}\n\n"
        f"🔗 *Proof:* {catalyst.recency_proof}\n"
        f"_Generated at {ny_time}_"
    )

def send_telegram_alert(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown", "disable_web_page_preview": True}
    try: requests.post(url, json=payload).raise_for_status()
    except Exception as e: print(f"[!] Telegram Error: {e}")

# --- MAIN EXECUTION ---
def main():
    print("[-] Stock Scout initializing (Hybrid Quant-Fundamental Mode)...")
    try:
        report, is_fallback = get_alpha_scout_response()
        if not report or not report.catalysts:
            print("[!] No catalysts found.")
            return
        
        print(f"[-] Analyzing {len(report.catalysts)} raw candidates...")
        valid_catalysts = []
        
        for c in report.catalysts:
            # 1. Basic Filters
            mcap_millions = parse_market_cap_to_millions(c.market_cap)
            is_liquid = mcap_millions >= 300
            if c.sentiment != "Bullish" or not is_liquid: continue

            # 2. QUANTITATIVE VERIFICATION
            enriched_c = enrich_with_technical_data(c)
            if not enriched_c: continue

            # 3. Final Scoring
            if enriched_c.conviction_score >= 7:
                valid_catalysts.append(enriched_c)

        if not valid_catalysts:
            print("[-] No catalysts met the Hybrid criteria.")
            return
        
        valid_catalysts.sort(key=lambda c: c.conviction_score, reverse=True)
        top_picks = valid_catalysts[:3]
        
        save_to_json(ScoutReport(catalysts=valid_catalysts))
        
        for item in top_picks:
            # LOG FIRST (Persistent & Duplicate Checked)
            log_to_performance_csv(item)
            # THEN ALERT
            msg = format_telegram_message(item, is_fallback=is_fallback)
            send_telegram_alert(msg)
            time.sleep(1)
        
        print(f"[*] Processed {len(valid_catalysts)} valid signals. Sent {len(top_picks)} alerts.")
        


    except Exception as e:
        print(f"[!] Critical Error: {e}")

if __name__ == "__main__":
    main()