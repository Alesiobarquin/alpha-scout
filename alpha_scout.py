import os
import json
import time
import requests
import re
import csv
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from google.genai import types
import pytz

# --- CONFIGURATION ---
# Use a model capable of complex reasoning (Gemini 1.5 Pro or newer)
MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-1.5-pro-002") 
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DATA_FILE = "data/latest_report.json"
PERFORMANCE_LOG_FILE = "data/performance_log.csv"

# --- DATA MODELS (ENHANCED) ---
class Catalyst(BaseModel):
    ticker: str = Field(..., description="Stock Ticker (e.g., AAPL)")
    current_price: float = Field(..., description="The stock price at the time of alert generation.")
    market_cap: str = Field(..., description="Market Cap string (e.g., '$450M', '$2.1B').")
    conviction_score: int = Field(..., description="1-10 Score. 8+ requires hard date and institutional backing.")
    thesis: str = Field(..., description="1-2 sentence thesis focusing on the inefficiency.")
    catalyst_details: str = Field(..., description="Specific event details and timing.")
    absorption_status: str = Field(..., description="Explicit reason why the market hasn't fully reacted (e.g., 'News broke post-market', 'Low relative volume', 'Overshadowed by macro').")
    earnings_date: str = Field(..., description="Next earnings date. Mark 'Past' if recently reported, or specific date.")
    relative_volume: str = Field(..., description="Current vol vs 30d avg (e.g., '2.5x 30d avg').")
    stop_loss_trigger: str = Field(..., description="Specific price or event that invalidates the thesis.")
    sentiment: str = Field(..., description="Bullish, Bearish, or Mixed")
    prediction_market: str = Field(..., description="Source, Odds, and 24h change if available.")
    recency_proof: str = Field(..., description="Source Link and Timestamp.")
    risk: str = Field(..., description="Primary invalidation factor.")
    expected_upside: str = Field(..., description="Quantified potential (e.g., '10-20% drift').")
    mispricing_evidence: str = Field(..., description="Why not priced in (e.g., 'Price flat despite news').")
    x_sentiment: Optional[str] = Field(None, description="X buzz summary.")

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

# --- PERFORMANCE LOGGER (NY TIME & CSV SAFETY) ---
def log_alert_to_csv(catalyst: Catalyst):
    """Appends a catalyst alert to the performance log CSV in NY Time."""
    os.makedirs(os.path.dirname(PERFORMANCE_LOG_FILE), exist_ok=True)
    file_exists = os.path.isfile(PERFORMANCE_LOG_FILE)
    
    # Get NY Time
    ny_tz = pytz.timezone('America/New_York')
    ny_time = datetime.now(ny_tz).strftime('%Y-%m-%d %H:%M:%S')
    
    row_data = [
        ny_time,
        catalyst.ticker,
        catalyst.current_price,
        catalyst.conviction_score,
        catalyst.market_cap,
        catalyst.expected_upside,
        catalyst.thesis,
        catalyst.absorption_status,
        'OPEN'
    ]
    
    # Use QUOTE_ALL to handle commas in thesis/status safely
    with open(PERFORMANCE_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if not file_exists:
            headers = ["Timestamp_NY", "Ticker", "Entry_Price", "Conviction", "Market_Cap", "Expected_Upside", "Thesis", "Absorption_Status", "Status"]
            writer.writerow(headers)
        writer.writerow(row_data)
    
    print(f"[*] Logged alert for {catalyst.ticker} to {PERFORMANCE_LOG_FILE}")

# --- AGENT SETUP ---
def get_alpha_scout_response():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Date Context
    now = datetime.now()
    current_year = now.year
    three_days_ago = (now - timedelta(days=3)).strftime('%Y-%m-%d')
    today_str = now.strftime('%Y-%m-%d')

    system_instruction = f"""
    Role: You are “Alpha Scout,” a Quant-Fundamental Analyst finding "Asymmetric Upside."
    
    STRATEGY FOCUS:
    1. Post-Earnings Announcement Drift (PEAD): Companies beating earnings >20% but price moved <5%.
    2. Biotech PDUFA Run-ups: Confirmed FDA dates 30-60 days out.
    3. Insider Aggression: Form 4 filings showing "Open Market" purchases >$100k.
    4. Macro Rotations: Small-caps benefiting from immediate policy shifts.

    STRICT FILTERS:
    1. THE 8% RULE: Check the stock's performance TODAY. If it is ALREADY up >8% on the news, DISCARD IT. The alpha is gone. We only want "Slow-Reaction" events.
    2. Liquidity: Focus on Market Caps between $500M and $10B. Avoid Micro-caps (<$300M).
    3. Source Weighting: Prioritize SEC Edgar (Form 4, 8-K) and FDA Calendars.
    
    FORBIDDEN DATES: Today is {today_str} ({current_year}). You MUST strictly ignore any search results from {current_year - 1} or earlier.
    """

    prompt = f"""
    Current Date: {today_str}
    
    Perform a deep sweep for unpriced bullish catalysts published ONLY between {three_days_ago} and {today_str}.
    
    CRITICAL REQUIREMENTS:
    1. Find the 'current_price' for every candidate.
    2. Analyze the 'absorption_status': Why hasn't the price spiked yet? (e.g., "News broke post-market", "Volume is 50% below average", "Overshadowed by CPI data").
    3. Verify the "8% Rule": If the stock is up 15% today, do NOT include it.
    
    Compile the JSON report.
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
    print(f"[*] Data saved to {DATA_FILE}")

def format_telegram_message(catalyst: Catalyst) -> str:
    """Formats the catalyst data into a highly readable Telegram message with UX enhancements."""
    try:
        ny_tz = pytz.timezone("America/New_York")
        now_ny = datetime.now(ny_tz).strftime('%H:%M %Z')
    except Exception:
        now_ny = datetime.now().strftime('%H:%M UTC')

    robinhood_link = f"https://robinhood.com/us/en/stocks/{catalyst.ticker}/"

    return (
        f"🚀 *Alpha Scout Signal: ${catalyst.ticker}*\n"
        f"📈 *Price:* ${catalyst.current_price:.2f} | *Mkt Cap:* {catalyst.market_cap}\n"
        f"🔗 *[Trade on Robinhood]({robinhood_link})*\n"
        f"━━━━━\n"
        f"*Conviction: {catalyst.conviction_score}/10*\n"
        f"*Upside:* {catalyst.expected_upside}\n"
        f"━━━━━\n"
        f"💡 *Thesis:*\n{catalyst.thesis}\n\n"
        f"⏳ *Absorption Status:*\n{catalyst.absorption_status}\n\n"
        f"📅 *Catalyst:*\n{catalyst.catalyst_details}\n\n"
        f"🛑 *Stop Loss:*\n{catalyst.stop_loss_trigger}\n"
        f"━━━━━\n"
        f"🔗 *Proof:* {catalyst.recency_proof}\n"
        f"_Generated at {now_ny}_"
    )

def send_telegram_alert(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[!] Telegram credentials not found. Skipping message.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True  # Keeps the alert clean
    }
    
    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        print(f"[*] Telegram sent.")
    except Exception as e:
        print(f"[!] Failed to send Telegram: {e}")

# --- MAIN EXECUTION ---
def main():
    print("[-] Alpha Scout initializing (Quant-Fundamental Mode)...")
    try:
        report = get_alpha_scout_response()
        
        if not report or not report.catalysts:
            print("[!] No catalysts found.")
            return
        
        print(f"[-] Analyzing {len(report.catalysts)} raw candidates...")
        filtered_catalysts = []
        
        for c in report.catalysts:
            # 1. Parse Metrics
            upside_val = parse_upside_percentage(c.expected_upside)
            mcap_millions = parse_market_cap_to_millions(c.market_cap)
            
            # 2. Liquidity Guard
            is_liquid_enough = mcap_millions >= 300
            is_sweet_spot = 500 <= mcap_millions <= 10000
            
            # 3. Scoring Logic
            final_score = c.conviction_score
            if is_sweet_spot:
                final_score += 0.5
            
            # 4. Filter Criteria
            if (c.sentiment == "Bullish" and 
                upside_val >= 8 and 
                is_liquid_enough and 
                final_score >= 7.5):
                
                c.conviction_score = min(10, int(final_score))
                filtered_catalysts.append(c)

        if not filtered_catalysts:
            print("[-] No catalysts met the strict Quant-Fundamental criteria.")
            save_to_json(report)
            return
        
        # Sort: Highest Conviction first
        filtered_catalysts.sort(key=lambda c: c.conviction_score, reverse=True)
        
        # Select Top 3
        top_picks = filtered_catalysts[:3]
        
        # Save Report
        final_report = ScoutReport(catalysts=filtered_catalysts)
        save_to_json(final_report)
        
        # Log and Alert
        for item in top_picks:
            log_alert_to_csv(item)
            msg = format_telegram_message(item)
            send_telegram_alert(msg)
            time.sleep(1)
        
        print(f"[*] Processed {len(filtered_catalysts)} valid signals. Sent {len(top_picks)} alerts.")

    except Exception as e:
        print(f"[!] Critical Error: {e}")

if __name__ == "__main__":
    main()