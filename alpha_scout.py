import os
import json
import time
import requests
import re
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from google.genai import types

# --- CONFIGURATION ---
# Ensure you are using a model capable of complex reasoning
MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview") 
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DATA_FILE = "data/latest_report.json"

# --- DATA MODELS ---
class Catalyst(BaseModel):
    ticker: str = Field(..., description="Stock Ticker (e.g., AAPL)")
    market_cap: str = Field(..., description="Market Cap string (e.g., '$450M', '$2.1B'). Essential for liquidity checks.")
    conviction_score: int = Field(..., description="1-10 Score. 8+ requires hard date and institutional backing.")
    thesis: str = Field(..., description="1-2 sentence thesis focusing on the inefficiency.")
    catalyst_details: str = Field(..., description="Specific event details and timing.")
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
    """Converts strings like '$450M' or '$2.1B' into millions (float)."""
    clean = cap_str.upper().replace('$', '').replace(',', '').strip()
    try:
        if 'B' in clean:
            return float(re.search(r"[\d\.]+", clean).group()) * 1000
        elif 'M' in clean:
            return float(re.search(r"[\d\.]+", clean).group())
        elif 'T' in clean:
            return float(re.search(r"[\d\.]+", clean).group()) * 1000000
        return 0.0
    except:
        return 0.0

def parse_upside_percentage(upside_str: str) -> float:
    """Extracts percentage. If range '10-20%', returns average (15.0)."""
    matches = re.findall(r'(\d+(?:\.\d+)?)%', upside_str)
    if not matches:
        return 0.0
    values = [float(x) for x in matches]
    return sum(values) / len(values)

# --- AGENT SETUP ---
def get_alpha_scout_response():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Dynamic Date Context
    now = datetime.now()
    current_year = now.year
    prev_year = current_year - 1
    three_days_ago = (now - timedelta(days=3)).strftime('%Y-%m-%d')
    today_str = now.strftime('%Y-%m-%d')

    # System Instructions
    system_instruction = f"""
    Role: You are “Alpha Scout,” a Quant-Fundamental Analyst. Your goal is to find "Asymmetric Upside" where the potential gain is 3x the risk.

    STRATEGY FOCUS:
    1. Post-Earnings Announcement Drift (PEAD): Look for companies that beat earnings >20% but price moved <5%.
    2. Biotech PDUFA Run-ups: Confirmed FDA dates 30-60 days out.
    3. Insider Aggression: Form 4 filings showing "Open Market" purchases >$100k by C-suite.
    4. Macro/Sector Rotation: Small-caps benefiting from immediate rate/policy shifts.

    STRICT FILTERS:
    1. "Priced-In" Check: DISCARD any stock that has already moved >8% on the day of the news. The alpha is gone.
    2. Source Weighting: Prioritize SEC Edgar (Form 4, 8-K) and FDA Calendars over social media.
    3. Liquidity: Focus on Market Caps between $500M and $10B (Mid/Small Cap). Avoid Micro-caps (<$300M).
    4. Risk Management: You must identify a specific 'Stop Loss Trigger' (technical level or news event).

    FORBIDDEN DATES:
    - Today is {today_str} ({current_year}).
    - You MUST strictly ignore any search results from {prev_year} or earlier.
    - Verify every timestamp. If a result says "Dec 28" but the year is {prev_year}, DISCARD IT.
    """

    # Tool Configuration
    tools = [types.Tool(google_search=types.GoogleSearch())]

    # Prompt Construction
    prompt = f"""
    Current Date: {today_str}
    
    Perform a deep sweep for unpriced bullish catalysts published ONLY between {three_days_ago} and {today_str}.

    Execution Steps:
    1. Search SEC filings (Form 4, 8-K) and FDA databases for events in the last 72h.
    2. Filter for "Quiet Winners": Stocks with good news but low relative volume or muted price action (<5% move).
    3. Cross-reference with Prediction Markets (Polymarket/Kalshi) or Options Flow (Call/Put skew).
    4. For each candidate, explicitly check the Market Cap.
    5. Compile the JSON report.
    
    CRITICAL: 
    - Ensure 'earnings_date' is accurate. If earnings are within 7 days, flag as HIGH RISK.
    - Ensure 'market_cap' is formatted like '$500M' or '$2.5B'.
    """

    # Generate Content
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
    return (
        f"🚀 *{catalyst.ticker}* ({catalyst.market_cap}) - *{catalyst.conviction_score}/10*\n"
        f"📊 *Vol:* {catalyst.relative_volume} | *Upside:* {catalyst.expected_upside}\n\n"
        f"💡 *Thesis:* {catalyst.thesis}\n"
        f"📅 *Catalyst:* {catalyst.catalyst_details}\n"
        f"🛑 *Stop Loss:* {catalyst.stop_loss_trigger}\n"
        f"⚠️ *Earnings:* {catalyst.earnings_date}\n"
        f"🎲 *Pred. Market:* {catalyst.prediction_market}\n"
        f"🔗 *Proof:* {catalyst.recency_proof}\n"
        f"📉 *Risk:* {catalyst.risk}\n"
        f"🧠 *Mispricing:* {catalyst.mispricing_evidence}"
    )

def send_telegram_alert(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[!] Telegram credentials not found. Skipping message.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
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
        
        filtered_catalysts = []
        
        print(f"[-] Analyzing {len(report.catalysts)} raw candidates...")

        for c in report.catalysts:
            # 1. Parse Metrics
            upside_val = parse_upside_percentage(c.expected_upside)
            mcap_millions = parse_market_cap_to_millions(c.market_cap)
            
            # 2. Liquidity Guard (Sweet Spot: $500M - $10B)
            # We allow slightly outside this range if conviction is extremely high (9+), 
            # otherwise we strictly enforce >$300M to avoid penny stocks.
            is_liquid_enough = mcap_millions >= 300 
            is_sweet_spot = 500 <= mcap_millions <= 10000
            
            # 3. Scoring Logic
            # Boost score if in sweet spot
            final_score = c.conviction_score
            if is_sweet_spot:
                final_score += 0.5
            
            # 4. Filter Criteria
            # - Must be Bullish
            # - Must have >8% upside potential
            # - Must be liquid enough
            # - Score must be >= 7.5 (adjusted)
            if (c.sentiment == "Bullish" and 
                upside_val >= 8 and 
                is_liquid_enough and 
                final_score >= 7.5):
                
                # Update the object with the adjusted score (optional, for sorting)
                c.conviction_score = int(final_score) if final_score > 10 else int(final_score)
                filtered_catalysts.append(c)

        if not filtered_catalysts:
            print("[-] No catalysts met the strict Quant-Fundamental criteria.")
            save_to_json(report) # Save raw data for debugging
            return
        
        # Sort: Highest Conviction first
        filtered_catalysts.sort(key=lambda c: c.conviction_score, reverse=True)
        
        # Select Top 3
        top_picks = filtered_catalysts[:3]
        
        # Save Report
        final_report = ScoutReport(catalysts=filtered_catalysts)
        save_to_json(final_report)
        
        # Alert
        for item in top_picks:
            msg = format_telegram_message(item)
            send_telegram_alert(msg)
            time.sleep(1)
        
        print(f"[*] Processed {len(filtered_catalysts)} valid signals. Sent {len(top_picks)} alerts.")

    except Exception as e:
        print(f"[!] Critical Error: {e}")

if __name__ == "__main__":
    main()