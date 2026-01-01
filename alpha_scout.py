import os
import json
import time
import requests
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from google.genai import types

# --- CONFIGURATION ---
# Latest model as of December 2025: gemini-3-pro-preview
MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview") 
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DATA_FILE = "data/latest_report.json"

# --- DATA MODELS ---
class Catalyst(BaseModel):
    ticker: str = Field(..., description="Stock Ticker (e.g., AAPL)")
    conviction_score: int = Field(..., description="1-10 Score. 8+ requires hard date.")
    thesis: str = Field(..., description="1-2 sentence thesis.")
    catalyst_details: str = Field(..., description="Specific event details and timing.")
    sentiment: str = Field(..., description="Bullish, Bearish, or Mixed")
    prediction_market: str = Field(..., description="Source, Odds, and 24h change if available.")
    recency_proof: str = Field(..., description="Source Link and Timestamp.")
    risk: str = Field(..., description="Primary invalidation factor.")

class ScoutReport(BaseModel):
    catalysts: List[Catalyst]

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
    Role: You are “Alpha Scout,” a senior event‑driven analyst specializing in uncovering undervalued bullish catalysts with high profit potential, focusing on market inefficiencies where events are not yet fully priced in (e.g., pre-announcement rumors, sentiment mismatches, or small-cap drifts).

    Constraints:
    1. 72h Recency: ONLY items published in the last 72 hours. ABSOLUTELY NO items from {prev_year} or earlier.
    Verify the year of every source. If the current year is {current_year}, every catalyst MUST have been published in {current_year}.
    2. Source Priority: Start with SEC.gov, FDA.gov, Official IR, Tier-1 news (Reuters/Bloomberg), then supplement with X (Twitter) via searches like "site:x.com" for emerging narratives, and niche forums (e.g., Reddit via "site:reddit.com").
    3. Prediction Markets: MUST cross-reference with Polymarket or Kalshi odds, focusing on bullish probabilities (>50%) that suggest mispricing (e.g., odds higher than implied by stock options or analyst consensus). If no direct market, use proxies and estimate upside edge.
    4. Logic: Deduplicate news; Normalize names to Tickers. Prioritize diverse sectors with inefficiencies (e.g., biotech, small-caps, emerging tech). Focus on signals with evidence of incomplete pricing (e.g., low volume reaction, high short interest).
    5. Conviction: Rate 1–10 based on profit potential (upside asymmetry). Score 8+ requires a "Hard Date" or verifiable rumor, supportive prediction odds (>60% bullish), AND signs of mispricing (e.g., options skew or X buzz outpacing news).

    Task:
    Search for undervalued bullish catalysts from the last 72 hours that show strong signs of upside not yet fully priced in, such as pre-event rumors, positive surprises in inefficient markets, or emerging narratives on X/social media. Include only those with bullish sentiment and evidence of profit edge (e.g., historical post-event drifts averaging +10-20%, sentiment deltas indicating momentum buildup).
    Verify with prediction markets and quantify potential returns (e.g., based on analogs). Return a ranked list (by conviction descending) of 3-7 catalysts to avoid noise.
    """

    # Tool Configuration: Google Search
    tools = [types.Tool(google_search=types.GoogleSearch())]

    # Prompt Construction
    prompt = f"""
    Current Full Timestamp: {now.strftime('%Y-%m-%d %H:%M:%S')}
    Today is in the year {current_year}.

    Perform a deep sweep for undervalued bullish catalysts ONLY between {three_days_ago} and {today_str}, emphasizing pre-event buildups, mispricings, and emerging narratives likely to drive multi-day upside (e.g., rumor leaks, unusual SEC filings with X buzz, positive macro shifts in small-caps).

    CRITICAL:
    - You MUST ignore any search results from {prev_year}.
    - Many articles from late {prev_year} may appear in search results; you are FORBIDDEN from using them.
    - For each candidate, verify the publication year is {current_year}.
    - If a result is from "Dec 30" but the year is {prev_year}, DISCARD IT.
    - Discard bearish/mixed or fully priced-in events (e.g., if stock already up >10% post-news); only include those with incomplete absorption (e.g., flat/low volume reaction).

    1. Search for trending bullish signals on SEC, FDA, Reuters, Bloomberg, and X (use "site:x.com" + keywords like "rumor" or "leak" for narratives) in the last 72h of {current_year}. Target inefficiencies: small/mid-caps (market cap < $10B), high-vol sectors.
    2. For each candidate, SEARCH for prediction market odds on Polymarket/Kalshi, plus proxies like options activity (e.g., "unusual call volume {ticker}"), short interest, or X sentiment (e.g., "site:x.com {ticker} bullish" with engagement metrics).
    3. Identify mispricings: Compare odds to market reaction (e.g., if odds imply 70% upside but stock flat, flag as edge). Search for historical analogs (e.g., "similar {event} stock performance history") to estimate drifts/returns.
    4. Enhance with quantitative insights: Expected impact (e.g., "% price target upside per analysts"), risk-reward ratio, or narrative strength (e.g., X virality score via retweets/mentions).
    5. Compile into JSON schema, ensuring each has bullish sentiment, mispricing evidence, and profit rationale.
    """

    # Generate Content with Structured Output
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
        f"🚀 *{catalyst.ticker}* - *{catalyst.conviction_score}/10*\n"
        f"- *Thesis:* {catalyst.thesis}\n"
        f"- *Catalyst:* {catalyst.catalyst_details}\n"
        f"- *Sentiment:* {catalyst.sentiment}\n"
        f"- *Prediction Market:* {catalyst.prediction_market}\n"
        f"- *Recency Proof:* {catalyst.recency_proof}\n"
        f"- *Risk:* {catalyst.risk}"
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
        print(f"[*] Telegram sent for ticker.")
    except Exception as e:
        print(f"[!] Failed to send Telegram: {e}")

# --- MAIN EXECUTION ---
def main():
    print("[-] Alpha Scout initializing...")
    try:
        report = get_alpha_scout_response()
        
        if not report or not report.catalysts:
            print("[!] No catalysts found.")
            return

        # Filter for Conviction >= 7
        high_conviction = [c for c in report.catalysts if c.conviction_score >= 7]
        
        if not high_conviction:
            print("[-] No catalysts met the conviction threshold (>=7).")
            # We still save the full report for records, or you can choose to save empty
            save_to_json(report) 
            return

        # Save filtered list to JSON (or full list, depending on preference. 
        # Instructions say "Overwrite with the result", implying the filtered result or full result.
        # We will save the high conviction ones to be safe).
        filtered_report = ScoutReport(catalysts=high_conviction)
        save_to_json(filtered_report)

        # Send Telegram Messages
        for item in high_conviction:
            msg = format_telegram_message(item)
            send_telegram_alert(msg)
            time.sleep(1) # Rate limit safety

    except Exception as e:
        print(f"[!] Critical Error: {e}")
        # Optional: Send error log to Telegram

if __name__ == "__main__":
    main()