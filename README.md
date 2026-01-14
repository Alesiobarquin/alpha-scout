# Stock Scout Terminal

A hybrid Quant-Fundamental stock scouting platform that identifies unpriced bullish catalysts using AI and validates them with technical analysis.

 **Quick Link:** [Live Deployment](https://alpha-scout-mrl6d3cjejyksiwqdt94rq.streamlit.app)

---

##  Overview

Stock Scout identifies high-conviction trading opportunities by combining large-scale news/event analysis (Fundamental) with volatility-adjusted technical levels (Quant). It specifically looks for:
- **PEAD** (Post-Earnings Announcement Drift)
- **Biotech PDUFA** dates
- **Insider Aggression**
- **Unpriced Catalysts** with a focus on market absorption.

##  Key Features

- **AI-Powered Scouting**: Uses Gemini to scan for catalysts and generate investment theses.
- **Quantitative Enrichment**: Automatically calculates **ATR-based Stop Loss** and **2:1 Reward/Risk Targets** using live market data via `yfinance`.
- **Live Terminal**: A Streamlit dashboard to track active signals, real-time price action, and performance history.
- **Automated Alerting**: Integration with Telegram for real-time signal notifications.
- **Persistent Logging**: Tracks every signal in a performance log for long-term auditing and P/L calculation.

##  Tech Stack

- **Core**: Python 3.3
- **Frontend**: Streamlit
- **AI**: Google Gemini API (`gemini-3-pro-preview`)
- **Market Data**: Yahoo Finance (`yfinance`)
- **Data Engineering**: Pandas, JSON/CSV
- **Automation**: GitHub Actions (Daily Scans)

##  Getting Started

### Prerequisites

1.  **Gemini API Key**: Obtain from [Google AI Studio](https://aistudio.google.com/).
2.  **Telegram Bot**: (Optional) For alerts, create a bot via `@BotFather`.
3.  **Python 3.10+**

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/stock-scout.git
cd stock-scout

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file or export the following:
```bash
export GEMINI_API_KEY="your_api_key"
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

### Usage

#### 1. Run the Scout
This script performs the catalyst search, technical enrichment, logs the data, and sends alerts.
```bash
python alpha_scout.py
```

#### 2. Launch the Terminal
Visualize the active signals and history.
```bash
streamlit run app.py
```

## 📂 Project Structure

- `alpha_scout.py`: The "Brain". Handles scouting, technical calculations, and logging.
- `app.py`: The "Interface". Streamlit dashboard for real-time monitoring.
- `data/`: Contains `latest_report.json` (active signal) and `performance_log.csv` (historical data).
- `.github/workflows/`: Contains `daily_scan.yml` for automated daily execution.

---

*Disclaimer: This tool is for educational and research purposes only. Not financial advice.*
