# User Manual — Institutional Options Intelligence Platform

## What This Platform Does

This is a web-based dashboard that scans the stock and options market to help you identify trading opportunities. It pulls live data from Yahoo Finance, applies technical analysis, and surfaces the most actionable options contracts across 70+ tickers. It is designed for traders who want a structured daily workflow for options ideas — not a prediction machine, but a systematic filter for what deserves attention.

---

## Getting Started

### Opening the Dashboard

Open your browser and navigate to the Streamlit URL (typically `http://localhost:8501` if running locally). The dashboard loads with six tabs across the top.

### When to Run

| Time | Day | What to Run |
|---|---|---|
| 9:45 – 10:15 AM ET | Tue–Thu | Strategy Ideas, Market Screener |
| 11:00 – 11:30 AM ET | Tue–Thu | Strategy Ideas (confirm signals) |
| 10:00 – 11:00 AM ET | Monday | Strategy Ideas (weekly bias) |
| Avoid | Any | 9:30–9:40 AM (erratic open noise) |
| Avoid | Friday | After 2 PM (premium crush) |
| Avoid | Weekends | Data is stale/unavailable |

---

## Tab 1 — Market Options Screener

**What it does:** Shows the most actively traded option contracts across your core ticker list. This is a raw look at where volume and open interest are concentrating right now.

### How to Use

1. Click **Run Market Options Screener**
2. Adjust the filters:
   - **Call or Put** — filter to just calls or puts, or leave as All
   - **In The Money** — filter to ITM, OTM, or both
   - **Minimum Volume** — hide low-activity contracts (default: 1)
   - **Contracts Per Ticker** — how many top contracts to fetch per ticker (3–20)
3. The table and scatter plot update automatically when you change filters — you do not need to re-run

### Column Meanings

| Column | Meaning |
|---|---|
| Ticker | The underlying stock |
| Expiration | Date the contract expires |
| Call or Put | Contract direction |
| Strike | Price at which the option can be exercised |
| Last Price | Most recent trade price of the option (may be stale for illiquid contracts) |
| Bid / Ask | Current buy/sell prices — the spread between them is your transaction cost |
| Volume | Number of contracts traded today |
| Open Interest | Total outstanding contracts |
| In The Money | Yes = strike is favorable vs current stock price |

### Scatter Plot

X-axis = strike price, Y-axis = option price, bubble size = volume. Calls in one colour, puts in another. Large bubbles at certain strikes show where the market is concentrating — useful for identifying key price levels.

---

## Tab 2 — Market Volume Leaders

**What it does:** Ranks tickers by total options volume (calls + puts combined) for today, with a 7-day trend. Use this to find where institutional activity is heaviest.

### How to Use

1. Click **Run Market Volume Leaders**
2. Adjust **Volume Leaders Rows** slider to show more or fewer rows

### Column Meanings

| Column | Meaning |
|---|---|
| Ticker | Stock symbol |
| Calls or Puts | Which side has more volume today |
| 1 Day Volume | Total options contracts traded today |
| % of Scanned Day Total | This ticker's share of the total volume across all 70+ scanned tickers |
| Last 1 Week Total | Rolling 7-day volume total |
| 1 Week Trend | Is volume increasing, decreasing, or flat vs. the prior half-week? |
| Call Volume | Calls-only today |
| Put Volume | Puts-only today |

### What to Look For

- Tickers with **high % of day total** are getting unusual institutional attention
- A ticker showing **"increasing"** trend combined with high call volume = potential bullish setup
- **"increasing"** trend with high put volume = potential defensive hedging or bearish bet

---

## Tab 3 — Rapid Movers

**What it does:** Scores every ticker in the scanned universe for near-term directional potential (1-week and 1-month views). Shows the top candidates sorted by score.

### How to Use

1. Click **Run Rapid Movers**
2. Adjust **Mover Rows** slider (10–20)

### Column Meanings

| Column | Meaning |
|---|---|
| Ticker | Stock symbol |
| Last Price | Most recent daily closing price |
| 1 Week View | "Grow Rapidly" or "Fall Steeply" — the model's 1-week directional bias |
| 1 Week Score | Numeric score behind the view. Higher = stronger signal |
| 1 Month View | Same, but for a 1-month horizon |
| 1 Month Score | Numeric score for the 1-month view |
| 5 Day Return % | The stock's actual return over the past 5 trading days |
| 5 Day Range % | Average intraday high-low range over 5 days — a volatility proxy |
| Volume Ratio | Today's volume vs the 5-day average. Above 1.0 means elevated activity |

### What to Look For

- **High score + "Grow Rapidly" + volume ratio > 1.2** = strong bullish candidate
- **High score + "Fall Steeply" + volume ratio > 1.2** = strong bearish candidate
- Low 5 Day Range with high score = trend move without much noise — cleaner entry

---

## Tab 4 — Strategy Ideas

**What it does:** The main feature. Combines the Rapid Movers signals with options volume data to produce concrete, actionable trade ideas: a specific contract (ticker, expiration, strike, call/put), confidence score, and complete entry/exit rules.

### How to Use

1. Click **Run Strategy**
2. Adjust **Strategy Rows** slider (5–15)
3. Read the status caption to understand data quality (see below)

### Status Messages

| Status | Meaning |
|---|---|
| Fresh data | Full pipeline ran successfully |
| Partial results | Some tickers timed out or had no contracts — fewer ideas than requested |
| data_unavailable | Either no market data or all signals failed confidence threshold |
| no_contracts | Signals found but no options passed the liquidity filters (common on weekends) |

### Main Table Columns

| Column | Meaning |
|---|---|
| Ticker | Underlying stock |
| Horizon | Time window: "Next Fri", "Fri+2", "+5d", "+10d", "+15d", "+3wk" |
| Market View | "Grow Rapidly" (bullish) or "Fall Steeply" (bearish) |
| Call or Put | Which contract type to trade |
| Expiration | The specific expiry date selected |
| Strike Price | The strike of the recommended contract |
| Option Value | Current fair value estimate (mid-price = (bid+ask)/2) |
| Bid / Ask | Live bid and ask |
| Spread % | (Ask−Bid)/Mid × 100. Under 12% is tight, 12–18% is acceptable |
| Open Interest | Existing contracts — higher is more liquid |
| Option Volume | Today's trading volume on this contract |
| Contract Quality | 0–100 score combining liquidity, spread, and proximity to target strike |
| Strategy Score | Combined directional + volume signal strength |
| Confidence % | Model confidence (0–100%). Below 70% is weaker |
| % of Day Volume | This ticker's share of total scanned volume — size of the institutional footprint |
| Underlying 5D Move % | How much the stock actually moved over the last 5 days |

### Trade Rules Table

Below the main table is a **Trade Rules** section. For each idea it shows:

- **Entry Rule** — when to enter and what mid-price to target
- **Stop Rule** — stop loss level and % (1.5× ATR from entry)
- **Take Profit Rule** — two-stage exit: take 50% off at first target, trail rest to second
- **11 AM Check** — whether to stay in or exit at mid-morning based on momentum
- **Day Trader Plan** — plain-language summary of the full trade plan

### Best Times Table

Always visible at the bottom — shows the best windows to run the strategy scan and why.

### Market Regime Banner

The caption shows the current market regime (Bullish / Bearish / Neutral based on SPY's 3-month trend):
- **Bullish:** Put signals are suppressed unless their score is very high
- **Bearish:** Call signals are suppressed unless their score is very high
- **Neutral:** All signals treated equally

---

## Tab 5 — Options Chain Explorer

**What it does:** Shows the full options chain for a specific ticker and expiry. Also highlights unusual flow and shows the dealer gamma exposure (GEX) chart.

### How to Use

1. Select a **Ticker** from the dropdown
2. Select an **Expiration** date
3. Click **Run Options Chain Explorer**

### Unusual Options Flow

Contracts that had:
- Volume > 500 contracts, AND
- Volume/Open Interest ratio > 2 (i.e., today's trading is more than twice the existing positions)

These are potential institutional "smart money" prints.

### Dealer Gamma Exposure (GEX)

The bar chart shows GEX by strike price. GEX = Gamma × Open Interest × 100 × Strike.

- **Positive GEX bar** at a strike = dealers are long gamma there; they buy dips and sell rips → acts as a price magnet / stabilizer
- **Negative GEX bar** = dealers are short gamma; they amplify moves → price can accelerate through that level
- The **largest positive GEX strike** is often a key resistance/support level for the day

---

## Tab 6 — Price Forecast

**What it does:** Generates estimated 1-week and 2-week percentage moves for tickers with strong directional signals. Shows top 10 predicted gainers and top 10 predicted losers.

### How to Use

1. Click **Run Price Forecast**

### Column Meanings

| Column | Meaning |
|---|---|
| Ticker | Stock symbol |
| Last Price | Most recent close |
| 1W Direction | "Grow Rapidly" or "Fall Steeply" |
| Est. 1W Move % | Estimated percentage move over the next ~5 trading days |
| Est. 2W Move % | Estimated percentage move over the next ~10 trading days |
| Confidence % | Model confidence — how aligned the signals are |
| RSI (14) | Relative Strength Index (14-day). 30–70 is neutral zone |
| ADX (14) | Average Directional Index. Below 20 = weak trend, above 40 = strong trend |
| 5D Range % | Average intraday range — how volatile the stock has been |
| Vol Ratio | Volume vs 5-day average |

### Important Disclaimer

These are **directional watchlist estimates**, not price predictions. They are derived entirely from technical momentum (trend, RSI, MACD, ADX, MA alignment). Do not use them as standalone buy/sell signals. Always verify with your own analysis.

---

## Common Questions

**Why is the data showing zeros or "stale" values?**
You're likely running outside market hours or on a weekend. yfinance returns stale or empty data outside US market hours. Use the dashboard on trading days between 9:45 AM and 3:30 PM ET.

**Why does "No contracts passed the liquidity and spread filters" appear?**
See Tab 4 status notes above. Most commonly happens on weekends (volume = 0) or for thinly-traded tickers.

**What does "Confidence lower than 0.7 may not be recommended" mean?**
The strategy confidence score combines trend strength (ADX), volume activity, and RSI alignment. Scores below 70% mean the signals are weak or conflicting — treat those ideas with extra skepticism.

**Should I trade every idea from the Strategy tab?**
No. The tab is a structured filter, not a buy/sell signal generator. Cross-reference with your own view of market conditions, news, and risk tolerance. The trade rules give you a framework, not a guarantee.

**The Rapid Movers and Strategy tabs show different tickers. Why?**
Rapid Movers ranks all 70+ tickers by technical score. Strategy adds a second filter — it only shows tickers that also have enough options volume activity (from the Market Volume Leaders scan). A ticker can have strong price momentum but thin options liquidity, and will not appear in Strategy.
