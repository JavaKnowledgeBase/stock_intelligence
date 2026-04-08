# Short Squeeze Scanner — What Was Built & Where

## Summary
A new **Short Squeeze Scanner** tab was built for the Streamlit dashboard.
It was accidentally committed to the `stock_predictor` folder instead of
`stock_intelligence_main_worktree`. The files need to be ported here.

---

## Files to Add / Modify

### 1. NEW FILE → `short_squeeze.py`
Copy from: `c:\Users\rkafl\Documents\Projects\stock_predictor\short_squeeze.py`

**What it does:**
- Scans a list of tickers using free yfinance data
- Filters to large-cap only (market cap > $5B, price > $10)
- Fetches `shortPercentOfFloat`, `shortRatio` (days to cover) from yfinance `info`
- Computes from 3-month price history: RSI(14), 20-day MA position,
  3d/5d momentum, volume ratio (5-day vs 20-day avg)
- Scores each ticker 0–100:
  - Short float % → max 40 pts
  - Days to cover → max 20 pts
  - Price momentum + MA position → max 25 pts
  - Volume ratio → max 15 pts
- Assigns Tier 1 / Tier 2 / Tier 3 setup quality labels
- Generates a plain-English **buy timing recommendation** per ticker
- Returns a ranked `pd.DataFrame` via `scan_short_squeeze(tickers, min_short_float)`

---

### 2. MODIFY → `dashboard.py`

**Three changes needed:**

#### A. Add import at top (after `from r2_storage import ensure_assets_available`):
```python
from short_squeeze import scan_short_squeeze
```

#### B. Add two session state keys to the existing defaults block:
```python
("squeeze_df", None),
("squeeze_fetched_at", None),
```

#### C. Add `tab7` to the tab list:
Change:
```python
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([...])
```
To:
```python
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    ...,
    "Short Squeeze Scanner",
])
```

#### D. Add the full tab7 block at the bottom of `dashboard.py`
Copy the entire `with tab7:` block from:
`c:\Users\rkafl\Documents\Projects\stock_predictor\dashboard.py`
(starts at the `# SHORT SQUEEZE SCANNER` comment, end of file)

---

## Tab7 UI Features
- **Run button** → triggers `scan_short_squeeze(MARKET_SCAN_TICKERS)`
- **Min Short Float % slider** (5–25%, default 8%)
- **Max Results slider** (5–30, default 15)
- **Ranked metrics table** — ticker, short float %, days-to-cover, volume ratio,
  momentum, RSI, MA position, squeeze score, tier
- **Bar chart** — squeeze scores color-coded by tier
- **Expandable detail cards** per ticker:
  - Key metrics in columns
  - Bullet-point reasons why it's a candidate
  - Plain-English buy timing guidance (e.g. "WAIT — overbought" vs "BUY ZONE — breakout setup")
  - Earnings date shown as catalyst window when available

---

## Status
- [x] Copy `short_squeeze.py` into this folder
- [x] Apply 4 changes to `dashboard.py`
- [ ] Commit and push from `stock_intelligence_main_worktree`
