# Visual Architecture — Institutional Options Intelligence Platform

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                                   │
│                                                                         │
│   ┌──────────────────────┐          ┌──────────────────────────────┐   │
│   │   Streamlit Dashboard │          │   FastAPI REST API (api.py)  │   │
│   │   (dashboard.py)      │          │   GET /predictions           │   │
│   │   Port 8501           │          │   GET /predict/{ticker}      │   │
│   └──────────┬───────────┘          └───────────────┬──────────────┘   │
└──────────────┼──────────────────────────────────────┼───────────────────┘
               │                                      │
               ▼                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        BUSINESS LOGIC LAYER                              │
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────┐    │
│   │                     options_data.py                            │    │
│   │                                                               │    │
│   │  build_market_volume_table()   build_market_movers_table()   │    │
│   │  build_price_forecast_table()  build_strategy_table()         │    │
│   │  _pick_strategy_contract()     _build_strategy_recommendation()│   │
│   │  get_price_history()           get_options_chain()             │    │
│   └───────────────────────────────────────────────────────────────┘    │
│                                                                         │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐    │
│   │ build_features.py│   │  gex_engine.py  │   │ options_flow.py  │   │
│   │ 60+ technical    │   │ GEX by strike   │   │ unusual flow     │   │
│   │ indicators       │   │                 │   │ detection        │   │
│   └─────────────────┘   └─────────────────┘   └─────────────────┘    │
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────┐    │
│   │                     nightly_monitor.py                         │    │
│   │  Runs at 8 PM ET · builds forecast + strategy · emails report  │    │
│   └───────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                       │
│                                                                         │
│   ┌────────────────────────┐     ┌──────────────────────────────────┐  │
│   │  In-Memory Cache        │     │  Local Filesystem                │  │
│   │  options_data.py        │     │                                  │  │
│   │  TTL: 900 seconds       │     │  data/features/{ticker}.csv      │  │
│   │  Thread-safe Lock       │     │  data/options_market_snapshots/  │  │
│   │  Keys: (prefix,ticker,  │     │  models/{ticker}.pkl             │  │
│   │         period,interval)│     │  reports/nightly/*.json          │  │
│   └────────────────────────┘     └──────────────────────────────────┘  │
│                                                                         │
│   ┌────────────────────────┐     ┌──────────────────────────────────┐  │
│   │  Yahoo Finance (yfinance│     │  Cloudflare R2 Object Storage   │  │
│   │  Ticker.history()       │     │  r2_storage.py                   │  │
│   │  option_chain()         │     │  Syncs models/ and data/ dirs    │  │
│   │  options expiries       │     │  Boto3 S3-compatible client      │  │
│   └────────────────────────┘     └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Dashboard Tab Data Flow

```
User Clicks "Run Strategy"
         │
         ▼
build_strategy_table(MARKET_SCAN_TICKERS, top_n=10)
         │
         ├─── ThreadPoolExecutor ──► build_market_movers_table()
         │         │                      │
         │         │                      ├── _build_mover_row(ticker) × N tickers
         │         │                      │       │
         │         │                      │       ├── get_price_history() → yfinance
         │         │                      │       ├── build_features() → 60+ indicators
         │         │                      │       └── score 1W/1M upside/downside
         │         │                      │
         │         │                 sort by 1W score ↓
         │         │
         ├─── ThreadPoolExecutor ──► build_market_volume_table()
         │                               │
         │                               ├── get_full_market_options_snapshot(ticker) × N
         │                               │       └── get_options_chain() → yfinance
         │                               ├── sum call/put volume per ticker
         │                               └── save daily CSV snapshot
         │
         ├── merge movers + volume on ticker
         ├── compute strategy_confidence (ADX × vol_ratio × RSI_alignment)
         ├── filter confidence >= 0.60
         ├── detect market regime (SPY 3-month trend)
         ├── regime filter (suppress counter-trend signals)
         ├── expand top_N tickers × 2 horizons = candidate rows
         │
         └── ThreadPoolExecutor ──► _build_strategy_recommendation(row) × candidates
                   │
                   ├── _pick_strategy_contract()
                   │       ├── get_options_chain() → yfinance
                   │       ├── filter strike range ±15% (calls) / -15%+5% (puts)
                   │       ├── two-pass liquidity filter
                   │       └── score: 55% liquidity + 30% spread + 15% proximity
                   │
                   ├── ATR-based stop/target sizing
                   └── generate entry/stop/TP/midday/daily rules
```

---

## Strategy Scoring Pipeline

```
Raw Market Data (yfinance)
         │
         ▼
build_features()  ──────────────────────────────────────────────────────►
         │                                                               │
         │  Close Return     Volatility      Volume         Momentum     │
         │  ─────────────    ──────────      ──────         ────────     │
         │  close_ret_1d     volatility_5d   volume_        rsi_14       │
         │  close_ret_3d     volatility_10d  ratio_5/20     macd_hist    │
         │  close_ret_5d     atr_14          vol_direction  adx_14       │
         │                                   _ratio         ma_alignment │
         │                                                  obv_slope    │
         ▼
_build_mover_row_inner()
         │
         ├── trend_factor    = ret_5d × 0.65 + ret_3d × 0.35
         ├── swing_factor    = vol_5d × 0.6  + vol_10d × 0.4
         ├── volume_factor   = vol_ratio_5 × 0.6 + vol_ratio_20 × 0.4
         ├── extension_factor= dist_ma_20 × 0.6 + dist_ema_20 × 0.4
         │
         ├── RSI adj:    +2.0 if 50≤RSI≤70 (calls)  │  +2.0 if 30≤RSI≤50 (puts)
         ├── MACD adj:   +1.5 if hist > 0             │  −1.5
         ├── ADX adj:    −2.0 if ADX<20, +2.5 if >40
         ├── MA stack:   +2.0 if bullish stack        │  −2.0
         ├── OBV slope:  +0.75 if slope > 5           │  −0.75
         ├── 52W high:   +1.0 if within 3%            │  −0.5 if >20% away
         └── Rel vs SPY: +1.0 if outperforming >2%    │  −1.0
         │
         ▼
1W_upside  = trend×0.9 + swing×0.8 + max(vol-1,0)×6 + max(ext,0)×0.35 + adjustments
1W_downside= (−trend)×0.9 + swing×0.8 + max(vol-1,0)×6 + max(−ext,0)×0.35 + adjustments
         │
         ▼
direction  = "Grow Rapidly" if upside ≥ downside, else "Fall Steeply"
score      = max(upside, downside)
```

---

## Contract Selection Flow

```
_pick_strategy_contract(ticker, contract_type, close_price, hv_30d, expiry)
         │
         ▼
get_options_chain(ticker, expiry)  ── yfinance tk.option_chain()
         │
         ▼
Filter by contract_type (call/put)
         │
         ▼
Coerce numeric: strike, last_price, bid, ask, volume, open_interest
Compute: mid_price = (bid+ask)/2
         spread_pct = (ask−bid)/mid × 100
         │
         ▼
base_filter: ask>0, bid≥0, mid>0
         │
         ▼
Strike Range Filter  ◄─────────── PREVENTS deep ITM accumulation issue
  calls:  95%–115% of close_price
  puts:   85%–105% of close_price
         │
         ▼
Two-Pass Liquidity Filter
  TIGHT:  volume≥100, OI≥100, spread≤12%
  LOOSE:  volume≥50,  OI≥100, spread≤18%
  (use tight if available, else loose)
         │
         ▼
Scoring:
  target_strike = close × 1.02 (call) or close × 0.98 (put)
  liquidity_score = volume + OI × 0.5
  distance_pct    = |strike − target| / close × 100

  quality_score = (liq/max_liq × 55)
                + ((18−spread%).clip(0)/18 × 30)
                + ((10−distance%).clip(0)/10 × 15)
         │
         ▼
Select highest quality_score contract
Return: strike, option_value(mid_price), bid, ask, spread_pct, OI, volume,
        quality_score, iv_hv_ratio
```

---

## Caching Architecture

```
Request: get_price_history("AAPL", "1y", "1d")
         │
         ▼
_get_cached(("history","AAPL","1y","1d"))
         │
         ├── HIT  ── return df.copy()
         │
         └── MISS
              │
              ▼
         yfinance.Ticker("AAPL").history(period="1y", interval="1d")
              │
              ▼
         _set_cached(("history","AAPL","1y","1d"), df.copy())
              │
              ▼
         return df (original)

Cache invalidation: TTL = 900 seconds (15 minutes)
Thread safety: threading.Lock() wraps all reads and writes
Cache keys by type:
  ("history",  ticker, period, interval)
  ("chain",    ticker, expiry)
  ("expirations", ticker)
  ("ticker",   ticker)   ← yfinance.Ticker object
```

---

## Persistent Storage Map

```
stock_intelligence_main_worktree/
│
├── data/
│   ├── features/
│   │   ├── AAPL.csv        ← 60+ technical features, one row per trading day
│   │   ├── SPY.csv
│   │   └── ... (25+ tickers)
│   │
│   └── options_market_snapshots/
│       ├── 2026-03-24.csv  ← daily call/put volume per ticker
│       ├── 2026-03-25.csv
│       └── ... (rolling 7-day window used for trend)
│
├── models/
│   ├── AAPL.pkl            ← joblib: sklearn model + metadata dict
│   ├── SPY.pkl
│   └── ... (25+ tickers)
│
└── reports/
    └── nightly/
        ├── 20260331_200000.json   ← full summary JSON
        ├── 20260331_200000_forecast.csv
        └── 20260331_200000_strategy.csv
```

---

## Threading Model

```
Dashboard "Run Strategy" button pressed
         │
         ▼
Streamlit main thread
         │
         ├── ThreadPoolExecutor(max_workers=8)
         │     ├── Thread 1: _build_mover_row("AAPL")
         │     │               └── get_price_history() ← _CACHE_LOCK protects
         │     ├── Thread 2: _build_mover_row("MSFT")
         │     ├── Thread 3: _build_mover_row("SPY")
         │     └── ... (up to 8 concurrent)
         │
         └── ThreadPoolExecutor(max_workers=8)
               ├── Thread 1: _build_strategy_recommendation(AAPL_NextFri_row)
               ├── Thread 2: _build_strategy_recommendation(AAPL_FriPlus2_row)
               └── ... (candidates × 2 horizons each)

Each thread uses: Ticker.history() — per-ticker session, thread-safe
                  _CACHE_LOCK — protects shared in-memory dict
```

---

## Module Dependency Graph

```
dashboard.py
    │
    ├── options_data.py ──────────────────────────────────────────────────┐
    │       ├── build_features.py                                         │
    │       ├── config.py                                                 │
    │       └── yfinance                                                  │
    │                                                                     │
    ├── gex_engine.py                                                     │
    ├── options_flow.py                                                   │
    ├── r2_storage.py ──────────────────────────── boto3 (R2/S3)         │
    └── config.py                                                         │
                                                                          │
api.py                                                                    │
    ├── options_data.py (same) ────────────────────────────────────────── │
    ├── r2_storage.py                                                      │
    ├── train_models.py ── scikit-learn (RF, ET, GBM)                    │
    └── config.py                                                         │
                                                                          │
nightly_monitor.py                                                        │
    ├── options_data.py (same) ────────────────────────────────────────── ┘
    └── config.py
```

---

## Market Regime Filter

```
_get_market_regime()
         │
         ├── get SPY 3-month daily history
         ├── compute ret_5d  = (close[-1] − close[-6]) / close[-6] × 100
         ├── compute ma_20   = mean(close[-20:])
         ├── above_ma20      = close[-1] > ma_20
         │
         ├── ret_5d > 1.0  AND above_ma20  → "bullish"
         ├── ret_5d < −1.0 AND below_ma20  → "bearish"
         └── else                           → "neutral"
         │
         ▼
In build_strategy_table():
  if bullish or bearish:
      score_threshold = strategy_score.quantile(0.60)
      aligned_side = "call" if bullish else "put"
      keep if: contract_type == aligned_side
            OR strategy_score >= score_threshold
```
