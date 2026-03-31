# File Reference: nightly_monitor.py

## Purpose

An automated health-check and email reporting system that runs nightly after market close. It re-runs the full forecast and strategy pipeline, compares results to the previous run, generates human-readable recommendations, writes JSON/CSV reports to disk, and optionally sends an email summary via SMTP.

Designed to be triggered by cron (Linux) or Windows Task Scheduler at 8:00 PM ET on trading days.

---

## Libraries Used

| Library | Why |
|---|---|
| `options_data` | `build_price_forecast_table()`, `build_strategy_table()` |
| `config` | `MARKET_SCAN_TICKERS` |
| `os` | Read environment variables, path operations |
| `json` | Write JSON report files |
| `smtplib` | Send email via SMTP |
| `email.mime` | Construct multipart email with HTML body |
| `pathlib` | Report directory management |
| `datetime` | Timestamp generation |
| `subprocess` | Get git commit hash for build label |
| `dotenv` | Load `.env` credentials |
| `pandas` | DataFrame operations for CSV output |

---

## Configuration Constants

```python
MIN_FORECAST_ROWS    = 5    # Minimum acceptable gainers+losers rows
MIN_STRATEGY_ROWS    = 5    # Minimum acceptable strategy rows
RECOMMENDED_CONFIDENCE = 70  # % threshold for median confidence
REPORT_DIR = Path("reports/nightly")
```

---

## Main Function: `run_nightly_monitor()`

**Entry point** — called when script runs directly (`if __name__ == "__main__"`).

**Steps:**
1. Call `build_price_forecast_table(MARKET_SCAN_TICKERS, top_n=10)` → gainers, losers
2. Call `build_strategy_table(MARKET_SCAN_TICKERS, top_n=10)` → strategy_df, diagnostics
3. Read previous run from `reports/nightly/` for comparison
4. Call `build_summary()` → structured dict
5. Call `_write_report_files(summary, gainers, losers, strategy_df)` → writes JSON + 3 CSVs
6. Call `_send_email(summary)` → sends HTML email if SMTP is configured

---

## `_confidence_stats(df, col)` → dict

Computes descriptive statistics for a confidence column:
- count, mean, median, min, max

Returns a dict. Used for both forecast_confidence and strategy_confidence columns.

---

## `_build_recommendations(summary)` → list[str]

Generates plain-English action items based on metric checks:

| Check | Message Generated |
|---|---|
| forecast_rows < MIN_FORECAST_ROWS | "Forecast rows low (N) — consider expanding MARKET_SCAN_TICKERS or check yfinance availability" |
| strategy_rows < MIN_STRATEGY_ROWS | "Strategy rows low (N) — check liquidity filters and data availability" |
| strategy median confidence < RECOMMENDED_CONFIDENCE | "Median strategy confidence is X% — signals may be weak" |
| forecast median confidence < RECOMMENDED_CONFIDENCE | "Median forecast confidence is X% — signals may be weak" |
| coverage dropped > 30% vs previous | "Coverage dropped from N to M rows — possible data issue or market condition change" |
| diagnostics.status in {"data_unavailable","no_contracts"} | "Strategy scan status: {status} — {message}" |

---

## `build_summary(gainers, losers, strategy_df, diagnostics, prev_summary)` → dict

Assembles the full JSON summary:

```json
{
  "generated_at": "2026-03-31T20:00:00",
  "build": {"branch": "main", "commit": "f1d65c2"},
  "forecast": {
    "gainers_count": 10,
    "losers_count": 8,
    "total_rows": 18,
    "confidence_stats": {"count":18,"mean":78.2,"median":79.5,"min":65.0,"max":92.1}
  },
  "strategy": {
    "rows": 10,
    "status": "ok",
    "regime": "bullish",
    "confidence_stats": {"count":10,"mean":81.3,"median":83.0,"min":70.1,"max":95.2},
    "diagnostics": { ... full diagnostics dict ... }
  },
  "recommendations": [
    "Signals look healthy — no action needed"
  ],
  "previous_run": {
    "generated_at": "2026-03-30T20:00:00",
    "forecast_total_rows": 16,
    "strategy_rows": 9
  }
}
```

---

## `_write_report_files(summary, gainers, losers, strategy_df)` → Path

Creates `reports/nightly/` if it doesn't exist.

Writes three files with timestamp prefix `YYYYMMDD_HHMMSS`:

1. `{timestamp}.json` — full summary dict
2. `{timestamp}_forecast.csv` — gainers + losers DataFrames concatenated with a `direction` column added
3. `{timestamp}_strategy.csv` — strategy_df if not empty

Returns the JSON file path.

---

## `_send_email(summary)` → bool

Loads SMTP credentials from environment:
- `SMTP_HOST` (default: smtp.gmail.com)
- `SMTP_PORT` (default: 587)
- `SMTP_USER`
- `SMTP_PASSWORD`
- `MONITOR_EMAIL_RECIPIENT`

Returns False (silently) if any credential is missing — email is optional.

**Email structure:** HTML multipart message.

**Subject:** `[Options Monitor] {date} — {status} | Forecast: {N} rows | Strategy: {M} rows`

**Body sections:**
- Build info (branch, commit)
- Forecast summary table (gainers count, losers count, median confidence)
- Strategy summary (rows, status, regime, median confidence)
- Recommendations (bulleted list)
- Timestamp

Uses STARTTLS (port 587). For Gmail: requires an App Password (not account password) with 2FA enabled.

---

## Scheduling

### Windows Task Scheduler

1. Open Task Scheduler → Create Basic Task
2. Trigger: Daily, 8:00 PM
3. Action: Start Program → `powershell.exe`
4. Arguments: `-File "C:\path\to\run_nightly_monitor.ps1"`

### Linux Cron

```cron
0 20 * * 1-5 cd /app && python nightly_monitor.py >> logs/nightly.log 2>&1
```

(Monday–Friday at 8 PM)

---

## Report Retention

Reports accumulate in `reports/nightly/`. There is no automatic cleanup. If disk space is a concern, add a periodic cleanup task to remove files older than 30 days:

```python
import glob, os
for f in glob.glob("reports/nightly/*.json"):
    if os.path.getmtime(f) < time.time() - 30*86400:
        os.remove(f)
```

---

## Previous Run Comparison

The monitor reads the most recent existing JSON report from `reports/nightly/` (by filename sort, which is chronological since filenames start with YYYYMMDD) and extracts `forecast_total_rows` and `strategy_rows` for comparison. A >30% drop triggers a recommendation. This catches silent data degradation that wouldn't be obvious from a single run's output.
