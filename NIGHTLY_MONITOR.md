# Nightly Monitor

Run the nightly health check with:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_nightly_monitor.ps1
```

Recommended scheduler time:
- 8:00 PM America/New_York

What it does:
- runs the live Price Forecast scan
- runs the live Strategy Ideas scan
- saves JSON and CSV reports under `reports/nightly/`
- recommends tweaks when coverage or confidence degrades
- emails a summary if SMTP env vars are configured

Environment variables:
- `MONITOR_EMAIL_TO` recipient email address
- `SMTP_HOST` SMTP server host
- `SMTP_PORT` SMTP server port
- `SMTP_USERNAME` SMTP login username
- `SMTP_PASSWORD` SMTP login password
- `SMTP_FROM` optional sender address override
- `SMTP_USE_TLS` optional, default `true`
- `SMTP_USE_SSL` optional, default `false`
- `MONITOR_BRANCH_NAME` optional branch label override in reports

Suggested Windows Task Scheduler action:
- Program/script: `powershell.exe`
- Add arguments: `-ExecutionPolicy Bypass -File "C:\Users\rkafl\Documents\Projects\stock_intelligence_main_worktree\run_nightly_monitor.ps1"`
- Start in: `C:\Users\rkafl\Documents\Projects\stock_intelligence_main_worktree`

Notes:
- confidence below 0.7 is treated as lower conviction in recommendations
- the monitor recommends changes; it does not auto-edit production logic
- if SMTP is not configured, the run still completes and writes local reports
