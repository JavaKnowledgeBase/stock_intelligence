# Deployment Guide — 200 Concurrent Users

## Quick Start (single machine, local testing)

```powershell
# 1. Activate the venv
.\.venv\Scripts\Activate.ps1

# 2. Start the background worker (keep this running)
Start-Process python -ArgumentList "background_worker.py" -NoNewWindow

# 3. Start the Streamlit app
streamlit run dashboard.py
```

---

## Production Setup (200 concurrent users)

### Why the shared cache matters
yfinance calls are the bottleneck — each market scan touches 100 tickers × 3 API calls each.
Without caching, 200 users clicking "Run" simultaneously = 60,000 API calls → rate limits + timeouts.

With `cache_manager.py` + `background_worker.py`:
- The worker runs every 15 min and writes results to `data/shared_cache/`
- All 200 users read from that one file instantly (< 1ms)
- Users still see "cached · 8m ago" timestamps so they know the data age

### Option A — Single VPS (cheapest, handles ~200 users)

**Requirements:** 4 vCPU / 8 GB RAM VPS (e.g. Hetzner CX31 ~$12/mo, DigitalOcean 4GB Droplet ~$24/mo)

```bash
# Install nginx
sudo apt install nginx -y

# Run 4 Streamlit workers on ports 8501-8504
for PORT in 8501 8502 8503 8504; do
  nohup .venv/bin/streamlit run dashboard.py \
    --server.port $PORT \
    --server.headless true &
done

# Run background worker
nohup python background_worker.py &
```

**nginx config** (`/etc/nginx/sites-available/stockapp`):
```nginx
upstream streamlit_backend {
    least_conn;
    server 127.0.0.1:8501;
    server 127.0.0.1:8502;
    server 127.0.0.1:8503;
    server 127.0.0.1:8504;
}

server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://streamlit_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/stockapp /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### Option B — Streamlit Community Cloud (free tier, up to ~50 users)

1. Push this repo to GitHub (make sure `.env` is in `.gitignore`)
2. Go to share.streamlit.io → New app → select `dashboard.py`
3. Add secrets in the Streamlit Cloud dashboard (R2 keys, API keys)

Limitation: Community Cloud uses a single worker; the shared cache still helps but
you won't get true 200-user horizontal scaling without nginx+multi-worker.

### Option C — Docker (recommended for paid hosting)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "dashboard.py"]
```

```bash
# Build and run 4 replicas behind a Docker load balancer
docker build -t stockapp .
docker run -d -p 8501:8501 --name app1 stockapp
docker run -d -p 8502:8501 --name app2 stockapp
docker run -d -p 8503:8501 --name app3 stockapp
docker run -d -p 8504:8501 --name app4 stockapp
# Point nginx upstream to 8501-8504
```

---

## Environment Variables

Create a `.env` file (never commit this):

```env
# Cloudflare R2 (for syncing models + data from cloud)
R2_BUCKET_NAME=your-bucket-name
R2_ENDPOINT_URL=https://xxxx.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your-key-id
R2_SECRET_ACCESS_KEY=your-secret-key

# Polygon.io (for Ticker Analysis tab real-time data)
POLYGON_API_KEY=your-polygon-key

# Anthropic Claude API (optional — for AI trade thesis)
ANTHROPIC_API_KEY=your-anthropic-key

# Nightly monitor email alerts
MONITOR_EMAIL_TO=you@example.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=you@gmail.com
SMTP_PASSWORD=your-app-password
```

---

## Background Worker Schedule (Windows Task Scheduler)

To run the background worker every 15 minutes during market hours:

1. Open Task Scheduler → Create Basic Task
2. Name: `StockApp Background Worker`
3. Trigger: Daily, repeat every 15 minutes from 9:00 AM to 5:00 PM
4. Action: Start a program
   - Program: `C:\Users\DueDiligence\Desktop\stock streamlit\.venv\Scripts\python.exe`
   - Arguments: `background_worker.py`
   - Start in: `C:\Users\DueDiligence\Desktop\stock streamlit`

---

## Monetisation Checklist

- [ ] Deploy to a VPS with a domain + SSL (Let's Encrypt)
- [ ] Add Stripe payment link (buy.stripe.com → password-protected page)
- [ ] Gate access with `streamlit-authenticator` (pip package)
- [ ] Price: $29/mo basic, $59/mo pro (with email alerts)
- [ ] Target: 70 basic subscribers = $2,030/mo
- [ ] Promote on: r/stocks, r/options, r/wallstreetbets, Twitter/X fintwit, StockTwits

```
70 users × $29/mo = $2,030/mo  ← minimum target
25 users × $59/mo = $1,475/mo  ← alternative with pro tier only
```
