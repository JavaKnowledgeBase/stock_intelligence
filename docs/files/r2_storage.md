# File Reference: r2_storage.py

## Purpose

Manages synchronisation of trained models and feature data between Cloudflare R2 object storage and the local filesystem. Called on application startup to ensure a fresh deployment has all required assets without needing them bundled in the repository.

---

## Libraries Used

| Library | Why |
|---|---|
| `boto3` | AWS S3-compatible client — works with Cloudflare R2's S3 API |
| `os` | Read environment variables |
| `pathlib` | Check if local directories/files exist |
| `dotenv` | Load `.env` credentials before boto3 initialisation |

---

## Environment Variables Required

```
R2_BUCKET_NAME        = your-bucket-name
R2_ENDPOINT_URL       = https://<account_id>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID      = your-access-key
R2_SECRET_ACCESS_KEY  = your-secret-key
```

If any of these are absent, the function skips R2 and returns `"local_only"`.

---

## Main Function: `ensure_assets_available(force=False)` → dict

**Return values:**

| Status | Meaning |
|---|---|
| `"synced"` | Successfully downloaded assets from R2 |
| `"local_only"` | No R2 credentials — using whatever is already local |
| `"local_fallback"` | R2 failed but local assets exist — continuing |
| Raises `RuntimeError` | R2 failed AND local assets are missing |

**Logic:**
```
1. Load .env
2. Check R2 credentials — if missing, return "local_only"
3. Connect: boto3.client("s3", endpoint_url=R2_ENDPOINT_URL, ...)
4. List R2 objects in models/ and data/ prefixes
5. For each object:
     if not exists locally OR force=True:
         download to local path
6. Return "synced"

On exception:
  if local models/ and data/ both exist: return "local_fallback"
  else: raise RuntimeError("Assets unavailable from R2 and not found locally")
```

**`force=False`** — only downloads missing files. Set `force=True` to re-sync everything (e.g., after retraining models).

---

## R2 Directory Structure (mirrors local)

```
bucket/
├── models/
│   ├── AAPL.pkl
│   ├── SPY.pkl
│   └── ...
└── data/
    └── features/
        ├── AAPL.csv
        └── ...
```

---

## Uploading Assets to R2 (manual)

No auto-upload is implemented. After retraining, manually upload:

```python
import boto3, os
from dotenv import load_dotenv
load_dotenv()

s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("R2_ENDPOINT_URL"),
    aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
)

# Upload all model files
import glob
for path in glob.glob("models/*.pkl"):
    s3.upload_file(path, os.getenv("R2_BUCKET_NAME"), path)
    print(f"Uploaded {path}")
```

---

## Why Cloudflare R2 Instead of S3

R2 has no egress fees (free data transfer out), making it cost-effective for a dashboard that downloads the full model set on every cold start. The S3-compatible API means standard `boto3` works without modification — just point `endpoint_url` to the R2 endpoint.

---

## Called From

- `dashboard.py` — `ensure_assets_available()` called at the very top before Streamlit setup
- `api.py` — same call in the FastAPI startup event handler
