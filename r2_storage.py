import os
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv


load_dotenv()

_SYNCED = False
_BUCKET_ENV = "R2_BUCKET_NAME"
_ENDPOINT_ENV = "R2_ENDPOINT_URL"
_ACCESS_KEY_ENV = "R2_ACCESS_KEY_ID"
_SECRET_KEY_ENV = "R2_SECRET_ACCESS_KEY"
_PREFIXES = ("models", "data")


def _local_assets_present() -> bool:
    return all(Path(prefix).exists() for prefix in _PREFIXES)


def _missing_config() -> list[str]:
    required = [
        _BUCKET_ENV,
        _ENDPOINT_ENV,
        _ACCESS_KEY_ENV,
        _SECRET_KEY_ENV,
    ]
    return [name for name in required if not os.getenv(name)]


def r2_is_configured() -> bool:
    return not _missing_config()


def _build_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ[_ENDPOINT_ENV],
        aws_access_key_id=os.environ[_ACCESS_KEY_ENV],
        aws_secret_access_key=os.environ[_SECRET_KEY_ENV],
        region_name="auto",
    )


def ensure_assets_available(force: bool = False) -> dict:
    global _SYNCED

    if _SYNCED and not force:
        return {"status": "already_synced"}

    missing_config = _missing_config()
    if missing_config:
        if _local_assets_present():
            return {
                "status": "local_only",
                "detail": f"R2 not configured; using local assets. Missing: {', '.join(missing_config)}",
            }
        raise RuntimeError(
            "R2 credentials are not configured and local assets are missing. "
            f"Missing environment variables: {', '.join(missing_config)}"
        )

    bucket_name = os.environ[_BUCKET_ENV]
    client = _build_client()
    downloaded = 0

    try:
        paginator = client.get_paginator("list_objects_v2")
        for prefix in _PREFIXES:
            for page in paginator.paginate(Bucket=bucket_name, Prefix=f"{prefix}/"):
                for item in page.get("Contents", []):
                    key = item["Key"]
                    if key.endswith("/"):
                        continue

                    destination = Path(key)
                    destination.parent.mkdir(parents=True, exist_ok=True)

                    if destination.exists() and not force:
                        continue

                    client.download_file(bucket_name, key, str(destination))
                    downloaded += 1
    except (BotoCoreError, ClientError) as exc:
        if _local_assets_present():
            return {
                "status": "local_fallback",
                "detail": f"R2 sync failed; using local assets instead: {exc}",
            }
        raise RuntimeError(f"Failed to sync assets from R2: {exc}") from exc

    _SYNCED = True
    return {"status": "synced", "downloaded_files": downloaded}
