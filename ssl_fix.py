"""
ssl_fix.py — Must be imported first in dashboard.py and any other entry point.

Sets CURL_CA_BUNDLE, REQUESTS_CA_BUNDLE, and SSL_CERT_FILE to a combined
PEM bundle (certifi + Windows corporate CA certs) so that:
  - yfinance (curl_cffi)        ← reads CURL_CA_BUNDLE
  - requests / urllib3           ← reads REQUESTS_CA_BUNDLE
  - Anthropic / httpx            ← reads SSL_CERT_FILE
  - Python ssl module            ← truststore handles this

If the bundle is missing or stale (>30 days), it is rebuilt automatically.
"""
from __future__ import annotations

import base64
import os
import ssl
import time
from pathlib import Path

import certifi

_BUNDLE = Path(__file__).parent / "data" / "ca_bundle.pem"
_MAX_AGE = 30 * 24 * 3600  # rebuild after 30 days


def _needs_rebuild() -> bool:
    if not _BUNDLE.exists():
        return True
    return time.time() - _BUNDLE.stat().st_mtime > _MAX_AGE


def build_bundle() -> None:
    _BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    with open(certifi.where(), "r") as f:
        base = f.read()
    extra: list[str] = []
    for store in ("CA", "ROOT", "MY", "AuthRoot"):
        try:
            for cert_data, enc, _trust in ssl.enum_certificates(store):
                if enc == "x509_asn":
                    pem = "-----BEGIN CERTIFICATE-----\n"
                    pem += base64.encodebytes(cert_data).decode()
                    pem += "-----END CERTIFICATE-----\n"
                    extra.append(pem)
        except Exception:
            pass
    with open(_BUNDLE, "w") as f:
        f.write(base + "\n" + "\n".join(extra))


def apply() -> None:
    if _needs_rebuild():
        build_bundle()

    bundle = str(_BUNDLE)
    os.environ.setdefault("CURL_CA_BUNDLE", bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
    os.environ.setdefault("SSL_CERT_FILE", bundle)

    # Also inject into Python's ssl module for httpx / anthropic
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:
        pass


apply()
