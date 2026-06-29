"""
Macro Dashboard — key economic indicators from primary government sources.

Sources (all free, no API key):
  FRED CSV endpoint   — fred.stlouisfed.org (Federal Reserve St. Louis)
  Treasury.gov XML    — home.treasury.gov  (daily yield curve)
  BLS JSON API        — api.bls.gov        (CPI, jobs — no key for public series)
"""

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import requests

_TIMEOUT = 15
_HEADERS = {"User-Agent": "MarketIntelligencePlatform admin@marketintelligence.app"}

# FRED public CSV endpoint — no API key required
_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

FRED_SERIES = {
    "fed_funds": {
        "id": "DFF",
        "label": "Fed Funds Rate",
        "unit": "%",
        "description": "Effective Federal Funds Rate (daily)",
    },
    "cpi": {
        "id": "CPIAUCSL",
        "label": "CPI (All Urban)",
        "unit": "Index",
        "description": "Consumer Price Index, Seasonally Adjusted (monthly)",
    },
    "cpi_yoy": {
        "id": "CPIAUCSL",
        "label": "CPI YoY %",
        "unit": "%",
        "description": "CPI Year-over-Year % change (computed)",
        "transform": "pct_change_12",
    },
    "unemployment": {
        "id": "UNRATE",
        "label": "Unemployment Rate",
        "unit": "%",
        "description": "Civilian Unemployment Rate (monthly)",
    },
    "yield_spread": {
        "id": "T10Y2Y",
        "label": "10Y-2Y Spread",
        "unit": "%",
        "description": "10-Year minus 2-Year Treasury spread (inverted = recession signal)",
    },
    "vix": {
        "id": "VIXCLS",
        "label": "VIX",
        "unit": "Points",
        "description": "CBOE Volatility Index (daily)",
    },
    "breakeven_inflation": {
        "id": "T10YIE",
        "label": "10Y Breakeven Inflation",
        "unit": "%",
        "description": "Market-implied 10-year inflation expectation",
    },
    "m2_money": {
        "id": "M2SL",
        "label": "M2 Money Supply",
        "unit": "$B",
        "description": "M2 money stock in billions (monthly)",
    },
}


def _fetch_fred_series(series_id: str, years_back: int = 3) -> pd.DataFrame:
    """Fetch a FRED series as a DataFrame with DATE and VALUE columns."""
    start = (datetime.now() - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")
    url = f"{_FRED_CSV}?id={series_id}&vintage_date=&cosd={start}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "value"])


def get_macro_indicators(years_back: int = 3) -> dict[str, pd.DataFrame]:
    """
    Fetch all key macro series.
    Returns dict: series_key → DataFrame(date, value)
    """
    results = {}
    for key, meta in FRED_SERIES.items():
        if meta.get("transform") == "pct_change_12":
            # Compute YoY % change from level data
            df = _fetch_fred_series(meta["id"], years_back + 1)
            if not df.empty:
                df["value"] = df["value"].pct_change(12) * 100
                df = df.dropna().tail(365 * years_back).reset_index(drop=True)
        else:
            df = _fetch_fred_series(meta["id"], years_back)
        results[key] = df

    return results


def get_latest_values(indicators: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """
    Extract the most recent value for each indicator.
    Returns dict: series_key → {value, date, label, unit, description}
    """
    latest = {}
    for key, df in indicators.items():
        meta = FRED_SERIES[key]
        if df.empty:
            latest[key] = {**meta, "value": None, "date": None}
        else:
            row = df.iloc[-1]
            latest[key] = {
                **meta,
                "value": round(float(row["value"]), 2),
                "date": row["date"].strftime("%Y-%m-%d"),
            }
    return latest


# ── Treasury Yield Curve ──────────────────────────────────────────────────────

_TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value={yyyymm}"
)

_YIELD_FIELDS = [
    ("BC_1MONTH", "1M"),
    ("BC_3MONTH", "3M"),
    ("BC_6MONTH", "6M"),
    ("BC_1YEAR", "1Y"),
    ("BC_2YEAR", "2Y"),
    ("BC_3YEAR", "3Y"),
    ("BC_5YEAR", "5Y"),
    ("BC_7YEAR", "7Y"),
    ("BC_10YEAR", "10Y"),
    ("BC_20YEAR", "20Y"),
    ("BC_30YEAR", "30Y"),
]


def _fetch_yield_curve_month(yyyymm: str) -> pd.DataFrame:
    """Fetch daily yield curve data for a given month (e.g. '202506')."""
    url = _TREASURY_URL.format(yyyymm=yyyymm)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "d": "http://schemas.microsoft.com/ado/2007/08/dataservices",
            "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
        }
        rows = []
        for entry in root.findall("atom:entry", ns):
            props = entry.find(".//m:properties", ns)
            if props is None:
                continue
            date_el = props.find("d:NEW_DATE", ns)
            if date_el is None or not date_el.text:
                continue
            row = {"date": pd.to_datetime(date_el.text[:10], errors="coerce")}
            for field, label in _YIELD_FIELDS:
                el = props.find(f"d:{field}", ns)
                row[label] = float(el.text) if el is not None and el.text else None
            rows.append(row)
        return pd.DataFrame(rows).dropna(subset=["date"])
    except Exception:
        return pd.DataFrame()


def get_yield_curve(months_back: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch yield curve data for the past N months.

    Returns
    -------
    (history_df, latest_curve_df)
    history_df : all daily rows — useful for time-series of individual tenors
    latest_curve_df : single row representing most recent yield curve
    """
    frames = []
    now = datetime.now()
    for i in range(months_back + 1):
        dt = now - timedelta(days=30 * i)
        yyyymm = dt.strftime("%Y%m")
        df = _fetch_yield_curve_month(yyyymm)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    history = pd.concat(frames, ignore_index=True)
    history = history.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    # Most recent complete row
    latest_row = history.dropna(how="any").iloc[-1:] if not history.empty else pd.DataFrame()
    if not latest_row.empty:
        tenors = [label for _, label in _YIELD_FIELDS]
        curve = pd.DataFrame({
            "Tenor": tenors,
            "Yield %": [latest_row[t].values[0] for t in tenors],
        })
    else:
        curve = pd.DataFrame()

    return history, curve


# ── Market Regime Indicator ───────────────────────────────────────────────────

def classify_market_regime(latest: dict[str, dict]) -> dict:
    """
    Simple rule-based market regime from macro signals.
    Returns {regime, color, signals}
    """
    signals = []
    bullish = 0
    bearish = 0

    fed = (latest.get("fed_funds") or {}).get("value")
    cpi_yoy = (latest.get("cpi_yoy") or {}).get("value")
    unemployment = (latest.get("unemployment") or {}).get("value")
    spread = (latest.get("yield_spread") or {}).get("value")
    vix = (latest.get("vix") or {}).get("value")

    if fed is not None:
        if fed < 3.0:
            signals.append("✅ Low rates — accommodative Fed")
            bullish += 1
        elif fed > 5.0:
            signals.append("⚠️ High rates — restrictive Fed")
            bearish += 1
        else:
            signals.append("➡️ Neutral rates")

    if cpi_yoy is not None:
        if cpi_yoy < 2.5:
            signals.append("✅ Inflation near target")
            bullish += 1
        elif cpi_yoy > 4.0:
            signals.append("⚠️ Elevated inflation")
            bearish += 1
        else:
            signals.append("➡️ Inflation moderating")

    if unemployment is not None:
        if unemployment < 4.5:
            signals.append("✅ Tight labour market")
            bullish += 1
        elif unemployment > 6.0:
            signals.append("⚠️ Rising unemployment")
            bearish += 1
        else:
            signals.append("➡️ Labour market softening")

    if spread is not None:
        if spread < 0:
            signals.append(f"⚠️ Yield curve inverted ({spread:+.2f}%) — recession signal")
            bearish += 2
        elif spread > 0.5:
            signals.append(f"✅ Yield curve normal ({spread:+.2f}%)")
            bullish += 1
        else:
            signals.append(f"➡️ Yield curve flat ({spread:+.2f}%)")

    if vix is not None:
        if vix < 18:
            signals.append(f"✅ Low volatility (VIX {vix:.1f}) — complacency or calm")
            bullish += 1
        elif vix > 30:
            signals.append(f"⚠️ Elevated fear (VIX {vix:.1f}) — potential opportunity")
            bearish += 1
        else:
            signals.append(f"➡️ Moderate volatility (VIX {vix:.1f})")

    if bullish > bearish + 1:
        regime, color = "🟢 Risk-On", "green"
    elif bearish > bullish + 1:
        regime, color = "🔴 Risk-Off", "red"
    else:
        regime, color = "🟡 Mixed / Transitioning", "orange"

    return {"regime": regime, "color": color, "signals": signals}
