"""
Hedge Fund 13F Tracker — quarterly institutional holdings from SEC EDGAR.

Primary source: https://data.sec.gov  (official SEC, free, no API key)
SEC rate limit: 10 requests/second — we stay well under it.

13F-HR filings are filed within 45 days of quarter-end by any institution
managing >$100M in equities. Holdings represent the snapshot at quarter-end.
"""

import re
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

_HEADERS = {"User-Agent": "MarketIntelligencePlatform admin@marketintelligence.app"}
_SEC_BASE = "https://data.sec.gov"
_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
_SLEEP = 0.12  # ~8 req/sec, safely under SEC 10 req/sec limit

# Known hedge funds and institutions with verified SEC CIK numbers
KNOWN_FUNDS: dict[str, str] = {
    "Berkshire Hathaway (Buffett)": "1067983",
    "Bridgewater Associates": "1350694",
    "Renaissance Technologies": "1037038",
    "Pershing Square (Ackman)": "1336528",
    "Tiger Global Management": "1167483",
    "Third Point (Loeb)": "1040273",
    "Viking Global Investors": "1103804",
    "Coatue Management": "1336771",
    "D1 Capital Partners": "1726445",
    "Greenlight Capital (Einhorn)": "1079114",
    "Duquesne Family Office (Druckenmiller)": "1446194",
    "Appaloosa Management (Tepper)": "813672",
    "Elliott Management (Singer)": "1048268",
    "Baupost Group (Klarman)": "893482",
    "Lone Pine Capital": "1014932",
    "Two Sigma Investments": "1446933",
    "Citadel Advisors": "1423689",
    "Point72 Asset Management (Cohen)": "1603466",
    "Soros Fund Management": "1029160",
    "Icahn Associates": "921738",
}


def _get(url: str, timeout: int = 15) -> requests.Response | None:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        time.sleep(_SLEEP)
        return resp
    except Exception:
        return None


def _get_latest_13f(cik: str) -> tuple[str, str] | None:
    """
    Returns (accession_number, filing_date) for the most recent 13F-HR,
    or None if the fund has no 13F on record.
    """
    cik_padded = cik.zfill(10)
    resp = _get(f"{_SEC_BASE}/submissions/CIK{cik_padded}.json")
    if not resp:
        return None

    recent = resp.json().get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])

    for form, acc, date in zip(forms, accessions, dates):
        if form == "13F-HR":
            return acc, date

    # Sometimes older filings are in a paginated "files" list
    return None


def _find_infotable_url(cik: str, accession: str) -> str | None:
    """Find the InfoTable XML URL inside a 13F-HR filing."""
    cik_clean = cik.lstrip("0") or "0"
    acc_nodash = accession.replace("-", "")
    index_url = f"{_ARCHIVES}/{cik_clean}/{acc_nodash}/{accession}-index.json"

    resp = _get(index_url)
    if not resp:
        return None

    try:
        items = resp.json().get("directory", {}).get("item", [])
    except Exception:
        return None

    for item in items:
        name = item.get("name", "").lower()
        if "infotable" in name and name.endswith(".xml"):
            return f"{_ARCHIVES}/{cik_clean}/{acc_nodash}/{item['name']}"

    # Fallback: any secondary .xml that isn't the primary submission doc
    for item in items:
        name = item.get("name", "")
        if name.lower().endswith(".xml") and "primary_doc" not in name.lower():
            return f"{_ARCHIVES}/{cik_clean}/{acc_nodash}/{name}"

    return None


def _parse_infotable(xml_text: str) -> list[dict]:
    """Parse 13F InfoTable XML into holding dicts."""
    # Strip namespaces to simplify ElementTree queries
    xml_clean = re.sub(r'\sxmlns(?::\w+)?="[^"]*"', "", xml_text)
    xml_clean = re.sub(r"<(\w+:)", "<", xml_clean)
    xml_clean = re.sub(r"</(\w+:)", "</", xml_clean)

    try:
        root = ET.fromstring(xml_clean)
    except ET.ParseError:
        return []

    rows = []
    for entry in root.findall(".//infoTable"):
        name = (entry.findtext("nameOfIssuer") or "").strip()
        cusip = (entry.findtext("cusip") or "").strip()

        value_raw = (entry.findtext("value") or "0").replace(",", "")
        try:
            value_usd = int(value_raw) * 1_000
        except ValueError:
            value_usd = 0

        shares_elem = entry.find(".//sshPrnamt")
        try:
            shares = int((shares_elem.text or "0").replace(",", "")) if shares_elem is not None else 0
        except ValueError:
            shares = 0

        prnamt_type = (entry.findtext(".//sshPrnamtType") or "SH").strip()

        if name and value_usd > 0:
            rows.append({
                "company": name,
                "cusip": cusip,
                "shares": shares,
                "unit": prnamt_type,
                "value_usd": value_usd,
            })

    return rows


def get_fund_holdings(fund_name: str) -> tuple[pd.DataFrame, str, str]:
    """
    Fetch the latest 13F-HR holdings for a named fund.

    Returns
    -------
    (holdings_df, filing_date, fund_name)
    holdings_df columns: company, cusip, shares, unit, value_usd, pct_portfolio
    """
    cik = KNOWN_FUNDS.get(fund_name)
    if not cik:
        return pd.DataFrame(), "", fund_name

    result = _get_latest_13f(cik)
    if not result:
        return pd.DataFrame(), "", fund_name

    accession, filing_date = result

    info_url = _find_infotable_url(cik, accession)
    if not info_url:
        return pd.DataFrame(), filing_date, fund_name

    resp = _get(info_url)
    if not resp:
        return pd.DataFrame(), filing_date, fund_name

    rows = _parse_infotable(resp.text)
    if not rows:
        return pd.DataFrame(), filing_date, fund_name

    df = pd.DataFrame(rows)
    total = df["value_usd"].sum()
    df["pct_portfolio"] = (df["value_usd"] / total * 100).round(2) if total > 0 else 0.0
    df = df.sort_values("value_usd", ascending=False).reset_index(drop=True)

    return df, filing_date, fund_name


def get_multi_fund_top_holdings(
    fund_names: list[str],
    top_n_per_fund: int = 15,
) -> pd.DataFrame:
    """
    Fetch top holdings across multiple funds and combine into one DataFrame.
    Useful for finding convergence trades (stocks multiple funds own).
    """
    all_rows = []
    for fund in fund_names:
        df, filing_date, _ = get_fund_holdings(fund)
        if df.empty:
            continue
        top = df.head(top_n_per_fund).copy()
        top["fund"] = fund
        top["filing_date"] = filing_date
        all_rows.append(top)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    return combined


def get_conviction_plays(combined_df: pd.DataFrame, min_funds: int = 2) -> pd.DataFrame:
    """
    Find stocks held by multiple funds simultaneously — high-conviction ideas.
    """
    if combined_df.empty or "company" not in combined_df.columns:
        return pd.DataFrame()

    agg = (
        combined_df.groupby("company")
        .agg(
            funds_holding=("fund", "nunique"),
            fund_names=("fund", lambda x: ", ".join(sorted(x.unique()))),
            total_value=("value_usd", "sum"),
            avg_pct=("pct_portfolio", "mean"),
            cusip=("cusip", "first"),
        )
        .reset_index()
    )
    return (
        agg[agg["funds_holding"] >= min_funds]
        .sort_values("total_value", ascending=False)
        .reset_index(drop=True)
    )
