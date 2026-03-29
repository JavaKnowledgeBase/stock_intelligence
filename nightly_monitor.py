import json
import os
import smtplib
import subprocess
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from config import MARKET_SCAN_TICKERS
from options_data import build_price_forecast_table, build_strategy_table
from r2_storage import ensure_assets_available


TIMEZONE = ZoneInfo("America/New_York")
REPORT_ROOT = Path("reports") / "nightly"
MIN_FORECAST_ROWS = 5
MIN_STRATEGY_ROWS = 5
RECOMMENDED_CONFIDENCE = 70.0


def _now_et():
    return datetime.now(TIMEZONE)


def _git_output(args):
    try:
        result = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_build_metadata():
    branch = os.getenv("MONITOR_BRANCH_NAME") or _git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    commit = (
        os.getenv("STREAMLIT_BUILD_COMMIT")
        or os.getenv("GITHUB_SHA")
        or os.getenv("COMMIT_SHA")
        or _git_output(["git", "rev-parse", "--short", "HEAD"])
    )
    return {"branch": branch, "commit": commit[:7] if commit != "unknown" else commit}


def _read_previous_summary(current_path: Path):
    if not REPORT_ROOT.exists():
        return None
    candidates = sorted(p for p in REPORT_ROOT.glob("*.json") if p != current_path)
    if not candidates:
        return None
    try:
        return json.loads(candidates[-1].read_text())
    except Exception:
        return None


def _confidence_stats(df: pd.DataFrame, column: str):
    if df is None or df.empty or column not in df.columns:
        return {"count": 0, "avg": None, "median": None, "min": None, "max": None}
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return {"count": 0, "avg": None, "median": None, "min": None, "max": None}
    return {
        "count": int(series.shape[0]),
        "avg": round(float(series.mean()), 2),
        "median": round(float(series.median()), 2),
        "min": round(float(series.min()), 2),
        "max": round(float(series.max()), 2),
    }


def _build_recommendations(gainers, losers, strategy_df, previous_summary):
    recommendations = []
    total_forecast = len(gainers) + len(losers)
    strategy_count = len(strategy_df)
    forecast_conf = _confidence_stats(pd.concat([gainers, losers], ignore_index=True) if total_forecast else pd.DataFrame(), "forecast_confidence")
    strategy_conf = _confidence_stats(strategy_df, "strategy_confidence")

    if len(gainers) < MIN_FORECAST_ROWS:
        recommendations.append(
            f"Forecast gainers are thin ({len(gainers)} rows). Consider slightly relaxing the forecast confidence floor or widening the candidate universe if this persists."
        )
    if len(losers) < MIN_FORECAST_ROWS:
        recommendations.append(
            f"Forecast losers are thin ({len(losers)} rows). Consider slightly relaxing the forecast confidence floor or widening the candidate universe if this persists."
        )
    if strategy_count < MIN_STRATEGY_ROWS:
        recommendations.append(
            f"Strategy Ideas returned only {strategy_count} rows. Review the confidence floor and contract liquidity filters if this repeats."
        )
    if forecast_conf["median"] is not None and forecast_conf["median"] < RECOMMENDED_CONFIDENCE:
        recommendations.append(
            f"Median forecast confidence is {forecast_conf['median']}%, below the preferred 70% mark. Treat the table as exploratory rather than high conviction."
        )
    if strategy_conf["median"] is not None and strategy_conf["median"] < RECOMMENDED_CONFIDENCE:
        recommendations.append(
            f"Median strategy confidence is {strategy_conf['median']}%, below the preferred 70% mark. Consider tightening risk sizing or rechecking filters."
        )

    if previous_summary:
        prev_total_forecast = int(previous_summary.get("forecast", {}).get("total_rows", 0))
        prev_strategy_count = int(previous_summary.get("strategy", {}).get("count", 0))
        if prev_total_forecast and total_forecast < max(1, int(prev_total_forecast * 0.6)):
            recommendations.append(
                f"Forecast coverage dropped materially versus the last run ({total_forecast} vs {prev_total_forecast}). Review whether market conditions or filters changed too much."
            )
        if prev_strategy_count and strategy_count < max(1, int(prev_strategy_count * 0.6)):
            recommendations.append(
                f"Strategy coverage dropped materially versus the last run ({strategy_count} vs {prev_strategy_count}). Review confidence and liquidity thresholds."
            )

    if not recommendations:
        recommendations.append("No urgent logic tweaks recommended from this run. Continue monitoring confidence distribution and row counts.")

    return recommendations, forecast_conf, strategy_conf


def _table_preview(df: pd.DataFrame, columns):
    if df is None or df.empty:
        return []
    available = [col for col in columns if col in df.columns]
    return df[available].head(5).to_dict(orient="records")


def build_summary(gainers, losers, strategy_df, diagnostics, previous_summary, metadata):
    recommendations, forecast_conf, strategy_conf = _build_recommendations(gainers, losers, strategy_df, previous_summary)
    timestamp = _now_et()
    return {
        "generated_at": timestamp.isoformat(),
        "generated_at_label": timestamp.strftime("%Y-%m-%d %I:%M:%S %p %Z"),
        "build": metadata,
        "forecast": {
            "gainers": len(gainers),
            "losers": len(losers),
            "total_rows": len(gainers) + len(losers),
            "confidence": forecast_conf,
            "top_gainers_preview": _table_preview(gainers, ["ticker", "est_1w_pct", "forecast_confidence"]),
            "top_losers_preview": _table_preview(losers, ["ticker", "est_1w_pct", "forecast_confidence"]),
        },
        "strategy": {
            "count": len(strategy_df),
            "confidence": strategy_conf,
            "diagnostics": diagnostics,
            "top_preview": _table_preview(strategy_df, ["ticker", "horizon", "strategy_score", "strategy_confidence"]),
        },
        "recommendations": recommendations,
    }


def _write_report_files(summary, gainers, losers, strategy_df):
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = _now_et().strftime("%Y%m%d_%H%M%S")
    summary_path = REPORT_ROOT / f"nightly_monitor_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    gainers.to_csv(REPORT_ROOT / f"forecast_gainers_{stamp}.csv", index=False)
    losers.to_csv(REPORT_ROOT / f"forecast_losers_{stamp}.csv", index=False)
    strategy_df.to_csv(REPORT_ROOT / f"strategy_ideas_{stamp}.csv", index=False)
    return summary_path


def _email_enabled():
    required = [
        "SMTP_HOST",
        "SMTP_PORT",
        "SMTP_USERNAME",
        "SMTP_PASSWORD",
        "MONITOR_EMAIL_TO",
    ]
    return all(os.getenv(name) for name in required)


def _send_email(summary):
    if not _email_enabled():
        return {"status": "skipped", "detail": "SMTP env vars not fully configured."}

    msg = EmailMessage()
    recipient = os.environ["MONITOR_EMAIL_TO"]
    sender = os.getenv("SMTP_FROM") or os.environ["SMTP_USERNAME"]
    msg["Subject"] = f"Nightly Stock Intelligence Check - {summary['generated_at_label']}"
    msg["From"] = sender
    msg["To"] = recipient

    recommendation_lines = "\n".join(f"- {item}" for item in summary["recommendations"])
    body = f"""Nightly monitor completed.

Build: {summary['build']['branch']} @ {summary['build']['commit']}
Generated: {summary['generated_at_label']}

Forecast:
- Gainers: {summary['forecast']['gainers']}
- Losers: {summary['forecast']['losers']}
- Median Confidence: {summary['forecast']['confidence']['median']}

Strategy:
- Rows: {summary['strategy']['count']}
- Median Confidence: {summary['strategy']['confidence']['median']}
- Status: {summary['strategy']['diagnostics'].get('status')}

Recommendations:
{recommendation_lines}
"""
    msg.set_content(body)

    host = os.environ["SMTP_HOST"]
    port = int(os.environ["SMTP_PORT"])
    username = os.environ["SMTP_USERNAME"]
    password = os.environ["SMTP_PASSWORD"]
    use_ssl = os.getenv("SMTP_USE_SSL", "false").lower() == "true"

    if use_ssl:
        with smtplib.SMTP_SSL(host, port, timeout=30) as server:
            server.login(username, password)
            server.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.ehlo()
            if os.getenv("SMTP_USE_TLS", "true").lower() == "true":
                server.starttls()
                server.ehlo()
            server.login(username, password)
            server.send_message(msg)

    return {"status": "sent", "detail": f"Email sent to {recipient}"}


def _safe_forecast_run():
    try:
        gainers, losers = build_price_forecast_table(MARKET_SCAN_TICKERS, top_n=10)
        return gainers, losers, {"status": "ok", "detail": ""}
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), {"status": "error", "detail": str(exc)}


def _safe_strategy_run():
    try:
        strategy_df, diagnostics = build_strategy_table(MARKET_SCAN_TICKERS, top_n=10)
        return strategy_df, diagnostics
    except Exception as exc:
        return pd.DataFrame(), {"status": "error", "message": str(exc)}


def main():
    load_dotenv()
    ensure_assets_available()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    metadata = get_build_metadata()
    gainers, losers, forecast_status = _safe_forecast_run()
    strategy_df, diagnostics = _safe_strategy_run()

    temp_summary_path = REPORT_ROOT / "_temp.json"
    previous_summary = _read_previous_summary(temp_summary_path)
    summary = build_summary(gainers, losers, strategy_df, diagnostics, previous_summary, metadata)
    summary["forecast"]["run_status"] = forecast_status
    summary_path = _write_report_files(summary, gainers, losers, strategy_df)
    email_result = _send_email(summary)
    summary["email"] = email_result
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
