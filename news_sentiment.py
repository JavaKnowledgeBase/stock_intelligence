"""
News sentiment scanner — fetches recent headlines from Finviz and scores
each one with a keyword-based bullish/bearish signal. No ML dependency.
"""

import concurrent.futures
import re

import pandas as pd

# ── Sentiment word lists ──────────────────────────────────────────────────────

_BULLISH = {
    "upgrade", "upgrades", "upgraded", "beat", "beats", "surpass", "surpasses",
    "surpassed", "outperform", "outperforms", "record", "records", "profit",
    "profits", "growth", "rally", "rallies", "rallied", "buy", "strong",
    "strength", "positive", "bullish", "raises", "raised", "raise", "exceeds",
    "exceeded", "exceed", "breakout", "soars", "soar", "surged", "surge",
    "jumps", "jump", "climbs", "climb", "higher", "gains", "gain", "won",
    "wins", "approval", "approved", "partnership", "deal", "contract",
    "dividend", "buyback", "repurchase", "expansion", "launches", "launch",
    "innovative", "breakthrough", "milestone", "optimistic", "confident",
    "tops", "top", "boosts", "boost", "accelerates", "accelerate",
}

_BEARISH = {
    "downgrade", "downgrades", "downgraded", "miss", "misses", "missed",
    "disappoint", "disappoints", "disappointed", "underperform", "underperforms",
    "loss", "losses", "decline", "declines", "sell", "weak", "weakness",
    "negative", "bearish", "lowers", "lowered", "lower", "falls", "fall",
    "fell", "slump", "slumps", "slumped", "drops", "drop", "plunges",
    "plunge", "crash", "crashes", "cut", "cuts", "layoff", "layoffs",
    "recall", "fine", "fines", "lawsuit", "litigation", "fraud", "probe",
    "investigation", "warning", "warns", "warn", "concern", "concerns",
    "risks", "risk", "headwinds", "slowdown", "slow", "missed", "guidance",
    "reduced", "reduce", "shrinks", "shrink", "bankruptcy", "default",
    "debt", "deficit", "loss", "losses", "disappointing", "disappoints",
}


def _score_headline(title: str) -> float:
    """Return sentiment score in [-1, 1]: positive = bullish, negative = bearish."""
    words = re.findall(r"[a-z]+", title.lower())
    if not words:
        return 0.0
    bull = sum(1 for w in words if w in _BULLISH)
    bear = sum(1 for w in words if w in _BEARISH)
    if bull == 0 and bear == 0:
        return 0.0
    return (bull - bear) / (bull + bear)


def _label(score: float) -> str:
    if score >= 0.3:
        return "Bullish"
    if score <= -0.3:
        return "Bearish"
    return "Neutral"


def _fetch_ticker_news(ticker: str, max_headlines: int = 10) -> list[dict]:
    try:
        from finvizfinance.quote import finvizfinance
        news_df = finvizfinance(ticker).ticker_news()
        rows = []
        for _, row in news_df.head(max_headlines).iterrows():
            title = str(row.get("Title", "")).strip()
            score = _score_headline(title)
            rows.append({
                "ticker": ticker,
                "date": row.get("Date"),
                "title": title,
                "source": str(row.get("Source", "")).strip(),
                "link": str(row.get("Link", "")).strip(),
                "score": round(score, 3),
                "sentiment": _label(score),
            })
        return rows
    except Exception:
        return []


def get_news_sentiment(
    tickers: list[str],
    max_headlines_per_ticker: int = 5,
    workers: int = 8,
) -> pd.DataFrame:
    """
    Fetch and score recent headlines for each ticker.
    Returns a DataFrame sorted by date descending.
    """
    all_rows: list[dict] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_ticker_news, t, max_headlines_per_ticker): t
            for t in tickers
        }
        for future in concurrent.futures.as_completed(futures):
            all_rows.extend(future.result())

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    return df


def aggregate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise per-ticker sentiment: avg score, headline count, bullish/bearish counts.
    """
    if df.empty:
        return pd.DataFrame()

    agg = (
        df.groupby("ticker")
        .agg(
            headlines=("title", "count"),
            avg_score=("score", "mean"),
            bullish=("sentiment", lambda s: (s == "Bullish").sum()),
            neutral=("sentiment", lambda s: (s == "Neutral").sum()),
            bearish=("sentiment", lambda s: (s == "Bearish").sum()),
        )
        .reset_index()
    )
    agg["avg_score"] = agg["avg_score"].round(3)
    agg["overall"] = agg["avg_score"].apply(_label)
    return agg.sort_values("avg_score", ascending=False).reset_index(drop=True)
