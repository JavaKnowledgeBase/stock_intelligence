"""
Earnings Surprise Analytics — pre-earnings intelligence for traders.

Computes from yfinance (free, no API key):
  - Options-implied expected move (ATM straddle price / spot)
  - Historical beat/miss rate and EPS surprise history
  - Average post-earnings stock move (last 8 quarters)
  - IV Rank approximation (current ATM IV vs 1-year realized vol range)
  - Concrete straddle cost, breakeven prices, days until earnings
  - Strategy suggestion based on the above signals
"""

import numpy as np
import pandas as pd
import yfinance as yf


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mid(row) -> float:
    """Return mid-price or fall back to lastPrice."""
    bid = row.get("bid", 0) or 0
    ask = row.get("ask", 0) or 0
    last = row.get("lastPrice", 0) or 0
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    return last


def _find_post_earnings_expiration(expirations: list[str], earnings_date: pd.Timestamp) -> str | None:
    """Return the first expiration on or after the earnings date."""
    for exp in expirations:
        if pd.Timestamp(exp) >= earnings_date:
            return exp
    return None


# ── Core analytics ────────────────────────────────────────────────────────────

def get_atm_straddle(t: yf.Ticker, expiration: str, spot: float) -> dict:
    """
    Price the ATM straddle for a given expiration.
    Returns: atm_strike, call_price, put_price, straddle_cost,
             implied_move_pct, upside_be, downside_be, atm_iv
    """
    try:
        chain = t.option_chain(expiration)
        calls, puts = chain.calls, chain.puts
        if calls.empty or puts.empty:
            return {}

        # ATM = strike closest to spot
        atm_idx = (calls["strike"] - spot).abs().idxmin()
        atm_strike = calls.loc[atm_idx, "strike"]

        call_row = calls[calls["strike"] == atm_strike].iloc[0]
        put_row = puts[puts["strike"] == atm_strike].iloc[0]

        call_price = _mid(call_row)
        put_price = _mid(put_row)
        straddle = call_price + put_price

        call_iv = (call_row.get("impliedVolatility") or 0) * 100
        put_iv = (put_row.get("impliedVolatility") or 0) * 100
        atm_iv = (call_iv + put_iv) / 2

        return {
            "atm_strike": atm_strike,
            "call_price": round(call_price, 2),
            "put_price": round(put_price, 2),
            "straddle_cost": round(straddle, 2),
            "implied_move_pct": round(straddle / spot * 100, 1) if spot else 0.0,
            "upside_be": round(spot + straddle, 2),
            "downside_be": round(spot - straddle, 2),
            "atm_iv": round(atm_iv, 1),
            "expiration": expiration,
        }
    except Exception:
        return {}


def get_historical_earnings(t: yf.Ticker) -> pd.DataFrame:
    """
    Return past earnings with EPS estimate, actual, surprise %.
    Filters to rows where earnings have already occurred (epsActual not NaN).
    """
    try:
        df = t.earnings_dates
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        df = df.rename(columns={
            "EPS Estimate": "eps_estimate",
            "Reported EPS": "eps_actual",
            "Surprise(%)": "surprise_pct",
        })

        # Keep only past quarters with real data
        past = df[df["eps_actual"].notna()].copy()
        past = past.sort_index(ascending=False).head(12)  # last 12 quarters

        past["beat"] = past["eps_actual"] > past["eps_estimate"]
        past["surprise_pct"] = pd.to_numeric(past["surprise_pct"], errors="coerce")
        past.index.name = "earnings_date"
        return past.reset_index()
    except Exception:
        return pd.DataFrame()


def get_post_earnings_moves(t: yf.Ticker, earnings_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each past earnings date, calculate the actual 1-day stock move.
    Uses close-to-close: day before earnings vs day after announcement.
    """
    if earnings_df.empty:
        return pd.DataFrame()

    try:
        hist = t.history(period="3y", auto_adjust=True)
        if hist.empty:
            return pd.DataFrame()

        hist.index = hist.index.tz_localize(None)
        rows = []

        for _, row in earnings_df.iterrows():
            edate = pd.Timestamp(row["earnings_date"])
            try:
                before_prices = hist[hist.index < edate]["Close"]
                after_prices = hist[hist.index > edate]["Close"]
                if before_prices.empty or after_prices.empty:
                    continue
                close_before = before_prices.iloc[-1]
                close_after = after_prices.iloc[0]
                move_pct = (close_after - close_before) / close_before * 100
                rows.append({
                    "earnings_date": edate.strftime("%Y-%m-%d"),
                    "close_before": round(close_before, 2),
                    "close_after": round(close_after, 2),
                    "move_pct": round(move_pct, 1),
                    "direction": "🟢 Up" if move_pct > 0 else "🔴 Down",
                    "eps_beat": row.get("beat", None),
                    "surprise_pct": row.get("surprise_pct", None),
                })
            except (IndexError, KeyError):
                continue

        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def get_iv_rank(t: yf.Ticker, current_iv: float) -> float | None:
    """
    Approximate IV Rank: (current IV - 52w low) / (52w high - 52w low) × 100.
    We proxy historical IV with rolling 30-day annualised realised vol.
    """
    try:
        hist = t.history(period="1y", auto_adjust=True)
        if hist.empty or len(hist) < 40:
            return None
        rets = hist["Close"].pct_change().dropna()
        rolling_rv = rets.rolling(30).std() * (252 ** 0.5) * 100
        rolling_rv = rolling_rv.dropna()
        if rolling_rv.empty:
            return None
        lo, hi = rolling_rv.min(), rolling_rv.max()
        if hi == lo:
            return 50.0
        rank = (current_iv - lo) / (hi - lo) * 100
        return round(float(np.clip(rank, 0, 100)), 0)
    except Exception:
        return None


def _suggest_strategy(
    implied_move_pct: float,
    avg_hist_move: float,
    beat_rate: float,
    iv_rank: float | None,
    days_to_earnings: int,
) -> list[str]:
    tips = []

    # Implied vs historical move comparison
    if avg_hist_move > 0:
        ratio = implied_move_pct / avg_hist_move
        if ratio < 0.75:
            tips.append(
                f"📉 Options UNDERPRICING historical move "
                f"(implied {implied_move_pct:.1f}% vs avg {avg_hist_move:.1f}%) "
                f"— straddle/strangle buying favoured"
            )
        elif ratio > 1.35:
            tips.append(
                f"📈 Options OVERPRICING historical move "
                f"(implied {implied_move_pct:.1f}% vs avg {avg_hist_move:.1f}%) "
                f"— premium selling / iron condor favoured"
            )
        else:
            tips.append(
                f"⚖️ Options fairly priced vs history "
                f"(implied {implied_move_pct:.1f}% vs avg {avg_hist_move:.1f}%)"
            )

    # Beat rate
    if beat_rate >= 75:
        tips.append(
            f"✅ Strong beat history ({beat_rate:.0f}% of quarters) — slight bullish directional bias"
        )
    elif beat_rate <= 40:
        tips.append(
            f"⚠️ Weak beat history ({beat_rate:.0f}% of quarters) — slight bearish directional bias"
        )
    else:
        tips.append(f"➡️ Mixed beat history ({beat_rate:.0f}%) — no clear directional edge")

    # IV Rank
    if iv_rank is not None:
        if iv_rank >= 75:
            tips.append(f"🔥 IV Rank {iv_rank:.0f}/100 — options expensive, favour selling premium")
        elif iv_rank <= 30:
            tips.append(f"💤 IV Rank {iv_rank:.0f}/100 — options cheap, favour buying premium")
        else:
            tips.append(f"📊 IV Rank {iv_rank:.0f}/100 — neutral, no strong vol bias")

    # Timing
    if days_to_earnings <= 3:
        tips.append(
            "⏰ Earnings in ≤3 days — theta decay accelerating, time-spread "
            "risk high; size positions conservatively"
        )
    elif days_to_earnings <= 7:
        tips.append("📅 1 week to earnings — straddle decay picking up; optimal entry window")

    return tips


# ── Main entry point ──────────────────────────────────────────────────────────

def analyse_earnings(ticker: str) -> dict:
    """
    Full pre-earnings intelligence package for a single ticker.

    Returns a dict with keys:
      ticker, spot, next_earnings_date, days_to_earnings,
      straddle (dict), iv_rank, earnings_history (df),
      post_moves (df), avg_hist_move, beat_rate, strategy_tips
    """
    result: dict = {"ticker": ticker.upper(), "error": None}

    try:
        t = yf.Ticker(ticker)
        spot = getattr(t.fast_info, "last_price", None) or 0.0
        if not spot:
            result["error"] = f"Could not fetch price for {ticker}"
            return result
        result["spot"] = round(spot, 2)

        # ── Upcoming earnings date ────────────────────────────────────────────
        next_earnings = None
        try:
            cal = t.calendar
            if cal is not None and not cal.empty:
                dates = cal.get("Earnings Date") or cal.get("Earnings Date", [None])
                if hasattr(dates, '__iter__') and not isinstance(dates, str):
                    for d in dates:
                        ts = pd.Timestamp(d)
                        if ts >= pd.Timestamp.now():
                            next_earnings = ts
                            break
        except Exception:
            pass

        if next_earnings is None:
            # Fallback: check earnings_dates for future entries
            try:
                df_ed = t.earnings_dates
                if df_ed is not None and not df_ed.empty:
                    future = df_ed[df_ed["Reported EPS"].isna()]
                    if not future.empty:
                        future_idx = pd.to_datetime(future.index, utc=True).tz_localize(None)
                        future_idx = future_idx[future_idx > pd.Timestamp.now()]
                        if len(future_idx) > 0:
                            next_earnings = future_idx.min()
            except Exception:
                pass

        result["next_earnings_date"] = (
            next_earnings.strftime("%Y-%m-%d") if next_earnings else None
        )
        result["days_to_earnings"] = (
            (next_earnings - pd.Timestamp.now()).days if next_earnings else None
        )

        # ── ATM Straddle ─────────────────────────────────────────────────────
        straddle = {}
        expirations = t.options or []
        if expirations and next_earnings:
            target_exp = _find_post_earnings_expiration(expirations, next_earnings)
            if target_exp:
                straddle = get_atm_straddle(t, target_exp, spot)
        elif expirations:
            straddle = get_atm_straddle(t, expirations[0], spot)
        result["straddle"] = straddle

        # ── IV Rank ───────────────────────────────────────────────────────────
        atm_iv = straddle.get("atm_iv", 0)
        result["iv_rank"] = get_iv_rank(t, atm_iv) if atm_iv else None

        # ── Earnings history (EPS beat/miss) ──────────────────────────────────
        earnings_hist = get_historical_earnings(t)
        result["earnings_history"] = earnings_hist

        if not earnings_hist.empty and "beat" in earnings_hist.columns:
            beats = earnings_hist["beat"].dropna()
            result["beat_rate"] = round(beats.mean() * 100, 1) if len(beats) > 0 else None
            result["beat_count"] = int(beats.sum())
            result["total_quarters"] = int(len(beats))
        else:
            result["beat_rate"] = None
            result["beat_count"] = 0
            result["total_quarters"] = 0

        # ── Post-earnings price moves ─────────────────────────────────────────
        post_moves = get_post_earnings_moves(t, earnings_hist)
        result["post_moves"] = post_moves

        if not post_moves.empty:
            abs_moves = post_moves["move_pct"].abs()
            result["avg_hist_move"] = round(float(abs_moves.mean()), 1)
            result["max_hist_move"] = round(float(abs_moves.max()), 1)
            result["median_hist_move"] = round(float(abs_moves.median()), 1)
        else:
            result["avg_hist_move"] = None
            result["max_hist_move"] = None
            result["median_hist_move"] = None

        # ── Strategy tips ─────────────────────────────────────────────────────
        result["strategy_tips"] = _suggest_strategy(
            implied_move_pct=straddle.get("implied_move_pct", 0),
            avg_hist_move=result["avg_hist_move"] or 0,
            beat_rate=result["beat_rate"] or 50,
            iv_rank=result["iv_rank"],
            days_to_earnings=result["days_to_earnings"] or 999,
        )

    except Exception as e:
        result["error"] = str(e)

    return result
