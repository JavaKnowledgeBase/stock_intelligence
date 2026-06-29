import pandas as pd


def calculate_gex(chain):
    """Calculate dealer Gamma Exposure (GEX) by strike — backward-compat entry point."""
    if chain is None or chain.empty:
        return None
    if "gamma" not in chain.columns:
        return None
    chain = chain.copy()
    chain["gex"] = chain["gamma"] * chain["openInterest"] * 100 * chain["strike"]
    gex_by_strike = chain.groupby("strike")["gex"].sum()
    return gex_by_strike.sort_index()


def calculate_gex_enhanced(chain: pd.DataFrame, spot_price: float = 0.0) -> dict:
    """
    Full GEX profile with key dealer levels.

    Returns a dict:
      gex_by_strike  — pd.Series indexed by strike
      net_gex        — aggregate net GEX (positive = long gamma / stabilising)
      call_wall      — strike with highest positive GEX (resistance)
      put_wall       — strike with most negative GEX (support)
      gamma_flip     — price where dealer net GEX crosses zero
      dealer_bias    — human-readable bias string
    """
    if chain is None or chain.empty or "gamma" not in chain.columns:
        return {}

    chain = chain.copy()
    for col in ["gamma", "openInterest", "strike"]:
        chain[col] = pd.to_numeric(chain.get(col, 0), errors="coerce").fillna(0)

    # Standard GEX sign convention:
    # Dealers sell calls → net short calls → positive GEX (they buy dips)
    # Dealers buy puts → net long puts  → negative GEX (they sell rallies)
    has_type = "type" in chain.columns
    if has_type:
        calls = chain[chain["type"] == "call"].copy()
        puts = chain[chain["type"] == "put"].copy()
        calls["gex"] = calls["gamma"] * calls["openInterest"] * 100 * calls["strike"]
        puts["gex"] = -puts["gamma"] * puts["openInterest"] * 100 * puts["strike"]
        combined = pd.concat([calls[["strike", "gex"]], puts[["strike", "gex"]]])
    else:
        # Fallback: treat all rows as calls (old format)
        chain["gex"] = chain["gamma"] * chain["openInterest"] * 100 * chain["strike"]
        combined = chain[["strike", "gex"]]

    gex_by_strike = combined.groupby("strike")["gex"].sum().sort_index()

    if gex_by_strike.empty:
        return {}

    net_gex = gex_by_strike.sum()
    call_wall = float(gex_by_strike.idxmax())
    put_wall = float(gex_by_strike.idxmin())

    # Gamma flip: nearest strike where cumulative sign changes
    gamma_flip = None
    if spot_price > 0:
        # Look at strikes within ±15% of spot
        near = gex_by_strike[
            (gex_by_strike.index >= spot_price * 0.85)
            & (gex_by_strike.index <= spot_price * 1.15)
        ]
        if len(near) >= 2:
            signs = near.apply(lambda v: 1 if v >= 0 else -1)
            changes = signs[signs != signs.shift(1)].index
            if len(changes) > 0:
                # Pick the flip closest to spot
                gamma_flip = float(changes[abs(changes - spot_price).argmin()])

    if net_gex > 0:
        dealer_bias = f"Long Gamma — dealers stabilise price (absorb moves)"
    else:
        dealer_bias = f"Short Gamma — dealers amplify price moves (momentum)"

    return {
        "gex_by_strike": gex_by_strike,
        "net_gex": net_gex,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "gamma_flip": gamma_flip,
        "dealer_bias": dealer_bias,
    }


def get_gex_for_ticker(ticker: str, expiration: str | None = None) -> tuple[float, dict]:
    """
    Fetch options chain for ticker and return (spot_price, gex_data).
    Uses first 2 expirations if none specified.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        spot = getattr(t.fast_info, "last_price", None) or 0.0
        exps = t.options
        if not exps:
            return spot, {}

        target_exps = [expiration] if expiration else exps[:2]
        chains = []
        for exp in target_exps:
            try:
                opt = t.option_chain(exp)
                calls = opt.calls.copy()
                calls["type"] = "call"
                puts = opt.puts.copy()
                puts["type"] = "put"
                chains.extend([calls, puts])
            except Exception:
                continue

        if not chains:
            return spot, {}

        chain = pd.concat(chains, ignore_index=True)
        gex_data = calculate_gex_enhanced(chain, spot_price=spot)
        return spot, gex_data
    except Exception:
        return 0.0, {}
