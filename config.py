# config.py
"""
Central configuration file.
Change values here — the rest of the project adapts automatically.
"""

# ── Deep-analysis tickers (have pre-trained ML models) ────────────────────────
TICKERS = [
    # Mega Cap Tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "AMD", "NFLX", "AVGO", "INTU",
    # Large Tech / Software
    "CRM", "ADBE", "INTC", "ORCL", "CSCO",
    "NOW", "PANW", "SNOW", "SHOP", "UBER",
    # Financials
    "JPM", "BAC", "GS", "MS", "WFC",
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "XLF",
    "XLK", "XLE", "GLD", "SLV", "TLT",
    # Cyclicals
    "XOM", "CVX", "BA", "CAT", "GE", "KO",
    # High volatility
    "COIN", "MARA", "RIVN", "PLTR", "MSTR", "WBD",
]

# ── Market scan universe — 500 tickers ────────────────────────────────────────
# Used by all scanners: Volume Leaders, Rapid Movers, Strategy, Forecast,
# Short Squeeze, Earnings Calendar, Screener, Insider Flow.
MARKET_SCAN_TICKERS = [
    # ── Mega / Large Cap Tech ──────────────────────────────────────────────
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO",
    "AMD", "ORCL", "CSCO", "INTC", "ADBE", "CRM", "INTU", "NOW", "PANW",
    "SNOW", "SHOP", "UBER", "LYFT", "DDOG", "NET", "CRWD", "ZS", "OKTA",
    "MDB", "GTLB", "PATH", "AFRM", "HOOD", "SOFI", "UPST", "BILL",
    "HUBS", "TWLO", "ZM", "DOCU", "TEAM", "ESTC", "COUP",
    "NTNX", "PSTG", "WD", "SMAR", "NCNO", "ASAN", "BRZE",

    # ── Semiconductors ─────────────────────────────────────────────────────
    "MU", "QCOM", "TXN", "AMAT", "LRCX", "KLAC", "MRVL", "SMCI", "ARM",
    "ON", "MPWR", "SWKS", "QRVO", "MCHP", "ADI", "NXPI", "WOLF", "AMBA",
    "FORM", "SLAB", "DIOD", "SITM", "COHR", "AEHR", "CEVA",

    # ── Internet / E-Commerce ──────────────────────────────────────────────
    "NFLX", "SPOT", "ABNB", "DASH", "YELP", "TRIP", "EXPE", "BKNG", "VRBO",
    "ETSY", "EBAY", "W", "CHWY", "CVNA", "CARG", "OPEN",
    "RDFN", "Z", "ZG", "MTTR",

    # ── Fintech / Payments ─────────────────────────────────────────────────
    "V", "MA", "PYPL", "SQ", "AFRM", "COIN", "MARA", "RIOT", "CLSK",
    "MSTR", "HUT", "BITF", "CIFR", "BTBT", "WULF",

    # ── Large Cap Financials ───────────────────────────────────────────────
    "JPM", "BAC", "GS", "MS", "WFC", "C", "USB", "PNC", "TFC", "SCHW",
    "AXP", "COF", "DFS", "SYF", "ALLY", "BX", "KKR", "APO", "ARES",
    "BLK", "TROW", "STT", "BK", "NTRS", "FRC", "SIVB", "ZION",
    "CFG", "RF", "KEY", "FITB", "HBAN", "MTB", "WAL", "EWBC",
    "ICE", "CME", "CBOE", "NDAQ", "MKTX", "GPN", "FIS", "FISV",
    "WEX", "JKHY", "EPAM",

    # ── Healthcare / Biotech ───────────────────────────────────────────────
    "LLY", "JNJ", "UNH", "PFE", "MRK", "ABBV", "BMY", "GILD", "AMGN",
    "BIIB", "REGN", "VRTX", "MRNA", "BNTX", "NVAX", "SGEN", "ALNY",
    "ARWR", "BEAM", "EDIT", "CRSP", "NTLA", "FATE", "BLUE", "RCKT",
    "HALO", "INCY", "EXEL", "JAZZ", "ITCI", "ACAD", "SAGE", "PRAX",
    "LGND", "FOLD", "ARCT", "OCGN", "CTIC",
    "CVS", "MCK", "ABC", "CAH", "HSIC", "PDCO",
    "CI", "HUM", "ELV", "MOH", "CNC", "HCA", "THC", "UHS",
    "TMO", "DHR", "A", "BIO", "IDXX", "MTD", "WAT", "ILMN",
    "DXCM", "ABT", "STE", "BAX", "BDX", "ZBH", "SYK", "ISRG", "MDT",
    "EW", "BSX", "HOLX", "ALGN", "NUVA", "TMDX",

    # ── Consumer / Retail ──────────────────────────────────────────────────
    "WMT", "COST", "TGT", "KR", "DG", "DLTR", "BJ",
    "MCD", "SBUX", "YUM", "CMG", "DPZ", "QSR", "JACK", "SHAK",
    "NKE", "LULU", "UAA", "VFC", "PVH", "RL", "TPR", "CPRI",
    "HD", "LOW", "TJX", "ROST", "BURL", "ANF", "AEO", "GPS",
    "AMZN", "ETSY", "EBAY", "CHWY", "PDD", "BABA", "JD",
    "PEP", "KO", "MDLZ", "GIS", "CPB", "HRL", "SJM", "MKC",
    "PM", "MO", "BTI", "STZ", "BUD", "TAP", "SAM",
    "EL", "CLX", "PG", "CL", "CHD", "ENR", "REV", "COTY",

    # ── Energy ─────────────────────────────────────────────────────────────
    "XOM", "CVX", "COP", "EOG", "PXD", "DVN", "FANG", "MRO", "APA",
    "HAL", "SLB", "BKR", "NOV", "HP", "RIG", "VAL",
    "ET", "EPD", "MMP", "OKE", "KMI", "WMB", "TRGP", "LNG", "CQP",
    "VLO", "PSX", "MPC", "HES", "OXY",

    # ── Industrials / Aerospace ────────────────────────────────────────────
    "BA", "RTX", "LMT", "NOC", "GD", "HII", "L3H", "KTOS", "AXON",
    "CAT", "DE", "CMI", "PCAR", "IR", "ROK", "EMR", "HON",
    "GE", "MMM", "ITW", "PH", "ETN", "AME", "VRSK",
    "UPS", "FDX", "XPO", "ODFL", "SAIA", "JBHT", "CHRW",
    "CSX", "UNP", "NSC", "CP", "CNI",

    # ── Real Estate / REITs ────────────────────────────────────────────────
    "AMT", "PLD", "EQIX", "CCI", "SPG", "O", "VICI", "WELL",
    "AVB", "EQR", "MAA", "NLY", "AGNC",

    # ── Utilities ──────────────────────────────────────────────────────────
    "NEE", "DUK", "SO", "D", "EXC", "XEL", "WEC", "ES", "AEP",

    # ── Telecom / Media ────────────────────────────────────────────────────
    "T", "VZ", "TMUS", "CMCSA", "CHTR", "LUMN", "DISH",
    "DIS", "PARA", "WBD", "FOXA", "NWSA", "NYT",
    "RDDT", "SNAP", "PINS", "MTCH", "BMBL", "IAC",

    # ── Auto / EV ──────────────────────────────────────────────────────────
    "TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI", "F", "GM",
    "STLA", "TM", "HMC", "RACE", "APTV", "LEA",

    # ── ETFs ───────────────────────────────────────────────────────────────
    "SPY", "QQQ", "IWM", "DIA", "MDY", "VXX", "UVXY",
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLU", "XLB", "XLRE",
    "GLD", "SLV", "GDX", "GDXJ", "USO", "UNG",
    "TLT", "IEF", "HYG", "LQD", "JNK",
    "ARKK", "ARKG", "ARKW", "ARKF",
    "SOXS", "SOXL", "TQQQ", "SQQQ", "SPXU", "UPRO",

    # ── High-volatility / Meme / Momentum ─────────────────────────────────
    "GME", "AMC", "BBBY", "PLTR", "MSTR", "COIN", "MARA", "RIOT",
    "DKNG", "RBLX", "U", "RBLX", "HOOD", "SOFI", "AFRM", "OPEN",
    "SPCE", "JOBY", "LILM", "ACHR", "BLDE", "EH",
    "SMCI", "ANET", "DELL", "HPE", "HPQ", "NTAP", "STX", "WDC",

    # ── China ADRs ─────────────────────────────────────────────────────────
    "BABA", "JD", "BIDU", "PDD", "NIO", "XPEV", "LI", "BILI",
    "DIDI", "EDU", "TAL", "VIPS", "IQ", "HUYA", "DOYU",
]

# Deduplicate while preserving order
_seen: set = set()
_deduped: list = []
for _t in MARKET_SCAN_TICKERS:
    if _t not in _seen:
        _seen.add(_t)
        _deduped.append(_t)
MARKET_SCAN_TICKERS = _deduped

INTRADAY_DAYS = 360
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"
DATA_DIR = "data"
