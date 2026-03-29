# config.py

"""
Central configuration file.
Change values here — the rest of the project adapts automatically.
"""

TICKERS = [

    # Mega Cap Tech
    "AAPL","MSFT","NVDA","AMZN","GOOGL",
    "META","TSLA","AMD","NFLX","AVGO","INTU",

    # Large Tech / Software
    "CRM","ADBE","INTC","ORCL","CSCO",
    "NOW","PANW","SNOW","SHOP","UBER",

    # Financials
    "JPM","BAC","GS","MS","WFC",

    # ETFs
    "SPY","QQQ","IWM","DIA","XLF",
    "XLK","XLE","GLD","SLV","TLT",

    # Cyclicals
    "XOM","CVX","BA","CAT","GE","KO",

    # High volatility
    "COIN","MARA","RIVN","PLTR","MSTR","WBD"
]

MARKET_SCAN_TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "AMD", "NFLX", "AVGO", "INTC", "ORCL", "CSCO", "CRM",
    "ADBE", "NOW", "PANW", "SNOW", "SHOP", "UBER", "JPM", "BAC", "GS",
    "MS", "WFC", "XLF", "XLE", "XLK", "XOM", "CVX", "BA", "CAT", "GE",
    "KO", "MCD", "DIS", "WMT", "COST", "TGT", "NKE", "SBUX", "PEP", "PFE",
    "LLY", "JNJ", "MRK", "ABBV", "UNH", "V", "MA", "PYPL", "COIN", "MARA",
    "RIVN", "PLTR", "MSTR", "MU", "ARM", "SMCI", "ANET", "DELL", "RDDT",
    "HOOD", "DKNG", "SOFI", "BABA", "BIDU", "PDD", "NIO", "XPEV", "LI",
    "F", "GM", "LCID", "RBLX", "NET", "CRWD", "ZS", "OKTA", "DDOG", "MDB",
    "ROKU", "SQ", "SHOP", "TLT", "GLD", "SLV", "USO", "ARKK", "HYG", "LQD"
]

INTRADAY_DAYS = 360
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"
DATA_DIR = "data"
