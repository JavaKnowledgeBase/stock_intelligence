# config.py  single source of truth

"""
Central configuration file.
Change values here — the rest of the project adapts automatically.
"""

# Stocks included in the pooled model
# TICKERS = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"]

TICKERS = ["NVDA", "AAPL", "GOOG", "GOOGL", "MSFT", "AMZN", "AVGO", 
           "TSLA", "WMT", "MU", "BAC", "ORCL", 
           "NFLX", "AMD", "CSCO", "PLTR", "INTC", "VZ", "T", "UBER", 
           "PFE", "CMCSA", "HOOD", "WBD", "CCL", "HPE", "SMCI"]

# How many calendar days of intraday data to download
# (Yahoo limits intraday history per request)
INTRADAY_DAYS = 360

# Market hours (US equities)
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"

# Base data directory
DATA_DIR = "data"
