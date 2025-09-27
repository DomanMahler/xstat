"""
Data feeds for xstat platform.
"""

from xstat.datafeeds.yfinance_feed import YFinanceFeed
from xstat.datafeeds.ccxt_feed import CCXTFeed
from xstat.datafeeds.parquet_feed import ParquetFeed

__all__ = [
    "YFinanceFeed",
    "CCXTFeed", 
    "ParquetFeed",
]
