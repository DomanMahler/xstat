"""
Data feeds for xstat platform.
"""

from .yfinance_feed import YFinanceFeed
from .ccxt_feed import CCXTFeed
from .parquet_feed import ParquetFeed

__all__ = [
    "YFinanceFeed",
    "CCXTFeed", 
    "ParquetFeed",
]
