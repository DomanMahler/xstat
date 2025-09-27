"""
Data interfaces and adapters for xstat platform.
"""

from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field


class BarData(BaseModel):
    """Standardized bar data structure."""
    
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str = "1d"
    
    class Config:
        arbitrary_types_allowed = True


class MicrostructureData(BaseModel):
    """Order book microstructure data."""
    
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    mid_price: float
    spread: float
    
    class Config:
        arbitrary_types_allowed = True


class DataFeed(ABC):
    """Abstract base class for data feeds."""
    
    @abstractmethod
    async def get_symbols(
        self, 
        universe: str, 
        start: date, 
        end: date
    ) -> List[str]:
        """Get available symbols for a universe."""
        pass
    
    @abstractmethod
    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: date,
        end: date,
        **kwargs
    ) -> pd.DataFrame:
        """Get OHLCV bar data for a symbol."""
        pass
    
    @abstractmethod
    async def get_microstructure(
        self,
        symbol: str,
        start: date,
        end: date,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """Get order book microstructure data (optional)."""
        pass
    
    def validate_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean bar data."""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Validate OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if invalid_ohlc.any():
            print(f"Warning: {invalid_ohlc.sum()} invalid OHLC bars removed")
            df = df[~invalid_ohlc]
        
        # Ensure volume is non-negative
        df = df[df['volume'] >= 0]
        
        return df


class DataCache:
    """Simple file-based caching for data feeds."""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        import os
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, key: str) -> str:
        """Get cache file path for a key."""
        import hashlib
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return f"{self.cache_dir}/{hash_key}.parquet"
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached data."""
        import os
        cache_path = self.get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                return None
        return None
    
    def set(self, key: str, data: pd.DataFrame) -> None:
        """Cache data."""
        cache_path = self.get_cache_path(key)
        data.to_parquet(cache_path)
    
    def clear(self) -> None:
        """Clear all cached data."""
        import os
        import glob
        for file in glob.glob(f"{self.cache_dir}/*.parquet"):
            os.remove(file)


class DataManager:
    """Centralized data management with caching."""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache = DataCache(cache_dir)
        self.feeds: Dict[str, DataFeed] = {}
    
    def register_feed(self, name: str, feed: DataFeed) -> None:
        """Register a data feed."""
        self.feeds[name] = feed
    
    async def get_bars(
        self,
        feed_name: str,
        symbol: str,
        timeframe: str,
        start: date,
        end: date,
        use_cache: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """Get bars with optional caching."""
        if feed_name not in self.feeds:
            raise ValueError(f"Unknown feed: {feed_name}")
        
        # Create cache key
        cache_key = f"{feed_name}_{symbol}_{timeframe}_{start}_{end}"
        
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Fetch from feed
        feed = self.feeds[feed_name]
        data = await feed.get_bars(symbol, timeframe, start, end, **kwargs)
        
        # Validate and cache
        data = feed.validate_bars(data)
        if use_cache:
            self.cache.set(cache_key, data)
        
        return data
    
    async def get_multiple_bars(
        self,
        feed_name: str,
        symbols: List[str],
        timeframe: str,
        start: date,
        end: date,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Get bars for multiple symbols."""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = await self.get_bars(
                    feed_name, symbol, timeframe, start, end, **kwargs
                )
            except Exception as e:
                print(f"Failed to get data for {symbol}: {e}")
                continue
        return results
