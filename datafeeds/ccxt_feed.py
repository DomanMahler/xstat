"""
CCXT data feed for cryptocurrency data.
"""

import asyncio
from typing import List, Optional, Dict, Any
from datetime import date, datetime
import pandas as pd
import ccxt
from xstat.core.data import DataFeed, BarData, MicrostructureData


class CCXTFeed(DataFeed):
    """CCXT data feed for cryptocurrency data."""
    
    def __init__(self, exchange_name: str = "binance", cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.exchange_name = exchange_name
        
        # Initialize exchange
        try:
            exchange_class = getattr(ccxt, exchange_name)
            self.exchange = exchange_class({
                'apiKey': '',  # Add API key if needed
                'secret': '',  # Add secret if needed
                'sandbox': True,  # Use sandbox for testing
                'rateLimit': 1200,  # Rate limit in ms
                'enableRateLimit': True,
            })
        except Exception as e:
            print(f"Error initializing {exchange_name} exchange: {e}")
            self.exchange = None
        
        # Common crypto symbols
        self._crypto_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT',
            'SOL/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'SHIB/USDT',
            'MATIC/USDT', 'LTC/USDT', 'UNI/USDT', 'LINK/USDT', 'ATOM/USDT',
            'FTM/USDT', 'ALGO/USDT', 'VET/USDT', 'ICP/USDT', 'FIL/USDT',
            'TRX/USDT', 'ETC/USDT', 'XLM/USDT', 'BCH/USDT', 'EOS/USDT',
            'AAVE/USDT', 'SUSHI/USDT', 'COMP/USDT', 'YFI/USDT', 'SNX/USDT',
            'MKR/USDT', 'CRV/USDT', '1INCH/USDT', 'BAL/USDT', 'ZRX/USDT',
            'ENJ/USDT', 'MANA/USDT', 'SAND/USDT', 'AXS/USDT', 'GALA/USDT',
            'CHZ/USDT', 'FLOW/USDT', 'NEAR/USDT', 'FTT/USDT', 'SRM/USDT',
            'RAY/USDT', 'SERUM/USDT', 'STEP/USDT', 'ORCA/USDT', 'JUP/USDT'
        ]
    
    async def get_symbols(
        self, 
        universe: str, 
        start: date, 
        end: date
    ) -> List[str]:
        """Get available symbols for a universe."""
        if universe == "crypto_majors":
            return self._crypto_symbols[:20]  # Top 20
        elif universe == "crypto_top50":
            return self._crypto_symbols[:50]
        elif universe == "crypto_all":
            return self._crypto_symbols
        else:
            # For custom universes, return the universe name as a single symbol
            return [universe]
    
    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: date,
        end: date,
        **kwargs
    ) -> pd.DataFrame:
        """Get OHLCV bar data for a symbol."""
        if not self.exchange:
            return pd.DataFrame()
        
        try:
            # Convert timeframe to CCXT format
            ccxt_timeframe = self._convert_timeframe(timeframe)
            
            # Convert dates to timestamps
            start_ts = int(datetime.combine(start, datetime.min.time()).timestamp() * 1000)
            end_ts = int(datetime.combine(end, datetime.min.time()).timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = await self._fetch_ohlcv(symbol, ccxt_timeframe, start_ts, end_ts)
            
            if not ohlcv:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Validate and clean data
            data = self.validate_bars(df)
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_microstructure(
        self,
        symbol: str,
        start: date,
        end: date,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """Get order book microstructure data."""
        if not self.exchange:
            return None
        
        try:
            # Get order book
            orderbook = await self._fetch_orderbook(symbol)
            
            if not orderbook:
                return None
            
            # Extract top of book data
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return None
            
            # Create microstructure data
            microstructure_data = []
            timestamp = datetime.now()
            
            if bids and asks:
                bid_price, bid_size = bids[0]
                ask_price, ask_size = asks[0]
                mid_price = (bid_price + ask_price) / 2
                spread = ask_price - bid_price
                
                microstructure_data.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'bid_price': bid_price,
                    'ask_price': ask_price,
                    'bid_size': bid_size,
                    'ask_size': ask_size,
                    'mid_price': mid_price,
                    'spread': spread
                })
            
            df = pd.DataFrame(microstructure_data)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching microstructure data for {symbol}: {e}")
            return None
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert standard timeframe to CCXT format."""
        timeframe_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
            '1w': '1w',
            '1M': '1M'
        }
        return timeframe_map.get(timeframe, '1d')
    
    async def _fetch_ohlcv(self, symbol: str, timeframe: str, start_ts: int, end_ts: int) -> List:
        """Fetch OHLCV data from exchange."""
        try:
            # Use asyncio to run in thread pool
            loop = asyncio.get_event_loop()
            ohlcv = await loop.run_in_executor(
                None,
                lambda: self.exchange.fetch_ohlcv(symbol, timeframe, start_ts, limit=1000)
            )
            return ohlcv
        except Exception as e:
            print(f"Error fetching OHLCV for {symbol}: {e}")
            return []
    
    async def _fetch_orderbook(self, symbol: str) -> Dict:
        """Fetch order book from exchange."""
        try:
            loop = asyncio.get_event_loop()
            orderbook = await loop.run_in_executor(
                None,
                lambda: self.exchange.fetch_order_book(symbol)
            )
            return orderbook
        except Exception as e:
            print(f"Error fetching order book for {symbol}: {e}")
            return {}
    
    def get_supported_exchanges(self) -> List[str]:
        """Get list of supported exchanges."""
        return [
            'binance', 'coinbase', 'kraken', 'bitfinex', 'huobi', 'okx',
            'bybit', 'kucoin', 'gateio', 'bitget', 'mexc', 'cryptocom'
        ]
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information."""
        if not self.exchange:
            return {}
        
        return {
            'name': self.exchange.name,
            'countries': self.exchange.countries,
            'rateLimit': self.exchange.rateLimit,
            'has': self.exchange.has,
            'urls': self.exchange.urls,
            'version': self.exchange.version,
            'fees': self.exchange.fees
        }
    
    def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees for a symbol."""
        if not self.exchange:
            return {}
        
        try:
            markets = self.exchange.load_markets()
            if symbol in markets:
                market = markets[symbol]
                return {
                    'maker': market.get('maker', 0.001),
                    'taker': market.get('taker', 0.001),
                    'percentage': market.get('percentage', True)
                }
        except Exception as e:
            print(f"Error getting fees for {symbol}: {e}")
        
        return {}
    
    def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """Get market information for a symbol."""
        if not self.exchange:
            return {}
        
        try:
            markets = self.exchange.load_markets()
            if symbol in markets:
                market = markets[symbol]
                return {
                    'id': market.get('id'),
                    'symbol': market.get('symbol'),
                    'base': market.get('base'),
                    'quote': market.get('quote'),
                    'active': market.get('active'),
                    'precision': market.get('precision'),
                    'limits': market.get('limits'),
                    'info': market.get('info')
                }
        except Exception as e:
            print(f"Error getting market info for {symbol}: {e}")
        
        return {}
