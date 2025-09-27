"""
YFinance data feed for equities and FX data.
"""

import asyncio
from typing import List, Optional, Dict, Any
from datetime import date, datetime
import pandas as pd
import yfinance as yf
from xstat.core.data import DataFeed, BarData


class YFinanceFeed(DataFeed):
    """YFinance data feed for equities and FX."""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self._universe_symbols = {
            'sp100': self._get_sp100_symbols(),
            'sp500': self._get_sp500_symbols(),
            'nasdaq100': self._get_nasdaq100_symbols(),
            'dow': self._get_dow_symbols(),
            'fx_majors': self._get_fx_majors(),
        }
    
    async def get_symbols(
        self, 
        universe: str, 
        start: date, 
        end: date
    ) -> List[str]:
        """Get available symbols for a universe."""
        if universe in self._universe_symbols:
            return self._universe_symbols[universe]
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
        try:
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Download data
            data = ticker.history(
                start=start,
                end=end,
                interval=timeframe,
                auto_adjust=True,
                prepost=False,
                threads=True
            )
            
            if data.empty:
                return pd.DataFrame()
            
            # Standardize column names
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Add symbol column
            data['symbol'] = symbol
            
            # Validate and clean data
            data = self.validate_bars(data)
            
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
        """YFinance doesn't provide microstructure data."""
        return None
    
    def _get_sp100_symbols(self) -> List[str]:
        """Get S&P 100 symbols."""
        # Top 100 S&P 500 stocks by market cap (simplified list)
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
            'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE',
            'NFLX', 'CRM', 'INTC', 'CMCSA', 'PFE', 'ABT', 'TMO', 'ACN', 'CSCO',
            'AVGO', 'PEP', 'COST', 'TXN', 'QCOM', 'DHR', 'VZ', 'ADP', 'NKE',
            'T', 'LIN', 'ABBV', 'ORCL', 'MRK', 'UNP', 'PM', 'NEE', 'RTX',
            'HON', 'LOW', 'UPS', 'SPGI', 'INTU', 'IBM', 'AMD', 'CAT', 'GE',
            'AMT', 'BKNG', 'CVS', 'DE', 'EL', 'FIS', 'ISRG', 'LMT', 'MDT',
            'MMM', 'NOW', 'PGR', 'SYK', 'TGT', 'USB', 'WBA', 'ZTS', 'AON',
            'AXP', 'BA', 'BLK', 'CB', 'CL', 'COP', 'CVX', 'D', 'EMR', 'EXC',
            'FDX', 'GD', 'GM', 'GS', 'IBM', 'JCI', 'KMB', 'KO', 'LLY', 'MCD',
            'MO', 'NOC', 'PFE', 'PLD', 'REGN', 'SO', 'SPG', 'TMO', 'TXN', 'WMT'
        ]
    
    def _get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 symbols (simplified)."""
        # This would typically be loaded from a file or API
        # For now, return a subset
        return self._get_sp100_symbols() + [
            'A', 'AA', 'AAL', 'AAP', 'ABBV', 'ABC', 'ABMD', 'ABT', 'ACN',
            'ADBE', 'ADI', 'ADM', 'ADP', 'ADS', 'ADSK', 'AEE', 'AEP', 'AES',
            'AFL', 'A', 'AIG', 'AIV', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN',
            'ALK', 'ALL', 'ALLE', 'ALXN', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMG',
            'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'ANTM', 'AON', 'AOS',
            'APA', 'APD', 'APH', 'APTV', 'ARE', 'ATO', 'ATVI', 'AVB', 'AVGO',
            'AVY', 'AWK', 'AXP', 'AZO', 'BA', 'BAC', 'BAX', 'BBY', 'BDX',
            'BEN', 'BF-B', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLK', 'BLL',
            'BMY', 'BR', 'BRK-B', 'BSX', 'BWA', 'BXP', 'C', 'CAG', 'CAH',
            'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CE',
            'CERN', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL',
            'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP',
            'COF', 'COG', 'COO', 'COP', 'COST', 'COTY', 'CPB', 'CPRT', 'CRM',
            'CSCO', 'CSX', 'CTAS', 'CTLT', 'CTSH', 'CTVA', 'CTXS', 'CVS',
            'CVX', 'CXO', 'D', 'DAL', 'DD', 'DE', 'DFS', 'DG', 'DGX', 'DHI',
            'DHR', 'DIS', 'DISCA', 'DISCK', 'DISH', 'DLR', 'DLTR', 'DOV',
            'DOW', 'DPZ', 'DRE', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM',
            'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMN', 'EMR',
            'ENPH', 'EOG', 'EQIX', 'EQR', 'ES', 'ESS', 'ETN', 'ETR', 'EVRG',
            'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FB',
            'FBHS', 'FCX', 'FDX', 'FE', 'FFIV', 'FIS', 'FISV', 'FITB', 'FLT',
            'FMC', 'FOX', 'FOXA', 'FRC', 'FRT', 'FTI', 'FTNT', 'FTV', 'GD',
            'GE', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GOOG', 'GOOGL', 'GPC',
            'GPN', 'GPS', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HBI',
            'HCA', 'HD', 'HES', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HRL',
            'HSIC', 'HST', 'HSY', 'HUM', 'HWM', 'IBM', 'ICE', 'IDXX', 'IEX',
            'IFF', 'ILMN', 'INCY', 'INFO', 'INTC', 'INTU', 'ISRG', 'IT', 'ITW',
            'IVZ', 'J', 'JBHT', 'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'JWN',
            'K', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI',
            'KMX', 'KO', 'KR', 'KSU', 'L', 'LB', 'LDOS', 'LEG', 'LEN', 'LH',
            'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNC', 'LNT', 'LOW', 'LRCX',
            'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'M', 'MA', 'MAA', 'MAC', 'MAR',
            'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'MGM',
            'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH',
            'MOS', 'MPC', 'MRK', 'MRNA', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI',
            'MTB', 'MTD', 'MU', 'MXIM', 'MYL', 'NCLH', 'NDAQ', 'NEE', 'NEM',
            'NFLX', 'NKE', 'NKTR', 'NLSN', 'NOC', 'NOV', 'NOW', 'NRG', 'NSC',
            'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWL', 'NWS', 'NWSA', 'NXT',
            'O', 'ODFL', 'OKE', 'OMC', 'ORCL', 'ORLY', 'OXY', 'PAYC', 'PAYX',
            'PBCT', 'PCAR', 'PEAK', 'PEG', 'PENN', 'PEP', 'PFE', 'PFG', 'PG',
            'PGR', 'PH', 'PHM', 'PKG', 'PKI', 'PLD', 'PM', 'PNC', 'PNR',
            'PNW', 'PPG', 'PPL', 'PRGO', 'PRU', 'PSA', 'PSX', 'PTC', 'PWR',
            'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'RE', 'REG', 'REGN', 'RF',
            'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG',
            'RTX', 'SBAC', 'SBUX', 'SCHW', 'SEE', 'SHW', 'SIVB', 'SJM', 'SLB',
            'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STT', 'STX',
            'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY',
            'TECH', 'TEL', 'TER', 'TFC', 'TGT', 'TJX', 'TMO', 'TMUS', 'TPG',
            'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 'TWTR', 'TXN',
            'TXT', 'TYL', 'UA', 'UAA', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH',
            'UNM', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VFC', 'VIAC', 'VLO',
            'VMC', 'VRSK', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT',
            'WBA', 'WEC', 'WELL', 'WFC', 'WHR', 'WLTW', 'WM', 'WMB', 'WMT',
            'WRB', 'WST', 'WU', 'WY', 'WYNN', 'XEL', 'XLNX', 'XOM', 'XRAY',
            'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZEN', 'ZION', 'ZTS'
        ]
    
    def _get_nasdaq100_symbols(self) -> List[str]:
        """Get NASDAQ 100 symbols."""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA',
            'AVGO', 'PEP', 'COST', 'TMUS', 'CMCSA', 'TXN', 'QCOM', 'CHTR',
            'ADBE', 'NFLX', 'INTC', 'AMD', 'PYPL', 'CSCO', 'AMAT', 'GILD',
            'INTU', 'ISRG', 'ADP', 'BKNG', 'REGN', 'VRTX', 'FISV', 'MDLZ',
            'ATVI', 'ILMN', 'WBA', 'ADSK', 'CSX', 'MRNA', 'LRCX', 'SNPS',
            'CTAS', 'KLAC', 'ORLY', 'DXCM', 'EXC', 'PAYX', 'BIIB', 'IDXX',
            'MCHP', 'ALGN', 'CTSH', 'FAST', 'WDAY', 'ROST', 'XEL', 'PCAR',
            'INCY', 'VRSK', 'SGEN', 'SIRI', 'NTES', 'VRSN', 'CHKP', 'SWKS',
            'MXIM', 'LULU', 'CTXS', 'WLTW', 'CSGP', 'ANSS', 'MELI', 'NTAP',
            'CDNS', 'AMGN', 'CTSH', 'BIDU', 'JD', 'BABA', 'NFLX', 'TSLA'
        ]
    
    def _get_dow_symbols(self) -> List[str]:
        """Get Dow Jones Industrial Average symbols."""
        return [
            'AAPL', 'MSFT', 'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'MA',
            'DIS', 'PYPL', 'ADBE', 'NFLX', 'CRM', 'INTC', 'CMCSA', 'PFE',
            'ABT', 'TMO', 'ACN', 'CSCO', 'AVGO', 'PEP', 'COST', 'TXN',
            'QCOM', 'DHR', 'VZ', 'ADP', 'NKE', 'T', 'LIN', 'ABBV', 'ORCL',
            'MRK', 'UNP', 'PM', 'NEE', 'RTX', 'HON', 'LOW', 'UPS', 'SPGI',
            'INTU', 'IBM', 'AMD', 'CAT', 'GE', 'AMT', 'BKNG', 'CVS', 'DE',
            'EL', 'FIS', 'ISRG', 'LMT', 'MDT', 'MMM', 'NOW', 'PGR', 'SYK',
            'TGT', 'USB', 'WBA', 'ZTS'
        ]
    
    def _get_fx_majors(self) -> List[str]:
        """Get major FX pairs."""
        return [
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X',
            'USDCAD=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X',
            'AUDJPY=X', 'EURCHF=X', 'GBPCHF=X', 'AUDCHF=X', 'CADCHF=X',
            'NZDCHF=X', 'EURCAD=X', 'GBPCAD=X', 'AUDCAD=X', 'NZDCAD=X',
            'EURAUD=X', 'GBPAUD=X', 'AUDNZD=X', 'EURNZD=X', 'GBPNZD=X'
        ]
