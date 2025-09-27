"""
Parquet data feed for local cached data.
"""

import os
from typing import List, Optional, Dict, Any
from datetime import date, datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from xstat.core.data import DataFeed


class ParquetFeed(DataFeed):
    """Parquet data feed for local cached data."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    async def get_symbols(
        self, 
        universe: str, 
        start: date, 
        end: date
    ) -> List[str]:
        """Get available symbols from parquet files."""
        symbols = []
        
        # Scan data directory for parquet files
        for file in os.listdir(self.data_dir):
            if file.endswith('.parquet'):
                symbol = file.replace('.parquet', '')
                symbols.append(symbol)
        
        return symbols
    
    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: date,
        end: date,
        **kwargs
    ) -> pd.DataFrame:
        """Get OHLCV bar data from parquet file."""
        try:
            file_path = os.path.join(self.data_dir, f"{symbol}.parquet")
            
            if not os.path.exists(file_path):
                return pd.DataFrame()
            
            # Read parquet file
            df = pd.read_parquet(file_path)
            
            # Filter by date range
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            
            # Filter by date range
            start_dt = datetime.combine(start, datetime.min.time())
            end_dt = datetime.combine(end, datetime.min.time())
            
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            
            # Validate and clean data
            data = self.validate_bars(df)
            
            return data
            
        except Exception as e:
            print(f"Error reading parquet file for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_microstructure(
        self,
        symbol: str,
        start: date,
        end: date,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """Get microstructure data from parquet file."""
        try:
            file_path = os.path.join(self.data_dir, f"{symbol}_microstructure.parquet")
            
            if not os.path.exists(file_path):
                return None
            
            # Read parquet file
            df = pd.read_parquet(file_path)
            
            # Filter by date range
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            
            # Filter by date range
            start_dt = datetime.combine(start, datetime.min.time())
            end_dt = datetime.combine(end, datetime.min.time())
            
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            
            return df
            
        except Exception as e:
            print(f"Error reading microstructure data for {symbol}: {e}")
            return None
    
    def save_bars(self, symbol: str, data: pd.DataFrame) -> None:
        """Save bar data to parquet file."""
        try:
            file_path = os.path.join(self.data_dir, f"{symbol}.parquet")
            data.to_parquet(file_path)
            print(f"Saved {len(data)} bars for {symbol} to {file_path}")
        except Exception as e:
            print(f"Error saving data for {symbol}: {e}")
    
    def save_microstructure(self, symbol: str, data: pd.DataFrame) -> None:
        """Save microstructure data to parquet file."""
        try:
            file_path = os.path.join(self.data_dir, f"{symbol}_microstructure.parquet")
            data.to_parquet(file_path)
            print(f"Saved {len(data)} microstructure records for {symbol} to {file_path}")
        except Exception as e:
            print(f"Error saving microstructure data for {symbol}: {e}")
    
    def get_file_info(self, symbol: str) -> Dict[str, Any]:
        """Get information about a parquet file."""
        file_path = os.path.join(self.data_dir, f"{symbol}.parquet")
        
        if not os.path.exists(file_path):
            return {}
        
        try:
            # Get file stats
            stat = os.stat(file_path)
            
            # Read parquet metadata
            parquet_file = pq.ParquetFile(file_path)
            metadata = parquet_file.metadata
            
            return {
                'file_path': file_path,
                'file_size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'num_rows': metadata.num_rows,
                'num_columns': len(metadata.schema),
                'schema': [field.name for field in metadata.schema]
            }
        except Exception as e:
            print(f"Error getting file info for {symbol}: {e}")
            return {}
    
    def list_available_data(self) -> Dict[str, Any]:
        """List all available data files."""
        data_info = {}
        
        for file in os.listdir(self.data_dir):
            if file.endswith('.parquet'):
                symbol = file.replace('.parquet', '')
                if symbol.endswith('_microstructure'):
                    symbol = symbol.replace('_microstructure', '')
                    data_type = 'microstructure'
                else:
                    data_type = 'bars'
                
                if symbol not in data_info:
                    data_info[symbol] = {}
                
                data_info[symbol][data_type] = self.get_file_info(symbol)
        
        return data_info
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        """Clean up old data files."""
        cutoff_date = datetime.now() - pd.Timedelta(days=days_to_keep)
        
        for file in os.listdir(self.data_dir):
            if file.endswith('.parquet'):
                file_path = os.path.join(self.data_dir, file)
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_time < cutoff_date:
                    try:
                        os.remove(file_path)
                        print(f"Removed old file: {file}")
                    except Exception as e:
                        print(f"Error removing file {file}: {e}")
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of available data."""
        summary = {
            'total_files': 0,
            'total_size': 0,
            'symbols': set(),
            'date_ranges': {}
        }
        
        for file in os.listdir(self.data_dir):
            if file.endswith('.parquet'):
                file_path = os.path.join(self.data_dir, file)
                symbol = file.replace('.parquet', '')
                
                if symbol.endswith('_microstructure'):
                    symbol = symbol.replace('_microstructure', '')
                
                summary['total_files'] += 1
                summary['total_size'] += os.path.getsize(file_path)
                summary['symbols'].add(symbol)
                
                # Get date range for this symbol
                try:
                    df = pd.read_parquet(file_path)
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        summary['date_ranges'][symbol] = {
                            'start': df['timestamp'].min(),
                            'end': df['timestamp'].max(),
                            'count': len(df)
                        }
                except Exception:
                    continue
        
        summary['symbols'] = list(summary['symbols'])
        return summary
