"""
Utility functions for xstat platform.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


def align_dataframes(
    dataframes: Dict[str, pd.DataFrame], 
    method: str = "inner"
) -> Dict[str, pd.DataFrame]:
    """
    Align multiple DataFrames to common index.
    
    Args:
        dataframes: Dictionary of symbol -> DataFrame
        method: Alignment method ('inner', 'outer', 'left', 'right')
        
    Returns:
        Dictionary of aligned DataFrames
    """
    if not dataframes:
        return {}
    
    # Get common index
    indices = [df.index for df in dataframes.values()]
    common_index = indices[0]
    
    for idx in indices[1:]:
        if method == "inner":
            common_index = common_index.intersection(idx)
        elif method == "outer":
            common_index = common_index.union(idx)
        elif method == "left":
            common_index = common_index.intersection(idx)
        elif method == "right":
            common_index = idx.intersection(common_index)
    
    # Align all DataFrames
    aligned = {}
    for symbol, df in dataframes.items():
        aligned[symbol] = df.reindex(common_index)
    
    return aligned


def calculate_returns(prices: pd.Series, method: str = "simple") -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series
        method: Return calculation method ('simple', 'log')
        
    Returns:
        Returns series
    """
    if method == "simple":
        return prices.pct_change()
    elif method == "log":
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown return method: {method}")


def calculate_volatility(
    returns: pd.Series, 
    window: int = 252, 
    annualize: bool = True
) -> pd.Series:
    """
    Calculate rolling volatility.
    
    Args:
        returns: Returns series
        window: Rolling window size
        annualize: Whether to annualize volatility
        
    Returns:
        Volatility series
    """
    vol = returns.rolling(window=window).std()
    
    if annualize:
        vol = vol * np.sqrt(252)
    
    return vol


def calculate_correlation(
    series1: pd.Series, 
    series2: pd.Series, 
    window: Optional[int] = None
) -> Union[float, pd.Series]:
    """
    Calculate correlation between two series.
    
    Args:
        series1: First series
        series2: Second series
        window: Rolling window (None for full correlation)
        
    Returns:
        Correlation coefficient or rolling correlation
    """
    if window is None:
        return series1.corr(series2)
    else:
        return series1.rolling(window=window).corr(series2)


def calculate_beta(
    asset_returns: pd.Series, 
    market_returns: pd.Series,
    window: Optional[int] = None
) -> Union[float, pd.Series]:
    """
    Calculate beta coefficient.
    
    Args:
        asset_returns: Asset returns
        market_returns: Market returns
        window: Rolling window (None for full beta)
        
    Returns:
        Beta coefficient or rolling beta
    """
    if window is None:
        covariance = np.cov(asset_returns.dropna(), market_returns.dropna())[0, 1]
        market_variance = np.var(market_returns.dropna())
        return covariance / market_variance if market_variance > 0 else 0
    else:
        def _beta_calc(asset_chunk, market_chunk):
            if len(asset_chunk) < 2 or len(market_chunk) < 2:
                return np.nan
            covariance = np.cov(asset_chunk, market_chunk)[0, 1]
            market_variance = np.var(market_chunk)
            return covariance / market_variance if market_variance > 0 else 0
        
        return asset_returns.rolling(window=window).apply(
            lambda x: _beta_calc(x, market_returns.iloc[x.index])
        )


def detect_outliers(
    series: pd.Series, 
    method: str = "iqr", 
    threshold: float = 1.5
) -> pd.Series:
    """
    Detect outliers in a series.
    
    Args:
        series: Input series
        method: Detection method ('iqr', 'zscore', 'modified_zscore')
        threshold: Outlier threshold
        
    Returns:
        Boolean series indicating outliers
    """
    if method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    elif method == "zscore":
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    elif method == "modified_zscore":
        median = series.median()
        mad = np.median(np.abs(series - median))
        modified_z_scores = 0.6745 * (series - median) / mad
        return np.abs(modified_z_scores) > threshold
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def clean_data(
    df: pd.DataFrame, 
    method: str = "forward_fill"
) -> pd.DataFrame:
    """
    Clean data by handling missing values and outliers.
    
    Args:
        df: Input DataFrame
        method: Cleaning method ('forward_fill', 'backward_fill', 'interpolate', 'drop')
        
    Returns:
        Cleaned DataFrame
    """
    cleaned = df.copy()
    
    if method == "forward_fill":
        cleaned = cleaned.fillna(method='ffill')
    elif method == "backward_fill":
        cleaned = cleaned.fillna(method='bfill')
    elif method == "interpolate":
        cleaned = cleaned.interpolate()
    elif method == "drop":
        cleaned = cleaned.dropna()
    
    return cleaned


def resample_data(
    df: pd.DataFrame, 
    frequency: str, 
    method: str = "last"
) -> pd.DataFrame:
    """
    Resample data to different frequency.
    
    Args:
        df: Input DataFrame
        frequency: Target frequency ('1D', '1H', '1W', '1M')
        method: Resampling method ('last', 'first', 'mean', 'sum')
        
    Returns:
        Resampled DataFrame
    """
    if method == "last":
        return df.resample(frequency).last()
    elif method == "first":
        return df.resample(frequency).first()
    elif method == "mean":
        return df.resample(frequency).mean()
    elif method == "sum":
        return df.resample(frequency).sum()
    else:
        raise ValueError(f"Unknown resampling method: {method}")


def calculate_rolling_statistics(
    series: pd.Series, 
    window: int, 
    stat: str = "mean"
) -> pd.Series:
    """
    Calculate rolling statistics.
    
    Args:
        series: Input series
        window: Rolling window size
        stat: Statistic to calculate ('mean', 'std', 'min', 'max', 'median')
        
    Returns:
        Rolling statistics series
    """
    if stat == "mean":
        return series.rolling(window=window).mean()
    elif stat == "std":
        return series.rolling(window=window).std()
    elif stat == "min":
        return series.rolling(window=window).min()
    elif stat == "max":
        return series.rolling(window=window).max()
    elif stat == "median":
        return series.rolling(window=window).median()
    else:
        raise ValueError(f"Unknown statistic: {stat}")


def calculate_technical_indicators(
    df: pd.DataFrame, 
    window: int = 20
) -> pd.DataFrame:
    """
    Calculate basic technical indicators.
    
    Args:
        df: DataFrame with OHLCV data
        window: Window size for calculations
        
    Returns:
        DataFrame with technical indicators
    """
    result = df.copy()
    
    # Simple Moving Average
    result['SMA'] = df['close'].rolling(window=window).mean()
    
    # Exponential Moving Average
    result['EMA'] = df['close'].ewm(span=window).mean()
    
    # Bollinger Bands
    sma = result['SMA']
    std = df['close'].rolling(window=window).std()
    result['BB_Upper'] = sma + (std * 2)
    result['BB_Lower'] = sma - (std * 2)
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    result['RSI'] = 100 - (100 / (1 + rs))
    
    return result


def calculate_portfolio_metrics(
    returns: pd.Series, 
    benchmark_returns: Optional[pd.Series] = None
) -> Dict[str, float]:
    """
    Calculate portfolio performance metrics.
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns (optional)
        
    Returns:
        Dictionary of performance metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['total_return'] = (1 + returns).prod() - 1
    metrics['annual_return'] = returns.mean() * 252
    metrics['volatility'] = returns.std() * np.sqrt(252)
    metrics['sharpe_ratio'] = metrics['annual_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
    
    # Drawdown metrics
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min()
    
    # Benchmark comparison
    if benchmark_returns is not None:
        excess_returns = returns - benchmark_returns
        metrics['alpha'] = excess_returns.mean() * 252
        metrics['beta'] = returns.cov(benchmark_returns) / benchmark_returns.var()
        metrics['information_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    return metrics


def create_synthetic_data(
    n_periods: int = 1000,
    n_assets: int = 2,
    correlation: float = 0.5,
    volatility: float = 0.02,
    drift: float = 0.001
) -> pd.DataFrame:
    """
    Create synthetic correlated time series data.
    
    Args:
        n_periods: Number of time periods
        n_assets: Number of assets
        correlation: Correlation between assets
        volatility: Volatility of returns
        drift: Drift (expected return)
        
    Returns:
        DataFrame with synthetic price data
    """
    # Generate correlated random returns
    np.random.seed(42)  # For reproducibility
    
    # Create correlation matrix
    corr_matrix = np.full((n_assets, n_assets), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Generate random returns
    returns = np.random.multivariate_normal(
        mean=[drift] * n_assets,
        cov=corr_matrix * volatility**2,
        size=n_periods
    )
    
    # Convert to price series
    prices = np.exp(np.cumsum(returns, axis=0))
    
    # Create DataFrame
    columns = [f'Asset_{i+1}' for i in range(n_assets)]
    df = pd.DataFrame(prices, columns=columns)
    df.index = pd.date_range(start='2020-01-01', periods=n_periods, freq='D')
    
    return df


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality and return quality metrics.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with quality metrics
    """
    quality = {}
    
    # Missing data
    quality['missing_data_pct'] = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    
    # Duplicate timestamps
    quality['duplicate_timestamps'] = df.index.duplicated().sum()
    
    # Data gaps
    if hasattr(df.index, 'freq') and df.index.freq:
        expected_periods = pd.date_range(
            start=df.index.min(), 
            end=df.index.max(), 
            freq=df.index.freq
        )
        quality['data_gaps'] = len(expected_periods) - len(df)
    else:
        quality['data_gaps'] = 0
    
    # Outliers (using IQR method)
    outlier_counts = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
        outlier_counts[col] = outliers
    
    quality['outliers'] = outlier_counts
    quality['total_outliers'] = sum(outlier_counts.values())
    
    return quality
