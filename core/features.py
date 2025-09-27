"""
Feature engineering for statistical arbitrage.
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class FeatureEngineer:
    """Feature engineering for statistical arbitrage strategies."""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def rolling_zscore(
        self, 
        series: pd.Series, 
        window: int = 20
    ) -> pd.Series:
        """Calculate rolling z-score."""
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        return (series - rolling_mean) / rolling_std
    
    def rolling_beta(
        self, 
        y: pd.Series, 
        x: pd.Series, 
        window: int = 60
    ) -> pd.Series:
        """Calculate rolling beta coefficient."""
        def _beta_calc(y_chunk, x_chunk):
            if len(y_chunk) < 2 or y_chunk.std() == 0 or x_chunk.std() == 0:
                return np.nan
            return np.cov(y_chunk, x_chunk)[0, 1] / np.var(x_chunk)
        
        return y.rolling(window=window).apply(
            lambda y_chunk: _beta_calc(y_chunk, x.iloc[y_chunk.index])
        )
    
    def kalman_filter_beta(
        self, 
        y: pd.Series, 
        x: pd.Series,
        initial_beta: float = 1.0,
        initial_var: float = 1.0,
        process_var: float = 0.01,
        obs_var: float = 0.1
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Kalman filter for time-varying beta estimation.
        
        Args:
            y: Dependent variable
            x: Independent variable
            initial_beta: Initial beta estimate
            initial_var: Initial variance estimate
            process_var: Process noise variance
            obs_var: Observation noise variance
            
        Returns:
            Tuple of (beta_estimates, beta_variances)
        """
        n = len(y)
        beta_estimates = np.zeros(n)
        beta_vars = np.zeros(n)
        
        # Initialize
        beta_estimates[0] = initial_beta
        beta_vars[0] = initial_var
        
        for t in range(1, n):
            # Prediction step
            beta_pred = beta_estimates[t-1]
            var_pred = beta_vars[t-1] + process_var
            
            # Update step
            if not np.isnan(x.iloc[t]) and not np.isnan(y.iloc[t]):
                # Kalman gain
                K = var_pred / (var_pred + obs_var)
                
                # Update estimates
                beta_estimates[t] = beta_pred + K * (y.iloc[t] - beta_pred * x.iloc[t])
                beta_vars[t] = (1 - K) * var_pred
            else:
                beta_estimates[t] = beta_pred
                beta_vars[t] = var_pred
        
        return pd.Series(beta_estimates, index=y.index), pd.Series(beta_vars, index=y.index)
    
    def bollinger_bands(
        self, 
        series: pd.Series, 
        window: int = 20, 
        num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        return upper_band, rolling_mean, lower_band
    
    def rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def macd(
        self, 
        series: pd.Series, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def volatility_features(
        self, 
        returns: pd.Series, 
        window: int = 20
    ) -> Dict[str, pd.Series]:
        """Calculate various volatility features."""
        features = {}
        
        # Rolling volatility
        features['volatility'] = returns.rolling(window=window).std()
        
        # GARCH-like volatility (EWMA)
        features['ewma_vol'] = returns.ewm(span=window).std()
        
        # Parkinson volatility (using high-low)
        # Note: This requires OHLC data, simplified here
        features['parkinson_vol'] = returns.rolling(window=window).std() * np.sqrt(2)
        
        # Volatility of volatility
        vol = features['volatility']
        features['vol_of_vol'] = vol.rolling(window=window).std()
        
        return features
    
    def regime_features(
        self, 
        spread: pd.Series, 
        volatility: pd.Series,
        window: int = 20
    ) -> Dict[str, pd.Series]:
        """Calculate regime detection features."""
        features = {}
        
        # Spread momentum
        features['spread_momentum'] = spread.rolling(window=window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        
        # Volatility regime
        vol_percentile = volatility.rolling(window=window*2).rank(pct=True)
        features['high_vol_regime'] = (vol_percentile > 0.8).astype(int)
        features['low_vol_regime'] = (vol_percentile < 0.2).astype(int)
        
        # Spread regime (trending vs mean-reverting)
        spread_abs = spread.abs()
        features['high_spread_regime'] = (spread_abs > spread_abs.rolling(window=window).quantile(0.8)).astype(int)
        
        return features
    
    def correlation_features(
        self, 
        y: pd.Series, 
        x: pd.Series, 
        window: int = 60
    ) -> Dict[str, pd.Series]:
        """Calculate correlation-based features."""
        features = {}
        
        # Rolling correlation
        features['correlation'] = y.rolling(window=window).corr(x)
        
        # Correlation stability (rolling std of correlation)
        corr = features['correlation']
        features['corr_stability'] = corr.rolling(window=window//2).std()
        
        # Correlation regime changes
        corr_diff = corr.diff()
        features['corr_regime_change'] = (corr_diff.abs() > corr_diff.rolling(window=window).std()).astype(int)
        
        return features
    
    def technical_indicators(
        self, 
        series: pd.Series
    ) -> Dict[str, pd.Series]:
        """Calculate comprehensive technical indicators."""
        indicators = {}
        
        # Price-based indicators
        indicators['sma_20'] = series.rolling(window=20).mean()
        indicators['sma_50'] = series.rolling(window=50).mean()
        indicators['ema_12'] = series.ewm(span=12).mean()
        indicators['ema_26'] = series.ewm(span=26).mean()
        
        # Momentum indicators
        indicators['rsi'] = self.rsi(series)
        macd, signal, hist = self.macd(series)
        indicators['macd'] = macd
        indicators['macd_signal'] = signal
        indicators['macd_histogram'] = hist
        
        # Volatility indicators
        bb_upper, bb_middle, bb_lower = self.bollinger_bands(series)
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        indicators['bb_width'] = (bb_upper - bb_lower) / bb_middle
        indicators['bb_position'] = (series - bb_lower) / (bb_upper - bb_lower)
        
        return indicators
    
    def create_spread_features(
        self, 
        y: pd.Series, 
        x: pd.Series, 
        hedge_ratio: float = 1.0
    ) -> Dict[str, pd.Series]:
        """Create comprehensive features for spread analysis."""
        features = {}
        
        # Basic spread
        spread = y - hedge_ratio * x
        features['spread'] = spread
        
        # Spread statistics
        features['spread_zscore'] = self.rolling_zscore(spread)
        features['spread_returns'] = spread.pct_change()
        
        # Volatility features
        vol_features = self.volatility_features(features['spread_returns'])
        features.update(vol_features)
        
        # Technical indicators on spread
        spread_indicators = self.technical_indicators(spread)
        features.update(spread_indicators)
        
        # Regime features
        regime_features = self.regime_features(spread, vol_features['volatility'])
        features.update(regime_features)
        
        # Correlation features
        corr_features = self.correlation_features(y, x)
        features.update(corr_features)
        
        return features
    
    def create_pair_features(
        self, 
        y: pd.Series, 
        x: pd.Series,
        hedge_ratio: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create comprehensive features for a pair."""
        if hedge_ratio is None:
            # Estimate hedge ratio using OLS
            from statsmodels.tools import add_constant
            from statsmodels.regression.linear_model import OLS
            
            X = add_constant(x)
            model = OLS(y, X).fit()
            hedge_ratio = model.params[1]
        
        # Create spread features
        features = self.create_spread_features(y, x, hedge_ratio)
        
        # Add pair-specific features
        features['hedge_ratio'] = pd.Series([hedge_ratio] * len(y), index=y.index)
        features['y_price'] = y
        features['x_price'] = x
        features['y_returns'] = y.pct_change()
        features['x_returns'] = x.pct_change()
        
        return features
