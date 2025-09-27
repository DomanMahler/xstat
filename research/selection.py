"""
Pair selection and ranking for statistical arbitrage.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from xstat.core.stat_tests import CointegrationTests
from xstat.core.features import FeatureEngineer
from xstat.core.utils import calculate_correlation, calculate_volatility


@dataclass
class PairCandidate:
    """Candidate pair for statistical arbitrage."""
    symbol_y: str
    symbol_x: str
    hedge_ratio: float
    correlation: float
    cointegration_score: float
    half_life: float
    spread_volatility: float
    stability_score: float
    tradeability_score: float
    risk_score: float
    overall_score: float
    test_results: Dict[str, Any]


class PairSelector:
    """Select and rank pairs for statistical arbitrage."""
    
    def __init__(
        self,
        min_correlation: float = 0.5,
        max_half_life: float = 30.0,
        min_cointegration_score: float = 0.7,
        max_spread_volatility: float = 0.05
    ):
        self.min_correlation = min_correlation
        self.max_half_life = max_half_life
        self.min_cointegration_score = min_cointegration_score
        self.max_spread_volatility = max_spread_volatility
        
        self.cointegration_tests = CointegrationTests()
        self.feature_engineer = FeatureEngineer()
    
    def scan_pairs(
        self,
        price_data: Dict[str, pd.DataFrame],
        universe: List[str],
        min_lookback: int = 252
    ) -> List[PairCandidate]:
        """
        Scan universe for potential pairs.
        
        Args:
            price_data: Dictionary of symbol -> price DataFrame
            universe: List of symbols to scan
            min_lookback: Minimum lookback period for analysis
            
        Returns:
            List of PairCandidate objects
        """
        candidates = []
        
        # Generate all possible pairs
        pairs = self._generate_pairs(universe)
        
        for symbol_y, symbol_x in pairs:
            try:
                # Get price data for both symbols
                y_data = price_data.get(symbol_y)
                x_data = price_data.get(symbol_x)
                
                if y_data is None or x_data is None:
                    continue
                
                # Align data
                aligned_data = self._align_data(y_data, x_data)
                if len(aligned_data) < min_lookback:
                    continue
                
                # Extract price series
                y_prices = aligned_data[symbol_y]['close']
                x_prices = aligned_data[symbol_x]['close']
                
                # Calculate returns
                y_returns = y_prices.pct_change().dropna()
                x_returns = x_prices.pct_change().dropna()
                
                # Run cointegration tests
                test_results = self.cointegration_tests.comprehensive_test(y_prices, x_prices)
                
                # Calculate features
                features = self._calculate_pair_features(y_prices, x_prices, test_results)
                
                # Create candidate
                candidate = PairCandidate(
                    symbol_y=symbol_y,
                    symbol_x=symbol_x,
                    hedge_ratio=features['hedge_ratio'],
                    correlation=features['correlation'],
                    cointegration_score=features['cointegration_score'],
                    half_life=features['half_life'],
                    spread_volatility=features['spread_volatility'],
                    stability_score=features['stability_score'],
                    tradeability_score=features['tradeability_score'],
                    risk_score=features['risk_score'],
                    overall_score=features['overall_score'],
                    test_results=test_results
                )
                
                # Filter candidates
                if self._is_valid_candidate(candidate):
                    candidates.append(candidate)
                
            except Exception as e:
                print(f"Error analyzing pair {symbol_y}-{symbol_x}: {e}")
                continue
        
        # Sort by overall score
        candidates.sort(key=lambda x: x.overall_score, reverse=True)
        
        return candidates
    
    def _generate_pairs(self, universe: List[str]) -> List[Tuple[str, str]]:
        """Generate all possible pairs from universe."""
        pairs = []
        for i, symbol_y in enumerate(universe):
            for j, symbol_x in enumerate(universe[i+1:], i+1):
                pairs.append((symbol_y, symbol_x))
        return pairs
    
    def _align_data(
        self, 
        y_data: pd.DataFrame, 
        x_data: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Align two price DataFrames."""
        # Find common index
        common_index = y_data.index.intersection(x_data.index)
        
        if len(common_index) == 0:
            return {}
        
        return {
            'y': y_data.loc[common_index],
            'x': x_data.loc[common_index]
        }
    
    def _calculate_pair_features(
        self,
        y_prices: pd.Series,
        x_prices: pd.Series,
        test_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate features for a pair."""
        features = {}
        
        # Basic statistics
        features['correlation'] = y_prices.corr(x_prices)
        features['hedge_ratio'] = test_results.get('engle_granger', {}).get('hedge_ratio', 1.0)
        features['half_life'] = test_results.get('half_life', np.nan)
        
        # Cointegration score
        eg_result = test_results.get('engle_granger', {})
        johansen_result = test_results.get('johansen', {})
        
        cointegration_score = 0.0
        if eg_result.get('is_cointegrated', False):
            cointegration_score += 0.5
        if johansen_result.get('n_cointegrating_relations', 0) > 0:
            cointegration_score += 0.5
        
        features['cointegration_score'] = cointegration_score
        
        # Spread analysis
        spread = y_prices - features['hedge_ratio'] * x_prices
        features['spread_volatility'] = spread.std()
        features['spread_mean'] = spread.mean()
        
        # Stability score (persistence of hedge ratio)
        features['stability_score'] = self._calculate_stability_score(y_prices, x_prices)
        
        # Tradeability score
        features['tradeability_score'] = self._calculate_tradeability_score(
            spread, features['spread_volatility']
        )
        
        # Risk score
        features['risk_score'] = self._calculate_risk_score(
            y_prices, x_prices, spread
        )
        
        # Overall score
        features['overall_score'] = self._calculate_overall_score(features)
        
        return features
    
    def _calculate_stability_score(
        self, 
        y_prices: pd.Series, 
        x_prices: pd.Series,
        window: int = 60
    ) -> float:
        """Calculate stability of hedge ratio."""
        if len(y_prices) < window:
            return 0.0
        
        # Calculate rolling hedge ratios
        rolling_betas = []
        for i in range(window, len(y_prices)):
            y_chunk = y_prices.iloc[i-window:i]
            x_chunk = x_prices.iloc[i-window:i]
            
            if len(y_chunk) > 1 and len(x_chunk) > 1:
                beta = np.cov(y_chunk, x_chunk)[0, 1] / np.var(x_chunk)
                rolling_betas.append(beta)
        
        if len(rolling_betas) < 2:
            return 0.0
        
        # Calculate stability as inverse of variance
        beta_std = np.std(rolling_betas)
        stability = 1.0 / (1.0 + beta_std)  # Normalize to [0, 1]
        
        return stability
    
    def _calculate_tradeability_score(
        self, 
        spread: pd.Series, 
        spread_volatility: float
    ) -> float:
        """Calculate tradeability score."""
        # Higher volatility = more trading opportunities
        # But not too high (risk)
        optimal_vol = 0.02  # 2% daily volatility
        
        if spread_volatility <= 0:
            return 0.0
        
        # Score based on distance from optimal volatility
        vol_score = 1.0 - abs(np.log(spread_volatility / optimal_vol))
        vol_score = max(0.0, min(1.0, vol_score))
        
        # Add frequency of mean reversion
        spread_returns = spread.pct_change().dropna()
        mean_reversion_freq = (spread_returns * spread_returns.shift(1) < 0).mean()
        
        return (vol_score + mean_reversion_freq) / 2
    
    def _calculate_risk_score(
        self, 
        y_prices: pd.Series, 
        x_prices: pd.Series, 
        spread: pd.Series
    ) -> float:
        """Calculate risk score (lower is better)."""
        risk_factors = []
        
        # Price volatility
        y_vol = y_prices.pct_change().std()
        x_vol = x_prices.pct_change().std()
        vol_risk = (y_vol + x_vol) / 2
        risk_factors.append(vol_risk)
        
        # Spread drift (trend)
        spread_trend = abs(spread.diff().mean())
        risk_factors.append(spread_trend)
        
        # Correlation stability
        returns_y = y_prices.pct_change().dropna()
        returns_x = x_prices.pct_change().dropna()
        rolling_corr = returns_y.rolling(60).corr(returns_x)
        corr_stability = 1.0 - rolling_corr.std()
        risk_factors.append(corr_stability)
        
        # Combine risk factors
        overall_risk = np.mean(risk_factors)
        
        # Convert to score (lower risk = higher score)
        risk_score = 1.0 / (1.0 + overall_risk)
        
        return risk_score
    
    def _calculate_overall_score(self, features: Dict[str, float]) -> float:
        """Calculate overall score for ranking."""
        weights = {
            'cointegration_score': 0.3,
            'stability_score': 0.25,
            'tradeability_score': 0.25,
            'risk_score': 0.2
        }
        
        overall_score = 0.0
        for feature, weight in weights.items():
            if feature in features:
                overall_score += features[feature] * weight
        
        return overall_score
    
    def _is_valid_candidate(self, candidate: PairCandidate) -> bool:
        """Check if candidate meets minimum requirements."""
        return (
            candidate.correlation >= self.min_correlation and
            candidate.half_life <= self.max_half_life and
            candidate.cointegration_score >= self.min_cointegration_score and
            candidate.spread_volatility <= self.max_spread_volatility and
            not np.isnan(candidate.half_life) and
            candidate.half_life > 0
        )
    
    def rank_pairs(
        self, 
        candidates: List[PairCandidate],
        top_n: int = 50
    ) -> List[PairCandidate]:
        """Rank and filter top pairs."""
        # Sort by overall score
        ranked = sorted(candidates, key=lambda x: x.overall_score, reverse=True)
        
        # Return top N
        return ranked[:top_n]
    
    def create_pair_summary(self, candidates: List[PairCandidate]) -> pd.DataFrame:
        """Create summary DataFrame of candidates."""
        if not candidates:
            return pd.DataFrame()
        
        data = []
        for candidate in candidates:
            data.append({
                'symbol_y': candidate.symbol_y,
                'symbol_x': candidate.symbol_x,
                'hedge_ratio': candidate.hedge_ratio,
                'correlation': candidate.correlation,
                'cointegration_score': candidate.cointegration_score,
                'half_life': candidate.half_life,
                'spread_volatility': candidate.spread_volatility,
                'stability_score': candidate.stability_score,
                'tradeability_score': candidate.tradeability_score,
                'risk_score': candidate.risk_score,
                'overall_score': candidate.overall_score
            })
        
        return pd.DataFrame(data)
    
    def filter_by_sector(
        self, 
        candidates: List[PairCandidate],
        sector_mapping: Dict[str, str]
    ) -> List[PairCandidate]:
        """Filter pairs to only include same-sector pairs."""
        filtered = []
        
        for candidate in candidates:
            y_sector = sector_mapping.get(candidate.symbol_y)
            x_sector = sector_mapping.get(candidate.symbol_x)
            
            if y_sector and x_sector and y_sector == x_sector:
                filtered.append(candidate)
        
        return filtered
    
    def calculate_pair_capacity(
        self, 
        candidate: PairCandidate,
        price_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Estimate trading capacity for a pair."""
        y_data = price_data.get(candidate.symbol_y)
        x_data = price_data.get(candidate.symbol_x)
        
        if y_data is None or x_data is None:
            return 0.0
        
        # Calculate average daily volume
        y_volume = y_data['volume'].mean() if 'volume' in y_data.columns else 0
        x_volume = x_data['volume'].mean() if 'volume' in x_data.columns else 0
        
        # Estimate capacity as percentage of average volume
        capacity_pct = 0.01  # 1% of daily volume
        capacity = min(y_volume, x_volume) * capacity_pct
        
        return capacity
