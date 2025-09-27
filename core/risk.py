"""
Risk management and position sizing.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings("ignore")


class RiskModel(Enum):
    """Risk model types."""
    FIXED = "fixed"
    VOLATILITY_TARGET = "volatility_target"
    KELLY = "kelly"
    VAR_BASED = "var_based"


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_leverage: float = 2.0
    max_position_size: float = 0.2  # 20% of capital per position
    max_portfolio_exposure: float = 1.0  # 100% of capital
    target_volatility: float = 0.15  # 15% annual volatility
    var_confidence: float = 0.05  # 5% VaR
    max_drawdown: float = 0.20  # 20% max drawdown
    risk_model: RiskModel = RiskModel.VOLATILITY_TARGET


class RiskManager:
    """Risk management and position sizing."""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.current_exposure = 0.0
        self.positions = {}
    
    def calculate_position_size(
        self, 
        signal_strength: float, 
        volatility: float, 
        available_capital: float,
        current_exposure: float = 0.0
    ) -> float:
        """Calculate optimal position size based on risk model."""
        
        if self.config.risk_model == RiskModel.FIXED:
            return self._fixed_sizing(signal_strength, available_capital)
        
        elif self.config.risk_model == RiskModel.VOLATILITY_TARGET:
            return self._volatility_target_sizing(
                signal_strength, volatility, available_capital, current_exposure
            )
        
        elif self.config.risk_model == RiskModel.KELLY:
            return self._kelly_sizing(signal_strength, volatility, available_capital)
        
        elif self.config.risk_model == RiskModel.VAR_BASED:
            return self._var_based_sizing(signal_strength, volatility, available_capital)
        
        else:
            return self._fixed_sizing(signal_strength, available_capital)
    
    def _fixed_sizing(
        self, 
        signal_strength: float, 
        available_capital: float
    ) -> float:
        """Fixed position sizing."""
        base_size = signal_strength * available_capital * 0.1  # 10% per signal
        return min(base_size, available_capital * self.config.max_position_size)
    
    def _volatility_target_sizing(
        self, 
        signal_strength: float, 
        volatility: float, 
        available_capital: float,
        current_exposure: float
    ) -> float:
        """Volatility targeting position sizing."""
        if volatility <= 0:
            return self._fixed_sizing(signal_strength, available_capital)
        
        # Calculate target position size based on volatility
        target_vol = self.config.target_volatility
        position_vol = volatility * signal_strength
        
        if position_vol > 0:
            vol_scalar = target_vol / position_vol
            base_size = signal_strength * available_capital * vol_scalar * 0.1
        else:
            base_size = signal_strength * available_capital * 0.1
        
        # Apply constraints
        max_size = available_capital * self.config.max_position_size
        remaining = available_capital * self.config.max_portfolio_exposure - current_exposure
        
        return min(base_size, max_size, remaining)
    
    def _kelly_sizing(
        self, 
        signal_strength: float, 
        volatility: float, 
        available_capital: float
    ) -> float:
        """Kelly criterion position sizing."""
        # Simplified Kelly calculation
        # In practice, you'd need historical win rate and average win/loss
        win_rate = 0.55  # Assume 55% win rate
        avg_win = 0.02   # 2% average win
        avg_loss = 0.015 # 1.5% average loss
        
        if avg_loss <= 0:
            return self._fixed_sizing(signal_strength, available_capital)
        
        # Kelly = (bp - q) / b
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        base_size = kelly_fraction * available_capital * signal_strength
        return min(base_size, available_capital * self.config.max_position_size)
    
    def _var_based_sizing(
        self, 
        signal_strength: float, 
        volatility: float, 
        available_capital: float
    ) -> float:
        """VaR-based position sizing."""
        if volatility <= 0:
            return self._fixed_sizing(signal_strength, available_capital)
        
        # Calculate VaR
        confidence_level = self.config.var_confidence
        var_multiplier = 1.96  # 95% confidence (simplified)
        var = volatility * var_multiplier
        
        # Size position to limit VaR
        max_var = available_capital * 0.02  # 2% max VaR
        if var > 0:
            position_size = (max_var / var) * available_capital * signal_strength
        else:
            position_size = signal_strength * available_capital * 0.1
        
        return min(position_size, available_capital * self.config.max_position_size)
    
    def check_risk_limits(
        self, 
        new_position_size: float, 
        current_exposure: float,
        portfolio_value: float
    ) -> Tuple[bool, str]:
        """Check if new position violates risk limits."""
        
        # Check leverage limit
        total_exposure = current_exposure + new_position_size
        if total_exposure > portfolio_value * self.config.max_leverage:
            return False, f"Exceeds leverage limit: {total_exposure/portfolio_value:.2f} > {self.config.max_leverage}"
        
        # Check position size limit
        if new_position_size > portfolio_value * self.config.max_position_size:
            return False, f"Exceeds position size limit: {new_position_size/portfolio_value:.2f} > {self.config.max_position_size}"
        
        # Check portfolio exposure limit
        if total_exposure > portfolio_value * self.config.max_portfolio_exposure:
            return False, f"Exceeds portfolio exposure limit: {total_exposure/portfolio_value:.2f} > {self.config.max_portfolio_exposure}"
        
        return True, "Risk limits OK"
    
    def calculate_portfolio_var(
        self, 
        positions: Dict[str, float], 
        volatilities: Dict[str, float],
        correlations: Optional[Dict[Tuple[str, str], float]] = None
    ) -> float:
        """Calculate portfolio Value at Risk."""
        if not positions:
            return 0.0
        
        # Calculate portfolio variance
        portfolio_var = 0.0
        
        for symbol, weight in positions.items():
            vol = volatilities.get(symbol, 0.02)
            portfolio_var += (weight * vol) ** 2
        
        # Add correlation terms
        if correlations:
            symbols = list(positions.keys())
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    corr = correlations.get((symbol1, symbol2), 0.0)
                    vol1 = volatilities.get(symbol1, 0.02)
                    vol2 = volatilities.get(symbol2, 0.02)
                    portfolio_var += 2 * positions[symbol1] * positions[symbol2] * vol1 * vol2 * corr
        
        # Calculate VaR
        portfolio_vol = np.sqrt(portfolio_var)
        var_multiplier = 1.96  # 95% confidence
        return portfolio_vol * var_multiplier
    
    def calculate_maximum_drawdown_control(
        self, 
        current_equity: float, 
        peak_equity: float,
        max_drawdown_limit: Optional[float] = None
    ) -> Tuple[bool, float]:
        """Check maximum drawdown control."""
        if max_drawdown_limit is None:
            max_drawdown_limit = self.config.max_drawdown
        
        current_dd = (peak_equity - current_equity) / peak_equity
        
        if current_dd > max_drawdown_limit:
            return True, current_dd  # Drawdown limit exceeded
        else:
            return False, current_dd
    
    def calculate_stop_loss(
        self, 
        entry_price: float, 
        volatility: float,
        confidence_level: float = 0.95
    ) -> float:
        """Calculate stop loss level."""
        # Simple stop loss based on volatility
        stop_multiplier = 2.0  # 2 standard deviations
        stop_distance = entry_price * volatility * stop_multiplier
        
        return entry_price - stop_distance
    
    def calculate_position_limits(
        self, 
        symbol: str, 
        daily_volume: float,
        position_size: float
    ) -> Tuple[bool, str]:
        """Check position size against market capacity."""
        # Simple capacity check
        max_position_pct = 0.01  # 1% of daily volume
        max_position = daily_volume * max_position_pct
        
        if position_size > max_position:
            return False, f"Position size {position_size:.0f} exceeds capacity limit {max_position:.0f}"
        
        return True, "Position size OK"
    
    def calculate_correlation_risk(
        self, 
        new_symbol: str, 
        existing_positions: Dict[str, float],
        correlation_matrix: Dict[Tuple[str, str], float]
    ) -> float:
        """Calculate correlation risk for new position."""
        if not existing_positions:
            return 0.0
        
        total_correlation_risk = 0.0
        
        for existing_symbol, weight in existing_positions.items():
            corr = correlation_matrix.get((new_symbol, existing_symbol), 0.0)
            total_correlation_risk += abs(corr) * weight
        
        return total_correlation_risk / len(existing_positions)
    
    def optimize_portfolio_weights(
        self, 
        expected_returns: Dict[str, float],
        volatilities: Dict[str, float],
        correlations: Dict[Tuple[str, str], float],
        target_return: Optional[float] = None
    ) -> Dict[str, float]:
        """Optimize portfolio weights using mean-variance optimization."""
        # Simplified optimization - in practice, use scipy.optimize
        symbols = list(expected_returns.keys())
        n_assets = len(symbols)
        
        if n_assets == 0:
            return {}
        
        # Equal weight as starting point
        weights = {symbol: 1.0 / n_assets for symbol in symbols}
        
        # Simple rebalancing based on risk-adjusted returns
        risk_adjusted_returns = {
            symbol: expected_returns[symbol] / volatilities[symbol] 
            for symbol in symbols if volatilities[symbol] > 0
        }
        
        if not risk_adjusted_returns:
            return weights
        
        # Weight by risk-adjusted returns
        total_rar = sum(risk_adjusted_returns.values())
        if total_rar > 0:
            weights = {
                symbol: risk_adjusted_returns[symbol] / total_rar 
                for symbol in symbols
            }
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {symbol: weight / total_weight for symbol, weight in weights.items()}
        
        return weights
