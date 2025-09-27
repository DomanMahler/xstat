"""
Cost models for realistic backtesting.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings("ignore")


class CostModel(Enum):
    """Cost model types."""
    FIXED = "fixed"
    LINEAR = "linear"
    SQUARE_ROOT = "square_root"
    ADV_BASED = "adv_based"


@dataclass
class CostConfig:
    """Configuration for cost models."""
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005   # 0.05%
    borrow_rate: float = 0.02       # 2% annual
    stamp_duty_rate: float = 0.005  # 0.5% (UK)
    sec_fee_rate: float = 0.0000229  # SEC fee
    min_commission: float = 1.0     # Minimum commission
    max_commission: float = 1000.0  # Maximum commission
    cost_model: CostModel = CostModel.LINEAR


class CostCalculator:
    """Calculate realistic trading costs."""
    
    def __init__(self, config: CostConfig):
        self.config = config
    
    def calculate_commission(
        self, 
        notional: float, 
        side: str = "buy"
    ) -> float:
        """Calculate commission costs."""
        commission = notional * self.config.commission_rate
        
        # Apply min/max bounds
        commission = max(commission, self.config.min_commission)
        commission = min(commission, self.config.max_commission)
        
        # Add SEC fee for US equities
        if side == "sell":
            commission += notional * self.config.sec_fee_rate
        
        return commission
    
    def calculate_slippage(
        self, 
        notional: float, 
        volatility: float = 0.02,
        model: Optional[CostModel] = None
    ) -> float:
        """Calculate slippage costs."""
        if model is None:
            model = self.config.cost_model
        
        if model == CostModel.FIXED:
            return notional * self.config.slippage_rate
        
        elif model == CostModel.LINEAR:
            # Linear in notional size
            return notional * self.config.slippage_rate
        
        elif model == CostModel.SQUARE_ROOT:
            # Square root impact model
            impact = self.config.slippage_rate * np.sqrt(notional / 1000000)  # Normalize to $1M
            return notional * impact
        
        elif model == CostModel.ADV_BASED:
            # Based on average daily volume
            # Simplified: assume ADV is proportional to volatility
            adv_factor = min(volatility * 10, 1.0)  # Cap at 1.0
            return notional * self.config.slippage_rate * adv_factor
        
        else:
            return notional * self.config.slippage_rate
    
    def calculate_borrow_costs(
        self, 
        short_notional: float, 
        holding_days: int
    ) -> float:
        """Calculate borrowing costs for short positions."""
        daily_rate = self.config.borrow_rate / 365
        return short_notional * daily_rate * holding_days
    
    def calculate_stamp_duty(
        self, 
        notional: float, 
        side: str = "buy"
    ) -> float:
        """Calculate stamp duty (UK equities)."""
        if side == "buy":
            return notional * self.config.stamp_duty_rate
        return 0.0
    
    def calculate_total_costs(
        self, 
        trade_details: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate total costs for a trade."""
        notional_y = trade_details.get('notional_y', 0)
        notional_x = trade_details.get('notional_x', 0)
        side_y = trade_details.get('side_y', 'buy')
        side_x = trade_details.get('side_x', 'buy')
        volatility_y = trade_details.get('volatility_y', 0.02)
        volatility_x = trade_details.get('volatility_x', 0.02)
        holding_days = trade_details.get('holding_days', 1)
        
        # Commission costs
        commission_y = self.calculate_commission(notional_y, side_y)
        commission_x = self.calculate_commission(notional_x, side_x)
        
        # Slippage costs
        slippage_y = self.calculate_slippage(notional_y, volatility_y)
        slippage_x = self.calculate_slippage(notional_x, volatility_x)
        
        # Borrow costs (for short positions)
        borrow_y = self.calculate_borrow_costs(
            abs(notional_y) if side_y == "sell" else 0, holding_days
        )
        borrow_x = self.calculate_borrow_costs(
            abs(notional_x) if side_x == "sell" else 0, holding_days
        )
        
        # Stamp duty (for UK equities)
        stamp_y = self.calculate_stamp_duty(notional_y, side_y)
        stamp_x = self.calculate_stamp_duty(notional_x, side_x)
        
        return {
            'commission_y': commission_y,
            'commission_x': commission_x,
            'slippage_y': slippage_y,
            'slippage_x': slippage_x,
            'borrow_y': borrow_y,
            'borrow_x': borrow_x,
            'stamp_y': stamp_y,
            'stamp_x': stamp_x,
            'total': commission_y + commission_x + slippage_y + slippage_x + 
                    borrow_y + borrow_x + stamp_y + stamp_x
        }
    
    def calculate_crypto_costs(
        self, 
        trade_details: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate costs for crypto trading."""
        notional_y = trade_details.get('notional_y', 0)
        notional_x = trade_details.get('notional_x', 0)
        side_y = trade_details.get('side_y', 'buy')
        side_x = trade_details.get('side_x', 'buy')
        
        # Crypto typically has maker/taker fees
        maker_rate = 0.001  # 0.1%
        taker_rate = 0.001  # 0.1%
        
        # Assume market orders (taker)
        fee_y = notional_y * taker_rate
        fee_x = notional_x * taker_rate
        
        # Crypto slippage is typically higher
        slippage_y = notional_y * 0.001  # 0.1%
        slippage_x = notional_x * 0.001
        
        return {
            'fee_y': fee_y,
            'fee_x': fee_x,
            'slippage_y': slippage_y,
            'slippage_x': slippage_x,
            'total': fee_y + fee_x + slippage_y + slippage_x
        }
    
    def calculate_futures_costs(
        self, 
        trade_details: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate costs for futures trading."""
        notional_y = trade_details.get('notional_y', 0)
        notional_x = trade_details.get('notional_x', 0)
        
        # Futures typically have lower costs
        commission_rate = 0.0001  # 0.01%
        slippage_rate = 0.0001   # 0.01%
        
        commission_y = notional_y * commission_rate
        commission_x = notional_x * commission_rate
        slippage_y = notional_y * slippage_rate
        slippage_x = notional_x * slippage_rate
        
        return {
            'commission_y': commission_y,
            'commission_x': commission_x,
            'slippage_y': slippage_y,
            'slippage_x': slippage_x,
            'total': commission_y + commission_x + slippage_y + slippage_x
        }
    
    def calculate_fx_costs(
        self, 
        trade_details: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate costs for FX trading."""
        notional_y = trade_details.get('notional_y', 0)
        notional_x = trade_details.get('notional_x', 0)
        
        # FX spreads are typically very tight
        spread_rate = 0.0001  # 0.01%
        
        spread_y = notional_y * spread_rate
        spread_x = notional_x * spread_rate
        
        return {
            'spread_y': spread_y,
            'spread_x': spread_x,
            'total': spread_y + spread_x
        }


class CostOptimizer:
    """Optimize costs through smart execution."""
    
    def __init__(self, cost_calculator: CostCalculator):
        self.cost_calculator = cost_calculator
    
    def suggest_execution_timing(
        self, 
        signals: List[Any], 
        volatility_data: Dict[str, pd.Series]
    ) -> List[Dict[str, Any]]:
        """Suggest optimal execution timing to minimize costs."""
        optimized_signals = []
        
        for signal in signals:
            # Get current volatility
            symbol_y = signal.metadata.get('symbol_y', 'Y') if signal.metadata else 'Y'
            symbol_x = signal.metadata.get('symbol_x', 'X') if signal.metadata else 'X'
            
            vol_y = volatility_data.get(symbol_y, pd.Series([0.02])).iloc[-1]
            vol_x = volatility_data.get(symbol_x, pd.Series([0.02])).iloc[-1]
            
            # Suggest execution during low volatility periods
            if vol_y < 0.01 and vol_x < 0.01:  # Low volatility
                signal.metadata = signal.metadata or {}
                signal.metadata['execution_priority'] = 'high'
            elif vol_y > 0.05 or vol_x > 0.05:  # High volatility
                signal.metadata = signal.metadata or {}
                signal.metadata['execution_priority'] = 'low'
            else:
                signal.metadata = signal.metadata or {}
                signal.metadata['execution_priority'] = 'medium'
            
            optimized_signals.append(signal)
        
        return optimized_signals
    
    def calculate_cost_impact(
        self, 
        trade_size: float, 
        market_impact: float = 0.001
    ) -> float:
        """Calculate market impact costs."""
        # Simplified market impact model
        impact = market_impact * (trade_size / 1000000) ** 0.5  # Square root impact
        return trade_size * impact
    
    def optimize_position_sizing(
        self, 
        signals: List[Any], 
        available_capital: float
    ) -> List[Dict[str, Any]]:
        """Optimize position sizing to balance returns and costs."""
        optimized_signals = []
        
        for signal in signals:
            # Calculate optimal size based on signal strength and costs
            base_size = signal.strength * available_capital * 0.1  # 10% per signal
            
            # Adjust for cost considerations
            if signal.metadata and signal.metadata.get('execution_priority') == 'low':
                base_size *= 0.5  # Reduce size in high volatility
            elif signal.metadata and signal.metadata.get('execution_priority') == 'high':
                base_size *= 1.2  # Increase size in low volatility
            
            # Cap at maximum position size
            max_size = available_capital * 0.2  # 20% max per position
            optimal_size = min(base_size, max_size)
            
            signal.strength = optimal_size / available_capital
            optimized_signals.append(signal)
        
        return optimized_signals
