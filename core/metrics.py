"""
Performance metrics and risk calculations.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


class PerformanceMetrics:
    """Comprehensive performance metrics calculation."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_returns(self, equity_curve: List[Dict[str, Any]]) -> pd.Series:
        """Calculate returns from equity curve."""
        if not equity_curve:
            return pd.Series(dtype=float)
        
        equity_values = [eq['equity'] for eq in equity_curve]
        dates = [eq['date'] for eq in equity_curve]
        
        equity_series = pd.Series(equity_values, index=dates)
        returns = equity_series.pct_change().dropna()
        
        return returns
    
    def calculate_sharpe_ratio(
        self, 
        returns: pd.Series, 
        annualization_factor: int = 252
    ) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / annualization_factor
        return excess_returns.mean() / returns.std() * np.sqrt(annualization_factor)
    
    def calculate_sortino_ratio(
        self, 
        returns: pd.Series, 
        annualization_factor: int = 252
    ) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / annualization_factor
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return excess_returns.mean() / downside_returns.std() * np.sqrt(annualization_factor)
    
    def calculate_max_drawdown(self, equity_curve: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate maximum drawdown and related metrics."""
        if not equity_curve:
            return {'max_drawdown': 0.0, 'max_drawdown_duration': 0, 'recovery_time': 0}
        
        equity_values = [eq['equity'] for eq in equity_curve]
        dates = [eq['date'] for eq in equity_curve]
        
        equity_series = pd.Series(equity_values, index=dates)
        
        # Calculate running maximum
        running_max = equity_series.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_series - running_max) / running_max
        
        # Maximum drawdown
        max_dd = drawdown.min()
        
        # Drawdown duration
        dd_duration = self._calculate_drawdown_duration(drawdown)
        
        # Recovery time
        recovery_time = self._calculate_recovery_time(drawdown)
        
        return {
            'max_drawdown': abs(max_dd),
            'max_drawdown_duration': dd_duration,
            'recovery_time': recovery_time,
            'drawdown_series': drawdown
        }
    
    def calculate_var_cvar(
        self, 
        returns: pd.Series, 
        confidence_level: float = 0.05
    ) -> Dict[str, float]:
        """Calculate Value at Risk and Conditional Value at Risk."""
        if len(returns) == 0:
            return {'var': 0.0, 'cvar': 0.0}
        
        # Value at Risk
        var = np.percentile(returns, confidence_level * 100)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar = returns[returns <= var].mean()
        
        return {
            'var': var,
            'cvar': cvar
        }
    
    def calculate_calmar_ratio(
        self, 
        returns: pd.Series, 
        max_drawdown: float
    ) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return 0.0
        
        annual_return = returns.mean() * 252
        return annual_return / abs(max_drawdown)
    
    def calculate_information_ratio(
        self, 
        strategy_returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate Information ratio."""
        if len(strategy_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align returns
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_index) == 0:
            return 0.0
        
        strategy_aligned = strategy_returns.loc[common_index]
        benchmark_aligned = benchmark_returns.loc[common_index]
        
        excess_returns = strategy_aligned - benchmark_aligned
        
        if excess_returns.std() == 0:
            return 0.0
        
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    def calculate_turnover(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate portfolio turnover."""
        if not trades:
            return 0.0
        
        total_volume = sum(abs(trade.get('quantity_y', 0)) + abs(trade.get('quantity_x', 0)) 
                          for trade in trades)
        
        # This is a simplified calculation
        # In practice, you'd need to track portfolio value over time
        return total_volume
    
    def calculate_hit_rate(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate hit rate (percentage of profitable trades)."""
        if not trades:
            return 0.0
        
        profitable_trades = sum(1 for trade in trades if trade.get('net_pnl', 0) > 0)
        return profitable_trades / len(trades)
    
    def calculate_profit_factor(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate profit factor."""
        if not trades:
            return 0.0
        
        gross_profit = sum(trade.get('net_pnl', 0) for trade in trades if trade.get('net_pnl', 0) > 0)
        gross_loss = abs(sum(trade.get('net_pnl', 0) for trade in trades if trade.get('net_pnl', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def calculate_kelly_criterion(
        self, 
        trades: List[Dict[str, Any]], 
        win_rate: float, 
        avg_win: float, 
        avg_loss: float
    ) -> float:
        """Calculate Kelly criterion for position sizing."""
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        # Kelly = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - win_rate
        
        kelly = (b * p - q) / b
        return max(0, min(kelly, 1))  # Cap between 0 and 1
    
    def calculate_capacity_estimate(
        self, 
        trades: List[Dict[str, Any]], 
        avg_trade_size: float
    ) -> Dict[str, float]:
        """Estimate strategy capacity."""
        if not trades:
            return {'capacity': 0.0, 'utilization': 0.0}
        
        # Simple capacity estimate based on trade frequency and size
        total_volume = sum(abs(trade.get('quantity_y', 0)) + abs(trade.get('quantity_x', 0)) 
                          for trade in trades)
        
        # Estimate capacity (simplified)
        capacity = total_volume * 10  # Assume 10x current volume is sustainable
        
        return {
            'capacity': capacity,
            'utilization': total_volume / capacity if capacity > 0 else 0.0
        }
    
    def calculate_rolling_metrics(
        self, 
        returns: pd.Series, 
        window: int = 252
    ) -> Dict[str, pd.Series]:
        """Calculate rolling performance metrics."""
        if len(returns) == 0:
            return {}
        
        rolling_sharpe = returns.rolling(window=window).apply(
            lambda x: self.calculate_sharpe_ratio(x) if len(x) > 1 else 0
        )
        
        rolling_volatility = returns.rolling(window=window).std() * np.sqrt(252)
        
        rolling_return = returns.rolling(window=window).mean() * 252
        
        return {
            'rolling_sharpe': rolling_sharpe,
            'rolling_volatility': rolling_volatility,
            'rolling_return': rolling_return
        }
    
    def calculate_comprehensive_metrics(
        self, 
        equity_curve: List[Dict[str, Any]], 
        trades: List[Dict[str, Any]],
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not equity_curve:
            return {}
        
        # Calculate returns
        returns = self.calculate_returns(equity_curve)
        
        # Basic metrics
        total_return = (equity_curve[-1]['equity'] - equity_curve[0]['equity']) / equity_curve[0]['equity']
        annual_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        
        # Risk metrics
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        sortino_ratio = self.calculate_sortino_ratio(returns)
        max_dd_metrics = self.calculate_max_drawdown(equity_curve)
        var_cvar = self.calculate_var_cvar(returns)
        calmar_ratio = self.calculate_calmar_ratio(returns, max_dd_metrics['max_drawdown'])
        
        # Trade metrics
        hit_rate = self.calculate_hit_rate(trades)
        profit_factor = self.calculate_profit_factor(trades)
        turnover = self.calculate_turnover(trades)
        
        # Rolling metrics
        rolling_metrics = self.calculate_rolling_metrics(returns)
        
        # Benchmark comparison
        information_ratio = 0.0
        if benchmark_returns is not None:
            information_ratio = self.calculate_information_ratio(returns, benchmark_returns)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_dd_metrics['max_drawdown'],
            'max_drawdown_duration': max_dd_metrics['max_drawdown_duration'],
            'recovery_time': max_dd_metrics['recovery_time'],
            'var_5pct': var_cvar['var'],
            'cvar_5pct': var_cvar['cvar'],
            'calmar_ratio': calmar_ratio,
            'hit_rate': hit_rate,
            'profit_factor': profit_factor,
            'turnover': turnover,
            'information_ratio': information_ratio,
            'rolling_metrics': rolling_metrics,
            'returns': returns
        }
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in periods."""
        if len(drawdown) == 0:
            return 0
        
        # Find consecutive periods in drawdown
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def _calculate_recovery_time(self, drawdown: pd.Series) -> int:
        """Calculate average recovery time from drawdowns."""
        if len(drawdown) == 0:
            return 0
        
        # Find drawdown periods and their recovery times
        recovery_times = []
        in_drawdown = False
        drawdown_start = None
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                drawdown_start = i
            elif dd >= 0 and in_drawdown:
                # End of drawdown
                in_drawdown = False
                if drawdown_start is not None:
                    recovery_time = i - drawdown_start
                    recovery_times.append(recovery_time)
                drawdown_start = None
        
        return int(np.mean(recovery_times)) if recovery_times else 0
