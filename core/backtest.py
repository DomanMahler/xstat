"""
Event-driven backtester for statistical arbitrage strategies.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


@dataclass
class Trade:
    """Individual trade record."""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    symbol_y: str
    symbol_x: str
    side: str  # 'long' or 'short'
    quantity_y: float
    quantity_x: float
    entry_price_y: float
    entry_price_x: float
    exit_price_y: Optional[float]
    exit_price_x: Optional[float]
    hedge_ratio: float
    pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Current position state."""
    symbol_y: str
    symbol_x: str
    quantity_y: float
    quantity_x: float
    entry_time: pd.Timestamp
    entry_price_y: float
    entry_price_x: float
    hedge_ratio: float
    unrealized_pnl: float = 0.0


class Backtester:
    """Event-driven backtester for statistical arbitrage."""
    
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        commission_rate: float = 0.001,  # 0.1%
        slippage_rate: float = 0.0005,   # 0.05%
        borrow_rate: float = 0.02,       # 2% annual
        max_leverage: float = 2.0,
        min_trade_size: float = 1000.0
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.borrow_rate = borrow_rate
        self.max_leverage = max_leverage
        self.min_trade_size = min_trade_size
        
        # State variables
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_pnl: List[Dict[str, Any]] = []
        self.current_date = None
        
        # Performance tracking
        self.equity_curve = []
        self.drawdown_curve = []
        self.max_drawdown = 0.0
        self.peak_equity = initial_capital
    
    def reset(self) -> None:
        """Reset backtester state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.daily_pnl.clear()
        self.equity_curve.clear()
        self.drawdown_curve.clear()
        self.max_drawdown = 0.0
        self.peak_equity = self.initial_capital
        self.current_date = None
    
    def run_backtest(
        self,
        signals: List[Any],  # Signal objects
        price_data: Dict[str, pd.DataFrame],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Run backtest with signals and price data.
        
        Args:
            signals: List of Signal objects
            price_data: Dict of symbol -> DataFrame with OHLCV data
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Dict with backtest results
        """
        self.reset()
        
        # Filter signals by date range
        if start_date or end_date:
            signals = self._filter_signals_by_date(signals, start_date, end_date)
        
        # Process signals chronologically
        for signal in signals:
            self._process_signal(signal, price_data)
        
        # Close any remaining positions
        self._close_all_positions(price_data)
        
        # Calculate final results
        return self._calculate_results()
    
    def _filter_signals_by_date(
        self,
        signals: List[Any],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[Any]:
        """Filter signals by date range."""
        filtered = []
        for signal in signals:
            if start_date and signal.timestamp < start_date:
                continue
            if end_date and signal.timestamp > end_date:
                continue
            filtered.append(signal)
        return filtered
    
    def _process_signal(self, signal: Any, price_data: Dict[str, pd.DataFrame]) -> None:
        """Process a single signal."""
        self.current_date = signal.timestamp
        
        if signal.signal_type.value == 0:  # Neutral/Exit signal
            self._close_position(signal, price_data)
        else:
            self._open_position(signal, price_data)
        
        # Update daily P&L
        self._update_daily_pnl()
    
    def _open_position(self, signal: Any, price_data: Dict[str, pd.DataFrame]) -> None:
        """Open a new position based on signal."""
        # Extract symbols from signal metadata or use defaults
        symbol_y = signal.metadata.get('symbol_y', 'Y') if signal.metadata else 'Y'
        symbol_x = signal.metadata.get('symbol_x', 'X') if signal.metadata else 'X'
        
        # Get current prices
        try:
            price_y = self._get_price(signal.timestamp, symbol_y, price_data)
            price_x = self._get_price(signal.timestamp, symbol_x, price_data)
        except KeyError:
            return  # Skip if price data not available
        
        # Calculate position size
        hedge_ratio = signal.metadata.get('hedge_ratio', 1.0) if signal.metadata else 1.0
        position_value = self.cash * signal.strength * 0.1  # Use 10% of capital per signal
        
        if position_value < self.min_trade_size:
            return
        
        # Calculate quantities
        if signal.signal_type.value == 1:  # Long spread
            quantity_y = position_value / price_y
            quantity_x = -(position_value * hedge_ratio) / price_x
        else:  # Short spread
            quantity_y = -(position_value / price_y)
            quantity_x = (position_value * hedge_ratio) / price_x
        
        # Check leverage constraints
        total_exposure = abs(quantity_y * price_y) + abs(quantity_x * price_x)
        if total_exposure > self.cash * self.max_leverage:
            return
        
        # Calculate costs
        costs = self._calculate_trade_costs(quantity_y, price_y, quantity_x, price_x)
        
        # Update cash
        self.cash -= costs
        
        # Create position
        position_key = f"{symbol_y}_{symbol_x}"
        self.positions[position_key] = Position(
            symbol_y=symbol_y,
            symbol_x=symbol_x,
            quantity_y=quantity_y,
            quantity_x=quantity_x,
            entry_time=signal.timestamp,
            entry_price_y=price_y,
            entry_price_x=price_x,
            hedge_ratio=hedge_ratio
        )
    
    def _close_position(self, signal: Any, price_data: Dict[str, pd.DataFrame]) -> None:
        """Close existing positions."""
        for position_key, position in list(self.positions.items()):
            # Get current prices
            try:
                price_y = self._get_price(signal.timestamp, position.symbol_y, price_data)
                price_x = self._get_price(signal.timestamp, position.symbol_x, price_data)
            except KeyError:
                continue
            
            # Calculate P&L
            pnl_y = position.quantity_y * (price_y - position.entry_price_y)
            pnl_x = position.quantity_x * (price_x - position.entry_price_x)
            total_pnl = pnl_y + pnl_x
            
            # Calculate exit costs
            exit_costs = self._calculate_trade_costs(
                -position.quantity_y, price_y,
                -position.quantity_x, price_x
            )
            
            # Update cash
            self.cash += total_pnl - exit_costs
            
            # Record trade
            trade = Trade(
                entry_time=position.entry_time,
                exit_time=signal.timestamp,
                symbol_y=position.symbol_y,
                symbol_x=position.symbol_x,
                side='long' if position.quantity_y > 0 else 'short',
                quantity_y=position.quantity_y,
                quantity_x=position.quantity_x,
                entry_price_y=position.entry_price_y,
                entry_price_x=position.entry_price_x,
                exit_price_y=price_y,
                exit_price_x=price_x,
                hedge_ratio=position.hedge_ratio,
                pnl=total_pnl,
                costs=exit_costs,
                net_pnl=total_pnl - exit_costs,
                metadata=signal.metadata or {}
            )
            self.trades.append(trade)
            
            # Remove position
            del self.positions[position_key]
    
    def _close_all_positions(self, price_data: Dict[str, pd.DataFrame]) -> None:
        """Close all remaining positions at the end."""
        if not self.positions:
            return
        
        # Use the last available date
        last_date = max([df.index[-1] for df in price_data.values()])
        
        for position_key, position in list(self.positions.items()):
            try:
                price_y = self._get_price(last_date, position.symbol_y, price_data)
                price_x = self._get_price(last_date, position.symbol_x, price_data)
            except KeyError:
                continue
            
            # Calculate P&L
            pnl_y = position.quantity_y * (price_y - position.entry_price_y)
            pnl_x = position.quantity_x * (price_x - position.entry_price_x)
            total_pnl = pnl_y + pnl_x
            
            # Calculate exit costs
            exit_costs = self._calculate_trade_costs(
                -position.quantity_y, price_y,
                -position.quantity_x, price_x
            )
            
            # Update cash
            self.cash += total_pnl - exit_costs
            
            # Record trade
            trade = Trade(
                entry_time=position.entry_time,
                exit_time=last_date,
                symbol_y=position.symbol_y,
                symbol_x=position.symbol_x,
                side='long' if position.quantity_y > 0 else 'short',
                quantity_y=position.quantity_y,
                quantity_x=position.quantity_x,
                entry_price_y=position.entry_price_y,
                entry_price_x=position.entry_price_x,
                exit_price_y=price_y,
                exit_price_x=price_x,
                hedge_ratio=position.hedge_ratio,
                pnl=total_pnl,
                costs=exit_costs,
                net_pnl=total_pnl - exit_costs,
                metadata={'exit_reason': 'end_of_backtest'}
            )
            self.trades.append(trade)
            
            # Remove position
            del self.positions[position_key]
    
    def _get_price(self, timestamp: pd.Timestamp, symbol: str, price_data: Dict[str, pd.DataFrame]) -> float:
        """Get price for a symbol at a given timestamp."""
        if symbol not in price_data:
            raise KeyError(f"Symbol {symbol} not found in price data")
        
        df = price_data[symbol]
        
        # Find the closest available price
        if timestamp in df.index:
            return df.loc[timestamp, 'close']
        
        # Find the most recent price before timestamp
        available_dates = df.index[df.index <= timestamp]
        if len(available_dates) == 0:
            raise KeyError(f"No price data available for {symbol} before {timestamp}")
        
        latest_date = available_dates[-1]
        return df.loc[latest_date, 'close']
    
    def _calculate_trade_costs(
        self,
        quantity_y: float,
        price_y: float,
        quantity_x: float,
        price_x: float
    ) -> float:
        """Calculate trading costs for a position."""
        # Commission costs
        commission_y = abs(quantity_y * price_y) * self.commission_rate
        commission_x = abs(quantity_x * price_x) * self.commission_rate
        
        # Slippage costs
        slippage_y = abs(quantity_y * price_y) * self.slippage_rate
        slippage_x = abs(quantity_x * price_x) * self.slippage_rate
        
        return commission_y + commission_x + slippage_y + slippage_x
    
    def _update_daily_pnl(self) -> None:
        """Update daily P&L tracking."""
        if not self.current_date:
            return
        
        # Calculate unrealized P&L for open positions
        unrealized_pnl = 0.0
        for position in self.positions.values():
            # This would require current prices, simplified here
            unrealized_pnl += position.unrealized_pnl
        
        # Calculate total equity
        total_equity = self.cash + sum(pos.quantity_y * pos.entry_price_y + pos.quantity_x * pos.entry_price_x 
                                     for pos in self.positions.values())
        
        # Update equity curve
        self.equity_curve.append({
            'date': self.current_date,
            'equity': total_equity,
            'cash': self.cash,
            'unrealized_pnl': unrealized_pnl
        })
        
        # Update drawdown
        if total_equity > self.peak_equity:
            self.peak_equity = total_equity
        
        current_drawdown = (self.peak_equity - total_equity) / self.peak_equity
        self.drawdown_curve.append(current_drawdown)
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate backtest results."""
        if not self.trades:
            return {
                'total_trades': 0,
                'total_pnl': 0.0,
                'total_costs': 0.0,
                'net_pnl': 0.0,
                'final_equity': self.cash,
                'return_pct': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'avg_trade_pnl': 0.0,
                'trades': []
            }
        
        # Basic statistics
        total_trades = len(self.trades)
        total_pnl = sum(trade.pnl for trade in self.trades)
        total_costs = sum(trade.costs for trade in self.trades)
        net_pnl = sum(trade.net_pnl for trade in self.trades)
        
        # Performance metrics
        final_equity = self.cash
        return_pct = (final_equity - self.initial_capital) / self.initial_capital
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t.net_pnl > 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_trade_pnl = net_pnl / total_trades if total_trades > 0 else 0
        
        # Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = pd.Series([eq['equity'] for eq in self.equity_curve]).pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0.0
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'total_costs': total_costs,
            'net_pnl': net_pnl,
            'final_equity': final_equity,
            'return_pct': return_pct,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'avg_trade_pnl': avg_trade_pnl,
            'trades': [
                {
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'symbol_y': trade.symbol_y,
                    'symbol_x': trade.symbol_x,
                    'side': trade.side,
                    'pnl': trade.pnl,
                    'costs': trade.costs,
                    'net_pnl': trade.net_pnl
                }
                for trade in self.trades
            ],
            'equity_curve': self.equity_curve,
            'drawdown_curve': self.drawdown_curve
        }
