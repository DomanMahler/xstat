"""
Signal generation for statistical arbitrage strategies.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings("ignore")


class SignalType(Enum):
    """Signal types."""
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


@dataclass
class Signal:
    """Signal data structure."""
    timestamp: pd.Timestamp
    signal_type: SignalType
    strength: float
    entry_price: float
    exit_price: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class SignalGenerator:
    """Generate trading signals for statistical arbitrage."""
    
    def __init__(
        self,
        entry_z_threshold: float = 2.0,
        exit_z_threshold: float = 0.5,
        max_position_size: float = 1.0,
        min_half_life: float = 1.0,
        max_half_life: float = 30.0,
        regime_filter: bool = True
    ):
        self.entry_z_threshold = entry_z_threshold
        self.exit_z_threshold = exit_z_threshold
        self.max_position_size = max_position_size
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.regime_filter = regime_filter
    
    def generate_zscore_signals(
        self,
        spread: pd.Series,
        zscore: pd.Series,
        hedge_ratio: float = 1.0,
        y_price: Optional[pd.Series] = None,
        x_price: Optional[pd.Series] = None
    ) -> List[Signal]:
        """
        Generate signals based on z-score thresholds.
        
        Args:
            spread: Spread time series
            zscore: Z-score of spread
            hedge_ratio: Hedge ratio for position sizing
            y_price: Price of first asset (for entry/exit prices)
            x_price: Price of second asset (for entry/exit prices)
            
        Returns:
            List of Signal objects
        """
        signals = []
        current_position = 0
        entry_price = None
        
        for i, (timestamp, z_val) in enumerate(zscore.items()):
            if pd.isna(z_val):
                continue
            
            # Entry signals
            if current_position == 0:
                if z_val > self.entry_z_threshold:
                    # Short spread (long x, short y)
                    signal_type = SignalType.SHORT
                    entry_price = self._get_entry_price(timestamp, y_price, x_price, hedge_ratio, signal_type)
                    signals.append(Signal(
                        timestamp=timestamp,
                        signal_type=signal_type,
                        strength=min(abs(z_val) / self.entry_z_threshold, self.max_position_size),
                        entry_price=entry_price,
                        metadata={'zscore': z_val, 'spread': spread.iloc[i]}
                    ))
                    current_position = -1
                    
                elif z_val < -self.entry_z_threshold:
                    # Long spread (short x, long y)
                    signal_type = SignalType.LONG
                    entry_price = self._get_entry_price(timestamp, y_price, x_price, hedge_ratio, signal_type)
                    signals.append(Signal(
                        timestamp=timestamp,
                        signal_type=signal_type,
                        strength=min(abs(z_val) / self.entry_z_threshold, self.max_position_size),
                        entry_price=entry_price,
                        metadata={'zscore': z_val, 'spread': spread.iloc[i]}
                    ))
                    current_position = 1
            
            # Exit signals
            elif current_position != 0:
                if abs(z_val) < self.exit_z_threshold:
                    # Exit position
                    exit_price = self._get_exit_price(timestamp, y_price, x_price, hedge_ratio, current_position)
                    signals.append(Signal(
                        timestamp=timestamp,
                        signal_type=SignalType.NEUTRAL,
                        strength=0.0,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        metadata={'zscore': z_val, 'spread': spread.iloc[i], 'exit_reason': 'zscore_exit'}
                    ))
                    current_position = 0
                    entry_price = None
        
        return signals
    
    def generate_bollinger_signals(
        self,
        spread: pd.Series,
        bb_upper: pd.Series,
        bb_lower: pd.Series,
        bb_middle: pd.Series,
        hedge_ratio: float = 1.0,
        y_price: Optional[pd.Series] = None,
        x_price: Optional[pd.Series] = None
    ) -> List[Signal]:
        """
        Generate signals based on Bollinger Bands.
        
        Args:
            spread: Spread time series
            bb_upper: Upper Bollinger Band
            bb_lower: Lower Bollinger Band
            bb_middle: Middle Bollinger Band (SMA)
            hedge_ratio: Hedge ratio for position sizing
            y_price: Price of first asset
            x_price: Price of second asset
            
        Returns:
            List of Signal objects
        """
        signals = []
        current_position = 0
        entry_price = None
        
        for i, timestamp in enumerate(spread.index):
            if pd.isna(spread.iloc[i]) or pd.isna(bb_upper.iloc[i]) or pd.isna(bb_lower.iloc[i]):
                continue
            
            spread_val = spread.iloc[i]
            upper_val = bb_upper.iloc[i]
            lower_val = bb_lower.iloc[i]
            middle_val = bb_middle.iloc[i]
            
            # Entry signals
            if current_position == 0:
                if spread_val > upper_val:
                    # Short spread (long x, short y)
                    signal_type = SignalType.SHORT
                    entry_price = self._get_entry_price(timestamp, y_price, x_price, hedge_ratio, signal_type)
                    strength = min((spread_val - middle_val) / (upper_val - middle_val), self.max_position_size)
                    signals.append(Signal(
                        timestamp=timestamp,
                        signal_type=signal_type,
                        strength=strength,
                        entry_price=entry_price,
                        metadata={'spread': spread_val, 'bb_position': (spread_val - lower_val) / (upper_val - lower_val)}
                    ))
                    current_position = -1
                    
                elif spread_val < lower_val:
                    # Long spread (short x, long y)
                    signal_type = SignalType.LONG
                    entry_price = self._get_entry_price(timestamp, y_price, x_price, hedge_ratio, signal_type)
                    strength = min((middle_val - spread_val) / (middle_val - lower_val), self.max_position_size)
                    signals.append(Signal(
                        timestamp=timestamp,
                        signal_type=signal_type,
                        strength=strength,
                        entry_price=entry_price,
                        metadata={'spread': spread_val, 'bb_position': (spread_val - lower_val) / (upper_val - lower_val)}
                    ))
                    current_position = 1
            
            # Exit signals
            elif current_position != 0:
                if (current_position == 1 and spread_val >= middle_val) or \
                   (current_position == -1 and spread_val <= middle_val):
                    # Exit position
                    exit_price = self._get_exit_price(timestamp, y_price, x_price, hedge_ratio, current_position)
                    signals.append(Signal(
                        timestamp=timestamp,
                        signal_type=SignalType.NEUTRAL,
                        strength=0.0,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        metadata={'spread': spread_val, 'exit_reason': 'bollinger_exit'}
                    ))
                    current_position = 0
                    entry_price = None
        
        return signals
    
    def generate_regime_filtered_signals(
        self,
        spread: pd.Series,
        zscore: pd.Series,
        regime_features: Dict[str, pd.Series],
        hedge_ratio: float = 1.0,
        y_price: Optional[pd.Series] = None,
        x_price: Optional[pd.Series] = None
    ) -> List[Signal]:
        """
        Generate signals with regime filtering.
        
        Args:
            spread: Spread time series
            zscore: Z-score of spread
            regime_features: Dictionary of regime features
            hedge_ratio: Hedge ratio for position sizing
            y_price: Price of first asset
            x_price: Price of second asset
            
        Returns:
            List of Signal objects
        """
        if not self.regime_filter:
            return self.generate_zscore_signals(spread, zscore, hedge_ratio, y_price, x_price)
        
        # Filter out high volatility regimes
        high_vol_regime = regime_features.get('high_vol_regime', pd.Series([0] * len(spread), index=spread.index))
        
        # Create filtered zscore
        filtered_zscore = zscore.copy()
        filtered_zscore[high_vol_regime == 1] = np.nan
        
        return self.generate_zscore_signals(spread, filtered_zscore, hedge_ratio, y_price, x_price)
    
    def generate_time_stop_signals(
        self,
        signals: List[Signal],
        max_holding_periods: int = 20
    ) -> List[Signal]:
        """
        Add time-based exit signals.
        
        Args:
            signals: List of existing signals
            max_holding_periods: Maximum periods to hold a position
            
        Returns:
            Updated list of signals with time stops
        """
        if not signals:
            return signals
        
        # Convert to DataFrame for easier manipulation
        signal_df = pd.DataFrame([
            {
                'timestamp': s.timestamp,
                'signal_type': s.signal_type,
                'strength': s.strength,
                'entry_price': s.entry_price,
                'exit_price': s.exit_price,
                'metadata': s.metadata
            }
            for s in signals
        ])
        
        # Add time stops
        updated_signals = []
        current_position = 0
        entry_time = None
        
        for i, row in signal_df.iterrows():
            timestamp = row['timestamp']
            signal_type = row['signal_type']
            
            if signal_type != SignalType.NEUTRAL:
                # Entry signal
                current_position = signal_type.value
                entry_time = timestamp
                updated_signals.append(signals[i])
                
            elif current_position != 0:
                # Check for time stop
                if entry_time and (timestamp - entry_time).days >= max_holding_periods:
                    # Time stop exit
                    updated_signals.append(Signal(
                        timestamp=timestamp,
                        signal_type=SignalType.NEUTRAL,
                        strength=0.0,
                        entry_price=signals[i].entry_price,
                        exit_price=signals[i].exit_price,
                        metadata={**(signals[i].metadata or {}), 'exit_reason': 'time_stop'}
                    ))
                    current_position = 0
                    entry_time = None
                else:
                    updated_signals.append(signals[i])
            else:
                updated_signals.append(signals[i])
        
        return updated_signals
    
    def _get_entry_price(
        self,
        timestamp: pd.Timestamp,
        y_price: Optional[pd.Series],
        x_price: Optional[pd.Series],
        hedge_ratio: float,
        signal_type: SignalType
    ) -> float:
        """Calculate entry price for a signal."""
        if y_price is None or x_price is None:
            return 0.0
        
        try:
            y_val = y_price.loc[timestamp]
            x_val = x_price.loc[timestamp]
            
            if signal_type == SignalType.LONG:
                # Long spread: short x, long y
                return y_val - hedge_ratio * x_val
            else:
                # Short spread: long x, short y
                return hedge_ratio * x_val - y_val
        except KeyError:
            return 0.0
    
    def _get_exit_price(
        self,
        timestamp: pd.Timestamp,
        y_price: Optional[pd.Series],
        x_price: Optional[pd.Series],
        hedge_ratio: float,
        position: int
    ) -> float:
        """Calculate exit price for a position."""
        if y_price is None or x_price is None:
            return 0.0
        
        try:
            y_val = y_price.loc[timestamp]
            x_val = x_price.loc[timestamp]
            
            if position == 1:  # Long spread
                return y_val - hedge_ratio * x_val
            else:  # Short spread
                return hedge_ratio * x_val - y_val
        except KeyError:
            return 0.0
    
    def validate_signals(self, signals: List[Signal]) -> List[Signal]:
        """Validate and clean signals."""
        if not signals:
            return signals
        
        valid_signals = []
        current_position = 0
        
        for signal in signals:
            # Check for valid signal type
            if signal.signal_type not in [SignalType.LONG, SignalType.SHORT, SignalType.NEUTRAL]:
                continue
            
            # Check position consistency
            if signal.signal_type == SignalType.NEUTRAL:
                if current_position == 0:
                    continue  # Skip neutral signals when no position
                current_position = 0
            else:
                if current_position != 0:
                    continue  # Skip entry signals when already in position
                current_position = signal.signal_type.value
            
            # Check for valid strength
            if signal.strength < 0 or signal.strength > self.max_position_size:
                continue
            
            valid_signals.append(signal)
        
        return valid_signals
