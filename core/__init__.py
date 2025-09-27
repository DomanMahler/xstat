"""
Core modules for xstat platform.
"""

from .data import DataFeed, BarData, MicrostructureData
from .stat_tests import CointegrationTests
from .signals import SignalGenerator
from .backtest import Backtester
from .metrics import PerformanceMetrics

__all__ = [
    "DataFeed",
    "BarData",
    "MicrostructureData", 
    "CointegrationTests",
    "SignalGenerator",
    "Backtester",
    "PerformanceMetrics",
]
