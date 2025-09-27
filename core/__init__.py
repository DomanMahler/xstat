"""
Core modules for xstat platform.
"""

from xstat.core.data import DataFeed, BarData, MicrostructureData
from xstat.core.stat_tests import CointegrationTests
from xstat.core.signals import SignalGenerator
from xstat.core.backtest import Backtester
from xstat.core.metrics import PerformanceMetrics

__all__ = [
    "DataFeed",
    "BarData",
    "MicrostructureData", 
    "CointegrationTests",
    "SignalGenerator",
    "Backtester",
    "PerformanceMetrics",
]
