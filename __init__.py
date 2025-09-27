"""
xstat - Cross-Asset Statistical Arbitrage Research Platform

A production-quality platform for discovering cointegrated baskets,
generating signals, backtesting with realistic frictions, and
auto-producing research reports.
"""

__version__ = "0.1.0"
__author__ = "xstat"
__email__ = "xstat@example.com"

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
