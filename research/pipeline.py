"""
End-to-end research pipeline for statistical arbitrage.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import date, datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
import yaml
import os

from xstat.core.data import DataManager
from xstat.datafeeds.yfinance_feed import YFinanceFeed
from xstat.datafeeds.ccxt_feed import CCXTFeed
from xstat.datafeeds.parquet_feed import ParquetFeed
from xstat.research.selection import PairSelector, PairCandidate
from xstat.core.stat_tests import CointegrationTests
from xstat.core.features import FeatureEngineer
from xstat.core.signals import SignalGenerator
from xstat.core.backtest import Backtester
from xstat.core.metrics import PerformanceMetrics


@dataclass
class ResearchConfig:
    """Configuration for research pipeline."""
    universe: str
    start_date: date
    end_date: date
    timeframe: str = "1d"
    min_lookback: int = 252
    max_pairs: int = 50
    train_ratio: float = 0.6
    test_ratio: float = 0.2
    validation_ratio: float = 0.2
    seed: int = 42


class ResearchPipeline:
    """End-to-end research pipeline."""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.data_manager = DataManager()
        self.pair_selector = PairSelector()
        self.cointegration_tests = CointegrationTests()
        self.feature_engineer = FeatureEngineer()
        self.signal_generator = SignalGenerator()
        self.backtester = Backtester()
        self.metrics_calculator = PerformanceMetrics()
        
        # Initialize data feeds
        self._setup_data_feeds()
    
    def _setup_data_feeds(self):
        """Setup data feeds."""
        self.data_manager.register_feed("yfinance", YFinanceFeed())
        self.data_manager.register_feed("ccxt", CCXTFeed())
        self.data_manager.register_feed("parquet", ParquetFeed())
    
    async def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete research pipeline."""
        print("Starting research pipeline...")
        
        # Step 1: Data collection
        print("Step 1: Collecting data...")
        price_data = await self._collect_data()
        
        if not price_data:
            return {"error": "No data collected"}
        
        # Step 2: Pair selection
        print("Step 2: Selecting pairs...")
        candidates = await self._select_pairs(price_data)
        
        if not candidates:
            return {"error": "No valid pairs found"}
        
        # Step 3: Feature engineering
        print("Step 3: Engineering features...")
        features = await self._engineer_features(price_data, candidates)
        
        # Step 4: Signal generation
        print("Step 4: Generating signals...")
        signals = await self._generate_signals(price_data, candidates, features)
        
        # Step 5: Backtesting
        print("Step 5: Running backtests...")
        results = await self._run_backtests(price_data, signals)
        
        # Step 6: Performance analysis
        print("Step 6: Analyzing performance...")
        performance = await self._analyze_performance(results)
        
        return {
            "data_summary": self._summarize_data(price_data),
            "candidates": self._summarize_candidates(candidates),
            "features": features,
            "signals": signals,
            "backtest_results": results,
            "performance": performance
        }
    
    async def _collect_data(self) -> Dict[str, pd.DataFrame]:
        """Collect data for the universe."""
        # Get symbols for universe
        if self.config.universe.startswith("crypto"):
            feed_name = "ccxt"
        else:
            feed_name = "yfinance"
        
        symbols = await self.data_manager.feeds[feed_name].get_symbols(
            self.config.universe, self.config.start_date, self.config.end_date
        )
        
        # Collect data for each symbol
        price_data = {}
        for symbol in symbols[:20]:  # Limit to first 20 symbols for demo
            try:
                data = await self.data_manager.get_bars(
                    feed_name, symbol, self.config.timeframe,
                    self.config.start_date, self.config.end_date
                )
                if not data.empty:
                    price_data[symbol] = data
            except Exception as e:
                print(f"Error collecting data for {symbol}: {e}")
                continue
        
        return price_data
    
    async def _select_pairs(self, price_data: Dict[str, pd.DataFrame]) -> List[PairCandidate]:
        """Select pairs from price data."""
        symbols = list(price_data.keys())
        
        # Use pair selector to find candidates
        candidates = self.pair_selector.scan_pairs(
            price_data, symbols, self.config.min_lookback
        )
        
        # Rank and filter
        ranked_candidates = self.pair_selector.rank_pairs(
            candidates, self.config.max_pairs
        )
        
        return ranked_candidates
    
    async def _engineer_features(
        self, 
        price_data: Dict[str, pd.DataFrame], 
        candidates: List[PairCandidate]
    ) -> Dict[str, Any]:
        """Engineer features for selected pairs."""
        features = {}
        
        for candidate in candidates:
            try:
                y_data = price_data[candidate.symbol_y]
                x_data = price_data[candidate.symbol_x]
                
                # Align data
                common_index = y_data.index.intersection(x_data.index)
                y_aligned = y_data.loc[common_index]
                x_aligned = x_data.loc[common_index]
                
                # Create features
                pair_features = self.feature_engineer.create_pair_features(
                    y_aligned['close'], x_aligned['close'], candidate.hedge_ratio
                )
                
                features[f"{candidate.symbol_y}_{candidate.symbol_x}"] = pair_features
                
            except Exception as e:
                print(f"Error engineering features for {candidate.symbol_y}-{candidate.symbol_x}: {e}")
                continue
        
        return features
    
    async def _generate_signals(
        self, 
        price_data: Dict[str, pd.DataFrame], 
        candidates: List[PairCandidate],
        features: Dict[str, Any]
    ) -> Dict[str, List[Any]]:
        """Generate signals for pairs."""
        signals = {}
        
        for candidate in candidates:
            try:
                y_data = price_data[candidate.symbol_y]
                x_data = price_data[candidate.symbol_x]
                
                # Align data
                common_index = y_data.index.intersection(x_data.index)
                y_aligned = y_data.loc[common_index]
                x_aligned = x_data.loc[common_index]
                
                # Create spread
                spread = y_aligned['close'] - candidate.hedge_ratio * x_aligned['close']
                
                # Calculate z-score
                zscore = self.feature_engineer.rolling_zscore(spread)
                
                # Generate signals
                pair_signals = self.signal_generator.generate_zscore_signals(
                    spread, zscore, candidate.hedge_ratio,
                    y_aligned['close'], x_aligned['close']
                )
                
                signals[f"{candidate.symbol_y}_{candidate.symbol_x}"] = pair_signals
                
            except Exception as e:
                print(f"Error generating signals for {candidate.symbol_y}-{candidate.symbol_x}: {e}")
                continue
        
        return signals
    
    async def _run_backtests(
        self, 
        price_data: Dict[str, pd.DataFrame], 
        signals: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Run backtests for all pairs."""
        results = {}
        
        for pair_name, pair_signals in signals.items():
            if not pair_signals:
                continue
            
            try:
                # Extract symbols from pair name
                symbols = pair_name.split('_')
                if len(symbols) != 2:
                    continue
                
                symbol_y, symbol_x = symbols
                
                # Prepare price data for backtest
                backtest_data = {
                    symbol_y: price_data[symbol_y],
                    symbol_x: price_data[symbol_x]
                }
                
                # Run backtest
                backtest_result = self.backtester.run_backtest(
                    pair_signals, backtest_data
                )
                
                results[pair_name] = backtest_result
                
            except Exception as e:
                print(f"Error running backtest for {pair_name}: {e}")
                continue
        
        return results
    
    async def _analyze_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance across all pairs."""
        if not results:
            return {}
        
        # Aggregate performance metrics
        total_trades = 0
        total_pnl = 0.0
        total_costs = 0.0
        winning_pairs = 0
        
        pair_performance = {}
        
        for pair_name, result in results.items():
            if not result:
                continue
            
            total_trades += result.get('total_trades', 0)
            total_pnl += result.get('total_pnl', 0)
            total_costs += result.get('total_costs', 0)
            
            if result.get('net_pnl', 0) > 0:
                winning_pairs += 1
            
            pair_performance[pair_name] = {
                'total_trades': result.get('total_trades', 0),
                'net_pnl': result.get('net_pnl', 0),
                'return_pct': result.get('return_pct', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0),
                'max_drawdown': result.get('max_drawdown', 0),
                'win_rate': result.get('win_rate', 0)
            }
        
        # Calculate portfolio-level metrics
        portfolio_metrics = {
            'total_pairs': len(results),
            'winning_pairs': winning_pairs,
            'win_rate': winning_pairs / len(results) if results else 0,
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'total_costs': total_costs,
            'net_pnl': total_pnl - total_costs,
            'avg_trade_pnl': (total_pnl - total_costs) / total_trades if total_trades > 0 else 0
        }
        
        return {
            'portfolio_metrics': portfolio_metrics,
            'pair_performance': pair_performance
        }
    
    def _summarize_data(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Summarize collected data."""
        if not price_data:
            return {}
        
        summary = {
            'num_symbols': len(price_data),
            'symbols': list(price_data.keys()),
            'date_range': {},
            'data_quality': {}
        }
        
        for symbol, data in price_data.items():
            summary['date_range'][symbol] = {
                'start': data.index.min(),
                'end': data.index.max(),
                'num_periods': len(data)
            }
            
            # Data quality metrics
            summary['data_quality'][symbol] = {
                'missing_data': data.isnull().sum().sum(),
                'duplicates': data.index.duplicated().sum()
            }
        
        return summary
    
    def _summarize_candidates(self, candidates: List[PairCandidate]) -> Dict[str, Any]:
        """Summarize pair candidates."""
        if not candidates:
            return {}
        
        return {
            'num_candidates': len(candidates),
            'top_pairs': [
                {
                    'symbol_y': c.symbol_y,
                    'symbol_x': c.symbol_x,
                    'overall_score': c.overall_score,
                    'correlation': c.correlation,
                    'half_life': c.half_life
                }
                for c in candidates[:10]
            ],
            'score_distribution': {
                'min': min(c.overall_score for c in candidates),
                'max': max(c.overall_score for c in candidates),
                'mean': np.mean([c.overall_score for c in candidates])
            }
        }
    
    def save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """Save research results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary
        summary_file = os.path.join(output_dir, "research_summary.yaml")
        with open(summary_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        # Save pair candidates
        if 'candidates' in results:
            candidates_df = self.pair_selector.create_pair_summary(results['candidates'])
            candidates_file = os.path.join(output_dir, "candidates.csv")
            candidates_df.to_csv(candidates_file, index=False)
        
        # Save performance metrics
        if 'performance' in results:
            performance_file = os.path.join(output_dir, "performance.json")
            import json
            with open(performance_file, 'w') as f:
                json.dump(results['performance'], f, indent=2)
        
        print(f"Results saved to {output_dir}")


def load_config(config_path: str) -> ResearchConfig:
    """Load research configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return ResearchConfig(
        universe=config_data['universe'],
        start_date=datetime.strptime(config_data['start_date'], '%Y-%m-%d').date(),
        end_date=datetime.strptime(config_data['end_date'], '%Y-%m-%d').date(),
        timeframe=config_data.get('timeframe', '1d'),
        min_lookback=config_data.get('min_lookback', 252),
        max_pairs=config_data.get('max_pairs', 50),
        train_ratio=config_data.get('train_ratio', 0.6),
        test_ratio=config_data.get('test_ratio', 0.2),
        validation_ratio=config_data.get('validation_ratio', 0.2),
        seed=config_data.get('seed', 42)
    )
