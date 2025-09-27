#!/usr/bin/env python3
"""
XSTAT Demo - Shows the platform working with real data
"""

import asyncio
import sys
import os
from datetime import datetime, date
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from core.data import DataManager
from datafeeds.yfinance_feed import YFinanceFeed
from datafeeds.parquet_feed import ParquetFeed
from research.selection import PairSelector
from core.stat_tests import StatisticalTests
from core.signals import SignalGenerator

async def demo_xstat():
    """Demonstrate XSTAT platform working with real data."""
    print("🚀 XSTAT - Cross-Asset Statistical Arbitrage Platform")
    print("=" * 60)
    print()
    
    # Step 1: Setup data feeds
    print("📊 Setting up data feeds...")
    data_manager = DataManager()
    data_manager.register_feed("yfinance", YFinanceFeed())
    data_manager.register_feed("parquet", ParquetFeed("data"))
    print("   ✓ Data feeds configured")
    print()
    
    # Step 2: Pull sample data
    print("📈 Pulling market data for demo...")
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    start_date = date(2022, 1, 1)
    end_date = date(2023, 12, 31)
    
    price_data = {}
    for symbol in symbols:
        try:
            print(f"   Downloading {symbol}...")
            data = await data_manager.get_bars("yfinance", symbol, "1d", start_date, end_date)
            if not data.empty:
                price_data[symbol] = data
                print(f"   ✓ {symbol}: {len(data)} bars")
            else:
                print(f"   ✗ {symbol}: No data")
        except Exception as e:
            print(f"   ✗ {symbol}: Error - {e}")
    
    if not price_data:
        print("   ❌ No data available. Please check your internet connection")
        return
    
    print(f"   ✓ Successfully loaded data for {len(price_data)} symbols")
    print()
    
    # Step 3: Statistical Analysis
    print("🔬 Running statistical analysis...")
    stat_tests = StatisticalTests()
    
    # Calculate returns
    returns_data = {}
    for symbol, data in price_data.items():
        returns = data['close'].pct_change().dropna()
        returns_data[symbol] = returns
        print(f"   {symbol}: Mean return = {returns.mean():.4f}, Volatility = {returns.std():.4f}")
    
    print()
    
    # Step 4: Pair Analysis
    print("🔍 Analyzing potential trading pairs...")
    selector = PairSelector()
    
    # Find potential pairs
    candidates = []
    symbols_list = list(price_data.keys())
    
    for i, symbol1 in enumerate(symbols_list):
        for j, symbol2 in enumerate(symbols_list[i+1:], i+1):
            try:
                # Get overlapping data
                data1 = price_data[symbol1]
                data2 = price_data[symbol2]
                
                # Align dates
                common_dates = data1.index.intersection(data2.index)
                if len(common_dates) < 100:  # Need sufficient data
                    continue
                
                prices1 = data1.loc[common_dates, 'close']
                prices2 = data2.loc[common_dates, 'close']
                
                # Calculate correlation
                correlation = prices1.corr(prices2)
                
                # Simple cointegration test (price ratio)
                ratio = prices1 / prices2
                ratio_returns = ratio.pct_change().dropna()
                
                # Calculate mean reversion metrics
                z_score = (ratio - ratio.mean()) / ratio.std()
                mean_reversion_strength = abs(z_score).mean()
                
                if correlation > 0.7:  # High correlation threshold
                    candidates.append({
                        'pair': f"{symbol1}-{symbol2}",
                        'correlation': correlation,
                        'mean_reversion': mean_reversion_strength,
                        'score': correlation * (1 - mean_reversion_strength)
                    })
                    
            except Exception as e:
                continue
    
    # Sort by score
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"   Found {len(candidates)} potential pairs")
    print("   Top 3 pairs:")
    for i, candidate in enumerate(candidates[:3]):
        print(f"   {i+1}. {candidate['pair']}: Correlation={candidate['correlation']:.3f}, Score={candidate['score']:.3f}")
    
    print()
    
    # Step 5: Signal Generation
    print("📡 Generating trading signals...")
    signal_gen = SignalGenerator()
    
    if candidates:
        best_pair = candidates[0]
        symbol1, symbol2 = best_pair['pair'].split('-')
        
        # Get aligned data
        data1 = price_data[symbol1]
        data2 = price_data[symbol2]
        common_dates = data1.index.intersection(data2.index)
        prices1 = data1.loc[common_dates, 'close']
        prices2 = data2.loc[common_dates, 'close']
        
        # Generate signals
        ratio = prices1 / prices2
        signals = signal_gen.generate_mean_reversion_signals(ratio)
        
        print(f"   Best pair: {symbol1}-{symbol2}")
        print(f"   Generated {len(signals)} signals")
        print(f"   Signal distribution: {signals.value_counts().to_dict()}")
    
    print()
    
    # Step 6: Performance Summary
    print("📊 Performance Summary")
    print("-" * 30)
    print(f"Data period: {start_date} to {end_date}")
    print(f"Symbols analyzed: {len(price_data)}")
    print(f"Potential pairs found: {len(candidates)}")
    print(f"Best correlation: {candidates[0]['correlation']:.3f}" if candidates else "No pairs found")
    print()
    
    # Step 7: Save results
    print("💾 Saving results...")
    os.makedirs("runs", exist_ok=True)
    
    # Save price data
    for symbol, data in price_data.items():
        data.to_csv(f"runs/{symbol}_prices.csv")
    
    # Save pair analysis
    if candidates:
        candidates_df = pd.DataFrame(candidates)
        candidates_df.to_csv("runs/pair_analysis.csv", index=False)
        print("   ✓ Results saved to 'runs/' directory")
    
    print()
    print("🎉 XSTAT Demo Complete!")
    print("=" * 60)
    print("Next steps:")
    print("1. Check the 'runs/' directory for results")
    print("2. Run: python3 demo.py web (for web dashboard)")
    print("3. View pair analysis in 'runs/pair_analysis.csv'")

def run_web_demo():
    """Start web dashboard demo."""
    print("🌐 Starting XSTAT Web Dashboard...")
    print("=" * 40)
    
    try:
        from web.app import create_app
        import uvicorn
        
        app = create_app("runs")
        print("   Dashboard available at: http://localhost:8000")
        print("   Press Ctrl+C to stop")
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
    except Exception as e:
        print(f"   Error starting web dashboard: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        run_web_demo()
    else:
        print("XSTAT - Cross-Asset Statistical Arbitrage Platform")
        print("Choose an option:")
        print("1. Run analysis demo: python3 demo.py")
        print("2. Start web dashboard: python3 demo.py web")
        print()
        
        # Run the analysis demo
        asyncio.run(demo_xstat())
