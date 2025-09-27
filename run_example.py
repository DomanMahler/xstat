#!/usr/bin/env python3
"""
Example script to run the complete XSTAT pipeline.
This demonstrates how to run all the code at once.
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from xstat.research.pipeline import ResearchPipeline, load_config
from xstat.core.data import DataManager
from xstat.datafeeds.yfinance_feed import YFinanceFeed
from xstat.datafeeds.parquet_feed import ParquetFeed
from xstat.report.build import ReportBuilder


async def run_complete_pipeline():
    """Run the complete XSTAT research pipeline."""
    print("🚀 Starting XSTAT Complete Pipeline")
    print("=" * 50)
    
    # Step 1: Setup data manager
    print("📊 Setting up data feeds...")
    data_manager = DataManager()
    data_manager.register_feed("yfinance", YFinanceFeed())
    data_manager.register_feed("parquet", ParquetFeed("data"))
    
    # Step 2: Load configuration
    print("⚙️  Loading configuration...")
    config = load_config("config/base.yaml")
    print(f"   Universe: {config.universe}")
    print(f"   Date range: {config.start_date} to {config.end_date}")
    
    # Step 3: Pull sample data (limited for demo)
    print("📈 Pulling market data...")
    try:
        # Get a few sample symbols
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        for symbol in symbols:
            try:
                data = await data_manager.get_bars(
                    "yfinance", symbol, "1d", 
                    config.start_date, config.end_date
                )
                if not data.empty:
                    data_manager.feeds["parquet"].save_bars(symbol, data)
                    print(f"   ✓ {symbol}: {len(data)} bars")
                else:
                    print(f"   ✗ {symbol}: No data")
            except Exception as e:
                print(f"   ✗ {symbol}: Error - {e}")
    except Exception as e:
        print(f"   Error pulling data: {e}")
        return
    
    # Step 4: Run research pipeline
    print("\n🔬 Running research pipeline...")
    try:
        pipeline = ResearchPipeline(config)
        results = await pipeline.run_full_pipeline()
        
        if "error" in results:
            print(f"   Pipeline failed: {results['error']}")
            return
        
        print("   ✓ Research pipeline completed!")
        print(f"   Results saved to: {results.get('output_dir', 'runs/latest')}")
        
    except Exception as e:
        print(f"   Error in research pipeline: {e}")
        return
    
    # Step 5: Generate report
    print("\n📋 Generating report...")
    try:
        report_builder = ReportBuilder()
        report_path = report_builder.build_html_report(
            results.get('output_dir', 'runs/latest'),
            "reports"
        )
        print(f"   ✓ Report generated: {report_path}")
    except Exception as e:
        print(f"   Error generating report: {e}")
    
    print("\n🎉 XSTAT Pipeline Complete!")
    print("=" * 50)
    print("Next steps:")
    print("1. View results: python -m xstat.cli data list")
    print("2. Launch web dashboard: python -m xstat.cli web serve")
    print("3. Check reports in the 'reports' directory")


def run_web_dashboard():
    """Start the web dashboard."""
    print("🌐 Starting XSTAT Web Dashboard...")
    print("=" * 50)
    
    try:
        from xstat.web.app import create_app
        import uvicorn
        
        app = create_app("runs")
        print("   Dashboard available at: http://localhost:8000")
        print("   Press Ctrl+C to stop")
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
    except Exception as e:
        print(f"   Error starting web dashboard: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        run_web_dashboard()
    else:
        print("XSTAT - Cross-Asset Statistical Arbitrage Platform")
        print("Choose an option:")
        print("1. Run complete pipeline: python run_example.py")
        print("2. Start web dashboard: python run_example.py web")
        print("3. Use CLI commands: python -m xstat.cli --help")
        print()
        
        # Run the complete pipeline by default
        asyncio.run(run_complete_pipeline())
