"""
Command-line interface for xstat platform.
"""

import asyncio
from typing import Optional, List
from datetime import date, datetime
import typer
from pathlib import Path
import yaml
import json

from xstat.research.pipeline import ResearchPipeline, ResearchConfig, load_config
from xstat.core.data import DataManager
from xstat.datafeeds.yfinance_feed import YFinanceFeed
from xstat.datafeeds.ccxt_feed import CCXTFeed
from xstat.datafeeds.parquet_feed import ParquetFeed
from xstat.report.build import ReportBuilder
from xstat.web.app import create_app

app = typer.Typer(help="xstat - Cross-Asset Statistical Arbitrage Research Platform")


@app.command()
def data(
    action: str = typer.Argument(..., help="Action: pull, list, clean"),
    source: str = typer.Option("yfinance", help="Data source: yfinance, ccxt, parquet"),
    universe: str = typer.Option("sp100", help="Universe to pull data for"),
    start: str = typer.Option("2020-01-01", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option("2023-01-01", help="End date (YYYY-MM-DD)"),
    output_dir: str = typer.Option("data", help="Output directory for data")
):
    """Data management commands."""
    
    if action == "pull":
        asyncio.run(_pull_data(source, universe, start, end, output_dir))
    elif action == "list":
        _list_data(output_dir)
    elif action == "clean":
        _clean_data(output_dir)
    else:
        typer.echo(f"Unknown action: {action}")
        raise typer.Exit(1)


@app.command()
def research(
    action: str = typer.Argument(..., help="Action: scan, run"),
    config: str = typer.Option("config/base.yaml", help="Configuration file"),
    output_dir: str = typer.Option("runs", help="Output directory for results")
):
    """Research commands."""
    
    if action == "scan":
        asyncio.run(_scan_pairs(config, output_dir))
    elif action == "run":
        asyncio.run(_run_research(config, output_dir))
    else:
        typer.echo(f"Unknown action: {action}")
        raise typer.Exit(1)


@app.command()
def backtest(
    config: str = typer.Option("config/base.yaml", help="Configuration file"),
    output_dir: str = typer.Option("runs", help="Output directory for results"),
    pairs: Optional[str] = typer.Option(None, help="Specific pairs to backtest (comma-separated)")
):
    """Run backtests."""
    asyncio.run(_run_backtest(config, output_dir, pairs))


@app.command()
def report(
    action: str = typer.Argument(..., help="Action: build, list"),
    run_dir: str = typer.Option("runs/latest", help="Run directory"),
    format: str = typer.Option("html", help="Report format: html, pdf"),
    output_dir: str = typer.Option("reports", help="Output directory for reports")
):
    """Report generation commands."""
    
    if action == "build":
        _build_report(run_dir, format, output_dir)
    elif action == "list":
        _list_reports(output_dir)
    else:
        typer.echo(f"Unknown action: {action}")
        raise typer.Exit(1)


@app.command()
def web(
    action: str = typer.Argument(..., help="Action: serve, status"),
    host: str = typer.Option("localhost", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    data_dir: str = typer.Option("runs", help="Data directory for web interface")
):
    """Web dashboard commands."""
    
    if action == "serve":
        _serve_web(host, port, data_dir)
    elif action == "status":
        _web_status(host, port)
    else:
        typer.echo(f"Unknown action: {action}")
        raise typer.Exit(1)


async def _pull_data(source: str, universe: str, start: str, end: str, output_dir: str):
    """Pull data from specified source."""
    typer.echo(f"Pulling data from {source} for universe {universe}")
    
    # Parse dates
    start_date = datetime.strptime(start, '%Y-%m-%d').date()
    end_date = datetime.strptime(end, '%Y-%m-%d').date()
    
    # Setup data manager
    data_manager = DataManager()
    
    if source == "yfinance":
        data_manager.register_feed("yfinance", YFinanceFeed())
        feed_name = "yfinance"
    elif source == "ccxt":
        data_manager.register_feed("ccxt", CCXTFeed())
        feed_name = "ccxt"
    elif source == "parquet":
        data_manager.register_feed("parquet", ParquetFeed())
        feed_name = "parquet"
    else:
        typer.echo(f"Unknown source: {source}")
        raise typer.Exit(1)
    
    # Get symbols
    symbols = await data_manager.feeds[feed_name].get_symbols(universe, start_date, end_date)
    typer.echo(f"Found {len(symbols)} symbols")
    
    # Pull data for each symbol
    parquet_feed = ParquetFeed(output_dir)
    success_count = 0
    
    for symbol in symbols[:20]:  # Limit to first 20 for demo
        try:
            data = await data_manager.get_bars(feed_name, symbol, "1d", start_date, end_date)
            if not data.empty:
                parquet_feed.save_bars(symbol, data)
                success_count += 1
                typer.echo(f"✓ {symbol}: {len(data)} bars")
            else:
                typer.echo(f"✗ {symbol}: No data")
        except Exception as e:
            typer.echo(f"✗ {symbol}: Error - {e}")
    
    typer.echo(f"Successfully pulled data for {success_count} symbols")


def _list_data(data_dir: str):
    """List available data."""
    parquet_feed = ParquetFeed(data_dir)
    summary = parquet_feed.get_data_summary()
    
    typer.echo(f"Data Summary:")
    typer.echo(f"  Total files: {summary['total_files']}")
    typer.echo(f"  Total size: {summary['total_size'] / (1024*1024):.2f} MB")
    typer.echo(f"  Symbols: {len(summary['symbols'])}")
    
    if summary['symbols']:
        typer.echo(f"  Available symbols: {', '.join(summary['symbols'][:10])}")
        if len(summary['symbols']) > 10:
            typer.echo(f"  ... and {len(summary['symbols']) - 10} more")


def _clean_data(data_dir: str):
    """Clean old data files."""
    parquet_feed = ParquetFeed(data_dir)
    parquet_feed.cleanup_old_data(days_to_keep=30)
    typer.echo(f"Cleaned old data files in {data_dir}")


async def _scan_pairs(config_path: str, output_dir: str):
    """Scan for potential pairs."""
    typer.echo(f"Scanning pairs using config: {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup data manager
    data_manager = DataManager()
    data_manager.register_feed("parquet", ParquetFeed("data"))
    
    # Get available symbols
    symbols = await data_manager.feeds["parquet"].get_symbols(
        config.universe, config.start_date, config.end_date
    )
    
    if not symbols:
        typer.echo("No symbols found. Run 'xstat data pull' first.")
        raise typer.Exit(1)
    
    typer.echo(f"Found {len(symbols)} symbols")
    
    # Load price data
    price_data = {}
    for symbol in symbols[:20]:  # Limit for demo
        try:
            data = await data_manager.get_bars(
                "parquet", symbol, config.timeframe,
                config.start_date, config.end_date
            )
            if not data.empty:
                price_data[symbol] = data
        except Exception as e:
            typer.echo(f"Error loading {symbol}: {e}")
    
    if not price_data:
        typer.echo("No price data available")
        raise typer.Exit(1)
    
    # Scan for pairs
    from xstat.research.selection import PairSelector
    selector = PairSelector()
    
    candidates = selector.scan_pairs(price_data, list(price_data.keys()), config.min_lookback)
    ranked_candidates = selector.rank_pairs(candidates, config.max_pairs)
    
    # Save results
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save candidates CSV
    candidates_df = selector.create_pair_summary(ranked_candidates)
    candidates_file = os.path.join(output_dir, "candidates.csv")
    candidates_df.to_csv(candidates_file, index=False)
    
    typer.echo(f"Found {len(ranked_candidates)} valid pairs")
    typer.echo(f"Top 5 pairs:")
    for i, candidate in enumerate(ranked_candidates[:5]):
        typer.echo(f"  {i+1}. {candidate.symbol_y}-{candidate.symbol_x} (score: {candidate.overall_score:.3f})")
    
    typer.echo(f"Results saved to {output_dir}")


async def _run_research(config_path: str, output_dir: str):
    """Run full research pipeline."""
    typer.echo(f"Running research pipeline with config: {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Create pipeline
    pipeline = ResearchPipeline(config)
    
    # Run pipeline
    results = await pipeline.run_full_pipeline()
    
    if "error" in results:
        typer.echo(f"Pipeline failed: {results['error']}")
        raise typer.Exit(1)
    
    # Save results
    pipeline.save_results(results, output_dir)
    
    typer.echo("Research pipeline completed successfully!")
    typer.echo(f"Results saved to {output_dir}")


async def _run_backtest(config_path: str, output_dir: str, pairs: Optional[str]):
    """Run backtests."""
    typer.echo(f"Running backtests with config: {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    # TODO: Implement backtest logic
    typer.echo("Backtest functionality not yet implemented")


def _build_report(run_dir: str, format: str, output_dir: str):
    """Build research report."""
    typer.echo(f"Building {format} report from {run_dir}")
    
    # TODO: Implement report building
    typer.echo("Report building functionality not yet implemented")


def _list_reports(output_dir: str):
    """List available reports."""
    typer.echo(f"Available reports in {output_dir}:")
    # TODO: Implement report listing


def _serve_web(host: str, port: int, data_dir: str):
    """Serve web dashboard."""
    typer.echo(f"Starting web server on {host}:{port}")
    
    import uvicorn
    app = create_app(data_dir)
    uvicorn.run(app, host=host, port=port)


def _web_status(host: str, port: int):
    """Check web server status."""
    typer.echo(f"Checking web server status at {host}:{port}")
    # TODO: Implement status check


if __name__ == "__main__":
    app()
