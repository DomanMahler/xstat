# XSTAT - Cross-Asset Statistical Arbitrage Research Platform

A comprehensive platform for automated cointegration scanning, mean-reversion signal generation, event-driven backtesting with realistic frictions, and auto-generated PDF/HTML performance reports.

## 🚀 Quick Start

### 1. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/DomanMahler/xstat.git
cd xstat

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

#### Option A: Command Line Interface (Recommended)
```bash
# 1. Pull market data
python -m xstat.cli data pull --source yfinance --universe sp100 --start 2020-01-01 --end 2023-01-01

# 2. Scan for trading pairs
python -m xstat.cli research scan --config config/base.yaml

# 3. Run full research pipeline
python -m xstat.cli research run --config config/base.yaml

# 4. Generate reports
python -m xstat.cli report build --run-dir runs/latest

# 5. Launch web dashboard
python -m xstat.cli web serve --host 0.0.0.0 --port 8000
```

#### Option B: Web Dashboard (Interactive)
```bash
# Start the web interface
python -m xstat.cli web serve

# Open browser to http://localhost:8000
```

#### Option C: Python API
```python
from xstat.research.pipeline import ResearchPipeline, load_config

# Load configuration
config = load_config("config/base.yaml")

# Create and run pipeline
pipeline = ResearchPipeline(config)
results = await pipeline.run_full_pipeline()
```

## 📊 Features

### Data Management
- **Multiple Data Sources**: YFinance, CCXT (crypto), Parquet files
- **Automated Data Pulling**: Batch download with error handling
- **Data Validation**: Quality checks and cleaning

### Research Pipeline
- **Pair Scanning**: Automated cointegration testing
- **Statistical Tests**: ADF, KPSS, Johansen cointegration
- **Signal Generation**: Mean-reversion signals with thresholds
- **Risk Management**: Position sizing and stop-losses

### Backtesting
- **Event-Driven**: Realistic trade execution
- **Transaction Costs**: Bid-ask spreads, commissions
- **Performance Metrics**: Sharpe ratio, max drawdown, etc.

### Reporting
- **HTML Reports**: Interactive dashboards
- **PDF Reports**: Professional research documents
- **Web Interface**: Real-time monitoring

## 🛠️ Configuration

### Base Configuration (`config/base.yaml`)
```yaml
universe: "sp100"           # Asset universe
start_date: "2020-01-01"   # Research start date
end_date: "2023-01-01"     # Research end date
timeframe: "1d"            # Data frequency
min_lookback: 252          # Minimum data points
max_pairs: 50             # Maximum pairs to analyze
```

### Crypto Configuration (`config/pairs_crypto.yaml`)
```yaml
universe: "crypto_top50"
exchanges: ["binance", "coinbase"]
min_volume: 1000000
```

## 📈 Usage Examples

### 1. Equity Pairs Research
```bash
# Pull S&P 100 data
python -m xstat.cli data pull --source yfinance --universe sp100

# Scan for pairs
python -m xstat.cli research scan --config config/base.yaml

# View results
python -m xstat.cli data list
```

### 2. Crypto Pairs Research
```bash
# Pull crypto data
python -m xstat.cli data pull --source ccxt --universe crypto_top50

# Scan crypto pairs
python -m xstat.cli research scan --config config/pairs_crypto.yaml
```

### 3. Web Dashboard
```bash
# Start web server
python -m xstat.cli web serve --port 8000

# Access dashboard at http://localhost:8000
```

## 🔧 Development

### Project Structure
```
xstat/
├── cli.py                 # Command-line interface
├── core/                  # Core statistical functions
│   ├── backtest.py        # Backtesting engine
│   ├── data.py           # Data management
│   ├── features.py       # Feature engineering
│   ├── metrics.py        # Performance metrics
│   ├── risk.py           # Risk management
│   ├── signals.py        # Signal generation
│   └── stat_tests.py     # Statistical tests
├── datafeeds/            # Data source adapters
│   ├── ccxt_feed.py      # Crypto data
│   ├── parquet_feed.py   # Local data storage
│   └── yfinance_feed.py  # Equity data
├── research/             # Research pipeline
│   ├── pipeline.py       # Main research pipeline
│   └── selection.py      # Pair selection
├── report/               # Report generation
│   └── build.py          # Report builder
└── web/                  # Web dashboard
    └── app.py            # FastAPI application
```

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black xstat/
flake8 xstat/
```

## 📋 Requirements

- Python 3.8+
- 4GB+ RAM (for large datasets)
- Internet connection (for data feeds)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

- GitHub Issues: [Report bugs and request features](https://github.com/DomanMahler/xstat/issues)
- Documentation: [Full documentation](https://github.com/DomanMahler/xstat/wiki)

---

**XSTAT** - Statistical Arbitrage Made Simple 🎯