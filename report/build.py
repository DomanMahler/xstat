"""
Report generation system for xstat platform.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import yaml
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Environment, FileSystemLoader
import weasyprint
from io import BytesIO
import base64


class ReportBuilder:
    """Generate research reports in HTML and PDF formats."""
    
    def __init__(self, template_dir: str = "xstat/report/templates"):
        self.template_dir = Path(template_dir)
        self.templates = Environment(loader=FileSystemLoader(str(self.template_dir)))
        
        # Set up matplotlib for better plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def build_report(
        self, 
        run_dir: str, 
        output_format: str = "html",
        output_dir: str = "reports"
    ) -> str:
        """
        Build research report.
        
        Args:
            run_dir: Directory containing research results
            output_format: Output format ('html' or 'pdf')
            output_dir: Output directory for report
            
        Returns:
            Path to generated report
        """
        run_path = Path(run_dir)
        if not run_path.exists():
            raise ValueError(f"Run directory not found: {run_dir}")
        
        # Load research data
        data = self._load_research_data(run_path)
        
        # Generate plots
        plots = self._generate_plots(data)
        
        # Build report
        if output_format == "html":
            report_path = self._build_html_report(data, plots, output_dir)
        elif output_format == "pdf":
            report_path = self._build_pdf_report(data, plots, output_dir)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return str(report_path)
    
    def _load_research_data(self, run_path: Path) -> Dict[str, Any]:
        """Load research data from run directory."""
        data = {}
        
        # Load summary
        summary_file = run_path / "research_summary.yaml"
        if summary_file.exists():
            with open(summary_file) as f:
                data['summary'] = yaml.safe_load(f)
        
        # Load candidates
        candidates_file = run_path / "candidates.csv"
        if candidates_file.exists():
            data['candidates'] = pd.read_csv(candidates_file)
        
        # Load performance
        performance_file = run_path / "performance.json"
        if performance_file.exists():
            with open(performance_file) as f:
                data['performance'] = json.load(f)
        
        # Load backtest results
        backtest_files = list(run_path.glob("backtest_*.json"))
        data['backtests'] = {}
        for file in backtest_files:
            with open(file) as f:
                data['backtests'][file.stem] = json.load(f)
        
        return data
    
    def _generate_plots(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Generate plots for the report."""
        plots = {}
        
        # Candidates score distribution
        if 'candidates' in data and not data['candidates'].empty:
            plots['score_distribution'] = self._plot_score_distribution(data['candidates'])
            plots['correlation_heatmap'] = self._plot_correlation_heatmap(data['candidates'])
        
        # Performance plots
        if 'performance' in data:
            plots['performance_summary'] = self._plot_performance_summary(data['performance'])
        
        # Backtest results
        if 'backtests' in data:
            plots['backtest_results'] = self._plot_backtest_results(data['backtests'])
        
        return plots
    
    def _plot_score_distribution(self, candidates: pd.DataFrame) -> str:
        """Plot score distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'overall_score' in candidates.columns:
            ax.hist(candidates['overall_score'], bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Overall Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Pair Scores')
            ax.grid(True, alpha=0.3)
        
        return self._fig_to_base64(fig)
    
    def _plot_correlation_heatmap(self, candidates: pd.DataFrame) -> str:
        """Plot correlation heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Select numeric columns for correlation
        numeric_cols = candidates.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = candidates[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Feature Correlation Matrix')
        
        return self._fig_to_base64(fig)
    
    def _plot_performance_summary(self, performance: Dict[str, Any]) -> str:
        """Plot performance summary."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        if 'pair_performance' in performance:
            pair_perf = performance['pair_performance']
            
            # Extract metrics
            returns = [pair.get('return_pct', 0) for pair in pair_perf.values()]
            sharpe_ratios = [pair.get('sharpe_ratio', 0) for pair in pair_perf.values()]
            max_drawdowns = [pair.get('max_drawdown', 0) for pair in pair_perf.values()]
            win_rates = [pair.get('win_rate', 0) for pair in pair_perf.values()]
            
            # Plot returns distribution
            axes[0, 0].hist(returns, bins=15, alpha=0.7, edgecolor='black')
            axes[0, 0].set_xlabel('Return (%)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Return Distribution')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot Sharpe ratios
            axes[0, 1].hist(sharpe_ratios, bins=15, alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Sharpe Ratio')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Sharpe Ratio Distribution')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot max drawdowns
            axes[1, 0].hist(max_drawdowns, bins=15, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Max Drawdown')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Max Drawdown Distribution')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot win rates
            axes[1, 1].hist(win_rates, bins=15, alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Win Rate')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Win Rate Distribution')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _plot_backtest_results(self, backtests: Dict[str, Any]) -> str:
        """Plot backtest results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract equity curves
        equity_curves = []
        for pair_name, backtest in backtests.items():
            if 'equity_curve' in backtest:
                equity_curves.append((pair_name, backtest['equity_curve']))
        
        # Plot equity curves
        for pair_name, equity_curve in equity_curves[:5]:  # Top 5 pairs
            if equity_curve:
                dates = [eq['date'] for eq in equity_curve]
                equity = [eq['equity'] for eq in equity_curve]
                axes[0, 0].plot(dates, equity, label=pair_name, alpha=0.7)
        
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Equity')
        axes[0, 0].set_title('Equity Curves (Top 5 Pairs)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot performance metrics
        metrics = ['total_trades', 'net_pnl', 'sharpe_ratio', 'max_drawdown']
        metric_values = {metric: [] for metric in metrics}
        
        for backtest in backtests.values():
            for metric in metrics:
                metric_values[metric].append(backtest.get(metric, 0))
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            ax.hist(metric_values[metric], bins=10, alpha=0.7, edgecolor='black')
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"
    
    def _build_html_report(
        self, 
        data: Dict[str, Any], 
        plots: Dict[str, str], 
        output_dir: str
    ) -> Path:
        """Build HTML report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load template
        template = self.templates.get_template("research_report.html")
        
        # Prepare template data
        template_data = {
            'data': data,
            'plots': plots,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'xstat_version': '0.1.0'
        }
        
        # Render template
        html_content = template.render(**template_data)
        
        # Save report
        report_path = output_path / f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path
    
    def _build_pdf_report(
        self, 
        data: Dict[str, Any], 
        plots: Dict[str, str], 
        output_dir: str
    ) -> Path:
        """Build PDF report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # First generate HTML
        html_path = self._build_html_report(data, plots, output_dir)
        
        # Convert to PDF
        pdf_path = html_path.with_suffix('.pdf')
        
        try:
            with open(html_path, 'r') as f:
                html_content = f.read()
            
            weasyprint.HTML(string=html_content).write_pdf(str(pdf_path))
            
            # Remove temporary HTML file
            html_path.unlink()
            
            return pdf_path
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return html_path  # Return HTML if PDF generation fails
