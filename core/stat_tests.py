"""
Statistical tests for cointegration and stationarity.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings

warnings.filterwarnings("ignore")


class CointegrationTests:
    """Comprehensive cointegration and stationarity testing."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def adf_test(
        self, 
        series: pd.Series, 
        max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Augmented Dickey-Fuller test for stationarity.
        
        Returns:
            Dict with test statistic, p-value, critical values, and conclusion
        """
        result = adfuller(series, maxlag=max_lags, autolag='AIC')
        
        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'critical_values': {
                '1%': result[4]['1%'],
                '5%': result[4]['5%'],
                '10%': result[4]['10%']
            },
            'is_stationary': result[1] < self.significance_level,
            'lags_used': result[2],
            'n_observations': result[3]
        }
    
    def kpss_test(
        self, 
        series: pd.Series, 
        regression: str = 'c'
    ) -> Dict[str, Any]:
        """
        KPSS test for stationarity.
        
        Args:
            series: Time series to test
            regression: 'c' for constant, 'ct' for constant and trend
            
        Returns:
            Dict with test results
        """
        result = kpss(series, regression=regression)
        
        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[3],
            'is_stationary': result[1] > self.significance_level,
            'lags_used': result[2]
        }
    
    def engle_granger_test(
        self, 
        y: pd.Series, 
        x: pd.Series,
        max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Engle-Granger cointegration test.
        
        Args:
            y: Dependent variable
            x: Independent variable
            max_lags: Maximum lags for ADF test on residuals
            
        Returns:
            Dict with test results including hedge ratio
        """
        # Step 1: Estimate cointegrating relationship
        X = add_constant(x)
        model = OLS(y, X).fit()
        
        # Get residuals
        residuals = model.resid
        
        # Step 2: Test residuals for stationarity
        adf_result = self.adf_test(residuals, max_lags=max_lags)
        
        return {
            'hedge_ratio': model.params[1],  # Beta coefficient
            'intercept': model.params[0],    # Alpha coefficient
            'r_squared': model.rsquared,
            'residual_adf': adf_result,
            'is_cointegrated': adf_result['is_stationary'],
            'residuals': residuals,
            'model': model
        }
    
    def johansen_test(
        self, 
        data: pd.DataFrame, 
        det_order: int = -1,
        k_ar_diff: int = 1
    ) -> Dict[str, Any]:
        """
        Johansen cointegration test.
        
        Args:
            data: DataFrame with time series (columns are variables)
            det_order: Deterministic order (-1, 0, 1, 2)
            k_ar_diff: Number of lags in differences
            
        Returns:
            Dict with test results
        """
        result = coint_johansen(data, det_order, k_ar_diff)
        
        # Trace test
        trace_stat = result.lr1
        trace_crit = result.cvt
        trace_pval = getattr(result, 'p1', None)
        
        # Max eigenvalue test
        eigen_stat = result.lr2
        eigen_crit = getattr(result, 'cve', result.cvt)  # Use cvt if cve not available
        eigen_pval = getattr(result, 'p2', None)
        
        # Determine number of cointegrating relationships
        n_coint = 0
        for i in range(len(trace_stat)):
            if trace_stat[i] > trace_crit[i, 1]:  # 5% critical value
                n_coint = i + 1
        
        return {
            'trace_statistics': trace_stat,
            'trace_critical_values': trace_crit,
            'trace_p_values': trace_pval,
            'eigen_statistics': eigen_stat,
            'eigen_critical_values': eigen_crit,
            'eigen_p_values': eigen_pval,
            'n_cointegrating_relations': n_coint,
            'cointegrating_vectors': result.evec,
            'eigenvalues': result.eig
        }
    
    def half_life_estimation(self, spread: pd.Series) -> float:
        """
        Estimate half-life of mean reversion using AR(1) model.
        
        Args:
            spread: Time series of spread
            
        Returns:
            Half-life in periods
        """
        # Fit AR(1) model: spread_t = alpha + beta * spread_{t-1} + epsilon_t
        spread_lag = spread.shift(1).dropna()
        spread_current = spread.iloc[1:]
        
        if len(spread_current) < 2:
            return np.nan
        
        # OLS regression
        X = add_constant(spread_lag)
        model = OLS(spread_current, X).fit()
        
        # Half-life = -log(2) / log(beta)
        beta = model.params.iloc[1] if hasattr(model.params, 'iloc') else model.params[1]
        if beta >= 1 or beta <= 0:
            return np.inf
        
        half_life = -np.log(2) / np.log(beta)
        return half_life
    
    def hurst_exponent(self, series: pd.Series) -> float:
        """
        Calculate Hurst exponent to measure long-term memory.
        
        Args:
            series: Time series
            
        Returns:
            Hurst exponent (0.5 = random walk, >0.5 = trending, <0.5 = mean-reverting)
        """
        def _hurst_rs(series, n):
            """Calculate R/S statistic for given n."""
            mean = series.mean()
            Y = series - mean
            Z = Y.cumsum()
            R = Z.max() - Z.min()
            S = series.std()
            if S == 0:
                return 0
            return R / S
        
        n = len(series)
        rs_values = []
        n_values = []
        
        # Use different window sizes
        for i in range(2, n // 2):
            if n % i == 0:
                chunks = [series.iloc[j:j+i] for j in range(0, n, i)]
                rs_chunk = np.mean([_hurst_rs(chunk, i) for chunk in chunks])
                rs_values.append(rs_chunk)
                n_values.append(i)
        
        if len(rs_values) < 2:
            return 0.5
        
        # Linear regression: log(R/S) = H * log(n) + c
        log_rs = np.log(rs_values)
        log_n = np.log(n_values)
        
        if len(log_rs) < 2:
            return 0.5
        
        slope, _ = np.polyfit(log_n, log_rs, 1)
        return slope
    
    def comprehensive_test(
        self, 
        y: pd.Series, 
        x: pd.Series
    ) -> Dict[str, Any]:
        """
        Run comprehensive cointegration and stationarity tests.
        
        Args:
            y: First time series
            x: Second time series
            
        Returns:
            Dict with all test results
        """
        results = {}
        
        # Individual stationarity tests
        results['y_adf'] = self.adf_test(y)
        results['x_adf'] = self.adf_test(x)
        results['y_kpss'] = self.kpss_test(y)
        results['x_kpss'] = self.kpss_test(x)
        
        # Engle-Granger test
        results['engle_granger'] = self.engle_granger_test(y, x)
        
        # Johansen test
        data = pd.DataFrame({'y': y, 'x': x}).dropna()
        if len(data) > 10:  # Need sufficient data
            results['johansen'] = self.johansen_test(data)
        else:
            results['johansen'] = {'error': 'Insufficient data for Johansen test'}
        
        # Half-life estimation
        if results['engle_granger']['is_cointegrated']:
            residuals = results['engle_granger']['residuals']
            results['half_life'] = self.half_life_estimation(residuals)
        else:
            results['half_life'] = np.nan
        
        # Hurst exponents
        results['y_hurst'] = self.hurst_exponent(y)
        results['x_hurst'] = self.hurst_exponent(x)
        
        # Overall assessment
        results['assessment'] = self._assess_cointegration(results)
        
        return results
    
    def _assess_cointegration(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess cointegration based on test results."""
        assessment = {
            'is_cointegrated': False,
            'confidence': 'low',
            'recommendations': []
        }
        
        # Check Engle-Granger
        eg_result = results['engle_granger']
        if eg_result['is_cointegrated'] and eg_result['r_squared'] > 0.5:
            assessment['is_cointegrated'] = True
            assessment['confidence'] = 'medium'
        
        # Check Johansen if available
        if 'johansen' in results and 'error' not in results['johansen']:
            johansen_result = results['johansen']
            if johansen_result['n_cointegrating_relations'] > 0:
                assessment['is_cointegrated'] = True
                if assessment['confidence'] == 'medium':
                    assessment['confidence'] = 'high'
        
        # Check half-life
        half_life = results.get('half_life', np.nan)
        if not np.isnan(half_life) and half_life < 30:  # Less than 30 periods
            assessment['recommendations'].append('Good mean reversion speed')
        elif not np.isnan(half_life) and half_life > 100:
            assessment['recommendations'].append('Slow mean reversion - consider longer holding periods')
        
        # Check Hurst exponents
        y_hurst = results.get('y_hurst', 0.5)
        x_hurst = results.get('x_hurst', 0.5)
        if y_hurst < 0.5 and x_hurst < 0.5:
            assessment['recommendations'].append('Both series show mean-reverting behavior')
        elif y_hurst > 0.5 or x_hurst > 0.5:
            assessment['recommendations'].append('One or both series show trending behavior')
        
        return assessment
