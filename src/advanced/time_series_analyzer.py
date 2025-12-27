"""
Time-Series Specific Features Module

Features:
- Seasonality detection
- Trend analysis
- Stationarity testing (ADF, KPSS)
- Temporal gaps detection
- Look-ahead bias checking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


@dataclass
class SeasonalityResult:
    """Container for seasonality detection result"""
    has_seasonality: bool
    period: Optional[int]
    strength: float
    pattern_type: str  # 'daily', 'weekly', 'monthly', 'yearly', 'custom'


@dataclass
class StationarityResult:
    """Container for stationarity test result"""
    is_stationary: bool
    adf_statistic: float
    adf_pvalue: float
    critical_values: Dict[str, float]
    recommendation: str


class TimeSeriesAnalyzer:
    """
    Time-series specific data quality analysis.
    
    Features:
    - Seasonality detection
    - Trend analysis
    - Stationarity testing (ADF, KPSS approximation)
    - Temporal gaps detection
    - Look-ahead bias checking
    - Autocorrelation analysis
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize the analyzer.
        
        Args:
            significance_level: Significance level for statistical tests
        """
        self.significance_level = significance_level
        self.results_ = None
        
    def analyze(
        self,
        data: pd.DataFrame,
        datetime_column: str,
        value_columns: List[str] = None,
        target_column: str = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive time-series analysis.
        
        Args:
            data: DataFrame with time-series data
            datetime_column: Column containing timestamps
            value_columns: Columns to analyze (default: all numeric)
            target_column: Target column for look-ahead bias check
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            'summary': {},
            'temporal_coverage': {},
            'gaps_analysis': {},
            'seasonality': {},
            'trend_analysis': {},
            'stationarity': {},
            'autocorrelation': {},
            'look_ahead_bias': {}
        }
        
        # Ensure datetime column is datetime type
        data = data.copy()
        data[datetime_column] = pd.to_datetime(data[datetime_column], errors='coerce')
        data = data.sort_values(datetime_column)
        
        # Get value columns
        if value_columns is None:
            value_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Temporal coverage analysis
        results['temporal_coverage'] = self._analyze_temporal_coverage(
            data, datetime_column
        )
        
        # Gaps analysis
        results['gaps_analysis'] = self._analyze_gaps(data, datetime_column)
        
        # Analyze each value column
        for col in value_columns:
            if col == datetime_column:
                continue
                
            series = data[col].dropna()
            if len(series) < 10:
                continue
            
            # Seasonality detection
            results['seasonality'][col] = self._detect_seasonality(
                data, datetime_column, col
            )
            
            # Trend analysis
            results['trend_analysis'][col] = self._analyze_trend(series)
            
            # Stationarity tests
            results['stationarity'][col] = self._test_stationarity(series)
            
            # Autocorrelation
            results['autocorrelation'][col] = self._analyze_autocorrelation(series)
        
        # Look-ahead bias check
        if target_column and target_column in data.columns:
            results['look_ahead_bias'] = self._check_look_ahead_bias(
                data, datetime_column, target_column, value_columns
            )
        
        # Summary
        results['summary'] = self._generate_summary(results)
        
        self.results_ = results
        return results
    
    def _analyze_temporal_coverage(
        self,
        data: pd.DataFrame,
        datetime_column: str
    ) -> Dict[str, Any]:
        """Analyze temporal coverage of the data"""
        dt_series = data[datetime_column].dropna()
        
        if len(dt_series) == 0:
            return {'error': 'No valid datetime values'}
        
        return {
            'start_date': str(dt_series.min()),
            'end_date': str(dt_series.max()),
            'total_records': len(dt_series),
            'date_range_days': (dt_series.max() - dt_series.min()).days,
            'unique_dates': dt_series.dt.date.nunique(),
            'records_per_day': len(dt_series) / max((dt_series.max() - dt_series.min()).days, 1),
            'most_common_hour': int(dt_series.dt.hour.mode().iloc[0]) if len(dt_series.dt.hour.mode()) > 0 else None,
            'most_common_day_of_week': int(dt_series.dt.dayofweek.mode().iloc[0]) if len(dt_series.dt.dayofweek.mode()) > 0 else None
        }
    
    def _analyze_gaps(
        self,
        data: pd.DataFrame,
        datetime_column: str
    ) -> Dict[str, Any]:
        """Detect and analyze temporal gaps"""
        dt_series = data[datetime_column].dropna().sort_values()
        
        if len(dt_series) < 2:
            return {'error': 'Insufficient data for gap analysis'}
        
        # Calculate time differences
        time_diffs = dt_series.diff().dropna()
        
        # Detect expected frequency
        median_diff = time_diffs.median()
        
        # Find gaps (significantly larger than median)
        gap_threshold = median_diff * 2
        gaps = time_diffs[time_diffs > gap_threshold]
        
        gap_details = []
        for idx in gaps.index[:20]:  # Top 20 gaps
            gap_start_idx = dt_series.index.get_loc(idx) - 1
            if gap_start_idx >= 0:
                gap_start = dt_series.iloc[gap_start_idx]
                gap_end = dt_series.loc[idx]
                gap_details.append({
                    'start': str(gap_start),
                    'end': str(gap_end),
                    'duration': str(gaps.loc[idx]),
                    'duration_seconds': gaps.loc[idx].total_seconds()
                })
        
        return {
            'expected_frequency': str(median_diff),
            'total_gaps_detected': len(gaps),
            'gaps_percentage': len(gaps) / len(time_diffs) * 100,
            'largest_gap': str(time_diffs.max()),
            'smallest_gap': str(time_diffs.min()),
            'gap_details': sorted(gap_details, key=lambda x: x['duration_seconds'], reverse=True)
        }
    
    def _detect_seasonality(
        self,
        data: pd.DataFrame,
        datetime_column: str,
        value_column: str
    ) -> Dict[str, Any]:
        """Detect seasonality patterns"""
        df = data[[datetime_column, value_column]].dropna()
        
        if len(df) < 30:
            return {'has_seasonality': False, 'reason': 'Insufficient data'}
        
        series = df[value_column].values
        
        # Calculate autocorrelation for common periods
        periods_to_check = {
            'hourly': 24,
            'daily': 7,
            'weekly': 4,
            'monthly': 12,
            'quarterly': 4,
            'yearly': 1
        }
        
        seasonality_scores = {}
        
        for name, period in periods_to_check.items():
            if len(series) > period * 3:
                # Calculate autocorrelation at this lag
                autocorr = self._autocorrelation(series, period)
                seasonality_scores[name] = {
                    'period': period,
                    'autocorrelation': float(autocorr),
                    'significant': abs(autocorr) > 0.3
                }
        
        # Find strongest seasonality
        if seasonality_scores:
            strongest = max(
                seasonality_scores.items(),
                key=lambda x: abs(x[1]['autocorrelation'])
            )
            has_seasonality = strongest[1]['significant']
        else:
            has_seasonality = False
            strongest = (None, {'autocorrelation': 0})
        
        return {
            'has_seasonality': has_seasonality,
            'strongest_pattern': strongest[0] if has_seasonality else None,
            'strength': float(abs(strongest[1]['autocorrelation'])),
            'all_patterns': seasonality_scores
        }
    
    def _autocorrelation(self, series: np.ndarray, lag: int) -> float:
        """Calculate autocorrelation at specific lag"""
        n = len(series)
        if lag >= n:
            return 0
        
        mean = np.mean(series)
        var = np.var(series)
        
        if var == 0:
            return 0
        
        autocov = np.sum((series[:-lag] - mean) * (series[lag:] - mean)) / n
        return autocov / var
    
    def _analyze_trend(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze trend in the time series"""
        n = len(series)
        x = np.arange(n)
        
        # Linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
        
        # Determine trend direction
        if p_value > self.significance_level:
            trend_direction = 'no_significant_trend'
        elif slope > 0:
            trend_direction = 'upward'
        else:
            trend_direction = 'downward'
        
        # Calculate trend strength
        trend_strength = abs(r_value)
        
        # Rolling mean for visual trend
        if n > 10:
            rolling_mean = series.rolling(window=min(n // 5, 30), min_periods=1).mean()
            trend_values = {
                'start': float(rolling_mean.iloc[0]),
                'end': float(rolling_mean.iloc[-1]),
                'change': float(rolling_mean.iloc[-1] - rolling_mean.iloc[0])
            }
        else:
            trend_values = {
                'start': float(series.iloc[0]),
                'end': float(series.iloc[-1]),
                'change': float(series.iloc[-1] - series.iloc[0])
            }
        
        return {
            'direction': trend_direction,
            'slope': float(slope),
            'strength': float(trend_strength),
            'p_value': float(p_value),
            'std_error': float(std_err),
            'trend_values': trend_values,
            'percentage_change': float((trend_values['change'] / abs(trend_values['start']) * 100)) 
                if trend_values['start'] != 0 else 0
        }
    
    def _test_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """
        Test stationarity using augmented Dickey-Fuller test approximation.
        
        Note: This is a simplified implementation. For production use,
        consider using statsmodels.tsa.stattools.adfuller.
        """
        n = len(series)
        
        if n < 20:
            return {
                'is_stationary': None,
                'reason': 'Insufficient data for stationarity test'
            }
        
        # Simple ADF approximation using first-order difference regression
        y = series.values
        dy = np.diff(y)
        y_lag = y[:-1]
        
        # Regression: dy = alpha + beta * y_lag + error
        X = np.column_stack([np.ones(len(y_lag)), y_lag])
        
        try:
            beta, residuals, rank, s = np.linalg.lstsq(X, dy, rcond=None)
            
            # Calculate t-statistic for beta[1] (coefficient on y_lag)
            if len(residuals) > 0:
                mse = residuals[0] / (len(dy) - 2)
                var_beta = mse * np.linalg.inv(X.T @ X)[1, 1]
                t_stat = beta[1] / np.sqrt(var_beta)
            else:
                # Fallback calculation
                y_pred = X @ beta
                residuals_vec = dy - y_pred
                mse = np.sum(residuals_vec ** 2) / (len(dy) - 2)
                var_beta = mse * np.linalg.inv(X.T @ X)[1, 1]
                t_stat = beta[1] / np.sqrt(var_beta)
            
            # Critical values for ADF test (approximate)
            critical_values = {
                '1%': -3.43,
                '5%': -2.86,
                '10%': -2.57
            }
            
            # Determine stationarity
            is_stationary = t_stat < critical_values['5%']
            
            # Calculate approximate p-value
            if t_stat < -4:
                p_value = 0.001
            elif t_stat < -3.5:
                p_value = 0.01
            elif t_stat < -3:
                p_value = 0.025
            elif t_stat < -2.5:
                p_value = 0.05
            elif t_stat < -2:
                p_value = 0.1
            else:
                p_value = 0.5
            
            return {
                'is_stationary': is_stationary,
                'adf_statistic': float(t_stat),
                'p_value': float(p_value),
                'critical_values': critical_values,
                'recommendation': 'Series is stationary' if is_stationary 
                    else 'Consider differencing or transformation'
            }
            
        except Exception as e:
            return {
                'is_stationary': None,
                'error': str(e)
            }
    
    def _analyze_autocorrelation(
        self,
        series: pd.Series,
        max_lag: int = 20
    ) -> Dict[str, Any]:
        """Analyze autocorrelation structure"""
        n = len(series)
        max_lag = min(max_lag, n // 3)
        
        if max_lag < 2:
            return {'error': 'Insufficient data for autocorrelation analysis'}
        
        autocorrelations = []
        for lag in range(1, max_lag + 1):
            autocorr = self._autocorrelation(series.values, lag)
            autocorrelations.append({
                'lag': lag,
                'autocorrelation': float(autocorr),
                'significant': abs(autocorr) > 1.96 / np.sqrt(n)
            })
        
        # Find significant lags
        significant_lags = [
            ac['lag'] for ac in autocorrelations 
            if ac['significant']
        ]
        
        # Ljung-Box approximation
        q_stat = n * (n + 2) * sum(
            ac['autocorrelation'] ** 2 / (n - ac['lag'])
            for ac in autocorrelations[:10]
        )
        
        return {
            'autocorrelations': autocorrelations[:10],
            'significant_lags': significant_lags[:10],
            'first_significant_lag': significant_lags[0] if significant_lags else None,
            'ljung_box_statistic': float(q_stat),
            'has_significant_autocorrelation': len(significant_lags) > 0
        }
    
    def _check_look_ahead_bias(
        self,
        data: pd.DataFrame,
        datetime_column: str,
        target_column: str,
        feature_columns: List[str]
    ) -> Dict[str, Any]:
        """Check for look-ahead bias in features"""
        results = {
            'potential_leakage_features': [],
            'temporal_correlation_issues': [],
            'warnings': []
        }
        
        df = data.sort_values(datetime_column).reset_index(drop=True)
        
        for col in feature_columns:
            if col == target_column or col == datetime_column:
                continue
            
            # Check 1: Future correlation
            # Shift target forward and check correlation
            try:
                forward_corr = df[col].corr(df[target_column].shift(-1))
                
                if not pd.isna(forward_corr) and abs(forward_corr) > 0.8:
                    results['potential_leakage_features'].append({
                        'feature': col,
                        'forward_correlation': float(forward_corr),
                        'risk': 'high',
                        'reason': 'High correlation with future target values'
                    })
            except Exception:
                pass
            
            # Check 2: Perfect correlation with current target
            try:
                current_corr = df[col].corr(df[target_column])
                
                if not pd.isna(current_corr) and abs(current_corr) > 0.95:
                    results['warnings'].append({
                        'feature': col,
                        'correlation': float(current_corr),
                        'warning': 'Near-perfect correlation with target - possible leakage'
                    })
            except Exception:
                pass
        
        results['has_potential_leakage'] = len(results['potential_leakage_features']) > 0
        
        return results
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary"""
        summary = {
            'temporal_coverage': results.get('temporal_coverage', {}).get('date_range_days', 0),
            'gaps_detected': results.get('gaps_analysis', {}).get('total_gaps_detected', 0),
            'columns_analyzed': len(results.get('seasonality', {})),
            'columns_with_seasonality': sum(
                1 for v in results.get('seasonality', {}).values()
                if v.get('has_seasonality', False)
            ),
            'columns_with_trend': sum(
                1 for v in results.get('trend_analysis', {}).values()
                if v.get('direction') != 'no_significant_trend'
            ),
            'stationary_columns': sum(
                1 for v in results.get('stationarity', {}).values()
                if v.get('is_stationary', False)
            ),
            'look_ahead_bias_risk': results.get('look_ahead_bias', {}).get('has_potential_leakage', False)
        }
        
        # Overall quality score
        quality_score = 100
        
        if summary['gaps_detected'] > 0:
            gap_penalty = min(summary['gaps_detected'] * 2, 20)
            quality_score -= gap_penalty
        
        if summary['look_ahead_bias_risk']:
            quality_score -= 30
        
        summary['quality_score'] = max(quality_score, 0)
        
        return summary
    
    def generate_report(self) -> str:
        """Generate human-readable time series report"""
        if self.results_ is None:
            return "No analysis results available. Run analyze() first."
        
        report = []
        report.append("=" * 60)
        report.append("TIME SERIES QUALITY REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        summary = self.results_['summary']
        report.append(f"Quality Score: {summary['quality_score']}/100")
        report.append(f"Temporal Coverage: {summary['temporal_coverage']} days")
        report.append(f"Gaps Detected: {summary['gaps_detected']}")
        report.append(f"Columns Analyzed: {summary['columns_analyzed']}")
        report.append(f"Look-Ahead Bias Risk: {'⚠️ YES' if summary['look_ahead_bias_risk'] else '✅ No'}")
        report.append("")
        
        # Temporal Coverage
        coverage = self.results_['temporal_coverage']
        report.append("-" * 60)
        report.append("TEMPORAL COVERAGE")
        report.append("-" * 60)
        report.append(f"Start: {coverage.get('start_date', 'N/A')}")
        report.append(f"End: {coverage.get('end_date', 'N/A')}")
        report.append(f"Records: {coverage.get('total_records', 'N/A')}")
        report.append("")
        
        # Gaps
        gaps = self.results_['gaps_analysis']
        if gaps.get('total_gaps_detected', 0) > 0:
            report.append("-" * 60)
            report.append("TEMPORAL GAPS")
            report.append("-" * 60)
            report.append(f"Total Gaps: {gaps['total_gaps_detected']}")
            report.append(f"Largest Gap: {gaps['largest_gap']}")
            for gap in gaps.get('gap_details', [])[:5]:
                report.append(f"  • {gap['start']} to {gap['end']} ({gap['duration']})")
            report.append("")
        
        # Stationarity
        report.append("-" * 60)
        report.append("STATIONARITY ANALYSIS")
        report.append("-" * 60)
        for col, result in self.results_['stationarity'].items():
            status = "✅ Stationary" if result.get('is_stationary') else "⚠️ Non-stationary"
            report.append(f"{col}: {status}")
            if 'adf_statistic' in result:
                report.append(f"  ADF Statistic: {result['adf_statistic']:.4f}")
        
        return "\n".join(report)

