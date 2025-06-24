"""
Market Regime Detection for Helformer Trading System
Implements Hurst exponent-based regime detection to adapt trading strategies
based on market conditions (trending vs mean-reverting).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from config_helformer import config
import warnings

@dataclass
class RegimeClassification:
    """Container for regime classification results"""
    regime_type: str
    hurst_exponent: float
    confidence_score: float
    supporting_metrics: Dict[str, float]
    timestamp: pd.Timestamp
    regime_parameters: Dict[str, float]

class MarketRegimeDetector:
    """
    Market regime detection using Hurst exponent analysis.
    
    Classifies market conditions as:
    - trending_strong: H > 0.65 (strong persistence/momentum)
    - trending_weak: 0.55 < H <= 0.65 (weak persistence)
    - neutral: 0.45 <= H <= 0.55 (random walk)
    - mean_reverting: H < 0.45 (strong mean reversion)
    """
    
    def __init__(self, 
                 window_size: int = 100,
                 max_lag: int = 50,
                 min_periods: int = 30,
                 confidence_threshold: float = 0.7):
        """
        Initialize regime detector.
        
        Args:
            window_size: Rolling window for regime detection
            max_lag: Maximum lag for R/S analysis
            min_periods: Minimum periods required for calculation
            confidence_threshold: Minimum confidence for regime classification
        """
        self.window_size = window_size
        self.max_lag = max_lag
        self.min_periods = min_periods
        self.confidence_threshold = confidence_threshold
        
        # Regime thresholds (based on Hurst exponent literature)
        self.regime_thresholds = {
            'trending_strong': 0.65,
            'trending_weak': 0.55,
            'neutral_upper': 0.55,
            'neutral_lower': 0.45,
            'mean_reverting': 0.45
        }
        
        # Cache for performance
        self._hurst_cache = {}
        self._regime_history = []
        
    def calculate_hurst_exponent(self, 
                                price_series: pd.Series, 
                                max_lag: Optional[int] = None) -> Tuple[float, float, Dict[str, float]]:
        """
        Calculate Hurst exponent using Rescaled Range (R/S) analysis.
        
        The Hurst exponent H indicates:
        - H > 0.5: Trending/persistent behavior
        - H = 0.5: Random walk (efficient market)
        - H < 0.5: Mean-reverting/anti-persistent behavior
        
        Args:
            price_series: Price time series
            max_lag: Maximum lag for analysis (uses self.max_lag if None)
            
        Returns:
            Tuple of (hurst_exponent, r_squared, supporting_metrics)
        """
        if max_lag is None:
            max_lag = self.max_lag
            
        # Convert to numpy for performance
        prices = np.array(price_series)
        n = len(prices)
        
        if n < self.min_periods:
            return 0.5, 0.0, {'error': 'insufficient_data', 'n': n}
        
        # Calculate log returns
        log_returns = np.diff(np.log(prices))
        
        if len(log_returns) == 0:
            return 0.5, 0.0, {'error': 'no_returns', 'n': n}
        
        # Remove any infinite or NaN values
        log_returns = log_returns[np.isfinite(log_returns)]
        
        if len(log_returns) < 10:
            return 0.5, 0.0, {'error': 'insufficient_valid_returns', 'n': len(log_returns)}
        
        # Calculate R/S statistics for different lags
        lags = np.arange(2, min(max_lag + 1, len(log_returns) // 2))
        rs_values = []
        
        for lag in lags:
            rs = self._calculate_rs_statistic(log_returns, lag)
            if rs > 0:  # Valid R/S ratio
                rs_values.append(rs)
            else:
                rs_values.append(np.nan)
        
        # Filter out invalid R/S values
        valid_mask = np.isfinite(rs_values)
        if np.sum(valid_mask) < 3:  # Need at least 3 points for regression
            return 0.5, 0.0, {'error': 'insufficient_valid_rs', 'valid_points': np.sum(valid_mask)}
        
        valid_lags = lags[valid_mask]
        valid_rs = np.array(rs_values)[valid_mask]
        
        # Linear regression: log(R/S) = H * log(lag) + constant
        # H is the slope (Hurst exponent)
        try:
            log_lags = np.log(valid_lags)
            log_rs = np.log(valid_rs)
            
            # Remove any remaining infinite values from log transformation
            finite_mask = np.isfinite(log_lags) & np.isfinite(log_rs)
            if np.sum(finite_mask) < 3:
                return 0.5, 0.0, {'error': 'insufficient_finite_values', 'finite_points': np.sum(finite_mask)}
            
            log_lags = log_lags[finite_mask]
            log_rs = log_rs[finite_mask]
            
            # Perform linear regression
            coefficients = np.polyfit(log_lags, log_rs, 1)
            hurst_exponent = coefficients[0]
            
            # Calculate R-squared for goodness of fit
            predicted_log_rs = np.polyval(coefficients, log_lags)
            ss_res = np.sum((log_rs - predicted_log_rs) ** 2)
            ss_tot = np.sum((log_rs - np.mean(log_rs)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Supporting metrics
            supporting_metrics = {
                'r_squared': r_squared,
                'n_observations': len(log_returns),
                'n_lags_used': len(log_lags),
                'mean_rs': np.mean(valid_rs),
                'volatility': np.std(log_returns),
                'mean_return': np.mean(log_returns),
                'price_range': (np.min(prices), np.max(prices))
            }
            
            # Bound Hurst exponent to reasonable range
            hurst_exponent = np.clip(hurst_exponent, 0.0, 1.0)
            
            return hurst_exponent, r_squared, supporting_metrics
            
        except Exception as e:
            warnings.warn(f"Error in Hurst calculation: {str(e)}")
            return 0.5, 0.0, {'error': str(e)}
    
    def _calculate_rs_statistic(self, returns: np.ndarray, lag: int) -> float:
        """
        Calculate the Rescaled Range (R/S) statistic for a given lag.
        
        R/S = (Max cumulative deviation - Min cumulative deviation) / Standard deviation
        
        Args:
            returns: Log returns array
            lag: Time lag for calculation
            
        Returns:
            R/S statistic
        """
        if lag >= len(returns) or lag < 2:
            return 0.0
        
        # Split returns into non-overlapping windows of size 'lag'
        n_windows = len(returns) // lag
        if n_windows < 1:
            return 0.0
        
        rs_values = []
        
        for i in range(n_windows):
            start_idx = i * lag
            end_idx = start_idx + lag
            window_returns = returns[start_idx:end_idx]
            
            # Calculate mean return for this window
            mean_return = np.mean(window_returns)
            
            # Calculate cumulative deviations from mean
            deviations = window_returns - mean_return
            cumulative_deviations = np.cumsum(deviations)
            
            # Calculate range (R)
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            
            # Calculate standard deviation (S)
            S = np.std(window_returns, ddof=1) if len(window_returns) > 1 else np.std(window_returns)
            
            # Avoid division by zero
            if S > 0 and R >= 0:
                rs_values.append(R / S)
        
        # Return mean R/S ratio across all windows
        return np.mean(rs_values) if rs_values else 0.0
    
    def classify_regime(self, hurst_exponent: float, confidence_score: float) -> str:
        """
        Classify market regime based on Hurst exponent.
        
        Args:
            hurst_exponent: Calculated Hurst exponent
            confidence_score: Confidence in the measurement (R-squared)
            
        Returns:
            Regime classification string
        """
        # If confidence is too low, default to neutral
        if confidence_score < self.confidence_threshold:
            return 'neutral'
        
        # Classify based on Hurst exponent thresholds
        if hurst_exponent > self.regime_thresholds['trending_strong']:
            return 'trending_strong'
        elif hurst_exponent > self.regime_thresholds['trending_weak']:
            return 'trending_weak'
        elif hurst_exponent >= self.regime_thresholds['neutral_lower']:
            return 'neutral'
        else:
            return 'mean_reverting'
    
    def detect_current_regime(self, 
                            df: pd.DataFrame, 
                            price_column: str = 'close',
                            window: Optional[int] = None) -> Dict:
        """
        Detect current market regime based on recent price data.
        
        Args:
            df: DataFrame with price data
            price_column: Name of price column to analyze
            window: Rolling window size (uses self.window_size if None)
            
        Returns:
            Dictionary with regime analysis results
        """
        if window is None:
            window = self.window_size
        
        if len(df) < window:
            # Not enough data - return neutral regime
            return {
                'regime_type': 'neutral',
                'hurst_exponent': 0.5,
                'confidence_score': 0.0,
                'supporting_metrics': {'error': 'insufficient_data', 'n': len(df)},
                'timestamp': df.index[-1] if len(df) > 0 else pd.Timestamp.now(),
                'regime_parameters': self.get_regime_parameters('neutral')
            }
        
        # Use most recent window of data
        recent_data = df.tail(window)
        price_series = recent_data[price_column]
        
        # Calculate Hurst exponent
        hurst_exponent, r_squared, supporting_metrics = self.calculate_hurst_exponent(price_series)
        
        # Classify regime
        regime_type = self.classify_regime(hurst_exponent, r_squared)
        
        # Get regime-specific parameters
        regime_parameters = self.get_regime_parameters(regime_type)
        
        # Create regime classification object for internal use
        classification = RegimeClassification(
            regime_type=regime_type,
            hurst_exponent=hurst_exponent,
            confidence_score=r_squared,
            supporting_metrics=supporting_metrics,
            timestamp=recent_data.index[-1],
            regime_parameters=regime_parameters
        )
        
        # Update regime history
        self._regime_history.append(classification)
        
        # Keep only recent history (last 100 classifications)
        if len(self._regime_history) > 100:
            self._regime_history = self._regime_history[-100:]
        
        # Return as dictionary for external use
        return {
            'regime_type': regime_type,
            'hurst_exponent': hurst_exponent,
            'confidence_score': r_squared,
            'supporting_metrics': supporting_metrics,
            'timestamp': recent_data.index[-1],
            'regime_parameters': regime_parameters
        }
    
    def get_regime_parameters(self, regime_type: str) -> Dict[str, float]:
        """
        Get trading parameters optimized for specific market regime.
        
        Args:
            regime_type: Market regime classification
            
        Returns:
            Dictionary of regime-specific trading parameters
        """
        # Default parameters from config
        base_params = {
            'position_multiplier': 1.0,
            'stop_loss_pct': config.STOP_LOSS_PCT,
            'take_profit_pct': config.TAKE_PROFIT_PCT,
            'min_confidence': config.CONFIDENCE_THRESHOLD,
            'transaction_cost_multiplier': 1.0,
            'max_holding_periods': 24,  # hours
            'volatility_scaling': 1.0
        }
        
        # Regime-specific adjustments
        regime_adjustments = {
            'trending_strong': {
                'position_multiplier': 2.0,  # Higher position size in strong trends
                'stop_loss_pct': 0.12,      # Wider stops to avoid whipsaws
                'take_profit_pct': 0.25,    # Higher profit targets
                'min_confidence': 0.75,     # Require higher confidence
                'transaction_cost_multiplier': 1.2,  # Account for slippage
                'max_holding_periods': 48,  # Hold longer in trends
                'volatility_scaling': 1.5   # Scale with volatility
            },
            'trending_weak': {
                'position_multiplier': 1.3,
                'stop_loss_pct': 0.08,
                'take_profit_pct': 0.15,
                'min_confidence': 0.70,
                'transaction_cost_multiplier': 1.1,
                'max_holding_periods': 24,
                'volatility_scaling': 1.2
            },
            'neutral': {
                'position_multiplier': 0.8,  # Smaller positions in choppy markets
                'stop_loss_pct': 0.06,      # Tighter stops
                'take_profit_pct': 0.10,    # Quick profits
                'min_confidence': 0.75,     # Higher confidence needed
                'transaction_cost_multiplier': 1.0,
                'max_holding_periods': 12,  # Quick trades
                'volatility_scaling': 1.0
            },
            'mean_reverting': {
                'position_multiplier': 0.6,  # Very conservative
                'stop_loss_pct': 0.04,      # Very tight stops
                'take_profit_pct': 0.08,    # Quick scalping profits
                'min_confidence': 0.85,     # Very high confidence required
                'transaction_cost_multiplier': 0.9,
                'max_holding_periods': 6,   # Very quick trades
                'volatility_scaling': 0.8   # Reduce with volatility
            }
        }
        
        # Apply regime-specific adjustments
        if regime_type in regime_adjustments:
            adjusted_params = base_params.copy()
            adjusted_params.update(regime_adjustments[regime_type])
            return adjusted_params
        else:
            return base_params
    
    def get_regime_stability(self, lookback_periods: int = 10) -> float:
        """
        Calculate regime stability over recent periods.
        
        Args:
            lookback_periods: Number of recent periods to analyze
            
        Returns:
            Stability score (0-1), where 1 = perfectly stable regime
        """
        if len(self._regime_history) < 2:
            return 0.0
        
        recent_regimes = self._regime_history[-lookback_periods:]
        
        if len(recent_regimes) < 2:
            return 0.0
        
        # Calculate regime consistency
        regime_types = [r.regime_type for r in recent_regimes]
        most_common_regime = max(set(regime_types), key=regime_types.count)
        stability = regime_types.count(most_common_regime) / len(regime_types)
        
        return stability
    
    def get_regime_trend(self, lookback_periods: int = 20) -> Dict[str, float]:
        """
        Analyze regime trends over time.
        
        Args:
            lookback_periods: Number of periods to analyze
            
        Returns:
            Dictionary with regime trend analysis
        """
        if len(self._regime_history) < lookback_periods:
            return {'trend_direction': 0.0, 'trend_strength': 0.0, 'confidence': 0.0}
        
        recent_history = self._regime_history[-lookback_periods:]
        
        # Extract Hurst exponents over time
        hurst_values = [r.hurst_exponent for r in recent_history]
        confidence_values = [r.confidence_score for r in recent_history]
        
        # Calculate trend in Hurst exponent (moving toward/away from trending)
        if len(hurst_values) >= 3:
            # Simple linear trend
            x = np.arange(len(hurst_values))
            coefficients = np.polyfit(x, hurst_values, 1)
            trend_direction = coefficients[0]  # Slope
            
            # Trend strength (R-squared of the trend)
            predicted = np.polyval(coefficients, x)
            ss_res = np.sum((hurst_values - predicted) ** 2)
            ss_tot = np.sum((hurst_values - np.mean(hurst_values)) ** 2)
            trend_strength = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Average confidence
            avg_confidence = np.mean(confidence_values)
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': max(0, trend_strength),
                'confidence': avg_confidence,
                'current_hurst': hurst_values[-1],
                'hurst_range': (min(hurst_values), max(hurst_values))
            }
        
        return {'trend_direction': 0.0, 'trend_strength': 0.0, 'confidence': 0.0}
    
    def should_update_regime(self, current_regime: str, new_regime: str, 
                           stability_threshold: float = 0.8) -> bool:
        """
        Determine if regime should be updated based on stability criteria.
        
        Args:
            current_regime: Current regime classification
            new_regime: Newly detected regime
            stability_threshold: Minimum stability required for regime change
            
        Returns:
            True if regime should be updated
        """
        if current_regime == new_regime:
            return False
        
        # Check regime stability
        stability = self.get_regime_stability()
        
        # Only update if the regime change is stable
        return stability < stability_threshold
    
    def get_regime_summary(self) -> Dict[str, any]:
        """
        Get comprehensive summary of current regime analysis.
        
        Returns:
            Dictionary with complete regime analysis
        """
        if not self._regime_history:
            return {'status': 'no_data'}
        
        current_regime = self._regime_history[-1]
        stability = self.get_regime_stability()
        trend_analysis = self.get_regime_trend()
        
        return {
            'current_regime': current_regime.regime_type,
            'hurst_exponent': current_regime.hurst_exponent,
            'confidence': current_regime.confidence_score,
            'stability': stability,
            'trend_analysis': trend_analysis,
            'regime_parameters': current_regime.regime_parameters,
            'supporting_metrics': current_regime.supporting_metrics,
            'timestamp': current_regime.timestamp,
            'history_length': len(self._regime_history)
        }

# Factory function for easy instantiation
def create_regime_detector(
    window_size: int = 100,
    max_lag: int = 50,
    confidence_threshold: float = 0.7
) -> MarketRegimeDetector:
    """
    Create a MarketRegimeDetector with standard parameters.
    
    Args:
        window_size: Rolling window for regime detection
        max_lag: Maximum lag for R/S analysis  
        confidence_threshold: Minimum confidence for classification
        
    Returns:
        Configured MarketRegimeDetector instance
    """
    return MarketRegimeDetector(
        window_size=window_size,
        max_lag=max_lag,
        confidence_threshold=confidence_threshold
    )

# Export main classes and functions
__all__ = [
    'MarketRegimeDetector',
    'RegimeClassification', 
    'create_regime_detector'
]