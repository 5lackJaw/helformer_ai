"""
Multi-Timeframe Analysis Module for Helformer
Provides cross-timeframe feature engineering and signal aggregation
"""

import pandas as pd
import numpy as np
import ccxt
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from config_helformer import config
from improved_training_utils import create_research_based_features as helformer_features

logger = logging.getLogger(__name__)

class MultiTimeframeAnalyzer:
    """
    Multi-timeframe analysis for enhanced prediction accuracy.
    
    Analyzes multiple timeframes simultaneously and combines signals
    for more robust trading decisions.
    """
    
    def __init__(self, exchange: ccxt.Exchange, symbol: str):
        """
        Initialize multi-timeframe analyzer.
        
        Args:
            exchange: CCXT exchange instance
            symbol: Trading symbol (e.g., 'BTC/USDT')
        """
        self.exchange = exchange
        self.symbol = symbol
        self.timeframes = [config.PRIMARY_TIMEFRAME] + config.SECONDARY_TIMEFRAMES
        self.data_cache = {}
        self.feature_cache = {}
        
        logger.info(f"Multi-timeframe analyzer initialized for {symbol}")
        logger.info(f"Timeframes: {self.timeframes}")
    
    def fetch_multi_timeframe_data(self, limit: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for all timeframes.
        
        Args:
            limit: Number of candles to fetch per timeframe
            
        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        data = {}
        
        for timeframe in self.timeframes:
            try:
                logger.debug(f"Fetching {timeframe} data for {self.symbol}")
                
                ohlcv = self.exchange.fetch_ohlcv(
                    self.symbol, 
                    timeframe=timeframe, 
                    limit=limit
                )
                
                df = pd.DataFrame(
                    ohlcv, 
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                data[timeframe] = df
                self.data_cache[timeframe] = df
                
                logger.debug(f"Fetched {len(df)} candles for {timeframe}")
                
            except Exception as e:
                logger.error(f"Error fetching {timeframe} data: {str(e)}")
                # Use cached data if available
                if timeframe in self.data_cache:
                    data[timeframe] = self.data_cache[timeframe]
                    logger.warning(f"Using cached data for {timeframe}")
                else:
                    logger.error(f"No data available for {timeframe}")
        
        return data
    
    def compute_multi_timeframe_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Compute features for each timeframe.
        
        Args:
            data: Dictionary mapping timeframe to OHLCV DataFrame
            
        Returns:
            Dictionary mapping timeframe to features DataFrame
        """
        features = {}
        
        for timeframe, df in data.items():
            if df is None or len(df) < 50:
                logger.warning(f"Insufficient data for {timeframe} features")
                continue
                
            try:
                logger.debug(f"Computing features for {timeframe}")
                
                # Compute standard Helformer features
                df_with_features = helformer_features(df.copy())
                
                # Add timeframe-specific features
                df_with_features = self._add_timeframe_specific_features(
                    df_with_features, timeframe
                )
                
                features[timeframe] = df_with_features
                self.feature_cache[timeframe] = df_with_features
                
                logger.debug(f"Computed {len(df_with_features.columns)} features for {timeframe}")
                
            except Exception as e:
                logger.error(f"Error computing features for {timeframe}: {str(e)}")
                # Use cached features if available
                if timeframe in self.feature_cache:
                    features[timeframe] = self.feature_cache[timeframe]
        
        return features
    
    def _add_timeframe_specific_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Add timeframe-specific features.
        
        Args:
            df: DataFrame with basic features
            timeframe: Timeframe string (e.g., '15m')
            
        Returns:
            DataFrame with additional timeframe features
        """
        # Extract timeframe multiplier for scaling
        if timeframe.endswith('m'):
            tf_minutes = int(timeframe[:-1])
        elif timeframe.endswith('h'):
            tf_minutes = int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            tf_minutes = int(timeframe[:-1]) * 1440
        else:
            tf_minutes = 15  # Default
        
        # Timeframe-adjusted volatility
        df[f'volatility_tf_{timeframe}'] = df['returns'].rolling(
            window=max(20, 20 * 15 // tf_minutes)  # Adjust window for timeframe
        ).std()
        
        # Timeframe-adjusted momentum
        momentum_periods = [5, 10, 20]
        for period in momentum_periods:
            adjusted_period = max(2, period * 15 // tf_minutes)
            if len(df) >= adjusted_period + 1:
                df[f'momentum_{period}_tf_{timeframe}'] = (
                    df['close'] / df['close'].shift(adjusted_period) - 1
                )
        
        # Timeframe weight (for ensemble)
        df[f'timeframe_weight'] = config.TIMEFRAME_WEIGHTS.get(timeframe, 0.1)
        
        return df
    
    def aggregate_cross_timeframe_signals(self, features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate signals across timeframes.
        
        Args:
            features: Dictionary mapping timeframe to features DataFrame
            
        Returns:
            DataFrame with aggregated cross-timeframe features
        """
        if not features or config.PRIMARY_TIMEFRAME not in features:
            raise ValueError("Primary timeframe data missing")
        
        primary_df = features[config.PRIMARY_TIMEFRAME].copy()
        
        # Initialize cross-timeframe features
        primary_df['trend_alignment_score'] = 0.0
        primary_df['timeframe_momentum_consensus'] = 0.0
        primary_df['cross_tf_volatility_ratio'] = 1.0
        primary_df['higher_tf_trend_strength'] = 0.0
        
        # Get latest values for alignment calculation
        latest_idx = primary_df.index[-1]
        trend_signals = []
        momentum_signals = []
        volatility_values = []
        
        for timeframe, df in features.items():
            if len(df) == 0:
                continue
                
            weight = config.TIMEFRAME_WEIGHTS.get(timeframe, 0.1)
            
            # Get latest values (use last available if exact timestamp not found)
            try:
                if latest_idx in df.index:
                    latest_data = df.loc[latest_idx]
                else:
                    # Use the most recent data available
                    latest_data = df.iloc[-1]
                
                # Trend signal (based on multiple moving averages)
                trend_signal = 0.0
                ma_columns = [col for col in df.columns if 'price_vs_sma' in col]
                for col in ma_columns:
                    if not pd.isna(latest_data.get(col, np.nan)):
                        trend_signal += np.sign(latest_data[col])
                
                if ma_columns:
                    trend_signal = trend_signal / len(ma_columns)
                
                trend_signals.append(trend_signal * weight)
                
                # Momentum signal
                momentum_cols = [col for col in df.columns if 'momentum' in col]
                momentum_signal = 0.0
                for col in momentum_cols:
                    if not pd.isna(latest_data.get(col, np.nan)):
                        momentum_signal += np.sign(latest_data[col])
                
                if momentum_cols:
                    momentum_signal = momentum_signal / len(momentum_cols)
                
                momentum_signals.append(momentum_signal * weight)
                
                # Volatility
                if 'returns' in df.columns:
                    vol = df['returns'].tail(20).std()
                    if not pd.isna(vol):
                        volatility_values.append(vol)
                
            except Exception as e:
                logger.warning(f"Error processing {timeframe} for aggregation: {str(e)}")
                continue
        
        # Calculate aggregated signals
        if trend_signals:
            primary_df.loc[latest_idx, 'trend_alignment_score'] = np.mean(trend_signals)
            
        if momentum_signals:
            primary_df.loc[latest_idx, 'timeframe_momentum_consensus'] = np.mean(momentum_signals)
        
        if len(volatility_values) > 1:
            # Ratio of current timeframe volatility to average
            primary_vol = primary_df['returns'].tail(20).std()
            avg_vol = np.mean(volatility_values)
            if avg_vol > 0:
                primary_df.loc[latest_idx, 'cross_tf_volatility_ratio'] = primary_vol / avg_vol
        
        # Higher timeframe trend strength
        higher_tf_timeframes = ['1h', '4h', '1d']
        higher_tf_trends = []
        
        for tf in higher_tf_timeframes:
            if tf in features and len(features[tf]) > 0:
                tf_df = features[tf]
                try:
                    if latest_idx in tf_df.index:
                        latest_tf_data = tf_df.loc[latest_idx]
                    else:
                        latest_tf_data = tf_df.iloc[-1]
                    
                    # Simple trend strength based on price vs longer MA
                    if 'price_vs_sma_50' in tf_df.columns:
                        trend_strength = latest_tf_data.get('price_vs_sma_50', 0)
                        if not pd.isna(trend_strength):
                            higher_tf_trends.append(trend_strength)
                            
                except Exception as e:
                    logger.warning(f"Error calculating trend strength for {tf}: {str(e)}")
        
        if higher_tf_trends:
            primary_df.loc[latest_idx, 'higher_tf_trend_strength'] = np.mean(higher_tf_trends)
        
        # Forward fill the new features for the entire DataFrame
        cross_tf_columns = [
            'trend_alignment_score', 'timeframe_momentum_consensus',
            'cross_tf_volatility_ratio', 'higher_tf_trend_strength'
        ]
        
        for col in cross_tf_columns:
            primary_df[col] = primary_df[col].fillna(method='ffill')
            primary_df[col] = primary_df[col].fillna(0.0)  # Fill any remaining NaN
        
        logger.info("Cross-timeframe features aggregated successfully")
        return primary_df
    
    def get_timeframe_confidence_score(self, features: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate confidence score based on timeframe alignment.
        
        Args:
            features: Dictionary mapping timeframe to features DataFrame
            
        Returns:
            Confidence score between 0 and 1
        """
        if not config.TIMEFRAME_CONFIDENCE_SCALING:
            return 1.0
        
        try:
            trend_signals = []
            
            for timeframe, df in features.items():
                if len(df) == 0:
                    continue
                
                weight = config.TIMEFRAME_WEIGHTS.get(timeframe, 0.1)
                latest_data = df.iloc[-1]
                
                # Calculate trend direction
                trend_indicators = 0
                total_indicators = 0
                
                # Moving average trends
                ma_columns = [col for col in df.columns if 'price_vs_sma' in col]
                for col in ma_columns:
                    value = latest_data.get(col, np.nan)
                    if not pd.isna(value):
                        trend_indicators += 1 if value > 0 else -1
                        total_indicators += 1
                
                # Momentum trends
                momentum_cols = [col for col in df.columns if 'momentum' in col]
                for col in momentum_cols:
                    value = latest_data.get(col, np.nan)
                    if not pd.isna(value):
                        trend_indicators += 1 if value > 0 else -1
                        total_indicators += 1
                
                if total_indicators > 0:
                    trend_score = trend_indicators / total_indicators
                    trend_signals.append((trend_score, weight))
            
            if not trend_signals:
                return 0.5  # Neutral confidence
            
            # Calculate weighted agreement
            total_weight = sum(weight for _, weight in trend_signals)
            if total_weight == 0:
                return 0.5
            
            # Agreement is measured by how similar the trend signals are
            weighted_trend = sum(trend * weight for trend, weight in trend_signals) / total_weight
            
            # Calculate variance in signals
            variance = sum(weight * (trend - weighted_trend) ** 2 for trend, weight in trend_signals) / total_weight
            
            # Convert variance to confidence (lower variance = higher confidence)
            confidence = max(0.0, min(1.0, 1.0 - variance))
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating timeframe confidence: {str(e)}")
            return 0.5  # Default to neutral confidence
    
    def should_trade_based_on_alignment(self, features: Dict[str, pd.DataFrame]) -> Tuple[bool, str]:
        """
        Determine if trade should proceed based on timeframe alignment.
        
        Args:
            features: Dictionary mapping timeframe to features DataFrame
            
        Returns:
            Tuple of (should_trade, reason)
        """
        if not config.REQUIRE_TREND_ALIGNMENT:
            return True, "Trend alignment not required"
        
        try:
            confidence = self.get_timeframe_confidence_score(features)
            
            if confidence >= config.MIN_TIMEFRAME_AGREEMENT:
                return True, f"Timeframe alignment sufficient: {confidence:.2f}"
            else:
                return False, f"Insufficient timeframe alignment: {confidence:.2f} < {config.MIN_TIMEFRAME_AGREEMENT}"
                
        except Exception as e:
            logger.error(f"Error checking timeframe alignment: {str(e)}")
            return False, f"Error in alignment check: {str(e)}"

def get_multi_timeframe_features(exchange: ccxt.Exchange, symbol: str, limit: int = 1000) -> pd.DataFrame:
    """
    Convenience function to get multi-timeframe features.
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading symbol
        limit: Number of candles per timeframe
        
    Returns:
        DataFrame with multi-timeframe features
    """
    analyzer = MultiTimeframeAnalyzer(exchange, symbol)
    
    # Fetch data for all timeframes
    data = analyzer.fetch_multi_timeframe_data(limit)
    
    # Compute features for each timeframe
    features = analyzer.compute_multi_timeframe_features(data)
    
    # Aggregate cross-timeframe signals
    aggregated_features = analyzer.aggregate_cross_timeframe_signals(features)
    
    return aggregated_features

if __name__ == "__main__":
    # Test the multi-timeframe analyzer
    import ccxt
    
    try:
        # Initialize exchange (using testnet for safety)
        exchange = ccxt.binance({
            'apiKey': 'your_api_key',
            'secret': 'your_secret',
            'sandbox': True,  # Use testnet
            'enableRateLimit': True,
        })
        
        # Test analyzer
        analyzer = MultiTimeframeAnalyzer(exchange, 'BTC/USDT')
        data = analyzer.fetch_multi_timeframe_data(500)
        
        print(f"Fetched data for {len(data)} timeframes")
        for tf, df in data.items():
            print(f"{tf}: {len(df)} candles")
        
        # Compute features
        features = analyzer.compute_multi_timeframe_features(data)
        print(f"Computed features for {len(features)} timeframes")
        
        # Aggregate signals
        if features:
            aggregated = analyzer.aggregate_cross_timeframe_signals(features)
            print(f"Aggregated features shape: {aggregated.shape}")
            
            # Check alignment
            should_trade, reason = analyzer.should_trade_based_on_alignment(features)
            print(f"Should trade: {should_trade} - {reason}")
            
    except Exception as e:
        print(f"Test failed: {str(e)}")
        print("Note: This test requires valid API keys for live testing")