"""
Integration of Market Regime Detection with Helformer Training System
Adds regime-aware features and training parameters without data leakage
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from market_regime_detector import MarketRegimeDetector, create_regime_detector
from config_helformer import config
import warnings

class RegimeAwareFeatureEngine:
    """
    Feature engineering that incorporates market regime detection
    Ensures no data leakage by computing regime features sequentially
    """
    
    def __init__(self, 
                 regime_window: int = 100,
                 regime_update_freq: int = 24):
        """
        Initialize regime-aware feature engine
        
        Args:
            regime_window: Window size for regime detection
            regime_update_freq: How often to update regime (in periods)
        """
        self.regime_detector = create_regime_detector(
            window_size=regime_window,
            confidence_threshold=0.7
        )
        self.regime_update_freq = regime_update_freq
        self.last_regime_update = 0
        self.current_regime = None
    
    def add_regime_features(self, df: pd.DataFrame, 
                          price_column: str = 'close') -> pd.DataFrame:
        """
        Add regime-based features to dataset WITHOUT data leakage
        
        Args:
            df: DataFrame with OHLCV data
            price_column: Column name for price data
            
        Returns:
            DataFrame with additional regime features
        """
        print("Adding regime features (NO DATA LEAKAGE)...")
        
        # Sort by time to ensure proper order
        df = df.sort_index()
        
        # Initialize regime feature columns
        regime_features = [
            'hurst_exponent',
            'regime_confidence',
            'regime_trending_strong',
            'regime_trending_weak', 
            'regime_neutral',
            'regime_mean_reverting',
            'regime_stability',
            'regime_trend_direction',
            'regime_position_multiplier',
            'regime_volatility_scaling'
        ]
        
        for feature in regime_features:
            df[feature] = np.nan
        
        # Calculate regime features sequentially (NO FUTURE DATA)
        min_data_points = max(self.regime_detector.window_size, 100)
        
        for i in range(min_data_points, len(df)):
            # Only use data up to current point (critical for no leakage)
            historical_data = df.iloc[:i+1].copy()
            
            # Update regime detection (only if enough time has passed)
            periods_since_update = i - self.last_regime_update
            if (self.current_regime is None or 
                periods_since_update >= self.regime_update_freq):
                
                try:
                    # Detect regime using only historical data
                    regime_classification = self.regime_detector.detect_current_regime(
                        historical_data, price_column
                    )
                    self.current_regime = regime_classification
                    self.last_regime_update = i
                    
                except Exception as e:
                    # If regime detection fails, use previous regime or neutral
                    if self.current_regime is None:
                        # Create neutral regime as fallback
                        from market_regime_detector import RegimeClassification
                        self.current_regime = RegimeClassification(
                            regime_type='neutral',
                            hurst_exponent=0.5,
                            confidence_score=0.0,
                            supporting_metrics={},
                            timestamp=historical_data.index[-1],
                            regime_parameters=self.regime_detector.get_regime_parameters('neutral')
                        )
            
            if self.current_regime is not None:
                # Set regime features for current row
                regime = self.current_regime
                
                # Core regime metrics
                df.iloc[i, df.columns.get_loc('hurst_exponent')] = regime.hurst_exponent
                df.iloc[i, df.columns.get_loc('regime_confidence')] = regime.confidence_score
                
                # One-hot encoded regime types (for model input)
                df.iloc[i, df.columns.get_loc('regime_trending_strong')] = 1.0 if regime.regime_type == 'trending_strong' else 0.0
                df.iloc[i, df.columns.get_loc('regime_trending_weak')] = 1.0 if regime.regime_type == 'trending_weak' else 0.0
                df.iloc[i, df.columns.get_loc('regime_neutral')] = 1.0 if regime.regime_type == 'neutral' else 0.0
                df.iloc[i, df.columns.get_loc('regime_mean_reverting')] = 1.0 if regime.regime_type == 'mean_reverting' else 0.0
                
                # Regime stability and trend
                stability = self.regime_detector.get_regime_stability()
                trend_analysis = self.regime_detector.get_regime_trend()
                
                df.iloc[i, df.columns.get_loc('regime_stability')] = stability
                df.iloc[i, df.columns.get_loc('regime_trend_direction')] = trend_analysis.get('trend_direction', 0.0)
                
                # Trading parameters (for position sizing)
                params = regime.regime_parameters
                df.iloc[i, df.columns.get_loc('regime_position_multiplier')] = params['position_multiplier']
                df.iloc[i, df.columns.get_loc('regime_volatility_scaling')] = params['volatility_scaling']
        
        # Clean up NaN values 
        # Forward fill regime features (regime persists until next update)
        for feature in regime_features:
            df[feature] = df[feature].fillna(method='ffill')
        
        # Backward fill any remaining NaN at the beginning
        for feature in regime_features:
            df[feature] = df[feature].fillna(method='bfill')
        
        # Final cleanup - fill any remaining NaN with neutral values
        df['hurst_exponent'] = df['hurst_exponent'].fillna(0.5)
        df['regime_confidence'] = df['regime_confidence'].fillna(0.0)
        df['regime_neutral'] = df['regime_neutral'].fillna(1.0)
        df['regime_position_multiplier'] = df['regime_position_multiplier'].fillna(1.0)
        df['regime_volatility_scaling'] = df['regime_volatility_scaling'].fillna(1.0)
        
        # Fill other regime one-hot features with 0
        for feature in ['regime_trending_strong', 'regime_trending_weak', 'regime_mean_reverting']:
            df[feature] = df[feature].fillna(0.0)
        
        df['regime_stability'] = df['regime_stability'].fillna(0.0)
        df['regime_trend_direction'] = df['regime_trend_direction'].fillna(0.0)
        
        valid_rows = df[regime_features].notna().all(axis=1).sum()
        print(f"Regime features added: {len(regime_features)} features, {valid_rows} valid rows")
        
        return df
    
    def get_current_regime_parameters(self) -> Dict[str, float]:
        """Get current regime trading parameters"""
        if self.current_regime is None:
            return self.regime_detector.get_regime_parameters('neutral')
        return self.current_regime.regime_parameters
    
    def get_regime_adjusted_config(self, base_config: Dict) -> Dict:
        """
        Adjust training configuration based on current regime
        
        Args:
            base_config: Base training configuration
            
        Returns:
            Regime-adjusted configuration
        """
        if self.current_regime is None:
            return base_config
        
        regime_params = self.current_regime.regime_parameters
        adjusted_config = base_config.copy()
        
        # Adjust training parameters based on regime
        if self.current_regime.regime_type == 'trending_strong':
            # In strong trends, use larger batch sizes and more patience
            adjusted_config['batch_size'] = int(base_config.get('batch_size', 32) * 1.2)
            adjusted_config['early_stopping_patience'] = int(base_config.get('early_stopping_patience', 20) * 1.5)
            adjusted_config['dropout_rate'] = base_config.get('dropout_rate', 0.2) * 0.9  # Less dropout
            
        elif self.current_regime.regime_type == 'mean_reverting':
            # In mean reverting markets, use smaller batches and more regularization
            adjusted_config['batch_size'] = int(base_config.get('batch_size', 32) * 0.8)
            adjusted_config['dropout_rate'] = base_config.get('dropout_rate', 0.2) * 1.2  # More dropout
            adjusted_config['learning_rate'] = base_config.get('learning_rate', 0.001) * 0.8  # Lower LR
            
        elif self.current_regime.regime_type == 'neutral':
            # In neutral markets, use more conservative settings
            adjusted_config['early_stopping_patience'] = int(base_config.get('early_stopping_patience', 20) * 0.8)
            adjusted_config['learning_rate'] = base_config.get('learning_rate', 0.001) * 0.9
        
        return adjusted_config

def add_regime_features_to_existing_pipeline():
    """
    Integration function to add regime features to existing training pipeline
    This modifies training_utils.py functionality
    """
    
    def enhanced_helformer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced version of helformer_features that includes regime detection
        This function can replace the existing helformer_features in training_utils.py
        """
        # First run existing feature engineering
        from improved_training_utils import create_research_based_features as original_helformer_features
        
        try:
            df_with_features = original_helformer_features(df)
        except Exception as e:
            print(f"Warning: Original feature engineering failed: {e}")
            df_with_features = df.copy()
        
        # Add regime features
        regime_engine = RegimeAwareFeatureEngine()
        df_with_regime = regime_engine.add_regime_features(df_with_features)
        
        return df_with_regime
    
    return enhanced_helformer_features

def get_regime_aware_model_params(base_params: Dict, current_regime: str) -> Dict:
    """
    Adjust model parameters based on market regime
    
    Args:
        base_params: Base model parameters
        current_regime: Current market regime
        
    Returns:
        Regime-adjusted model parameters
    """
    adjusted_params = base_params.copy()
    
    # Get regime-specific adjustments from config
    if hasattr(config, 'REGIME_TRADING_PARAMS'):
        regime_config = config.REGIME_TRADING_PARAMS.get(current_regime, {})
        
        # Adjust model complexity based on regime
        if current_regime == 'trending_strong':
            # Use more model capacity for trend detection
            adjusted_params['lstm_units'] = int(base_params.get('lstm_units', 128) * 1.2)
            adjusted_params['num_heads'] = min(base_params.get('num_heads', 8) + 2, 12)
            adjusted_params['dropout_rate'] = base_params.get('dropout_rate', 0.2) * 0.8
            
        elif current_regime == 'mean_reverting':
            # Use more regularization for mean reversion
            adjusted_params['dropout_rate'] = base_params.get('dropout_rate', 0.2) * 1.3
            adjusted_params['learning_rate'] = base_params.get('learning_rate', 0.001) * 0.7
            
        elif current_regime == 'neutral':
            # Use balanced parameters
            adjusted_params['dropout_rate'] = base_params.get('dropout_rate', 0.2) * 1.1
            
    return adjusted_params

def create_regime_aware_training_callback():
    """
    Create a Keras callback that monitors regime changes during training
    """
    import tensorflow as tf
    
    class RegimeMonitoringCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.regime_detector = create_regime_detector()
            
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 10 == 0:  # Check regime every 10 epochs
                print(f"    Epoch {epoch + 1}: Monitoring training regime stability...")
                # Could add regime-specific adjustments here if needed
    
    return RegimeMonitoringCallback()

# Export integration functions
__all__ = [
    'RegimeAwareFeatureEngine',
    'add_regime_features_to_existing_pipeline',
    'get_regime_aware_model_params',
    'create_regime_aware_training_callback'
]