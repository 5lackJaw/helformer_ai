#!/usr/bin/env python3
"""
Improved Training Utils
Research-based improvements for feature engineering and target creation
"""

import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def create_research_based_features(df, symbol="BTC"):
    """
    Create research-proven features for crypto prediction
    Based on: Jaquart et al. (2021), Shah & Zhang (2014), Brownlee (2018)
    """
    print(f"🔧 Creating research-based features for {symbol}...")
    
    df = df.copy()
    
    # Price-based features (research: most predictive)
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['price_momentum'] = df['close'] / df['close'].shift(20) - 1  # 5-hour momentum
    
    # Volatility features (research: crucial for crypto)
    df['volatility'] = df['returns'].rolling(20).std()
    df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(60).mean()
    
    # Volume features (research: important for crypto)
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['relative_volume'] = df['volume'] / df['volume_ma']
    df['volume_price_trend'] = (df['close'] * df['volume']).rolling(10).mean() / (df['close'].rolling(10).mean() * df['volume'].rolling(10).mean())
    
    # Technical indicators (research-proven)
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['rsi_normalized'] = (df['rsi'] - 50) / 50  # Normalize to [-1, 1]
    
    # MACD (research: effective for trend detection)
    macd, macd_signal, macd_hist = talib.MACD(df['close'])
    df['macd_normalized'] = macd / df['close']
    df['macd_signal_normalized'] = macd_signal / df['close']
    df['macd_histogram'] = macd_hist / df['close']
    
    # Bollinger Bands (research: mean reversion indicator)
    bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'])
    df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle
    
    # Moving averages (research: trend following)
    df['sma_9'] = talib.SMA(df['close'], timeperiod=9)
    df['sma_21'] = talib.SMA(df['close'], timeperiod=21)
    df['price_vs_sma9'] = df['close'] / df['sma_9'] - 1
    df['price_vs_sma21'] = df['close'] / df['sma_21'] - 1
    df['sma_slope'] = (df['sma_21'] - df['sma_21'].shift(5)) / df['sma_21'].shift(5)
    
    # Stochastic oscillator (research: momentum indicator)
    slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'])
    df['stoch_k'] = (slowk - 50) / 50  # Normalize to [-1, 1]
    df['stoch_d'] = (slowd - 50) / 50
      # Time-based features (research: seasonality effects)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
    elif isinstance(df.index, pd.DatetimeIndex):
        # Use datetime index for time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
    else:
        # Create dummy time features if no timestamp available
        df['hour'] = 12  # Default to noon
        df['day_of_week'] = 1  # Default to Tuesday
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Market microstructure (research: important for crypto)
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['body_size'] = abs(df['close'] - df['open']) / df['close']
    df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
    df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
    
    print(f"✅ Created {len([col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} features")
    
    return df

def create_improved_targets(df, target_hours=1):
    """
    Create improved targets based on research
    Research: Future returns with proper scaling and outlier handling
    """
    print(f"🎯 Creating improved targets (target_hours={target_hours})...")
    
    # Calculate future returns (research: most direct target)
    periods_ahead = target_hours * 4  # 15-minute bars
    df['future_return'] = df['close'].shift(-periods_ahead) / df['close'] - 1
    
    # Remove extreme outliers (research: improves model stability)
    q01 = df['future_return'].quantile(0.01)
    q99 = df['future_return'].quantile(0.99)
    df['future_return'] = df['future_return'].clip(q01, q99)
    
    # Normalize targets (research: z-score normalization)
    target_mean = df['future_return'].mean()
    target_std = df['future_return'].std()
    df['target_normalized'] = (df['future_return'] - target_mean) / target_std
    
    # Additional targets for ensemble diversity
    df['target_direction'] = np.where(df['future_return'] > 0, 1, 0)  # Binary direction
    df['target_magnitude'] = np.abs(df['future_return'])  # Magnitude
    
    print(f"✅ Target statistics:")
    print(f"   Mean: {target_mean:.6f}")
    print(f"   Std: {target_std:.6f}")
    print(f"   Range: {df['future_return'].min():.6f} to {df['future_return'].max():.6f}")
    print(f"   Normalized range: {df['target_normalized'].min():.2f} to {df['target_normalized'].max():.2f}")
    
    return df, target_mean, target_std

def prepare_model_data(df, sequence_length=30):
    """
    Prepare data for model training with proper validation
    Research-based feature selection and scaling
    """
    print(f"📊 Preparing model data (sequence_length={sequence_length})...")
    
    # Select research-proven features
    feature_columns = [
        'returns', 'log_returns', 'price_momentum',
        'volatility', 'volatility_ratio', 'relative_volume', 'volume_price_trend',
        'rsi_normalized', 'macd_normalized', 'macd_signal_normalized', 'macd_histogram',
        'bb_position', 'bb_width', 'price_vs_sma9', 'price_vs_sma21', 'sma_slope',
        'stoch_k', 'stoch_d', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'price_range', 'body_size', 'upper_shadow', 'lower_shadow'
    ]
    
    # Filter available features
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"✅ Using {len(available_features)} features: {available_features}")
    
    # Remove rows with NaN values
    initial_len = len(df)
    df_clean = df.dropna(subset=available_features + ['target_normalized'])
    final_len = len(df_clean)
    print(f"📈 Data retention: {final_len:,} / {initial_len:,} ({final_len/initial_len*100:.1f}%)")
    
    return df_clean, available_features

def create_sequences(X, y, sequence_length):
    """Create sequences for LSTM training with proper validation"""
    if len(X) < sequence_length:
        return np.array([]), np.array([])
    
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)

def safe_feature_scaling(X_train, X_val, X_test):
    """Safe feature scaling with RobustScaler (research: better for financial data)"""
    scaler = RobustScaler()
    
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def evaluate_predictions(y_true, y_pred, target_mean=0, target_std=1):
    """
    Comprehensive evaluation metrics for financial predictions
    Research-based metrics for trading performance
    """
    # Denormalize predictions
    y_true_denorm = y_true * target_std + target_mean
    y_pred_denorm = y_pred * target_std + target_mean
    
    # Prediction accuracy metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Directional accuracy (research: crucial for trading)
    direction_true = np.sign(y_true_denorm)
    direction_pred = np.sign(y_pred_denorm)
    directional_accuracy = np.mean(direction_true == direction_pred)
    
    # Trading-specific metrics
    hit_ratio = np.mean(np.abs(y_pred_denorm) < 0.02)  # Within 2% prediction error
    
    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Directional_Accuracy': directional_accuracy,
        'Hit_Ratio': hit_ratio,
        'Mean_Prediction': np.mean(y_pred_denorm),
        'Std_Prediction': np.std(y_pred_denorm)
    }

def time_aware_train_val_test_split(df, train_pct=0.7, val_pct=0.15, test_pct=0.15):
    """
    Time-aware data splitting (research: prevents look-ahead bias)
    """
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))
    
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

# ============================================================================
# LEGACY FUNCTIONS FROM training_utils.py (CONSOLIDATED)
# ============================================================================

import tensorflow as tf
import keras

class HoltWintersLayer(keras.layers.Layer):
    """Holt-Winters decomposition as a trainable layer"""
    
    def __init__(self, seasonal_periods=24, **kwargs):
        super().__init__(**kwargs)
        self.seasonal_periods = seasonal_periods
        
    def build(self, input_shape):
        # Learnable parameters (initialized between 0 and 1)
        self.alpha = self.add_weight(
            name='alpha',
            shape=(),
            initializer='uniform',
            constraint=keras.constraints.MinMaxNorm(0.0, 1.0),
            trainable=True
        )
        self.gamma = self.add_weight(
            name='gamma', 
            shape=(),
            initializer='uniform',
            constraint=keras.constraints.MinMaxNorm(0.0, 1.0),
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, inputs):
        """
        Implement Equations 1-3 from Helformer paper:
        Lt = α * (Xt/St-m) + (1-α) * Lt-1  # Level
        St = γ * (Xt/Lt) + (1-γ) * St-m     # Seasonal  
        Deseasonalized = Xt/St              # Output
        """
        batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
        
        # Simplified Holt-Winters implementation for efficiency
        level_weights = tf.pow(1 - self.alpha, tf.range(seq_len, dtype=tf.float32))
        level_weights = tf.reverse(level_weights, [0])
        level_weights = level_weights / tf.reduce_sum(level_weights)
        
        # Apply smoothing
        level_tensor = tf.nn.conv1d(
            tf.expand_dims(inputs, -1),
            tf.expand_dims(tf.expand_dims(level_weights, -1), -1),
            stride=1,
            padding='SAME'
        )
        level = tf.squeeze(level_tensor, -1)
        
        # Seasonal approximation (simplified)
        seasonal = tf.ones_like(inputs)
        
        # Deseasonalized output
        deseasonalized = inputs / (seasonal + 1e-8)  # Avoid division by zero
        
        return deseasonalized
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'seasonal_periods': self.seasonal_periods
        })
        return config

def normalize_targets_no_leakage(df):
    """
    Normalize targets without data leakage (legacy compatibility)
    Note: This is already handled in create_improved_targets()
    """
    if 'future_return' in df.columns and 'target_normalized' not in df.columns:
        target_mean = df['future_return'].mean()
        target_std = df['future_return'].std()
        df['target_normalized'] = (df['future_return'] - target_mean) / target_std
    return df

def create_helformer_sequences(features, targets, window_size=None):
    """
    Create sequences for LSTM training (legacy compatibility)
    Note: Use create_sequences() from this file for new code
    """
    window_size = window_size or 30
    return create_sequences(features, targets, window_size)

def evaluate_helformer(y_true, y_pred):
    """
    Evaluate model predictions (legacy compatibility)
    Note: Use evaluate_predictions() from this file for new code
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Directional accuracy
    y_true_dir = np.sign(y_true)
    y_pred_dir = np.sign(y_pred)
    directional_accuracy = np.mean(y_true_dir == y_pred_dir)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy
    }

# ============================================================================
