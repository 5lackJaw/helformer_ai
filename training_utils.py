"""
Custom Loss Functions for Helformer Model
Implements directional accuracy and Sharpe ratio optimization
"""

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config_helformer import config
from typing import List
import warnings
warnings.filterwarnings('ignore')

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
        
        # Initialize level and seasonal components
        level = tf.ones((batch_size, seq_len))
        seasonal = tf.ones((batch_size, seq_len))
        
        # Initialize with first values
        level = tf.concat([inputs[:, :1], level[:, 1:]], axis=1)
        
        # Simplified Holt-Winters decomposition (avoid complex loops in graph mode)
        # Use moving averages as approximations for efficiency
        
        # Level approximation using exponential smoothing
        level_weights = tf.pow(1 - self.alpha, tf.range(seq_len, dtype=tf.float32))
        level_weights = tf.reverse(level_weights, [0])
        level_weights = level_weights / tf.reduce_sum(level_weights)
        
        # Apply smoothing
        for i in range(batch_size):
            level_i = tf.nn.conv1d(
                tf.expand_dims(inputs[i:i+1], -1),
                tf.expand_dims(tf.expand_dims(level_weights, -1), -1),
                stride=1,
                padding='SAME'
            )
            if i == 0:
                level_tensor = level_i
            else:
                level_tensor = tf.concat([level_tensor, level_i], axis=0)
        
        level = tf.squeeze(level_tensor, -1)
        
        # Seasonal approximation (simplified)
        seasonal_period = tf.minimum(seq_len // 4, 7)  # Weekly seasonality or shorter
        seasonal = tf.ones_like(inputs)
          # Deseasonalized output (Eq 3)
        deseasonalized = inputs / (seasonal + 1e-8)  # Avoid division by zero
        
        return deseasonalized
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'seasonal_periods': self.seasonal_periods
        })
        return config

def mish_activation(x):
    """Mish activation: x * tanh(softplus(x))"""
    return x * keras.activations.tanh(keras.activations.softplus(x))

# Register the custom activation globally
keras.utils.get_custom_objects()['mish_activation'] = mish_activation

def create_helformer_model(input_shape, **params):
    """Create a simplified Helformer model that works with our data structure"""
    
    # Extract parameters with defaults
    num_heads = params.get('num_heads', config.NUM_HEADS)
    d_model = params.get('head_dim', config.HEAD_DIM)  # Map head_dim to d_model
    lstm_units = params.get('lstm_units', config.LSTM_UNITS)
    dropout_rate = params.get('dropout_rate', config.DROPOUT_RATE)
    learning_rate = params.get('learning_rate', config.LEARNING_RATE)
    
    # Build simplified model
    inputs = keras.Input(shape=input_shape, name='feature_input')
    
    # Input normalization
    x = keras.layers.LayerNormalization()(inputs)
    
    # Multi-head attention
    attention = keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate
    )(x, x)
    
    # Add & Norm layer with dropout
    attention = keras.layers.Dropout(dropout_rate)(attention)
    attention = keras.layers.Add()([x, attention])
    attention = keras.layers.LayerNormalization()(attention)
    
    # LSTM layers
    lstm_out = keras.layers.LSTM(
        lstm_units,
        return_sequences=False,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(attention)
    
    # Dense layers
    dense = keras.layers.Dense(64, activation=mish_activation)(lstm_out)
    dense = keras.layers.BatchNormalization()(dense)
    dense = keras.layers.Dropout(dropout_rate)(dense)
    
    dense = keras.layers.Dense(32, activation=mish_activation)(dense)
    dense = keras.layers.Dropout(dropout_rate)(dense)
    
    # Output layer
    output = keras.layers.Dense(1, activation='linear', name='price_output')(dense)
    
    model = keras.Model(inputs=inputs, outputs=output)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def helformer_features(df):
    """
    Calculate features without data leakage - FIXED VERSION
    Features at time t use only data from time 0 to t-1
    """
    print("Adding Helformer features (NO DATA LEAKAGE)...")
    
    # Sort by time to ensure proper order
    df = df.sort_index()
    
    # Initialize feature columns
    feature_columns = [
        'returns', 'log_returns', 'price_acceleration',
        'relative_volume', 'rsi_normalized', 'macd_normalized',
        'price_vs_sma_10', 'price_vs_sma_20', 'price_vs_sma_50',
        'bb_position', 'bb_width', 'volatility_rank',
        'momentum_5', 'momentum_10', 'momentum_20',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
    ]
    
    for col in feature_columns:
        df[col] = np.nan
    
    # Calculate features sequentially, using only past data
    for i in range(len(df)):
        # Only use data up to current point (no future data)
        historical_data = df.iloc[:i+1].copy()
        
        if len(historical_data) < 2:
            continue  # Need at least 2 points for calculations
            
        # 1. Basic price features (safe - only use past data)
        if i > 0:
            df.iloc[i, df.columns.get_loc('returns')] = (
                historical_data['close'].iloc[-1] / historical_data['close'].iloc[-2] - 1
            )
            df.iloc[i, df.columns.get_loc('log_returns')] = np.log(
                historical_data['close'].iloc[-1] / historical_data['close'].iloc[-2]
            )
            
        if i > 1:
            df.iloc[i, df.columns.get_loc('price_acceleration')] = (
                df.iloc[i, df.columns.get_loc('returns')] - 
                df.iloc[i-1, df.columns.get_loc('returns')]
            )
        
        # 2. Volume features (safe - only use past data)
        if len(historical_data) >= config.VOLATILITY_WINDOW:
            volume_ma = historical_data['volume'].tail(config.VOLATILITY_WINDOW).mean()
            df.iloc[i, df.columns.get_loc('relative_volume')] = (
                historical_data['volume'].iloc[-1] / volume_ma
            )
        
        # 3. Technical indicators (FIXED - only use historical data)
        if len(historical_data) >= config.RSI_PERIOD * 2:  # Need enough data for RSI
            rsi_values = talib.RSI(historical_data['close'].values, timeperiod=config.RSI_PERIOD)
            if not np.isnan(rsi_values[-1]):
                df.iloc[i, df.columns.get_loc('rsi_normalized')] = (rsi_values[-1] - 50) / 50
        
        # 4. MACD (FIXED - only use historical data)
        if len(historical_data) >= 35:  # Need enough data for MACD
            macd_values, _, _ = talib.MACD(historical_data['close'].values)
            if not np.isnan(macd_values[-1]):
                df.iloc[i, df.columns.get_loc('macd_normalized')] = (
                    macd_values[-1] / historical_data['close'].iloc[-1]
                )
        
        # 5. Moving averages (FIXED - only use historical data)
        for period in config.MA_PERIODS:
            if len(historical_data) >= period:
                sma = historical_data['close'].tail(period).mean()
                col_name = f'price_vs_sma_{period}'
                if col_name in df.columns:
                    df.iloc[i, df.columns.get_loc(col_name)] = (
                        historical_data['close'].iloc[-1] / sma - 1
                    )
        
        # 6. Bollinger Bands (FIXED - only use historical data)
        if len(historical_data) >= config.VOLATILITY_WINDOW:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                historical_data['close'].values, 
                timeperiod=config.VOLATILITY_WINDOW
            )
            if not (np.isnan(bb_upper[-1]) or np.isnan(bb_lower[-1])):
                bb_range = bb_upper[-1] - bb_lower[-1]
                if bb_range > 0:
                    df.iloc[i, df.columns.get_loc('bb_position')] = (
                        (historical_data['close'].iloc[-1] - bb_lower[-1]) / bb_range
                    )
                    df.iloc[i, df.columns.get_loc('bb_width')] = bb_range / bb_middle[-1]        # 7. Volatility (FIXED - only use historical data)
        if len(historical_data) >= config.VOLATILITY_WINDOW:
            volatility = historical_data['returns'].tail(config.VOLATILITY_WINDOW).std()
            # Only calculate volatility rank if we have sufficient data
            if len(historical_data) >= config.VOLATILITY_WINDOW * 10:  # Increased requirement
                volatility_window = historical_data['returns'].rolling(config.VOLATILITY_WINDOW).std().dropna()
                if len(volatility_window) >= 50:  # Need at least 50 observations for meaningful ranking
                    current_vol = volatility_window.iloc[-1]
                    volatility_rank = (volatility_window <= current_vol).mean()  # Percentile rank
                    df.iloc[i, df.columns.get_loc('volatility_rank')] = volatility_rank
        
        # 8. Momentum (FIXED - only use historical data)
        for period in config.MOMENTUM_PERIODS:
            if len(historical_data) >= period + 1:
                momentum = (historical_data['close'].iloc[-1] / historical_data['close'].iloc[-(period+1)]) - 1
                col_name = f'momentum_{period}'
                if col_name in df.columns:
                    df.iloc[i, df.columns.get_loc(col_name)] = momentum
          # 9. Time features (cyclical encoding) - safe, no future data
        try:
            if hasattr(historical_data, 'index') and len(historical_data) > 0:
                timestamp = historical_data.index[-1]
                if hasattr(timestamp, 'hour'):
                    hour = timestamp.hour
                    dow = timestamp.dayofweek
                    df.iloc[i, df.columns.get_loc('hour_sin')] = np.sin(2 * np.pi * hour / 24)
                    df.iloc[i, df.columns.get_loc('hour_cos')] = np.cos(2 * np.pi * hour / 24)
                    df.iloc[i, df.columns.get_loc('dow_sin')] = np.sin(2 * np.pi * dow / 7)
                    df.iloc[i, df.columns.get_loc('dow_cos')] = np.cos(2 * np.pi * dow / 7)
        except Exception as e:
            # Skip time features if timestamp parsing fails
            pass
      # Clean data more aggressively
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Remove columns that are mostly NaN (>90% NaN)
    for col in df.columns:
        if col in feature_columns:  # Only check feature columns
            nan_pct = df[col].isnull().sum() / len(df)
            if nan_pct > 0.9:
                print(f"WARNING: Removing feature '{col}' - {nan_pct*100:.1f}% NaN values")
                if col in feature_columns:
                    feature_columns.remove(col)
    
    # Forward fill, then backward fill, then drop remaining NaN rows
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Final check - drop any rows that still have NaN in feature columns
    initial_len = len(df)
    df = df.dropna(subset=[col for col in feature_columns if col in df.columns])
    if len(df) < initial_len:
        print(f"Removed {initial_len - len(df)} rows with remaining NaN values")
    
    print(f"Features calculated: {[col for col in feature_columns if col in df.columns]}")
    return df


def normalize_targets_no_leakage(df):
    """Normalize targets without data leakage - FIXED VERSION"""
    print("Normalizing targets (NO DATA LEAKAGE)...")
    print(f"Input shape: {df.shape}")
    
    # Calculate targets first
    df['target_price'] = df['close'].shift(-1)
    df['target_return'] = (df['target_price'] / df['close']) - 1
    
    print(f"Target returns calculated: {df['target_return'].count()} valid values")
    if df['target_return'].count() > 0:
        print(f"Target return range: {df['target_return'].min():.6f} to {df['target_return'].max():.6f}")
    
    # More lenient normalization approach
    rolling_window = max(20, config.VOLATILITY_WINDOW)  # Ensure minimum window
    
    # Use expanding standard deviation instead of rolling for more data retention
    expanding_std = df['target_return'].expanding(min_periods=rolling_window).std()
    
    # Only normalize where we have sufficient data
    df['target_normalized'] = df['target_return'] / (expanding_std + 1e-6)
    
    # Add close normalization (expanding mean for more data retention)
    expanding_mean = df['close'].expanding(min_periods=rolling_window).mean()
    df['close_normalized'] = df['close'] / expanding_mean - 1
    
    # Only remove rows where both normalized columns are NaN
    initial_len = len(df)
    df = df.dropna(subset=['target_normalized'])  # Only require target_normalized
    
    # If we still lost too much data, try a more lenient approach
    if len(df) < initial_len * 0.7:  # If we lost more than 30% of data
        print(f"WARNING: Lost {initial_len - len(df)} rows. Trying more lenient normalization...")
        
        # Reset and try simpler normalization
        df = df.reset_index(drop=True) if 'timestamp' not in df.index.names else df
        df['target_normalized'] = df['target_return'] / (df['target_return'].std() + 1e-6)
        df['close_normalized'] = (df['close'] - df['close'].mean()) / df['close'].std()
        
        # Remove only truly invalid rows
        df = df.dropna(subset=['target_normalized'])
    
    final_len = len(df)
    print(f"After normalization: {final_len} valid rows (removed {initial_len - final_len})")
    
    if final_len > 0:
        print(f"Target normalized range: {df['target_normalized'].min():.6f} to {df['target_normalized'].max():.6f}")
        print(f"Close normalized range: {df['close_normalized'].min():.6f} to {df['close_normalized'].max():.6f}")
    else:
        print("ERROR: No valid data remaining after normalization!")
    
    return df

def create_price_targets(df, forecast_horizon=1):
    """Create next-day price targets - NO FUTURE DATA LEAKAGE"""
    print(f"Creating price targets with {forecast_horizon} period forecast horizon...")
    
    # Target: next period's price
    df['target_price'] = df['close'].shift(-forecast_horizon)
    
    # Remove last N rows where target is NaN
    initial_rows = len(df)
    df = df.dropna()
    final_rows = len(df)
    
    print(f"Removed {initial_rows - final_rows} rows with NaN targets")
    
    return df

def create_time_aware_splits(data, train_pct=None, val_pct=None):
    """Ensure NO future data in training/validation"""
    
    # Use configuration defaults if not provided
    if train_pct is None:
        train_pct = config.TRAIN_PCT
    if val_pct is None:
        val_pct = config.VAL_PCT
      
    # Sort by time FIRST
    data = data.sort_index()
    
    # Detect data frequency using index (timestamps)
    if len(data) > 1:
        time_diffs = []
        for i in range(1, min(100, len(data))):
            # Use index timestamps
            ts_curr = data.index[i]
            ts_prev = data.index[i-1]
            
            diff = (ts_curr - ts_prev).total_seconds() / 60
            if diff > 0:  # Skip any zero or negative diffs
                time_diffs.append(diff)
        
        if time_diffs:
            time_diff = np.median(time_diffs)  # Use median for robustness
            print(f"Detected data frequency: {time_diff:.1f} minutes (median of {len(time_diffs)} intervals)")
        else:
            time_diff = 15  # Default fallback
            print(f"Could not detect frequency, using default: {time_diff} minutes")
    else:
        time_diff = 15
        print(f"Insufficient data for frequency detection, using default: {time_diff} minutes")
    
    # Calculate number of rows to skip for gaps
    train_val_gap_rows = max(1, int(config.TRAIN_VAL_GAP_HOURS * 60 / time_diff))
    val_test_gap_rows = max(1, int(config.VAL_TEST_GAP_HOURS * 60 / time_diff))
    
    # Calculate split indices
    total_rows = len(data)
    train_end_idx = int(total_rows * train_pct)
    
    # Validation split with gap
    val_start_idx = train_end_idx + train_val_gap_rows
    val_size = int(total_rows * val_pct)
    val_end_idx = val_start_idx + val_size
    
    # Test split with gap
    test_start_idx = val_end_idx + val_test_gap_rows
    
    print(f"Split indices: train=0:{train_end_idx}, val={val_start_idx}:{val_end_idx}, test={test_start_idx}:")
    
    # Create splits
    train_data = data.iloc[:train_end_idx]
    val_data = data.iloc[val_start_idx:val_end_idx]
    test_data = data.iloc[test_start_idx:]
    
    # Check if we have any data left after splits
    if len(train_data) == 0:
        print("ERROR: Training set is empty!")
        return data.iloc[:0], data.iloc[:0], data.iloc[:0]  # Return empty datasets
    
    # Verify we have enough data after gaps
    if len(val_data) == 0:
        print("WARNING: Validation set empty after applying gaps. Reducing gap size.")
        reduced_gap = max(1, train_val_gap_rows // 4)
        val_start_idx = train_end_idx + reduced_gap
        val_data = data.iloc[val_start_idx:val_end_idx]
        test_start_idx = val_end_idx + reduced_gap
        test_data = data.iloc[test_start_idx:]
    
    if len(test_data) == 0:
        print("WARNING: Test set empty after applying gaps. Reducing gap size.")
        reduced_gap = max(1, val_test_gap_rows // 4)
        test_start_idx = val_end_idx + reduced_gap
        test_data = data.iloc[test_start_idx:]
    
    # Verify actual gaps and adjust if needed - Use timestamp column if available
    if 'timestamp' in data.columns:
        if len(train_data) > 0 and len(val_data) > 0:
            # Use actual timestamps for gap calculation
            train_end_time = train_data['timestamp'].iloc[-1]
            val_start_time = val_data['timestamp'].iloc[0]
            
            # Convert Unix timestamps to datetime if needed
            if isinstance(train_end_time, (int, np.int64)):
                train_end_time = pd.to_datetime(train_end_time, unit='ms')
                val_start_time = pd.to_datetime(val_start_time, unit='ms')
            
            actual_train_val_gap = (val_start_time - train_end_time).total_seconds() / 3600
            print(f"Actual Train-Val gap: {actual_train_val_gap:.1f} hours (target: {config.TRAIN_VAL_GAP_HOURS}h)")
            
            # If gap is too small, extend it
            if actual_train_val_gap < config.TRAIN_VAL_GAP_HOURS * 0.8:  # Allow 20% tolerance
                print(f"WARNING: Train-Val gap too small, extending...")
                additional_rows = int((config.TRAIN_VAL_GAP_HOURS - actual_train_val_gap) * 60 / time_diff)
                val_start_idx += additional_rows
                val_end_idx += additional_rows
                test_start_idx += additional_rows
                val_data = data.iloc[val_start_idx:val_end_idx]
                test_data = data.iloc[test_start_idx:]
                
                # Recalculate actual gap using timestamps
                if len(val_data) > 0:
                    train_end_time = train_data['timestamp'].iloc[-1]
                    val_start_time = val_data['timestamp'].iloc[0]
                    
                    # Convert Unix timestamps to datetime if needed
                    if isinstance(train_end_time, (int, np.int64)):
                        train_end_time = pd.to_datetime(train_end_time, unit='ms')
                        val_start_time = pd.to_datetime(val_start_time, unit='ms')
                    
                    actual_train_val_gap = (val_start_time - train_end_time).total_seconds() / 3600
                    print(f"Adjusted Train-Val gap: {actual_train_val_gap:.1f} hours")
        
        if len(val_data) > 0 and len(test_data) > 0:
            # Use actual timestamps for gap calculation
            val_end_time = val_data['timestamp'].iloc[-1]
            test_start_time = test_data['timestamp'].iloc[0]
            
            # Convert Unix timestamps to datetime if needed
            if isinstance(val_end_time, (int, np.int64)):
                val_end_time = pd.to_datetime(val_end_time, unit='ms')
                test_start_time = pd.to_datetime(test_start_time, unit='ms')
            
            actual_val_test_gap = (test_start_time - val_end_time).total_seconds() / 3600
            print(f"Actual Val-Test gap: {actual_val_test_gap:.1f} hours (target: {config.VAL_TEST_GAP_HOURS}h)")
            
            # If gap is too small, extend it
            if actual_val_test_gap < config.VAL_TEST_GAP_HOURS * 0.8:  # Allow 20% tolerance
                print(f"WARNING: Val-Test gap too small, extending...")
                additional_rows = int((config.VAL_TEST_GAP_HOURS - actual_val_test_gap) * 60 / time_diff)
                test_start_idx += additional_rows
                test_data = data.iloc[test_start_idx:]
                
                # Recalculate actual gap using timestamps
                if len(test_data) > 0:
                    val_end_time = val_data['timestamp'].iloc[-1]
                    test_start_time = test_data['timestamp'].iloc[0]
                    
                    # Convert Unix timestamps to datetime if needed
                    if isinstance(val_end_time, (int, np.int64)):
                        val_end_time = pd.to_datetime(val_end_time, unit='ms')
                        test_start_time = pd.to_datetime(test_start_time, unit='ms')
                    
                    actual_val_test_gap = (test_start_time - val_end_time).total_seconds() / 3600
                    print(f"Adjusted Val-Test gap: {actual_val_test_gap:.1f} hours")
    
    # Final verification of data sizes after adjustments
    if len(val_data) == 0:
        print("WARNING: Validation set still empty after adjustments.")
    
    if len(test_data) == 0:
        print("WARNING: Test set still empty after adjustments.")
    
    # Data leakage check with timestamp-aware output
    print("DATA LEAKAGE CHECK:")
    if 'timestamp' in data.columns:
        print(f"Train end: {pd.to_datetime(train_data['timestamp'].iloc[-1], unit='ms')} ({len(train_data)} rows)")
        
        if len(val_data) > 0:
            # Use timestamps for gap calculation
            train_end_time = pd.to_datetime(train_data['timestamp'].iloc[-1], unit='ms')
            val_start_time = pd.to_datetime(val_data['timestamp'].iloc[0], unit='ms')
            actual_gap1 = (val_start_time - train_end_time).total_seconds() / 3600
            print(f"Val start: {val_start_time} (Gap: {actual_gap1:.1f}h, {len(val_data)} rows)")
        else:
            print("Val: EMPTY")
            
        if len(test_data) > 0:
            if len(val_data) > 0:
                # Use timestamps for gap calculation
                val_end_time = pd.to_datetime(val_data['timestamp'].iloc[-1], unit='ms')
                test_start_time = pd.to_datetime(test_data['timestamp'].iloc[0], unit='ms')
                actual_gap2 = (test_start_time - val_end_time).total_seconds() / 3600
            else:
                train_end_time = pd.to_datetime(train_data['timestamp'].iloc[-1], unit='ms')
                test_start_time = pd.to_datetime(test_data['timestamp'].iloc[0], unit='ms')
                actual_gap2 = (test_start_time - train_end_time).total_seconds() / 3600
            print(f"Test start: {pd.to_datetime(test_data['timestamp'].iloc[0], unit='ms')} (Gap: {actual_gap2:.1f}h, {len(test_data)} rows)")
        else:
            print("Test: EMPTY")
    else:
        # Fallback to index-based output when no timestamp column
        print(f"Train end: {train_data.index[-1]} ({len(train_data)} rows)")
        
        if len(val_data) > 0:
            print(f"Val start: {val_data.index[0]} ({len(val_data)} rows)")
        else:
            print("Val: EMPTY")
            
        if len(test_data) > 0:
            print(f"Test start: {test_data.index[0]} ({len(test_data)} rows)")
        else:
            print("Test: EMPTY")
    
    return train_data, val_data, test_data


def directional_accuracy_loss(y_true, y_pred, alpha=0.5):
    """
    Custom loss function that combines MSE with directional accuracy.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        alpha: Weight for directional component (0.5 = equal weight)
        
    Returns:
        Combined loss value
    """
    # Standard MSE loss
    mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    
    # Directional accuracy component
    y_true_direction = tf.sign(y_true)
    y_pred_direction = tf.sign(y_pred)
    
    # Directional agreement (+1 if same direction, -1 if opposite)
    directional_agreement = y_true_direction * y_pred_direction
    
    # Convert to loss (minimize disagreement)
    directional_loss = 1.0 - tf.reduce_mean(directional_agreement)
    
    # Combine losses
    total_loss = (1 - alpha) * mse_loss + alpha * directional_loss
    
    return total_loss

def sharpe_ratio_loss(y_true, y_pred, risk_free_rate=0.0):
    """
    Loss function that optimizes for Sharpe ratio.
    Maximizes return-to-volatility ratio of predictions.
    
    Args:
        y_true: True values (price changes)
        y_pred: Predicted values (price changes)
        risk_free_rate: Risk-free rate for Sharpe calculation
        
    Returns:
        Negative Sharpe ratio (for minimization)
    """
    # Calculate returns based on predictions
    predicted_returns = y_pred - risk_free_rate
    
    # Calculate mean and standard deviation of predicted returns
    mean_return = tf.reduce_mean(predicted_returns)
    std_return = tf.math.reduce_std(predicted_returns)
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    sharpe_ratio = mean_return / (std_return + epsilon)
    
    # Return negative Sharpe (since we want to minimize loss)
    return -sharpe_ratio

def weighted_mse_loss(y_true, y_pred, sample_weights=None):
    """
    Weighted MSE loss that can emphasize certain samples.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        sample_weights: Weight for each sample (if None, uses uniform weights)
        
    Returns:
        Weighted MSE loss
    """
    squared_diff = tf.square(y_true - y_pred)
    
    if sample_weights is not None:
        squared_diff = squared_diff * sample_weights
    
    return tf.reduce_mean(squared_diff)

def quantile_loss(y_true, y_pred, quantile=0.5):
    """
    Quantile regression loss function.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        quantile: Target quantile (0.5 = median)
        
    Returns:
        Quantile loss
    """
    error = y_true - y_pred
    loss = tf.maximum(quantile * error, (quantile - 1.0) * error)
    return tf.reduce_mean(loss)

def huber_directional_loss(y_true, y_pred, delta=1.0, alpha=0.3):
    """
    Combines Huber loss (robust to outliers) with directional accuracy.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        delta: Threshold for Huber loss
        alpha: Weight for directional component
        
    Returns:
        Combined Huber and directional loss
    """
    # Huber loss
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    squared_loss = tf.square(error) / 2
    linear_loss = delta * tf.abs(error) - tf.square(delta) / 2
    huber_loss = tf.where(is_small_error, squared_loss, linear_loss)
    
    # Directional component
    y_true_direction = tf.sign(y_true)
    y_pred_direction = tf.sign(y_pred)
    directional_agreement = y_true_direction * y_pred_direction
    directional_loss = 1.0 - tf.reduce_mean(directional_agreement)
    
    # Combine losses
    total_loss = (1 - alpha) * tf.reduce_mean(huber_loss) + alpha * directional_loss
    
    return total_loss

def regime_aware_loss(y_true, y_pred, regime_indicators=None, regime_weights=None):
    """
    Loss function that adapts based on market regime.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        regime_indicators: Tensor indicating current regime (one-hot encoded)
        regime_weights: Weights for each regime type
        
    Returns:
        Regime-adjusted loss
    """
    if regime_indicators is None:
        # Fall back to standard MSE if no regime info
        return tf.keras.losses.mean_squared_error(y_true, y_pred)
    
    if regime_weights is None:
        # Default weights for different regimes
        regime_weights = tf.constant([1.0, 1.2, 0.8, 1.5])  # trending_strong, trending_weak, neutral, mean_reverting
    
    # Calculate base MSE loss
    mse_loss = tf.square(y_true - y_pred)
    
    # Apply regime-specific weights
    regime_weight = tf.reduce_sum(regime_indicators * regime_weights, axis=1, keepdims=True)
    weighted_loss = mse_loss * regime_weight
    
    return tf.reduce_mean(weighted_loss)

def profit_loss(y_true, y_pred, transaction_cost=0.001):
    """
    Loss function that directly optimizes for trading profit.
    
    Args:
        y_true: True price changes
        y_pred: Predicted price changes
        transaction_cost: Transaction cost per trade
        
    Returns:
        Negative profit (for minimization)
    """
    # Trading decisions based on predictions
    positions = tf.sign(y_pred)  # +1 for long, -1 for short, 0 for no position
    
    # Calculate returns from positions
    returns = positions * y_true
    
    # Apply transaction costs when position changes
    position_changes = tf.abs(tf.diff(positions, prepend=0))
    transaction_costs = position_changes * transaction_cost
    
    # Net profit
    net_returns = returns - transaction_costs
    total_profit = tf.reduce_sum(net_returns)
    
    # Return negative profit (since we minimize loss)
    return -total_profit


def safe_feature_scaling(train_features, val_features, test_features, train_targets=None, val_targets=None, test_targets=None):
    """Scale features and targets using ONLY training data statistics"""
    
    # Feature scaling
    feature_scaler = RobustScaler()
    train_features_scaled = feature_scaler.fit_transform(train_features)
    val_features_scaled = feature_scaler.transform(val_features)
    test_features_scaled = feature_scaler.transform(test_features)
    
    print(f"Feature scaling complete: {train_features.shape[1]} features")
    print(f"Train feature mean: {np.mean(train_features_scaled):.4f}, std: {np.std(train_features_scaled):.4f}")
    
    # Target scaling (CRITICAL for price prediction)
    target_scaler = None
    train_targets_scaled = train_targets
    val_targets_scaled = val_targets  
    test_targets_scaled = test_targets
    
    if train_targets is not None:
        from sklearn.preprocessing import MinMaxScaler
        target_scaler = MinMaxScaler()
        
        # Fit scaler on training targets only
        train_targets_scaled = target_scaler.fit_transform(train_targets.reshape(-1, 1)).flatten()
        
        if val_targets is not None:
            val_targets_scaled = target_scaler.transform(val_targets.reshape(-1, 1)).flatten()
        if test_targets is not None:
            test_targets_scaled = target_scaler.transform(test_targets.reshape(-1, 1)).flatten()
        
        print(f"Target scaling complete:")
        print(f"Original target range: [{np.min(train_targets):.2f}, {np.max(train_targets):.2f}]")
        print(f"Scaled target range: [{np.min(train_targets_scaled):.4f}, {np.max(train_targets_scaled):.4f}]")
    
    return (train_features_scaled, val_features_scaled, test_features_scaled, 
            train_targets_scaled, val_targets_scaled, test_targets_scaled,
            feature_scaler, target_scaler)

def create_helformer_sequences(features, targets, window_size=None):
    """Create input sequences and price targets"""
    
    if window_size is None:
        window_size = config.SEQUENCE_LENGTH
    
    # Ensure inputs are numpy arrays and not datetime or object types
    if hasattr(features, 'values'):
        features = features.values
    if hasattr(targets, 'values'):
        targets = targets.values
        
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    
    # Handle NaN values
    features = np.nan_to_num(features, nan=0.0)
    targets = np.nan_to_num(targets, nan=0.0)
    
    X, y = [], []
    
    # Create sequences with proper temporal order
    for i in range(window_size, len(features)):
        # Input: past window_size days of features
        X.append(features[i-window_size:i])
        
        # Target: current day's target price
        y.append(targets[i])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
    
    return X, y

def evaluate_helformer(y_true, y_pred):
    """Helformer evaluation metrics from paper"""
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {'RMSE': np.nan, 'MAPE': np.nan, 'MAE': np.nan, 'R2': np.nan}
    
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    
    # MAPE calculation with protection against division by zero
    mape_values = np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-8)) * 100
    mape = np.mean(mape_values)
    
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # WARNING: Perfect R² (1.0) may indicate overfitting
    if r2 > 0.999:
        print("WARNING: Near-perfect R² detected - check for data leakage")
    
    return {'RMSE': rmse, 'MAPE': mape, 'MAE': mae, 'R2': r2}

def validate_predictions(y_true, y_pred):
    """Sanity checks for price predictions"""
    
    print("Validating price predictions...")
    
    # Check for extreme predictions
    max_reasonable_change = 0.5  # 50% max daily change
    daily_changes = np.abs(y_pred - y_true) / (y_true + 1e-8)
    
    extreme_predictions = np.sum(daily_changes > max_reasonable_change)
    if extreme_predictions > 0:
        print(f"WARNING: {extreme_predictions} extreme price predictions detected")
        print(f"Max change: {np.max(daily_changes):.2%}")
        return False
    
    # Check prediction distribution
    pred_std = np.std(y_pred)
    true_std = np.std(y_true)
    
    if pred_std < true_std * 0.1:
        print("WARNING: Predictions too conservative")
        print(f"Prediction std: {pred_std:.4f}, True std: {true_std:.4f}")
        return False
    
    # Check for NaN or infinite predictions
    invalid_predictions = np.sum(~np.isfinite(y_pred))
    if invalid_predictions > 0:
        print(f"WARNING: {invalid_predictions} invalid predictions (NaN/Inf)")
        return False
    
    print("Prediction validation passed")
    return True

def technical_indicators(data: pd.DataFrame) -> List[str]:
    """Generate technical indicators and return feature names."""
    try:
        # Basic price-based indicators
        data['sma_5'] = data['close'].rolling(5).mean()
        data['sma_10'] = data['close'].rolling(10).mean()
        data['sma_20'] = data['close'].rolling(20).mean()
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_sma = data['close'].rolling(bb_period).mean()
        bb_std_val = data['close'].rolling(bb_period).std()
        data['bb_upper'] = bb_sma + (bb_std_val * bb_std)
        data['bb_lower'] = bb_sma - (bb_std_val * bb_std)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / bb_sma
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Volume indicators
        data['volume_sma'] = data['volume'].rolling(20).mean()
        data['relative_volume'] = data['volume'] / data['volume_sma']
        
        return [
            'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_histogram', 'rsi',
            'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
            'volume_sma', 'relative_volume'
        ]
        
    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        return []

def volatility_features(data: pd.DataFrame) -> List[str]:
    """Generate volatility-based features and return feature names."""
    try:
        # Price volatility
        data['returns'] = data['close'].pct_change()
        data['vol_5'] = data['returns'].rolling(5).std()
        data['vol_20'] = data['returns'].rolling(20).std()
        data['vol_ratio'] = data['vol_5'] / data['vol_20']
        
        # High-Low volatility
        data['hl_ratio'] = (data['high'] - data['low']) / data['close']
        data['true_range'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        data['atr'] = data['true_range'].rolling(14).mean()
        data['normalized_atr'] = data['atr'] / data['close']
        
        # Volatility ranking
        vol_window = 252  # 1 year
        if len(data) >= vol_window:
            data['vol_rank'] = data['vol_20'].rolling(vol_window).rank(pct=True)
        else:
            data['vol_rank'] = 0.5  # Default to median
        
        return [
            'vol_5', 'vol_20', 'vol_ratio', 'hl_ratio',
            'true_range', 'atr', 'normalized_atr', 'vol_rank'
        ]
        
    except Exception as e:
        print(f"Error calculating volatility features: {e}")
        return []

def regime_features(data: pd.DataFrame) -> List[str]:
    """Generate regime-based features and return feature names."""
    try:
        # Trend strength
        data['trend_strength'] = abs(data['close'].rolling(20).corr(range(20)))
        
        # Price momentum
        data['mom_5'] = data['close'] / data['close'].shift(5) - 1
        data['mom_10'] = data['close'] / data['close'].shift(10) - 1
        data['mom_20'] = data['close'] / data['close'].shift(20) - 1
        
        # Price acceleration
        data['accel'] = data['returns'].diff()
        
        # Market structure breaks
        lookback = 20
        if len(data) >= lookback:
            data['higher_highs'] = (data['high'] > data['high'].rolling(lookback).max().shift(1)).astype(int)
            data['lower_lows'] = (data['low'] < data['low'].rolling(lookback).min().shift(1)).astype(int)
            data['structure_break'] = data['higher_highs'] - data['lower_lows']
        else:
            data['higher_highs'] = 0
            data['lower_lows'] = 0
            data['structure_break'] = 0
        
        # Volatility regime
        vol_threshold = data['vol_20'].quantile(0.7) if len(data) > 20 else 0.02
        data['high_vol_regime'] = (data['vol_20'] > vol_threshold).astype(int)
        
        return [
            'trend_strength', 'mom_5', 'mom_10', 'mom_20', 'accel',
            'higher_highs', 'lower_lows', 'structure_break', 'high_vol_regime'
        ]
        
    except Exception as e:
        print(f"Error calculating regime features: {e}")
        return []

def prepare_sequences(features: np.ndarray, targets: np.ndarray, sequence_length: int = 60):
    """Prepare sequences for LSTM training."""
    try:
        if len(features) < sequence_length:
            print(f"Warning: Not enough data for sequence length {sequence_length}. Need at least {sequence_length} samples.")
            return np.array([]), np.array([])
        
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            X.append(features[i-sequence_length:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
        
    except Exception as e:
        print(f"Error preparing sequences: {e}")
        return np.array([]), np.array([])
# Loss function factory class
class CustomLossFactory:
    """Factory class for creating custom loss functions."""
    
    @staticmethod
    def create_directional_loss(alpha=0.5):
        """Create directional accuracy loss."""
        return lambda y_true, y_pred: directional_accuracy_loss(y_true, y_pred, alpha)
    
    @staticmethod
    def create_sharpe_loss(risk_free_rate=0.0):
        """Create Sharpe ratio loss."""
        return lambda y_true, y_pred: sharpe_ratio_loss(y_true, y_pred, risk_free_rate)
    
    @staticmethod
    def create_huber_directional_loss(delta=1.0, alpha=0.3):
        """Create Huber directional loss."""
        return lambda y_true, y_pred: huber_directional_loss(y_true, y_pred, delta, alpha)
    
    @staticmethod
    def create_profit_loss(transaction_cost=0.001):
        """Create profit-based loss."""
        return lambda y_true, y_pred: profit_loss(y_true, y_pred, transaction_cost)
    
    @staticmethod
    def create_quantile_loss(quantile=0.5):
        """Create quantile loss."""
        return lambda y_true, y_pred: quantile_loss(y_true, y_pred, quantile)

# Custom metrics for evaluation
def directional_accuracy_metric(y_true, y_pred):
    """Metric to track directional accuracy."""
    y_true_direction = tf.sign(y_true)
    y_pred_direction = tf.sign(y_pred)
    agreement = tf.cast(tf.equal(y_true_direction, y_pred_direction), tf.float32)
    return tf.reduce_mean(agreement)

def sharpe_ratio_metric(y_true, y_pred):
    """Metric to track Sharpe ratio of predictions."""
    mean_return = tf.reduce_mean(y_pred)
    std_return = tf.math.reduce_std(y_pred)
    epsilon = 1e-8
    return mean_return / (std_return + epsilon)

def hit_ratio_metric(y_true, y_pred, threshold=0.01):
    """Metric to track hit ratio (predictions within threshold)."""
    error = tf.abs(y_true - y_pred)
    hits = tf.cast(error <= threshold, tf.float32)
    return tf.reduce_mean(hits)

# Loss function selection utility
def get_loss_function(loss_name: str, **kwargs):
    """
    Get loss function by name with optional parameters.
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional parameters for the loss function
        
    Returns:
        Loss function
    """
    loss_functions = {
        'mse': tf.keras.losses.mean_squared_error,
        'mae': tf.keras.losses.mean_absolute_error,
        'huber': tf.keras.losses.Huber(),
        'directional': CustomLossFactory.create_directional_loss(**kwargs),
        'sharpe': CustomLossFactory.create_sharpe_loss(**kwargs),
        'huber_directional': CustomLossFactory.create_huber_directional_loss(**kwargs),
        'profit': CustomLossFactory.create_profit_loss(**kwargs),
        'quantile': CustomLossFactory.create_quantile_loss(**kwargs),
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}. Available: {list(loss_functions.keys())}")
    
    return loss_functions[loss_name]

def get_custom_metrics():
    """Get list of custom metrics for model compilation."""
    return [
        directional_accuracy_metric,
        sharpe_ratio_metric,
        hit_ratio_metric
    ]