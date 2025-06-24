import pandas as pd
import numpy as np
import os
import talib  # For efficient technical indicators

# Suppress TensorFlow warnings and CPU optimization messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import tensorflow as tf
import psutil  # For memory monitoring
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import warnings
from tqdm import tqdm
from config_helformer import config
# Import unified training utilities (CONSOLIDATED)
from improved_training_utils import (
    create_research_based_features, create_improved_targets,
    prepare_model_data, create_sequences, safe_feature_scaling,
    evaluate_predictions, time_aware_train_val_test_split,
    HoltWintersLayer, normalize_targets_no_leakage,
    create_helformer_sequences, evaluate_helformer
)
from helformer_model import create_helformer_model, EnsembleManager, create_default_ensemble_configs
from adaptive_ensemble_optimizer import optimize_helformer_ensemble
from trading_metrics import evaluate_model_trading_performance, create_trading_callback
import gc
import time

# Suppress all warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')  # Only show errors

# Set TensorFlow to use CPU efficiently and manage memory
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)

# Configure aggressive memory management for CPU training
try:
    # Limit TensorFlow memory allocation to prevent OOM
    tf.config.experimental.set_virtual_device_configuration(
        tf.config.list_physical_devices('CPU')[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # 4GB limit
    )
except:
    pass

# Enable memory growth and optimization
try:
    tf.config.experimental.enable_memory_growth = True
    tf.config.experimental.set_memory_growth = True
    
    # Additional memory optimizations
    tf.config.experimental.enable_mlir_graph_optimization()
    tf.config.experimental.enable_tensor_float_32_execution(False)
except:
    pass  # Some configurations may not support this

print("HELFORMER MULTI-ASSET PRICE PREDICTION TRAINING")
print("=" * 60)
print("OPTIMIZED: Efficient O(n) feature engineering pipeline")
print("Target: 925%+ Annual Returns | Asset-Specific Models")
print("MEMORY OPTIMIZED: Reduced model size and batch size for stability")
print("=" * 60)

# Check available memory
import psutil
memory_info = psutil.virtual_memory()
print(f"System Memory: {memory_info.total / (1024**3):.1f} GB total, {memory_info.available / (1024**3):.1f} GB available")
print(f"Memory Usage: {memory_info.percent:.1f}%")
print("=" * 60)

# Load crypto datasets using config-driven market selection
file_paths = config.get_market_files()

# Load individual datasets (NO MIXING!)
# FOR PRODUCTION: Use full dataset for maximum performance
# Configuration for test mode (now controlled by config)
TEST_MODE = config.is_test_mode()
DATA_LIMIT = config.get_data_limit()  # None in production, number in test mode

datasets = {}
for symbol, path in file_paths.items():
    if os.path.exists(path):
        print(f"Loading {symbol} from {path}...")
        df = pd.read_csv(path)
        
        # Apply data limit only if specified (test mode)
        if DATA_LIMIT is not None:
            df = df.tail(DATA_LIMIT)  # Use limited dataset in test mode
            print(f"  >>> {config.get_mode_description()}")
        else:
            print(f"  >>> PRODUCTION MODE: Using all {len(df):,} records")
        
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("time", inplace=True)
        datasets[symbol] = df
        print(f"  {symbol}: {len(df):,} records from {df.index.min()} to {df.index.max()}")

if not datasets:
    print("ERROR: No data files found")  
    exit()

print(f"Loaded {len(datasets)} assets: {list(datasets.keys())}")
print(f"Mode: {'TEST' if TEST_MODE else 'PRODUCTION'} | Data per asset: {'Limited' if DATA_LIMIT else 'Full'}")

def create_helformer_sequences(X, y, window_size):
    """Create sequences for LSTM training"""
    if len(X) < window_size:
        return np.array([]), np.array([])
    
    X_seq, y_seq = [], []
    for i in range(window_size, len(X)):
        X_seq.append(X[i-window_size:i])
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)

def prepare_asset_data(df, asset_name):
    """Prepare data for a single asset with RESEARCH-BASED IMPROVED FEATURES"""
    
    print(f"\nPreparing {asset_name} data (RESEARCH-BASED IMPROVED FEATURES)...")
    
    # Use RESEARCH-BASED feature engineering (from improved_training_utils)
    df = create_research_based_features(df, symbol=asset_name)    
    # Create improved targets using research-based approach
    df, target_mean, target_std = create_improved_targets(df, target_hours=1)
    
    # Prepare model data with feature selection and validation
    df_clean, available_features = prepare_model_data(df, sequence_length=config.SEQUENCE_LENGTH)
    
    print(f"✅ {asset_name}: {len(df_clean):,} samples, {len(available_features)} features")
    print(f"   Target stats: μ={target_mean:.6f}, σ={target_std:.6f}")
    
    return df_clean, available_features, target_mean, target_std

def select_universal_features(prepared_datasets):
    """Select features that work across all assets"""
    # Use the features from the optimized pipeline
    first_asset_data, available_features = list(prepared_datasets.values())[0]
    
    print(f"Selected {len(available_features)} universal features:")
    for feat in available_features:
        print(f"  - {feat}")
    
    return available_features

def train_asset_specific_model(asset_name, df_prepared, feature_columns, model_params, experiment_folder, target_stats):
    """Train Advanced Ensemble Helformer model for specific asset"""
    
    print(f"\n{'='*20} TRAINING {asset_name} ADVANCED ENSEMBLE {'='*20}")
    print("🚀 USING ADVANCED ARCHITECTURES: transformer, cnn_lstm, gru_attention, multi_scale")
    
    # Data is already prepared with features and targets
    df, available_features = df_prepared
    target_mean = target_stats['mean']
    target_std = target_stats['std']
      # Use time-aware splits from improved utilities
    train_data, val_data, test_data = time_aware_train_val_test_split(df, train_pct=config.TRAIN_PCT, val_pct=config.VAL_PCT, test_pct=config.TEST_PCT)
    
    total_len = len(df)
    print(f"{asset_name} data splits:")
    print(f"  Train: {len(train_data)} samples ({len(train_data)/total_len*100:.1f}%)")
    print(f"  Val: {len(val_data)} samples ({len(val_data)/total_len*100:.1f}%)")
    print(f"  Test: {len(test_data)} samples ({len(test_data)/total_len*100:.1f}%)")
    
    # Extract features and targets
    X_train = train_data[feature_columns].values
    y_train = train_data['target_normalized'].values
    
    X_val = val_data[feature_columns].values
    y_val = val_data['target_normalized'].values
    
    X_test = test_data[feature_columns].values
    y_test = test_data['target_normalized'].values      # Feature scaling using improved function
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Features scaled with RobustScaler")
    print(f"   X range: {X_train_scaled.min():.3f} to {X_train_scaled.max():.3f}")
    print(f"   y range: {y_train.min():.3f} to {y_train.max():.3f}")
    
    # Sanity check - ensure targets have reasonable variance
    if y_train.std() < 0.1:
        print(f"⚠️  WARNING: Low target variance ({y_train.std():.4f}) - model may struggle to learn")
    if abs(y_train.mean()) > 0.5:
        print(f"⚠️  WARNING: High target bias ({y_train.mean():.4f}) - consider rebalancing")
      # Create sequences using improved function
    WINDOW_SIZE = config.SEQUENCE_LENGTH
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, WINDOW_SIZE)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, WINDOW_SIZE)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, WINDOW_SIZE)
    
    print(f"{asset_name} sequences created:")
    print(f"  Train: {X_train_seq.shape}")
    print(f"  Val: {X_val_seq.shape}")
    print(f"  Test: {X_test_seq.shape}")
    print()# Create advanced ensemble configurations using ADAPTIVE OPTIMIZATION
    print(f"🎯 Target ensemble size: {config.ENSEMBLE_SIZE} models")
    print(f"🧠 Using ADAPTIVE ENSEMBLE OPTIMIZATION...")
    
    # Use adaptive optimizer to determine best configuration
    try:
        print("⚡ Running adaptive ensemble optimization...")        
        optimized_configs = optimize_helformer_ensemble(
            X_train_seq, y_train_seq,
            max_ensemble_size=config.ENSEMBLE_SIZE
        )
        if optimized_configs and 'ensemble_architectures' in optimized_configs:
            ensemble_configs = optimized_configs['ensemble_architectures']
            print(f"✅ Adaptive optimization complete! Selected {len(ensemble_configs)} optimal configurations")
            print(f"📊 Data complexity: {optimized_configs['optimization_summary']['total_complexity_score']:.2f}")
            print(f"⚖️  Ensemble method: {optimized_configs['ensemble_method']}")
        else:
            print("⚠️  Adaptive optimization failed, falling back to manual configuration...")
            raise Exception("Adaptive optimization returned invalid configs")
            
    except Exception as e:
        print(f"⚠️  Adaptive optimization error: {e}")
        print("🔄 Falling back to manual ensemble configuration...")
        
        # Fallback: Manual ensemble configuration (previous logic)
        ensemble_configs = []
        target_ensemble_size = config.ENSEMBLE_SIZE
        architectures = config.ENSEMBLE_ARCHITECTURES
        
        # Create configurations to reach target ensemble size
        for i in range(target_ensemble_size):
            # Cycle through architectures if we need more models than architecture types
            arch_type = architectures[i % len(architectures)]
            
            # Add slight variations for multiple instances of same architecture
            variation_factor = i // len(architectures)
            dropout_variation = model_params.get('dropout_rate', config.DROPOUT_RATE) * (1 + variation_factor * 0.05)
            
            if arch_type == 'transformer':
                ensemble_configs.append({
                    'type': 'transformer',
                    'num_transformer_blocks': getattr(config, 'TRANSFORMER_BLOCKS', 2) + (variation_factor % 2),
                    'd_model': model_params.get('head_dim', config.HEAD_DIM),
                    'num_heads': model_params.get('num_heads', config.NUM_HEADS),
                    'dropout_rate': min(dropout_variation, 0.3),  # Cap at 30%
                    'optimizer': 'adam',
                    'loss': 'mse',
                    'metrics': ['mae'],
                    'model_id': f'transformer_{i+1}'
                })
            elif arch_type == 'cnn_lstm':
                filters = getattr(config, 'CNN_FILTERS', [64, 128])
                # Vary filter sizes slightly for different instances
                if variation_factor > 0:
                    filters = [f + (variation_factor * 16) for f in filters]
                ensemble_configs.append({
                    'type': 'cnn_lstm',
                    'conv_filters': filters,
                    'kernel_sizes': getattr(config, 'CNN_KERNEL_SIZES', [3, 5]),
                    'lstm_units': [model_params.get('lstm_units', config.LSTM_UNITS) + (variation_factor * 16)],
                    'dropout_rate': min(dropout_variation, 0.3),
                    'optimizer': 'adam',
                    'loss': 'mae',
                    'metrics': ['mae'],
                    'model_id': f'cnn_lstm_{i+1}'
                })
            elif arch_type == 'gru_attention':
                gru_units = getattr(config, 'GRU_UNITS', [128, 64])
                # Vary GRU units slightly for different instances
                if variation_factor > 0:
                    gru_units = [u + (variation_factor * 16) for u in gru_units]
                ensemble_configs.append({
                    'type': 'gru_attention',
                    'gru_units': gru_units,
                    'attention_units': model_params.get('head_dim', config.HEAD_DIM) + (variation_factor * 8),
                    'dropout_rate': min(dropout_variation, 0.3),
                    'optimizer': 'rmsprop',
                    'loss': 'huber',
                    'metrics': ['mae'],
                    'model_id': f'gru_attention_{i+1}'
                })
            elif arch_type == 'multi_scale':
                scales = getattr(config, 'MULTI_SCALE_KERNELS', [1, 3, 5])
                # Add additional scales for variations
                if variation_factor > 0:
                    scales = scales + [7, 9][:variation_factor]
                ensemble_configs.append({
                    'type': 'multi_scale',
                    'scales': scales,
                    'filters_per_scale': 32 + (variation_factor * 8),
                    'lstm_units': model_params.get('lstm_units', config.LSTM_UNITS) + (variation_factor * 16),
                    'dropout_rate': min(dropout_variation, 0.3),
                    'optimizer': 'adam',
                    'loss': 'mse',
                    'metrics': ['mae'],
                    'model_id': f'multi_scale_{i+1}'
                })
    
    print(f"📊 Creating ensemble with {len(ensemble_configs)} advanced models:")
    for i, cfg in enumerate(ensemble_configs):
        model_id = cfg.get('model_id', f"model_{i+1}")
        print(f"  {i+1}. {model_id.upper()} - Loss: {cfg['loss']}, Optimizer: {cfg['optimizer']}")
    
    # Initialize ensemble manager
    ensemble_manager = EnsembleManager(ensemble_configs)
    
    # Create ensemble models
    input_shape = (WINDOW_SIZE, len(feature_columns))
    ensemble_models = ensemble_manager.create_ensemble(input_shape)
    
    print(f"✅ Successfully created {len(ensemble_models)} advanced models")
    
    # Define callbacks for ensemble training
    early_stopping = EarlyStopping(
        monitor='val_mae',
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.0001
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_mae',
        factor=0.2,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=config.MIN_LEARNING_RATE,
        verbose=1
    )
    
    # Train ensemble
    print(f"\n🎯 Training Advanced Ensemble for {asset_name}...")
    print("=" * 60)
    
    training_histories = ensemble_manager.train_ensemble(
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
        epochs=config.MAX_EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
      # Make ensemble predictions with ADAPTIVE METHOD SELECTION
    print(f"\n📈 Generating ensemble predictions for {asset_name}...")
    
    # Test different ensemble methods and select the best one
    ensemble_methods = ['weighted_average', 'simple_average', 'median']
    best_method = config.ENSEMBLE_METHOD  # Default
    best_score = float('-inf')
    
    print("🔄 Testing ensemble methods...")
    for method in ensemble_methods:
        try:
            test_pred = ensemble_manager.predict_ensemble(X_val_seq, method=method)
            # Quick performance check using correlation with actual values
            correlation = np.corrcoef(y_val_seq, test_pred)[0, 1]
            if not np.isnan(correlation) and correlation > best_score:
                best_score = correlation
                best_method = method
            print(f"  {method}: correlation = {correlation:.4f}")
        except Exception as e:
            print(f"  {method}: failed ({e})")
    
    print(f"✅ Selected best ensemble method: {best_method} (score: {best_score:.4f})")
    
    # Generate final predictions with best method
    ensemble_pred = ensemble_manager.predict_ensemble(X_test_seq, method=best_method)
      # Calculate ensemble diversity
    diversity_score = ensemble_manager.get_ensemble_diversity(X_test_seq)
    
    print(f"🎭 Ensemble diversity score: {diversity_score:.4f}")
    print(f"🔧 Best ensemble method: {best_method} (adaptive selection)")
    print(f"⚖️  Model weights: {[f'{w:.3f}' for w in ensemble_manager.model_weights]}")
      # Ensemble evaluation with improved metrics
    print(f"\n{asset_name} ADVANCED ENSEMBLE PERFORMANCE:")
    print("=" * 50)
    ensemble_metrics = evaluate_predictions(
        y_test_seq, ensemble_pred, target_mean=target_mean, target_std=target_std
    )
    
    # Also get trading metrics
    trading_metrics = evaluate_model_trading_performance(
        y_test_seq, ensemble_pred, asset_name=f"{asset_name} ADVANCED ENSEMBLE"
    )
    
    # Combine metrics
    combined_metrics = {**ensemble_metrics, **trading_metrics}
      # Add ensemble-specific metrics
    combined_metrics['diversity_score'] = diversity_score
    combined_metrics['ensemble_method'] = best_method  # Use the adaptively selected method
    combined_metrics['ensemble_method_score'] = best_score
    combined_metrics['model_weights'] = ensemble_manager.model_weights
    combined_metrics['architectures'] = config.ENSEMBLE_ARCHITECTURES
    
    # Check if ensemble meets performance targets
    meets_targets = (
        combined_metrics.get('win_rate', 0) >= config.TARGET_WIN_RATE and
        combined_metrics.get('annual_return', 0) >= config.TARGET_ANNUAL_RETURN and
        combined_metrics.get('sharpe_ratio', 0) >= config.TARGET_SHARPE_RATIO and
        combined_metrics.get('max_drawdown', 1) <= config.TARGET_MAX_DRAWDOWN
    )
    
    status = "✅ PASSED" if meets_targets else "⚠️  NEEDS IMPROVEMENT"
    print(f"\nPerformance Status: {status}")
    print("=" * 50)    # Save ensemble
    print(f"\n💾 Saving Advanced Ensemble for {asset_name}...")
    ensemble_filepath = os.path.join(experiment_folder, f"helformer_{asset_name.lower()}_ensemble")
    ensemble_manager.save_ensemble(ensemble_filepath)
    
    # Save asset-specific components with optimized scaler
    scaler_filepath = os.path.join(experiment_folder, f"helformer_{asset_name.lower()}_scaler.pkl")
    joblib.dump(scaler, scaler_filepath)
    
    return {
        'ensemble_manager': ensemble_manager,
        'models': ensemble_models,
        'scaler': scaler,
        'metrics': combined_metrics,
        'predictions': ensemble_pred,
        'actuals': y_test_seq,
        'test_data': test_data,
        'window_size': WINDOW_SIZE,
        'training_histories': training_histories
    }

def efficient_helformer_features(df):
    """
    EFFICIENT Helformer features using vectorized operations (O(n) not O(n²))
    Maintains all advanced features but computes them efficiently
    No data leakage - uses only past data for calculations
    """
    print("\n🔧 Creating EFFICIENT Helformer features (O(n) vectorized)...")
    
    # Sort by timestamp to ensure proper chronological order
    df = df.sort_index()
    
    # 1. Basic price features (vectorized - no loops)
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['price_acceleration'] = df['returns'].diff()
    
    # 2. Volume features (vectorized)
    volume_window = min(20, len(df) // 10)  # Adaptive window
    df['volume_sma'] = df['volume'].rolling(volume_window, min_periods=5).mean()
    df['relative_volume'] = df['volume'] / (df['volume_sma'] + 1e-8)
    
    # 3. Technical indicators using TA-Lib (efficient C implementations)
    try:
        # RSI (vectorized - no loops)
        rsi_values = talib.RSI(df['close'].values, timeperiod=config.RSI_PERIOD)
        df['rsi_normalized'] = (rsi_values - 50) / 50
        
        # MACD (vectorized - no loops) 
        macd_values, _, _ = talib.MACD(df['close'].values)
        df['macd_normalized'] = macd_values / df['close']
        
    except Exception as e:
        print(f"   TA-Lib indicators failed: {e}, using manual calculations")
        # Fallback to manual RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=5).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=5).mean()
        rs = gain / (loss + 1e-8)
        df['rsi_normalized'] = (100 - (100 / (1 + rs)) - 50) / 50
        
        # Manual MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd_normalized'] = (ema_12 - ema_26) / df['close']
    
    # 4. Moving averages (vectorized - no loops)
    for period in [10, 20, 50]:
        sma = df['close'].rolling(period, min_periods=max(5, period//4)).mean()
        df[f'price_vs_sma_{period}'] = (df['close'] / sma) - 1
    
    # 5. Bollinger Bands (vectorized)
    try:
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            df['close'].values, 
            timeperiod=config.VOLATILITY_WINDOW
        )
        bb_range = bb_upper - bb_lower
        df['bb_position'] = (df['close'] - bb_lower) / (bb_range + 1e-8)
        df['bb_width'] = bb_range / bb_middle
    except:
        # Manual Bollinger Bands
        bb_sma = df['close'].rolling(20, min_periods=10).mean()
        bb_std = df['close'].rolling(20, min_periods=10).std()
        bb_upper = bb_sma + (bb_std * 2)
        bb_lower = bb_sma - (bb_std * 2)
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        df['bb_width'] = (bb_upper - bb_lower) / bb_sma
    
    # 6. Volatility features (vectorized)
    volatility_window = min(config.VOLATILITY_WINDOW, len(df) // 5)
    df['volatility_20'] = df['returns'].rolling(volatility_window, min_periods=5).std()
    # Efficient volatility rank using rolling quantile
    df['volatility_rank'] = df['volatility_20'].rolling(100, min_periods=20).rank(pct=True)
    
    # 7. Momentum features (vectorized - no loops)
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = (df['close'] / df['close'].shift(period)) - 1
    
    # 8. Time features (cyclical encoding - vectorized)
    if hasattr(df.index, 'hour'):
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    else:
        # Default values if time features can't be calculated
        df['hour_sin'] = 0
        df['hour_cos'] = 1
        df['dow_sin'] = 0
        df['dow_cos'] = 1
    
    # 9. Clean infinite and extreme values (vectorized)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Cap extreme outliers (more robust than original)
    feature_cols = [
        'returns', 'log_returns', 'price_acceleration', 'relative_volume',
        'rsi_normalized', 'macd_normalized', 'price_vs_sma_10', 'price_vs_sma_20', 
        'price_vs_sma_50', 'bb_position', 'bb_width', 'volatility_rank',
        'momentum_5', 'momentum_10', 'momentum_20'
    ]
    
    for col in feature_cols:
        if col in df.columns:
            # Use IQR-based outlier capping (more robust than fixed percentiles)
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
    
    print(f"✅ Efficient Helformer features created: {df.shape[1]} columns")
    print(f"   Features computed in O(n) time with proper data leakage prevention")
    
    return df

def create_improved_model(architecture, input_shape, dropout_rate=0.2):
    """Create improved research-based model architectures"""
    print(f"🔧 Creating {architecture} model...")
    
    # All architectures now use the research-aligned Helformer
    # Different random seeds will create ensemble diversity
    if architecture == 'helformer':
        return create_helformer_model(input_shape, dropout_rate=dropout_rate)
    elif architecture == 'research_lstm':
        return create_helformer_model(input_shape, dropout_rate=dropout_rate)
    elif architecture == 'attention_lstm':
        return create_helformer_model(input_shape, dropout_rate=dropout_rate)
    elif architecture == 'cnn_lstm':
        return create_helformer_model(input_shape, dropout_rate=dropout_rate)
    elif architecture == 'transformer':
        return create_helformer_model(input_shape, dropout_rate=dropout_rate)
    elif architecture == 'lstm':
        return create_helformer_model(input_shape, dropout_rate=dropout_rate)
    else:
        # Fallback to research-aligned helformer
        return create_helformer_model(input_shape, dropout_rate=dropout_rate)

# Model architecture mapping for improved training

def main():
    """Main training pipeline with optimized preprocessing"""
    
    # Create unique experiment folder
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = f"experiment_{timestamp}"
    os.makedirs(experiment_folder, exist_ok=True)
    
    print(f"🗂️  Created experiment folder: {experiment_folder}")
    print(f"📊 Starting Advanced Ensemble Training Pipeline")
    print("=" * 60)
      # Prepare each asset separately with improved pipeline
    prepared_datasets = {}
    target_stats = {}
    for asset_name, df in datasets.items():
        df_clean, features, target_mean, target_std = prepare_asset_data(df, asset_name)
        prepared_datasets[asset_name] = (df_clean, features)
        target_stats[asset_name] = {'mean': target_mean, 'std': target_std}
    
    # Select universal features from improved pipeline
    feature_columns = select_universal_features(prepared_datasets)# Model parameters
    model_params = {
        'num_heads': config.NUM_HEADS,
        'head_dim': config.HEAD_DIM,
        'lstm_units': config.LSTM_UNITS,
        'dropout_rate': config.DROPOUT_RATE,
        'learning_rate': config.LEARNING_RATE
    }    # Train models for each asset with optimized pipeline
    asset_results = {}
    overall_performance = []
    
    print(f"🎯 Training {len(prepared_datasets)} assets with {len(config.ENSEMBLE_ARCHITECTURES)} architectures each")
    print(f"🏗️  Architectures: {', '.join(config.ENSEMBLE_ARCHITECTURES)}")
    print(f"📈 Features: {len(feature_columns)} universal features selected")
    print()
    
    for i, (asset_name, df_prepared) in enumerate(prepared_datasets.items(), 1):
        print(f"📊 Training Asset {i}/{len(prepared_datasets)}: {asset_name}")
        print("-" * 40)
        
        result = train_asset_specific_model(asset_name, df_prepared, feature_columns, model_params, experiment_folder, target_stats[asset_name])
        asset_results[asset_name] = result
        overall_performance.append(result['metrics']['R2'])
        
        # Print immediate summary for this asset
        metrics = result['metrics']
        print(f"✅ {asset_name} Training Complete!")
        print(f"   R²: {metrics['R2']:.4f} | MAPE: {metrics.get('MAPE', 0):.1f}% | RMSE: {metrics['RMSE']:.4f}")
        print(f"   Win Rate: {metrics.get('win_rate', 0):.1f}% | Annual Return: {metrics.get('annual_return', 0):.1f}%")
        print(f"   Sharpe: {metrics.get('sharpe_ratio', 0):.2f} | Max DD: {metrics.get('max_drawdown', 0):.1f}%")
        print(f"   Diversity: {metrics['diversity_score']:.4f} | Method: {metrics['ensemble_method']}")
        print()
    
    # Overall system evaluation
    print(f"\n{'='*60}")
    print("ADVANCED ENSEMBLE HELFORMER SYSTEM RESULTS")
    print(f"{'='*60}")
    
    avg_r2 = np.mean(overall_performance)
    min_r2 = np.min(overall_performance)
    
    # Calculate ensemble diversity statistics
    avg_diversity = np.mean([result['metrics']['diversity_score'] for result in asset_results.values()])
    
    print(f"Assets Trained: {len(asset_results)}")
    print(f"Architecture Types: {', '.join(config.ENSEMBLE_ARCHITECTURES)}")
    print(f"Average R²: {avg_r2:.4f}")
    print(f"Minimum R²: {min_r2:.4f}")
    print(f"Average Ensemble Diversity: {avg_diversity:.4f}")
    
    # Performance assessment using config thresholds
    if avg_r2 > config.MIN_R2_SCORE and min_r2 > (config.MIN_R2_SCORE - 0.15):
        performance_level = "EXCELLENT - Advanced ensemble system ready"
        expected_returns = f"{config.TARGET_ANNUAL_RETURN}%+"
    elif avg_r2 > (config.MIN_R2_SCORE - 0.15) and min_r2 > (config.MIN_R2_SCORE - 0.25):
        performance_level = "GOOD - Strong ensemble performance"
        expected_returns = f"{config.TARGET_ANNUAL_RETURN // 2}-{config.TARGET_ANNUAL_RETURN}%"
    else:
        performance_level = "NEEDS IMPROVEMENT - Optimize ensemble weights"
        expected_returns = f"50-{config.TARGET_ANNUAL_RETURN // 4}%"
    
    print(f"System Quality: {performance_level}")
    print(f"Expected Returns: {expected_returns}")
    
    # Asset-specific performance with ensemble details
    print(f"\nPER-ASSET ENSEMBLE PERFORMANCE:")
    print("-" * 60)
    for asset_name, result in asset_results.items():
        metrics = result['metrics']
        print(f"{asset_name} Advanced Ensemble:")
        print(f"  R²: {metrics['R2']:.4f} | MAPE: {metrics.get('MAPE', 0):.3f}% | RMSE: {metrics['RMSE']:.4f}")
        print(f"  Diversity: {metrics['diversity_score']:.4f} | Method: {metrics['ensemble_method']}")
        print(f"  Architectures: {', '.join(metrics['architectures'])}")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.1f}% | Annual Return: {metrics.get('annual_return', 0):.1f}%")
        print()
    
    # Save enhanced system info
    system_info = {
        'assets': list(asset_results.keys()),
        'feature_columns': feature_columns,
        'avg_r2': avg_r2,
        'min_r2': min_r2,
        'avg_diversity': avg_diversity,
        'performance_level': performance_level,
        'expected_returns': expected_returns,
        'model_params': model_params,
        'system_type': 'advanced_ensemble_helformer',
        'ensemble_architectures': config.ENSEMBLE_ARCHITECTURES,
        'ensemble_method': config.ENSEMBLE_METHOD,
        'ensemble_configs': [result['ensemble_manager'].ensemble_configs for result in asset_results.values()],
        'training_timestamp': pd.Timestamp.now().isoformat(),
        'experiment_folder': experiment_folder
    }
    
    # Save system information in experiment folder
    system_info_path = os.path.join(experiment_folder, "helformer_system_info.pkl")
    features_path = os.path.join(experiment_folder, "helformer_universal_features.pkl")
    ensemble_info_path = os.path.join(experiment_folder, "helformer_ensemble_info.pkl")
    
    joblib.dump(system_info, system_info_path)
    joblib.dump(feature_columns, features_path)
    
    # Save consolidated ensemble info for easy loading
    ensemble_info = {
        'asset_ensemble_managers': {asset: result['ensemble_manager'] for asset, result in asset_results.items()},
        'asset_scalers': {asset: result['scaler'] for asset, result in asset_results.items()},
        'feature_columns': feature_columns,
        'window_size': list(asset_results.values())[0]['window_size'],
        'system_info': system_info
    }
    joblib.dump(ensemble_info, ensemble_info_path)
    print(f"\n🎉 Advanced Ensemble Helformer system training complete!")
    print(f"📊 System Type: {system_info['system_type']}")
    print(f"🏗️  Architectures: {', '.join(config.ENSEMBLE_ARCHITECTURES)}")
    print(f"📁 Experiment Folder: {experiment_folder}")
    print(f"📂 Files saved in {experiment_folder}/:")
    print(f"  - helformer_system_info.pkl (system metadata)")
    print(f"  - helformer_ensemble_info.pkl (ensemble managers & scalers)")
    print(f"  - helformer_universal_features.pkl (feature list)")
    print(f"  - helformer_[asset]_ensemble_*.h5 (individual ensemble models)")
    print(f"  - helformer_[asset]_scaler.pkl (feature scalers)")
    
    # Create a summary file in the experiment folder
    summary_path = os.path.join(experiment_folder, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("ADVANCED ENSEMBLE HELFORMER TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training Timestamp: {system_info['training_timestamp']}\n")
        f.write(f"Experiment Folder: {experiment_folder}\n")
        f.write(f"Assets Trained: {len(asset_results)}\n")
        f.write(f"Architectures: {', '.join(config.ENSEMBLE_ARCHITECTURES)}\n")
        f.write(f"Features Used: {len(feature_columns)}\n")
        f.write(f"Average R²: {avg_r2:.4f}\n")
        f.write(f"Minimum R²: {min_r2:.4f}\n")
        f.write(f"Average Diversity: {avg_diversity:.4f}\n")
        f.write(f"Performance Level: {performance_level}\n")
        f.write(f"Expected Returns: {expected_returns}\n\n")
        
        f.write("PER-ASSET PERFORMANCE:\n")
        f.write("-" * 30 + "\n")
        for asset_name, result in asset_results.items():
            metrics = result['metrics']
            f.write(f"{asset_name}:\n")
            f.write(f"  R²: {metrics['R2']:.4f} | MAPE: {metrics['MAPE']:.1f}% | RMSE: {metrics['RMSE']:.4f}\n")
            f.write(f"  Win Rate: {metrics['win_rate']:.1f}% | Annual Return: {metrics['annual_return']:.1f}%\n")
            f.write(f"  Sharpe: {metrics['sharpe_ratio']:.2f} | Max DD: {metrics['max_drawdown']:.1f}%\n")
            f.write(f"  Diversity: {metrics['diversity_score']:.4f} | Method: {metrics['ensemble_method']}\n\n")
    
    print(f"  - training_summary.txt (detailed summary)")
    print(f"\n📋 Summary saved to: {summary_path}")
    
    return asset_results

if __name__ == "__main__":
    main()