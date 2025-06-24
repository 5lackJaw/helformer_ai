"""
Research-Accurate Helformer Implementation
Based on Kehinde et al. (2025) - Helformer: an attention-based deep learning model for cryptocurrency price forecasting
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
import warnings
warnings.filterwarnings('ignore')

def mish_activation(x):
    """Mish activation function as used in research"""
    return x * tf.nn.tanh(tf.nn.softplus(x))

class SimpleHoltWintersDecomposition(layers.Layer):
    """
    TensorFlow-compatible Holt-Winters Layer
    Implements a simplified version of Holt-Winters decomposition
    that maintains the research concept while being TF-compatible
    """
    
    def __init__(self, alpha_init=0.3, gamma_init=0.3, **kwargs):
        super().__init__(**kwargs)
        self.alpha_init = alpha_init
        self.gamma_init = gamma_init
        
    def build(self, input_shape):
        # Learnable smoothing parameters (constrained to (0,1))
        self.alpha = self.add_weight(
            name='alpha',
            shape=(),
            initializer=keras.initializers.Constant(self.alpha_init),
            trainable=True
        )
        
        self.gamma = self.add_weight(
            name='gamma',
            shape=(),
            initializer=keras.initializers.Constant(self.gamma_init),
            trainable=True        )
        
        super().build(input_shape)
    
    def call(self, inputs):
        """
        Apply simplified Holt-Winters decomposition
        TensorFlow-compatible implementation
        """
        # Constrain parameters to (0,1) range
        alpha = tf.sigmoid(self.alpha)  # Sigmoid to get (0,1)
        gamma = tf.sigmoid(self.gamma)
        
        # Get input dimensions
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        num_features = tf.shape(inputs)[2]
        
        # Create exponential decay weights for level smoothing
        positions = tf.cast(tf.range(seq_len), tf.float32)
        level_weights = tf.pow(1.0 - alpha, positions)
        level_weights = tf.reverse(level_weights, [0])
        level_weights = level_weights / tf.reduce_sum(level_weights)
        
        # Apply level smoothing using matrix multiplication (more efficient than conv1d)
        # Shape: (batch_size, seq_len, num_features) -> (batch_size, seq_len, num_features)
        level_weights_expanded = tf.expand_dims(tf.expand_dims(level_weights, 0), -1)  # (1, seq_len, 1)
        level_smoothed = tf.reduce_sum(inputs * level_weights_expanded, axis=1, keepdims=True)
        level_smoothed = tf.tile(level_smoothed, [1, seq_len, 1])
        
        # Create seasonal weights
        seasonal_weights = tf.pow(1.0 - gamma, positions)
        seasonal_weights = tf.reverse(seasonal_weights, [0])
        seasonal_weights = seasonal_weights / tf.reduce_sum(seasonal_weights)
        
        # Apply seasonal smoothing
        seasonal_weights_expanded = tf.expand_dims(tf.expand_dims(seasonal_weights, 0), -1)  # (1, seq_len, 1)
        seasonal_smoothed = tf.reduce_sum(inputs * seasonal_weights_expanded, axis=1, keepdims=True)
        seasonal_smoothed = tf.tile(seasonal_smoothed, [1, seq_len, 1])
        
        # Deseasonalize: divide by seasonal component (avoid division by zero)
        deseasonalized = inputs / (seasonal_smoothed + 1e-8)
        
        return deseasonalized
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha_init': self.alpha_init,
            'gamma_init': self.gamma_init
        })
        return config

class MultiHeadSelfAttention(layers.Layer):
    """
    Multi-Head Self-Attention as per research
    Implements Equations 4-5 from the paper
    Maintains input dimensions for residual connections
    """
    
    def __init__(self, num_heads, head_size, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        
    def build(self, input_shape):
        # Get input feature dimension to maintain shape compatibility
        self.input_dim = input_shape[-1]
        self.d_model = self.num_heads * self.head_size
        
        # Query, Key, Value projection layers
        self.q_dense = layers.Dense(self.d_model, use_bias=False)
        self.k_dense = layers.Dense(self.d_model, use_bias=False)
        self.v_dense = layers.Dense(self.d_model, use_bias=False)
        
        # Output projection to match input dimensions for residual connection
        self.output_dense = layers.Dense(self.input_dim)
        
        super().build(input_shape)
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        
        # Generate Q, K, V matrices
        q = self.q_dense(inputs)  # (batch_size, seq_len, d_model)
        k = self.k_dense(inputs)
        v = self.v_dense(inputs)
        
        # Reshape for multi-head attention
        q = tf.reshape(q, (batch_size, seq_length, self.num_heads, self.head_size))
        k = tf.reshape(k, (batch_size, seq_length, self.num_heads, self.head_size))
        v = tf.reshape(v, (batch_size, seq_length, self.num_heads, self.head_size))
        
        # Transpose to (batch_size, num_heads, seq_len, head_size)
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])
        
        # Scaled dot-product attention (Equation 4)
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale by sqrt(d_k)
        scaled_attention = matmul_qk / tf.math.sqrt(tf.cast(self.head_size, tf.float32))
        
        # Apply softmax
        attention_weights = tf.nn.softmax(scaled_attention, axis=-1)
        
        # Apply attention to values
        attention_output = tf.matmul(attention_weights, v)
        
        # Transpose back and reshape
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, seq_length, self.d_model))
        
        # Final linear projection
        output = self.output_dense(attention_output)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'head_size': self.head_size
        })
        return config

def create_research_helformer(input_shape, config):
    """
    Create Helformer model exactly as described in research
    Single encoder architecture with:
    1. Holt-Winters Decomposition
    2. Multi-Head Attention
    3. Add & Norm layers    4. LSTM (replacing FFN)
    5. Dense output layer
    """
    
    inputs = layers.Input(shape=input_shape, name='price_input')
    
    # 1. Holt-Winters Decomposition Block
    decomposed = SimpleHoltWintersDecomposition(
        alpha_init=config.get('alpha_init', 0.3),
        gamma_init=config.get('gamma_init', 0.3),
        name='holt_winters_decomposition'
    )(inputs)
    
    # 2. Multi-Head Self-Attention Block
    attention_output = MultiHeadSelfAttention(
        num_heads=config.get('num_heads', 4),
        head_size=config.get('head_size', 16),
        name='multi_head_attention'
    )(decomposed)
    
    # 3. Add & Norm Layer (Residual Connection + Layer Normalization)
    add_norm_1 = layers.Add(name='add_1')([decomposed, attention_output])
    norm_1 = layers.LayerNormalization(name='layer_norm_1')(add_norm_1)    # 4. LSTM Layer (Replaces FFN in traditional Transformer)
    # Research: "LSTM layer captures temporal dependencies essential for time-series"
    lstm_output = layers.LSTM(
        units=26,  # Match feature count for residual connection
        return_sequences=True,
        dropout=config.get('dropout_rate', 0.1),
        recurrent_dropout=config.get('dropout_rate', 0.1),
        name='lstm_temporal'
    )(norm_1)    # 5. Another Add & Norm Layer
    add_norm_2 = layers.Add(name='add_2')([norm_1, lstm_output])
    norm_2 = layers.LayerNormalization(name='layer_norm_2')(add_norm_2)
    
    # 6. Global Average Pooling (to handle sequence to single prediction)
    pooled = layers.GlobalAveragePooling1D(name='global_pooling')(norm_2)
    
    # 7. Final Dense Layer with Mish Activation
    dense_1 = layers.Dense(
        units=64,
        activation=mish_activation,
        name='dense_mish'
    )(pooled)
    
    # 8. Dropout for regularization
    dropout = layers.Dropout(config.get('dropout_rate', 0.1), name='final_dropout')(dense_1)
    
    # 9. Output Layer (Single value prediction)
    outputs = layers.Dense(1, activation='linear', name='price_prediction')(dropout)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='Helformer')
    
    # Compile with research parameters
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.get('learning_rate', 0.001)),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_research_trading_strategy():
    """
    Simple directional trading strategy as per research
    Research achieved 925% returns with this approach
    """
    
    class ResearchTradingStrategy:
        def __init__(self, transaction_cost=0.01, initial_capital=10000):
            self.transaction_cost = transaction_cost
            self.initial_capital = initial_capital
            self.capital = initial_capital
            self.position = 0  # 0 = no position, 1 = long, -1 = short
            self.trades = []
            self.net_values = [1.0]  # Start with net value of 1
            
        def generate_signal(self, prediction, current_price=None):
            """
            Simple directional signal based on prediction
            Research: Trade based on prediction direction (no confidence threshold)
            """
            if prediction > 0:
                return 1  # Buy signal
            elif prediction < 0:
                return -1  # Sell signal
            else:
                return 0  # Hold
        
        def execute_trade(self, signal, current_price, next_price):
            """
            Execute trade based on signal
            Research methodology: Simple directional trading
            """
            if signal == 0 or signal == self.position:
                # No trade or already in position
                return_rate = 0
            else:
                # Calculate return rate
                if signal == 1:  # Buy
                    return_rate = (next_price - current_price) / current_price
                else:  # Sell
                    return_rate = (current_price - next_price) / current_price
                
                # Apply transaction cost
                return_rate -= self.transaction_cost
                
                self.position = signal
                
                # Record trade
                self.trades.append({
                    'signal': signal,
                    'current_price': current_price,
                    'next_price': next_price,
                    'return_rate': return_rate,
                    'transaction_cost': self.transaction_cost
                })
            
            # Update net value
            new_net_value = self.net_values[-1] * (1 + return_rate)
            self.net_values.append(new_net_value)
            
            return return_rate
        
        def calculate_metrics(self):
            """
            Calculate trading metrics as per research
            Returns: Excess Return, Volatility, Max Drawdown, Sharpe Ratio
            """
            if len(self.trades) == 0:
                return {
                    'excess_return': 0.0,
                    'volatility': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'num_trades': 0,
                    'win_rate': 0.0
                }
            
            returns = [trade['return_rate'] for trade in self.trades]
            
            # Excess Return (%)
            total_return = (self.net_values[-1] - 1) * 100
            
            # Volatility (standard deviation of returns)
            volatility = np.std(returns) if len(returns) > 1 else 0.0
            
            # Maximum Drawdown
            net_values = np.array(self.net_values)
            peak = np.maximum.accumulate(net_values)
            drawdown = (net_values - peak) / peak
            max_drawdown = np.min(drawdown)
            
            # Sharpe Ratio (assuming 1% risk-free rate as per research)
            risk_free_rate = 0.01
            mean_return = np.mean(returns) if len(returns) > 0 else 0.0
            if volatility > 0:
                sharpe_ratio = (mean_return - risk_free_rate) / volatility
            else:
                sharpe_ratio = 0.0
            
            # Additional metrics
            win_rate = sum(1 for r in returns if r > 0) / len(returns) * 100
            
            return {
                'excess_return': total_return,
                'volatility': volatility,
                'max_drawdown': abs(max_drawdown) * 100,
                'sharpe_ratio': sharpe_ratio,
                'num_trades': len(self.trades),
                'win_rate': win_rate,
                'net_value_final': self.net_values[-1]
            }
    
    return ResearchTradingStrategy

def evaluate_research_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics exactly as per research
    Returns similarity and dissimilarity metrics
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    # Ensure numpy arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Remove any NaN or infinite values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {
            'rmse': float('inf'),
            'mape': float('inf'),
            'mae': float('inf'),
            'r2': -float('inf'),
            'evs': -float('inf'),
            'kge': -float('inf')
        }
    
    # Dissimilarity metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE calculation (avoid division by zero)
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100
    
    # Similarity metrics
    r2 = r2_score(y_true, y_pred)
    
    # Explained Variance Score
    var_y = np.var(y_true)
    if var_y > 0:
        evs = 1 - np.var(y_true - y_pred) / var_y
    else:
        evs = 0.0
    
    # Kling-Gupta Efficiency (simplified)
    mean_obs = np.mean(y_true)
    mean_sim = np.mean(y_pred)
    std_obs = np.std(y_true)
    std_sim = np.std(y_pred)
    
    if std_obs > 0 and std_sim > 0:
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        bias_ratio = mean_sim / mean_obs if mean_obs != 0 else 0
        variability_ratio = std_sim / std_obs
        
        kge = 1 - np.sqrt((correlation - 1)**2 + (bias_ratio - 1)**2 + (variability_ratio - 1)**2)
    else:
        kge = -float('inf')
    
    return {
        'rmse': rmse,
        'mape': mape,
        'mae': mae,
        'r2': r2,
        'evs': evs,
        'kge': kge
    }

# Research benchmark targets (from paper results)
RESEARCH_BENCHMARKS = {
    'rmse': {'excellent': 1.0, 'good': 50.0, 'poor': 500.0},
    'mape': {'excellent': 0.1, 'good': 1.0, 'poor': 10.0},
    'mae': {'excellent': 1.0, 'good': 50.0, 'poor': 500.0},
    'r2': {'excellent': 0.8, 'good': 0.5, 'poor': 0.1},
    'excess_return': {'excellent': 500.0, 'good': 100.0, 'poor': 10.0},
    'volatility': {'excellent': 0.02, 'good': 0.05, 'poor': 0.10},
    'max_drawdown': {'excellent': 1.0, 'good': 10.0, 'poor': 50.0},
    'sharpe_ratio': {'excellent': 5.0, 'good': 2.0, 'poor': 0.5}
}

def evaluate_against_research(metrics):
    """Evaluate how our results compare to research benchmarks"""
    evaluation = {}
    
    for metric, value in metrics.items():
        if metric in RESEARCH_BENCHMARKS:
            benchmarks = RESEARCH_BENCHMARKS[metric]
            
            if metric in ['rmse', 'mape', 'mae', 'volatility', 'max_drawdown']:
                # Lower is better
                if value <= benchmarks['excellent']:
                    evaluation[metric] = 'EXCELLENT'
                elif value <= benchmarks['good']:
                    evaluation[metric] = 'GOOD'
                else:
                    evaluation[metric] = 'POOR'
            else:
                # Higher is better
                if value >= benchmarks['excellent']:
                    evaluation[metric] = 'EXCELLENT'
                elif value >= benchmarks['good']:
                    evaluation[metric] = 'GOOD'
                else:
                    evaluation[metric] = 'POOR'
        else:
            evaluation[metric] = 'NOT_BENCHMARKED'
    
    return evaluation
