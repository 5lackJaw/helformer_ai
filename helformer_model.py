"""
Enhanced Model Architectures and Ensemble Methods
Provides multiple model variants and ensemble techniques
"""

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Dense, LSTM, GRU, Conv1D, MultiHeadAttention, LayerNormalization, Dropout, Add, GlobalAveragePooling1D, TimeDistributed, Multiply, Concatenate
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional
import logging

# Import improved architectures
from improved_model_architectures import (
    create_research_lstm_model, create_attention_lstm_model, 
    create_cnn_lstm_model, create_transformer_model, 
    get_model_by_name, create_realistic_callbacks
)

# Import research-aligned components
from research_helformer import create_research_helformer, create_research_trading_strategy

# Import centralized configuration
from config_helformer import config

logger = logging.getLogger(__name__)

def mish_activation(x):
    """Mish activation: x * tanh(softplus(x))"""
    return x * keras.activations.tanh(keras.activations.softplus(x))

# Register the custom activation globally
keras.utils.get_custom_objects()['mish_activation'] = mish_activation

def create_helformer_model(input_shape, **params):
    """
    Create the main Helformer model (research-aligned version)
    This is the primary model creation function used throughout the system.
    
    Now uses the research-aligned architecture from Kehinde et al. (2025)
    """
    logger.info("Creating research-aligned Helformer model")
      # Use the research-aligned architecture
    return EnhancedHelformerArchitectures.create_research_aligned_helformer(input_shape, **params)

class EnhancedHelformerArchitectures:
    """
    Enhanced Helformer model architectures with multiple variants.
    """
    
    @staticmethod
    def create_transformer_variant(input_shape: Tuple, 
                                 d_model: int = None,
                                 num_heads: int = None,
                                 num_transformer_blocks: int = None,
                                 ff_dim: int = None,
                                 dropout_rate: float = None) -> Model:
        """
        Pure transformer variant of Helformer.
        
        Args:
            input_shape: Shape of input data
            d_model: Model dimension (uses config default if None)
            num_heads: Number of attention heads (uses config default if None)
            num_transformer_blocks: Number of transformer blocks (uses config default if None)
            ff_dim: Feed-forward network dimension (uses config default if None)
            dropout_rate: Dropout rate (uses config default if None)
            
        Returns:
            Compiled Keras model
        """
        # Use config defaults if parameters not provided
        defaults = config.MODEL_DEFAULTS['TRANSFORMER']
        d_model = d_model or defaults['d_model']
        num_heads = num_heads or defaults['num_heads']
        num_transformer_blocks = num_transformer_blocks or defaults['num_transformer_blocks']
        ff_dim = ff_dim or defaults['ff_dim']
        dropout_rate = dropout_rate or defaults['dropout_rate']
        inputs = Input(shape=input_shape)
        
        # Input projection
        x = Dense(d_model)(inputs)
        
        # Transformer blocks
        for i in range(num_transformer_blocks):
            # Multi-head self-attention
            attention_output = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model // num_heads,
                dropout=dropout_rate,
                name=f'mha_{i}'
            )(x, x)
            
            attention_output = Dropout(dropout_rate)(attention_output)
            x = Add()([x, attention_output])
            x = LayerNormalization()(x)
            
            # Feed-forward network
            ff_output = Dense(ff_dim, activation='relu')(x)
            ff_output = Dropout(dropout_rate)(ff_output)
            ff_output = Dense(d_model)(ff_output)
            ff_output = Dropout(dropout_rate)(ff_output)
            
            x = Add()([x, ff_output])
            x = LayerNormalization()(x)        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        
        # Output layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    @staticmethod
    def create_cnn_lstm_variant(input_shape: Tuple,
                              num_conv_layers: int = None,
                              conv_filters: List[int] = None,
                              kernel_sizes: List[int] = None,
                              lstm_units: List[int] = None,
                              dropout_rate: float = None) -> Model:
        """
        CNN-LSTM hybrid variant.
        
        Args:
            input_shape: Shape of input data
            num_conv_layers: Number of convolutional layers (uses config default if None)
            conv_filters: Number of filters per conv layer (uses config default if None)
            kernel_sizes: Kernel sizes for conv layers (uses config default if None)
            lstm_units: LSTM units for each layer (uses config default if None)
            dropout_rate: Dropout rate (uses config default if None)
            
        Returns:
            Compiled Keras model
        """        # Use config defaults if parameters not provided
        defaults = config.MODEL_DEFAULTS['CNN_LSTM']
        num_conv_layers = num_conv_layers or defaults['num_conv_layers']
        conv_filters = conv_filters or defaults['conv_filters']
        kernel_sizes = kernel_sizes or defaults['kernel_sizes']
        lstm_units = lstm_units or defaults['lstm_units']
        dropout_rate = dropout_rate or defaults['dropout_rate']
        
        inputs = Input(shape=input_shape)
        
        x = inputs
        
        # Convolutional layers
        for i in range(num_conv_layers):
            filters = conv_filters[i] if i < len(conv_filters) else conv_filters[-1]
            kernel_size = kernel_sizes[i] if i < len(kernel_sizes) else kernel_sizes[-1]
            
            x = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
            x = Dropout(dropout_rate)(x)
        
        # LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            x = LSTM(units, return_sequences=return_sequences, dropout=dropout_rate)(x)
        
        # Output layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    @staticmethod
    def create_gru_attention_variant(input_shape: Tuple,
                                   gru_units: List[int] = None,
                                   attention_units: int = None,
                                   dropout_rate: float = None) -> Model:
        """
        GRU with attention mechanism variant.
        
        Args:
            input_shape: Shape of input data
            gru_units: GRU units for each layer (uses config default if None)
            attention_units: Number of attention units (uses config default if None)
            dropout_rate: Dropout rate (uses config default if None)
            
        Returns:
            Compiled Keras model
        """
        # Use config defaults if parameters not provided
        defaults = config.MODEL_DEFAULTS['GRU_ATTENTION']
        gru_units = gru_units or defaults['gru_units']
        attention_units = attention_units or defaults['attention_units']
        dropout_rate = dropout_rate or defaults['dropout_rate']
        
        inputs = Input(shape=input_shape)
        
        x = inputs
        
        # GRU layers
        gru_outputs = []
        for i, units in enumerate(gru_units):
            x = GRU(units, return_sequences=True, dropout=dropout_rate)(x)
            gru_outputs.append(x)        # Attention mechanism
        attention_weights = TimeDistributed(Dense(attention_units, activation='tanh'))(x)
        attention_weights = TimeDistributed(Dense(1, activation='softmax'))(attention_weights)
        
        # Apply attention
        attention_applied = Multiply()([x, attention_weights])
        context_vector = GlobalAveragePooling1D()(attention_applied)
        
        # Output layers
        x = Dense(64, activation='relu')(context_vector)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    @staticmethod
    def create_multi_scale_variant(input_shape: Tuple,
                                 scales: List[int] = [1, 3, 5, 7],
                                 filters_per_scale: int = 32,
                                 lstm_units: int = 128,
                                 dropout_rate: float = 0.1) -> Model:
        """
        Multi-scale CNN variant for different temporal patterns.
        
        Args:
            input_shape: Shape of input data
            scales: Different kernel sizes for multi-scale analysis
            filters_per_scale: Number of filters per scale
            lstm_units: LSTM units
            dropout_rate: Dropout rate
            
        Returns:
            Compiled Keras model
        """
        inputs = Input(shape=input_shape)
        
        # Multi-scale convolutions
        scale_outputs = []
        for scale in scales:
            x = Conv1D(filters=filters_per_scale, kernel_size=scale, activation='relu', padding='same')(inputs)
            x = Dropout(dropout_rate)(x)
            scale_outputs.append(x)
          # Concatenate multi-scale features
        if len(scale_outputs) > 1:
            x = Concatenate(axis=-1)(scale_outputs)
        else:
            x = scale_outputs[0]
        
        # LSTM processing
        x = LSTM(lstm_units, return_sequences=False, dropout=dropout_rate)(x)
        
        # Output layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    @staticmethod
    def create_research_aligned_helformer(input_shape: Tuple, **params) -> Model:
        """
        Create research-aligned Helformer model based on Kehinde et al. (2025)
        
        This implements the exact architecture from the research paper:
        - Holt-Winters decomposition layer
        - Multi-head self-attention
        - LSTM (replaces FFN in traditional Transformer)
        - Add & Norm layers
        - Mish activation in output layer
        
        Args:
            input_shape: Shape of input data
            **params: Model parameters from config
            
        Returns:
            Compiled research-aligned Helformer model
        """
        # Create model configuration for research alignment
        research_config = {
            'num_heads': params.get('num_heads', 4),
            'head_size': params.get('head_size', 16),
            'lstm_units': params.get('lstm_units', 26),  # Match feature count
            'dropout_rate': params.get('dropout_rate', 0.1),
            'learning_rate': params.get('learning_rate', 0.001),
            'alpha_init': params.get('alpha_init', 0.3),
            'gamma_init': params.get('gamma_init', 0.3)
        }
        
        logger.info("Creating research-aligned Helformer model")
        logger.info(f"Model config: {research_config}")
        
        # Use the research implementation
        model = create_research_helformer(input_shape, research_config)
        
        return model

class EnsembleManager:
    """
    Manages ensemble of different model architectures.
    """
    
    def __init__(self, ensemble_configs: List[Dict]):
        """
        Initialize ensemble manager.
        
        Args:
            ensemble_configs: List of model configuration dictionaries
        """
        self.ensemble_configs = ensemble_configs
        self.models: List[Model] = []
        self.model_weights: List[float] = []
        self.training_histories: List[Dict] = []
        
        logger.info(f"Ensemble manager initialized with {len(ensemble_configs)} models")
    
    def create_ensemble(self, input_shape: Tuple) -> List[Model]:
        """
        Create ensemble of models based on configurations.
        
        Args:
            input_shape: Shape of input data
            
        Returns:
            List of compiled models
        """
        self.models = []
        
        for i, config in enumerate(self.ensemble_configs):
            model_type = config.get('type', 'transformer')
            
            try:
                # Filter out keys that are not valid for architecture creation
                valid_keys = ['d_model', 'num_heads', 'num_transformer_blocks', 'ff_dim', 'dropout_rate',
                             'cnn_filters', 'lstm_units', 'attention_heads', 'gru_units', 'scale_factors']
                filtered_config = {k: v for k, v in config.items() 
                                 if k != 'type' and k in valid_keys}
                if model_type == 'transformer':
                    model = create_transformer_model(
                        input_shape=input_shape,
                        dropout_rate=filtered_config.get('dropout_rate', 0.2)
                    )
                elif model_type == 'cnn_lstm':
                    model = create_cnn_lstm_model(
                        input_shape=input_shape,
                        dropout_rate=filtered_config.get('dropout_rate', 0.2)
                    )
                elif model_type == 'gru_attention':
                    model = create_attention_lstm_model(
                        input_shape=input_shape,
                        dropout_rate=filtered_config.get('dropout_rate', 0.2)
                    )
                elif model_type == 'multi_scale':
                    model = create_cnn_lstm_model(  # Use CNN-LSTM as multi-scale variant
                        input_shape=input_shape,
                        dropout_rate=filtered_config.get('dropout_rate', 0.2)
                    )
                elif model_type == 'lstm':
                    model = create_research_lstm_model(
                        input_shape=input_shape,
                        dropout_rate=filtered_config.get('dropout_rate', 0.2)
                    )
                else:
                    logger.warning(f"Unknown model type: {model_type}, using research LSTM")
                    model = create_research_lstm_model(input_shape, dropout_rate=0.2)# Compile model with proper optimizer handling
                optimizer_name = config.get('optimizer', 'adam').lower()
                learning_rate = config.get('learning_rate', 0.001)
                
                # Map optimizer strings to actual optimizers with fallback
                try:
                    optimizer_map = {
                        'adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        'adamw': tf.keras.optimizers.AdamW(learning_rate=learning_rate) if hasattr(tf.keras.optimizers, 'AdamW') else tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                        'sgd': tf.keras.optimizers.SGD(learning_rate=learning_rate)
                    }
                    
                    optimizer = optimizer_map.get(optimizer_name, optimizer_map['adam'])
                    logger.info(f"Using optimizer: {optimizer_name} -> {type(optimizer).__name__}")
                
                except Exception as e:
                    logger.warning(f"Error creating optimizer {optimizer_name}: {e}, falling back to Adam")
                    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                
                model.compile(
                    optimizer=optimizer,
                    loss=config.get('loss', 'mse'),
                    metrics=config.get('metrics', ['mae'])
                )
                
                self.models.append(model)
                logger.info(f"Created model {i+1}/{len(self.ensemble_configs)}: {model_type}")
                
            except Exception as e:
                logger.error(f"Error creating model {i}: {str(e)}")
                continue
        
        # Initialize equal weights
        self.model_weights = [1.0 / len(self.models)] * len(self.models)
        
        return self.models
    
    def train_ensemble(self, X_train, y_train, X_val, y_val, **kwargs) -> List[Dict]:
        """
        Train all models in the ensemble.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features  
            y_val: Validation targets
            **kwargs: Training parameters
            
        Returns:
            List of training histories
        """
        self.training_histories = []
        
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}")
            
            try:
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    verbose=kwargs.get('verbose', 1),
                    **{k: v for k, v in kwargs.items() if k != 'verbose'}
                )
                
                self.training_histories.append(history.history)
                logger.info(f"Model {i+1} training completed")
                
            except Exception as e:
                logger.error(f"Error training model {i}: {str(e)}")
                self.training_histories.append({})
        
        # Update ensemble weights based on validation performance
        self._update_ensemble_weights(X_val, y_val)
        
        return self.training_histories
    
    def predict_ensemble(self, X, method: str = 'weighted_average') -> np.ndarray:
        """
        Make predictions using ensemble of models.
        
        Args:
            X: Input features
            method: Ensemble method ('weighted_average', 'simple_average', 'median')
            
        Returns:
            Ensemble predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            try:
                pred = model.predict(X, verbose=0)
                predictions.append(pred.flatten())
            except Exception as e:
                logger.error(f"Error getting prediction from model: {str(e)}")
                continue
        
        if not predictions:
            raise ValueError("No valid predictions from ensemble")
        
        predictions = np.array(predictions)
        
        # Combine predictions
        if method == 'weighted_average':
            weights = np.array(self.model_weights[:len(predictions)])
            weights = weights / weights.sum()  # Normalize weights
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            
        elif method == 'simple_average':
            ensemble_pred = np.mean(predictions, axis=0)
            
        elif method == 'median':
            ensemble_pred = np.median(predictions, axis=0)
            
        else:
            logger.warning(f"Unknown ensemble method: {method}, using simple average")
            ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def _update_ensemble_weights(self, X_val, y_val):
        """Update ensemble weights based on validation performance."""
        
        if not self.models or len(X_val) == 0:
            return
        
        model_scores = []
        
        for model in self.models:
            try:
                # Evaluate model on validation set
                val_pred = model.predict(X_val, verbose=0)
                mse = np.mean((y_val - val_pred.flatten()) ** 2)
                
                # Convert MSE to score (lower MSE = higher score)
                score = 1.0 / (1.0 + mse)
                model_scores.append(score)
                
            except Exception as e:
                logger.error(f"Error evaluating model: {str(e)}")
                model_scores.append(0.1)  # Low score for failed models
        
        # Normalize scores to get weights
        if sum(model_scores) > 0:
            self.model_weights = [score / sum(model_scores) for score in model_scores]
        else:
            self.model_weights = [1.0 / len(self.models)] * len(self.models)
        
        logger.info(f"Updated ensemble weights: {[f'{w:.3f}' for w in self.model_weights]}")
    
    def get_ensemble_diversity(self, X) -> float:
        """
        Calculate diversity score of ensemble predictions.
        
        Args:
            X: Input features
            
        Returns:
            Diversity score (higher = more diverse)
        """
        if len(self.models) < 2:
            return 0.0
        
        try:
            # Get predictions from all models
            predictions = []
            for model in self.models:
                pred = model.predict(X, verbose=0)
                predictions.append(pred.flatten())
            
            predictions = np.array(predictions)
            
            # Calculate pairwise correlations
            correlations = []
            for i in range(len(predictions)):
                for j in range(i + 1, len(predictions)):
                    corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            if correlations:
                # Diversity is inverse of average correlation
                avg_correlation = np.mean(correlations)
                diversity = 1.0 - avg_correlation
                return max(0.0, diversity)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating ensemble diversity: {str(e)}")
            return 0.0
    
    def save_ensemble(self, filepath_prefix: str):
        """Save all models in the ensemble."""
        for i, model in enumerate(self.models):
            model.save(f"{filepath_prefix}_model_{i}.h5")
          # Save weights and configurations
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        metadata = {
            'model_weights': convert_numpy_types(self.model_weights),
            'ensemble_configs': convert_numpy_types(self.ensemble_configs),
            'training_histories': convert_numpy_types(self.training_histories)
        }
        
        with open(f"{filepath_prefix}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Ensemble saved with prefix: {filepath_prefix}")
    
    def load_ensemble(self, filepath_prefix: str):
        """Load ensemble from saved files."""
        import json
        import os
        
        # Load metadata
        metadata_path = f"{filepath_prefix}_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.model_weights = metadata.get('model_weights', [])
            self.ensemble_configs = metadata.get('ensemble_configs', [])
            self.training_histories = metadata.get('training_histories', [])
        
        # Load models
        self.models = []
        i = 0
        while True:
            model_path = f"{filepath_prefix}_model_{i}.h5"
            if os.path.exists(model_path):
                model = keras.models.load_model(model_path)
                self.models.append(model)
                i += 1
            else:
                break
        
        logger.info(f"Loaded ensemble with {len(self.models)} models")

def create_default_ensemble_configs() -> List[Dict]:
    """Create default ensemble configurations."""
    
    configs = [
        {
            'type': 'transformer',
            'num_transformer_blocks': 2,
            'd_model': 128,
            'num_heads': 8,
            'dropout_rate': 0.1,
            'optimizer': 'adam',
            'loss': 'mse'
        },
        {
            'type': 'cnn_lstm',
            'conv_filters': [64, 128],
            'kernel_sizes': [3, 5],
            'lstm_units': [128, 64],
            'dropout_rate': 0.15,
            'optimizer': 'adam',
            'loss': 'mae'
        },
        {
            'type': 'gru_attention',
            'gru_units': [128, 64],
            'attention_units': 64,
            'dropout_rate': 0.1,
            'optimizer': 'rmsprop',
            'loss': 'huber'
        },
        {
            'type': 'multi_scale',
            'scales': [1, 3, 5],
            'filters_per_scale': 32,
            'lstm_units': 128,
            'dropout_rate': 0.1,
            'optimizer': 'adam',
            'loss': 'mse'
        }
    ]
    
    return configs