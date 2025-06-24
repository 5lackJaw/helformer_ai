#!/usr/bin/env python3
"""
Improved Model Architectures for Financial Time Series Prediction
Research-based architectures optimized for realistic financial forecasting
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
import numpy as np

def create_research_lstm_model(input_shape, dropout_rate=0.2):
    """
    Research-based LSTM architecture for financial prediction
    Based on: Kim & Won (2018), Sezer et al. (2020)
    """
    inputs = layers.Input(shape=input_shape)
    
    # First LSTM layer with batch normalization
    x = layers.LSTM(64, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate)(inputs)
    x = layers.BatchNormalization()(x)
    
    # Second LSTM layer
    x = layers.LSTM(32, dropout=dropout_rate, recurrent_dropout=dropout_rate)(x)
    x = layers.BatchNormalization()(x)
    
    # Dense layers with proper regularization
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer for regression
    outputs = layers.Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='research_lstm')
    
    # Use appropriate optimizer and loss for financial data
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='huber',  # More robust to outliers than MSE
        metrics=['mae', 'mse']
    )
    
    return model

def create_attention_lstm_model(input_shape, dropout_rate=0.2):
    """
    LSTM with attention mechanism for financial prediction
    Based on: Bahdanau et al. (2015), Qin et al. (2017)
    """
    inputs = layers.Input(shape=input_shape)
    
    # LSTM layers with attention
    lstm_out = layers.LSTM(64, return_sequences=True, dropout=dropout_rate)(inputs)
    lstm_out = layers.BatchNormalization()(lstm_out)
    
    # Attention mechanism
    attention = layers.Dense(64, activation='tanh')(lstm_out)
    attention = layers.Dense(1, activation='softmax')(attention)
    context_vector = layers.Dot(axes=1)([attention, lstm_out])
    context_vector = layers.Flatten()(context_vector)
    
    # Dense layers
    x = layers.Dense(32, activation='relu')(context_vector)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='attention_lstm')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae', 'mse']
    )
    
    return model

def create_cnn_lstm_model(input_shape, dropout_rate=0.2):
    """
    CNN-LSTM hybrid for feature extraction and temporal modeling
    Based on: Selvin et al. (2017), Livieris et al. (2020)
    """
    inputs = layers.Input(shape=input_shape)
    
    # CNN layers for feature extraction
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # LSTM layers for temporal modeling
    x = layers.LSTM(50, return_sequences=True, dropout=dropout_rate)(x)
    x = layers.LSTM(25, dropout=dropout_rate)(x)
    x = layers.BatchNormalization()(x)
    
    # Dense layers
    x = layers.Dense(25, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='cnn_lstm')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae', 'mse']
    )
    
    return model

def create_transformer_model(input_shape, num_heads=4, key_dim=32, dropout_rate=0.2):
    """
    Simplified transformer for financial time series
    Based on: Vaswani et al. (2017), optimized for financial data
    """
    inputs = layers.Input(shape=input_shape)
    
    # Positional encoding
    x = inputs
    
    # Multi-head attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=key_dim,
        dropout=dropout_rate
    )(x, x)
    
    # Add & Norm
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)
    
    # Feed forward
    ffn = layers.Dense(64, activation='relu')(x)
    ffn = layers.Dropout(dropout_rate)(ffn)
    ffn = layers.Dense(input_shape[-1])(ffn)
    
    # Add & Norm
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization()(x)
    
    # Global average pooling and output
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='transformer')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae', 'mse']
    )
    
    return model

def create_ensemble_model(input_shape, num_models=3, dropout_rate=0.2):
    """
    Create ensemble of different architectures
    Research shows ensemble diversity improves performance
    """
    inputs = layers.Input(shape=input_shape)
    
    # Create different model branches
    lstm_branch = create_lstm_branch(inputs, dropout_rate)
    cnn_branch = create_cnn_branch(inputs, dropout_rate)
    attention_branch = create_attention_branch(inputs, dropout_rate)
    
    # Combine predictions
    ensemble_output = layers.Average()([lstm_branch, cnn_branch, attention_branch])
    
    model = Model(inputs=inputs, outputs=ensemble_output, name='ensemble_model')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae', 'mse']
    )
    
    return model

def create_lstm_branch(inputs, dropout_rate):
    """LSTM branch for ensemble"""
    x = layers.LSTM(32, return_sequences=True, dropout=dropout_rate)(inputs)
    x = layers.LSTM(16, dropout=dropout_rate)(x)
    x = layers.Dense(8, activation='relu')(x)
    return layers.Dense(1, activation='linear', name='lstm_out')(x)

def create_cnn_branch(inputs, dropout_rate):
    """CNN branch for ensemble"""
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(inputs)
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(8, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    return layers.Dense(1, activation='linear', name='cnn_out')(x)

def create_attention_branch(inputs, dropout_rate):
    """Attention branch for ensemble"""
    x = layers.MultiHeadAttention(num_heads=2, key_dim=16)(inputs, inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(8, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    return layers.Dense(1, activation='linear', name='attention_out')(x)

def create_financial_loss_function():
    """
    Custom loss function for financial prediction
    Combines MSE with directional accuracy penalty
    """
    def financial_loss(y_true, y_pred):
        # MSE component
        mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        
        # Directional accuracy component
        direction_true = tf.sign(y_true)
        direction_pred = tf.sign(y_pred)
        direction_loss = tf.reduce_mean(tf.cast(tf.not_equal(direction_true, direction_pred), tf.float32))
        
        # Combine losses
        return mse_loss + 0.1 * direction_loss
    
    return financial_loss

def get_model_by_name(model_name, input_shape, **kwargs):
    """Factory function to create models by name"""
    model_registry = {
        'lstm': create_research_lstm_model,
        'attention_lstm': create_attention_lstm_model,
        'cnn_lstm': create_cnn_lstm_model,
        'transformer': create_transformer_model,
        'ensemble': create_ensemble_model
    }
    
    if model_name not in model_registry:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_registry.keys())}")
    
    return model_registry[model_name](input_shape, **kwargs)

def create_realistic_callbacks(patience=15, min_lr=1e-6, monitor='val_loss'):
    """
    Create callbacks optimized for financial time series training
    Research-based early stopping and learning rate scheduling
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=patience//2,
            min_lr=min_lr,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model_weights.h5',
            monitor=monitor,
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        )
    ]
    
    return callbacks
