"""
Adaptive Ensemble Optimizer for Helformer
Automatically selects optimal architectures and ensemble methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger(__name__)

class AdaptiveEnsembleOptimizer:
    """
    Intelligently optimizes ensemble configuration based on data characteristics
    and performance testing
    """
    
    def __init__(self):
        self.architecture_scores = {}
        self.ensemble_method_scores = {}
        self.data_characteristics = {}
        
    def analyze_data_characteristics(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Analyze data to determine optimal architectures
        
        Returns:
            Dictionary with data characteristics and recommendations
        """
        characteristics = {}
        
        # Sequence length analysis
        seq_len = X.shape[1] if len(X.shape) > 1 else 1
        characteristics['sequence_length'] = seq_len
        
        # Feature dimensionality
        n_features = X.shape[-1] if len(X.shape) > 1 else 1
        characteristics['n_features'] = n_features
        
        # Volatility analysis
        target_volatility = np.std(y)
        characteristics['volatility'] = target_volatility
        
        # Trend analysis
        if len(y) > 1:
            trend_strength = abs(np.corrcoef(np.arange(len(y)), y)[0, 1])
            characteristics['trend_strength'] = trend_strength
        else:
            characteristics['trend_strength'] = 0.0
        
        # Autocorrelation analysis
        if len(y) > 10:
            autocorr_1 = np.corrcoef(y[:-1], y[1:])[0, 1] if len(y) > 1 else 0
            autocorr_5 = np.corrcoef(y[:-5], y[5:])[0, 1] if len(y) > 5 else 0
            autocorr_10 = np.corrcoef(y[:-10], y[10:])[0, 1] if len(y) > 10 else 0
        else:
            autocorr_1 = autocorr_5 = autocorr_10 = 0
            
        characteristics['autocorr_short'] = autocorr_1
        characteristics['autocorr_medium'] = autocorr_5
        characteristics['autocorr_long'] = autocorr_10
        
        # Frequency domain analysis (simplified)
        if len(y) > 20:
            fft_y = np.fft.fft(y)
            power_spectrum = np.abs(fft_y) ** 2
            dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
            dominant_freq_strength = power_spectrum[dominant_freq_idx] / np.sum(power_spectrum)
            characteristics['dominant_freq_strength'] = dominant_freq_strength
        else:
            characteristics['dominant_freq_strength'] = 0.0
        
        self.data_characteristics = characteristics
        return characteristics
    
    def recommend_architectures(self, data_chars: Dict) -> List[Dict]:
        """
        Recommend optimal architectures based on data characteristics
        
        Args:
            data_chars: Output from analyze_data_characteristics
            
        Returns:
            List of recommended architecture configurations
        """
        recommendations = []
        
        # Base architectures with reasoning
        arch_scores = {
            'transformer': 0.0,
            'cnn_lstm': 0.0,
            'gru_attention': 0.0,
            'multi_scale': 0.0,
            'lstm_attention': 0.0,  # Additional architecture
            'residual_lstm': 0.0,   # Additional architecture
        }
        
        # Score based on sequence length
        seq_len = data_chars.get('sequence_length', 30)
        if seq_len > 50:
            arch_scores['transformer'] += 0.3  # Transformers excel with long sequences
            arch_scores['gru_attention'] += 0.2
        elif seq_len > 20:
            arch_scores['cnn_lstm'] += 0.3     # CNN-LSTM good for medium sequences
            arch_scores['lstm_attention'] += 0.2
        else:
            arch_scores['multi_scale'] += 0.3  # Multi-scale for short sequences
            arch_scores['residual_lstm'] += 0.2
        
        # Score based on volatility
        volatility = data_chars.get('volatility', 0.5)
        if volatility > 1.0:  # High volatility
            arch_scores['multi_scale'] += 0.3  # Multi-scale handles noise well
            arch_scores['gru_attention'] += 0.2  # Attention focuses on important parts
        elif volatility < 0.2:  # Low volatility
            arch_scores['transformer'] += 0.3   # Transformers for subtle patterns
            arch_scores['lstm_attention'] += 0.2
        
        # Score based on trend strength
        trend_strength = data_chars.get('trend_strength', 0.0)
        if trend_strength > 0.5:  # Strong trend
            arch_scores['cnn_lstm'] += 0.3      # CNN-LSTM captures trends well
            arch_scores['residual_lstm'] += 0.2
        else:  # Weak/no trend
            arch_scores['transformer'] += 0.2   # Transformers for complex patterns
            arch_scores['gru_attention'] += 0.3
        
        # Score based on autocorrelation
        autocorr_short = data_chars.get('autocorr_short', 0.0)
        autocorr_long = data_chars.get('autocorr_long', 0.0)
        
        if abs(autocorr_long) > abs(autocorr_short):  # Long-term memory important
            arch_scores['transformer'] += 0.3
            arch_scores['lstm_attention'] += 0.2
        else:  # Short-term patterns dominant
            arch_scores['cnn_lstm'] += 0.3
            arch_scores['multi_scale'] += 0.2
        
        # Score based on frequency domain
        freq_strength = data_chars.get('dominant_freq_strength', 0.0)
        if freq_strength > 0.1:  # Strong frequency patterns
            arch_scores['multi_scale'] += 0.3   # Multi-scale captures frequencies
            arch_scores['cnn_lstm'] += 0.2
        
        # Normalize scores
        max_score = max(arch_scores.values()) if arch_scores.values() else 1.0
        if max_score > 0:
            arch_scores = {k: v/max_score for k, v in arch_scores.items()}
        
        # Select top architectures
        sorted_archs = sorted(arch_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create configurations for top architectures
        for arch_name, score in sorted_archs[:4]:  # Top 4 architectures
            if score > 0.3:  # Only include if reasonably good
                config = self._create_arch_config(arch_name, data_chars)
                if config:
                    config['selection_score'] = score
                    config['selection_reason'] = self._get_selection_reason(arch_name, data_chars)
                    recommendations.append(config)
        
        # Ensure we have at least 2 architectures
        if len(recommendations) < 2:
            # Add best fallback architectures
            fallbacks = ['transformer', 'cnn_lstm']
            for fallback in fallbacks:
                if not any(r['type'] == fallback for r in recommendations):
                    config = self._create_arch_config(fallback, data_chars)
                    if config:
                        config['selection_score'] = 0.5
                        config['selection_reason'] = "Fallback - reliable performer"
                        recommendations.append(config)
                        if len(recommendations) >= 2:
                            break
        
        return recommendations
    
    def _create_arch_config(self, arch_name: str, data_chars: Dict) -> Optional[Dict]:
        """Create configuration for specific architecture"""
        
        seq_len = data_chars.get('sequence_length', 30)
        n_features = data_chars.get('n_features', 20)
        volatility = data_chars.get('volatility', 0.5)
        
        # Adaptive parameters based on data characteristics
        if volatility > 1.0:
            dropout_rate = 0.2  # Higher dropout for noisy data
            learning_rate = 0.0005  # Lower LR for stability
        else:
            dropout_rate = 0.1
            learning_rate = 0.001
        
        configs = {
            'transformer': {
                'type': 'transformer',
                'num_transformer_blocks': 3 if seq_len > 40 else 2,
                'd_model': min(128, n_features * 4),
                'num_heads': 8 if n_features >= 20 else 4,
                'dropout_rate': dropout_rate,
                'optimizer': 'adamw',  # Better for transformers
                'loss': 'huber'
            },
            'cnn_lstm': {
                'type': 'cnn_lstm',
                'conv_filters': [64, 128] if seq_len > 30 else [32, 64],
                'kernel_sizes': [3, 5, 7] if seq_len > 40 else [3, 5],
                'lstm_units': [128, 64],
                'dropout_rate': dropout_rate,
                'optimizer': 'adam',
                'loss': 'mae'
            },
            'gru_attention': {
                'type': 'gru_attention',
                'gru_units': [128, 64],
                'attention_units': 64,
                'dropout_rate': dropout_rate,
                'optimizer': 'rmsprop',
                'loss': 'huber'
            },
            'multi_scale': {
                'type': 'multi_scale',
                'scales': [1, 3, 5, 7] if seq_len > 30 else [1, 3, 5],
                'filters_per_scale': 32,
                'lstm_units': 128,
                'dropout_rate': dropout_rate,
                'optimizer': 'adam',
                'loss': 'mse'
            },
            'lstm_attention': {
                'type': 'lstm_attention',
                'lstm_units': [128, 64],
                'attention_units': 64,
                'dropout_rate': dropout_rate,
                'optimizer': 'adam',
                'loss': 'mse'
            },
            'residual_lstm': {
                'type': 'residual_lstm',
                'lstm_units': [128, 128, 64],
                'residual_connections': True,
                'dropout_rate': dropout_rate,
                'optimizer': 'adam',
                'loss': 'huber'
            }
        }
        
        return configs.get(arch_name)
    
    def _get_selection_reason(self, arch_name: str, data_chars: Dict) -> str:
        """Get human-readable reason for architecture selection"""
        
        reasons = {
            'transformer': "Excellent for long sequences and complex patterns",
            'cnn_lstm': "Great for trend detection and local pattern recognition",
            'gru_attention': "Efficient memory with selective attention",
            'multi_scale': "Captures patterns at multiple time scales",
            'lstm_attention': "Strong memory with attention mechanism",
            'residual_lstm': "Deep architecture with residual connections"
        }
        
        base_reason = reasons.get(arch_name, "Reliable architecture")
        
        # Add data-specific reasoning
        seq_len = data_chars.get('sequence_length', 30)
        volatility = data_chars.get('volatility', 0.5)
        trend_strength = data_chars.get('trend_strength', 0.0)
        
        specifics = []
        if seq_len > 50:
            specifics.append("long sequence data")
        if volatility > 1.0:
            specifics.append("high volatility")
        if trend_strength > 0.5:
            specifics.append("strong trend patterns")
        
        if specifics:
            return f"{base_reason} - Optimal for {', '.join(specifics)}"
        
        return base_reason
    
    def recommend_ensemble_method(self, n_models: int, data_chars: Dict) -> str:
        """
        Recommend optimal ensemble method based on data and model diversity
        
        Args:
            n_models: Number of models in ensemble
            data_chars: Data characteristics
            
        Returns:
            Recommended ensemble method
        """
        volatility = data_chars.get('volatility', 0.5)
        
        if n_models < 3:
            return 'simple_average'  # Not enough models for sophisticated weighting
        
        if volatility > 1.0:
            return 'median'  # Robust to outliers in high volatility
        
        return 'weighted_average'  # Best general performance
    
    def optimize_ensemble_size(self, data_chars: Dict, max_size: int = 7) -> int:
        """
        Recommend optimal ensemble size based on data characteristics
        
        Args:
            data_chars: Data characteristics
            max_size: Maximum allowed ensemble size
            
        Returns:
            Recommended ensemble size
        """
        base_size = 3  # Minimum for meaningful ensemble
        
        # Increase size for complex data
        seq_len = data_chars.get('sequence_length', 30)
        n_features = data_chars.get('n_features', 20)
        volatility = data_chars.get('volatility', 0.5)
        
        complexity_score = 0
        if seq_len > 40:
            complexity_score += 1
        if n_features > 25:
            complexity_score += 1
        if volatility > 1.0:
            complexity_score += 1
        
        recommended_size = min(base_size + complexity_score, max_size)
        return recommended_size
    
    def create_optimized_config(self, X: np.ndarray, y: np.ndarray, 
                              max_ensemble_size: int = 7) -> Dict:
        """
        Create optimized ensemble configuration
        
        Args:
            X: Input features
            y: Target values
            max_ensemble_size: Maximum ensemble size
            
        Returns:
            Optimized configuration dictionary
        """
        # Analyze data
        data_chars = self.analyze_data_characteristics(X, y)
        
        # Get recommendations
        arch_configs = self.recommend_architectures(data_chars)
        ensemble_method = self.recommend_ensemble_method(len(arch_configs), data_chars)
        ensemble_size = self.optimize_ensemble_size(data_chars, max_ensemble_size)
        
        # Ensure we have enough architectures
        while len(arch_configs) < ensemble_size:
            # Duplicate best architectures with slight variations
            best_config = arch_configs[0].copy()
            best_config['dropout_rate'] += 0.05  # Slight variation
            arch_configs.append(best_config)
        
        # Trim to ensemble size
        arch_configs = arch_configs[:ensemble_size]
        
        config = {
            'data_characteristics': data_chars,
            'ensemble_architectures': arch_configs,
            'ensemble_method': ensemble_method,
            'ensemble_size': ensemble_size,
            'optimization_summary': {
                'total_complexity_score': self._calculate_complexity_score(data_chars),
                'recommended_method_reason': self._get_method_reason(ensemble_method, data_chars),
                'architecture_selection_summary': [
                    f"{cfg['type']}: {cfg['selection_reason']}" 
                    for cfg in arch_configs
                ]
            }
        }
        
        return config
    
    def _calculate_complexity_score(self, data_chars: Dict) -> float:
        """Calculate data complexity score"""
        score = 0.0
        
        seq_len = data_chars.get('sequence_length', 30)
        score += min(seq_len / 100.0, 1.0)  # Sequence complexity
        
        volatility = data_chars.get('volatility', 0.5)
        score += min(volatility, 1.0)  # Volatility complexity
        
        trend_strength = data_chars.get('trend_strength', 0.0)
        score += (1.0 - trend_strength)  # Less trend = more complex
        
        return score / 3.0  # Normalize to 0-1
    
    def _get_method_reason(self, method: str, data_chars: Dict) -> str:
        """Get reason for ensemble method selection"""
        reasons = {
            'weighted_average': "Optimal for balanced performance weighting",
            'simple_average': "Simple and robust for small ensembles",
            'median': "Robust to outliers in high volatility environments"
        }
        
        base_reason = reasons.get(method, "Default selection")
        volatility = data_chars.get('volatility', 0.5)
        
        if method == 'median' and volatility > 1.0:
            return f"{base_reason} - High volatility detected ({volatility:.2f})"
        
        return base_reason


def optimize_helformer_ensemble(X_train: np.ndarray, y_train: np.ndarray, 
                               max_ensemble_size: int = 7) -> Dict:
    """
    Main function to optimize Helformer ensemble configuration
    
    Args:
        X_train: Training features
        y_train: Training targets  
        max_ensemble_size: Maximum ensemble size
        
    Returns:
        Optimized configuration
    """
    optimizer = AdaptiveEnsembleOptimizer()
    config = optimizer.create_optimized_config(X_train, y_train, max_ensemble_size)
    
    logger.info("🧠 Adaptive Ensemble Optimization Complete!")
    logger.info(f"📊 Data Complexity Score: {config['optimization_summary']['total_complexity_score']:.2f}")
    logger.info(f"🔧 Recommended Ensemble Size: {config['ensemble_size']}")
    logger.info(f"⚖️  Recommended Method: {config['ensemble_method']}")
    logger.info(f"🏗️  Selected Architectures: {[cfg['type'] for cfg in config['ensemble_architectures']]}")
    
    return config
