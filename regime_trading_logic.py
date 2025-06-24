"""
Regime-Aware Trading Logic for Helformer System
Implements adaptive trading strategies based on market regime detection
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from config_helformer import config
from market_regime_detector import MarketRegimeDetector, RegimeClassification
import logging

logger = logging.getLogger(__name__)

@dataclass
class TradingDecision:
    """Container for trading decision with regime-specific parameters"""
    should_trade: bool
    direction: str  # 'LONG', 'SHORT', 'HOLD'
    position_size_multiplier: float
    stop_loss_pct: float
    take_profit_pct: float
    confidence_required: float
    max_holding_periods: int
    reason: str

class RegimeAwareTradingLogic:
    """
    Implements adaptive trading strategies based on market regime classification.
    
    Different strategies for different market conditions:
    - Trending markets: Momentum strategies with larger positions
    - Mean-reverting markets: Contrarian strategies with smaller positions
    - Neutral markets: Conservative approach with tight risk management
    """
    
    def __init__(self, regime_detector: MarketRegimeDetector = None):
        """
        Initialize regime-aware trading logic.
        
        Args:
            regime_detector: MarketRegimeDetector instance
        """
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self.regime_params = config.REGIME_TRADING_PARAMS
        self.current_regime = None
        self.regime_history = []
        
    def analyze_trading_opportunity(self, 
                                   asset_data: pd.DataFrame,
                                   prediction: float,
                                   confidence: float,
                                   current_price: float) -> TradingDecision:
        """
        Analyze trading opportunity with regime-aware logic.
        
        Args:
            asset_data: Historical price data
            prediction: Model prediction (normalized price change)
            confidence: Prediction confidence score
            current_price: Current asset price
            
        Returns:
            TradingDecision with regime-specific parameters
        """
        try:
            # Detect current market regime
            regime_classification = self.regime_detector.detect_regime(
                asset_data['close'], 
                asset_data.index[-1]
            )
            
            if regime_classification is None:
                return self._create_no_trade_decision("No regime classification available")
            
            self.current_regime = regime_classification
            self.regime_history.append(regime_classification)
            
            # Get regime-specific parameters
            regime_type = regime_classification.regime_type
            regime_params = self.regime_params.get(regime_type, self.regime_params['neutral'])
            
            # Calculate predicted price change
            predicted_change = prediction
            predicted_price = current_price * (1 + predicted_change)
            
            # Apply regime-specific trading logic
            trading_decision = self._apply_regime_strategy(
                regime_type, regime_params, predicted_change, 
                confidence, regime_classification
            )
            
            logger.info(f"Regime Analysis: {regime_type} (H={regime_classification.hurst_exponent:.3f})")
            logger.info(f"Trading Decision: {trading_decision.direction} - {trading_decision.reason}")
            
            return trading_decision
            
        except Exception as e:
            logger.error(f"Error in regime analysis: {str(e)}")
            return self._create_no_trade_decision(f"Analysis error: {str(e)}")
    
    def _apply_regime_strategy(self, 
                              regime_type: str,
                              regime_params: Dict,
                              predicted_change: float,
                              confidence: float,
                              regime_classification: RegimeClassification) -> TradingDecision:
        """Apply regime-specific trading strategy"""
        
        # Check minimum confidence requirement
        if confidence < regime_params['min_confidence']:
            return self._create_no_trade_decision(
                f"Confidence {confidence:.2f} below regime threshold {regime_params['min_confidence']:.2f}"
            )
        
        # Determine trade direction based on regime and prediction
        if regime_type in ['trending_strong', 'trending_weak']:
            return self._trending_market_strategy(regime_params, predicted_change, confidence, regime_classification)
        elif regime_type == 'mean_reverting':
            return self._mean_reverting_strategy(regime_params, predicted_change, confidence, regime_classification)
        else:  # neutral
            return self._neutral_market_strategy(regime_params, predicted_change, confidence, regime_classification)
    
    def _trending_market_strategy(self, 
                                 regime_params: Dict,
                                 predicted_change: float,
                                 confidence: float,
                                 regime_classification: RegimeClassification) -> TradingDecision:
        """Strategy for trending markets - momentum-based"""
        
        # In trending markets, follow the momentum
        min_change = config.MIN_PREDICTION_CONFIDENCE * 0.5  # Lower threshold for trends
        
        if abs(predicted_change) < min_change:
            return self._create_no_trade_decision(f"Change {predicted_change:.2%} too small for trending market")
        
        # Stronger trends allow for more aggressive positioning
        trend_strength = regime_classification.hurst_exponent
        position_multiplier = regime_params['position_multiplier'] * min(trend_strength * 1.5, 2.0)
        
        direction = 'LONG' if predicted_change > 0 else 'SHORT'
        
        return TradingDecision(
            should_trade=True,
            direction=direction,
            position_size_multiplier=position_multiplier,
            stop_loss_pct=regime_params['stop_loss_pct'],
            take_profit_pct=regime_params['take_profit_pct'],
            confidence_required=regime_params['min_confidence'],
            max_holding_periods=regime_params['max_holding_periods'],
            reason=f"Trending market momentum ({trend_strength:.2f} Hurst) - {direction}"
        )
    
    def _mean_reverting_strategy(self, 
                                regime_params: Dict,
                                predicted_change: float,
                                confidence: float,
                                regime_classification: RegimeClassification) -> TradingDecision:
        """Strategy for mean-reverting markets - contrarian approach"""
        
        # In mean-reverting markets, we need higher confidence for smaller moves
        min_change = config.MIN_PREDICTION_CONFIDENCE * 1.5  # Higher threshold for mean reversion
        
        if abs(predicted_change) < min_change:
            return self._create_no_trade_decision(f"Change {predicted_change:.2%} too small for mean-reverting market")
        
        # Mean reversion requires very high confidence
        if confidence < 0.85:
            return self._create_no_trade_decision(f"Confidence {confidence:.2f} insufficient for mean reversion")
        
        # Adjust position size based on mean reversion strength
        reversion_strength = 1.0 - regime_classification.hurst_exponent  # Lower Hurst = stronger reversion
        position_multiplier = regime_params['position_multiplier'] * (reversion_strength * 0.8)
        
        # In mean-reverting markets, we can trade against small moves but with tight stops
        direction = 'LONG' if predicted_change > 0 else 'SHORT'
        
        return TradingDecision(
            should_trade=True,
            direction=direction,
            position_size_multiplier=position_multiplier,
            stop_loss_pct=regime_params['stop_loss_pct'],
            take_profit_pct=regime_params['take_profit_pct'],
            confidence_required=regime_params['min_confidence'],
            max_holding_periods=regime_params['max_holding_periods'],
            reason=f"Mean-reverting scalp ({reversion_strength:.2f} reversion) - {direction}"
        )
    
    def _neutral_market_strategy(self, 
                                regime_params: Dict,
                                predicted_change: float,
                                confidence: float,
                                regime_classification: RegimeClassification) -> TradingDecision:
        """Strategy for neutral markets - conservative approach"""
        
        # Neutral markets require larger moves and higher confidence
        min_change = config.MIN_PREDICTION_CONFIDENCE * 1.2
        
        if abs(predicted_change) < min_change:
            return self._create_no_trade_decision(f"Change {predicted_change:.2%} too small for neutral market")
        
        # Conservative position sizing in uncertain conditions
        direction = 'LONG' if predicted_change > 0 else 'SHORT'
        
        return TradingDecision(
            should_trade=True,
            direction=direction,
            position_size_multiplier=regime_params['position_multiplier'],
            stop_loss_pct=regime_params['stop_loss_pct'],
            take_profit_pct=regime_params['take_profit_pct'],
            confidence_required=regime_params['min_confidence'],
            max_holding_periods=regime_params['max_holding_periods'],
            reason=f"Neutral market conservative - {direction}"
        )
    
    def _create_no_trade_decision(self, reason: str) -> TradingDecision:
        """Create a no-trade decision with reason"""
        return TradingDecision(
            should_trade=False,
            direction='HOLD',
            position_size_multiplier=0.0,
            stop_loss_pct=0.0,
            take_profit_pct=0.0,
            confidence_required=1.0,
            max_holding_periods=0,
            reason=reason
        )
    
    def get_current_regime_info(self) -> Dict:
        """Get information about current market regime"""
        if self.current_regime is None:
            return {'regime': 'unknown', 'hurst': 0.5, 'confidence': 0.0}
        
        return {
            'regime': self.current_regime.regime_type,
            'hurst': self.current_regime.hurst_exponent,
            'confidence': self.current_regime.confidence_score,
            'timestamp': self.current_regime.timestamp,
            'supporting_metrics': self.current_regime.supporting_metrics
        }
    
    def get_regime_statistics(self) -> Dict:
        """Get statistics about regime transitions"""
        if not self.regime_history:
            return {}
        
        regime_counts = {}
        for regime in self.regime_history:
            regime_type = regime.regime_type
            regime_counts[regime_type] = regime_counts.get(regime_type, 0) + 1
        
        total_regimes = len(self.regime_history)
        regime_percentages = {k: (v/total_regimes)*100 for k, v in regime_counts.items()}
        
        return {
            'total_classifications': total_regimes,
            'regime_distribution': regime_percentages,
            'current_regime': self.current_regime.regime_type if self.current_regime else 'unknown',
            'avg_hurst': np.mean([r.hurst_exponent for r in self.regime_history]),
            'avg_confidence': np.mean([r.confidence_score for r in self.regime_history])
        }

# Global instance for easy access
regime_trading_logic = RegimeAwareTradingLogic()