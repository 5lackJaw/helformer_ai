"""
Enhanced Helformer Integration Module
Integrates all new features into a comprehensive trading system
"""

import asyncio
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Import core modules
from config_helformer import config
from helformer_model import EnsembleManager, create_default_ensemble_configs, create_helformer_model
from improved_training_utils import (
    create_research_based_features, create_sequences
)

# Import new feature modules
from multi_timeframe_analyzer import MultiTimeframeAnalyzer, get_multi_timeframe_features
from advanced_order_manager import AdvancedOrderManager, OrderType
from realtime_data_manager import RealTimeDataManager, TickData, OrderBookSnapshot, BarData
from performance_analytics import PerformanceAnalyzer, TradeAnalysis, PerformanceMetrics

# Import existing modules
from market_regime_detector import MarketRegimeDetector
from portfolio_risk_manager import PortfolioRiskManager
from execution_simulator import ExecutionSimulator
from exchange_manager import ExchangeManager

logger = logging.getLogger(__name__)

class EnhancedHelformerSystem:
    """
    Enhanced Helformer trading system with all institutional-grade features.
    
    Integrates:
    - Multi-timeframe analysis
    - Custom loss functions and ensemble models
    - Advanced order management    - Real-time data streaming
    - Performance analytics and monitoring
    - Market regime detection
    - Portfolio risk management
    - Realistic execution simulation
    """
    
    def __init__(self, exchange_configs: Dict[str, Dict]):
        """
        Initialize the enhanced Helformer system.
        
        Args:
            exchange_configs: Dictionary of exchange configurations
        """
        # Initialize exchanges
        self.exchanges = []
        for exchange_name, config_dict in exchange_configs.items():
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class(config_dict)
            self.exchanges.append(exchange)
        
        self.primary_exchange = self.exchanges[0] if self.exchanges else None
        
        # Initialize core components
        self.regime_detector = MarketRegimeDetector()
        self.risk_manager = PortfolioRiskManager()
        self.execution_simulator = ExecutionSimulator()
        self.exchange_manager = ExchangeManager(exchange_configs) if exchange_configs else None
        
        # Initialize new components
        self.multi_tf_analyzer = None
        self.order_manager = None
        self.realtime_data_manager = None
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Model management
        self.ensemble_manager = None
        self.models = {}
        self.scalers = {}
          # System state
        self.is_running = False
        self.symbols = config.DEFAULT_SYMBOLS
        self.current_positions = {}
        self.prediction_cache = {}
        
        # Performance tracking
        self.trade_counter = 0
        self.last_analytics_update = datetime.now()
        self.last_drift_check = datetime.now()
        
        logger.info("Enhanced Helformer System initialized")
    
    async def initialize(self):
        """Initialize all system components."""
        logger.info("Initializing Enhanced Helformer System...")
        
        # Initialize real-time data manager
        if config.MODULES_ENABLED.get('realtime_data', False) and self.exchanges:
            self.realtime_data_manager = RealTimeDataManager(self.exchanges)
            await self._setup_realtime_callbacks()
        
        # Initialize order manager
        if config.MODULES_ENABLED.get('advanced_orders', False) and self.primary_exchange:
            self.order_manager = AdvancedOrderManager(self.primary_exchange)
        
        # Initialize multi-timeframe analyzer
        if config.MODULES_ENABLED.get('multi_timeframe', False) and self.primary_exchange:
            self.multi_tf_analyzer = MultiTimeframeAnalyzer(self.primary_exchange, self.symbols[0])
        
        # Initialize ensemble model
        if config.MODULES_ENABLED.get('enhanced_models', False):
            await self._initialize_ensemble_models()
        
        # Load existing models and scalers
        await self._load_models_and_scalers()
        
        logger.info("System initialization completed")
    
    async def start_live_trading(self):
        """Start live trading with all enhanced features."""
        if self.is_running:
            logger.warning("System already running")
            return
        
        self.is_running = True
        logger.info("Starting Enhanced Helformer live trading...")
        
        # Start real-time data streaming
        if self.realtime_data_manager:
            await self.realtime_data_manager.start()
            
            # Subscribe to symbols and timeframes
            for symbol in self.symbols:
                await self.realtime_data_manager.subscribe_ticker(symbol)
                await self.realtime_data_manager.subscribe_orderbook(symbol)
                
                for timeframe in config.SECONDARY_TIMEFRAMES:
                    await self.realtime_data_manager.subscribe_bars(symbol, timeframe)
        
        # Start main trading loop
        await self._main_trading_loop()
    
    async def stop_live_trading(self):
        """Stop live trading."""
        self.is_running = False
        logger.info("Stopping Enhanced Helformer live trading...")
        
        # Stop real-time data streaming
        if self.realtime_data_manager:
            await self.realtime_data_manager.stop()
        
        # Cancel all active orders
        if self.order_manager:
            for order_id in list(self.order_manager.active_orders.keys()):
                await self.order_manager.cancel_order(order_id)
        
        # Generate final performance report
        if config.MODULES_ENABLED.get('performance_analytics', False):
            await self._generate_performance_report()
        
        logger.info("Live trading stopped")
    
    async def _main_trading_loop(self):
        """Main trading loop with enhanced features."""
        logger.info("Starting main trading loop...")
        
        while self.is_running:
            try:
                # Process each symbol
                for symbol in self.symbols:
                    await self._process_symbol(symbol)
                
                # Periodic system maintenance
                await self._periodic_maintenance()
                  # Wait before next iteration
                await asyncio.sleep(config.PREDICTION_COOLDOWN)
                
            except Exception as e:
                logger.error(f"Error in main trading loop: {str(e)}")
                await asyncio.sleep(config.ERROR_RETRY_DELAY)  # Use config value for retry delay
    
    async def _process_symbol(self, symbol: str):
        """Process trading signals for a single symbol."""
        try:
            # Get multi-timeframe features
            features_df = await self._get_enhanced_features(symbol)
            if features_df is None or len(features_df) < config.SEQUENCE_LENGTH:
                return
            
            # Check timeframe alignment
            if config.MODULES_ENABLED.get('multi_timeframe', False) and self.multi_tf_analyzer:
                should_trade, reason = self.multi_tf_analyzer.should_trade_based_on_alignment({
                    config.PRIMARY_TIMEFRAME: features_df
                })
                if not should_trade:
                    logger.debug(f"Skipping {symbol}: {reason}")
                    return
            
            # Detect market regime
            regime_info = self.regime_detector.detect_current_regime(features_df)
            
            # Generate ensemble prediction
            prediction_info = await self._generate_ensemble_prediction(symbol, features_df)
            if prediction_info is None:
                return
              # Assess risk
            risk_assessment = await self._assess_trading_risk(symbol, prediction_info, regime_info)
            if risk_assessment['should_trade']:
                # Execute trade
                await self._execute_enhanced_trade(symbol, prediction_info, risk_assessment, regime_info)
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
    
    async def _get_enhanced_features(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get enhanced features with multi-timeframe analysis."""
        try:
            if config.MODULES_ENABLED.get('multi_timeframe', False) and self.multi_tf_analyzer:
                # Get multi-timeframe features
                features_df = get_multi_timeframe_features(
                    self.primary_exchange, symbol, limit=config.DATA_FETCH_LIMIT
                )
            else:
                # Get standard features
                ohlcv = self.primary_exchange.fetch_ohlcv(symbol, timeframe=config.DEFAULT_TIMEFRAME, limit=config.DATA_FETCH_LIMIT)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                features_df = create_research_based_features(df, symbol=symbol)
            
            # Add regime features if enabled
            if config.MODULES_ENABLED.get('regime_detection', False):
                from regime_training_integration import RegimeAwareFeatureEngine
                regime_engine = RegimeAwareFeatureEngine()
                features_df = regime_engine.add_regime_features(features_df)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error getting features for {symbol}: {str(e)}")
            return None
    
    async def _generate_ensemble_prediction(self, symbol: str, features_df: pd.DataFrame) -> Optional[Dict]:
        """Generate prediction using ensemble models."""
        try:
            if not self.ensemble_manager or not self.ensemble_manager.models:
                logger.warning("No ensemble models available")
                return None
            
            # Prepare features for prediction
            feature_columns = [col for col in features_df.columns 
                             if col not in ['close', 'target_price', 'target_normalized']]
            features = features_df[feature_columns].values
            
            # Create sequences
            if len(features) < config.SEQUENCE_LENGTH:
                return None
            
            X = features[-config.SEQUENCE_LENGTH:].reshape(1, config.SEQUENCE_LENGTH, -1)
            
            # Generate ensemble prediction
            prediction = self.ensemble_manager.predict_ensemble(
                X, method=config.ENSEMBLE_METHOD
            )[0]
            
            # Calculate confidence based on ensemble diversity
            confidence = 1.0 - self.ensemble_manager.get_ensemble_diversity(X)
            
            # Get timeframe confidence if available
            if config.MODULES_ENABLED.get('multi_timeframe', False) and self.multi_tf_analyzer:
                tf_confidence = self.multi_tf_analyzer.get_timeframe_confidence_score({
                    config.PRIMARY_TIMEFRAME: features_df
                })
                confidence = (confidence + tf_confidence) / 2
            
            current_price = features_df['close'].iloc[-1]
            predicted_price = current_price * (1 + prediction)
            
            prediction_info = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'current_price': current_price,
                'predicted_price': predicted_price,
                'predicted_return': prediction,
                'confidence': confidence,
                'raw_prediction': prediction
            }
            
            # Store prediction for monitoring
            self.performance_analyzer.add_prediction(
                symbol=symbol,
                timestamp=prediction_info['timestamp'],
                predicted_value=prediction,
                confidence=confidence
            )
            
            return prediction_info
            
        except Exception as e:
            logger.error(f"Error generating ensemble prediction for {symbol}: {str(e)}")
            return None
    
    async def _assess_trading_risk(self, symbol: str, prediction_info: Dict, 
                                 regime_info: Dict) -> Dict:
        """Assess trading risk with enhanced risk management."""
        try:
            # Basic confidence check
            if prediction_info['confidence'] < config.CONFIDENCE_THRESHOLD:
                return {'should_trade': False, 'reason': 'Low confidence'}
            
            # Minimum prediction threshold check
            if abs(prediction_info['predicted_return']) < config.MIN_PREDICTION_CONFIDENCE:
                return {'should_trade': False, 'reason': 'Prediction below threshold'}
            
            # Portfolio risk assessment
            if config.MODULES_ENABLED.get('risk_management', False):
                risk_metrics = self.risk_manager.calculate_portfolio_risk(self.current_positions)
                
                if risk_metrics['portfolio_var'] > config.MAX_PORTFOLIO_VAR_PCT:
                    return {'should_trade': False, 'reason': 'Portfolio VaR exceeded'}
            
            # Regime-based risk adjustment
            position_multiplier = 1.0
            if regime_info and 'regime_parameters' in regime_info:
                regime_params = regime_info['regime_parameters']
                position_multiplier = regime_params.get('position_multiplier', 1.0)
            
            # Calculate position size
            base_position_size = config.BASE_POSITION_SIZE * position_multiplier
            
            # Risk-adjusted position sizing
            if config.MODULES_ENABLED.get('risk_management', False):
                risk_adjusted_size = self.risk_manager.calculate_kelly_position_size(
                    expected_return=prediction_info['predicted_return'],
                    win_probability=prediction_info['confidence'],
                    portfolio_value=self._get_portfolio_value()
                )
                base_position_size = min(base_position_size, risk_adjusted_size)
            
            position_size = min(base_position_size, config.MAX_POSITION_SIZE)
            
            return {
                'should_trade': True,
                'position_size': position_size,
                'position_multiplier': position_multiplier,
                'risk_metrics': risk_metrics if 'risk_metrics' in locals() else {}
            }
            
        except Exception as e:
            logger.error(f"Error assessing trading risk for {symbol}: {str(e)}")
            return {'should_trade': False, 'reason': f'Risk assessment error: {str(e)}'}
    
    async def _execute_enhanced_trade(self, symbol: str, prediction_info: Dict, 
                                    risk_assessment: Dict, regime_info: Dict):
        """Execute trade using advanced order management."""
        try:
            side = 'buy' if prediction_info['predicted_return'] > 0 else 'sell'
            quantity = risk_assessment['position_size'] * self._get_portfolio_value() / prediction_info['current_price']
            
            # Determine order type based on configuration and market conditions
            order_type = self._determine_optimal_order_type(symbol, quantity, regime_info)
            
            if config.MODULES_ENABLED.get('advanced_orders', False) and self.order_manager:
                # Use advanced order management
                order_id = await self.order_manager.submit_advanced_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=order_type,
                    urgency='medium'
                )
                
                logger.info(f"Advanced order submitted: {order_id} ({order_type.value})")
                
                # Monitor order execution
                asyncio.create_task(self._monitor_order_execution(order_id, prediction_info))
                
            else:
                # Use standard market order
                if side == 'buy':
                    result = self.primary_exchange.create_market_buy_order(symbol, quantity)
                else:
                    result = self.primary_exchange.create_market_sell_order(symbol, quantity)
                
                # Record trade for analysis
                await self._record_trade(result, prediction_info, risk_assessment)
                
                logger.info(f"Market order executed: {side} {quantity} {symbol}")
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {str(e)}")
    
    def _determine_optimal_order_type(self, symbol: str, quantity: float, 
                                    regime_info: Dict) -> OrderType:
        """Determine optimal order type based on market conditions."""
        
        if not config.MODULES_ENABLED.get('advanced_orders', False):
            return OrderType.MARKET
        
        # Get current market conditions
        orderbook = None
        if self.realtime_data_manager:
            orderbook = self.realtime_data_manager.get_latest_orderbook(symbol)
        
        # Large orders use TWAP/VWAP
        portfolio_value = self._get_portfolio_value()
        order_value = quantity * self._get_current_price(symbol)
        
        if order_value > portfolio_value * 0.1:  # Large order (>10% of portfolio)
            return OrderType.VWAP if regime_info.get('volatility', 0) > 0.02 else OrderType.TWAP
        
        # High volatility regimes use VWAP
        if regime_info.get('regime_type') in ['trending_strong', 'mean_reverting']:
            return OrderType.VWAP
        
        # Default to TWAP for medium orders
        return OrderType.TWAP if order_value > portfolio_value * 0.05 else OrderType.MARKET
    
    async def _monitor_order_execution(self, order_id: str, prediction_info: Dict):
        """Monitor advanced order execution."""
        try:
            while True:
                order = self.order_manager.get_order_status(order_id)
                if not order or order.status.value in ['filled', 'cancelled', 'failed']:
                    break
                
                await asyncio.sleep(30)  # Check every 30 seconds
            
            if order and order.status.value == 'filled':
                # Record completed trade
                trade_analysis = TradeAnalysis(
                    trade_id=order_id,
                    symbol=order.symbol,
                    entry_time=order.start_time,
                    exit_time=datetime.now(),
                    side=order.side,
                    entry_price=order.avg_fill_price,
                    exit_price=order.avg_fill_price,  # Same for immediate execution
                    quantity=order.filled_quantity,
                    pnl=0.0,  # Will be calculated when position is closed
                    pnl_pct=0.0,
                    fees=order.total_fees,
                    hold_time_hours=0.0,
                    predicted_return=prediction_info['predicted_return'],
                    confidence_score=prediction_info['confidence']
                )
                
                self.performance_analyzer.add_trade(trade_analysis)
                logger.info(f"Order {order_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error monitoring order {order_id}: {str(e)}")
    
    async def _periodic_maintenance(self):
        """Perform periodic system maintenance tasks."""
        now = datetime.now()
        
        # Performance analytics update
        if (config.MODULES_ENABLED.get('performance_analytics', False) and
            (now - self.last_analytics_update).total_seconds() > 3600):  # Every hour
            
            await self._update_performance_analytics()
            self.last_analytics_update = now
        
        # Model drift detection
        if (config.MODULES_ENABLED.get('performance_analytics', False) and
            config.ENABLE_MODEL_DRIFT_DETECTION and
            (now - self.last_drift_check).total_seconds() > config.DRIFT_CHECK_FREQUENCY * 3600):
            
            await self._check_model_drift()
            self.last_drift_check = now
        
        # Ensemble weight updates
        if (config.MODULES_ENABLED.get('enhanced_models', False) and
            config.UPDATE_ENSEMBLE_WEIGHTS and
            len(self.performance_analyzer.model_predictions) % config.ENSEMBLE_WEIGHT_UPDATE_FREQUENCY == 0):
            
            await self._update_ensemble_weights()
    
    async def _update_performance_analytics(self):
        """Update performance analytics."""
        try:
            # Generate performance report
            if config.ANALYTICS_REPORTING_FREQUENCY == 'daily':
                report = self.performance_analyzer.generate_performance_report()
                logger.info("Daily performance report generated")
                
                # Save report if configured
                if config.AUTO_EXPORT_DATA:
                    os.makedirs(config.EXPORT_DIRECTORY, exist_ok=True)
                    report_path = f"{config.EXPORT_DIRECTORY}/daily_report_{datetime.now().strftime('%Y%m%d')}.txt"
                    with open(report_path, 'w') as f:
                        f.write(report)
            
            # Generate charts if configured
            if config.AUTO_GENERATE_CHARTS:
                os.makedirs(config.CHART_SAVE_DIRECTORY, exist_ok=True)
                self.performance_analyzer.plot_performance_charts(config.CHART_SAVE_DIRECTORY)
                
        except Exception as e:
            logger.error(f"Error updating performance analytics: {str(e)}")
    
    async def _check_model_drift(self):
        """Check for model performance drift."""
        try:
            drift_analysis = self.performance_analyzer.detect_model_drift(
                window_size=config.DRIFT_DETECTION_WINDOW,
                threshold=config.DRIFT_DETECTION_THRESHOLD
            )
            
            if drift_analysis['drift_detected']:
                logger.warning(f"Model drift detected: {drift_analysis}")
                
                # Trigger model retraining if severely drifted
                if drift_analysis['drift_ratio'] > 1.5:  # 50% performance degradation
                    logger.critical("Severe model drift detected - consider retraining")
                    # Could automatically trigger retraining here
                    
        except Exception as e:
            logger.error(f"Error checking model drift: {str(e)}")
    
    async def _update_ensemble_weights(self):
        """Update ensemble model weights based on recent performance."""
        try:
            if self.ensemble_manager and len(self.performance_analyzer.model_predictions) > 100:
                # Get recent predictions for validation
                recent_predictions = self.performance_analyzer.model_predictions[-100:]
                
                # Create synthetic validation data from recent predictions
                X_val = np.random.randn(100, config.SEQUENCE_LENGTH, 20)  # Placeholder
                y_val = np.array([p['actual_value'] for p in recent_predictions if p['actual_value'] is not None])
                
                if len(y_val) > 50:  # Need sufficient actual values
                    X_val = X_val[:len(y_val)]
                    self.ensemble_manager._update_ensemble_weights(X_val, y_val)
                    logger.info("Ensemble weights updated based on recent performance")
                    
        except Exception as e:
            logger.error(f"Error updating ensemble weights: {str(e)}")
    
    async def _setup_realtime_callbacks(self):
        """Setup callbacks for real-time data processing."""
        
        def on_tick_data(tick: TickData):
            """Process real-time tick data."""
            # Update current price cache
            self.prediction_cache[f"{tick.symbol}_price"] = tick.price
            
            # Trigger regime detection update if needed
            # (Implementation would check timing and update regime)
        
        def on_orderbook_data(orderbook: OrderBookSnapshot):
            """Process real-time order book data."""
            # Update liquidity metrics
            if orderbook.best_bid and orderbook.best_ask:
                spread = orderbook.spread
                # Store spread for execution analysis
        
        def on_bar_data(bar: BarData):
            """Process real-time bar data."""
            # Update features and trigger prediction if needed
            # (Implementation would update feature cache)
            pass
        
        # Add callbacks to real-time data manager
        if self.realtime_data_manager:
            self.realtime_data_manager.add_tick_callback(on_tick_data)
            self.realtime_data_manager.add_orderbook_callback(on_orderbook_data)
            self.realtime_data_manager.add_bar_callback(on_bar_data)
    
    async def _initialize_ensemble_models(self):
        """Initialize ensemble models with different architectures."""
        try:
            if not config.MODULES_ENABLED.get('enhanced_models', False):
                return
            
            # Create ensemble configurations
            ensemble_configs = create_default_ensemble_configs()
            
            # Override with custom configurations if specified
            if hasattr(config, 'ENSEMBLE_ARCHITECTURES'):
                ensemble_configs = [
                    cfg for cfg in ensemble_configs 
                    if cfg['type'] in config.ENSEMBLE_ARCHITECTURES
                ]
            
            # Create ensemble manager
            self.ensemble_manager = EnsembleManager(ensemble_configs)
            
            # Note: Model training would happen separately during training phase
            logger.info(f"Ensemble manager initialized with {len(ensemble_configs)} architectures")
            
        except Exception as e:
            logger.error(f"Error initializing ensemble models: {str(e)}")
    
    async def _load_models_and_scalers(self):
        """Load pre-trained models and scalers."""
        try:
            # Load ensemble if available
            if self.ensemble_manager:
                try:
                    self.ensemble_manager.load_ensemble('./models/helformer_ensemble')
                    logger.info("Ensemble models loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load ensemble models: {str(e)}")
            
            # Load individual models as fallback
            for symbol in self.symbols:
                try:
                    # Load symbol-specific models (implementation depends on existing structure)
                    pass
                except Exception as e:
                    logger.warning(f"Could not load models for {symbol}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error loading models and scalers: {str(e)}")
    
    async def _record_trade(self, order_result: Dict, prediction_info: Dict, risk_assessment: Dict):
        """Record trade for performance analysis."""
        try:
            self.trade_counter += 1
            
            trade_analysis = TradeAnalysis(
                trade_id=f"trade_{self.trade_counter}",
                symbol=prediction_info['symbol'],
                entry_time=datetime.now(),
                exit_time=datetime.now(),  # For market orders
                side=order_result.get('side', 'unknown'),
                entry_price=order_result.get('average', prediction_info['current_price']),
                exit_price=order_result.get('average', prediction_info['current_price']),
                quantity=order_result.get('filled', 0),
                pnl=0.0,  # Will be calculated when position is closed
                pnl_pct=0.0,
                fees=order_result.get('fee', {}).get('cost', 0),
                hold_time_hours=0.0,
                predicted_return=prediction_info['predicted_return'],
                confidence_score=prediction_info['confidence']
            )
            
            self.performance_analyzer.add_trade(trade_analysis)
            
        except Exception as e:
            logger.error(f"Error recording trade: {str(e)}")
    
    def _get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        # Simplified implementation - would need proper balance tracking
        return 10000.0  # Placeholder
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            if self.realtime_data_manager:
                tick = self.realtime_data_manager.get_latest_tick(symbol)
                if tick:
                    return tick.price
            
            # Fallback to exchange API
            ticker = self.primary_exchange.fetch_ticker(symbol)
            return ticker['last']
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return 0.0
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        status = {
            'is_running': self.is_running,
            'timestamp': datetime.now().isoformat(),
            'exchanges': len(self.exchanges),
            'symbols': self.symbols,
            'modules_enabled': config.MODULES_ENABLED,
            'components': {
                'regime_detector': self.regime_detector is not None,
                'risk_manager': self.risk_manager is not None,
                'execution_simulator': self.execution_simulator is not None,
                'exchange_manager': self.exchange_manager is not None,
                'multi_tf_analyzer': self.multi_tf_analyzer is not None,
                'order_manager': self.order_manager is not None,
                'realtime_data_manager': self.realtime_data_manager is not None,
                'performance_analyzer': self.performance_analyzer is not None,
                'ensemble_manager': self.ensemble_manager is not None
            }
        }
        
        # Add performance summary
        if self.performance_analyzer.trades:
            latest_metrics = self.performance_analyzer.calculate_performance_metrics()
            status['performance'] = {
                'total_trades': len(self.performance_analyzer.trades),
                'win_rate': latest_metrics.win_rate,
                'total_return': latest_metrics.total_return,
                'sharpe_ratio': latest_metrics.sharpe_ratio
            }
        
        return status

def create_enhanced_helformer_system(exchange_configs: List[Dict]) -> EnhancedHelformerSystem:
    """Factory function to create an enhanced Helformer system."""
    return EnhancedHelformerSystem(exchange_configs)

if __name__ == "__main__":
    # Example usage
    async def main():
        # Example exchange configuration
        exchange_configs = [
            {
                'exchange': 'binance',
                'params': {
                    'apiKey': 'your_api_key',
                    'secret': 'your_secret',
                    'sandbox': True,  # Use testnet for safety
                    'enableRateLimit': True
                }
            }
        ]
        
        # Create and initialize system
        system = create_enhanced_helformer_system(exchange_configs)
        await system.initialize()
        
        # Get system status
        status = system.get_system_status()
        print(f"System Status: {status}")
        
        # Start live trading (commented out for safety)
        # await system.start_live_trading()
    
    # Run example
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("System stopped by user")
    except Exception as e:
        print(f"System error: {e}")