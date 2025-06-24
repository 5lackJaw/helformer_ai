"""
Enhanced Live Trading Engine with Complete Implementation
Production-ready trading system with advanced features
"""

import ccxt
import pandas as pd
import numpy as np
import os
import logging
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import signal
import sys
import threading
from typing import Dict, List, Optional, Tuple

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from config_helformer import config
from training_utils import helformer_features
from regime_trading_logic import regime_trading_logic, TradingDecision
from model_status import model_monitor
from market_regime_detector import MarketRegimeDetector
from exchange_manager import ExchangeManager, initialize_exchange_manager
from portfolio_risk_manager import PortfolioRiskManager, initialize_portfolio_risk_manager
from execution_simulator import ExecutionSimulator, initialize_execution_simulator
import warnings
warnings.filterwarnings('ignore')

# Enhanced logging setup
def setup_enhanced_logging():
    """Setup comprehensive logging system"""
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Main logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler with rotation
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        'logs/helformer_trading.log', 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Detailed formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(name)20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_enhanced_logging()

# Load environment
load_dotenv()

class EnhancedHelformerBot:
    """
    Enhanced production-ready Helformer trading bot with:
    - Complete regime-aware trading logic
    - Advanced risk management
    - Real-time performance monitoring
    - Error recovery and failover
    - Portfolio optimization
    """
    
    def __init__(self):
        """Initialize enhanced trading bot"""
        
        # API Configuration
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        self.mode = os.getenv('MODE', 'testnet')
        
        if not self.api_key or not self.api_secret:
            logger.error("API credentials not found. Check .env file.")
            sys.exit(1)
        
        # Initialize exchange
        self.exchange = self._initialize_exchange()
        
        # Load trading system
        self._load_trading_system()
        
        # Initialize components
        self.regime_detector = MarketRegimeDetector()
        self.regime_logic = regime_trading_logic
        self.monitor = model_monitor
        
        # Initialize advanced components
        self.exchange_manager = None
        self.portfolio_risk_manager = None
        self.execution_simulator = None
        
        # Initialize if multi-exchange is enabled
        if config.ENABLE_MULTI_EXCHANGE:
            self._initialize_exchange_manager()
        
        # Initialize portfolio risk management
        self._initialize_portfolio_risk_manager()
        
        # Initialize execution simulator for testing
        if config.TESTNET_MODE:
            self._initialize_execution_simulator()
        
        # Trading state
        self.active_positions: Dict[str, Dict] = {}
        self.portfolio_value = 0.0
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        
        # Control flags
        self.running = False
        self.shutdown_requested = False
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'start_time': datetime.now(),
            'last_update': datetime.now()
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        logger.info("Enhanced Helformer Bot initialized successfully")
    
    def _initialize_exchange_manager(self):
        """Initialize multi-exchange manager"""
        try:
            if config.ENABLE_MULTI_EXCHANGE:
                self.exchange_manager = initialize_exchange_manager(config.EXCHANGE_CONFIGS)
                self.exchange_manager.start_orderbook_updates(config.DEFAULT_SYMBOLS)
                logger.info("Exchange manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize exchange manager: {str(e)}")
    
    def _initialize_portfolio_risk_manager(self):
        """Initialize portfolio risk management"""
        try:
            self.portfolio_risk_manager = initialize_portfolio_risk_manager(
                lookback_days=config.VAR_LOOKBACK_DAYS
            )
            logger.info("Portfolio risk manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize portfolio risk manager: {str(e)}")
    
    def _initialize_execution_simulator(self):
        """Initialize execution simulator for testing"""
        try:
            self.execution_simulator = initialize_execution_simulator(
                latency_ms=config.BASE_EXECUTION_LATENCY_MS,
                spread_bps=config.BASE_SPREAD_BPS
            )
            logger.info("Execution simulator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize execution simulator: {str(e)}")
    
    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initialize exchange connection with error handling"""
        try:
            exchange = ccxt.bybit({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'sandbox': self.mode == 'testnet',
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'linear',
                    'apiVersion': 'v5',
                },
                'timeout': 30000,  # 30 second timeout
                'rateLimit': 100,  # Rate limit in ms
            })
            
            # Test connection
            balance = exchange.fetch_balance()
            logger.info(f"Exchange connected successfully ({self.mode} mode)")
            logger.info(f"Account balance: {balance.get('USDT', {}).get('total', 0):.2f} USDT")
            
            return exchange
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {str(e)}")
            raise
    
    def _load_trading_system(self):
        """Load models and trading system"""
        try:
            import joblib
            import tensorflow as tf
            from training_utils import HoltWintersLayer
            
            # Load system info
            self.system_info = joblib.load("helformer_system_info.pkl")
            self.universal_features = joblib.load("helformer_universal_features.pkl")
            
            # Load asset-specific models
            self.asset_models = {}
            self.asset_scalers = {}
            
            for asset in self.system_info['assets']:
                asset_lower = asset.lower()
                
                # Load ensemble models
                models = []
                for i in range(config.ENSEMBLE_SIZE):
                    model_path = f"helformer_{asset_lower}_model_{i}.h5"
                    if os.path.exists(model_path):
                        model = tf.keras.models.load_model(
                            model_path,
                            custom_objects={'HoltWintersLayer': HoltWintersLayer}
                        )
                        models.append(model)
                
                # Load scaler
                scaler_path = f"helformer_{asset_lower}_scaler.pkl"
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    self.asset_scalers[asset] = scaler
                
                if models:
                    self.asset_models[asset] = models
                    logger.info(f"Loaded {len(models)} models for {asset}")
            
            if not self.asset_models:
                raise Exception("No models loaded!")
            
            logger.info(f"Trading system loaded: {list(self.asset_models.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to load trading system: {str(e)}")
            raise
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Shutdown signal received ({signum})")
        self.shutdown_requested = True
    
    def get_account_balance(self) -> float:
        """Get current account balance with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                balance = self.exchange.fetch_balance()
                return balance.get('USDT', {}).get('total', 0.0)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch balance after {max_retries} attempts: {str(e)}")
                    return 0.0
                time.sleep(1)
    
    def fetch_market_data(self, symbol: str, timeframe: str = '15m', limit: int = 200) -> pd.DataFrame:
        """Fetch market data with error handling and retry"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()
                
                return df
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
                    return pd.DataFrame()
                time.sleep(1)
    
    def get_enhanced_prediction(self, symbol: str, market_data: pd.DataFrame) -> Tuple[Optional[float], float, Dict]:
        """Get prediction with enhanced regime analysis"""
        try:
            asset = symbol.split('/')[0]
            
            if asset not in self.asset_models:
                return None, 0.0, {'error': f'No model for {asset}'}
            
            # Prepare features
            data_with_features = helformer_features(market_data.copy())
            
            if len(data_with_features) < config.SEQUENCE_LENGTH:
                return None, 0.0, {'error': 'Insufficient data'}
            
            # Get feature sequence
            feature_sequence = data_with_features[self.universal_features].tail(config.SEQUENCE_LENGTH).values
            
            # Scale features
            if asset in self.asset_scalers:
                scaler = self.asset_scalers[asset]
                feature_sequence_scaled = scaler.transform(
                    feature_sequence.reshape(-1, feature_sequence.shape[-1])
                ).reshape(feature_sequence.shape)
            else:
                return None, 0.0, {'error': f'No scaler for {asset}'}
            
            # Get ensemble predictions
            predictions = []
            start_time = time.time()
            
            for model in self.asset_models[asset]:
                pred = model.predict(
                    feature_sequence_scaled.reshape(1, *feature_sequence_scaled.shape),
                    verbose=0
                )[0][0]
                predictions.append(pred)
            
            execution_time = (time.time() - start_time) * 1000  # ms
            
            # Ensemble results
            ensemble_prediction = np.mean(predictions)
            prediction_std = np.std(predictions)
            confidence = 1 - (prediction_std / (abs(ensemble_prediction) + 1e-8))
            confidence = min(max(confidence, 0.0), 1.0)  # Clamp to [0,1]
            
            # Track prediction
            self.monitor.track_prediction(
                model_name=f"helformer_ensemble",
                asset=asset,
                prediction=ensemble_prediction,
                confidence=confidence,
                execution_time_ms=execution_time
            )
            
            prediction_info = {
                'ensemble_prediction': ensemble_prediction,
                'individual_predictions': predictions,
                'confidence': confidence,
                'prediction_std': prediction_std,
                'execution_time_ms': execution_time,
                'feature_count': len(self.universal_features)
            }
            
            return ensemble_prediction, confidence, prediction_info
            
        except Exception as e:
            logger.error(f"Error getting prediction for {symbol}: {str(e)}")
            return None, 0.0, {'error': str(e)}
    
    def analyze_trading_opportunity(self, symbol: str, market_data: pd.DataFrame) -> Optional[TradingDecision]:
        """Analyze trading opportunity with regime awareness"""
        try:
            # Get prediction
            prediction, confidence, pred_info = self.get_enhanced_prediction(symbol, market_data)
            
            if prediction is None:
                return None
            
            current_price = market_data['close'].iloc[-1]
            
            # Use regime-aware trading logic
            trading_decision = self.regime_logic.analyze_trading_opportunity(
                asset_data=market_data,
                prediction=prediction,
                confidence=confidence,
                current_price=current_price
            )
            
            # Add prediction info to decision
            trading_decision.prediction_info = pred_info
            
            return trading_decision
            
        except Exception as e:
            logger.error(f"Error analyzing opportunity for {symbol}: {str(e)}")
            return None
    
    def execute_trade(self, symbol: str, decision: TradingDecision, current_price: float) -> bool:
        """Execute trade with enhanced error handling and position management"""
        try:
            if not decision.should_trade:
                return False
            
            # Calculate position size
            account_balance = self.get_account_balance()
            if account_balance <= 0:
                logger.warning("Insufficient account balance")
                return False
            
            # Risk-adjusted position sizing
            base_size = account_balance * config.BASE_POSITION_SIZE
            adjusted_size = base_size * decision.position_size_multiplier
            
            # Apply additional risk controls
            max_position_value = account_balance * config.MAX_POSITION_SIZE
            position_value = min(adjusted_size, max_position_value)
            
            if position_value < config.MIN_TRADE_SIZE:
                logger.info(f"Position size {position_value:.2f} below minimum {config.MIN_TRADE_SIZE}")
                return False
            
            # Calculate quantity
            quantity = self.exchange.amount_to_precision(symbol, position_value / current_price)
            
            # Execute order based on direction
            order = None
            if decision.direction == 'LONG':
                order = self.exchange.create_market_buy_order(symbol, quantity)
            elif decision.direction == 'SHORT':
                # For derivatives/futures
                order = self.exchange.create_market_sell_order(symbol, quantity)
            
            if order and order['id']:
                # Store position
                self._store_position(symbol, decision, order, current_price)
                
                logger.info(f"TRADE EXECUTED: {decision.direction} {symbol}")
                logger.info(f"  Quantity: {quantity} | Value: ${position_value:.2f}")
                logger.info(f"  Entry Price: ${current_price:.4f}")
                logger.info(f"  Reason: {decision.reason}")
                
                return True
            else:
                logger.error(f"Failed to execute order for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {str(e)}")
            return False
    
    def _store_position(self, symbol: str, decision: TradingDecision, order: Dict, entry_price: float):
        """Store position information for tracking"""
        
        stop_loss_price = entry_price * (1 - decision.stop_loss_pct) if decision.direction == 'LONG' else entry_price * (1 + decision.stop_loss_pct)
        take_profit_price = entry_price * (1 + decision.take_profit_pct) if decision.direction == 'LONG' else entry_price * (1 - decision.take_profit_pct)
        
        self.active_positions[symbol] = {
            'order_id': order['id'],
            'direction': decision.direction,
            'entry_price': entry_price,
            'quantity': order['amount'],
            'entry_time': datetime.now(),
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'max_holding_periods': decision.max_holding_periods,
            'regime_type': self.regime_logic.get_current_regime_info().get('regime', 'unknown'),
            'confidence': getattr(decision, 'prediction_info', {}).get('confidence', 0.0)
        }
        
        # Update performance tracking
        self.performance_metrics['total_trades'] += 1
    
    def manage_positions(self):
        """Manage active positions with enhanced logic"""
        positions_to_close = []
        
        for symbol, position in self.active_positions.items():
            try:
                # Get current price
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # Check exit conditions
                should_close = False
                exit_reason = ""
                
                # Stop loss check
                if position['direction'] == 'LONG' and current_price <= position['stop_loss']:
                    should_close = True
                    exit_reason = "Stop Loss"
                elif position['direction'] == 'SHORT' and current_price >= position['stop_loss']:
                    should_close = True
                    exit_reason = "Stop Loss"
                
                # Take profit check
                elif position['direction'] == 'LONG' and current_price >= position['take_profit']:
                    should_close = True
                    exit_reason = "Take Profit"
                elif position['direction'] == 'SHORT' and current_price <= position['take_profit']:
                    should_close = True
                    exit_reason = "Take Profit"
                
                # Time-based exit
                elif datetime.now() - position['entry_time'] > timedelta(hours=position['max_holding_periods']):
                    should_close = True
                    exit_reason = "Time Limit"
                
                if should_close:
                    if self._close_position(symbol, current_price, exit_reason):
                        positions_to_close.append(symbol)
                
            except Exception as e:
                logger.error(f"Error managing position {symbol}: {str(e)}")
        
        # Remove closed positions
        for symbol in positions_to_close:
            del self.active_positions[symbol]
    
    def _close_position(self, symbol: str, exit_price: float, reason: str) -> bool:
        """Close position and calculate P&L"""
        try:
            position = self.active_positions[symbol]
            
            # Execute closing order
            if position['direction'] == 'LONG':
                order = self.exchange.create_market_sell_order(symbol, position['quantity'])
            else:
                order = self.exchange.create_market_buy_order(symbol, position['quantity'])
            
            if order and order['id']:
                # Calculate P&L
                if position['direction'] == 'LONG':
                    pnl = (exit_price - position['entry_price']) * position['quantity']
                else:
                    pnl = (position['entry_price'] - exit_price) * position['quantity']
                
                pnl_pct = (pnl / (position['entry_price'] * position['quantity'])) * 100
                
                # Update performance metrics
                if pnl > 0:
                    self.performance_metrics['winning_trades'] += 1
                
                self.performance_metrics['total_pnl'] += pnl
                self.daily_pnl += pnl
                
                logger.info(f"POSITION CLOSED: {symbol} - {reason}")
                logger.info(f"  Entry: ${position['entry_price']:.4f} | Exit: ${exit_price:.4f}")
                logger.info(f"  P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
                logger.info(f"  Duration: {datetime.now() - position['entry_time']}")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {str(e)}")
            return False
    
    def run_trading_cycle(self):
        """Execute one complete trading cycle"""
        try:
            logger.info("Starting trading cycle...")
            
            # Get enabled symbols
            symbols = os.getenv('SYMBOLS', 'BTC/USDT,ETH/USDT').split(',')
            symbols = [s.strip() for s in symbols if s.strip()]
            
            # Filter symbols with available models
            available_symbols = []
            for symbol in symbols:
                asset = symbol.split('/')[0]
                if asset in self.asset_models:
                    available_symbols.append(symbol)
            
            logger.info(f"Analyzing {len(available_symbols)} symbols: {available_symbols}")
            
            # Analyze each symbol
            for symbol in available_symbols:
                try:
                    # Skip if already in position
                    if symbol in self.active_positions:
                        continue
                    
                    # Fetch market data
                    market_data = self.fetch_market_data(symbol)
                    if market_data.empty:
                        continue
                    
                    # Analyze opportunity
                    decision = self.analyze_trading_opportunity(symbol, market_data)
                    if decision is None:
                        continue
                    
                    current_price = market_data['close'].iloc[-1]
                    
                    if decision.should_trade:
                        # Execute trade
                        success = self.execute_trade(symbol, decision, current_price)
                        if success:
                            time.sleep(1)  # Brief pause between trades
                    else:
                        logger.info(f"NO TRADE: {symbol} - {decision.reason}")
                
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    continue
            
            # Manage existing positions
            self.manage_positions()
            
            # Update portfolio value
            self.portfolio_value = self.get_account_balance()
            
            # Log performance summary
            self._log_performance_summary()
            
            # Save status
            self.monitor.save_status()
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {str(e)}")
    
    def _log_performance_summary(self):
        """Log performance summary"""
        total_trades = self.performance_metrics['total_trades']
        if total_trades > 0:
            win_rate = (self.performance_metrics['winning_trades'] / total_trades) * 100
            avg_pnl = self.performance_metrics['total_pnl'] / total_trades
        else:
            win_rate = 0.0
            avg_pnl = 0.0
        
        # System health
        health = self.monitor.get_system_health()
        
        logger.info("="*50)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*50)
        logger.info(f"Portfolio Value: ${self.portfolio_value:.2f}")
        logger.info(f"Daily P&L: ${self.daily_pnl:.2f}")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.info(f"Total P&L: ${self.performance_metrics['total_pnl']:.2f}")
        logger.info(f"Active Positions: {len(self.active_positions)}")
        logger.info(f"System Health: {health.models_healthy}/{health.models_total} models healthy")
        logger.info("="*50)
    
    def start_trading(self):
        """Start the main trading loop"""
        try:
            self.running = True
            logger.info("ENHANCED HELFORMER BOT STARTED")
            logger.info(f"Mode: {self.mode}")
            logger.info(f"Assets: {list(self.asset_models.keys())}")
            
            # Initial cycle
            self.run_trading_cycle()
            
            # Main loop
            while self.running and not self.shutdown_requested:
                try:
                    # Calculate next run time (every 15 minutes)
                    next_run = datetime.now().replace(second=0, microsecond=0)
                    if next_run.minute % 15 != 0:
                        next_run = next_run.replace(minute=(next_run.minute // 15 + 1) * 15)
                    else:
                        next_run += timedelta(minutes=15)
                    
                    # Sleep until next run
                    sleep_seconds = (next_run - datetime.now()).total_seconds()
                    
                    if sleep_seconds > 0:
                        logger.info(f"Next cycle in {int(sleep_seconds)} seconds at {next_run.strftime('%H:%M')}")
                        
                        # Sleep in chunks to allow for shutdown
                        while sleep_seconds > 0 and not self.shutdown_requested:
                            sleep_time = min(30, sleep_seconds)
                            time.sleep(sleep_time)
                            sleep_seconds -= sleep_time
                            
                            # Quick position check every 30 seconds
                            if len(self.active_positions) > 0:
                                self.manage_positions()
                    
                    if not self.shutdown_requested:
                        self.run_trading_cycle()
                
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    time.sleep(60)  # Wait 1 minute before retrying
        
        except Exception as e:
            logger.error(f"Critical error in trading bot: {str(e)}")
        finally:
            self._shutdown()
    
    def _shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Enhanced Helformer Bot...")
        
        # Close all positions
        if self.active_positions:
            logger.info(f"Closing {len(self.active_positions)} active positions...")
            for symbol in list(self.active_positions.keys()):
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    self._close_position(symbol, current_price, "Shutdown")
                except Exception as e:
                    logger.error(f"Error closing position {symbol}: {str(e)}")
        
        # Save final status
        self.monitor.save_status()
        
        # Generate final report
        final_report = self.monitor.generate_status_report()
        logger.info("\n" + final_report)
        
        self.running = False
        logger.info("Enhanced Helformer Bot shutdown complete")

def main():
    """Main entry point"""
    try:
        bot = EnhancedHelformerBot()
        bot.start_trading()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()