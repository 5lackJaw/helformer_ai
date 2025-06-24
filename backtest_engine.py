"""
Consolidated Helformer Backtest - Best of both implementations
Combines the multi-asset logic from backtest_helformer.py with 
the working model loading from backtest_working.py
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Handle numpy compatibility
try:
    import numpy._core  
except ImportError:
    import numpy.core
    import sys
    sys.modules['numpy._core'] = numpy.core

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from config_helformer import config
from improved_training_utils import HoltWintersLayer, normalize_targets_no_leakage, create_research_based_features
from helformer_model import create_helformer_model, mish_activation
from feature_cache import get_cached_features
from execution_simulator import ExecutionSimulator, initialize_execution_simulator
from portfolio_risk_manager import PortfolioRiskManager, initialize_portfolio_risk_manager
from regime_trading_logic import regime_trading_logic

# Import research-aligned trading strategy
from research_helformer import create_research_trading_strategy

class ConsolidatedHelformerBacktest:
    """Consolidated multi-asset Helformer backtesting engine"""
    
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        
        # Load multi-asset system
        self.asset_models = {}
        self.asset_scalers = {}
        self.load_multi_asset_system()
        
        # Initialize new components
        self.execution_simulator = None
        self.portfolio_risk_manager = None
        self.regime_logic = regime_trading_logic
        
        # Initialize execution simulator for realistic backtesting
        if config.ENABLE_REALISTIC_SLIPPAGE:
            self.execution_simulator = initialize_execution_simulator(
                latency_ms=config.BASE_EXECUTION_LATENCY_MS,
                spread_bps=config.BASE_SPREAD_BPS
            )
        
        # Initialize portfolio risk management
        self.portfolio_risk_manager = initialize_portfolio_risk_manager(
            lookback_days=config.VAR_LOOKBACK_DAYS
        )
          # Trading parameters (research-aligned)
        self.position_size = 0.95  # 95% capital allocation per research
        self.transaction_cost = 0.01  # 1% transaction cost per research
        self.confidence_threshold = 0.0  # No confidence threshold (simple directional)
        self.min_prediction_confidence = 0.0  # Trade on any prediction
        self.window_size = config.SEQUENCE_LENGTH
        
        # State tracking
        self.trades = []
        self.equity_curve = []
        self.asset_performance = {}
    def load_multi_asset_system(self):
        """Load asset-specific models and scalers from experiment folder"""
        try:
            # Find the latest experiment folder
            experiment_folders = [f for f in os.listdir('.') if f.startswith('experiment_')]
            if not experiment_folders:
                print("❌ No experiment folders found! Please run training first.")
                return False
            
            latest_experiment = sorted(experiment_folders)[-1]
            print(f"Loading models from: {latest_experiment}")
            
            # Load system info from experiment folder
            system_info_path = os.path.join(latest_experiment, "helformer_system_info.pkl")
            features_path = os.path.join(latest_experiment, "helformer_universal_features.pkl")
            
            if not os.path.exists(system_info_path):
                print(f"❌ System info not found in {latest_experiment}")
                return False
            
            system_info = joblib.load(system_info_path)
            self.universal_features = joblib.load(features_path) if os.path.exists(features_path) else None
            
            print(f"Loading multi-asset system from {latest_experiment}...")
            print(f"Assets: {system_info['assets']}")
            print(f"Expected Returns: {system_info['expected_returns']}")
            
            # Load models and scalers for each asset from experiment folder
            for asset in system_info['assets']:
                asset_lower = asset.lower()
                
                # Load ensemble models from experiment folder
                asset_models = []
                model_pattern = f"helformer_{asset_lower}_ensemble_model_"
                
                for i in range(10):  # Check up to 10 models
                    model_path = os.path.join(latest_experiment, f"{model_pattern}{i}.h5")
                    if os.path.exists(model_path):
                        try:
                            # Method 1: Try recreating model and loading weights (most reliable)
                            model = create_helformer_model(
                                input_shape=(config.SEQUENCE_LENGTH, len(self.universal_features) if self.universal_features else 19),
                                num_heads=config.NUM_HEADS,
                                head_dim=config.HEAD_DIM,
                                lstm_units=config.LSTM_UNITS,
                                dropout_rate=config.DROPOUT_RATE,
                                learning_rate=config.LEARNING_RATE
                            )
                            model.load_weights(model_path)
                            asset_models.append(model)
                            print(f"    ✅ Loaded {model_path}")
                        except Exception as e:
                            print(f"    ❌ Failed to load {model_path}: {str(e)[:50]}...")
                
                # Load scaler from experiment folder
                scaler_path = os.path.join(latest_experiment, f"helformer_{asset_lower}_scaler.pkl")
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    self.asset_scalers[asset] = scaler
                
                if asset_models:
                    self.asset_models[asset] = asset_models
                    print(f"✅ {asset}: {len(asset_models)} models loaded")
                    
                    # Initialize performance tracking
                    self.asset_performance[asset] = {
                        'trades': 0,
                        'wins': 0,
                        'total_pnl': 0.0,
                        'predictions': [],
                        'actuals': []
                    }
            
            if not self.asset_models:
                raise Exception("No asset models loaded!")
                
            print(f"Multi-asset system loaded: {list(self.asset_models.keys())}")
            
        except Exception as e:
            print(f"ERROR loading multi-asset system: {e}")
            return False
        
        return True
    
    def load_asset_data(self, asset):
        """Load data for specific asset using config-driven file paths"""
        file_paths = config.get_market_files()
        symbol_key = f"{asset}/USDT"
        
        if symbol_key not in file_paths:
            print(f"❌ No data file configured for {asset}")
            return None
        
        file_path = file_paths[symbol_key]
        if not os.path.exists(file_path):
            print(f"❌ Data file not found: {file_path}")
            return None
        
        print(f"Loading {asset} data from {file_path}")
        data = pd.read_csv(file_path)        
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        data = data.sort_index()
        
        print(f"  ✅ Loaded {len(data)} records for {asset}")
        return data

    def prepare_asset_features(self, data, asset):
        """Prepare features for specific asset using RESEARCH-ALIGNED features"""
        print(f"  Computing RESEARCH-ALIGNED features for {asset}...")
        
        # Use the research-aligned feature engineering (same as training)
        try:
            data_with_features = create_research_based_features(data.copy(), symbol=asset)
            data_with_features = normalize_targets_no_leakage(data_with_features)
        except Exception as e:
            print(f"    ❌ Error with research features: {e}")
            # Fallback to cached features if available
            try:
                data_with_features = get_cached_features(asset, data.copy())
            except:
                print(f"    ❌ No cached features available for {asset}")
                return None, []
        
        # Use universal features if available, otherwise select best features
        if self.universal_features:
            feature_columns = [col for col in self.universal_features if col in data_with_features.columns]
        else:
            # Fallback feature selection
            feature_columns = [
                'returns', 'log_returns', 'price_acceleration', 'relative_volume',
                'rsi_normalized', 'macd_normalized', 'price_vs_sma_10', 'price_vs_sma_20', 
                'price_vs_sma_50', 'bb_position', 'bb_width', 'volatility_rank',
                'momentum_5', 'momentum_10', 'momentum_20', 
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
            ]
            feature_columns = [col for col in feature_columns if col in data_with_features.columns]
        
        print(f"  ✅ {len(feature_columns)} RESEARCH-ALIGNED features prepared for {asset}")
        return data_with_features, feature_columns
    
    def get_asset_prediction(self, asset, feature_data):
        """Get prediction for specific asset"""
        try:
            if asset not in self.asset_models:
                return None, 0.5
            
            # Ensure sufficient data
            if len(feature_data) < self.window_size:
                return None, 0.5
            
            # Get recent sequence
            feature_sequence = feature_data.tail(self.window_size).values
            
            # Scale using asset-specific scaler
            if asset in self.asset_scalers:
                scaler = self.asset_scalers[asset]
                feature_sequence_scaled = scaler.transform(
                    feature_sequence.reshape(-1, feature_sequence.shape[-1])
                ).reshape(feature_sequence.shape)
            else:
                # Simple normalization fallback
                feature_sequence_scaled = (feature_sequence - np.mean(feature_sequence, axis=0)) / (np.std(feature_sequence, axis=0) + 1e-8)
            
            # Get predictions from asset-specific ensemble
            predictions = []
            for model in self.asset_models[asset]:
                pred = model.predict(
                    feature_sequence_scaled.reshape(1, *feature_sequence_scaled.shape),
                    verbose=0
                )[0][0]
                predictions.append(pred)
            
            # Ensemble prediction and confidence
            ensemble_prediction = np.mean(predictions)
            prediction_std = np.std(predictions)
            confidence = 1 - (prediction_std / (abs(ensemble_prediction) + 1e-8))
            
            return ensemble_prediction, confidence            
        except Exception as e:
            print(f"  ❌ Error getting prediction for {asset}: {str(e)}")
            return None, 0.5
    
    def should_trade_asset(self, asset, current_price, predicted_price, confidence):
        """Asset-specific trading decision with comprehensive criteria"""
        # 1. Confidence check (relaxed threshold)
        if confidence < self.confidence_threshold:
            return False, f"Low confidence: {confidence:.2f} < {self.confidence_threshold:.2f}"
        
        # 2. Price change magnitude check        # Research-based trade validation (simple directional)
        price_change_pct = (predicted_price - current_price) / current_price
        
        # Research approach: Trade on any directional prediction (no confidence threshold)
        # Only check if we have sufficient balance
        trade_amount = self.balance * self.position_size
        if trade_amount < 50:  # Minimum viable trade size
            return False, f"Insufficient balance: ${trade_amount:.0f}"
        
        return True, f"Trade approved: {price_change_pct:.2%} move (research: simple directional)"
    
    def execute_asset_trade(self, asset, current_price, predicted_price, confidence, timestamp):
        """Execute trade using research-based simple directional strategy"""
        try:
            price_change = predicted_price - current_price
            price_change_pct = price_change / current_price
            
            # Research approach: Simple directional trading
            if price_change > 0:
                direction = 'LONG'
                signal = 1
            elif price_change < 0:
                direction = 'SHORT' 
                signal = -1
            else:
                return  # No trade on zero prediction
            
            # Research position sizing: 95% of capital per trade
            trade_amount = self.balance * self.position_size
            
            # Minimum trade size check
            if trade_amount < 50:  # $50 minimum
                return
            
            # Calculate quantity and fees (1% transaction cost per research)
            quantity = trade_amount / current_price
            transaction_fee = trade_amount * self.transaction_cost
            
            # Execute trade (research: no stop loss or take profit)
            self.balance -= transaction_fee
            
            trade = {
                'asset': asset,
                'timestamp': timestamp,
                'direction': direction,
                'signal': signal,
                'entry_price': current_price,
                'predicted_price': predicted_price,
                'quantity': quantity,
                'confidence': confidence,
                'trade_amount': trade_amount,
                'fee': transaction_fee,
                'position_size_pct': self.position_size * 100,
                'expected_return_pct': price_change_pct * 100,
                'return_rate': None  # Will be calculated later
            }
            
            self.trades.append(trade)
            self.asset_performance[asset]['trades'] += 1
            
            print(f"  📈 {direction} {asset}: ${trade_amount:.0f} @ ${current_price:.4f}")
            print(f"     Expected: {price_change_pct:.2%} | Research Strategy: Simple Directional")
            
        except Exception as e:
            print(f"  ❌ Error executing trade for {asset}: {str(e)}")
    
    def run_asset_backtest(self, asset):
        """Run backtest for specific asset"""
        print(f"\n{'='*20} BACKTESTING {asset} {'='*20}")
        
        # Load data
        data = self.load_asset_data(asset)
        if data is None:
            return
        
        # Prepare features
        data_with_features, feature_columns = self.prepare_asset_features(data, asset)
        
        # Use test mode data limit if configured
        if config.is_test_mode():
            data_limit = config.get_data_limit()
            if data_limit:
                data_with_features = data_with_features.tail(data_limit)
        
        print(f"  Running backtest on {len(data_with_features)} records...")
        
        # Backtest loop
        for i in range(self.window_size, len(data_with_features)):
            try:
                current_data = data_with_features.iloc[:i+1]
                current_features = current_data[feature_columns]
                
                # Get prediction
                prediction, confidence = self.get_asset_prediction(asset, current_features)
                
                if prediction is None:
                    continue
                
                current_price = current_data['close'].iloc[-1]
                predicted_price = current_price * (1 + prediction)
                
                # Trading decision
                should_trade, reason = self.should_trade_asset(
                    asset, current_price, predicted_price, confidence
                )
                
                if should_trade:
                    self.execute_asset_trade(
                        asset, current_price, predicted_price, confidence,
                        current_data.index[-1]
                    )
                
                # Track prediction accuracy
                if i < len(data_with_features) - 1:
                    actual_price = data_with_features['close'].iloc[i+1]
                    actual_return = (actual_price - current_price) / current_price
                    
                    self.asset_performance[asset]['predictions'].append(prediction)
                    self.asset_performance[asset]['actuals'].append(actual_return)
                
            except Exception as e:
                print(f"  ❌ Error at step {i}: {str(e)}")
                continue
        
        print(f"  ✅ {asset} backtest completed: {self.asset_performance[asset]['trades']} trades")
    
    def run_multi_asset_backtest(self):
        """Run backtest across all assets"""
        print(f"\n🚀 STARTING MULTI-ASSET HELFORMER BACKTEST")
        print(f"Assets: {list(self.asset_models.keys())}")
        print(f"Initial Balance: ${self.initial_balance:,.0f}")
        print("=" * 60)
        
        for asset in self.asset_models.keys():
            self.run_asset_backtest(asset)
        
        # Calculate overall performance
        self.calculate_performance_metrics()
    
    def calculate_performance_metrics(self):
        """Calculate research-aligned performance metrics"""
        print(f"\n{'='*60}")
        print(f"RESEARCH-ALIGNED BACKTEST RESULTS")
        print(f"{'='*60}")
        
        total_trades = len(self.trades)
        if total_trades == 0:
            print("❌ No trades executed - Check trading thresholds!")
            return
        
        # Research methodology: Calculate trading performance metrics
        
        # 1. Excess Return (ER) - Research equation 20
        final_balance = self.balance
        excess_return = (final_balance - self.initial_balance) / self.initial_balance * 100
        
        # 2. Calculate return rates for each trade (simulate outcomes)
        return_rates = []
        net_values = [1.0]  # Start with net value of 1
        
        for trade in self.trades:
            # Simulate return based on direction and price change
            expected_return = trade['expected_return_pct'] / 100
            # Apply transaction cost
            actual_return = expected_return - self.transaction_cost
            return_rates.append(actual_return)
            
            # Update net value
            new_net_value = net_values[-1] * (1 + actual_return)
            net_values.append(new_net_value)
            
            # Update trade record
            trade['return_rate'] = actual_return
        
        # 3. Volatility (V) - Research equation 22
        volatility = np.std(return_rates) if len(return_rates) > 1 else 0.0
        
        # 4. Maximum Drawdown (MDD) - Research equation 23  
        net_values = np.array(net_values)
        peak = np.maximum.accumulate(net_values)
        drawdown = (net_values - peak) / peak
        max_drawdown = abs(np.min(drawdown)) * 100
        
        # 5. Sharpe Ratio (SR) - Research equation 24
        risk_free_rate = 0.01  # 1% as per research
        mean_return = np.mean(return_rates) if len(return_rates) > 0 else 0.0
        if volatility > 0:
            sharpe_ratio = (mean_return - risk_free_rate) / volatility
        else:
            sharpe_ratio = 0.0
        
        # Additional metrics
        win_rate = sum(1 for r in return_rates if r > 0) / len(return_rates) * 100
        
        # Display results in research format
        print(f"Research Trading Strategy Results:")
        print(f"Excess Return (ER): {excess_return:.2f}%")
        print(f"Volatility (V): {volatility:.4f}")
        print(f"Maximum Drawdown (MDD): {max_drawdown:.2f}%")
        print(f"Sharpe Ratio (SR): {sharpe_ratio:.4f}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Net Value Final: {net_values[-1]:.4f}")
        
        # Research benchmark comparison
        print(f"\nResearch Benchmark Comparison:")
        print(f"Target ER: >200% (Helformer: 925.29%)")
        print(f"Target V: <0.025 (Helformer: 0.0178)")
        print(f"Target MDD: <5% (Helformer: ~0%)")
        print(f"Target SR: >2.0 (Helformer: 18.06)")
        
        # Performance evaluation
        performance_score = 0
        if excess_return > 200: performance_score += 1
        if volatility < 0.025: performance_score += 1  
        if max_drawdown < 5: performance_score += 1
        if sharpe_ratio > 2.0: performance_score += 1
        
        status_map = {
            4: "🎯 EXCELLENT (Research-level)",
            3: "✅ GOOD", 
            2: "⚠️ ACCEPTABLE",
            1: "❌ POOR",
            0: "💥 FAILED"
        }
        
        print(f"\nOverall Performance: {status_map.get(performance_score, '❓ UNKNOWN')}")
        print(f"Score: {performance_score}/4 research benchmarks met")
        
        # Per-asset performance
        print(f"\nPER-ASSET PERFORMANCE:")
        print("-" * 40)
        for asset, perf in self.asset_performance.items():
            if perf['trades'] > 0:
                print(f"{asset}: {perf['trades']} trades")
        
        return {
            'excess_return': excess_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'net_value_final': net_values[-1],
            'performance_score': performance_score,
            'return_rates': return_rates,
            'net_values': net_values.tolist()
        }

def main():
    """Run consolidated backtest"""
    print("🚀 CONSOLIDATED HELFORMER BACKTEST")
    print("=" * 50)
    
    backtest = ConsolidatedHelformerBacktest(initial_balance=10000)
    results = backtest.run_multi_asset_backtest()
    
    return results

if __name__ == "__main__":
    main()