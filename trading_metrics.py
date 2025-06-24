"""
Trading Performance Metrics for Helformer Training
Calculate win rate, annual return, Sharpe ratio, and profit factor from predictions
"""

import numpy as np
import pandas as pd
from config_helformer import config

def calculate_trading_metrics(y_true, y_pred, prices=None, timeframe_hours=0.25):
    """
    Calculate trading performance metrics from price predictions
    
    Args:
        y_true: Actual normalized price changes
        y_pred: Predicted normalized price changes  
        prices: Optional actual prices for more accurate calculations
        timeframe_hours: Hours between predictions (0.25 = 15min)
    
    Returns:
        Dict with trading metrics: win_rate, annual_return, sharpe_ratio, profit_factor
    """
    
    # Convert normalized predictions to trading signals
    signals = generate_trading_signals(y_pred)
    
    # Calculate returns based on signals and actual price movements
    if prices is not None:
        returns = calculate_returns_from_prices(signals, prices)
    else:        # Use normalized changes as proxy for returns (NO CAPS - let actual performance show)
        returns = signals * y_true
    
    # Filter out zero returns (no trades)
    active_returns = returns[returns != 0]
    
    if len(active_returns) == 0:
        return {
            'win_rate': 0.0,
            'annual_return': 0.0, 
            'sharpe_ratio': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'max_drawdown': 0.0
        }
    
    # 1. Win Rate
    winning_trades = np.sum(active_returns > 0)
    total_trades = len(active_returns)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0    # 2. Annual Return (FIXED - realistic calculation)
    if len(active_returns) > 0:
        # Calculate cumulative return
        cumulative_return = np.prod(1 + active_returns) - 1
        
        # Annualize based on sample period
        periods_per_year = (365 * 24) / timeframe_hours
        sample_periods = len(returns)
        sample_years = sample_periods / periods_per_year if periods_per_year > 0 else 1
        
        # Compound annual growth rate (CAGR)
        if sample_years > 0 and cumulative_return > -1:
            annual_return = ((1 + cumulative_return) ** (1/sample_years) - 1) * 100
        else:
            annual_return = 0.0
    else:
        annual_return = 0.0# 3. Sharpe Ratio
    if len(active_returns) > 1:
        mean_return = np.mean(active_returns)
        return_std = np.std(active_returns)
        if return_std > 0:
            # Annualize Sharpe ratio
            periods_per_year_sqrt = np.sqrt(periods_per_year)
            sharpe_ratio = (mean_return / return_std) * periods_per_year_sqrt
        else:
            sharpe_ratio = 0
    else:
        sharpe_ratio = 0
    
    # 4. Profit Factor
    winning_returns = active_returns[active_returns > 0]
    losing_returns = active_returns[active_returns < 0]
    
    gross_profit = np.sum(winning_returns) if len(winning_returns) > 0 else 0
    gross_loss = abs(np.sum(losing_returns)) if len(losing_returns) > 0 else 0
    
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf    # 5. Maximum Drawdown (FIXED - more realistic)
    if len(active_returns) > 1:
        # Calculate cumulative returns properly
        cumulative_returns = np.cumprod(1 + active_returns)
        # Find running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        # Calculate drawdown as percentage from peak
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(np.min(drawdown)) * 100
        
        # Cap maximum drawdown at reasonable level (100%)
        max_drawdown = min(max_drawdown, 100.0)
    else:
        max_drawdown = 0.0
    return {
        'win_rate': win_rate * 100,  # Convert to percentage
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio, 
        'profit_factor': profit_factor,
        'total_trades': total_trades,
        'max_drawdown': max_drawdown
    }

def generate_trading_signals(predictions, confidence_threshold=None, method='adaptive'):
    """
    Convert price predictions to trading signals with stable threshold calculation
    
    Args:
        predictions: Array of normalized price change predictions
        confidence_threshold: Minimum prediction confidence to trade
        method: 'adaptive', 'percentile', or 'fixed'
    
    Returns:
        Array of trading signals: 1 (long), -1 (short), 0 (no trade)
    """
    predictions = np.array(predictions).flatten()
    
    if confidence_threshold is None:
        if method == 'adaptive':
            # More conservative adaptive threshold (was causing too many trades)
            pred_std = np.std(predictions)
            confidence_threshold = max(0.005, pred_std * 0.5)  # Back to 0.5 std dev (was 0.67)
        
        elif method == 'percentile':
            # Use percentile-based threshold to control trade frequency
            abs_predictions = np.abs(predictions)
            confidence_threshold = np.percentile(abs_predictions, 75)  # Top 25% of predictions
            
        else:  # 'fixed'
            confidence_threshold = config.MIN_PREDICTION_CONFIDENCE
    
    signals = np.zeros_like(predictions)
    
    # Only trade if prediction confidence exceeds threshold
    strong_up = predictions > confidence_threshold
    strong_down = predictions < -confidence_threshold
    
    signals[strong_up] = 1   # Long position
    signals[strong_down] = -1  # Short position
    # signals remain 0 for weak predictions (no trade)
    
    return signals

def calculate_returns_from_prices(signals, prices):
    """
    Calculate actual trading returns from signals and price data
    
    Args:
        signals: Trading signals (1, -1, 0)
        prices: Actual price data
    
    Returns:
        Array of trade returns
    """
    returns = np.zeros_like(signals, dtype=float)
    
    # Calculate price changes
    price_changes = np.diff(prices) / prices[:-1]
    
    # Align arrays (signals are 1 element shorter after price diff)
    if len(price_changes) == len(signals) - 1:
        signals = signals[1:]  # Remove first signal
    elif len(price_changes) == len(signals):
        pass  # Arrays already aligned
    else:
        # Truncate to minimum length
        min_len = min(len(price_changes), len(signals))
        price_changes = price_changes[:min_len]
        signals = signals[:min_len]
    
    # Calculate returns: signal * price_change - transaction_costs
    transaction_cost = config.TRANSACTION_COST
    
    for i in range(len(signals)):
        if signals[i] != 0:  # Only when we have a position
            # Return = signal * price_change - transaction_costs
            trade_return = signals[i] * price_changes[i] - transaction_cost
            returns[i] = trade_return
    
    return returns

def evaluate_model_trading_performance(y_true, y_pred, prices=None, asset_name=""):
    """
    Evaluate model performance using both prediction accuracy and trading metrics
    
    Args:
        y_true: Actual normalized price changes
        y_pred: Predicted normalized price changes
        prices: Optional actual prices
        asset_name: Name of the asset for logging
    
    Returns:
        Combined metrics dictionary
    """
    from improved_training_utils import evaluate_helformer
    
    # Get prediction accuracy metrics
    accuracy_metrics = evaluate_helformer(y_true, y_pred)
    
    # Get trading performance metrics
    trading_metrics = calculate_trading_metrics(y_true, y_pred, prices)
    
    # Combine all metrics
    combined_metrics = {**accuracy_metrics, **trading_metrics}
    
    # Print summary
    print(f"\n{asset_name} MODEL PERFORMANCE:")
    print(f"  Prediction Accuracy:")
    print(f"    R²: {accuracy_metrics['R2']:.4f}")
    print(f"    MAPE: {accuracy_metrics['MAPE']:.2f}%")
    print(f"  Trading Performance:")
    print(f"    Win Rate: {trading_metrics['win_rate']:.1f}%")
    print(f"    Annual Return: {trading_metrics['annual_return']:.1f}%")
    print(f"    Sharpe Ratio: {trading_metrics['sharpe_ratio']:.2f}")
    print(f"    Profit Factor: {trading_metrics['profit_factor']:.2f}")
    print(f"    Max Drawdown: {trading_metrics['max_drawdown']:.1f}%")
    print(f"    Total Trades: {trading_metrics['total_trades']}")
    
    # Performance warnings
    if trading_metrics['win_rate'] < config.TARGET_WIN_RATE:
        print(f"    ⚠️  Win rate below target ({config.TARGET_WIN_RATE}%)")
    
    if trading_metrics['annual_return'] < config.TARGET_ANNUAL_RETURN:
        print(f"    ⚠️  Annual return below target ({config.TARGET_ANNUAL_RETURN}%)")
    
    if trading_metrics['sharpe_ratio'] < config.TARGET_SHARPE_RATIO:
        print(f"    ⚠️  Sharpe ratio below target ({config.TARGET_SHARPE_RATIO})")
    
    if trading_metrics['max_drawdown'] > config.TARGET_MAX_DRAWDOWN:
        print(f"    ⚠️  Max drawdown above limit ({config.TARGET_MAX_DRAWDOWN}%)")
    
    return combined_metrics

def create_trading_callback(validation_data, asset_name="Model"):
    """
    Create a Keras callback that calculates trading metrics during training
    
    Args:
        validation_data: Tuple of (X_val, y_val) for validation
        asset_name: Name for logging
    
    Returns:
        Keras callback that tracks trading performance
    """
    import tensorflow as tf    
    class TradingMetricsCallback(tf.keras.callbacks.Callback):
        def __init__(self, val_data, asset_name):
            super().__init__()
            self.X_val, self.y_val = val_data
            self.asset_name = asset_name
            self.trading_history = []
        
        def on_epoch_end(self, epoch, logs=None):
            # Get predictions for validation set
            y_pred = self.model.predict(self.X_val, verbose=0).flatten()
            
            # Calculate trading metrics with debugging
            trading_metrics = calculate_trading_metrics(self.y_val, y_pred)
              # Store in history
            epoch_metrics = {
                'epoch': epoch + 1,
                'val_loss': logs.get('val_loss', 0),
                'val_r2': logs.get('val_r2', 0),
                **trading_metrics
            }
            self.trading_history.append(epoch_metrics)
              
            # DISABLED: Epoch trading metrics were unreliable/misleading
            # Only print basic health check every 10 epochs
            if (epoch + 1) % 10 == 0:
                pred_std = np.std(y_pred)
                pred_range = f"[{np.min(y_pred):.6f}, {np.max(y_pred):.6f}]"
                print(f"    Prediction Health Check - Std: {pred_std:.6f}, Range: {np.max(y_pred) - np.min(y_pred):.6f}")
                # NOTE: Full trading metrics calculated only at final evaluation
    
    return TradingMetricsCallback(validation_data, asset_name)