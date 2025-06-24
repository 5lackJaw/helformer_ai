"""
Performance Analytics and Model Monitoring System
Provides comprehensive performance tracking, attribution analysis, and model drift detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
import json
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config_helformer import config

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: datetime
    period: str  # 'daily', 'weekly', 'monthly'
    
    # Return metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall: float = 0.0
    
    # Trading metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade_duration: float = 0.0
    
    # Model metrics
    prediction_accuracy: float = 0.0
    directional_accuracy: float = 0.0
    model_confidence: float = 0.0
    
    # Attribution
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0

@dataclass
class TradeAnalysis:
    """Individual trade analysis."""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    side: str  # 'buy' or 'sell'
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    fees: float
    hold_time_hours: float
    
    # Attribution
    predicted_return: float = 0.0
    actual_return: float = 0.0
    prediction_error: float = 0.0
    market_regime: str = ""
    confidence_score: float = 0.0

class PerformanceAnalyzer:
    """
    Comprehensive performance analytics system.
    """
    
    def __init__(self):
        """Initialize performance analyzer."""
        self.trades: List[TradeAnalysis] = []
        self.daily_returns: pd.DataFrame = pd.DataFrame()
        self.performance_history: List[PerformanceMetrics] = []
        self.benchmark_returns: pd.DataFrame = pd.DataFrame()
        
        # Model monitoring
        self.model_predictions: List[Dict] = []
        self.feature_importance_history: List[Dict] = []
        self.model_drift_alerts: List[Dict] = []
        
        logger.info("Performance Analyzer initialized")
    
    def add_trade(self, trade: TradeAnalysis):
        """Add a trade for analysis."""
        self.trades.append(trade)
        logger.debug(f"Added trade: {trade.trade_id}")
    
    def add_prediction(self, symbol: str, timestamp: datetime, 
                      predicted_value: float, actual_value: Optional[float] = None,
                      confidence: float = 0.0, features: Optional[Dict] = None):
        """Add model prediction for monitoring."""
        prediction = {
            'symbol': symbol,
            'timestamp': timestamp,
            'predicted_value': predicted_value,
            'actual_value': actual_value,
            'confidence': confidence,
            'features': features or {}
        }
        self.model_predictions.append(prediction)
    
    def calculate_performance_metrics(self, start_date: Optional[datetime] = None,
                                    end_date: Optional[datetime] = None) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            PerformanceMetrics object
        """
        # Filter trades by date range
        filtered_trades = self._filter_trades_by_date(start_date, end_date)
        
        if not filtered_trades:
            return PerformanceMetrics(
                timestamp=datetime.now(),
                period=f"{start_date} to {end_date}" if start_date and end_date else "all"
            )
        
        # Calculate returns
        returns = pd.Series([trade.pnl_pct for trade in filtered_trades])
        trade_dates = pd.Series([trade.exit_time for trade in filtered_trades])
        
        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        annual_return = self._annualize_return(total_return, len(returns))
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Risk-adjusted metrics
        risk_free_rate = 0.02  # 2% annual risk-free rate
        excess_returns = returns - (risk_free_rate / 252)
        
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Downside metrics
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 0 else 0
        sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Drawdown metrics
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # VaR and Expected Shortfall
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        expected_shortfall = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        
        # Trading metrics
        winning_trades = [t for t in filtered_trades if t.pnl > 0]
        losing_trades = [t for t in filtered_trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(filtered_trades) if filtered_trades else 0
        avg_win = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0
        
        profit_factor = (sum(t.pnl for t in winning_trades) / 
                        abs(sum(t.pnl for t in losing_trades))) if losing_trades else float('inf')
        
        avg_trade_duration = np.mean([t.hold_time_hours for t in filtered_trades])
        
        # Model performance metrics
        prediction_accuracy = self._calculate_prediction_accuracy()
        directional_accuracy = self._calculate_directional_accuracy()
        model_confidence = np.mean([p['confidence'] for p in self.model_predictions]) if self.model_predictions else 0
        
        # Create performance metrics object
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            period=f"{start_date} to {end_date}" if start_date and end_date else "all",
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade_duration=avg_trade_duration,
            prediction_accuracy=prediction_accuracy,
            directional_accuracy=directional_accuracy,
            model_confidence=model_confidence
        )
        
        return metrics
    
    def generate_performance_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive performance report.
        
        Args:
            save_path: Path to save the report
            
        Returns:
            Report as string
        """
        # Calculate metrics for different periods
        now = datetime.now()
        periods = {
            'Last 7 Days': now - timedelta(days=7),
            'Last 30 Days': now - timedelta(days=30),
            'Last 90 Days': now - timedelta(days=90),
            'Year to Date': datetime(now.year, 1, 1),
            'All Time': None
        }
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("HELFORMER PERFORMANCE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        for period_name, start_date in periods.items():
            metrics = self.calculate_performance_metrics(start_date, now)
            
            report_lines.append(f"\n{period_name.upper()}")
            report_lines.append("-" * 40)
            report_lines.append(f"Total Return:         {metrics.total_return:>10.2%}")
            report_lines.append(f"Annual Return:        {metrics.annual_return:>10.2%}")
            report_lines.append(f"Volatility:           {metrics.volatility:>10.2%}")
            report_lines.append(f"Sharpe Ratio:         {metrics.sharpe_ratio:>10.2f}")
            report_lines.append(f"Sortino Ratio:        {metrics.sortino_ratio:>10.2f}")
            report_lines.append(f"Max Drawdown:         {metrics.max_drawdown:>10.2%}")
            report_lines.append(f"Win Rate:             {metrics.win_rate:>10.2%}")
            report_lines.append(f"Profit Factor:        {metrics.profit_factor:>10.2f}")
            report_lines.append(f"Prediction Accuracy:  {metrics.prediction_accuracy:>10.2%}")
            report_lines.append(f"Directional Accuracy: {metrics.directional_accuracy:>10.2%}")
        
        # Add trade summary
        report_lines.append(f"\n\nTRADE SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Trades:         {len(self.trades):>10}")
        
        if self.trades:
            symbols = set(trade.symbol for trade in self.trades)
            report_lines.append(f"Symbols Traded:       {len(symbols):>10}")
            
            # Best and worst trades
            best_trade = max(self.trades, key=lambda t: t.pnl_pct)
            worst_trade = min(self.trades, key=lambda t: t.pnl_pct)
            
            report_lines.append(f"Best Trade:           {best_trade.pnl_pct:>10.2%} ({best_trade.symbol})")
            report_lines.append(f"Worst Trade:          {worst_trade.pnl_pct:>10.2%} ({worst_trade.symbol})")
        
        # Model performance section
        report_lines.append(f"\n\nMODEL PERFORMANCE")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Predictions:    {len(self.model_predictions):>10}")
        
        if self.model_predictions:
            recent_predictions = [p for p in self.model_predictions 
                                if p['timestamp'] > now - timedelta(days=7)]
            report_lines.append(f"Recent Predictions:   {len(recent_predictions):>10}")
            
            avg_confidence = np.mean([p['confidence'] for p in self.model_predictions])
            report_lines.append(f"Avg Confidence:       {avg_confidence:>10.2%}")
        
        # Risk metrics
        latest_metrics = self.calculate_performance_metrics()
        report_lines.append(f"\n\nRISK METRICS")
        report_lines.append("-" * 40)
        report_lines.append(f"VaR (95%):            {latest_metrics.var_95:>10.2%}")
        report_lines.append(f"VaR (99%):            {latest_metrics.var_99:>10.2%}")
        report_lines.append(f"Expected Shortfall:   {latest_metrics.expected_shortfall:>10.2%}")
        
        report_text = "\n".join(report_lines)
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Performance report saved to {save_path}")
        
        return report_text
    
    def plot_performance_charts(self, save_dir: Optional[str] = None):
        """
        Generate performance visualization charts.
        
        Args:
            save_dir: Directory to save charts
        """
        if not self.trades:
            logger.warning("No trades available for plotting")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Helformer Performance Analysis', fontsize=16)
        
        # 1. Cumulative Returns
        trade_returns = pd.Series([t.pnl_pct for t in self.trades])
        cumulative_returns = (1 + trade_returns).cumprod()
        
        axes[0, 0].plot(cumulative_returns.values)
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].grid(True)
        
        # 2. Drawdown
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        
        axes[0, 1].fill_between(range(len(drawdowns)), drawdowns.values, 0, alpha=0.7, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown %')
        axes[0, 1].grid(True)
        
        # 3. Returns Distribution
        axes[0, 2].hist(trade_returns.values, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Returns Distribution')
        axes[0, 2].set_xlabel('Return %')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True)
        
        # 4. Monthly Returns Heatmap
        if len(self.trades) > 12:
            trade_dates = pd.to_datetime([t.exit_time for t in self.trades])
            monthly_data = pd.DataFrame({
                'date': trade_dates,
                'return': trade_returns.values
            })
            monthly_data['year'] = monthly_data['date'].dt.year
            monthly_data['month'] = monthly_data['date'].dt.month
            
            monthly_returns = monthly_data.groupby(['year', 'month'])['return'].sum().unstack(fill_value=0)
            
            if not monthly_returns.empty:
                sns.heatmap(monthly_returns, annot=True, fmt='.2%', cmap='RdYlGn', 
                           center=0, ax=axes[1, 0])
                axes[1, 0].set_title('Monthly Returns Heatmap')
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient data\nfor monthly heatmap', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # 5. Win/Loss Analysis
        winning_trades = [t.pnl_pct for t in self.trades if t.pnl > 0]
        losing_trades = [t.pnl_pct for t in self.trades if t.pnl < 0]
        
        axes[1, 1].bar(['Wins', 'Losses'], [len(winning_trades), len(losing_trades)], 
                      color=['green', 'red'], alpha=0.7)
        axes[1, 1].set_title('Win/Loss Count')
        axes[1, 1].set_ylabel('Number of Trades')
        
        # 6. Prediction vs Actual Scatter
        if self.model_predictions:
            predictions_with_actual = [p for p in self.model_predictions if p['actual_value'] is not None]
            
            if predictions_with_actual:
                predicted = [p['predicted_value'] for p in predictions_with_actual]
                actual = [p['actual_value'] for p in predictions_with_actual]
                
                axes[1, 2].scatter(predicted, actual, alpha=0.6)
                
                # Add perfect prediction line
                min_val = min(min(predicted), min(actual))
                max_val = max(max(predicted), max(actual))
                axes[1, 2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
                axes[1, 2].set_xlabel('Predicted')
                axes[1, 2].set_ylabel('Actual')
                axes[1, 2].set_title('Prediction vs Actual')
                axes[1, 2].grid(True)
            else:
                axes[1, 2].text(0.5, 0.5, 'No actual values\navailable for predictions', 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
        else:
            axes[1, 2].text(0.5, 0.5, 'No predictions\navailable', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
        
        plt.tight_layout()
        
        # Save if directory provided
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/performance_analysis.png", dpi=300, bbox_inches='tight')
            logger.info(f"Performance charts saved to {save_dir}")
        
        plt.show()
    
    def detect_model_drift(self, window_size: int = 100, threshold: float = 0.1) -> Dict:
        """
        Detect model performance drift.
        
        Args:
            window_size: Size of rolling window for analysis
            threshold: Threshold for drift detection
            
        Returns:
            Drift analysis results
        """
        if len(self.model_predictions) < window_size * 2:
            return {'drift_detected': False, 'reason': 'Insufficient data'}
        
        # Get recent predictions with actual values
        predictions_with_actual = [
            p for p in self.model_predictions 
            if p['actual_value'] is not None
        ]
        
        if len(predictions_with_actual) < window_size * 2:
            return {'drift_detected': False, 'reason': 'Insufficient actual values'}
        
        # Calculate rolling accuracy
        errors = []
        for pred in predictions_with_actual:
            error = abs(pred['predicted_value'] - pred['actual_value'])
            errors.append(error)
        
        errors_series = pd.Series(errors)
        rolling_mean_error = errors_series.rolling(window=window_size).mean()
        
        # Compare recent performance to historical
        recent_error = rolling_mean_error.iloc[-window_size:].mean()
        historical_error = rolling_mean_error.iloc[:-window_size].mean()
        
        drift_ratio = recent_error / historical_error if historical_error > 0 else 1.0
        drift_detected = drift_ratio > (1 + threshold)
        
        drift_analysis = {
            'drift_detected': drift_detected,
            'drift_ratio': drift_ratio,
            'recent_error': recent_error,
            'historical_error': historical_error,
            'threshold': threshold,
            'window_size': window_size,
            'timestamp': datetime.now()
        }
        
        if drift_detected:
            self.model_drift_alerts.append(drift_analysis)
            logger.warning(f"Model drift detected! Ratio: {drift_ratio:.3f}")
        
        return drift_analysis
    
    def _filter_trades_by_date(self, start_date: Optional[datetime], 
                              end_date: Optional[datetime]) -> List[TradeAnalysis]:
        """Filter trades by date range."""
        filtered_trades = self.trades
        
        if start_date:
            filtered_trades = [t for t in filtered_trades if t.exit_time >= start_date]
        
        if end_date:
            filtered_trades = [t for t in filtered_trades if t.exit_time <= end_date]
        
        return filtered_trades
    
    def _annualize_return(self, total_return: float, num_periods: int) -> float:
        """Annualize return based on number of periods."""
        if num_periods <= 0:
            return 0.0
        
        # Assume daily trading frequency
        periods_per_year = 252
        years = num_periods / periods_per_year
        
        if years <= 0:
            return 0.0
        
        return (1 + total_return) ** (1 / years) - 1
    
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate model prediction accuracy."""
        predictions_with_actual = [
            p for p in self.model_predictions 
            if p['actual_value'] is not None
        ]
        
        if not predictions_with_actual:
            return 0.0
        
        errors = []
        for pred in predictions_with_actual:
            error = abs(pred['predicted_value'] - pred['actual_value'])
            errors.append(error)
        
        mean_error = np.mean(errors)
        # Convert to accuracy percentage (lower error = higher accuracy)
        accuracy = max(0.0, 1.0 - mean_error)
        
        return accuracy
    
    def _calculate_directional_accuracy(self) -> float:
        """Calculate directional prediction accuracy."""
        predictions_with_actual = [
            p for p in self.model_predictions 
            if p['actual_value'] is not None
        ]
        
        if not predictions_with_actual:
            return 0.0
        
        correct_directions = 0
        for pred in predictions_with_actual:
            predicted_direction = np.sign(pred['predicted_value'])
            actual_direction = np.sign(pred['actual_value'])
            
            if predicted_direction == actual_direction:
                correct_directions += 1
        
        return correct_directions / len(predictions_with_actual)
    
    def export_data(self, filepath: str):
        """Export all data to file."""
        data = {
            'trades': [
                {
                    'trade_id': t.trade_id,
                    'symbol': t.symbol,
                    'entry_time': t.entry_time.isoformat(),
                    'exit_time': t.exit_time.isoformat(),
                    'side': t.side,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'quantity': t.quantity,
                    'pnl': t.pnl,
                    'pnl_pct': t.pnl_pct,
                    'fees': t.fees,
                    'hold_time_hours': t.hold_time_hours
                }
                for t in self.trades
            ],
            'model_predictions': [
                {
                    'symbol': p['symbol'],
                    'timestamp': p['timestamp'].isoformat(),
                    'predicted_value': p['predicted_value'],
                    'actual_value': p['actual_value'],
                    'confidence': p['confidence']
                }
                for p in self.model_predictions
            ],
            'drift_alerts': [
                {
                    'timestamp': alert['timestamp'].isoformat(),
                    'drift_ratio': alert['drift_ratio'],
                    'recent_error': alert['recent_error'],
                    'historical_error': alert['historical_error']
                }
                for alert in self.model_drift_alerts
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Performance data exported to {filepath}")

def create_performance_analyzer() -> PerformanceAnalyzer:
    """Factory function to create a performance analyzer."""
    return PerformanceAnalyzer()

if __name__ == "__main__":
    # Test the performance analyzer
    analyzer = PerformanceAnalyzer()
    
    # Add some sample trades
    from datetime import datetime, timedelta
    
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(50):
        trade = TradeAnalysis(
            trade_id=f"trade_{i}",
            symbol="BTC/USDT",
            entry_time=base_time + timedelta(hours=i*12),
            exit_time=base_time + timedelta(hours=i*12 + 6),
            side="buy" if i % 2 == 0 else "sell",
            entry_price=50000 + np.random.normal(0, 1000),
            exit_price=50000 + np.random.normal(100, 1000),
            quantity=0.01,
            pnl=np.random.normal(50, 200),
            pnl_pct=np.random.normal(0.01, 0.05),
            fees=10,
            hold_time_hours=6
        )
        analyzer.add_trade(trade)
    
    # Add some predictions
    for i in range(100):
        analyzer.add_prediction(
            symbol="BTC/USDT",
            timestamp=base_time + timedelta(hours=i*6),
            predicted_value=np.random.normal(0, 0.02),
            actual_value=np.random.normal(0, 0.02),
            confidence=np.random.uniform(0.6, 0.9)
        )
    
    # Generate report
    report = analyzer.generate_performance_report()
    print(report)
    
    # Check for drift
    drift_analysis = analyzer.detect_model_drift()
    print(f"\nDrift Analysis: {drift_analysis}")
    
    # Generate charts (will show if running interactively)
    try:
        analyzer.plot_performance_charts()
    except Exception as e:
        print(f"Could not generate charts: {e}")