"""
Model Status and Health Monitoring for Helformer System
Tracks model performance, prediction accuracy, and system health
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Container for model performance metrics"""
    model_name: str
    asset: str
    prediction_count: int
    accuracy_score: float
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    r2_score: float
    correlation: float
    last_updated: datetime
    confidence_distribution: Dict[str, float]
    prediction_distribution: Dict[str, float]

@dataclass
class SystemHealth:
    """Container for overall system health metrics"""
    timestamp: datetime
    total_predictions: int
    average_accuracy: float
    models_healthy: int
    models_total: int
    memory_usage_mb: float
    prediction_latency_ms: float
    error_rate: float
    uptime_hours: float

class ModelStatusMonitor:
    """
    Comprehensive monitoring system for Helformer models and predictions.
    
    Tracks:
    - Individual model performance metrics
    - Prediction accuracy over time
    - System health and resource usage
    - Alert conditions and anomalies
    """
    
    def __init__(self, status_file: str = "model_status.json"):
        """
        Initialize model status monitor.
        
        Args:
            status_file: JSON file to persist status data
        """
        self.status_file = status_file
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.system_health_history: List[SystemHealth] = []
        self.prediction_history: List[Dict] = []
        self.start_time = datetime.now()
        self.error_count = 0
        self.total_predictions = 0
        
        # Load existing status if available
        self.load_status()
        
    def track_prediction(self, 
                        model_name: str,
                        asset: str,
                        prediction: float,
                        confidence: float,
                        actual: Optional[float] = None,
                        execution_time_ms: float = 0.0):
        """
        Track a model prediction and update metrics.
        
        Args:
            model_name: Name/ID of the model
            asset: Asset being predicted
            prediction: Model prediction value
            confidence: Prediction confidence score
            actual: Actual value (for accuracy calculation)
            execution_time_ms: Prediction execution time in milliseconds
        """
        try:
            timestamp = datetime.now()
            
            # Record prediction
            prediction_record = {
                'timestamp': timestamp,
                'model_name': model_name,
                'asset': asset,
                'prediction': prediction,
                'confidence': confidence,
                'actual': actual,
                'execution_time_ms': execution_time_ms,
                'error': abs(prediction - actual) if actual is not None else None
            }
            
            self.prediction_history.append(prediction_record)
            self.total_predictions += 1
            
            # Keep only last 1000 predictions
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            # Update model metrics
            self._update_model_metrics(model_name, asset, prediction_record)
            
            # Log prediction
            if actual is not None:
                error = abs(prediction - actual)
                logger.info(f"Model {model_name} ({asset}): pred={prediction:.4f}, actual={actual:.4f}, error={error:.4f}, conf={confidence:.2f}")
            else:
                logger.info(f"Model {model_name} ({asset}): pred={prediction:.4f}, conf={confidence:.2f}")
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error tracking prediction: {str(e)}")
    
    def _update_model_metrics(self, model_name: str, asset: str, prediction_record: Dict):
        """Update metrics for specific model"""
        
        model_key = f"{model_name}_{asset}"
        
        # Get recent predictions for this model
        model_predictions = [
            p for p in self.prediction_history 
            if p['model_name'] == model_name and p['asset'] == asset
            and p['actual'] is not None
        ]
        
        if len(model_predictions) == 0:
            return
        
        # Calculate metrics
        predictions = np.array([p['prediction'] for p in model_predictions])
        actuals = np.array([p['actual'] for p in model_predictions])
        confidences = np.array([p['confidence'] for p in model_predictions])
        
        mae = np.mean(np.abs(predictions - actuals))
        mse = np.mean((predictions - actuals) ** 2)
        
        # R² score
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Correlation
        correlation = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
        
        # Accuracy score (percentage within 5% of actual)
        accuracy_threshold = 0.05
        accurate_predictions = np.abs((predictions - actuals) / actuals) <= accuracy_threshold
        accuracy_score = np.mean(accurate_predictions) * 100
        
        # Confidence distribution
        conf_high = np.mean(confidences > 0.8) * 100
        conf_medium = np.mean((confidences > 0.6) & (confidences <= 0.8)) * 100
        conf_low = np.mean(confidences <= 0.6) * 100
        
        # Prediction distribution
        pred_positive = np.mean(predictions > 0) * 100
        pred_negative = np.mean(predictions < 0) * 100
        pred_neutral = np.mean(np.abs(predictions) < 0.01) * 100
        
        # Update model metrics
        self.model_metrics[model_key] = ModelMetrics(
            model_name=model_name,
            asset=asset,
            prediction_count=len(model_predictions),
            accuracy_score=accuracy_score,
            mae=mae,
            mse=mse,
            r2_score=r2,
            correlation=correlation,
            last_updated=datetime.now(),
            confidence_distribution={
                'high': conf_high,
                'medium': conf_medium,
                'low': conf_low
            },
            prediction_distribution={
                'positive': pred_positive,
                'negative': pred_negative,
                'neutral': pred_neutral
            }
        )
    
    def get_model_status(self, model_name: str = None, asset: str = None) -> Dict:
        """
        Get status for specific model or all models.
        
        Args:
            model_name: Specific model name (optional)
            asset: Specific asset (optional)
            
        Returns:
            Dictionary with model status information
        """
        
        if model_name and asset:
            model_key = f"{model_name}_{asset}"
            if model_key in self.model_metrics:
                return asdict(self.model_metrics[model_key])
            else:
                return {'error': f'No data for model {model_name} on {asset}'}
        
        # Return all models
        status = {}
        for model_key, metrics in self.model_metrics.items():
            status[model_key] = asdict(metrics)
        
        return status
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health metrics"""
        
        try:
            import psutil
            memory_usage = psutil.virtual_memory().used / 1024 / 1024  # MB
        except:
            memory_usage = 0.0
        
        # Calculate average metrics
        recent_predictions = [
            p for p in self.prediction_history 
            if p['timestamp'] > datetime.now() - timedelta(hours=1)
            and p['actual'] is not None
        ]
        
        if recent_predictions:
            avg_accuracy = np.mean([
                1.0 if abs(p['prediction'] - p['actual']) / abs(p['actual']) < 0.05 else 0.0
                for p in recent_predictions
            ]) * 100
            
            avg_latency = np.mean([p['execution_time_ms'] for p in recent_predictions])
        else:
            avg_accuracy = 0.0
            avg_latency = 0.0
        
        # Count healthy models (accuracy > 50%)
        healthy_models = sum(
            1 for metrics in self.model_metrics.values() 
            if metrics.accuracy_score > 50.0
        )
        
        uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        error_rate = (self.error_count / max(self.total_predictions, 1)) * 100
        
        health = SystemHealth(
            timestamp=datetime.now(),
            total_predictions=self.total_predictions,
            average_accuracy=avg_accuracy,
            models_healthy=healthy_models,
            models_total=len(self.model_metrics),
            memory_usage_mb=memory_usage,
            prediction_latency_ms=avg_latency,
            error_rate=error_rate,
            uptime_hours=uptime_hours
        )
        
        # Store in history
        self.system_health_history.append(health)
        if len(self.system_health_history) > 100:  # Keep last 100 health checks
            self.system_health_history = self.system_health_history[-100:]
        
        return health
    
    def check_alert_conditions(self) -> List[Dict]:
        """Check for alert conditions that require attention"""
        
        alerts = []
        
        # Check model performance
        for model_key, metrics in self.model_metrics.items():
            if metrics.accuracy_score < 40.0:  # Low accuracy alert
                alerts.append({
                    'type': 'low_accuracy',
                    'severity': 'warning',
                    'message': f'Model {model_key} accuracy dropped to {metrics.accuracy_score:.1f}%',
                    'timestamp': datetime.now()
                })
            
            if metrics.prediction_count < 10:  # Insufficient data alert
                alerts.append({
                    'type': 'insufficient_data',
                    'severity': 'info',
                    'message': f'Model {model_key} has only {metrics.prediction_count} predictions',
                    'timestamp': datetime.now()
                })
        
        # Check system health
        health = self.get_system_health()
        
        if health.error_rate > 10.0:  # High error rate
            alerts.append({
                'type': 'high_error_rate',
                'severity': 'critical',
                'message': f'System error rate is {health.error_rate:.1f}%',
                'timestamp': datetime.now()
            })
        
        if health.models_healthy < health.models_total * 0.7:  # Many unhealthy models
            alerts.append({
                'type': 'models_unhealthy',
                'severity': 'warning',
                'message': f'Only {health.models_healthy}/{health.models_total} models are healthy',
                'timestamp': datetime.now()
            })
        
        return alerts
    
    def generate_status_report(self) -> str:
        """Generate comprehensive status report"""
        
        health = self.get_system_health()
        alerts = self.check_alert_conditions()
        
        report = []
        report.append("="*60)
        report.append("HELFORMER SYSTEM STATUS REPORT")
        report.append("="*60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # System Health
        report.append("SYSTEM HEALTH:")
        report.append(f"  Uptime: {health.uptime_hours:.1f} hours")
        report.append(f"  Total Predictions: {health.total_predictions}")
        report.append(f"  Average Accuracy: {health.average_accuracy:.1f}%")
        report.append(f"  Error Rate: {health.error_rate:.2f}%")
        report.append(f"  Memory Usage: {health.memory_usage_mb:.0f} MB")
        report.append(f"  Avg Latency: {health.prediction_latency_ms:.1f} ms")
        report.append("")
        
        # Model Performance
        report.append("MODEL PERFORMANCE:")
        for model_key, metrics in self.model_metrics.items():
            report.append(f"  {model_key}:")
            report.append(f"    Predictions: {metrics.prediction_count}")
            report.append(f"    Accuracy: {metrics.accuracy_score:.1f}%")
            report.append(f"    R² Score: {metrics.r2_score:.3f}")
            report.append(f"    Correlation: {metrics.correlation:.3f}")
            report.append(f"    MAE: {metrics.mae:.4f}")
        report.append("")
        
        # Alerts
        if alerts:
            report.append("ACTIVE ALERTS:")
            for alert in alerts:
                report.append(f"  [{alert['severity'].upper()}] {alert['message']}")
        else:
            report.append("ALERTS: None")
        
        report.append("="*60)
        
        return "\n".join(report)
    
    def save_status(self):
        """Save current status to file"""
        try:
            status_data = {
                'model_metrics': {k: asdict(v) for k, v in self.model_metrics.items()},
                'system_health_history': [asdict(h) for h in self.system_health_history[-10:]],  # Last 10
                'start_time': self.start_time.isoformat(),
                'error_count': self.error_count,
                'total_predictions': self.total_predictions,
                'last_saved': datetime.now().isoformat()
            }
            
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving status: {str(e)}")
    
    def load_status(self):
        """Load status from file"""
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r') as f:
                    status_data = json.load(f)
                
                # Load model metrics
                for key, data in status_data.get('model_metrics', {}).items():
                    data['last_updated'] = datetime.fromisoformat(data['last_updated'])
                    self.model_metrics[key] = ModelMetrics(**data)
                
                # Load system health history
                for health_data in status_data.get('system_health_history', []):
                    health_data['timestamp'] = datetime.fromisoformat(health_data['timestamp'])
                    self.system_health_history.append(SystemHealth(**health_data))
                
                # Load other data
                if 'start_time' in status_data:
                    self.start_time = datetime.fromisoformat(status_data['start_time'])
                self.error_count = status_data.get('error_count', 0)
                self.total_predictions = status_data.get('total_predictions', 0)
                
                logger.info(f"Loaded model status from {self.status_file}")
                
        except Exception as e:
            logger.warning(f"Could not load existing status: {str(e)}")

# Global instance for easy access
model_monitor = ModelStatusMonitor()