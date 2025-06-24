#!/usr/bin/env python3
"""
Performance Evaluation Script
Evaluates trained ensemble models and provides comprehensive performance metrics.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import project modules
from config_helformer import HelformerConfig
from backtest_engine import ConsolidatedHelformerBacktest
from helformer_model import HelformerEnsembleManager
from improved_training_utils import prepare_model_data

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)8s | %(name)20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def evaluate_model_performance(config, logger):
    """Evaluate trained models with comprehensive metrics"""
    
    # Find latest experiment folder
    experiment_folders = [f for f in os.listdir('.') if f.startswith('experiment_')]
    if not experiment_folders:
        logger.error("No experiment folders found!")
        return
    
    latest_experiment = max(experiment_folders, key=lambda x: x.split('_')[1] + '_' + x.split('_')[2])
    experiment_path = Path(latest_experiment)
    
    logger.info(f"Evaluating models from: {latest_experiment}")
    
    results = {}
    
    for asset in config.ASSETS_TO_TRADE:
        logger.info(f"\n=== EVALUATING {asset} ENSEMBLE ===")
        
        # Check if models exist
        model_files = list(experiment_path.glob(f"helformer_{asset.lower()}_ensemble_model_*.h5"))
        metadata_file = experiment_path / f"helformer_{asset.lower()}_ensemble_metadata.json"
        scaler_file = experiment_path / f"helformer_{asset.lower()}_scaler.pkl"
        
        if not model_files or not metadata_file.exists() or not scaler_file.exists():
            logger.warning(f"Missing files for {asset}, skipping...")
            continue
        
        try:
            # Load data
            data_file = f"data/{asset}_USDT_data.csv"
            if not os.path.exists(data_file):
                logger.warning(f"Data file not found: {data_file}")
                continue
            
            df = pd.read_csv(data_file)
            logger.info(f"Loaded {len(df)} rows of data for {asset}")
            
            # Prepare features and targets
            df_clean, feature_columns = prepare_model_data(df, config.SEQUENCE_LENGTH)
            logger.info(f"Prepared data: {df_clean.shape}, features: {len(feature_columns)}")
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Extract training metrics
            ensemble_configs = metadata.get('ensemble_configs', [])
            model_weights = metadata.get('model_weights', [])
            
            logger.info(f"Found {len(model_files)} models with weights: {model_weights}")
            
            # Calculate model quality metrics
            model_metrics = []
            for i, config_data in enumerate(ensemble_configs):
                if 'history' in config_data:
                    history = config_data['history']
                    final_val_loss = history.get('val_loss', [])[-1] if history.get('val_loss') else None
                    final_val_mae = history.get('val_mae', [])[-1] if history.get('val_mae') else None
                    
                    model_metrics.append({
                        'model_type': config_data.get('type', 'unknown'),
                        'selection_score': config_data.get('selection_score', 0),
                        'final_val_loss': final_val_loss,
                        'final_val_mae': final_val_mae,
                        'weight': model_weights[i] if i < len(model_weights) else 0
                    })            # Run backtest
            logger.info("Running backtest...")
            backtest_engine = ConsolidatedHelformerBacktest(config)
            
            # Run backtest for the asset
            backtest_results = backtest_engine.run_asset_backtest(asset)
            
            if backtest_results:
                # Extract key performance metrics
                metrics = backtest_results.get('performance_metrics', {})
                trades = backtest_results.get('trades', [])
                
                results[asset] = {
                    'model_metrics': model_metrics,
                    'backtest_metrics': metrics,
                    'num_trades': len(trades),
                    'experiment_path': str(experiment_path)
                }
                
                # Log key results
                logger.info(f"=== {asset} PERFORMANCE SUMMARY ===")
                logger.info(f"Total Return: {metrics.get('total_return', 0):.2%}")
                logger.info(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
                logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
                logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
                logger.info(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
                logger.info(f"Number of Trades: {len(trades)}")
                
                # Model quality summary
                avg_val_loss = np.mean([m['final_val_loss'] for m in model_metrics if m['final_val_loss']])
                avg_val_mae = np.mean([m['final_val_mae'] for m in model_metrics if m['final_val_mae']])
                avg_selection_score = np.mean([m['selection_score'] for m in model_metrics])
                
                logger.info(f"Average Validation Loss: {avg_val_loss:.4f}")
                logger.info(f"Average Validation MAE: {avg_val_mae:.4f}")
                logger.info(f"Average Selection Score: {avg_selection_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error evaluating {asset}: {e}")
            continue
    
    # Overall summary
    if results:
        logger.info("\n" + "="*60)
        logger.info("OVERALL PERFORMANCE SUMMARY")
        logger.info("="*60)
        
        total_returns = []
        sharpe_ratios = []
        max_drawdowns = []
        win_rates = []
        
        for asset, metrics in results.items():
            bm = metrics['backtest_metrics']
            total_returns.append(bm.get('total_return', 0))
            sharpe_ratios.append(bm.get('sharpe_ratio', 0))
            max_drawdowns.append(bm.get('max_drawdown', 0))
            win_rates.append(bm.get('win_rate', 0))
        
        if total_returns:
            logger.info(f"Average Total Return: {np.mean(total_returns):.2%}")
            logger.info(f"Average Sharpe Ratio: {np.mean(sharpe_ratios):.3f}")
            logger.info(f"Average Max Drawdown: {np.mean(max_drawdowns):.2%}")
            logger.info(f"Average Win Rate: {np.mean(win_rates):.2%}")
            
            # Check if we're meeting performance targets
            target_return = 9.25  # 925% annual return target
            avg_annualized = np.mean([results[asset]['backtest_metrics'].get('annualized_return', 0) 
                                    for asset in results])
            
            if avg_annualized >= target_return:
                logger.info(f"✅ PERFORMANCE TARGET MET! Average annualized return: {avg_annualized:.2%}")
            else:
                logger.warning(f"⚠️  Performance below target. Average: {avg_annualized:.2%}, Target: {target_return:.2%}")
                
                # Provide improvement suggestions
                logger.info("\n=== IMPROVEMENT SUGGESTIONS ===")
                if np.mean(sharpe_ratios) < 2.0:
                    logger.info("- Consider improving signal quality (Sharpe < 2.0)")
                if np.mean(max_drawdowns) > 0.15:
                    logger.info("- Implement stronger risk controls (Drawdown > 15%)")
                if np.mean(win_rates) < 0.55:
                    logger.info("- Optimize entry/exit signals (Win rate < 55%)")
    
    return results

def main():
    """Main evaluation function"""
    logger = setup_logging()
    logger.info("Starting performance evaluation...")
    
    try:
        config = HelformerConfig()
        results = evaluate_model_performance(config, logger)
        
        if results:
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"performance_evaluation_{timestamp}.json"
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            results_clean = convert_numpy(results)
            
            with open(results_file, 'w') as f:
                json.dump(results_clean, f, indent=2)
            
            logger.info(f"Results saved to: {results_file}")
        
        logger.info("Performance evaluation completed!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
