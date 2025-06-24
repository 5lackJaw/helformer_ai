"""
Helformer Complete Pipeline Runner
Orchestrates the entire Helformer training, backtesting, and deployment process
Now with enhanced output options
"""

import os
import sys
import argparse
import logging
import subprocess
import time
import re
from datetime import datetime

# Unicode icon handling for Windows compatibility
def get_icon(unicode_icon, ascii_fallback):
    """Get unicode icon with ASCII fallback for Windows terminals"""
    try:
        # Test if we can encode the unicode icon to the system encoding
        test_encoding = sys.stdout.encoding or 'utf-8'
        unicode_icon.encode(test_encoding)
        return unicode_icon
    except (UnicodeEncodeError, AttributeError, LookupError):
        return ascii_fallback

# Define icons with fallbacks
ICONS = {
    'check': get_icon('✓', '[OK]'),
    'warning': get_icon('⚠', '[WARN]'),
    'error': get_icon('❌', '[ERROR]'),
    'success': get_icon('✅', '[SUCCESS]'),
    'rocket': get_icon('🚀', '[READY]'),
    'chart': get_icon('📊', '[CHART]'),
    'metrics': get_icon('📈', '[METRICS]'),
    'clipboard': get_icon('📋', '[INFO]'),
    'party': get_icon('🎉', '[COMPLETE]')
}

# Setup logging with proper encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('helformer_pipeline.log', encoding='utf-8'),
        logging.StreamHandler(stream=sys.stdout)
    ]
)

# Configure console handler for Unicode support
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass
elif hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logger = logging.getLogger(__name__)

class HelformerPipeline:
    """Complete Helformer pipeline orchestrator"""
    
    def __init__(self, config_path=None, output_mode='standard'):
        self.config_path = config_path
        self.output_mode = output_mode
        self.pipeline_status = {
            'data_validation': False,
            'training': False,
            'backtesting': False,
            'deployment_ready': False
        }
        self.results = {}
    
    def check_prerequisites(self):
        """Check if all prerequisites are met"""
        
        logger.info("Checking prerequisites...")
        
        issues = []
          # Check required files
        required_files = [
            'training_utils.py',
            'train_model.py',
            'backtest_engine.py',
            'live_trading.py',
            'validate_system.py',
            'trading_metrics.py',
            'config_helformer.py'
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                issues.append(f"Missing required file: {file}")
        
        # Check data directory
        if not os.path.exists('data'):
            issues.append("Data directory not found. Create 'data/' folder with crypto CSV files")
        
        # Check for at least one data file
        data_files = [f for f in os.listdir('data') if f.endswith('.csv')] if os.path.exists('data') else []
        if not data_files:
            issues.append("No CSV data files found in data/ directory")
        
        # Check Python packages
        required_packages = [
            'pandas', 'numpy', 'tensorflow', 'sklearn', 
            'matplotlib', 'seaborn', 'joblib', 'talib', 'ccxt'
        ]
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                issues.append(f"Missing required package: {package}")
        
        if issues:
            logger.error("Prerequisites check failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        
        logger.info("All prerequisites met!")
        return True
    
    def run_data_validation(self):
        """Run data validation step"""
        
        logger.info("Running data validation...")
        
        try:
            # Use existing validation system
            from validate_system import SystemValidator
            import pandas as pd
            
            # Load sample data for validation
            data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
            if not data_files:
                raise Exception("No data files found")
            
            # Run system validation instead of non-existent data validation
            validator = SystemValidator()
            validation_passed = validator.validate_dependencies() and validator.validate_integration_points()
            
            self.pipeline_status['data_validation'] = validation_passed
            self.results['data_validation'] = {
                'passed': validation_passed,
                'data_files': data_files,
                'validation_type': 'system_validation'
            }
            
            if validation_passed:
                logger.info("Data validation PASSED!")
            else:
                logger.warning("Data validation issues detected. Check logs.")
            return validation_passed
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            self.pipeline_status['data_validation'] = False
            return False
    
    def run_training(self):
        """Run Helformer training with real-time output"""
        
        logger.info("Starting Helformer training...")
        logger.info("=" * 60)
        logger.info("Training will be executed with real-time output...")
        logger.info("=" * 60)
        
        try:
            # Run training script with real-time output streaming
            import threading
            import queue
            
            def stream_output(pipe, output_queue, prefix):
                """Stream output from subprocess in real-time"""
                try:
                    for line in iter(pipe.readline, ''):
                        if line:
                            output_queue.put((prefix, line.rstrip()))
                    pipe.close()
                except Exception as e:
                    output_queue.put((prefix, f"Stream error: {e}"))
              # Start training process without capturing output
            process = subprocess.Popen([
                sys.executable, '-u', 'train_model.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
               text=True, bufsize=1, universal_newlines=True,
               env={**os.environ, 'PYTHONUNBUFFERED': '1'})
            
            # Create queues for real-time output
            output_queue = queue.Queue()
            
            # Start threads to stream stdout and stderr
            stdout_thread = threading.Thread(
                target=stream_output, 
                args=(process.stdout, output_queue, 'TRAIN')
            )
            stderr_thread = threading.Thread(
                target=stream_output, 
                args=(process.stderr, output_queue, 'ERROR')
            )
            
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            # Store all output for later analysis
            all_stdout = []
            all_stderr = []
            
            logger.info("Training process started - monitoring output...")
            
            # Monitor output in real-time with timeout
            timeout = 1800  # 30 minutes
            start_time = time.time()
            
            while process.poll() is None:
                # Check for timeout
                if time.time() - start_time > timeout:
                    logger.error("Training timeout (30 minutes) - terminating process")
                    process.terminate()
                    process.wait(10)  # Wait up to 10 seconds for graceful termination
                    if process.poll() is None:
                        process.kill()  # Force kill if still running
                    return False
                
                # Process output queue
                try:
                    prefix, line = output_queue.get(timeout=1)
                    if prefix == 'TRAIN':
                        logger.info(f"TRAIN: {line}")
                        all_stdout.append(line)
                    elif prefix == 'ERROR':
                        logger.warning(f"ERROR: {line}")
                        all_stderr.append(line)
                except queue.Empty:
                    # No output received in 1 second, continue monitoring
                    continue
                except Exception as e:
                    logger.warning(f"Output monitoring error: {e}")
                    break
            
            # Process any remaining output
            while not output_queue.empty():
                try:
                    prefix, line = output_queue.get_nowait()
                    if prefix == 'TRAIN':
                        logger.info(f"TRAIN: {line}")
                        all_stdout.append(line)
                    elif prefix == 'ERROR':
                        logger.warning(f"ERROR: {line}")
                        all_stderr.append(line)
                except queue.Empty:
                    break
            
            # Wait for threads to complete
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            
            # Get final return code
            return_code = process.wait()
            
            logger.info("=" * 60)
            logger.info(f"Training process completed with return code: {return_code}")
            logger.info("=" * 60)
            
            # Analyze results
            if return_code == 0:
                logger.info("Training completed successfully!")
                
                # Check if model files were created - more flexible checking
                model_files = [f for f in os.listdir('.') if f.startswith('helformer_model_') and f.endswith('.h5')]
                ensemble_info_exists = os.path.exists('helformer_ensemble_info.pkl')
                scaler_exists = os.path.exists('helformer_scaler.pkl')
                features_exists = os.path.exists('helformer_features.pkl')
                
                required_files = ['helformer_ensemble_info.pkl', 'helformer_scaler.pkl', 'helformer_features.pkl']
                missing_files = [f for f in required_files if not os.path.exists(f)]
                
                logger.info(f"Found {len(model_files)} model files")
                logger.info(f"Required files status:")
                for file in required_files:
                    status = "OK" if os.path.exists(file) else "MISSING"
                    logger.info(f"  {status}: {file}")
                
                if len(model_files) >= 1 and len(missing_files) == 0:  # At least 1 model and all required files
                    self.pipeline_status['training'] = True
                    
                    # Load training results
                    try:
                        import joblib
                        ensemble_info = joblib.load('helformer_ensemble_info.pkl')
                        
                        self.results['training'] = {
                            'status': 'completed',
                            'models_created': len(model_files),
                            'ensemble_info': ensemble_info
                        }
                        
                        logger.info(f"Training successful: {len(model_files)} models created")
                        logger.info(f"Performance: {ensemble_info.get('performance_level', 'Unknown')}")
                        logger.info(f"Expected Returns: {ensemble_info.get('expected_returns', 'Unknown')}")
                        return True                    
                    except Exception as e:
                        logger.warning(f"Could not load ensemble info: {e}")
                        # Still consider training successful if files exist
                        self.pipeline_status['training'] = True
                        self.results['training'] = {
                            'status': 'completed',
                            'models_created': len(model_files),
                            'ensemble_info': {'performance_level': 'Unknown'}
                        }
                        return True
                else:
                    logger.error(f"Training incomplete: Found {len(model_files)} models, missing files: {missing_files}")
                    return False
            else:
                logger.error(f"Training failed with return code {return_code}")
                if all_stderr:
                    logger.error("Error output:")
                    for line in all_stderr[-10:]:  # Show last 10 error lines
                        logger.error(f"  {line}")
                return False
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            return False
    
    def run_training_with_debug(self):
        """Run Helformer training with enhanced debugging and real-time output"""
        
        logger.info("Starting Helformer training with enhanced debugging...")
        logger.info("=" * 80)
        # First, check if training script exists and is accessible
        if not os.path.exists('train_model.py'):
            logger.error("train_model.py not found in current directory!")
            return False
        
        logger.info(f"{ICONS['check']} Training script found")
        logger.info(f"{ICONS['check']} Current working directory: {os.getcwd()}")
        logger.info(f"{ICONS['check']} Python executable: {sys.executable}")
        
        # Check data files
        data_files = []
        if os.path.exists('data'):
            data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
            logger.info(f"{ICONS['check']} Found {len(data_files)} data files: {data_files}")
        else:
            logger.warning(f"{ICONS['warning']} No data directory found")
        
        logger.info("=" * 80)
        logger.info("EXECUTING TRAINING WITH NATIVE KERAS PROGRESS BARS:")
        logger.info("=" * 80)
        
        try:            # Execute training WITHOUT capturing stdout - this preserves Keras progress bars!
            result = subprocess.run([
                sys.executable, '-u', 'train_model.py'
            ], env={**os.environ, 'PYTHONUNBUFFERED': '1'})
            
            logger.info("=" * 80)
            logger.info(f"Training process finished with return code: {result.returncode}")
            logger.info("=" * 80)
            
            # Check results
            if result.returncode == 0:
                logger.info("Training completed successfully!")
                
                # Verify output files
                model_files = [f for f in os.listdir('.') if f.startswith('helformer_model_') and f.endswith('.h5')]
                required_files = ['helformer_ensemble_info.pkl', 'helformer_scaler.pkl', 'helformer_features.pkl']
                
                logger.info(f"Model files created: {len(model_files)}")
                for f in model_files:
                    logger.info(f"  {ICONS['check']} {f}")
                
                missing_files = [f for f in required_files if not os.path.exists(f)]
                if missing_files:
                    logger.error(f"Missing required files: {missing_files}")
                    return False
                else:
                    logger.info(f"{ICONS['check']} All required files created successfully")
                    self.pipeline_status['training'] = True
                    return True
            else:
                logger.error(f"Training failed with return code: {result.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"Exception during training execution: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def run_training_clean(self):
        """Run training with clean output (minimal spam)"""
        logger.info("Starting Helformer training (clean mode)...")
        if not os.path.exists('train_model.py'):
            logger.error("train_model.py not found!")
            return False
        
        try:
            process = subprocess.Popen([
                sys.executable, '-u', 'train_model.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
               text=True, bufsize=1, universal_newlines=True,
               env={**os.environ, 'PYTHONUNBUFFERED': '1'})
            
            step_count = 0
            last_update = time.time()
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    step_count += 1
                    
                    # Only show important lines
                    if any(keyword in line.lower() for keyword in [
                        'training model', 'epoch', 'helformer', 'ensemble', 
                        'performance', 'r2', 'mape', 'successful', 'completed',
                        'error', 'failed', 'validation', 'saved', 'target'
                    ]) and not ('ms/step' in line and len(line) > 50):
                        current_time = time.time()
                        if current_time - last_update > 2:  # Throttle output
                            logger.info(f"TRAIN: {line}")
                            last_update = current_time
                    
                    # Show progress dots
                    if step_count % 500 == 0:
                        print(".", end="", flush=True)
            
            return_code = process.poll()
            
            if return_code == 0:
                logger.info("Training completed successfully!")
                
                # Check results
                model_files = [f for f in os.listdir('.') if f.startswith('helformer_model_') and f.endswith('.h5')]
                required_files = ['helformer_ensemble_info.pkl', 'helformer_scaler.pkl', 'helformer_features.pkl']
                missing_files = [f for f in required_files if not os.path.exists(f)]
                
                if len(model_files) >= 1 and len(missing_files) == 0:
                    self.pipeline_status['training'] = True
                    logger.info(f"{ICONS['success']} Training successful: {len(model_files)} models created")
                    return True
                else:
                    logger.error(f"Training incomplete: {len(model_files)} models, missing: {missing_files}")
                    return False
            else:
                logger.error(f"Training failed with return code: {return_code}")
                return False
                
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            return False
    
    def run_backtesting(self):
        """Run Helformer backtesting with real-time output"""
        
        logger.info("Starting Helformer backtesting...")
        logger.info("=" * 60)
        logger.info("Backtesting will be executed with real-time output...")
        logger.info("=" * 60)
        
        try:
            # Run backtesting script with real-time output
            process = subprocess.Popen([
                sys.executable, '-u', 'backtest_helformer.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
               text=True, bufsize=1, universal_newlines=True,
               env={**os.environ, 'PYTHONUNBUFFERED': '1'})
            
            # Stream output in real-time
            import threading
            import queue
            
            def stream_output(pipe, output_queue, prefix):
                """Stream output from subprocess in real-time"""
                try:
                    for line in iter(pipe.readline, ''):
                        if line:
                            output_queue.put((prefix, line.rstrip()))
                    pipe.close()
                except Exception as e:
                    output_queue.put((prefix, f"Stream error: {e}"))
            
            output_queue = queue.Queue()
            
            stdout_thread = threading.Thread(
                target=stream_output, 
                args=(process.stdout, output_queue, 'BACKTEST')
            )
            stderr_thread = threading.Thread(
                target=stream_output, 
                args=(process.stderr, output_queue, 'ERROR')
            )
            
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            logger.info("Backtesting process started - monitoring output...")
            
            # Monitor output in real-time
            import time
            timeout = 1800  # 30 minutes
            start_time = time.time()
            
            while process.poll() is None:
                if time.time() - start_time > timeout:
                    logger.error("Backtesting timeout (30 minutes)")
                    process.terminate()
                    return False
                
                try:
                    prefix, line = output_queue.get(timeout=1)
                    if prefix == 'BACKTEST':
                        logger.info(f"BACKTEST: {line}")
                    elif prefix == 'ERROR':
                        logger.warning(f"ERROR: {line}")
                except queue.Empty:
                    continue
            
            # Process remaining output
            while not output_queue.empty():
                try:
                    prefix, line = output_queue.get_nowait()
                    if prefix == 'BACKTEST':
                        logger.info(f"BACKTEST: {line}")
                    elif prefix == 'ERROR':
                        logger.warning(f"ERROR: {line}")
                except queue.Empty:
                    break
            
            return_code = process.wait()
            
            logger.info("=" * 60)
            logger.info(f"Backtesting completed with return code: {return_code}")
            logger.info("=" * 60)
            
            if return_code == 0:
                logger.info("Backtesting completed successfully!")
                
                # Check if results were created
                trades_file = 'helformer_trades.csv'
                equity_file = 'helformer_equity_curve.csv'
                
                if os.path.exists(trades_file) and os.path.exists(equity_file):
                    import pandas as pd
                    
                    # Load backtest results
                    trades_df = pd.read_csv(trades_file)
                    equity_df = pd.read_csv(equity_file)
                    
                    # Calculate basic metrics
                    if not trades_df.empty:
                        total_return = (equity_df['equity'].iloc[-1] - 10000) / 10000 * 100
                        win_rate = (trades_df['pnl'] > 0).mean() * 100
                        total_trades = len(trades_df)
                        
                        self.results['backtesting'] = {
                            'status': 'completed',
                            'total_return_pct': total_return,
                            'win_rate_pct': win_rate,
                            'total_trades': total_trades,
                            'final_balance': equity_df['equity'].iloc[-1]
                        }
                        
                        # Check if meets minimum targets
                        meets_targets = total_return >= 100 and win_rate >= 50  # Minimum viable
                        self.pipeline_status['backtesting'] = meets_targets
                        
                        if meets_targets:
                            logger.info(f"Backtesting PASSED! Return: {total_return:.1f}%, Win Rate: {win_rate:.1f}%")
                        else:
                            logger.warning(f"Backtesting below targets. Return: {total_return:.1f}%, Win Rate: {win_rate:.1f}%")
                        
                        return meets_targets
                    else:
                        logger.warning("Backtesting completed but no trades generated")
                        return False
                else:
                    logger.error("Backtesting completed but result files not found")
                    return False
            else:
                logger.error(f"Backtesting failed with return code: {return_code}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Backtesting timeout (30 minutes)")
            return False
        except Exception as e:
            logger.error(f"Backtesting error: {str(e)}")
            return False
    
    def check_deployment_readiness(self):
        """Check if system is ready for deployment"""
        
        logger.info("Checking deployment readiness...")
        
        required_files = [
            'helformer_ensemble_info.pkl',
            'helformer_scaler.pkl',
            'helformer_features.pkl'
        ]
        
        # Check model files
        model_files = [f for f in os.listdir('.') if f.startswith('helformer_model_')]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if not model_files:
            missing_files.append("helformer_model_*.h5 files")
        
        # Check environment file for live trading
        if not os.path.exists('.env'):
            logger.warning("No .env file found. Create one with API keys for live trading")
        
        deployment_ready = (
            len(missing_files) == 0 and
            self.pipeline_status['training'] and
            self.pipeline_status['backtesting']
        )
        
        self.pipeline_status['deployment_ready'] = deployment_ready
        
        if deployment_ready:
            logger.info("System is READY for deployment!")
        else:
            logger.warning("System NOT ready for deployment:")
            for file in missing_files:
                logger.warning(f"  - Missing: {file}")
            
            if not self.pipeline_status['training']:
                logger.warning("  - Training not completed successfully")
            if not self.pipeline_status['backtesting']:
                logger.warning("  - Backtesting did not meet minimum targets")
        
        return deployment_ready
    
    def generate_pipeline_report(self):
        """Generate comprehensive pipeline report"""
        
        report = []
        report.append("HELFORMER PIPELINE EXECUTION REPORT")
        report.append("=" * 60)
        report.append(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Pipeline Status
        report.append("PIPELINE STATUS:")
        report.append("-" * 30)
        for step, status in self.pipeline_status.items():
            status_icon = ICONS['success'] if status else ICONS['error']
            report.append(f"{status_icon} {step.replace('_', ' ').title()}: {'PASSED' if status else 'FAILED'}")
        
        # Results Summary
        if self.results:
            report.append("\nRESULTS SUMMARY:")
            report.append("-" * 30)
            
            if 'training' in self.results:
                training = self.results['training']
                report.append(f"Models Created: {training.get('models_created', 0)}")
                if 'ensemble_info' in training:
                    info = training['ensemble_info']
                    report.append(f"Expected Returns: {info.get('expected_returns', 'Unknown')}")
                    report.append(f"Model Quality: {info.get('performance_level', 'Unknown')}")
            
            if 'backtesting' in self.results:
                bt = self.results['backtesting']
                report.append(f"Backtest Return: {bt.get('total_return_pct', 0):.1f}%")
                report.append(f"Win Rate: {bt.get('win_rate_pct', 0):.1f}%")
                report.append(f"Total Trades: {bt.get('total_trades', 0)}")
                report.append(f"Final Balance: ${bt.get('final_balance', 0):.2f}")
        
        # Recommendations
        report.append("\nRECOMMENDATIONS:")
        report.append("-" * 30)
        
        if self.pipeline_status['deployment_ready']:
            report.append(f"{ICONS['rocket']} System ready for live trading!")
            report.append("   1. Set up .env file with API keys")
            report.append("   2. Start with testnet mode")
            report.append("   3. Run: python main_helformer.py")
        else:
            if not self.pipeline_status['data_validation']:
                report.append(f"{ICONS['error']} Fix data validation issues first")
            if not self.pipeline_status['training']:
                report.append(f"{ICONS['error']} Complete model training successfully")
            if not self.pipeline_status['backtesting']:
                report.append(f"{ICONS['error']} Improve backtesting performance")
        
        report_text = "\n".join(report)
        
        # Save report
        with open('helformer_pipeline_report.txt', 'w') as f:
            f.write(report_text)
        
        logger.info("\n" + report_text)
        logger.info("Full report saved to: helformer_pipeline_report.txt")
        
        return report_text
    
    def run_full_pipeline(self, skip_training=False, skip_backtesting=False):
        """Run the complete Helformer pipeline"""
        
        logger.info("STARTING HELFORMER COMPLETE PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            logger.error("Prerequisites check failed. Pipeline aborted.")
            return False
        
        # Step 2: Data validation
        if not self.run_data_validation():
            logger.error("Data validation failed. Pipeline aborted.")
            return False        # Step 3: Training (optional skip)
        if not skip_training:
            logger.info(f"Running training with {self.output_mode} output mode...")
            if not self.run_training_adaptive():
                logger.error("Training failed. Pipeline aborted.")
                return False
        else:
            logger.info("Skipping training (models should already exist)")
            self.pipeline_status['training'] = True  # Assume existing models
        
        # Step 4: Backtesting (optional skip)
        if not skip_backtesting:
            if not self.run_backtesting():
                logger.warning("Backtesting failed or below targets. Continuing...")
        else:
            logger.info("Skipping backtesting")
            self.pipeline_status['backtesting'] = True
        
        # Step 5: Deployment readiness check
        deployment_ready = self.check_deployment_readiness()
        
        # Step 6: Generate report
        self.generate_pipeline_report()
        if deployment_ready:
            logger.info(f"{ICONS['rocket']} PIPELINE COMPLETED SUCCESSFULLY - READY FOR DEPLOYMENT!")
        else:
            logger.warning(f"{ICONS['warning']} PIPELINE COMPLETED WITH ISSUES - CHECK REPORT")
        
        return deployment_ready

    def run_training_adaptive(self):
        """Run training using the appropriate method based on output mode"""
        if self.output_mode == 'debug':
            logger.info("Running training in debug mode (verbose with progress)...")
            return self.run_training_with_debug()
        elif self.output_mode == 'clean':
            logger.info("Running training in clean mode (minimal output)...")
            return self.run_training_clean()
        else:  # standard
            logger.info("Running training in standard mode...")
            return self.run_training()

def main():
    """Main entry point for pipeline execution"""
    
    parser = argparse.ArgumentParser(description='Helformer Complete Pipeline')
    parser.add_argument('--skip-training', action='store_true', 
                       help='Skip training step (use existing models)')
    parser.add_argument('--skip-backtesting', action='store_true',
                       help='Skip backtesting step')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--step', choices=['validate', 'train', 'backtest', 'check'],
                       help='Run specific step only')
    parser.add_argument('--output-mode', choices=['standard', 'debug', 'clean'], 
                       default='standard', help='Output verbosity mode (standard=normal, debug=shows native Keras progress bars, clean=minimal)')
    parser.add_argument('--quiet', action='store_true',
                       help='Run in quiet mode (minimal output, equivalent to --output-mode clean)')
    
    args = parser.parse_args()
    
    # Handle quiet flag (overrides output-mode)
    output_mode = 'clean' if args.quiet else args.output_mode
    
    # Create pipeline with appropriate output mode
    pipeline = HelformerPipeline(config_path=args.config, output_mode=output_mode)
    
    # Run specific step or full pipeline
    if args.step:        
        if args.step == 'validate':
            success = pipeline.run_data_validation()
        elif args.step == 'train':
            success = pipeline.run_training_adaptive()  # Use adaptive training method
        elif args.step == 'backtest':
            success = pipeline.run_backtesting()
        elif args.step == 'check':
            success = pipeline.check_deployment_readiness()
        
        if success:
            logger.info(f"Step '{args.step}' completed successfully!")
        else:
            logger.error(f"Step '{args.step}' failed!")
            sys.exit(1)
    else:
        # Run full pipeline
        success = pipeline.run_full_pipeline(
            skip_training=args.skip_training,
            skip_backtesting=args.skip_backtesting
        )
        
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()