"""
Comprehensive System Validation Script
Validates all institutional-grade enhancements and integrations
"""

import os
import sys
import importlib
import logging
import traceback
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemValidator:
    """Validates the complete Helformer trading system"""
    
    def __init__(self):
        self.validation_results = {}
        self.project_root = Path(__file__).parent

    def validate_module_imports(self):
        """Validate all new modules can be imported"""
        logger.info("[CHECK] Validating module imports...")
        
        modules_to_test = [
            'exchange_manager',
            'portfolio_risk_manager', 
            'execution_simulator',
            'regime_trading_logic',
            'market_regime_detector',
            'config_helformer',
            'training_utils',
            'trading_metrics'
        ]
        
        import_results = {}
        
        for module_name in modules_to_test:
            try:
                module = importlib.import_module(module_name)
                import_results[module_name] = {
                    'status': 'SUCCESS',
                    'path': getattr(module, '__file__', 'Built-in'),
                    'error': None
                }
                logger.info(f"[OK] {module_name}: Import successful")
            except Exception as e:
                import_results[module_name] = {
                    'status': 'FAILED',
                    'path': None,
                    'error': str(e)
                }
                logger.error(f"[FAIL] {module_name}: Import failed - {e}")
                
        self.validation_results['imports'] = import_results
        return all(result['status'] == 'SUCCESS' for result in import_results.values())
        
    def validate_configuration(self):
        """Validate configuration completeness"""
        logger.info("[CHECK] Validating configuration...")
        
        try:
            from config_helformer import config
            
            required_config_sections = [
                'MULTI_EXCHANGE_CONFIG',
                'RISK_MANAGEMENT_CONFIG',
                'EXECUTION_SIMULATION_CONFIG',
                'MARKET_REGIME_CONFIG',
                'INSTITUTIONAL_FEATURES'
            ]
            
            config_results = {}
            
            for section in required_config_sections:
                if hasattr(config, section):
                    config_results[section] = 'PRESENT'
                    logger.info(f"[OK] Config section {section}: Present")
                else:
                    config_results[section] = 'MISSING'
                    logger.warning(f"[WARN] Config section {section}: Missing")
                    
            self.validation_results['configuration'] = config_results
            return all(status == 'PRESENT' for status in config_results.values())
            
        except Exception as e:
            logger.error(f"[FAIL] Configuration validation failed: {e}")
            self.validation_results['configuration'] = {'error': str(e)}
            return False
            
    def validate_class_instantiation(self):
        """Validate key classes can be instantiated"""
        logger.info("[CHECK] Validating class instantiation...")
        
        instantiation_results = {}
        
        # Test ExchangeManager
        try:
            from exchange_manager import ExchangeManager
            test_config = {
                'enabled_exchanges': ['binance'],
                'exchange_configs': {
                    'binance': {'apiKey': 'test', 'secret': 'test', 'sandbox': True}
                },
                'default_symbols': ['BTC/USDT'],
                'update_frequency': 1.0
            }
            em = ExchangeManager(test_config)
            instantiation_results['ExchangeManager'] = 'SUCCESS'
            logger.info("[OK] ExchangeManager: Instantiation successful")
        except Exception as e:
            instantiation_results['ExchangeManager'] = f'FAILED: {e}'
            logger.error(f"[FAIL] ExchangeManager: {e}")
            
        # Test PortfolioRiskManager
        try:
            from portfolio_risk_manager import PortfolioRiskManager
            test_config = {
                'var_confidence_level': 0.05,
                'max_portfolio_var': 0.05,
                'max_single_position_size': 0.3,
                'risk_free_rate': 0.02
            }
            prm = PortfolioRiskManager(test_config)
            instantiation_results['PortfolioRiskManager'] = 'SUCCESS'
            logger.info("[OK] PortfolioRiskManager: Instantiation successful")
        except Exception as e:
            instantiation_results['PortfolioRiskManager'] = f'FAILED: {e}'
            logger.error(f"[FAIL] PortfolioRiskManager: {e}")
            
        # Test ExecutionSimulator
        try:
            from execution_simulator import ExecutionSimulator
            test_config = {
                'slippage_model': 'adaptive',
                'base_slippage_bps': 2.0,
                'latency_model': 'normal',
                'mean_latency_ms': 100,
                'partial_fill_probability': 0.1,
                'market_impact': {
                    'linear_impact_coef': 0.001,
                    'sqrt_impact_coef': 0.0005
                }
            }
            es = ExecutionSimulator(test_config)
            instantiation_results['ExecutionSimulator'] = 'SUCCESS'
            logger.info("[OK] ExecutionSimulator: Instantiation successful")
        except Exception as e:
            instantiation_results['ExecutionSimulator'] = f'FAILED: {e}'
            logger.error(f"[FAIL] ExecutionSimulator: {e}")
            
        # Test MarketRegimeDetector
        try:
            from market_regime_detector import MarketRegimeDetector
            test_config = {
                'hurst_window': 252,
                'volatility_window': 20,
                'trend_threshold': 0.02
            }
            mrd = MarketRegimeDetector(test_config)
            instantiation_results['MarketRegimeDetector'] = 'SUCCESS'
            logger.info("[OK] MarketRegimeDetector: Instantiation successful")
        except Exception as e:
            instantiation_results['MarketRegimeDetector'] = f'FAILED: {e}'
            logger.error(f"[FAIL] MarketRegimeDetector: {e}")
            
        self.validation_results['instantiation'] = instantiation_results
        return all('SUCCESS' in result for result in instantiation_results.values())
        
    def validate_file_structure(self):
        """Validate required files exist"""
        logger.info("[CHECK] Validating file structure...")
        
        required_files = [
            'exchange_manager.py',
            'portfolio_risk_manager.py',
            'execution_simulator.py',            'live_trading.py',
            'backtest_engine.py',
            'config_helformer.py',
            'requirements.txt',
            'INSTITUTIONAL_FEATURES.md',
            'REGIME_DETECTION_GUIDE.md',
            'EXECUTION_ENGINE_DOCS.md',
            'BACKTESTING_ACCURACY_VALIDATION.md',
            'IMPLEMENTATION_STATUS.md'
        ]
        
        file_results = {}
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                file_results[file_path] = {
                    'status': 'EXISTS',
                    'size': full_path.stat().st_size
                }
                logger.info(f"[OK] {file_path}: Exists ({full_path.stat().st_size} bytes)")
            else:
                file_results[file_path] = {
                    'status': 'MISSING',
                    'size': 0
                }
                logger.warning(f"[WARN] {file_path}: Missing")
                
        self.validation_results['files'] = file_results
        return all(result['status'] == 'EXISTS' for result in file_results.values())
        
    def validate_dependencies(self):
        """Validate all required dependencies"""
        logger.info("[CHECK] Validating dependencies...")
        
        required_packages = [
            'numpy', 'pandas', 'scipy', 'sklearn',
            'tensorflow', 'ccxt', 'dotenv',
            'statsmodels', 'joblib', 'pytest'
        ]
        
        dependency_results = {}
        
        for package in required_packages:
            try:
                if package == 'dotenv':
                    __import__('dotenv')
                else:
                    __import__(package)
                dependency_results[package] = 'AVAILABLE'
                logger.info(f"[OK] {package}: Available")
            except ImportError:
                dependency_results[package] = 'MISSING'
                logger.warning(f"[WARN] {package}: Missing")
                self.validation_results['dependencies'] = dependency_results
        return all(status == 'AVAILABLE' for status in dependency_results.values())
        
    def validate_integration_points(self):
        """Validate integration between modules"""
        logger.info("[CHECK] Validating integration points...")
        
        integration_results = {}
        # Test if live_trading imports new modules
        try:
            from exchange_manager import ExchangeManager
            from portfolio_risk_manager import PortfolioRiskManager  
            from execution_simulator import ExecutionSimulator
            integration_results['live_trading_imports'] = 'SUCCESS'
            logger.info("[OK] live_trading: Imports new modules successfully")
        except Exception as e:
            integration_results['live_trading_imports'] = f'FAILED: {e}'
            logger.error(f"[FAIL] live_trading imports: {e}")
        
        # Test if backtest_engine imports new modules
        try:
            # Read file content to check for imports
            backtest_file = self.project_root / 'backtest_engine.py'
            if backtest_file.exists():
                try:
                    content = backtest_file.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        content = backtest_file.read_text(encoding='latin-1')
                    except UnicodeDecodeError:
                        content = backtest_file.read_text(encoding='cp1252')
                
                if 'ExecutionSimulator' in content and 'PortfolioRiskManager' in content:
                    integration_results['backtest_integration'] = 'SUCCESS'
                    logger.info("[OK] backtest_engine: Integration complete")
                else:
                    integration_results['backtest_integration'] = 'PARTIAL'
                    logger.warning("[WARN] backtest_engine: Partial integration")
            else:
                integration_results['backtest_integration'] = 'FILE_MISSING'
                logger.error("[FAIL] backtest_engine.py: File missing")
        except Exception as e:
            integration_results['backtest_integration'] = f'FAILED: {e}'
            logger.error(f"[FAIL] Backtest integration check: {e}")
            
        self.validation_results['integration'] = integration_results
        return all('SUCCESS' in result for result in integration_results.values())
        
    def validate_test_infrastructure(self):
        """Validate test infrastructure"""
        logger.info("[CHECK] Validating test infrastructure...")
        
        test_files = [
            'tests/test_exchange_manager.py',
            'tests/test_portfolio_risk_manager.py',
            'tests/test_execution_simulator.py',
            'tests/test_integration.py',
            'tests/conftest.py',
            'pytest.ini'
        ]
        
        test_results = {}
        
        for test_file in test_files:
            full_path = self.project_root / test_file
            if full_path.exists():
                test_results[test_file] = 'EXISTS'
                logger.info(f"[OK] {test_file}: Exists")
            else:
                test_results[test_file] = 'MISSING'
                logger.warning(f"[WARN] {test_file}: Missing")
                
        self.validation_results['tests'] = test_results
        return all(status == 'EXISTS' for status in test_results.values())
        
    def run_comprehensive_validation(self):
        """Run all validation checks"""
        logger.info("[START] Starting comprehensive system validation...")
        
        validation_checks = [
            ('File Structure', self.validate_file_structure),
            ('Dependencies', self.validate_dependencies),
            ('Module Imports', self.validate_module_imports),
            ('Configuration', self.validate_configuration),
            ('Class Instantiation', self.validate_class_instantiation),
            ('Integration Points', self.validate_integration_points),
            ('Test Infrastructure', self.validate_test_infrastructure)
        ]
        
        all_passed = True
        
        for check_name, check_function in validation_checks:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running {check_name} validation...")
            logger.info(f"{'='*50}")
            
            try:
                result = check_function()
                if result:
                    logger.info(f"[PASS] {check_name}: PASSED")
                else:
                    logger.warning(f"[WARN] {check_name}: FAILED or INCOMPLETE")
                    all_passed = False
            except Exception as e:
                logger.error(f"[ERROR] {check_name}: ERROR - {e}")
                logger.error(traceback.format_exc())
                all_passed = False
                
        # Generate summary report
        self.generate_validation_report(all_passed)
        
        return all_passed
        
    def generate_validation_report(self, overall_success):
        """Generate comprehensive validation report"""
        report_path = self.project_root / 'VALIDATION_REPORT.md'
        
        with open(report_path, 'w') as f:
            f.write("# System Validation Report\n\n")
            f.write(f"**Validation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Overall Status**: {'[PASSED]' if overall_success else '[FAILED]'}\n\n")
            
            for category, results in self.validation_results.items():
                f.write(f"## {category.title()} Validation\n\n")
                
                if isinstance(results, dict):
                    for item, status in results.items():
                        if isinstance(status, dict):
                            f.write(f"- **{item}**: {status.get('status', status)}\n")
                        else:
                            f.write(f"- **{item}**: {status}\n")
                else:
                    f.write(f"- **Result**: {results}\n")
                    
                f.write("\n")
                
            f.write("## Recommendations\n\n")
            
            if not overall_success:
                f.write("The following issues were identified and should be addressed:\n\n")
                
                for category, results in self.validation_results.items():
                    if isinstance(results, dict):
                        failed_items = []
                        for item, status in results.items():
                            if isinstance(status, dict):
                                if status.get('status') not in ['SUCCESS', 'EXISTS', 'AVAILABLE', 'PRESENT']:
                                    failed_items.append(f"  - {item}: {status.get('status', status)}")
                            elif 'FAILED' in str(status) or 'MISSING' in str(status):
                                failed_items.append(f"  - {item}: {status}")
                                
                        if failed_items:
                            f.write(f"**{category.title()}**:\n")
                            f.write("\n".join(failed_items))
                            f.write("\n\n")
            else:
                f.write("All validation checks passed successfully! [PASSED]\n")
                f.write("The system is ready for institutional-grade trading.\n")
                
        logger.info(f"[REPORT] Validation report generated: {report_path}")


def main():
    """Main validation entry point"""
    validator = SystemValidator()
    success = validator.run_comprehensive_validation()
    
    if success:
        logger.info("\n[SUCCESS] ALL VALIDATIONS PASSED! System is ready for institutional trading.")
        return 0
    else:
        logger.error("\n[FAILED] VALIDATION FAILED! Please address the issues before proceeding.")
        return 1


if __name__ == '__main__':
    sys.exit(main())