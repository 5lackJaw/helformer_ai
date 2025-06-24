"""
Portfolio Risk Management System for Helformer
Implements comprehensive risk management including VaR, correlation analysis, and stress testing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.covariance import LedoitWolf
import warnings
from config_helformer import config

logger = logging.getLogger(__name__)

@dataclass
class PositionRisk:
    """Risk metrics for individual position"""
    symbol: str
    position_size: float
    market_value: float
    portfolio_weight: float
    daily_var_95: float
    daily_var_99: float
    beta: float
    volatility: float
    max_drawdown: float
    sharpe_ratio: float
    correlation_to_portfolio: float

@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    total_value: float
    daily_var_95: float
    daily_var_99: float
    expected_shortfall_95: float
    portfolio_volatility: float
    portfolio_beta: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    correlation_concentration: float
    sector_concentration: float
    risk_contribution: Dict[str, float]
    timestamp: datetime

@dataclass
class StressTestScenario:
    """Stress testing scenario definition"""
    name: str
    description: str
    market_shocks: Dict[str, float]  # symbol -> shock percentage
    correlation_multiplier: float = 1.0
    volatility_multiplier: float = 1.0

@dataclass
class StressTestResult:
    """Result of stress testing"""
    scenario_name: str
    portfolio_pnl: float
    portfolio_pnl_pct: float
    position_pnls: Dict[str, float]
    var_breach: bool
    max_loss_position: str
    recovery_days: int
    timestamp: datetime

class PortfolioRiskManager:
    """
    Comprehensive portfolio risk management system.
    
    Features:
    - Value at Risk (VaR) calculation using multiple methods
    - Correlation and concentration analysis
    - Stress testing and scenario analysis
    - Risk-adjusted position sizing
    - Real-time risk monitoring
    - Risk limits enforcement
    """
    
    def __init__(self, 
                 lookback_days: int = 252,
                 confidence_levels: List[float] = [0.95, 0.99],
                 risk_free_rate: float = 0.02):
        """
        Initialize portfolio risk manager.
        
        Args:
            lookback_days: Historical data window for calculations
            confidence_levels: VaR confidence levels
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.lookback_days = lookback_days
        self.confidence_levels = confidence_levels
        self.risk_free_rate = risk_free_rate
        
        # Portfolio data
        self.positions: Dict[str, Dict] = {}
        self.price_history: Dict[str, pd.Series] = {}
        self.returns_history: Dict[str, pd.Series] = {}
        self.benchmark_returns: Optional[pd.Series] = None
        
        # Risk limits
        self.max_portfolio_var = 0.05  # 5% daily VaR limit
        self.max_position_weight = 0.20  # 20% max position size
        self.max_correlation = 0.80  # 80% max correlation
        self.max_sector_concentration = 0.40  # 40% max sector allocation
        self.min_diversification_ratio = 0.70  # 70% min diversification
        
        # Risk calculations cache
        self.last_risk_calculation = None
        self.portfolio_risk_cache = None
        self.position_risks_cache = {}
        
        # Stress testing scenarios
        self.stress_scenarios = self._initialize_stress_scenarios()
        
        logger.info("PortfolioRiskManager initialized")
    
    def _initialize_stress_scenarios(self) -> List[StressTestScenario]:
        """Initialize predefined stress testing scenarios"""
        scenarios = [
            StressTestScenario(
                name="2008_Crisis",
                description="2008 Financial Crisis scenario",
                market_shocks={
                    'BTC/USDT': -0.50,  # 50% drop
                    'ETH/USDT': -0.60,  # 60% drop
                    'SOL/USDT': -0.70,  # 70% drop
                    'XRP/USDT': -0.55,  # 55% drop
                },
                correlation_multiplier=1.5,  # Correlations spike in crisis
                volatility_multiplier=2.0
            ),
            StressTestScenario(
                name="Flash_Crash",
                description="Flash crash scenario",
                market_shocks={
                    'BTC/USDT': -0.30,
                    'ETH/USDT': -0.35,
                    'SOL/USDT': -0.40,
                    'XRP/USDT': -0.32,
                },
                correlation_multiplier=2.0,
                volatility_multiplier=3.0
            ),
            StressTestScenario(
                name="Crypto_Winter",
                description="Extended bear market",
                market_shocks={
                    'BTC/USDT': -0.80,  # 80% drop over time
                    'ETH/USDT': -0.85,
                    'SOL/USDT': -0.90,
                    'XRP/USDT': -0.75,
                },
                correlation_multiplier=1.2,
                volatility_multiplier=1.5
            ),
            StressTestScenario(
                name="Regulatory_Ban",
                description="Major regulatory restriction",
                market_shocks={
                    'BTC/USDT': -0.40,
                    'ETH/USDT': -0.45,
                    'SOL/USDT': -0.50,
                    'XRP/USDT': -0.60,  # More regulatory sensitive
                },
                correlation_multiplier=1.8,
                volatility_multiplier=2.5
            )
        ]
        
        return scenarios
    
    def update_positions(self, positions: Dict[str, Dict]):
        """
        Update current portfolio positions.
        
        Args:
            positions: Dict mapping symbols to position info
                Example: {
                    'BTC/USDT': {
                        'size': 0.5,
                        'entry_price': 45000,
                        'current_price': 47000,
                        'market_value': 23500,
                        'unrealized_pnl': 1000
                    }
                }
        """
        self.positions = positions.copy()
        self._invalidate_cache()
        logger.debug(f"Updated positions for {len(positions)} symbols")
    
    def update_price_history(self, symbol: str, prices: pd.Series):
        """Update price history for a symbol"""
        # Keep only required lookback period
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days * 1.5)
        
        if isinstance(prices.index[0], str):
            prices.index = pd.to_datetime(prices.index)
        
        # Filter to recent data
        recent_prices = prices[prices.index >= cutoff_date]
        self.price_history[symbol] = recent_prices
        
        # Calculate returns
        returns = recent_prices.pct_change().dropna()
        self.returns_history[symbol] = returns
        
        self._invalidate_cache()
    
    def set_benchmark(self, benchmark_returns: pd.Series):
        """Set benchmark returns for beta calculation"""
        if isinstance(benchmark_returns.index[0], str):
            benchmark_returns.index = pd.to_datetime(benchmark_returns.index)
        
        self.benchmark_returns = benchmark_returns
        self._invalidate_cache()
    
    def _invalidate_cache(self):
        """Invalidate cached risk calculations"""
        self.last_risk_calculation = None
        self.portfolio_risk_cache = None
        self.position_risks_cache = {}
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95,
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk using multiple methods.
        
        Args:
            returns: Return series
            confidence_level: Confidence level (0.95 for 95% VaR)
            method: 'historical', 'parametric', or 'monte_carlo'
            
        Returns:
            VaR value (positive number representing potential loss)
        """
        if len(returns) < 30:
            return 0.0
        
        returns = returns.dropna()
        
        if method == 'historical':
            # Historical simulation
            var = -np.percentile(returns, (1 - confidence_level) * 100)
            
        elif method == 'parametric':
            # Parametric VaR assuming normal distribution
            mean_return = returns.mean()
            std_return = returns.std()
            z_score = stats.norm.ppf(confidence_level)
            var = -(mean_return - z_score * std_return)
            
        elif method == 'monte_carlo':
            # Monte Carlo simulation
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Generate random scenarios
            np.random.seed(42)
            simulated_returns = np.random.normal(mean_return, std_return, 10000)
            var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
            
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        return max(var, 0.0)  # VaR should be positive
    
    def calculate_expected_shortfall(self, returns: pd.Series, 
                                   confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        if len(returns) < 30:
            return 0.0
        
        returns = returns.dropna()
        cutoff = np.percentile(returns, (1 - confidence_level) * 100)
        tail_returns = returns[returns <= cutoff]
        
        if len(tail_returns) == 0:
            return 0.0
        
        return -tail_returns.mean()
    
    def calculate_position_risk(self, symbol: str) -> Optional[PositionRisk]:
        """Calculate risk metrics for individual position"""
        if symbol not in self.positions or symbol not in self.returns_history:
            return None
        
        position = self.positions[symbol]
        returns = self.returns_history[symbol]
        
        if len(returns) < 30:
            return None
        
        # Basic metrics
        market_value = position['market_value']
        total_portfolio_value = sum(pos['market_value'] for pos in self.positions.values())
        portfolio_weight = market_value / total_portfolio_value if total_portfolio_value > 0 else 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized
        var_95 = self.calculate_var(returns, 0.95) * market_value
        var_99 = self.calculate_var(returns, 0.99) * market_value
        
        # Performance metrics
        mean_return = returns.mean() * 252  # Annualized
        sharpe_ratio = (mean_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Beta calculation
        beta = 0.0
        if self.benchmark_returns is not None:
            # Align returns with benchmark
            aligned_returns, aligned_benchmark = returns.align(self.benchmark_returns, join='inner')
            if len(aligned_returns) > 30:
                covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                benchmark_variance = np.var(aligned_benchmark)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Maximum drawdown
        prices = self.price_history.get(symbol, pd.Series())
        if len(prices) > 0:
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())
        else:
            max_drawdown = 0.0
        
        # Correlation to portfolio
        portfolio_returns = self._calculate_portfolio_returns()
        correlation_to_portfolio = 0.0
        if portfolio_returns is not None and len(portfolio_returns) > 30:
            aligned_pos, aligned_port = returns.align(portfolio_returns, join='inner')
            if len(aligned_pos) > 10:
                correlation_to_portfolio = np.corrcoef(aligned_pos, aligned_port)[0, 1]
                if np.isnan(correlation_to_portfolio):
                    correlation_to_portfolio = 0.0
        
        return PositionRisk(
            symbol=symbol,
            position_size=position.get('size', 0.0),
            market_value=market_value,
            portfolio_weight=portfolio_weight,
            daily_var_95=var_95,
            daily_var_99=var_99,
            beta=beta,
            volatility=volatility,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            correlation_to_portfolio=correlation_to_portfolio
        )
    
    def _calculate_portfolio_returns(self) -> Optional[pd.Series]:
        """Calculate portfolio returns based on position weights"""
        if not self.positions or not self.returns_history:
            return None
        
        # Get total portfolio value
        total_value = sum(pos['market_value'] for pos in self.positions.values())
        if total_value <= 0:
            return None
        
        # Calculate weights
        weights = {}
        for symbol, position in self.positions.items():
            weights[symbol] = position['market_value'] / total_value
        
        # Get aligned returns
        returns_df = pd.DataFrame()
        for symbol in weights.keys():
            if symbol in self.returns_history:
                returns_df[symbol] = self.returns_history[symbol]
        
        if returns_df.empty:
            return None
        
        # Calculate weighted portfolio returns
        portfolio_returns = pd.Series(0.0, index=returns_df.index)
        for symbol, weight in weights.items():
            if symbol in returns_df.columns:
                portfolio_returns += returns_df[symbol] * weight
        
        return portfolio_returns.dropna()
    
    def calculate_portfolio_risk(self, use_cache: bool = True) -> Optional[PortfolioRisk]:
        """Calculate comprehensive portfolio risk metrics"""
        if use_cache and self.portfolio_risk_cache is not None:
            return self.portfolio_risk_cache
        
        if not self.positions:
            return None
        
        # Portfolio returns
        portfolio_returns = self._calculate_portfolio_returns()
        if portfolio_returns is None or len(portfolio_returns) < 30:
            return None
        
        # Basic portfolio metrics
        total_value = sum(pos['market_value'] for pos in self.positions.values())
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # VaR calculations
        var_95 = self.calculate_var(portfolio_returns, 0.95) * total_value
        var_99 = self.calculate_var(portfolio_returns, 0.99) * total_value
        expected_shortfall = self.calculate_expected_shortfall(portfolio_returns, 0.95) * total_value
        
        # Performance metrics
        mean_return = portfolio_returns.mean() * 252
        sharpe_ratio = (mean_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (mean_return - self.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        # Beta calculation
        portfolio_beta = 0.0
        if self.benchmark_returns is not None:
            aligned_port, aligned_bench = portfolio_returns.align(self.benchmark_returns, join='inner')
            if len(aligned_port) > 30:
                covariance = np.cov(aligned_port, aligned_bench)[0, 1]
                benchmark_variance = np.var(aligned_bench)
                portfolio_beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Concentration metrics
        weights = [pos['market_value'] / total_value for pos in self.positions.values()]
        correlation_concentration = self._calculate_correlation_concentration()
        sector_concentration = max(weights) if weights else 0.0  # Simplified sector concentration
        
        # Risk contribution analysis
        risk_contribution = self._calculate_risk_contribution(portfolio_returns)
        
        portfolio_risk = PortfolioRisk(
            total_value=total_value,
            daily_var_95=var_95,
            daily_var_99=var_99,
            expected_shortfall_95=expected_shortfall,
            portfolio_volatility=portfolio_volatility,
            portfolio_beta=portfolio_beta,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            correlation_concentration=correlation_concentration,
            sector_concentration=sector_concentration,
            risk_contribution=risk_contribution,
            timestamp=datetime.now()
        )
        
        # Cache result
        if use_cache:
            self.portfolio_risk_cache = portfolio_risk
            self.last_risk_calculation = datetime.now()
        
        return portfolio_risk
    
    def _calculate_correlation_concentration(self) -> float:
        """Calculate portfolio correlation concentration"""
        if len(self.returns_history) < 2:
            return 0.0
        
        # Create correlation matrix
        returns_df = pd.DataFrame()
        for symbol, returns in self.returns_history.items():
            if symbol in self.positions:
                returns_df[symbol] = returns
        
        if returns_df.shape[1] < 2:
            return 0.0
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        # Calculate average correlation (excluding diagonal)
        n = correlation_matrix.shape[0]
        if n < 2:
            return 0.0
        
        total_correlation = correlation_matrix.sum().sum() - n  # Exclude diagonal
        avg_correlation = total_correlation / (n * (n - 1))
        
        return abs(avg_correlation)
    
    def _calculate_risk_contribution(self, portfolio_returns: pd.Series) -> Dict[str, float]:
        """Calculate risk contribution of each position"""
        risk_contribution = {}
        total_value = sum(pos['market_value'] for pos in self.positions.values())
        
        if total_value <= 0:
            return risk_contribution
        
        portfolio_var = portfolio_returns.var()
        
        for symbol, position in self.positions.items():
            if symbol not in self.returns_history:
                risk_contribution[symbol] = 0.0
                continue
            
            weight = position['market_value'] / total_value
            asset_returns = self.returns_history[symbol]
            
            # Align returns
            aligned_asset, aligned_portfolio = asset_returns.align(portfolio_returns, join='inner')
            
            if len(aligned_asset) > 30:
                # Calculate marginal contribution to risk
                covariance = np.cov(aligned_asset, aligned_portfolio)[0, 1]
                marginal_risk = covariance / portfolio_var if portfolio_var > 0 else 0
                risk_contribution[symbol] = weight * marginal_risk
            else:
                risk_contribution[symbol] = 0.0
        
        # Normalize to sum to 1
        total_contribution = sum(risk_contribution.values())
        if total_contribution > 0:
            risk_contribution = {k: v/total_contribution for k, v in risk_contribution.items()}
        
        return risk_contribution
    
    def run_stress_test(self, scenario: StressTestScenario) -> StressTestResult:
        """Run stress test scenario on portfolio"""
        portfolio_pnl = 0.0
        position_pnls = {}
        
        for symbol, position in self.positions.items():
            current_value = position['market_value']
            
            # Apply shock if defined for this symbol
            if symbol in scenario.market_shocks:
                shock = scenario.market_shocks[symbol]
                position_pnl = current_value * shock
            else:
                # Apply average shock to non-specified positions
                avg_shock = np.mean(list(scenario.market_shocks.values()))
                position_pnl = current_value * avg_shock * 0.5  # Reduced impact
            
            position_pnls[symbol] = position_pnl
            portfolio_pnl += position_pnl
        
        # Calculate portfolio percentage
        total_value = sum(pos['market_value'] for pos in self.positions.values())
        portfolio_pnl_pct = portfolio_pnl / total_value if total_value > 0 else 0.0
        
        # Check if this breaches VaR
        current_risk = self.calculate_portfolio_risk()
        var_breach = False
        if current_risk:
            var_breach = abs(portfolio_pnl) > current_risk.daily_var_99
        
        # Find worst performing position
        max_loss_position = min(position_pnls.keys(), 
                              key=lambda x: position_pnls[x]) if position_pnls else ""
        
        # Estimate recovery time (simplified)
        if portfolio_pnl < 0:
            # Assume 1% daily recovery
            recovery_days = int(abs(portfolio_pnl_pct) / 0.01)
        else:
            recovery_days = 0
        
        return StressTestResult(
            scenario_name=scenario.name,
            portfolio_pnl=portfolio_pnl,
            portfolio_pnl_pct=portfolio_pnl_pct,
            position_pnls=position_pnls,
            var_breach=var_breach,
            max_loss_position=max_loss_position,
            recovery_days=recovery_days,
            timestamp=datetime.now()
        )
    
    def run_all_stress_tests(self) -> List[StressTestResult]:
        """Run all predefined stress test scenarios"""
        results = []
        
        for scenario in self.stress_scenarios:
            try:
                result = self.run_stress_test(scenario)
                results.append(result)
                logger.info(f"Stress test {scenario.name}: {result.portfolio_pnl_pct:.2%} portfolio impact")
            except Exception as e:
                logger.error(f"Error running stress test {scenario.name}: {str(e)}")
        
        return results
    
    def check_risk_limits(self) -> List[Dict]:
        """Check portfolio against risk limits"""
        violations = []
        
        # Calculate current portfolio risk
        portfolio_risk = self.calculate_portfolio_risk()
        if portfolio_risk is None:
            return violations
        
        # Portfolio VaR limit
        if portfolio_risk.daily_var_95 > (self.max_portfolio_var * portfolio_risk.total_value):
            violations.append({
                'type': 'portfolio_var',
                'description': f'Portfolio VaR {portfolio_risk.daily_var_95:.0f} exceeds limit {self.max_portfolio_var:.2%}',
                'severity': 'high',
                'current_value': portfolio_risk.daily_var_95 / portfolio_risk.total_value,
                'limit': self.max_portfolio_var
            })
        
        # Position size limits
        for symbol, position in self.positions.items():
            weight = position['market_value'] / portfolio_risk.total_value
            if weight > self.max_position_weight:
                violations.append({
                    'type': 'position_size',
                    'description': f'Position {symbol} weight {weight:.2%} exceeds limit {self.max_position_weight:.2%}',
                    'severity': 'medium',
                    'symbol': symbol,
                    'current_value': weight,
                    'limit': self.max_position_weight
                })
        
        # Correlation concentration
        if portfolio_risk.correlation_concentration > self.max_correlation:
            violations.append({
                'type': 'correlation',
                'description': f'Portfolio correlation {portfolio_risk.correlation_concentration:.2%} exceeds limit {self.max_correlation:.2%}',
                'severity': 'medium',
                'current_value': portfolio_risk.correlation_concentration,
                'limit': self.max_correlation
            })
        
        # Sector concentration
        if portfolio_risk.sector_concentration > self.max_sector_concentration:
            violations.append({
                'type': 'sector_concentration',
                'description': f'Sector concentration {portfolio_risk.sector_concentration:.2%} exceeds limit {self.max_sector_concentration:.2%}',
                'severity': 'medium',
                'current_value': portfolio_risk.sector_concentration,
                'limit': self.max_sector_concentration
            })
        
        return violations
    
    def calculate_optimal_position_size(self, symbol: str, expected_return: float,
                                      confidence: float, current_price: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion with risk adjustments.
        
        Args:
            symbol: Trading symbol
            expected_return: Expected return percentage
            confidence: Prediction confidence
            current_price: Current asset price
            
        Returns:
            Optimal position size as percentage of portfolio
        """
        if symbol not in self.returns_history:
            return 0.0
        
        returns = self.returns_history[symbol]
        if len(returns) < 30:
            return 0.0
        
        # Calculate asset volatility
        volatility = returns.std()
        
        # Kelly Criterion: f = (bp - q) / b
        # where b = odds, p = probability of win, q = probability of loss
        
        # Estimate win probability from confidence and historical win rate
        historical_positive = (returns > 0).mean()
        adjusted_win_prob = confidence * 0.5 + historical_positive * 0.5
        
        # Calculate Kelly fraction
        if volatility > 0 and adjusted_win_prob > 0.5:
            kelly_fraction = (expected_return * adjusted_win_prob - (1 - adjusted_win_prob)) / volatility
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        else:
            kelly_fraction = 0.0
        
        # Apply risk adjustments
        portfolio_risk = self.calculate_portfolio_risk()
        if portfolio_risk:
            # Reduce size if portfolio VaR is high
            var_adjustment = 1.0 - (portfolio_risk.daily_var_95 / (portfolio_risk.total_value * 0.05))
            var_adjustment = max(0.1, min(var_adjustment, 1.0))
            
            # Reduce size if correlation is high
            position_risk = self.calculate_position_risk(symbol)
            if position_risk and abs(position_risk.correlation_to_portfolio) > 0.7:
                correlation_adjustment = 1.0 - abs(position_risk.correlation_to_portfolio) * 0.5
            else:
                correlation_adjustment = 1.0
            
            kelly_fraction *= var_adjustment * correlation_adjustment
        
        # Apply maximum position size limit
        kelly_fraction = min(kelly_fraction, self.max_position_weight)
        
        return kelly_fraction
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        portfolio_risk = self.calculate_portfolio_risk()
        if portfolio_risk is None:
            return {'error': 'Insufficient data for risk calculation'}
        
        # Position summaries
        position_summaries = {}
        for symbol in self.positions.keys():
            pos_risk = self.calculate_position_risk(symbol)
            if pos_risk:
                position_summaries[symbol] = {
                    'weight': pos_risk.portfolio_weight,
                    'var_95': pos_risk.daily_var_95,
                    'volatility': pos_risk.volatility,
                    'sharpe': pos_risk.sharpe_ratio,
                    'correlation': pos_risk.correlation_to_portfolio
                }
        
        # Risk violations
        risk_violations = self.check_risk_limits()
        
        # Stress test summary
        stress_results = self.run_all_stress_tests()
        worst_stress = min(stress_results, key=lambda x: x.portfolio_pnl_pct) if stress_results else None
        
        return {
            'portfolio': {
                'total_value': portfolio_risk.total_value,
                'daily_var_95': portfolio_risk.daily_var_95,
                'daily_var_99': portfolio_risk.daily_var_99,
                'volatility': portfolio_risk.portfolio_volatility,
                'sharpe_ratio': portfolio_risk.sharpe_ratio,
                'max_drawdown': portfolio_risk.max_drawdown,
                'correlation_concentration': portfolio_risk.correlation_concentration
            },
            'positions': position_summaries,
            'risk_violations': risk_violations,
            'stress_test': {
                'worst_scenario': worst_stress.scenario_name if worst_stress else 'N/A',
                'worst_loss': worst_stress.portfolio_pnl_pct if worst_stress else 0.0,
                'scenarios_run': len(stress_results)
            },
            'timestamp': datetime.now()
        }
    
    def calculate_portfolio_value(self, portfolio: Dict[str, Dict], current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value given positions and current prices.
        
        Args:
            portfolio: Dict with symbol -> {'quantity': float, 'avg_price': float}
            current_prices: Dict with symbol -> current_price
            
        Returns:
            Total portfolio value in USD
        """
        try:
            total_value = 0.0
            
            for symbol, position in portfolio.items():
                if symbol in current_prices:
                    quantity = position.get('quantity', 0)
                    current_price = current_prices[symbol]
                    position_value = quantity * current_price
                    total_value += position_value
                    
                    logger.debug(f"Position {symbol}: {quantity} @ ${current_price} = ${position_value}")
            
            logger.info(f"Total portfolio value: ${total_value:,.2f}")
            return total_value
            
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return 0.0
    
    def calculate_position_size(self, symbol: str, price: float, **kwargs) -> float:
        """
        Simple wrapper for position sizing that delegates to calculate_optimal_position_size.
        
        Args:
            symbol: Trading symbol
            price: Current price
            **kwargs: Additional parameters for risk calculation
            
        Returns:
            Position size (quantity)
        """
        try:
            # Extract parameters from kwargs
            max_position_size = kwargs.get('max_position_size', 0.1)  # 10% default
            portfolio_value = kwargs.get('portfolio_value', 10000)   # $10k default
            volatility = kwargs.get('volatility', 0.02)              # 2% default
            expected_return = kwargs.get('expected_return', 0.001)   # 0.1% default
            
            # Calculate position value based on max position size
            max_position_value = portfolio_value * max_position_size
            
            # Simple position size calculation
            position_quantity = max_position_value / price
            
            # Apply volatility adjustment (reduce size for higher vol)
            vol_adjustment = min(1.0, 0.02 / max(volatility, 0.001))
            adjusted_quantity = position_quantity * vol_adjustment
            
            logger.debug(f"Position size for {symbol}: {adjusted_quantity:.6f} (value: ${adjusted_quantity * price:,.2f})")
            return adjusted_quantity
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0

# Factory function
def create_portfolio_risk_manager(lookback_days: int = 252) -> PortfolioRiskManager:
    """Create portfolio risk manager with configuration"""
    return PortfolioRiskManager(
        lookback_days=lookback_days,
        confidence_levels=[0.95, 0.99],
        risk_free_rate=0.02
    )


# Global instance
portfolio_risk_manager = None

def get_portfolio_risk_manager() -> Optional[PortfolioRiskManager]:
    """Get global portfolio risk manager instance"""
    return portfolio_risk_manager

def initialize_portfolio_risk_manager(**kwargs) -> PortfolioRiskManager:
    """Initialize global portfolio risk manager"""
    global portfolio_risk_manager
    portfolio_risk_manager = create_portfolio_risk_manager(**kwargs)
    return portfolio_risk_manager