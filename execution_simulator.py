"""
Realistic Execution Simulator for Backtesting
Simulates realistic order execution including slippage, latency, partial fills, and market impact
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from enum import Enum
import random
from config_helformer import config

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

@dataclass
class OrderExecution:
    """Individual order execution record"""
    execution_id: str
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    execution_latency_ms: float
    slippage_bps: float
    fee_amount: float

@dataclass
class SimulatedOrder:
    """Simulated order with realistic execution characteristics"""
    order_id: str
    symbol: str
    side: str
    order_type: OrderType
    quantity: float
    target_price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    quantity_filled: float
    quantity_remaining: float
    average_fill_price: float
    total_fees: float
    executions: List[OrderExecution]
    creation_time: datetime
    completion_time: Optional[datetime]
    slippage_bps: float
    market_impact_bps: float

@dataclass
class MarketConditions:
    """Current market conditions affecting execution"""
    volatility: float
    spread_bps: float
    volume: float
    trend_strength: float
    liquidity_score: float
    timestamp: datetime

class ExecutionSimulator:
    """
    Realistic execution simulator for backtesting.
    
    Simulates:
    - Latency delays
    - Bid-ask spreads
    - Market impact
    - Slippage
    - Partial fills
    - Order rejections
    - Fee structures
    - Volume constraints
    """
    
    def __init__(self, 
                 base_latency_ms: float = 50.0,
                 base_spread_bps: float = 5.0,
                 market_impact_factor: float = 0.1,
                 max_market_participation: float = 0.10):
        """
        Initialize execution simulator.
        
        Args:
            base_latency_ms: Base execution latency in milliseconds
            base_spread_bps: Base bid-ask spread in basis points
            market_impact_factor: Market impact scaling factor
            max_market_participation: Maximum volume participation rate
        """
        self.base_latency_ms = base_latency_ms
        self.base_spread_bps = base_spread_bps
        self.market_impact_factor = market_impact_factor
        self.max_market_participation = max_market_participation
        
        # Execution tracking
        self.pending_orders: Dict[str, SimulatedOrder] = {}
        self.completed_orders: Dict[str, SimulatedOrder] = {}
        self.execution_history: List[OrderExecution] = []
        
        # Market data for simulation
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.current_conditions: Dict[str, MarketConditions] = {}
        
        # Fee structure (basis points)
        self.fee_structure = {
            'maker': 10,  # 0.1%
            'taker': 15,  # 0.15%
            'withdrawal': 0  # No withdrawal fees in simulation
        }
        
        # Simulation parameters
        self.rejection_probability = 0.02  # 2% order rejection rate
        self.partial_fill_probability = 0.15  # 15% partial fill rate
        self.min_fill_ratio = 0.30  # Minimum 30% fill on partial fills
        
        # Random seed for reproducible backtests
        self.random_seed = 42
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        logger.info("ExecutionSimulator initialized")
    
    def update_market_data(self, symbol: str, data: pd.DataFrame):
        """Update market data for execution simulation"""
        self.market_data[symbol] = data.copy()
        
        # Calculate current market conditions
        if len(data) >= 20:
            self._update_market_conditions(symbol, data)
    
    def _update_market_conditions(self, symbol: str, data: pd.DataFrame):
        """Calculate current market conditions"""
        # Use recent data for conditions
        recent_data = data.tail(20)
        
        # Calculate volatility
        returns = recent_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.1
        
        # Estimate spread from high-low
        spread_estimate = (recent_data['high'] - recent_data['low']) / recent_data['close']
        avg_spread_bps = spread_estimate.mean() * 10000  # Convert to basis points
        
        # Volume analysis
        avg_volume = recent_data['volume'].mean()
        current_volume = recent_data['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Trend strength
        short_ma = recent_data['close'].rolling(5).mean().iloc[-1]
        long_ma = recent_data['close'].rolling(15).mean().iloc[-1]
        trend_strength = abs((short_ma - long_ma) / long_ma) if long_ma > 0 else 0.0
        
        # Liquidity score (higher volume = better liquidity)
        liquidity_score = min(volume_ratio, 2.0) / 2.0  # Normalize to 0-1
        
        self.current_conditions[symbol] = MarketConditions(
            volatility=volatility,
            spread_bps=max(avg_spread_bps, self.base_spread_bps),
            volume=current_volume,
            trend_strength=trend_strength,
            liquidity_score=liquidity_score,
            timestamp=datetime.now()
        )
    
    def submit_order(self, symbol: str, side: str, quantity: float,
                    order_type: OrderType = OrderType.MARKET,
                    target_price: Optional[float] = None,
                    stop_price: Optional[float] = None) -> str:
        """
        Submit order for execution simulation.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            order_type: Order type
            target_price: Target price for limit orders
            stop_price: Stop price for stop orders
            
        Returns:
            Order ID string
        """
        order_id = f"SIM_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.pending_orders)}"
        
        # Check for immediate rejection
        if self._should_reject_order(symbol, quantity):
            logger.warning(f"Order {order_id} rejected: {symbol} {side} {quantity}")
            return order_id
        
        # Create simulated order
        order = SimulatedOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            target_price=target_price,
            stop_price=stop_price,
            status=OrderStatus.PENDING,
            quantity_filled=0.0,
            quantity_remaining=quantity,
            average_fill_price=0.0,
            total_fees=0.0,
            executions=[],
            creation_time=datetime.now(),
            completion_time=None,
            slippage_bps=0.0,
            market_impact_bps=0.0
        )
        
        self.pending_orders[order_id] = order
        
        # Immediate execution for market orders
        if order_type == OrderType.MARKET:
            self._execute_market_order(order_id)
        
        logger.debug(f"Order submitted: {order_id} - {symbol} {side} {quantity}")
        return order_id
    
    def _should_reject_order(self, symbol: str, quantity: float) -> bool:
        """Determine if order should be rejected"""
        # Random rejection based on probability
        if np.random.random() < self.rejection_probability:
            return True
        
        # Reject if no market data
        if symbol not in self.market_data:
            return True
        
        # Reject if quantity is too large relative to volume
        if symbol in self.current_conditions:
            conditions = self.current_conditions[symbol]
            if conditions.volume > 0:
                participation_rate = quantity / conditions.volume
                if participation_rate > self.max_market_participation * 2:  # 2x max for rejection
                    return True
        
        return False
    
    def _execute_market_order(self, order_id: str):
        """Execute market order with realistic simulation"""
        if order_id not in self.pending_orders:
            return
        
        order = self.pending_orders[order_id]
        symbol = order.symbol
        
        # Get current market price
        current_price = self._get_current_price(symbol)
        if current_price is None:
            order.status = OrderStatus.REJECTED
            return
        
        # Calculate execution latency
        latency_ms = self._calculate_execution_latency(symbol)
        
        # Calculate slippage and market impact
        slippage_bps, market_impact_bps = self._calculate_slippage_and_impact(order)
        
        # Determine if partial fill
        fill_ratio = 1.0
        if np.random.random() < self.partial_fill_probability:
            fill_ratio = np.random.uniform(self.min_fill_ratio, 1.0)
        
        # Calculate execution price
        total_impact_bps = slippage_bps + market_impact_bps
        if order.side == 'buy':
            execution_price = current_price * (1 + total_impact_bps / 10000)
        else:
            execution_price = current_price * (1 - total_impact_bps / 10000)
        
        # Execute order
        fill_quantity = order.quantity * fill_ratio
        self._execute_fill(order, fill_quantity, execution_price, latency_ms, 
                          slippage_bps, market_impact_bps)
        
        # Update order status
        if fill_ratio >= 1.0:
            order.status = OrderStatus.FILLED
            order.completion_time = datetime.now()
            self.completed_orders[order_id] = order
            del self.pending_orders[order_id]
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
    
    def _execute_fill(self, order: SimulatedOrder, quantity: float, price: float,
                     latency_ms: float, slippage_bps: float, market_impact_bps: float):
        """Execute a fill for an order"""
        # Calculate fees
        fee_rate = self.fee_structure['taker'] / 10000  # Convert to decimal
        fee_amount = quantity * price * fee_rate
        
        # Create execution record
        execution = OrderExecution(
            execution_id=f"EXEC_{len(self.execution_history)}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            timestamp=datetime.now(),
            execution_latency_ms=latency_ms,
            slippage_bps=slippage_bps,
            fee_amount=fee_amount
        )
        
        # Update order
        order.executions.append(execution)
        order.quantity_filled += quantity
        order.quantity_remaining -= quantity
        order.total_fees += fee_amount
        order.slippage_bps = slippage_bps
        order.market_impact_bps = market_impact_bps
        
        # Update average fill price
        total_value = sum(exec.quantity * exec.price for exec in order.executions)
        total_quantity = sum(exec.quantity for exec in order.executions)
        order.average_fill_price = total_value / total_quantity if total_quantity > 0 else 0.0
        
        # Add to execution history
        self.execution_history.append(execution)
        
        logger.debug(f"Fill executed: {execution.execution_id} - {quantity:.4f} @ {price:.4f}")
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol"""
        if symbol not in self.market_data:
            return None
        
        data = self.market_data[symbol]
        if len(data) == 0:
            return None
        
        return data['close'].iloc[-1]
    
    def _calculate_execution_latency(self, symbol: str) -> float:
        """Calculate execution latency in milliseconds"""
        base_latency = self.base_latency_ms
        
        # Add volatility-based latency
        if symbol in self.current_conditions:
            conditions = self.current_conditions[symbol]
            volatility_multiplier = 1.0 + (conditions.volatility * 2.0)
            liquidity_multiplier = 1.0 + (1.0 - conditions.liquidity_score)
            
            latency = base_latency * volatility_multiplier * liquidity_multiplier
        else:
            latency = base_latency
        
        # Add random component
        latency *= np.random.uniform(0.5, 2.0)
        
        return latency
    
    def _calculate_slippage_and_impact(self, order: SimulatedOrder) -> Tuple[float, float]:
        """Calculate slippage and market impact in basis points"""
        symbol = order.symbol
        
        # Base slippage from spread
        if symbol in self.current_conditions:
            conditions = self.current_conditions[symbol]
            base_slippage = conditions.spread_bps / 2  # Half spread as base slippage
            
            # Volatility impact
            volatility_impact = conditions.volatility * 100  # Convert to bps
            
            # Volume impact (market impact)
            volume_participation = order.quantity / conditions.volume if conditions.volume > 0 else 0.01
            market_impact = volume_participation * self.market_impact_factor * 1000  # To bps
            
            # Trend impact (higher in trending markets)
            trend_impact = conditions.trend_strength * 20  # Up to 20 bps
            
            # Liquidity impact (worse execution in low liquidity)
            liquidity_impact = (1.0 - conditions.liquidity_score) * 30  # Up to 30 bps
            
            total_slippage = base_slippage + volatility_impact + trend_impact + liquidity_impact
            
        else:
            # Default values if no market conditions
            total_slippage = self.base_spread_bps
            market_impact = 5.0  # 5 bps default
        
        # Add randomness
        slippage_noise = np.random.uniform(-0.5, 1.5)  # Mostly upward bias
        total_slippage *= (1.0 + slippage_noise * 0.3)
        
        market_impact_noise = np.random.uniform(0.5, 1.5)
        market_impact *= market_impact_noise
        
        # Ensure positive values
        total_slippage = max(total_slippage, 0.1)
        market_impact = max(market_impact, 0.1)
        
        return total_slippage, market_impact
    
    def get_order_status(self, order_id: str) -> Optional[SimulatedOrder]:
        """Get current status of an order"""
        if order_id in self.pending_orders:
            return self.pending_orders[order_id]
        elif order_id in self.completed_orders:
            return self.completed_orders[order_id]
        else:
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        if order_id not in self.pending_orders:
            return False
        
        order = self.pending_orders[order_id]
        order.status = OrderStatus.CANCELLED
        order.completion_time = datetime.now()
        
        self.completed_orders[order_id] = order
        del self.pending_orders[order_id]
        
        logger.debug(f"Order cancelled: {order_id}")
        return True
    
    def process_pending_orders(self, current_time: datetime):
        """Process pending limit and stop orders"""
        orders_to_execute = []
        
        for order_id, order in self.pending_orders.items():
            if order.order_type == OrderType.LIMIT:
                current_price = self._get_current_price(order.symbol)
                if current_price is not None:
                    # Check if limit order should execute
                    if ((order.side == 'buy' and current_price <= order.target_price) or
                        (order.side == 'sell' and current_price >= order.target_price)):
                        orders_to_execute.append(order_id)
            
            elif order.order_type == OrderType.STOP_LOSS:
                current_price = self._get_current_price(order.symbol)
                if current_price is not None:
                    # Check if stop loss should trigger
                    if ((order.side == 'sell' and current_price <= order.stop_price) or
                        (order.side == 'buy' and current_price >= order.stop_price)):
                        orders_to_execute.append(order_id)
        
        # Execute triggered orders
        for order_id in orders_to_execute:
            self._execute_market_order(order_id)
    
    def get_execution_statistics(self, symbol: Optional[str] = None) -> Dict:
        """Get execution statistics"""
        executions = self.execution_history
        
        if symbol:
            executions = [e for e in executions if e.symbol == symbol]
        
        if not executions:
            return {}
        
        latencies = [e.execution_latency_ms for e in executions]
        slippages = [e.slippage_bps for e in executions]
        fees = [e.fee_amount for e in executions]
        
        return {
            'total_executions': len(executions),
            'avg_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'avg_slippage_bps': np.mean(slippages),
            'median_slippage_bps': np.median(slippages),
            'total_fees': sum(fees),
            'avg_fee_per_execution': np.mean(fees),
            'symbols_traded': len(set(e.symbol for e in executions)),
            'buy_executions': len([e for e in executions if e.side == 'buy']),
            'sell_executions': len([e for e in executions if e.side == 'sell'])
        }
    
    def get_slippage_analysis(self) -> Dict:
        """Analyze slippage patterns"""
        if not self.execution_history:
            return {}
        
        # Group by symbol
        symbol_slippage = {}
        for execution in self.execution_history:
            if execution.symbol not in symbol_slippage:
                symbol_slippage[execution.symbol] = []
            symbol_slippage[execution.symbol].append(execution.slippage_bps)
        
        analysis = {}
        for symbol, slippages in symbol_slippage.items():
            analysis[symbol] = {
                'avg_slippage_bps': np.mean(slippages),
                'median_slippage_bps': np.median(slippages),
                'max_slippage_bps': max(slippages),
                'slippage_std': np.std(slippages),
                'executions_count': len(slippages)
            }
        
        return analysis
    
    def simulate_backtest_execution(self, trades_df: pd.DataFrame, 
                                   market_data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Simulate realistic execution for backtest trades.
        
        Args:
            trades_df: DataFrame with columns ['timestamp', 'symbol', 'side', 'quantity', 'price']
            market_data_dict: Dict mapping symbols to market data
            
        Returns:
            DataFrame with realistic execution results
        """
        # Update market data
        for symbol, data in market_data_dict.items():
            self.update_market_data(symbol, data)
        
        executed_trades = []
        
        for _, trade in trades_df.iterrows():
            symbol = trade['symbol']
            side = trade['side']
            quantity = trade['quantity']
            target_price = trade.get('price', None)
            
            # Submit order
            order_id = self.submit_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET,
                target_price=target_price
            )
            
            # Get execution result
            order = self.get_order_status(order_id)
            if order and order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                # Add realistic execution to results
                executed_trade = {
                    'original_timestamp': trade['timestamp'],
                    'execution_timestamp': order.completion_time or datetime.now(),
                    'symbol': symbol,
                    'side': side,
                    'target_quantity': quantity,
                    'executed_quantity': order.quantity_filled,
                    'target_price': target_price,
                    'executed_price': order.average_fill_price,
                    'slippage_bps': order.slippage_bps,
                    'market_impact_bps': order.market_impact_bps,
                    'fees': order.total_fees,
                    'latency_ms': np.mean([e.execution_latency_ms for e in order.executions]) if order.executions else 0,
                    'status': order.status.value
                }
                executed_trades.append(executed_trade)
        
        return pd.DataFrame(executed_trades)
    
    def simulate_market_order(self, symbol: str, side: str, amount: float, current_price: float) -> Dict:
        """
        Simulate execution of a market order with realistic market effects.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            side: 'buy' or 'sell'
            amount: Order quantity
            current_price: Current market price
            
        Returns:
            Dict with execution results including filled_price, slippage, fees
        """
        try:
            # Generate unique order ID
            order_id = f"sim_{symbol}_{side}_{int(datetime.now().timestamp())}"
            
            # Calculate slippage based on order size and market conditions
            # Market orders typically face slippage due to spread and market impact
            spread_bps = self.base_spread_bps
            
            # Add market impact based on order size (simplified)
            market_impact_bps = self.market_impact_factor * np.sqrt(amount) * 10
            
            # Total slippage
            total_slippage_bps = spread_bps + market_impact_bps
            
            # Calculate fill price (worse for market taker)
            if side.lower() == 'buy':
                slippage_factor = 1 + (total_slippage_bps / 10000)
                filled_price = current_price * slippage_factor
            else:  # sell
                slippage_factor = 1 - (total_slippage_bps / 10000)
                filled_price = current_price * slippage_factor
            
            # Calculate fees (market orders are taker orders)
            fee_rate = self.fee_structure['taker'] / 10000  # Convert bps to decimal
            notional_value = amount * filled_price
            fee_amount = notional_value * fee_rate
            
            # Simulate execution latency
            latency_ms = self._calculate_execution_latency(symbol) if symbol in self.current_conditions else self.base_latency_ms
            
            # Create execution record
            execution = OrderExecution(
                execution_id=f"exec_{order_id}",
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=amount,
                price=filled_price,
                timestamp=datetime.now(),
                execution_latency_ms=latency_ms,
                slippage_bps=total_slippage_bps,
                fee_amount=fee_amount
            )
            
            # Store execution
            self.execution_history.append(execution)
            
            # Return execution result
            result = {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': amount,
                'filled_price': round(filled_price, 2),
                'filled_quantity': amount,  # Market orders fill completely
                'slippage_bps': round(total_slippage_bps, 2),
                'fee_amount': round(fee_amount, 4),
                'latency_ms': round(latency_ms, 2),
                'status': 'filled',
                'timestamp': execution.timestamp
            }
            
            logger.debug(f"Market order simulated: {symbol} {side} {amount} @ ${filled_price:.2f} (slippage: {total_slippage_bps:.1f}bps)")
            return result
            
        except Exception as e:
            logger.error(f"Error simulating market order: {e}")
            return {
                'order_id': 'error',
                'status': 'rejected',
                'error': str(e),
                'filled_price': current_price,
                'filled_quantity': 0,
                'slippage_bps': 0,
                'fee_amount': 0,
                'latency_ms': 0
            }
    
    def reset_simulator(self):
        """Reset simulator state"""
        self.pending_orders.clear()
        self.completed_orders.clear()
        self.execution_history.clear()
        self.current_conditions.clear()
        
        # Reset random seed for reproducibility
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        logger.info("ExecutionSimulator reset")


# Factory function
def create_execution_simulator(latency_ms: float = 50.0,
                             spread_bps: float = 5.0) -> ExecutionSimulator:
    """Create execution simulator with configuration"""
    return ExecutionSimulator(
        base_latency_ms=latency_ms,
        base_spread_bps=spread_bps,
        market_impact_factor=0.1,
        max_market_participation=0.10
    )


# Global instance
execution_simulator = None

def get_execution_simulator() -> Optional[ExecutionSimulator]:
    """Get global execution simulator instance"""
    return execution_simulator

def initialize_execution_simulator(**kwargs) -> ExecutionSimulator:
    """Initialize global execution simulator"""
    global execution_simulator
    execution_simulator = create_execution_simulator(**kwargs)
    return execution_simulator