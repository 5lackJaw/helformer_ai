"""
Advanced Execution Engine for Helformer Trading System
Sub-second execution with sophisticated order management and slippage protection
"""

import ccxt
import pandas as pd
import numpy as np
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from config_helformer import config

logger = logging.getLogger(__name__)

@dataclass
class ExecutionMetrics:
    """Container for execution quality metrics"""
    fill_rate: float
    avg_slippage: float
    avg_latency_ms: float
    execution_cost: float
    timestamp: datetime

@dataclass
class OrderResult:
    """Container for order execution results"""
    success: bool
    order_id: Optional[str]
    filled_quantity: float
    avg_fill_price: float
    slippage: float
    latency_ms: float
    total_cost: float
    error_message: Optional[str]

class AdvancedExecutionEngine:
    """
    Advanced order execution engine with:
    - Sub-second execution speed
    - Slippage prediction and protection
    - Multi-venue optimization
    - Order book analysis
    - Execution quality monitoring
    """
    
    def __init__(self, exchanges: List[ccxt.Exchange]):
        """
        Initialize advanced execution engine.
        
        Args:
            exchanges: List of configured exchange instances
        """
        self.exchanges = {exchange.id: exchange for exchange in exchanges}
        self.primary_exchange = exchanges[0] if exchanges else None
        
        # Execution monitoring
        self.execution_metrics: List[ExecutionMetrics] = []
        self.order_history: List[OrderResult] = []
        
        # Performance tracking
        self.latency_history: List[float] = []
        self.slippage_history: List[float] = []
        
        # Configuration
        self.max_slippage_tolerance = 0.005  # 0.5% maximum slippage
        self.execution_timeout = 10.0  # 10 seconds maximum execution time
        self.min_liquidity_usd = 1000  # Minimum liquidity requirement
        
        logger.info(f"Advanced Execution Engine initialized with {len(self.exchanges)} exchanges")
    
    async def execute_smart_order(self, 
                                 symbol: str,
                                 side: str,  # 'buy' or 'sell'
                                 quantity: float,
                                 urgency: str = 'normal',
                                 max_slippage: Optional[float] = None) -> OrderResult:
        """
        Execute order with intelligent routing and slippage protection.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            side: Order side ('buy' or 'sell')
            quantity: Quantity to trade
            urgency: Execution urgency ('low', 'normal', 'high', 'urgent')
            max_slippage: Maximum acceptable slippage
            
        Returns:
            OrderResult with execution details
        """
        start_time = time.time()
        
        try:
            # Set urgency-based parameters
            urgency_params = self._get_urgency_parameters(urgency)
            max_slippage = max_slippage or urgency_params['max_slippage']
            
            # Get market data for all exchanges
            market_data = await self._get_multi_exchange_data(symbol)
            
            if not market_data:
                return OrderResult(
                    success=False,
                    order_id=None,
                    filled_quantity=0.0,
                    avg_fill_price=0.0,
                    slippage=0.0,
                    latency_ms=(time.time() - start_time) * 1000,
                    total_cost=0.0,
                    error_message="No market data available"
                )
            
            # Analyze best execution venue
            best_venue = self._select_best_venue(market_data, side, quantity, urgency_params)
            
            if not best_venue:
                return OrderResult(
                    success=False,
                    order_id=None,
                    filled_quantity=0.0,
                    avg_fill_price=0.0,
                    slippage=0.0,
                    latency_ms=(time.time() - start_time) * 1000,
                    total_cost=0.0,
                    error_message="No suitable venue found"
                )
            
            # Predict slippage
            predicted_slippage = self._predict_slippage(best_venue['exchange'], symbol, side, quantity, market_data)
            
            if predicted_slippage > max_slippage:
                # Try order splitting if slippage too high
                return await self._execute_split_order(
                    symbol, side, quantity, urgency_params, max_slippage, start_time
                )
            
            # Execute order
            exchange = self.exchanges[best_venue['exchange']]
            order_result = await self._execute_single_order(
                exchange, symbol, side, quantity, urgency_params, start_time
            )
            
            # Update metrics
            self._update_execution_metrics(order_result)
            
            return order_result
            
        except Exception as e:
            logger.error(f"Error in smart order execution: {str(e)}")
            return OrderResult(
                success=False,
                order_id=None,
                filled_quantity=0.0,
                avg_fill_price=0.0,
                slippage=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                total_cost=0.0,
                error_message=str(e)
            )
    
    def _get_urgency_parameters(self, urgency: str) -> Dict:
        """Get execution parameters based on urgency level"""
        
        urgency_configs = {
            'low': {
                'max_slippage': 0.002,  # 0.2%
                'timeout': 30.0,
                'order_type': 'limit',
                'split_threshold': 0.1,  # Split if >10% of liquidity
                'patience': 'high'
            },
            'normal': {
                'max_slippage': 0.005,  # 0.5%
                'timeout': 10.0,
                'order_type': 'market',
                'split_threshold': 0.15,
                'patience': 'medium'
            },
            'high': {
                'max_slippage': 0.01,   # 1.0%
                'timeout': 5.0,
                'order_type': 'market',
                'split_threshold': 0.2,
                'patience': 'low'
            },
            'urgent': {
                'max_slippage': 0.02,   # 2.0%
                'timeout': 2.0,
                'order_type': 'market',
                'split_threshold': 0.3,
                'patience': 'none'
            }
        }
        
        return urgency_configs.get(urgency, urgency_configs['normal'])
    
    async def _get_multi_exchange_data(self, symbol: str) -> Dict:
        """Get market data from all available exchanges"""
        
        market_data = {}
        
        async def fetch_exchange_data(exchange_id: str, exchange: ccxt.Exchange):
            try:
                # Fetch ticker and order book
                ticker = exchange.fetch_ticker(symbol)
                order_book = exchange.fetch_order_book(symbol, limit=20)
                
                return exchange_id, {
                    'ticker': ticker,
                    'order_book': order_book,
                    'bid': ticker['bid'],
                    'ask': ticker['ask'],
                    'spread': ticker['ask'] - ticker['bid'],
                    'volume': ticker['baseVolume'],
                    'liquidity': self._calculate_liquidity(order_book),
                    'timestamp': time.time()
                }
            except Exception as e:
                logger.warning(f"Failed to fetch data from {exchange_id}: {str(e)}")
                return exchange_id, None
        
        # Fetch data from all exchanges concurrently
        tasks = [
            fetch_exchange_data(exchange_id, exchange) 
            for exchange_id, exchange in self.exchanges.items()
        ]
        
        # Use ThreadPoolExecutor for non-async exchange calls
        with ThreadPoolExecutor(max_workers=len(self.exchanges)) as executor:
            future_to_exchange = {
                executor.submit(self._fetch_sync_data, exchange_id, exchange, symbol): exchange_id
                for exchange_id, exchange in self.exchanges.items()
            }
            
            for future in as_completed(future_to_exchange, timeout=5.0):
                exchange_id = future_to_exchange[future]
                try:
                    data = future.result()
                    if data:
                        market_data[exchange_id] = data
                except Exception as e:
                    logger.warning(f"Failed to fetch data from {exchange_id}: {str(e)}")
        
        return market_data
    
    def _fetch_sync_data(self, exchange_id: str, exchange: ccxt.Exchange, symbol: str) -> Optional[Dict]:
        """Synchronous data fetching for thread pool"""
        try:
            ticker = exchange.fetch_ticker(symbol)
            order_book = exchange.fetch_order_book(symbol, limit=20)
            
            return {
                'ticker': ticker,
                'order_book': order_book,
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'spread': ticker['ask'] - ticker['bid'],
                'volume': ticker['baseVolume'],
                'liquidity': self._calculate_liquidity(order_book),
                'timestamp': time.time()
            }
        except Exception as e:
            logger.warning(f"Failed to fetch sync data from {exchange_id}: {str(e)}")
            return None
    
    def _calculate_liquidity(self, order_book: Dict) -> float:
        """Calculate available liquidity in USD"""
        
        total_liquidity = 0.0
        
        # Calculate bid side liquidity
        for bid_level in order_book['bids'][:10]:  # Top 10 levels
            price, quantity = bid_level
            total_liquidity += price * quantity
        
        # Calculate ask side liquidity
        for ask_level in order_book['asks'][:10]:  # Top 10 levels
            price, quantity = ask_level
            total_liquidity += price * quantity
        
        return total_liquidity
    
    def _select_best_venue(self, market_data: Dict, side: str, quantity: float, urgency_params: Dict) -> Optional[Dict]:
        """Select best exchange for execution"""
        
        best_venue = None
        best_score = -1
        
        for exchange_id, data in market_data.items():
            if not data:
                continue
            
            # Calculate execution score
            score = self._calculate_venue_score(data, side, quantity, urgency_params)
            
            if score > best_score:
                best_score = score
                best_venue = {
                    'exchange': exchange_id,
                    'data': data,
                    'score': score
                }
        
        return best_venue
    
    def _calculate_venue_score(self, data: Dict, side: str, quantity: float, urgency_params: Dict) -> float:
        """Calculate execution score for a venue"""
        
        # Base price score
        if side == 'buy':
            price_score = 1.0 / (data['ask'] + 1e-8)  # Lower ask = higher score
        else:
            price_score = data['bid']  # Higher bid = higher score
        
        # Liquidity score
        liquidity_score = min(data['liquidity'] / self.min_liquidity_usd, 1.0)
        
        # Spread score (lower spread = higher score)
        spread_pct = data['spread'] / ((data['bid'] + data['ask']) / 2)
        spread_score = max(0, 1.0 - spread_pct * 100)  # Penalize high spreads
        
        # Volume score
        volume_score = min(data['volume'] / 1000, 1.0)  # Normalize to 1000 BTC volume
        
        # Combined score with weights
        total_score = (
            price_score * 0.4 +
            liquidity_score * 0.3 +
            spread_score * 0.2 +
            volume_score * 0.1
        )
        
        return total_score
    
    def _predict_slippage(self, exchange_id: str, symbol: str, side: str, quantity: float, market_data: Dict) -> float:
        """Predict slippage for order execution"""
        
        data = market_data.get(exchange_id)
        if not data:
            return 0.02  # 2% default high slippage
        
        order_book = data['order_book']
        
        if side == 'buy':
            levels = order_book['asks']
            market_price = data['ask']
        else:
            levels = order_book['bids']
            market_price = data['bid']
        
        # Simulate order execution through order book
        remaining_quantity = quantity
        total_cost = 0.0
        
        for price, available_quantity in levels:
            if remaining_quantity <= 0:
                break
            
            executed_quantity = min(remaining_quantity, available_quantity)
            total_cost += executed_quantity * price
            remaining_quantity -= executed_quantity
        
        if remaining_quantity > 0:
            # Not enough liquidity - high slippage
            return 0.05  # 5% slippage
        
        avg_fill_price = total_cost / quantity
        slippage = abs(avg_fill_price - market_price) / market_price
        
        return slippage
    
    async def _execute_split_order(self, symbol: str, side: str, quantity: float, 
                                  urgency_params: Dict, max_slippage: float, start_time: float) -> OrderResult:
        """Execute large order by splitting into smaller chunks"""
        
        # Determine split size
        split_size = quantity * 0.3  # Split into 30% chunks
        splits = []
        remaining = quantity
        
        while remaining > 0:
            chunk_size = min(split_size, remaining)
            splits.append(chunk_size)
            remaining -= chunk_size
        
        logger.info(f"Splitting order into {len(splits)} chunks: {splits}")
        
        # Execute splits with delays
        total_filled = 0.0
        total_cost = 0.0
        max_slippage_seen = 0.0
        
        for i, chunk_size in enumerate(splits):
            # Add delay between chunks (except first)
            if i > 0:
                await asyncio.sleep(1.0)  # 1 second delay
            
            # Execute chunk
            chunk_result = await self.execute_smart_order(
                symbol, side, chunk_size, 'normal', max_slippage
            )
            
            if chunk_result.success:
                total_filled += chunk_result.filled_quantity
                total_cost += chunk_result.avg_fill_price * chunk_result.filled_quantity
                max_slippage_seen = max(max_slippage_seen, chunk_result.slippage)
            else:
                # If any chunk fails, return partial result
                break
        
        avg_fill_price = total_cost / total_filled if total_filled > 0 else 0.0
        
        return OrderResult(
            success=total_filled > 0,
            order_id=f"split_{int(time.time())}",
            filled_quantity=total_filled,
            avg_fill_price=avg_fill_price,
            slippage=max_slippage_seen,
            latency_ms=(time.time() - start_time) * 1000,
            total_cost=total_cost,
            error_message=None if total_filled > 0 else "Split order failed"
        )
    
    async def _execute_single_order(self, exchange: ccxt.Exchange, symbol: str, side: str, 
                                   quantity: float, urgency_params: Dict, start_time: float) -> OrderResult:
        """Execute single order on specified exchange"""
        
        try:
            order_start = time.time()
            
            # Execute market order for speed
            if side == 'buy':
                order = exchange.create_market_buy_order(symbol, quantity)
            else:
                order = exchange.create_market_sell_order(symbol, quantity)
            
            if not order or not order.get('id'):
                raise Exception("Order execution failed - no order ID returned")
            
            # Get order details
            order_details = exchange.fetch_order(order['id'], symbol)
            
            # Calculate metrics
            filled_quantity = order_details.get('filled', 0.0)
            avg_fill_price = order_details.get('average', 0.0)
            
            # Get current market price for slippage calculation
            ticker = exchange.fetch_ticker(symbol)
            market_price = ticker['ask'] if side == 'buy' else ticker['bid']
            
            slippage = abs(avg_fill_price - market_price) / market_price if market_price > 0 else 0.0
            latency = (time.time() - order_start) * 1000
            
            # Calculate total cost including fees
            fee = order_details.get('fee', {}).get('cost', 0.0) or 0.0
            total_cost = (avg_fill_price * filled_quantity) + fee
            
            return OrderResult(
                success=filled_quantity > 0,
                order_id=order['id'],
                filled_quantity=filled_quantity,
                avg_fill_price=avg_fill_price,
                slippage=slippage,
                latency_ms=latency,
                total_cost=total_cost,
                error_message=None
            )
            
        except Exception as e:
            return OrderResult(
                success=False,
                order_id=None,
                filled_quantity=0.0,
                avg_fill_price=0.0,
                slippage=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                total_cost=0.0,
                error_message=str(e)
            )
    
    def _update_execution_metrics(self, order_result: OrderResult):
        """Update execution quality metrics"""
        
        if order_result.success:
            self.latency_history.append(order_result.latency_ms)
            self.slippage_history.append(order_result.slippage)
            
            # Keep only last 100 orders
            if len(self.latency_history) > 100:
                self.latency_history = self.latency_history[-100:]
            if len(self.slippage_history) > 100:
                self.slippage_history = self.slippage_history[-100:]
        
        self.order_history.append(order_result)
        if len(self.order_history) > 1000:
            self.order_history = self.order_history[-1000:]
    
    def get_execution_metrics(self) -> ExecutionMetrics:
        """Get current execution quality metrics"""
        
        recent_orders = [o for o in self.order_history if o.success][-50:]  # Last 50 successful orders
        
        if not recent_orders:
            return ExecutionMetrics(
                fill_rate=0.0,
                avg_slippage=0.0,
                avg_latency_ms=0.0,
                execution_cost=0.0,
                timestamp=datetime.now()
            )
        
        fill_rate = len(recent_orders) / max(len(self.order_history[-50:]), 1)
        avg_slippage = np.mean([o.slippage for o in recent_orders])
        avg_latency = np.mean([o.latency_ms for o in recent_orders])
        avg_cost = np.mean([o.total_cost for o in recent_orders])
        
        return ExecutionMetrics(
            fill_rate=fill_rate,
            avg_slippage=avg_slippage,
            avg_latency_ms=avg_latency,
            execution_cost=avg_cost,
            timestamp=datetime.now()
        )
    
    def monitor_execution_quality(self) -> Dict:
        """Monitor and report execution quality"""
        
        metrics = self.get_execution_metrics()
        
        # Quality assessment
        quality_score = 100.0
        issues = []
        
        if metrics.fill_rate < 0.95:
            quality_score -= (0.95 - metrics.fill_rate) * 100
            issues.append(f"Low fill rate: {metrics.fill_rate:.1%}")
        
        if metrics.avg_slippage > 0.01:  # >1% slippage
            quality_score -= metrics.avg_slippage * 1000
            issues.append(f"High slippage: {metrics.avg_slippage:.2%}")
        
        if metrics.avg_latency_ms > 5000:  # >5 seconds
            quality_score -= (metrics.avg_latency_ms - 5000) / 100
            issues.append(f"High latency: {metrics.avg_latency_ms:.0f}ms")
        
        quality_score = max(0, quality_score)
        
        return {
            'metrics': metrics,
            'quality_score': quality_score,
            'issues': issues,
            'status': 'excellent' if quality_score > 90 else 'good' if quality_score > 70 else 'poor'
        }

# Convenience function for creating execution engine
def create_execution_engine(exchanges: List[ccxt.Exchange]) -> AdvancedExecutionEngine:
    """Create and configure advanced execution engine"""
    return AdvancedExecutionEngine(exchanges)