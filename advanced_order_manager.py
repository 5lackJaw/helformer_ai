"""
Advanced Order Management System for Helformer
Implements TWAP, VWAP, Iceberg, and other advanced order types
"""

import asyncio
import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from config_helformer import config

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Advanced order types supported by the system."""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"              # Time-Weighted Average Price
    VWAP = "vwap"              # Volume-Weighted Average Price
    ICEBERG = "iceberg"        # Large order split into smaller pieces
    POV = "pov"                # Percentage of Volume
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    ARRIVAL_PRICE = "arrival_price"

class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "pending"
    ACTIVE = "active"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"

@dataclass
class OrderSlice:
    """Individual order slice for advanced order types."""
    slice_id: str
    parent_order_id: str
    symbol: str
    side: str
    quantity: float
    price: Optional[float]
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    exchange_order_id: Optional[str] = None

@dataclass
class AdvancedOrder:
    """Advanced order with execution strategy."""
    order_id: str
    symbol: str
    side: str
    total_quantity: float
    order_type: OrderType
    status: OrderStatus = OrderStatus.PENDING
    
    # Execution parameters
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_participation_rate: float = 0.1  # Max 10% of volume
    min_slice_size: float = 0.001
    max_slice_size: Optional[float] = None
    
    # Execution tracking
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    total_fees: float = 0.0
    slices: List[OrderSlice] = None
    
    # Performance metrics
    arrival_price: Optional[float] = None
    implementation_shortfall: float = 0.0
    participation_rate: float = 0.0
    
    def __post_init__(self):
        if self.slices is None:
            self.slices = []

class AdvancedOrderManager:
    """
    Advanced order management system supporting institutional-grade order types.
    """
    
    def __init__(self, exchange: ccxt.Exchange):
        """
        Initialize advanced order manager.
        
        Args:
            exchange: CCXT exchange instance
        """
        self.exchange = exchange
        self.active_orders: Dict[str, AdvancedOrder] = {}
        self.order_history: List[AdvancedOrder] = []
        self.market_data_cache = {}
        self.volume_profiles = {}
        
        # Execution settings
        self.slice_interval_seconds = 30  # Execute slice every 30 seconds
        self.market_data_update_interval = 5  # Update market data every 5 seconds
        
        logger.info("Advanced Order Manager initialized")
    
    async def submit_advanced_order(self,
                                  symbol: str,
                                  side: str,
                                  quantity: float,
                                  order_type: OrderType,
                                  **kwargs) -> str:
        """
        Submit an advanced order for execution.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Total quantity to trade
            order_type: Type of advanced order
            **kwargs: Additional parameters specific to order type
            
        Returns:
            Order ID
        """
        order_id = f"{symbol}_{order_type.value}_{int(time.time() * 1000)}"
        
        # Get current market price for arrival price calculation
        ticker = await self._get_ticker(symbol)
        arrival_price = ticker['last'] if ticker else None
        
        # Create advanced order
        order = AdvancedOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            order_type=order_type,
            arrival_price=arrival_price,
            start_time=datetime.now(),
            **{k: v for k, v in kwargs.items() if hasattr(AdvancedOrder, k)}
        )
        
        # Set order-specific parameters
        await self._configure_order_parameters(order, **kwargs)
        
        # Add to active orders
        self.active_orders[order_id] = order
        
        # Start execution
        asyncio.create_task(self._execute_advanced_order(order))
        
        logger.info(f"Advanced order submitted: {order_id} ({order_type.value})")
        return order_id
    
    async def _configure_order_parameters(self, order: AdvancedOrder, **kwargs):
        """Configure order-specific parameters based on order type."""
        
        if order.order_type == OrderType.TWAP:
            # Time-Weighted Average Price parameters
            duration_hours = kwargs.get('duration_hours', 1.0)
            order.end_time = order.start_time + timedelta(hours=duration_hours)
            order.max_participation_rate = kwargs.get('max_participation_rate', 0.2)
            
        elif order.order_type == OrderType.VWAP:
            # Volume-Weighted Average Price parameters
            order.max_participation_rate = kwargs.get('max_participation_rate', 0.15)
            # VWAP typically executes over a shorter period
            duration_minutes = kwargs.get('duration_minutes', 30)
            order.end_time = order.start_time + timedelta(minutes=duration_minutes)
            
        elif order.order_type == OrderType.ICEBERG:
            # Iceberg order parameters
            order.max_slice_size = kwargs.get('slice_size', order.total_quantity * 0.1)
            order.min_slice_size = min(order.max_slice_size, kwargs.get('min_slice_size', 0.001))
            
        elif order.order_type == OrderType.POV:
            # Percentage of Volume parameters
            order.max_participation_rate = kwargs.get('participation_rate', 0.1)
            duration_hours = kwargs.get('duration_hours', 2.0)
            order.end_time = order.start_time + timedelta(hours=duration_hours)
            
        elif order.order_type == OrderType.IMPLEMENTATION_SHORTFALL:
            # Implementation Shortfall parameters
            urgency = kwargs.get('urgency', 'medium')  # low, medium, high
            urgency_params = {
                'low': {'duration_hours': 4.0, 'participation_rate': 0.05},
                'medium': {'duration_hours': 2.0, 'participation_rate': 0.1},
                'high': {'duration_hours': 0.5, 'participation_rate': 0.2}
            }
            params = urgency_params.get(urgency, urgency_params['medium'])
            order.end_time = order.start_time + timedelta(hours=params['duration_hours'])
            order.max_participation_rate = params['participation_rate']
            
        # Set default end time if not specified
        if order.end_time is None:
            order.end_time = order.start_time + timedelta(hours=1.0)
    
    async def _execute_advanced_order(self, order: AdvancedOrder):
        """Execute an advanced order using the specified strategy."""
        
        try:
            order.status = OrderStatus.ACTIVE
            logger.info(f"Starting execution of {order.order_type.value} order {order.order_id}")
            
            while (order.filled_quantity < order.total_quantity and 
                   datetime.now() < order.end_time and 
                   order.status == OrderStatus.ACTIVE):
                
                # Calculate next slice
                slice_size = await self._calculate_slice_size(order)
                
                if slice_size > 0:
                    # Execute slice
                    await self._execute_order_slice(order, slice_size)
                
                # Wait before next slice
                await asyncio.sleep(self.slice_interval_seconds)
            
            # Finalize order
            if order.filled_quantity >= order.total_quantity * 0.99:  # 99% filled
                order.status = OrderStatus.FILLED
            elif order.filled_quantity > 0:
                order.status = OrderStatus.PARTIALLY_FILLED
            
            # Calculate final metrics
            await self._calculate_execution_metrics(order)
            
            # Move to history
            self.order_history.append(order)
            del self.active_orders[order.order_id]
            
            logger.info(f"Order {order.order_id} completed: {order.status.value}")
            
        except Exception as e:
            logger.error(f"Error executing order {order.order_id}: {str(e)}")
            order.status = OrderStatus.FAILED
            self.order_history.append(order)
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
    
    async def _calculate_slice_size(self, order: AdvancedOrder) -> float:
        """Calculate the size of the next order slice."""
        
        remaining_quantity = order.total_quantity - order.filled_quantity
        if remaining_quantity <= 0:
            return 0
        
        if order.order_type == OrderType.TWAP:
            # Time-based slicing
            total_duration = (order.end_time - order.start_time).total_seconds()
            elapsed_time = (datetime.now() - order.start_time).total_seconds()
            remaining_time = max(1, total_duration - elapsed_time)
            
            # Target slice size based on remaining time
            target_slices_remaining = max(1, remaining_time / self.slice_interval_seconds)
            slice_size = remaining_quantity / target_slices_remaining
            
        elif order.order_type == OrderType.VWAP:
            # Volume-based slicing using historical volume profile
            volume_profile = await self._get_volume_profile(order.symbol)
            current_hour = datetime.now().hour
            expected_volume_pct = volume_profile.get(current_hour, 1/24)  # Default to uniform
            
            # Get recent volume data
            recent_volume = await self._get_recent_volume(order.symbol)
            target_participation = min(order.max_participation_rate, 0.2)
            
            slice_size = recent_volume * target_participation * expected_volume_pct
            
        elif order.order_type == OrderType.ICEBERG:
            # Fixed slice size for iceberg orders
            slice_size = min(order.max_slice_size, remaining_quantity)
            
        elif order.order_type == OrderType.POV:
            # Percentage of volume
            recent_volume = await self._get_recent_volume(order.symbol)
            slice_size = recent_volume * order.max_participation_rate
            
        else:
            # Default: equal time slicing
            total_duration = (order.end_time - order.start_time).total_seconds()
            elapsed_time = (datetime.now() - order.start_time).total_seconds()
            remaining_time = max(1, total_duration - elapsed_time)
            
            target_slices_remaining = max(1, remaining_time / self.slice_interval_seconds)
            slice_size = remaining_quantity / target_slices_remaining
        
        # Apply size constraints
        slice_size = max(order.min_slice_size, slice_size)
        slice_size = min(remaining_quantity, slice_size)
        
        return slice_size
    
    async def _execute_order_slice(self, order: AdvancedOrder, slice_size: float):
        """Execute a single order slice."""
        
        try:
            # Create slice
            slice_id = f"{order.order_id}_slice_{len(order.slices)}"
            order_slice = OrderSlice(
                slice_id=slice_id,
                parent_order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=slice_size,
                price=None,  # Market order for now
                timestamp=datetime.now()
            )
            
            # Determine order price (for now, use market orders)
            ticker = await self._get_ticker(order.symbol)
            if not ticker:
                logger.warning(f"No ticker data for {order.symbol}, skipping slice")
                return
            
            # Execute order on exchange
            try:
                if order.side == 'buy':
                    result = self.exchange.create_market_buy_order(
                        order.symbol, slice_size
                    )
                else:
                    result = self.exchange.create_market_sell_order(
                        order.symbol, slice_size
                    )
                
                # Update slice with execution results
                order_slice.exchange_order_id = result.get('id')
                order_slice.filled_quantity = result.get('filled', 0)
                order_slice.avg_fill_price = result.get('average', ticker['last'])
                order_slice.status = OrderStatus.FILLED if order_slice.filled_quantity == slice_size else OrderStatus.PARTIALLY_FILLED
                
                # Update parent order
                order.filled_quantity += order_slice.filled_quantity
                order.total_fees += result.get('fee', {}).get('cost', 0)
                
                # Update average fill price
                if order.filled_quantity > 0:
                    total_value = order.avg_fill_price * (order.filled_quantity - order_slice.filled_quantity)
                    total_value += order_slice.avg_fill_price * order_slice.filled_quantity
                    order.avg_fill_price = total_value / order.filled_quantity
                
                logger.debug(f"Slice executed: {slice_size} @ {order_slice.avg_fill_price}")
                
            except Exception as e:
                logger.error(f"Failed to execute slice: {str(e)}")
                order_slice.status = OrderStatus.FAILED
            
            order.slices.append(order_slice)
            
        except Exception as e:
            logger.error(f"Error in slice execution: {str(e)}")
    
    async def _get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get current ticker data with caching."""
        try:
            # Check cache first
            cache_key = f"ticker_{symbol}"
            if (cache_key in self.market_data_cache and 
                time.time() - self.market_data_cache[cache_key]['timestamp'] < self.market_data_update_interval):
                return self.market_data_cache[cache_key]['data']
            
            # Fetch fresh data
            ticker = self.exchange.fetch_ticker(symbol)
            
            # Cache the result
            self.market_data_cache[cache_key] = {
                'data': ticker,
                'timestamp': time.time()
            }
            
            return ticker
            
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {str(e)}")
            return None
    
    async def _get_recent_volume(self, symbol: str, minutes: int = 5) -> float:
        """Get recent trading volume."""
        try:
            # Fetch recent trades or use ticker volume
            ticker = await self._get_ticker(symbol)
            if ticker and 'quoteVolume' in ticker:
                # Estimate volume for the specified period
                daily_volume = ticker['quoteVolume']
                period_volume = daily_volume * (minutes / (24 * 60))  # Rough estimate
                return period_volume
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting recent volume for {symbol}: {str(e)}")
            return 0.0
    
    async def _get_volume_profile(self, symbol: str) -> Dict[int, float]:
        """Get historical volume profile by hour."""
        
        if symbol in self.volume_profiles:
            return self.volume_profiles[symbol]
        
        try:
            # Fetch historical data to build volume profile
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=168)  # 7 days
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['hour'] = df['timestamp'].dt.hour
            
            # Calculate average volume by hour
            hourly_volume = df.groupby('hour')['volume'].mean()
            total_volume = hourly_volume.sum()
            
            # Convert to percentages
            volume_profile = {}
            for hour in range(24):
                volume_profile[hour] = hourly_volume.get(hour, 0) / total_volume if total_volume > 0 else 1/24
            
            # Cache the profile
            self.volume_profiles[symbol] = volume_profile
            
            return volume_profile
            
        except Exception as e:
            logger.error(f"Error building volume profile for {symbol}: {str(e)}")
            # Return uniform distribution as fallback
            return {hour: 1/24 for hour in range(24)}
    
    async def _calculate_execution_metrics(self, order: AdvancedOrder):
        """Calculate execution quality metrics."""
        
        if order.filled_quantity == 0 or order.arrival_price is None:
            return
        
        # Implementation shortfall
        benchmark_value = order.arrival_price * order.total_quantity
        actual_value = order.avg_fill_price * order.filled_quantity
        
        if order.side == 'buy':
            order.implementation_shortfall = (actual_value - benchmark_value) / benchmark_value
        else:
            order.implementation_shortfall = (benchmark_value - actual_value) / benchmark_value
        
        # Participation rate
        if order.slices:
            execution_duration = (order.slices[-1].timestamp - order.slices[0].timestamp).total_seconds()
            if execution_duration > 0:
                # This is a simplified calculation
                order.participation_rate = order.filled_quantity / execution_duration * 60  # Per minute
    
    def get_order_status(self, order_id: str) -> Optional[AdvancedOrder]:
        """Get current status of an order."""
        if order_id in self.active_orders:
            return self.active_orders[order_id]
        
        # Check history
        for order in self.order_history:
            if order.order_id == order_id:
                return order
        
        return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        if order_id not in self.active_orders:
            return False
        
        try:
            order = self.active_orders[order_id]
            order.status = OrderStatus.CANCELLED
            
            # Cancel any pending slices on the exchange
            for slice_order in order.slices:
                if slice_order.exchange_order_id and slice_order.status == OrderStatus.ACTIVE:
                    try:
                        self.exchange.cancel_order(slice_order.exchange_order_id, order.symbol)
                        slice_order.status = OrderStatus.CANCELLED
                    except Exception as e:
                        logger.warning(f"Failed to cancel slice {slice_order.slice_id}: {str(e)}")
            
            # Move to history
            self.order_history.append(order)
            del self.active_orders[order_id]
            
            logger.info(f"Order {order_id} cancelled")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return False
    
    def get_execution_statistics(self) -> Dict:
        """Get overall execution statistics."""
        
        if not self.order_history:
            return {}
        
        filled_orders = [o for o in self.order_history if o.status == OrderStatus.FILLED]
        
        if not filled_orders:
            return {"message": "No completed orders"}
        
        # Calculate statistics
        implementation_shortfalls = [o.implementation_shortfall for o in filled_orders if o.implementation_shortfall != 0]
        participation_rates = [o.participation_rate for o in filled_orders if o.participation_rate > 0]
        
        stats = {
            "total_orders": len(self.order_history),
            "filled_orders": len(filled_orders),
            "fill_rate": len(filled_orders) / len(self.order_history),
            "avg_implementation_shortfall": np.mean(implementation_shortfalls) if implementation_shortfalls else 0,
            "avg_participation_rate": np.mean(participation_rates) if participation_rates else 0,
            "total_volume": sum(o.filled_quantity for o in filled_orders),
            "total_fees": sum(o.total_fees for o in filled_orders)
        }
        
        return stats

def create_advanced_order_manager(exchange: ccxt.Exchange) -> AdvancedOrderManager:
    """Factory function to create an advanced order manager."""
    return AdvancedOrderManager(exchange)

if __name__ == "__main__":
    # Test the advanced order manager
    async def test_order_manager():
        try:
            # Initialize exchange (use testnet for safety)
            exchange = ccxt.binance({
                'apiKey': 'your_api_key',
                'secret': 'your_secret',
                'sandbox': True,
                'enableRateLimit': True,
            })
            
            # Create order manager
            order_manager = AdvancedOrderManager(exchange)
            
            # Test TWAP order
            order_id = await order_manager.submit_advanced_order(
                symbol='BTC/USDT',
                side='buy',
                quantity=0.01,
                order_type=OrderType.TWAP,
                duration_hours=0.5,
                max_participation_rate=0.1
            )
            
            print(f"TWAP order submitted: {order_id}")
            
            # Monitor execution
            for i in range(10):
                await asyncio.sleep(30)
                order = order_manager.get_order_status(order_id)
                if order:
                    print(f"Order status: {order.status.value}, Filled: {order.filled_quantity}")
                    if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.FAILED]:
                        break
            
            # Get statistics
            stats = order_manager.get_execution_statistics()
            print(f"Execution statistics: {stats}")
            
        except Exception as e:
            print(f"Test failed: {str(e)}")
    
    # Run test
    asyncio.run(test_order_manager())