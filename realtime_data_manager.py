"""
Real-Time Data Streaming and WebSocket Manager for Helformer
Provides real-time market data feeds and order book management
"""

import asyncio
import websockets
import json
import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
import logging
import threading
import time
from config_helformer import config

logger = logging.getLogger(__name__)

@dataclass
class TickData:
    """Individual tick data point."""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    side: str  # 'buy' or 'sell'
    trade_id: Optional[str] = None

@dataclass
class OrderBookLevel:
    """Order book price level."""
    price: float
    size: float
    timestamp: datetime

@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot."""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    spread: float = 0.0

@dataclass
class BarData:
    """OHLCV bar data."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str

class RealTimeDataManager:
    """
    Real-time data streaming manager with WebSocket support.
    
    Manages multiple data feeds and provides callbacks for real-time processing.
    """
    
    def __init__(self, exchanges: List[ccxt.Exchange]):
        """
        Initialize real-time data manager.
        
        Args:
            exchanges: List of CCXT exchange instances
        """
        self.exchanges = {exchange.id: exchange for exchange in exchanges}
        self.primary_exchange = exchanges[0] if exchanges else None
        
        # Data storage
        self.tick_data: Dict[str, deque] = {}
        self.order_books: Dict[str, OrderBookSnapshot] = {}
        self.bar_data: Dict[str, Dict[str, deque]] = {}  # symbol -> timeframe -> bars
        
        # WebSocket connections
        self.ws_connections: Dict[str, Any] = {}
        self.ws_tasks: List[asyncio.Task] = []
        
        # Callbacks
        self.tick_callbacks: List[Callable[[TickData], None]] = []
        self.orderbook_callbacks: List[Callable[[OrderBookSnapshot], None]] = []
        self.bar_callbacks: List[Callable[[BarData], None]] = []
        
        # Configuration
        self.max_tick_history = 10000  # Keep last 10k ticks per symbol
        self.max_bar_history = 1000    # Keep last 1k bars per timeframe
        self.orderbook_depth = 20      # Top 20 levels
        
        # State tracking
        self.is_running = False
        self.subscribed_symbols = set()
        self.subscribed_timeframes = set()
        
        logger.info(f"Real-time data manager initialized with {len(self.exchanges)} exchanges")
    
    async def start(self):
        """Start real-time data streaming."""
        if self.is_running:
            logger.warning("Data manager already running")
            return
        
        self.is_running = True
        logger.info("Starting real-time data manager")
        
        # Start WebSocket connections for each exchange
        for exchange_id, exchange in self.exchanges.items():
            if hasattr(exchange, 'watch_ticker'):  # Check if exchange supports WebSocket
                task = asyncio.create_task(self._start_exchange_streams(exchange_id, exchange))
                self.ws_tasks.append(task)
        
        # Start bar aggregation task
        bar_task = asyncio.create_task(self._bar_aggregation_loop())
        self.ws_tasks.append(bar_task)
        
        logger.info("Real-time data streams started")
    
    async def stop(self):
        """Stop real-time data streaming."""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("Stopping real-time data manager")
        
        # Cancel all WebSocket tasks
        for task in self.ws_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.ws_tasks.clear()
        self.ws_connections.clear()
        
        logger.info("Real-time data streams stopped")
    
    async def subscribe_ticker(self, symbol: str, exchange_id: Optional[str] = None):
        """Subscribe to ticker updates for a symbol."""
        exchange = self._get_exchange(exchange_id)
        if not exchange:
            logger.error(f"Exchange {exchange_id} not found")
            return
        
        self.subscribed_symbols.add(symbol)
        
        # Initialize data structures
        if symbol not in self.tick_data:
            self.tick_data[symbol] = deque(maxlen=self.max_tick_history)
        
        logger.info(f"Subscribed to ticker for {symbol} on {exchange.id}")
    
    async def subscribe_orderbook(self, symbol: str, exchange_id: Optional[str] = None):
        """Subscribe to order book updates for a symbol."""
        exchange = self._get_exchange(exchange_id)
        if not exchange:
            logger.error(f"Exchange {exchange_id} not found")
            return
        
        self.subscribed_symbols.add(symbol)
        logger.info(f"Subscribed to order book for {symbol} on {exchange.id}")
    
    async def subscribe_bars(self, symbol: str, timeframe: str, exchange_id: Optional[str] = None):
        """Subscribe to bar data for a symbol and timeframe."""
        exchange = self._get_exchange(exchange_id)
        if not exchange:
            logger.error(f"Exchange {exchange_id} not found")
            return
        
        self.subscribed_symbols.add(symbol)
        self.subscribed_timeframes.add(timeframe)
        
        # Initialize data structures
        if symbol not in self.bar_data:
            self.bar_data[symbol] = {}
        if timeframe not in self.bar_data[symbol]:
            self.bar_data[symbol][timeframe] = deque(maxlen=self.max_bar_history)
        
        logger.info(f"Subscribed to {timeframe} bars for {symbol} on {exchange.id}")
    
    def add_tick_callback(self, callback: Callable[[TickData], None]):
        """Add callback for tick data updates."""
        self.tick_callbacks.append(callback)
    
    def add_orderbook_callback(self, callback: Callable[[OrderBookSnapshot], None]):
        """Add callback for order book updates."""
        self.orderbook_callbacks.append(callback)
    
    def add_bar_callback(self, callback: Callable[[BarData], None]):
        """Add callback for bar data updates."""
        self.bar_callbacks.append(callback)
    
    async def _start_exchange_streams(self, exchange_id: str, exchange: ccxt.Exchange):
        """Start WebSocket streams for an exchange."""
        try:
            while self.is_running:
                # Subscribe to all symbols for this exchange
                for symbol in self.subscribed_symbols:
                    try:
                        # Watch ticker
                        if hasattr(exchange, 'watch_ticker'):
                            ticker = await exchange.watch_ticker(symbol)
                            await self._process_ticker_update(symbol, ticker, exchange_id)
                        
                        # Watch order book
                        if hasattr(exchange, 'watch_order_book'):
                            orderbook = await exchange.watch_order_book(symbol, limit=self.orderbook_depth)
                            await self._process_orderbook_update(symbol, orderbook, exchange_id)
                        
                        # Watch trades
                        if hasattr(exchange, 'watch_trades'):
                            trades = await exchange.watch_trades(symbol)
                            for trade in trades:
                                await self._process_trade_update(symbol, trade, exchange_id)
                        
                    except Exception as e:
                        logger.warning(f"Error in {exchange_id} stream for {symbol}: {str(e)}")
                
                await asyncio.sleep(0.1)  # Small delay to prevent overwhelming
                
        except asyncio.CancelledError:
            logger.info(f"WebSocket stream cancelled for {exchange_id}")
        except Exception as e:
            logger.error(f"Error in {exchange_id} WebSocket stream: {str(e)}")
            if self.is_running:
                # Restart after delay
                await asyncio.sleep(5)
                asyncio.create_task(self._start_exchange_streams(exchange_id, exchange))
    
    async def _process_ticker_update(self, symbol: str, ticker: Dict, exchange_id: str):
        """Process ticker update."""
        try:
            if not ticker or 'last' not in ticker:
                return
            
            # Create tick data
            tick = TickData(
                symbol=symbol,
                timestamp=datetime.now(),
                price=float(ticker['last']),
                volume=float(ticker.get('baseVolume', 0)),
                side='unknown'  # Ticker doesn't specify side
            )
            
            # Store tick data
            if symbol in self.tick_data:
                self.tick_data[symbol].append(tick)
            
            # Trigger callbacks
            for callback in self.tick_callbacks:
                try:
                    callback(tick)
                except Exception as e:
                    logger.error(f"Error in tick callback: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error processing ticker for {symbol}: {str(e)}")
    
    async def _process_orderbook_update(self, symbol: str, orderbook: Dict, exchange_id: str):
        """Process order book update."""
        try:
            timestamp = datetime.now()
            
            # Convert to OrderBookLevel objects
            bids = []
            asks = []
            
            for price, size in orderbook.get('bids', []):
                bids.append(OrderBookLevel(
                    price=float(price),
                    size=float(size),
                    timestamp=timestamp
                ))
            
            for price, size in orderbook.get('asks', []):
                asks.append(OrderBookLevel(
                    price=float(price),
                    size=float(size),
                    timestamp=timestamp
                ))
            
            # Create order book snapshot
            snapshot = OrderBookSnapshot(
                symbol=symbol,
                timestamp=timestamp,
                bids=bids,
                asks=asks
            )
            
            # Calculate best bid/ask and spread
            if bids:
                snapshot.best_bid = bids[0].price
            if asks:
                snapshot.best_ask = asks[0].price
            if snapshot.best_bid and snapshot.best_ask:
                snapshot.spread = snapshot.best_ask - snapshot.best_bid
            
            # Store order book
            self.order_books[symbol] = snapshot
            
            # Trigger callbacks
            for callback in self.orderbook_callbacks:
                try:
                    callback(snapshot)
                except Exception as e:
                    logger.error(f"Error in orderbook callback: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error processing orderbook for {symbol}: {str(e)}")
    
    async def _process_trade_update(self, symbol: str, trade: Dict, exchange_id: str):
        """Process individual trade update."""
        try:
            tick = TickData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(trade.get('timestamp', time.time()) / 1000),
                price=float(trade['price']),
                volume=float(trade['amount']),
                side=trade.get('side', 'unknown'),
                trade_id=trade.get('id')
            )
            
            # Store tick data
            if symbol in self.tick_data:
                self.tick_data[symbol].append(tick)
            
            # Trigger callbacks
            for callback in self.tick_callbacks:
                try:
                    callback(tick)
                except Exception as e:
                    logger.error(f"Error in trade callback: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error processing trade for {symbol}: {str(e)}")
    
    async def _bar_aggregation_loop(self):
        """Background loop to aggregate tick data into bars."""
        last_bar_times = {}
        
        try:
            while self.is_running:
                current_time = datetime.now()
                
                for symbol in self.subscribed_symbols:
                    for timeframe in self.subscribed_timeframes:
                        try:
                            # Calculate bar interval
                            bar_interval = self._get_bar_interval_seconds(timeframe)
                            
                            # Check if it's time for a new bar
                            last_bar_key = f"{symbol}_{timeframe}"
                            last_bar_time = last_bar_times.get(last_bar_key, current_time)
                            
                            if (current_time - last_bar_time).total_seconds() >= bar_interval:
                                # Create new bar from tick data
                                bar = self._create_bar_from_ticks(symbol, timeframe, last_bar_time, current_time)
                                if bar:
                                    # Store bar
                                    if symbol in self.bar_data and timeframe in self.bar_data[symbol]:
                                        self.bar_data[symbol][timeframe].append(bar)
                                    
                                    # Trigger callbacks
                                    for callback in self.bar_callbacks:
                                        try:
                                            callback(bar)
                                        except Exception as e:
                                            logger.error(f"Error in bar callback: {str(e)}")
                                
                                last_bar_times[last_bar_key] = current_time
                                
                        except Exception as e:
                            logger.error(f"Error aggregating bars for {symbol} {timeframe}: {str(e)}")
                
                await asyncio.sleep(1)  # Check every second
                
        except asyncio.CancelledError:
            logger.info("Bar aggregation loop cancelled")
        except Exception as e:
            logger.error(f"Error in bar aggregation loop: {str(e)}")
    
    def _create_bar_from_ticks(self, symbol: str, timeframe: str, start_time: datetime, end_time: datetime) -> Optional[BarData]:
        """Create OHLCV bar from tick data."""
        if symbol not in self.tick_data or not self.tick_data[symbol]:
            return None
        
        # Filter ticks for the time period
        ticks = [
            tick for tick in self.tick_data[symbol] 
            if start_time <= tick.timestamp < end_time
        ]
        
        if not ticks:
            return None
        
        # Calculate OHLCV
        prices = [tick.price for tick in ticks]
        volumes = [tick.volume for tick in ticks]
        
        bar = BarData(
            symbol=symbol,
            timestamp=start_time,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=sum(volumes),
            timeframe=timeframe
        )
        
        return bar
    
    def _get_bar_interval_seconds(self, timeframe: str) -> int:
        """Convert timeframe string to seconds."""
        if timeframe.endswith('s'):
            return int(timeframe[:-1])
        elif timeframe.endswith('m'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 3600
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 86400
        else:
            return 60  # Default to 1 minute
    
    def _get_exchange(self, exchange_id: Optional[str] = None) -> Optional[ccxt.Exchange]:
        """Get exchange instance by ID."""
        if exchange_id is None:
            return self.primary_exchange
        return self.exchanges.get(exchange_id)
    
    def get_latest_tick(self, symbol: str) -> Optional[TickData]:
        """Get latest tick for a symbol."""
        if symbol in self.tick_data and self.tick_data[symbol]:
            return self.tick_data[symbol][-1]
        return None
    
    def get_latest_orderbook(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get latest order book for a symbol."""
        return self.order_books.get(symbol)
    
    def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[BarData]:
        """Get latest bar for a symbol and timeframe."""
        if (symbol in self.bar_data and 
            timeframe in self.bar_data[symbol] and 
            self.bar_data[symbol][timeframe]):
            return self.bar_data[symbol][timeframe][-1]
        return None
    
    def get_historical_ticks(self, symbol: str, limit: int = 100) -> List[TickData]:
        """Get historical tick data."""
        if symbol not in self.tick_data:
            return []
        
        ticks = list(self.tick_data[symbol])
        return ticks[-limit:] if limit else ticks
    
    def get_historical_bars(self, symbol: str, timeframe: str, limit: int = 100) -> List[BarData]:
        """Get historical bar data."""
        if (symbol not in self.bar_data or 
            timeframe not in self.bar_data[symbol]):
            return []
        
        bars = list(self.bar_data[symbol][timeframe])
        return bars[-limit:] if limit else bars
    
    def get_market_summary(self) -> Dict:
        """Get summary of current market data."""
        summary = {
            'subscribed_symbols': list(self.subscribed_symbols),
            'subscribed_timeframes': list(self.subscribed_timeframes),
            'active_tickers': len(self.tick_data),
            'active_orderbooks': len(self.order_books),
            'total_ticks': sum(len(ticks) for ticks in self.tick_data.values()),
            'total_bars': sum(
                len(bars) for symbol_bars in self.bar_data.values() 
                for bars in symbol_bars.values()
            ),
            'exchanges': list(self.exchanges.keys()),
            'is_running': self.is_running
        }
        
        return summary

def create_realtime_data_manager(exchanges: List[ccxt.Exchange]) -> RealTimeDataManager:
    """Factory function to create a real-time data manager."""
    return RealTimeDataManager(exchanges)

# Example callbacks for demonstration
def example_tick_callback(tick: TickData):
    """Example callback for tick data."""
    print(f"Tick: {tick.symbol} @ {tick.price} (Vol: {tick.volume})")

def example_orderbook_callback(orderbook: OrderBookSnapshot):
    """Example callback for order book data."""
    print(f"OrderBook: {orderbook.symbol} - Bid: {orderbook.best_bid}, Ask: {orderbook.best_ask}")

def example_bar_callback(bar: BarData):
    """Example callback for bar data."""
    print(f"Bar: {bar.symbol} {bar.timeframe} - O:{bar.open} H:{bar.high} L:{bar.low} C:{bar.close}")

if __name__ == "__main__":
    # Test the real-time data manager
    async def test_data_manager():
        try:
            # Initialize exchange
            exchange = ccxt.binance({
                'apiKey': 'your_api_key',
                'secret': 'your_secret',
                'sandbox': True,
                'enableRateLimit': True,
            })
            
            # Create data manager
            data_manager = RealTimeDataManager([exchange])
            
            # Add callbacks
            data_manager.add_tick_callback(example_tick_callback)
            data_manager.add_orderbook_callback(example_orderbook_callback)
            data_manager.add_bar_callback(example_bar_callback)
            
            # Subscribe to data
            await data_manager.subscribe_ticker('BTC/USDT')
            await data_manager.subscribe_orderbook('BTC/USDT')
            await data_manager.subscribe_bars('BTC/USDT', '1m')
            
            # Start streaming
            await data_manager.start()
            
            # Run for a while
            await asyncio.sleep(60)  # Run for 1 minute
            
            # Get summary
            summary = data_manager.get_market_summary()
            print(f"Market summary: {summary}")
            
            # Stop streaming
            await data_manager.stop()
            
        except Exception as e:
            print(f"Test failed: {str(e)}")
    
    # Run test
    asyncio.run(test_data_manager())