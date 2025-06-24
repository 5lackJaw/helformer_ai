"""
Multi-Exchange Management System for Helformer
Handles unified order book, cross-exchange arbitrage, and execution routing
"""

import ccxt
import asyncio
import pandas as pd
import numpy as np
import threading
import time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import logging
from config_helformer import config

logger = logging.getLogger(__name__)

@dataclass
class OrderBookLevel:
    """Single order book level (bid/ask)"""
    price: float
    size: float
    exchange: str
    timestamp: datetime

@dataclass
class UnifiedOrderBook:
    """Unified order book across exchanges"""
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: datetime
    best_bid: Optional[OrderBookLevel] = None
    best_ask: Optional[OrderBookLevel] = None
    spread: float = 0.0

@dataclass
class ExchangeMetrics:
    """Performance metrics for each exchange"""
    exchange_name: str
    latency_ms: float
    success_rate: float
    volume_24h: float
    fees: Dict[str, float]
    reliability_score: float
    last_update: datetime

@dataclass
class ArbitrageOpportunity:
    """Cross-exchange arbitrage opportunity"""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    profit_pct: float
    max_volume: float
    timestamp: datetime

class ExchangeManager:
    """
    Manages multiple cryptocurrency exchanges with unified interface.
    
    Features:
    - Unified order book aggregation
    - Smart order routing
    - Cross-exchange arbitrage detection
    - Latency and reliability monitoring
    - Automatic failover
    """
    
    def __init__(self, exchange_configs: Dict[str, Dict]):
        """
        Initialize exchange manager.
        
        Args:
            exchange_configs: Dict mapping exchange names to config dicts
                Example: {
                    'binance': {'api_key': '...', 'secret': '...', 'sandbox': True},
                    'coinbase': {'api_key': '...', 'secret': '...', 'sandbox': True}
                }
        """
        self.exchange_configs = exchange_configs
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.exchange_metrics: Dict[str, ExchangeMetrics] = {}
        self.unified_orderbooks: Dict[str, UnifiedOrderBook] = {}
        self.arbitrage_opportunities: List[ArbitrageOpportunity] = []
        
        # Configuration
        self.enabled_exchanges = []
        self.primary_exchange = None
        self.orderbook_depth = 10
        self.update_frequency = 1.0  # seconds
        self.min_arbitrage_profit = 0.002  # 0.2%
        
        # Monitoring
        self.orderbook_updates = defaultdict(int)
        self.last_update = {}
        self.is_running = False
        self.update_thread = None
        
        # Initialize exchanges
        self._initialize_exchanges()
        
        logger.info(f"ExchangeManager initialized with {len(self.exchanges)} exchanges")
    
    def _initialize_exchanges(self):
        """Initialize exchange connections"""
        for exchange_name, config_dict in self.exchange_configs.items():
            try:
                # Create exchange instance
                exchange_class = getattr(ccxt, exchange_name)
                exchange = exchange_class({
                    'apiKey': config_dict.get('api_key', ''),
                    'secret': config_dict.get('secret', ''),
                    'password': config_dict.get('passphrase', ''),
                    'sandbox': config_dict.get('sandbox', True),
                    'enableRateLimit': True,
                    'timeout': 30000,
                })
                
                # Test connection
                if exchange.has['fetchTicker']:
                    test_ticker = exchange.fetch_ticker('BTC/USDT')
                    logger.info(f"Successfully connected to {exchange_name}")
                    
                    self.exchanges[exchange_name] = exchange
                    self.enabled_exchanges.append(exchange_name)
                    
                    # Initialize metrics
                    self.exchange_metrics[exchange_name] = ExchangeMetrics(
                        exchange_name=exchange_name,
                        latency_ms=0.0,
                        success_rate=100.0,
                        volume_24h=0.0,
                        fees=self._get_exchange_fees(exchange),
                        reliability_score=100.0,
                        last_update=datetime.now()
                    )
                    
                    # Set primary exchange if not set
                    if self.primary_exchange is None:
                        self.primary_exchange = exchange_name
                        
                else:
                    logger.warning(f"Exchange {exchange_name} doesn't support required features")
                    
            except Exception as e:
                logger.error(f"Failed to initialize {exchange_name}: {str(e)}")
    
    def _get_exchange_fees(self, exchange: ccxt.Exchange) -> Dict[str, float]:
        """Get exchange fee structure"""
        try:
            fees = exchange.describe().get('fees', {})
            trading_fees = fees.get('trading', {})
            
            return {
                'maker': trading_fees.get('maker', 0.001),
                'taker': trading_fees.get('taker', 0.001),
                'withdrawal': 0.0  # Would need symbol-specific lookup
            }
        except:
            return {'maker': 0.001, 'taker': 0.001, 'withdrawal': 0.0}
    
    def start_orderbook_updates(self, symbols: List[str]):
        """Start real-time order book updates"""
        if self.is_running:
            logger.warning("Order book updates already running")
            return
        
        self.symbols = symbols
        self.is_running = True
        self.update_thread = threading.Thread(
            target=self._orderbook_update_loop,
            daemon=True
        )
        self.update_thread.start()
        logger.info(f"Started order book updates for {len(symbols)} symbols")
    
    def stop_orderbook_updates(self):
        """Stop order book updates"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5.0)
        logger.info("Stopped order book updates")
    
    def _orderbook_update_loop(self):
        """Main loop for updating order books"""
        while self.is_running:
            try:
                for symbol in self.symbols:
                    self._update_unified_orderbook(symbol)
                    self._detect_arbitrage_opportunities(symbol)
                
                time.sleep(self.update_frequency)
                
            except Exception as e:
                logger.error(f"Error in orderbook update loop: {str(e)}")
                time.sleep(1.0)
    
    def _update_unified_orderbook(self, symbol: str):
        """Update unified order book for a symbol"""
        all_bids = []
        all_asks = []
        
        for exchange_name in self.enabled_exchanges:
            try:
                exchange = self.exchanges[exchange_name]
                start_time = time.time()
                
                # Fetch order book
                orderbook = exchange.fetch_order_book(symbol, limit=self.orderbook_depth)
                
                # Update latency metrics
                latency = (time.time() - start_time) * 1000
                self._update_exchange_metrics(exchange_name, latency, True)
                
                # Convert to OrderBookLevel objects
                timestamp = datetime.now()
                
                for price, size in orderbook['bids']:
                    all_bids.append(OrderBookLevel(
                        price=price,
                        size=size,
                        exchange=exchange_name,
                        timestamp=timestamp
                    ))
                
                for price, size in orderbook['asks']:
                    all_asks.append(OrderBookLevel(
                        price=price,
                        size=size,
                        exchange=exchange_name,
                        timestamp=timestamp
                    ))
                
                self.orderbook_updates[exchange_name] += 1
                
            except Exception as e:
                logger.error(f"Error updating orderbook for {symbol} on {exchange_name}: {str(e)}")
                self._update_exchange_metrics(exchange_name, 0, False)
        
        # Sort and create unified order book
        all_bids.sort(key=lambda x: x.price, reverse=True)  # Highest bid first
        all_asks.sort(key=lambda x: x.price)  # Lowest ask first
        
        # Create unified order book
        unified_ob = UnifiedOrderBook(
            symbol=symbol,
            bids=all_bids,
            asks=all_asks,
            timestamp=datetime.now()
        )
        
        # Set best bid/ask and spread
        if all_bids:
            unified_ob.best_bid = all_bids[0]
        if all_asks:
            unified_ob.best_ask = all_asks[0]
        if unified_ob.best_bid and unified_ob.best_ask:
            unified_ob.spread = unified_ob.best_ask.price - unified_ob.best_bid.price
        
        self.unified_orderbooks[symbol] = unified_ob
        self.last_update[symbol] = datetime.now()
    
    def _update_exchange_metrics(self, exchange_name: str, latency: float, success: bool):
        """Update exchange performance metrics"""
        if exchange_name not in self.exchange_metrics:
            return
        
        metrics = self.exchange_metrics[exchange_name]
        
        # Update latency (exponential moving average)
        if latency > 0:
            alpha = 0.1
            metrics.latency_ms = alpha * latency + (1 - alpha) * metrics.latency_ms
        
        # Update success rate
        alpha = 0.05
        current_success = 100.0 if success else 0.0
        metrics.success_rate = alpha * current_success + (1 - alpha) * metrics.success_rate
        
        # Update reliability score (combines latency and success rate)
        latency_score = max(0, 100 - (metrics.latency_ms / 10))  # Penalize high latency
        metrics.reliability_score = (metrics.success_rate + latency_score) / 2
        
        metrics.last_update = datetime.now()
    
    def _detect_arbitrage_opportunities(self, symbol: str):
        """Detect cross-exchange arbitrage opportunities"""
        if symbol not in self.unified_orderbooks:
            return
        
        unified_ob = self.unified_orderbooks[symbol]
        
        # Group by exchange
        exchange_best_prices = {}
        for bid in unified_ob.bids[:5]:  # Check top 5 bids
            if bid.exchange not in exchange_best_prices:
                exchange_best_prices[bid.exchange] = {'best_bid': bid}
            elif bid.price > exchange_best_prices[bid.exchange]['best_bid'].price:
                exchange_best_prices[bid.exchange]['best_bid'] = bid
        
        for ask in unified_ob.asks[:5]:  # Check top 5 asks
            if ask.exchange not in exchange_best_prices:
                exchange_best_prices[ask.exchange] = {'best_ask': ask}
            elif ask.price < exchange_best_prices[ask.exchange].get('best_ask', ask).price:
                exchange_best_prices[ask.exchange]['best_ask'] = ask
        
        # Find arbitrage opportunities
        for buy_exchange, buy_data in exchange_best_prices.items():
            for sell_exchange, sell_data in exchange_best_prices.items():
                if (buy_exchange != sell_exchange and 
                    'best_ask' in buy_data and 'best_bid' in sell_data):
                    
                    buy_price = buy_data['best_ask'].price
                    sell_price = sell_data['best_bid'].price
                    
                    if sell_price > buy_price:
                        profit_pct = (sell_price - buy_price) / buy_price
                        
                        if profit_pct >= self.min_arbitrage_profit:
                            # Consider transaction costs
                            buy_fees = self.exchange_metrics[buy_exchange].fees
                            sell_fees = self.exchange_metrics[sell_exchange].fees
                            total_fees = buy_fees['taker'] + sell_fees['taker']
                            
                            net_profit = profit_pct - total_fees
                            
                            if net_profit > 0:
                                max_volume = min(
                                    buy_data['best_ask'].size,
                                    sell_data['best_bid'].size
                                )
                                
                                opportunity = ArbitrageOpportunity(
                                    symbol=symbol,
                                    buy_exchange=buy_exchange,
                                    sell_exchange=sell_exchange,
                                    buy_price=buy_price,
                                    sell_price=sell_price,
                                    profit_pct=net_profit,
                                    max_volume=max_volume,
                                    timestamp=datetime.now()
                                )
                                
                                self.arbitrage_opportunities.append(opportunity)
                                
                                logger.info(f"Arbitrage opportunity: {symbol} "
                                          f"Buy {buy_exchange}@{buy_price:.4f} "
                                          f"Sell {sell_exchange}@{sell_price:.4f} "
                                          f"Profit: {net_profit:.2%}")
        
        # Keep only recent opportunities (last 5 minutes)
        cutoff = datetime.now() - timedelta(minutes=5)
        self.arbitrage_opportunities = [
            opp for opp in self.arbitrage_opportunities 
            if opp.timestamp > cutoff
        ]
    
    def get_best_execution_route(self, symbol: str, side: str, size: float) -> Tuple[str, float]:
        """
        Find best exchange and price for execution.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Order size
            
        Returns:
            Tuple of (exchange_name, expected_price)
        """
        if symbol not in self.unified_orderbooks:
            return self.primary_exchange, 0.0
        
        unified_ob = self.unified_orderbooks[symbol]
        
        # Calculate weighted average price considering size
        best_exchange = None
        best_price = None
        best_score = float('-inf')
        
        # Group levels by exchange
        exchange_levels = defaultdict(list)
        
        if side == 'buy':
            levels = unified_ob.asks[:20]  # Check deeper into order book
        else:
            levels = unified_ob.bids[:20]
        
        for level in levels:
            exchange_levels[level.exchange].append(level)
        
        # Evaluate each exchange
        for exchange_name, levels in exchange_levels.items():
            if exchange_name not in self.exchange_metrics:
                continue
            
            # Calculate volume-weighted average price
            total_volume = 0
            total_cost = 0
            
            for level in levels:
                if total_volume >= size:
                    break
                
                volume = min(level.size, size - total_volume)
                total_volume += volume
                total_cost += volume * level.price
            
            if total_volume > 0:
                avg_price = total_cost / total_volume
                
                # Consider exchange reliability and fees
                metrics = self.exchange_metrics[exchange_name]
                reliability_factor = metrics.reliability_score / 100.0
                fee_cost = metrics.fees['taker'] * avg_price
                
                # Score combines price, reliability, and fees
                if side == 'buy':
                    score = -avg_price - fee_cost + (reliability_factor * avg_price * 0.01)
                else:
                    score = avg_price - fee_cost + (reliability_factor * avg_price * 0.01)
                
                if score > best_score:
                    best_score = score
                    best_exchange = exchange_name
                    best_price = avg_price
        
        return best_exchange or self.primary_exchange, best_price or 0.0
    
    def execute_order(self, exchange_name: str, symbol: str, side: str, 
                     amount: float, price: Optional[float] = None,
                     order_type: str = 'market') -> Dict:
        """
        Execute order on specified exchange.
        
        Args:
            exchange_name: Exchange to execute on
            symbol: Trading symbol
            side: 'buy' or 'sell'
            amount: Order amount
            price: Limit price (for limit orders)
            order_type: 'market' or 'limit'
            
        Returns:
            Order result dictionary
        """
        if exchange_name not in self.exchanges:
            raise ValueError(f"Exchange {exchange_name} not available")
        
        exchange = self.exchanges[exchange_name]
        
        try:
            start_time = time.time()
            
            if order_type == 'market':
                if side == 'buy':
                    result = exchange.create_market_buy_order(symbol, amount)
                else:
                    result = exchange.create_market_sell_order(symbol, amount)
            else:  # limit order
                if price is None:
                    raise ValueError("Price required for limit orders")
                
                if side == 'buy':
                    result = exchange.create_limit_buy_order(symbol, amount, price)
                else:
                    result = exchange.create_limit_sell_order(symbol, amount, price)
            
            # Update metrics
            execution_time = (time.time() - start_time) * 1000
            self._update_exchange_metrics(exchange_name, execution_time, True)
            
            logger.info(f"Order executed on {exchange_name}: {side} {amount} {symbol} @ {order_type}")
            
            return result
            
        except Exception as e:
            logger.error(f"Order execution failed on {exchange_name}: {str(e)}")
            self._update_exchange_metrics(exchange_name, 0, False)
            raise
    
    def get_unified_ticker(self, symbol: str) -> Dict:
        """Get unified ticker data across exchanges"""
        if symbol not in self.unified_orderbooks:
            return {}
        
        unified_ob = self.unified_orderbooks[symbol]
        
        if not unified_ob.best_bid or not unified_ob.best_ask:
            return {}
        
        mid_price = (unified_ob.best_bid.price + unified_ob.best_ask.price) / 2
        
        return {
            'symbol': symbol,
            'bid': unified_ob.best_bid.price,
            'ask': unified_ob.best_ask.price,
            'last': mid_price,
            'spread': unified_ob.spread,
            'spread_pct': (unified_ob.spread / mid_price) * 100,
            'timestamp': unified_ob.timestamp,
            'exchanges_count': len(set(level.exchange for level in unified_ob.bids + unified_ob.asks))
        }
    
    def get_exchange_status(self) -> Dict[str, Dict]:
        """Get status of all exchanges"""
        status = {}
        
        for exchange_name, metrics in self.exchange_metrics.items():
            status[exchange_name] = {
                'enabled': exchange_name in self.enabled_exchanges,
                'latency_ms': metrics.latency_ms,
                'success_rate': metrics.success_rate,
                'reliability_score': metrics.reliability_score,
                'orderbook_updates': self.orderbook_updates.get(exchange_name, 0),
                'last_update': metrics.last_update,
                'fees': metrics.fees
            }
        
        return status
    
    def get_arbitrage_summary(self) -> Dict:
        """Get summary of arbitrage opportunities"""
        if not self.arbitrage_opportunities:
            return {'count': 0, 'avg_profit': 0.0, 'best_profit': 0.0}
        
        profits = [opp.profit_pct for opp in self.arbitrage_opportunities]
        
        return {
            'count': len(self.arbitrage_opportunities),
            'avg_profit': np.mean(profits),
            'best_profit': max(profits),
            'total_volume': sum(opp.max_volume for opp in self.arbitrage_opportunities),
            'latest_opportunities': [
                {
                    'symbol': opp.symbol,
                    'profit_pct': opp.profit_pct,
                    'buy_exchange': opp.buy_exchange,
                    'sell_exchange': opp.sell_exchange,
                    'timestamp': opp.timestamp
                }
                for opp in sorted(self.arbitrage_opportunities, 
                                key=lambda x: x.timestamp, reverse=True)[:5]
            ]
        }
    
    def disconnect_all(self):
        """Disconnect from all exchanges"""
        self.stop_orderbook_updates()
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                if hasattr(exchange, 'close'):
                    exchange.close()
                logger.info(f"Disconnected from {exchange_name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {exchange_name}: {str(e)}")
        
        self.exchanges.clear()
        self.enabled_exchanges.clear()


# Factory function for easy initialization
def create_exchange_manager(config_dict: Optional[Dict] = None) -> ExchangeManager:
    """
    Factory function to create ExchangeManager with default configuration.
    
    Args:
        config_dict: Optional custom configuration
        
    Returns:
        Configured ExchangeManager instance
    """
    if config_dict is None:
        # Default configuration for testing
        config_dict = {
            'binance': {
                'api_key': '',
                'secret': '',
                'sandbox': True
            }
        }
    
    return ExchangeManager(config_dict)


# Global instance for easy access
exchange_manager = None

def get_exchange_manager() -> Optional[ExchangeManager]:
    """Get the global exchange manager instance"""
    return exchange_manager

def initialize_exchange_manager(config_dict: Dict) -> ExchangeManager:
    """Initialize the global exchange manager"""
    global exchange_manager
    exchange_manager = create_exchange_manager(config_dict)
    return exchange_manager