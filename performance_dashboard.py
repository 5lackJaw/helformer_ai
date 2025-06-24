"""
Real-Time Performance Dashboard for Helformer Trading Bot
Web-based dashboard for monitoring trading performance and system health
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List
import threading
import time

try:
    from flask import Flask, render_template, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask")

from model_status import model_monitor
from config_helformer import config

class PerformanceDashboard:
    """
    Real-time performance dashboard for monitoring trading bot performance.
    
    Features:
    - Live P&L tracking
    - Position monitoring
    - Model performance metrics
    - System health indicators
    - Trading alerts
    """
    
    def __init__(self, port: int = 5000):
        """
        Initialize performance dashboard.
        
        Args:
            port: Port for web server
        """
        self.port = port
        self.app = None
        self.monitor = model_monitor
        
        # Dashboard data
        self.dashboard_data = {
            'equity_curve': [],
            'daily_pnl': [],
            'trade_history': [],
            'active_positions': [],
            'performance_metrics': {},
            'system_health': {},
            'regime_analysis': {},
            'alerts': []
        }
        
        # Data update thread
        self.update_thread = None
        self.running = False
        
        if FLASK_AVAILABLE:
            self._setup_flask_app()
    
    def _setup_flask_app(self):
        """Setup Flask web application"""
        self.app = Flask(__name__)
        
        @self.app.route('/')
        def dashboard():
            return self._render_dashboard()
        
        @self.app.route('/api/data')
        def get_data():
            return jsonify(self.dashboard_data)
        
        @self.app.route('/api/health')
        def get_health():
            health = self.monitor.get_system_health()
            return jsonify({
                'status': 'healthy' if health.models_healthy > 0 else 'warning',
                'timestamp': health.timestamp.isoformat(),
                'models_healthy': health.models_healthy,
                'total_models': health.models_total,
                'error_rate': health.error_rate,
                'uptime_hours': health.uptime_hours
            })
        
        @self.app.route('/api/positions')
        def get_positions():
            return jsonify(self.dashboard_data.get('active_positions', []))
        
        @self.app.route('/api/trades')
        def get_trades():
            return jsonify(self.dashboard_data.get('trade_history', []))
    
    def _render_dashboard(self) -> str:
        """Render the main dashboard HTML"""
        
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Helformer Trading Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #1e1e1e;
                    color: white;
                }
                .header {
                    text-align: center;
                    margin-bottom: 30px;
                }
                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }
                .metric-card {
                    background: #2d2d2d;
                    padding: 20px;
                    border-radius: 10px;
                    border: 1px solid #404040;
                }
                .metric-value {
                    font-size: 2em;
                    font-weight: bold;
                    margin: 10px 0;
                }
                .metric-label {
                    color: #888;
                    font-size: 0.9em;
                }
                .positive { color: #00ff88; }
                .negative { color: #ff4444; }
                .neutral { color: #888; }
                .chart-container {
                    background: #2d2d2d;
                    margin: 20px 0;
                    padding: 20px;
                    border-radius: 10px;
                    border: 1px solid #404040;
                }
                .positions-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }
                .positions-table th,
                .positions-table td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #404040;
                }
                .positions-table th {
                    background-color: #383838;
                }
                .status-indicator {
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    margin-right: 8px;
                }
                .status-healthy { background-color: #00ff88; }
                .status-warning { background-color: #ffaa00; }
                .status-error { background-color: #ff4444; }
                .refresh-button {
                    background: #007acc;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    margin: 10px 0;
                }
                .refresh-button:hover {
                    background: #005a9e;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🤖 Helformer Trading Dashboard</h1>
                <p>Real-time AI Trading Performance Monitor</p>
                <button class="refresh-button" onclick="refreshData()">🔄 Refresh Data</button>
            </div>
            
            <div class="metrics-grid" id="metricsGrid">
                <!-- Metrics will be populated by JavaScript -->
            </div>
            
            <div class="chart-container">
                <h3>📈 Equity Curve</h3>
                <div id="equityChart"></div>
            </div>
            
            <div class="chart-container">
                <h3>💰 Daily P&L</h3>
                <div id="pnlChart"></div>
            </div>
            
            <div class="chart-container">
                <h3>🎯 Model Performance</h3>
                <div id="modelChart"></div>
            </div>
            
            <div class="chart-container">
                <h3>📊 Active Positions</h3>
                <table class="positions-table" id="positionsTable">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Direction</th>
                            <th>Entry Price</th>
                            <th>Current P&L</th>
                            <th>Duration</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="positionsBody">
                        <!-- Positions will be populated by JavaScript -->
                    </tbody>
                </table>
            </div>
            
            <script>
                async function fetchDashboardData() {
                    try {
                        const response = await fetch('/api/data');
                        const data = await response.json();
                        return data;
                    } catch (error) {
                        console.error('Error fetching data:', error);
                        return null;
                    }
                }
                
                function updateMetrics(data) {
                    const metrics = data.performance_metrics || {};
                    const health = data.system_health || {};
                    
                    const metricsGrid = document.getElementById('metricsGrid');
                    metricsGrid.innerHTML = `
                        <div class="metric-card">
                            <div class="metric-label">Portfolio Value</div>
                            <div class="metric-value">$${(metrics.portfolio_value || 0).toLocaleString()}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Total P&L</div>
                            <div class="metric-value ${(metrics.total_pnl || 0) >= 0 ? 'positive' : 'negative'}">
                                $${(metrics.total_pnl || 0).toFixed(2)}
                            </div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Win Rate</div>
                            <div class="metric-value">${(metrics.win_rate || 0).toFixed(1)}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Total Trades</div>
                            <div class="metric-value">${metrics.total_trades || 0}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Active Positions</div>
                            <div class="metric-value">${(data.active_positions || []).length}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">System Health</div>
                            <div class="metric-value">
                                <span class="status-indicator ${health.models_healthy > 0 ? 'status-healthy' : 'status-warning'}"></span>
                                ${health.models_healthy || 0}/${health.models_total || 0}
                            </div>
                        </div>
                    `;
                }
                
                function updateCharts(data) {
                    // Equity Curve
                    const equityData = data.equity_curve || [];
                    if (equityData.length > 0) {
                        const equityTrace = {
                            x: equityData.map(d => d.timestamp),
                            y: equityData.map(d => d.value),
                            type: 'scatter',
                            mode: 'lines',
                            name: 'Portfolio Value',
                            line: {color: '#00ff88', width: 2}
                        };
                        
                        Plotly.newPlot('equityChart', [equityTrace], {
                            paper_bgcolor: 'rgba(0,0,0,0)',
                            plot_bgcolor: 'rgba(0,0,0,0)',
                            font: {color: 'white'},
                            xaxis: {gridcolor: '#404040'},
                            yaxis: {gridcolor: '#404040'}
                        });
                    }
                    
                    // Daily P&L
                    const pnlData = data.daily_pnl || [];
                    if (pnlData.length > 0) {
                        const pnlTrace = {
                            x: pnlData.map(d => d.date),
                            y: pnlData.map(d => d.pnl),
                            type: 'bar',
                            name: 'Daily P&L',
                            marker: {
                                color: pnlData.map(d => d.pnl >= 0 ? '#00ff88' : '#ff4444')
                            }
                        };
                        
                        Plotly.newPlot('pnlChart', [pnlTrace], {
                            paper_bgcolor: 'rgba(0,0,0,0)',
                            plot_bgcolor: 'rgba(0,0,0,0)',
                            font: {color: 'white'},
                            xaxis: {gridcolor: '#404040'},
                            yaxis: {gridcolor: '#404040'}
                        });
                    }
                }
                
                function updatePositions(data) {
                    const positions = data.active_positions || [];
                    const tbody = document.getElementById('positionsBody');
                    
                    if (positions.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #888;">No active positions</td></tr>';
                        return;
                    }
                    
                    tbody.innerHTML = positions.map(pos => `
                        <tr>
                            <td>${pos.symbol}</td>
                            <td>${pos.direction}</td>
                            <td>$${pos.entry_price.toFixed(4)}</td>
                            <td class="${pos.pnl >= 0 ? 'positive' : 'negative'}">$${pos.pnl.toFixed(2)}</td>
                            <td>${pos.duration}</td>
                            <td>
                                <span class="status-indicator status-healthy"></span>
                                Active
                            </td>
                        </tr>
                    `).join('');
                }
                
                async function refreshData() {
                    const data = await fetchDashboardData();
                    if (data) {
                        updateMetrics(data);
                        updateCharts(data);
                        updatePositions(data);
                    }
                }
                
                // Initial load
                refreshData();
                
                // Auto refresh every 30 seconds
                setInterval(refreshData, 30000);
                
                // Update timestamp
                document.addEventListener('DOMContentLoaded', function() {
                    setInterval(() => {
                        const now = new Date();
                        document.title = `Helformer Dashboard - ${now.toLocaleTimeString()}`;
                    }, 1000);
                });
            </script>
        </body>
        </html>
        """
        
        return html_template
    
    def update_dashboard_data(self, 
                             portfolio_value: float = 0.0,
                             daily_pnl: float = 0.0,
                             active_positions: List[Dict] = None,
                             trade_history: List[Dict] = None):
        """Update dashboard data"""
        
        # Performance metrics
        total_trades = len(trade_history) if trade_history else 0
        winning_trades = sum(1 for trade in (trade_history or []) if trade.get('pnl', 0) > 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum(trade.get('pnl', 0) for trade in (trade_history or []))
        
        self.dashboard_data.update({
            'performance_metrics': {
                'portfolio_value': portfolio_value,
                'daily_pnl': daily_pnl,
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'winning_trades': winning_trades
            },
            'active_positions': active_positions or [],
            'trade_history': trade_history or [],
            'system_health': self.monitor.get_system_health().__dict__ if hasattr(self.monitor.get_system_health(), '__dict__') else {},
            'last_updated': datetime.now().isoformat()
        })
        
        # Update equity curve
        self.dashboard_data['equity_curve'].append({
            'timestamp': datetime.now().isoformat(),
            'value': portfolio_value
        })
        
        # Keep only last 100 points
        if len(self.dashboard_data['equity_curve']) > 100:
            self.dashboard_data['equity_curve'] = self.dashboard_data['equity_curve'][-100:]
        
        # Update daily P&L
        today = datetime.now().date().isoformat()
        daily_pnl_data = self.dashboard_data['daily_pnl']
        
        # Update or add today's P&L
        today_entry = next((item for item in daily_pnl_data if item['date'] == today), None)
        if today_entry:
            today_entry['pnl'] = daily_pnl
        else:
            daily_pnl_data.append({'date': today, 'pnl': daily_pnl})
        
        # Keep only last 30 days
        if len(daily_pnl_data) > 30:
            self.dashboard_data['daily_pnl'] = daily_pnl_data[-30:]
    
    def generate_performance_report(self) -> str:
        """Generate text-based performance report"""
        
        metrics = self.dashboard_data.get('performance_metrics', {})
        health = self.dashboard_data.get('system_health', {})
        positions = self.dashboard_data.get('active_positions', [])
        
        report = []
        report.append("="*60)
        report.append("HELFORMER TRADING PERFORMANCE REPORT")
        report.append("="*60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Performance Summary
        report.append("PERFORMANCE SUMMARY:")
        report.append(f"  Portfolio Value: ${metrics.get('portfolio_value', 0):,.2f}")
        report.append(f"  Total P&L: ${metrics.get('total_pnl', 0):,.2f}")
        report.append(f"  Daily P&L: ${metrics.get('daily_pnl', 0):,.2f}")
        report.append(f"  Total Trades: {metrics.get('total_trades', 0)}")
        report.append(f"  Win Rate: {metrics.get('win_rate', 0):.1f}%")
        report.append("")
        
        # Active Positions
        report.append("ACTIVE POSITIONS:")
        if positions:
            for pos in positions:
                report.append(f"  {pos.get('symbol', 'N/A')} {pos.get('direction', 'N/A')}: ${pos.get('pnl', 0):.2f} P&L")
        else:
            report.append("  No active positions")
        report.append("")
        
        # System Health
        report.append("SYSTEM HEALTH:")
        report.append(f"  Models Healthy: {health.get('models_healthy', 0)}/{health.get('models_total', 0)}")
        report.append(f"  Error Rate: {health.get('error_rate', 0):.2f}%")
        report.append(f"  Uptime: {health.get('uptime_hours', 0):.1f} hours")
        report.append("")
        
        report.append("="*60)
        
        return "\n".join(report)
    
    def start_dashboard(self):
        """Start the web dashboard"""
        if not FLASK_AVAILABLE:
            print("Flask not available. Dashboard cannot start.")
            print("Install Flask with: pip install flask")
            return
        
        if self.app:
            print(f"Starting dashboard on http://localhost:{self.port}")
            print("Access the dashboard in your web browser")
            self.app.run(host='0.0.0.0', port=self.port, debug=False)
        else:
            print("Flask app not initialized")
    
    def start_data_update_thread(self):
        """Start background thread for data updates"""
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def _update_loop(self):
        """Background loop for updating dashboard data"""
        while self.running:
            try:
                # Update system health data
                health = self.monitor.get_system_health()
                self.dashboard_data['system_health'] = {
                    'timestamp': health.timestamp.isoformat(),
                    'models_healthy': health.models_healthy,
                    'models_total': health.models_total,
                    'error_rate': health.error_rate,
                    'uptime_hours': health.uptime_hours,
                    'average_accuracy': health.average_accuracy
                }
                
                # Check for alerts
                alerts = self.monitor.check_alert_conditions()
                self.dashboard_data['alerts'] = alerts
                
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                print(f"Error in dashboard update loop: {str(e)}")
                time.sleep(60)
    
    def stop(self):
        """Stop the dashboard"""
        self.running = False
        if self.update_thread:
            self.update_thread.join()

# Global dashboard instance
dashboard = PerformanceDashboard()

def main():
    """Run the dashboard as a standalone application"""
    print("Helformer Trading Dashboard")
    print("Starting dashboard server...")
    
    dashboard.start_data_update_thread()
    dashboard.start_dashboard()

if __name__ == "__main__":
    main()