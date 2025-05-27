"""
Backtesting engine for portfolio strategies.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

@dataclass
class BacktestResult:
    """Results from a backtest run."""
    portfolio_value: pd.Series
    returns: pd.Series
    weights_history: pd.DataFrame
    metrics: Dict[str, float]
    trades: pd.DataFrame

class BacktestEngine:
    """
    Backtesting engine for portfolio strategies with transaction costs and slippage.
    """
    def __init__(
        self,
        price_data: pd.DataFrame,
        initial_capital: float = 1_000_000.0,
        transaction_cost: float = 0.001,  # 10 bps
        slippage_model: str = 'sqrt',  # 'linear' or 'sqrt'
        rebalance_frequency: str = 'M'  # Monthly rebalancing
    ):
        """
        Initialize the backtesting engine.
        
        Args:
            price_data: DataFrame with price history (index: dates, columns: assets)
            initial_capital: Initial portfolio value
            transaction_cost: Transaction cost as a fraction of trade value
            slippage_model: Model for market impact ('linear' or 'sqrt')
            rebalance_frequency: Frequency of portfolio rebalancing
        """
        self.price_data = price_data
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage_model = slippage_model
        self.rebalance_frequency = rebalance_frequency
        self.assets = price_data.columns.tolist()
        
    def _calculate_slippage(self, trade_size: float, price: float) -> float:
        """
        Calculate slippage based on trade size.
        
        Args:
            trade_size: Size of the trade in currency units
            price: Current price of the asset
            
        Returns:
            Slippage as a fraction of price
        """
        if self.slippage_model == 'linear':
            return 0.0001 * (trade_size / 1_000_000)  # 1bp per $1M
        else:  # sqrt model
            return 0.0002 * np.sqrt(trade_size / 1_000_000)  # 2bp per sqrt($1M)
    
    def _calculate_transaction_costs(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float
    ) -> float:
        """
        Calculate transaction costs for rebalancing.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Current portfolio value
            
        Returns:
            Total transaction costs
        """
        costs = 0.0
        for asset in self.assets:
            trade_size = abs(
                target_weights[asset] - current_weights[asset]
            ) * portfolio_value
            
            if trade_size > 0:
                # Calculate slippage
                price = self.price_data[asset].iloc[-1]
                slippage = self._calculate_slippage(trade_size, price)
                
                # Total cost = transaction cost + slippage
                costs += trade_size * (self.transaction_cost + slippage)
        
        return costs
    
    def run_backtest(
        self,
        weight_generator: callable,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """
        Run backtest with the given weight generator function.
        
        Args:
            weight_generator: Function that generates target weights
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            BacktestResult object with results
        """
        # Filter price data for backtest period
        if start_date:
            price_data = self.price_data[self.price_data.index >= start_date]
        else:
            price_data = self.price_data
            
        if end_date:
            price_data = price_data[price_data.index <= end_date]
        
        # Initialize results
        portfolio_value = pd.Series(index=price_data.index, dtype=float)
        portfolio_value.iloc[0] = self.initial_capital
        
        weights_history = pd.DataFrame(
            index=price_data.index,
            columns=self.assets,
            dtype=float
        )
        
        trades = []
        current_weights = {asset: 0.0 for asset in self.assets}
        
        # Run backtest
        for i in range(1, len(price_data)):
            current_date = price_data.index[i]
            prev_date = price_data.index[i-1]
            
            # Check if rebalancing is needed
            if i == 1 or (current_date - prev_date).days >= pd.Timedelta(self.rebalance_frequency).days:
                # Get target weights
                target_weights = weight_generator(
                    price_data.iloc[:i],
                    current_weights,
                    portfolio_value.iloc[i-1]
                )
                
                # Calculate transaction costs
                costs = self._calculate_transaction_costs(
                    current_weights,
                    target_weights,
                    portfolio_value.iloc[i-1]
                )
                
                # Update portfolio value
                portfolio_value.iloc[i] = (
                    portfolio_value.iloc[i-1] * (1 - costs/portfolio_value.iloc[i-1])
                )
                
                # Record trade
                for asset in self.assets:
                    if target_weights[asset] != current_weights[asset]:
                        trades.append({
                            'date': current_date,
                            'asset': asset,
                            'old_weight': current_weights[asset],
                            'new_weight': target_weights[asset],
                            'trade_size': (
                                target_weights[asset] - current_weights[asset]
                            ) * portfolio_value.iloc[i-1]
                        })
                
                current_weights = target_weights
            else:
                # No rebalancing, just update portfolio value
                portfolio_value.iloc[i] = portfolio_value.iloc[i-1]
            
            # Update weights based on price changes
            price_changes = price_data.iloc[i] / price_data.iloc[i-1]
            for asset in self.assets:
                current_weights[asset] *= price_changes[asset]
            
            # Normalize weights
            total_weight = sum(current_weights.values())
            current_weights = {k: v/total_weight for k, v in current_weights.items()}
            
            # Record weights
            weights_history.iloc[i] = current_weights
        
        # Calculate returns
        returns = portfolio_value.pct_change()
        
        # Calculate metrics
        metrics = {
            'total_return': (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1,
            'annualized_return': returns.mean() * 252,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            'max_drawdown': (portfolio_value / portfolio_value.cummax() - 1).min(),
            'turnover': pd.DataFrame(trades)['trade_size'].abs().sum() / portfolio_value.mean()
        }
        
        return BacktestResult(
            portfolio_value=portfolio_value,
            returns=returns,
            weights_history=weights_history,
            metrics=metrics,
            trades=pd.DataFrame(trades)
        ) 