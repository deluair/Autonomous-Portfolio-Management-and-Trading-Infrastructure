"""
Example script demonstrating portfolio optimization with backtesting.
"""
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from apmti.core.optimization.portfolio_optimizer import PortfolioOptimizer
from apmti.data.simulation.market_data_generator import MarketDataGenerator, MarketParameters
from apmti.backtesting.backtest_engine import BacktestEngine

def create_optimizer(price_data: pd.DataFrame, esg_scores: dict) -> PortfolioOptimizer:
    """
    Create portfolio optimizer from price data.
    
    Args:
        price_data: DataFrame with price history
        esg_scores: Dictionary of ESG scores
        
    Returns:
        PortfolioOptimizer instance
    """
    # Calculate returns and covariance
    returns = price_data.pct_change().mean() * 252  # Annualized returns
    cov_matrix = price_data.pct_change().cov() * 252  # Annualized covariance
    
    return PortfolioOptimizer(
        returns=returns.values,
        cov_matrix=cov_matrix.values,
        esg_scores=esg_scores,
        risk_free_rate=0.02,
        min_controversy_score=0.5
    )

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create market parameters
    params = MarketParameters(
        n_assets=10,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31),
        base_volatility=0.15,
        base_return=0.08,
        correlation_range=(0.2, 0.8),
        esg_score_range=(0.0, 1.0),
        controversy_range=(0.0, 1.0)
    )
    
    # Generate market data
    generator = MarketDataGenerator(params)
    price_history = generator.generate_price_history()
    _, _, esg_scores = generator.generate_market_data()
    
    # Create backtesting engine
    engine = BacktestEngine(
        price_data=price_history,
        initial_capital=1_000_000.0,
        transaction_cost=0.001,  # 10 bps
        slippage_model='sqrt',
        rebalance_frequency='M'  # Monthly rebalancing
    )
    
    # Define weight generator function
    def weight_generator(price_data, current_weights, portfolio_value):
        """Generate optimal weights using portfolio optimizer."""
        optimizer = create_optimizer(price_data, esg_scores)
        result = optimizer.optimize(
            assets=generator.assets,
            target_return=0.10,  # 10% target return
            max_volatility=0.20  # 20% max volatility
        )
        return result['weights']
    
    # Run backtest
    result = engine.run_backtest(weight_generator)
    
    # Print results
    print("\nBacktest Results:")
    print("----------------")
    print(f"Total Return: {result.metrics['total_return']:.2%}")
    print(f"Annualized Return: {result.metrics['annualized_return']:.2%}")
    print(f"Volatility: {result.metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
    print(f"Annual Turnover: {result.metrics['turnover']:.2%}")
    
    # Plot portfolio value
    plt.figure(figsize=(12, 6))
    result.portfolio_value.plot()
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('portfolio_value.png')
    
    # Plot drawdown
    plt.figure(figsize=(12, 6))
    drawdown = (result.portfolio_value / result.portfolio_value.cummax() - 1)
    drawdown.plot()
    plt.title('Portfolio Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('portfolio_drawdown.png')
    
    # Plot rolling metrics
    plt.figure(figsize=(12, 6))
    rolling_returns = result.returns.rolling(window=252).mean() * 252
    rolling_vol = result.returns.rolling(window=252).std() * np.sqrt(252)
    rolling_sharpe = rolling_returns / rolling_vol
    
    rolling_sharpe.plot()
    plt.title('Rolling Sharpe Ratio (1-year window)')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('rolling_sharpe.png')
    
    # Plot weight evolution
    plt.figure(figsize=(12, 6))
    result.weights_history.plot()
    plt.title('Portfolio Weights Over Time')
    plt.xlabel('Date')
    plt.ylabel('Weight')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('weight_evolution.png')
    
    # Plot trade analysis
    plt.figure(figsize=(12, 6))
    trade_sizes = result.trades.groupby('date')['trade_size'].abs().sum()
    trade_sizes.plot(kind='bar')
    plt.title('Trading Activity')
    plt.xlabel('Date')
    plt.ylabel('Total Trade Size ($)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('trading_activity.png')

if __name__ == "__main__":
    main() 