"""
Example script demonstrating portfolio optimization with ESG constraints.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from apmti.core.optimization.portfolio_optimizer import PortfolioOptimizer
from apmti.data.simulation.market_data_generator import MarketDataGenerator, MarketParameters

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
    returns, cov_matrix, esg_scores = generator.generate_market_data()
    price_history = generator.generate_price_history()
    
    # Create portfolio optimizer
    optimizer = PortfolioOptimizer(
        returns=returns,
        cov_matrix=cov_matrix,
        esg_scores=esg_scores,
        risk_free_rate=0.02,
        min_controversy_score=0.5
    )
    
    # Optimize portfolio
    result = optimizer.optimize(
        assets=generator.assets,
        target_return=0.10,  # 10% target return
        max_volatility=0.20  # 20% max volatility
    )
    
    # Print results
    print("\nOptimization Results:")
    print("-------------------")
    print(f"Expected Return: {result['expected_return']:.2%}")
    print(f"Volatility: {result['volatility']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print(f"ESG Score: {result['esg_score']:.2f}")
    print(f"Controversy Score: {result['controversy_score']:.2f}")
    
    print("\nPortfolio Weights:")
    for asset, weight in result['weights'].items():
        print(f"{asset}: {weight:.2%}")
    
    # Plot portfolio weights
    plt.figure(figsize=(12, 6))
    weights_df = pd.Series(result['weights'])
    weights_df.plot(kind='bar')
    plt.title('Portfolio Weights')
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('portfolio_weights.png')
    
    # Plot price history
    plt.figure(figsize=(12, 6))
    price_history.plot()
    plt.title('Asset Price History')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('price_history.png')
    
    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pd.DataFrame(cov_matrix, index=generator.assets, columns=generator.assets),
        annot=True,
        cmap='coolwarm',
        center=0
    )
    plt.title('Asset Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')

if __name__ == "__main__":
    main() 