"""
Example script demonstrating factor model analysis with regime detection.
"""
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from apmti.data.simulation.market_data_generator import MarketDataGenerator, MarketParameters
from apmti.models.factors.factor_model import FactorModel

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create market parameters with enhanced factor structure
    params = MarketParameters(
        n_assets=50,  # Increased number of assets
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31),
        base_volatility=0.15,
        base_return=0.08,
        correlation_range=(0.2, 0.8),
        esg_score_range=(0.0, 1.0),
        controversy_range=(0.0, 1.0),
        n_factors=5,  # Increased number of factors
        factor_persistence=0.95,  # High persistence
        regime_change_prob=0.01,  # 1% chance of regime change per period
        factor_correlation=0.3  # Moderate factor correlation
    )
    
    # Generate market data
    generator = MarketDataGenerator(params)
    price_history = generator.generate_price_history()
    
    # Calculate returns
    returns = price_history.pct_change().dropna()
    
    # Create factor model with regime detection
    model = FactorModel(
        returns=returns,
        factor_data=returns,  # Using returns as factor data for PCA
        min_half_life=20,
        max_half_life=252,
        min_t_stat=2.0,
        max_factors=5,
        regime_window=63,  # 3-month window for regime detection
        regime_threshold=0.7
    )
    
    # Fit model
    model.fit()
    
    # Get factor importance
    importance = model.get_factor_importance()
    
    # Get regime information
    regimes = model.get_regime_info()
    
    # Print results
    print("\nFactor Model Analysis Results:")
    print("-----------------------------")
    print("\nFactor Importance:")
    print(importance)
    
    print("\nRegime Analysis:")
    print("----------------")
    for i, regime in enumerate(regimes):
        print(f"\nRegime {i+1} ({regime.start_date.date()} to {regime.end_date.date()}):")
        print(f"Duration: {(regime.end_date - regime.start_date).days} days")
        print("\nFactor Volatilities:")
        print(regime.factor_volatilities)
        print("\nFactor Correlations:")
        print(regime.factor_correlations)
    
    print("\nFactor Exposures Summary:")
    for asset in returns.columns[:5]:  # Show first 5 assets
        exposures = model.factor_exposures[asset]
        print(f"\n{asset}:")
        if not exposures:
            print("  [No significant exposures]")
        for factor, exp in exposures.items():
            print(f"  {factor}:")
            print(f"    Exposure: {exp.exposure:.4f}")
            print(f"    t-stat: {exp.t_stat:.2f}")
            print(f"    p-value: {exp.p_value:.4f}")
            print(f"    Half-life: {exp.half_life:.1f} days")
            print(f"    Regime Sensitivity: {exp.regime_sensitivity:.4f}")
            print(f"    Persistence: {exp.persistence:.4f}")
    
    # Plot factor returns
    plt.figure(figsize=(12, 6))
    model.factor_returns.cumsum().plot()
    plt.title('Cumulative Factor Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('factor_returns.png')
    
    # Plot factor importance
    if importance['n_significant'].sum() > 0:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance, x='factor', y='n_significant')
        plt.title('Number of Significant Exposures by Factor')
        plt.xlabel('Factor')
        plt.ylabel('Number of Significant Exposures')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('factor_importance.png')
    
    # Plot factor persistence
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance, x='factor', y='avg_persistence')
    plt.title('Average Factor Persistence')
    plt.xlabel('Factor')
    plt.ylabel('AR(1) Coefficient')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('factor_persistence.png')
    
    # Plot regime sensitivity
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance, x='factor', y='avg_regime_sensitivity')
    plt.title('Average Regime Sensitivity')
    plt.xlabel('Factor')
    plt.ylabel('Regime Sensitivity')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('regime_sensitivity.png')
    
    # Plot factor exposures heatmap
    exposure_matrix = pd.DataFrame(
        {asset: {factor: exp.exposure for factor, exp in exposures.items()}
         for asset, exposures in model.factor_exposures.items()}
    ).T
    if not exposure_matrix.empty and not exposure_matrix.isnull().all().all():
        plt.figure(figsize=(12, 8))
        sns.heatmap(exposure_matrix, cmap='RdBu', center=0)
        plt.title('Factor Exposures Heatmap')
        plt.xlabel('Factor')
        plt.ylabel('Asset')
        plt.tight_layout()
        plt.savefig('factor_exposures.png')
    
    # Plot regime transitions
    plt.figure(figsize=(12, 6))
    regime_dates = []
    regime_labels = []
    for regime in regimes:
        regime_dates.extend([regime.start_date, regime.end_date])
        regime_labels.extend([f'Regime {regime.regime_id + 1}'] * 2)
    
    regime_df = pd.DataFrame({
        'date': regime_dates,
        'regime': regime_labels
    })
    
    plt.plot(returns.index, returns.mean(axis=1), label='Average Return')
    for i in range(len(regime_dates) - 1):
        plt.axvspan(
            regime_dates[i],
            regime_dates[i + 1],
            alpha=0.2,
            label=regime_labels[i] if i % 2 == 0 else None
        )
    plt.title('Market Regimes and Average Returns')
    plt.xlabel('Date')
    plt.ylabel('Average Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('regime_transitions.png')

if __name__ == "__main__":
    main() 