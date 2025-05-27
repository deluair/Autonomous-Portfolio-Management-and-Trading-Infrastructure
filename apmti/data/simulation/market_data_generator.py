"""
Market data generator for simulation purposes.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from apmti.core.optimization.portfolio_optimizer import ESGScore

@dataclass
class MarketParameters:
    """Market simulation parameters."""
    n_assets: int
    start_date: datetime
    end_date: datetime
    base_volatility: float
    base_return: float
    correlation_range: Tuple[float, float]
    esg_score_range: Tuple[float, float]
    controversy_range: Tuple[float, float]
    n_factors: int = 3  # Number of latent factors
    factor_persistence: float = 0.95  # Factor persistence (AR(1) coefficient)
    regime_change_prob: float = 0.01  # Probability of regime change per period
    factor_correlation: float = 0.3  # Base correlation between factors

class MarketDataGenerator:
    """
    Generates realistic market data for simulation purposes.
    """
    def __init__(self, params: MarketParameters):
        """
        Initialize the market data generator.
        
        Args:
            params: Market simulation parameters
        """
        self.params = params
        self.assets = [f"ASSET_{i}" for i in range(params.n_assets)]
        self.factor_names = [f"FACTOR_{i}" for i in range(params.n_factors)]
        self.regime = 0  # Current regime
        self.regime_returns = None  # Regime-specific returns
        self.regime_volatilities = None  # Regime-specific volatilities
        
    def _generate_regime_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate regime-specific parameters."""
        # Generate regime-specific returns
        regime_returns = np.random.normal(
            self.params.base_return / 252,
            self.params.base_volatility / np.sqrt(252),
            (self.params.n_factors,)
        )
        
        # Generate regime-specific volatilities
        regime_vols = np.random.lognormal(
            np.log(self.params.base_volatility / np.sqrt(252)),
            0.2,
            self.params.n_factors
        )
        
        return regime_returns, regime_vols
    
    def _generate_factor_correlation_matrix(self) -> np.ndarray:
        """Generate factor correlation matrix."""
        # Start with identity matrix
        corr = np.eye(self.params.n_factors)
        
        # Add correlation between factors
        for i in range(self.params.n_factors):
            for j in range(i + 1, self.params.n_factors):
                corr[i, j] = corr[j, i] = np.random.uniform(
                    -self.params.factor_correlation,
                    self.params.factor_correlation
                )
        
        # Ensure positive definiteness
        min_eig = np.min(np.real(np.linalg.eigvals(corr)))
        if min_eig < 0:
            corr -= min_eig * np.eye(self.params.n_factors)
            corr /= (1 - min_eig)
            
        return corr
    
    def _generate_asset_exposures(self) -> np.ndarray:
        """Generate asset exposures to factors."""
        # Generate random exposures
        exposures = np.random.normal(0, 1, (self.params.n_assets, self.params.n_factors))
        
        # Normalize exposures
        exposures = exposures / np.linalg.norm(exposures, axis=1, keepdims=True)
        
        # Add some assets with concentrated exposures
        n_concentrated = int(self.params.n_assets * 0.2)  # 20% of assets
        concentrated_indices = np.random.choice(
            self.params.n_assets,
            n_concentrated,
            replace=False
        )
        
        for idx in concentrated_indices:
            factor_idx = np.random.randint(0, self.params.n_factors)
            exposures[idx] = 0
            exposures[idx, factor_idx] = 1
        
        return exposures
    
    def generate_price_history(
        self,
        n_factors: Optional[int] = None,
        factor_strength: float = 0.7,
        noise_strength: float = 0.3
    ) -> pd.DataFrame:
        """
        Generate historical price data with a true factor structure.
        
        Args:
            n_factors: Number of latent factors (overrides params if provided)
            factor_strength: Proportion of variance explained by factors
            noise_strength: Proportion of variance explained by idiosyncratic noise
            
        Returns:
            DataFrame with historical prices
        """
        if n_factors is None:
            n_factors = self.params.n_factors
            
        dates = pd.date_range(
            start=self.params.start_date,
            end=self.params.end_date,
            freq='B'  # Business days
        )
        n_days = len(dates)
        
        # Initialize regime parameters
        self.regime_returns, self.regime_volatilities = self._generate_regime_parameters()
        
        # Generate factor correlation matrix
        factor_corr = self._generate_factor_correlation_matrix()
        
        # Generate asset exposures
        exposures = self._generate_asset_exposures()
        
        # Initialize factor returns
        factor_returns = np.zeros((n_days, n_factors))
        
        # Generate factor returns with persistence and regime changes
        for t in range(1, n_days):
            # Check for regime change
            if np.random.random() < self.params.regime_change_prob:
                self.regime = (self.regime + 1) % 2
                self.regime_returns, self.regime_volatilities = self._generate_regime_parameters()
            
            # Generate factor returns with persistence
            factor_returns[t] = (
                self.params.factor_persistence * factor_returns[t-1] +
                np.random.multivariate_normal(
                    self.regime_returns,
                    np.diag(self.regime_volatilities ** 2) @ factor_corr
                )
            )
        
        # Generate idiosyncratic noise
        noise = np.random.normal(
            0,
            noise_strength * self.params.base_volatility / np.sqrt(252),
            (n_days, self.params.n_assets)
        )
        
        # Calculate asset returns
        asset_returns = factor_strength * (factor_returns @ exposures.T) + noise
        
        # Add small drift
        asset_returns += self.params.base_return / 252
        
        # Convert to prices
        prices = (1 + asset_returns).cumprod(axis=0)
        
        # Create DataFrame
        df = pd.DataFrame(prices, index=dates, columns=self.assets)
        
        # Store factor returns for analysis
        self.factor_returns = pd.DataFrame(
            factor_returns,
            index=dates,
            columns=self.factor_names
        )
        
        return df
    
    def get_factor_returns(self) -> pd.DataFrame:
        """Get the generated factor returns."""
        if not hasattr(self, 'factor_returns'):
            raise ValueError("Must generate price history first")
        return self.factor_returns
    
    def _generate_correlation_matrix(self) -> np.ndarray:
        """Generate a realistic correlation matrix."""
        # Generate random correlation matrix
        corr = np.random.uniform(
            self.params.correlation_range[0],
            self.params.correlation_range[1],
            (self.params.n_assets, self.params.n_assets)
        )
        
        # Make it symmetric and ensure positive definiteness
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)
        
        # Ensure positive definiteness
        min_eig = np.min(np.real(np.linalg.eigvals(corr)))
        if min_eig < 0:
            corr -= min_eig * np.eye(self.params.n_assets)
            corr /= (1 - min_eig)
            
        return corr
    
    def _generate_volatilities(self) -> np.ndarray:
        """Generate realistic asset volatilities."""
        return np.random.lognormal(
            np.log(self.params.base_volatility),
            0.2,
            self.params.n_assets
        )
    
    def _generate_returns(self) -> np.ndarray:
        """Generate realistic expected returns."""
        return np.random.normal(
            self.params.base_return,
            self.params.base_volatility,
            self.params.n_assets
        )
    
    def _generate_esg_scores(self) -> Dict[str, ESGScore]:
        """Generate realistic ESG scores."""
        esg_scores = {}
        for asset in self.assets:
            esg_scores[asset] = ESGScore(
                environmental=np.random.uniform(*self.params.esg_score_range),
                social=np.random.uniform(*self.params.esg_score_range),
                governance=np.random.uniform(*self.params.esg_score_range),
                controversy=np.random.uniform(*self.params.controversy_range)
            )
        return esg_scores
    
    def generate_market_data(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, ESGScore]]:
        """
        Generate complete market data set.
        
        Returns:
            Tuple of (returns, covariance_matrix, esg_scores)
        """
        # Generate correlation matrix
        corr_matrix = self._generate_correlation_matrix()
        
        # Generate volatilities
        volatilities = self._generate_volatilities()
        
        # Generate covariance matrix
        cov_matrix = np.diag(volatilities) @ corr_matrix @ np.diag(volatilities)
        
        # Generate expected returns
        returns = self._generate_returns()
        
        # Generate ESG scores
        esg_scores = self._generate_esg_scores()
        
        return returns, cov_matrix, esg_scores 