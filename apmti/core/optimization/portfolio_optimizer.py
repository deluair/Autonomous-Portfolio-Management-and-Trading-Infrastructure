"""
Portfolio optimization module implementing multi-objective optimization with ESG constraints.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass

@dataclass
class ESGScore:
    """ESG score data structure."""
    environmental: float
    social: float
    governance: float
    controversy: float

class PortfolioOptimizer:
    """
    Multi-objective portfolio optimizer with ESG constraints.
    """
    def __init__(
        self,
        returns: np.ndarray,
        cov_matrix: np.ndarray,
        esg_scores: Dict[str, ESGScore],
        risk_free_rate: float = 0.02,
        min_controversy_score: float = 0.5,
        esg_weights: Tuple[float, float, float] = (0.33, 0.33, 0.34)
    ):
        """
        Initialize the portfolio optimizer.
        
        Args:
            returns: Expected returns for each asset
            cov_matrix: Covariance matrix of returns
            esg_scores: Dictionary mapping asset symbols to ESG scores
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            min_controversy_score: Minimum acceptable controversy score
            esg_weights: Weights for E, S, G components in optimization
        """
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.esg_scores = esg_scores
        self.risk_free_rate = risk_free_rate
        self.min_controversy_score = min_controversy_score
        self.esg_weights = esg_weights
        self.n_assets = len(returns)

    def _portfolio_metrics(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio metrics for given weights.
        
        Args:
            weights: Asset weights
            
        Returns:
            Tuple of (expected_return, volatility, sharpe_ratio)
        """
        portfolio_return = np.sum(self.returns * weights)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
        return portfolio_return, portfolio_vol, sharpe_ratio

    def _esg_score(self, weights: np.ndarray, assets: List[str]) -> float:
        """
        Calculate weighted ESG score for portfolio.
        
        Args:
            weights: Asset weights
            assets: List of asset symbols
            
        Returns:
            Weighted ESG score
        """
        e_score = sum(weights[i] * self.esg_scores[asset].environmental 
                     for i, asset in enumerate(assets))
        s_score = sum(weights[i] * self.esg_scores[asset].social 
                     for i, asset in enumerate(assets))
        g_score = sum(weights[i] * self.esg_scores[asset].governance 
                     for i, asset in enumerate(assets))
        
        return (self.esg_weights[0] * e_score + 
                self.esg_weights[1] * s_score + 
                self.esg_weights[2] * g_score)

    def _controversy_constraint(self, weights: np.ndarray, assets: List[str]) -> float:
        """
        Calculate controversy constraint violation.
        
        Args:
            weights: Asset weights
            assets: List of asset symbols
            
        Returns:
            Constraint violation (should be <= 0)
        """
        controversy_score = sum(weights[i] * self.esg_scores[asset].controversy 
                              for i, asset in enumerate(assets))
        return self.min_controversy_score - controversy_score

    def optimize(
        self,
        assets: List[str],
        target_return: Optional[float] = None,
        max_volatility: Optional[float] = None
    ) -> Dict:
        """
        Optimize portfolio weights with ESG constraints.
        
        Args:
            assets: List of asset symbols
            target_return: Optional target return constraint
            max_volatility: Optional maximum volatility constraint
            
        Returns:
            Dictionary containing optimization results
        """
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
            {'type': 'ineq', 'fun': lambda x: self._controversy_constraint(x, assets)}
        ]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: self._portfolio_metrics(x)[0] - target_return
            })
        
        if max_volatility is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: max_volatility - self._portfolio_metrics(x)[1]
            })

        bounds = [(0, 1) for _ in range(self.n_assets)]  # long-only constraint
        
        # Initial guess
        x0 = np.array([1/self.n_assets] * self.n_assets)
        
        # Objective function: maximize Sharpe ratio while considering ESG score
        def objective(weights):
            ret, vol, sharpe = self._portfolio_metrics(weights)
            esg_score = self._esg_score(weights, assets)
            return -sharpe - 0.1 * esg_score  # negative because we minimize
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-8, 'disp': True}
        )
        
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        
        weights = result.x
        ret, vol, sharpe = self._portfolio_metrics(weights)
        esg_score = self._esg_score(weights, assets)
        
        return {
            'weights': dict(zip(assets, weights)),
            'expected_return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'esg_score': esg_score,
            'controversy_score': sum(weights[i] * self.esg_scores[asset].controversy 
                                   for i, asset in enumerate(assets))
        } 