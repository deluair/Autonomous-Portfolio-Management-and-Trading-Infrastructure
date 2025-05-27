"""
Tests for the portfolio optimizer.
"""
import pytest
import numpy as np
from apmti.core.optimization.portfolio_optimizer import PortfolioOptimizer, ESGScore

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    n_assets = 5
    returns = np.array([0.08, 0.12, 0.15, 0.10, 0.09])
    cov_matrix = np.array([
        [0.04, 0.02, 0.01, 0.03, 0.02],
        [0.02, 0.09, 0.03, 0.02, 0.01],
        [0.01, 0.03, 0.16, 0.01, 0.02],
        [0.03, 0.02, 0.01, 0.25, 0.03],
        [0.02, 0.01, 0.02, 0.03, 0.36]
    ])
    
    esg_scores = {
        f"ASSET_{i}": ESGScore(
            environmental=0.7 + 0.1 * np.random.random(),
            social=0.6 + 0.2 * np.random.random(),
            governance=0.8 + 0.1 * np.random.random(),
            controversy=0.2 + 0.1 * np.random.random()
        )
        for i in range(n_assets)
    }
    
    return returns, cov_matrix, esg_scores

def test_portfolio_optimizer_initialization(sample_data):
    """Test portfolio optimizer initialization."""
    returns, cov_matrix, esg_scores = sample_data
    optimizer = PortfolioOptimizer(
        returns=returns,
        cov_matrix=cov_matrix,
        esg_scores=esg_scores
    )
    
    assert optimizer.n_assets == len(returns)
    assert optimizer.risk_free_rate == 0.02
    assert optimizer.min_controversy_score == 0.5
    assert optimizer.esg_weights == (0.33, 0.33, 0.34)

def test_portfolio_metrics(sample_data):
    """Test portfolio metrics calculation."""
    returns, cov_matrix, esg_scores = sample_data
    optimizer = PortfolioOptimizer(
        returns=returns,
        cov_matrix=cov_matrix,
        esg_scores=esg_scores
    )
    
    # Test with equal weights
    weights = np.array([0.2] * 5)
    ret, vol, sharpe = optimizer._portfolio_metrics(weights)
    
    assert isinstance(ret, float)
    assert isinstance(vol, float)
    assert isinstance(sharpe, float)
    assert vol > 0
    assert abs(ret - np.mean(returns)) < 0.1

def test_esg_score(sample_data):
    """Test ESG score calculation."""
    returns, cov_matrix, esg_scores = sample_data
    optimizer = PortfolioOptimizer(
        returns=returns,
        cov_matrix=cov_matrix,
        esg_scores=esg_scores
    )
    
    # Test with equal weights
    weights = np.array([0.2] * 5)
    assets = list(esg_scores.keys())
    esg_score = optimizer._esg_score(weights, assets)
    
    assert isinstance(esg_score, float)
    assert 0 <= esg_score <= 1

def test_controversy_constraint(sample_data):
    """Test controversy constraint calculation."""
    returns, cov_matrix, esg_scores = sample_data
    optimizer = PortfolioOptimizer(
        returns=returns,
        cov_matrix=cov_matrix,
        esg_scores=esg_scores
    )
    
    # Test with equal weights
    weights = np.array([0.2] * 5)
    assets = list(esg_scores.keys())
    constraint = optimizer._controversy_constraint(weights, assets)
    
    assert isinstance(constraint, float)

def test_optimization(sample_data):
    """Test portfolio optimization."""
    returns, cov_matrix, esg_scores = sample_data
    optimizer = PortfolioOptimizer(
        returns=returns,
        cov_matrix=cov_matrix,
        esg_scores=esg_scores
    )
    
    assets = list(esg_scores.keys())
    result = optimizer.optimize(
        assets=assets,
        target_return=0.10,
        max_volatility=0.20
    )
    
    assert isinstance(result, dict)
    assert 'weights' in result
    assert 'expected_return' in result
    assert 'volatility' in result
    assert 'sharpe_ratio' in result
    assert 'esg_score' in result
    assert 'controversy_score' in result
    
    # Check weights sum to 1
    assert abs(sum(result['weights'].values()) - 1.0) < 1e-6
    
    # Check constraints
    assert result['expected_return'] >= 0.10
    assert result['volatility'] <= 0.20
    assert result['controversy_score'] >= 0.5

def test_optimization_without_constraints(sample_data):
    """Test portfolio optimization without return/volatility constraints."""
    returns, cov_matrix, esg_scores = sample_data
    optimizer = PortfolioOptimizer(
        returns=returns,
        cov_matrix=cov_matrix,
        esg_scores=esg_scores
    )
    
    assets = list(esg_scores.keys())
    result = optimizer.optimize(assets=assets)
    
    assert isinstance(result, dict)
    assert 'weights' in result
    assert abs(sum(result['weights'].values()) - 1.0) < 1e-6
    assert result['controversy_score'] >= 0.5 