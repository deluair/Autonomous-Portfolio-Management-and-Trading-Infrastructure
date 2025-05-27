"""
Tests for the factor model implementation.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from apmti.models.factors.factor_model import FactorModel, FactorExposure

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Generate dates
    dates = pd.date_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 12, 31),
        freq='B'
    )
    
    # Generate random returns
    n_assets = 5
    n_factors = 3
    
    # Generate factor returns
    factor_returns = pd.DataFrame(
        np.random.normal(0.0002, 0.02, (len(dates), n_factors)),
        index=dates,
        columns=[f"Factor_{i+1}" for i in range(n_factors)]
    )
    
    # Generate asset returns with factor exposure
    asset_returns = pd.DataFrame(
        np.random.normal(0.0001, 0.015, (len(dates), n_assets)),
        index=dates,
        columns=[f"Asset_{i+1}" for i in range(n_assets)]
    )
    
    # Add factor exposure to asset returns
    for i in range(n_assets):
        factor_exposure = np.random.normal(0.5, 0.2, n_factors)
        for j in range(n_factors):
            asset_returns[f"Asset_{i+1}"] += factor_exposure[j] * factor_returns[f"Factor_{j+1}"]
    
    return asset_returns, factor_returns

def test_factor_model_initialization(sample_data):
    """Test factor model initialization."""
    returns, factor_data = sample_data
    model = FactorModel(
        returns=returns,
        factor_data=factor_data,
        min_half_life=20,
        max_half_life=252,
        min_t_stat=2.0,
        max_factors=3
    )
    
    assert model.min_half_life == 20
    assert model.max_half_life == 252
    assert model.min_t_stat == 2.0
    assert model.max_factors == 3
    assert model.factor_exposures == {}
    assert model.factor_returns is None
    assert model.residual_returns is None

def test_half_life_calculation(sample_data):
    """Test half-life calculation."""
    returns, _ = sample_data
    model = FactorModel(returns=returns, factor_data=pd.DataFrame())
    
    # Test with mean-reverting series
    series = pd.Series(np.sin(np.linspace(0, 10, 100)))
    half_life = model._calculate_half_life(series)
    
    assert isinstance(half_life, float)
    assert half_life > 0

def test_factor_decay_analysis(sample_data):
    """Test factor decay analysis."""
    returns, factor_data = sample_data
    model = FactorModel(returns=returns, factor_data=factor_data)
    
    # Test with factor returns
    factor_returns = factor_data['Factor_1']
    half_life = model._analyze_factor_decay(factor_returns)
    
    assert isinstance(half_life, float)
    assert half_life > 0

def test_factor_exposure_calculation(sample_data):
    """Test factor exposure calculation."""
    returns, factor_data = sample_data
    model = FactorModel(returns=returns, factor_data=factor_data)
    
    # Test with asset returns
    asset_returns = returns['Asset_1']
    exposures = model._calculate_factor_exposures(asset_returns, factor_data)
    
    assert isinstance(exposures, dict)
    for factor, exposure in exposures.items():
        assert isinstance(exposure, FactorExposure)
        assert isinstance(exposure.exposure, float)
        assert isinstance(exposure.t_stat, float)
        assert isinstance(exposure.p_value, float)
        assert isinstance(exposure.half_life, float)

def test_model_fitting(sample_data):
    """Test model fitting."""
    returns, factor_data = sample_data
    model = FactorModel(
        returns=returns,
        factor_data=factor_data,
        min_half_life=20,
        max_half_life=252,
        min_t_stat=2.0,
        max_factors=3
    )
    
    model.fit()
    
    assert model.factor_returns is not None
    assert model.residual_returns is not None
    assert len(model.factor_exposures) == len(returns.columns)
    
    # Check factor returns
    assert isinstance(model.factor_returns, pd.DataFrame)
    assert len(model.factor_returns.columns) <= model.max_factors
    
    # Check residual returns
    assert isinstance(model.residual_returns, pd.DataFrame)
    assert model.residual_returns.shape == returns.shape

def test_return_prediction(sample_data):
    """Test return prediction."""
    returns, factor_data = sample_data
    model = FactorModel(returns=returns, factor_data=factor_data)
    
    model.fit()
    
    # Generate new factor returns
    new_dates = pd.date_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 10),
        freq='B'
    )
    new_factor_returns = pd.DataFrame(
        np.random.normal(0.0002, 0.02, (len(new_dates), len(model.factor_returns.columns))),
        index=new_dates,
        columns=model.factor_returns.columns
    )
    
    # Predict returns
    predicted_returns = model.predict_returns(new_factor_returns)
    
    assert isinstance(predicted_returns, pd.DataFrame)
    assert predicted_returns.shape[0] == len(new_dates)
    assert predicted_returns.shape[1] == len(returns.columns)

def test_factor_importance(sample_data):
    """Test factor importance calculation."""
    returns, factor_data = sample_data
    model = FactorModel(returns=returns, factor_data=factor_data)
    
    model.fit()
    
    importance = model.get_factor_importance()
    
    assert isinstance(importance, pd.DataFrame)
    assert 'factor' in importance.columns
    assert 'n_significant' in importance.columns
    assert 'avg_exposure' in importance.columns
    assert 'avg_half_life' in importance.columns
    assert 'volatility' in importance.columns
    
    # Check values
    assert importance['n_significant'].min() >= 0
    assert importance['avg_half_life'].min() >= model.min_half_life
    assert importance['avg_half_life'].max() <= model.max_half_life
    assert importance['volatility'].min() >= 0 