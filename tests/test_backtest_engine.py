"""
Tests for the backtesting engine.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from apmti.backtesting.backtest_engine import BacktestEngine, BacktestResult

@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 12, 31),
        freq='B'
    )
    
    # Generate random price data
    n_assets = 5
    prices = pd.DataFrame(
        np.random.lognormal(
            mean=0.0002,
            sigma=0.02,
            size=(len(dates), n_assets)
        ).cumprod(axis=0),
        index=dates,
        columns=[f"ASSET_{i}" for i in range(n_assets)]
    )
    
    return prices

def test_backtest_engine_initialization(sample_price_data):
    """Test backtesting engine initialization."""
    engine = BacktestEngine(
        price_data=sample_price_data,
        initial_capital=1_000_000.0,
        transaction_cost=0.001,
        slippage_model='sqrt',
        rebalance_frequency='M'
    )
    
    assert engine.initial_capital == 1_000_000.0
    assert engine.transaction_cost == 0.001
    assert engine.slippage_model == 'sqrt'
    assert engine.rebalance_frequency == 'M'
    assert len(engine.assets) == 5

def test_slippage_calculation(sample_price_data):
    """Test slippage calculation."""
    engine = BacktestEngine(sample_price_data)
    
    # Test linear model
    engine.slippage_model = 'linear'
    slippage_linear = engine._calculate_slippage(1_000_000, 100)
    assert isinstance(slippage_linear, float)
    assert slippage_linear > 0
    
    # Test sqrt model
    engine.slippage_model = 'sqrt'
    slippage_sqrt = engine._calculate_slippage(1_000_000, 100)
    assert isinstance(slippage_sqrt, float)
    assert slippage_sqrt > 0

def test_transaction_costs(sample_price_data):
    """Test transaction cost calculation."""
    engine = BacktestEngine(sample_price_data)
    
    current_weights = {asset: 0.2 for asset in engine.assets}
    target_weights = {asset: 0.25 for asset in engine.assets}
    portfolio_value = 1_000_000.0
    
    costs = engine._calculate_transaction_costs(
        current_weights,
        target_weights,
        portfolio_value
    )
    
    assert isinstance(costs, float)
    assert costs > 0

def test_backtest_run(sample_price_data):
    """Test backtest run with simple weight generator."""
    engine = BacktestEngine(sample_price_data)
    
    def weight_generator(price_data, current_weights, portfolio_value):
        """Simple equal-weight strategy."""
        return {asset: 1.0/len(engine.assets) for asset in engine.assets}
    
    result = engine.run_backtest(weight_generator)
    
    assert isinstance(result, BacktestResult)
    assert isinstance(result.portfolio_value, pd.Series)
    assert isinstance(result.returns, pd.Series)
    assert isinstance(result.weights_history, pd.DataFrame)
    assert isinstance(result.metrics, dict)
    assert isinstance(result.trades, pd.DataFrame)
    
    # Check metrics
    assert 'total_return' in result.metrics
    assert 'annualized_return' in result.metrics
    assert 'volatility' in result.metrics
    assert 'sharpe_ratio' in result.metrics
    assert 'max_drawdown' in result.metrics
    assert 'turnover' in result.metrics
    
    # Check portfolio value
    assert result.portfolio_value.iloc[0] == engine.initial_capital
    assert not result.portfolio_value.isnull().any()
    
    # Check weights
    assert not result.weights_history.isnull().any()
    for col in result.weights_history.columns:
        assert abs(result.weights_history[col].sum() - 1.0) < 1e-6

def test_backtest_date_range(sample_price_data):
    """Test backtest with specific date range."""
    engine = BacktestEngine(sample_price_data)
    
    def weight_generator(price_data, current_weights, portfolio_value):
        return {asset: 1.0/len(engine.assets) for asset in engine.assets}
    
    start_date = datetime(2020, 3, 1)
    end_date = datetime(2020, 6, 30)
    
    result = engine.run_backtest(
        weight_generator,
        start_date=start_date,
        end_date=end_date
    )
    
    assert result.portfolio_value.index[0] >= start_date
    assert result.portfolio_value.index[-1] <= end_date

def test_backtest_rebalancing(sample_price_data):
    """Test backtest rebalancing frequency."""
    engine = BacktestEngine(
        sample_price_data,
        rebalance_frequency='W'  # Weekly rebalancing
    )
    
    def weight_generator(price_data, current_weights, portfolio_value):
        return {asset: 1.0/len(engine.assets) for asset in engine.assets}
    
    result = engine.run_backtest(weight_generator)
    
    # Check that trades occur approximately weekly
    trade_dates = result.trades['date'].unique()
    trade_dates = sorted(trade_dates)
    
    for i in range(1, len(trade_dates)):
        days_between = (trade_dates[i] - trade_dates[i-1]).days
        assert 4 <= days_between <= 7  # Business days between trades 