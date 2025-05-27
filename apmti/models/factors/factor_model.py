"""
Factor model implementation with alpha decay analysis and regime detection.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import adfuller

@dataclass
class FactorExposure:
    """Factor exposure data structure."""
    factor_name: str
    exposure: float
    t_stat: float
    p_value: float
    half_life: float  # in days
    regime_sensitivity: float  # sensitivity to regime changes
    persistence: float  # AR(1) coefficient

@dataclass
class RegimeInfo:
    """Regime information data structure."""
    regime_id: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    factor_returns: pd.DataFrame
    factor_volatilities: pd.Series
    factor_correlations: pd.DataFrame

class FactorModel:
    """
    Factor model implementation with alpha decay analysis and regime detection.
    """
    def __init__(
        self,
        returns: pd.DataFrame,
        factor_data: pd.DataFrame,
        min_half_life: int = 20,  # minimum half-life in days
        max_half_life: int = 252,  # maximum half-life in days
        min_t_stat: float = 2.0,  # minimum t-statistic for factor significance
        max_factors: int = 5,  # maximum number of factors to use
        regime_window: int = 63,  # window size for regime detection (3 months)
        regime_threshold: float = 0.7  # threshold for regime change detection
    ):
        """
        Initialize the factor model.
        
        Args:
            returns: DataFrame of asset returns (index: dates, columns: assets)
            factor_data: DataFrame of factor returns (index: dates, columns: factors)
            min_half_life: Minimum half-life for factor decay
            max_half_life: Maximum half-life for factor decay
            min_t_stat: Minimum t-statistic for factor significance
            max_factors: Maximum number of factors to use
            regime_window: Window size for regime detection
            regime_threshold: Threshold for regime change detection
        """
        self.returns = returns
        self.factor_data = factor_data
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.min_t_stat = min_t_stat
        self.max_factors = max_factors
        self.regime_window = regime_window
        self.regime_threshold = regime_threshold
        self.factor_exposures: Dict[str, Dict[str, FactorExposure]] = {}
        self.factor_returns: Optional[pd.DataFrame] = None
        self.residual_returns: Optional[pd.DataFrame] = None
        self.regimes: List[RegimeInfo] = []
        
    def _detect_regimes(self) -> List[RegimeInfo]:
        """
        Detect market regimes using factor returns.
        
        Returns:
            List of regime information
        """
        if self.factor_returns is None:
            raise ValueError("Must fit model before detecting regimes")
            
        # Calculate rolling statistics
        rolling_vol = self.factor_returns.rolling(self.regime_window).std()
        rolling_corr = self.factor_returns.rolling(self.regime_window).corr()
        
        # Detect regime changes using K-means clustering
        features = pd.concat([
            rolling_vol,
            rolling_corr.unstack().rolling(self.regime_window).mean()
        ], axis=1)
        
        # Remove NaN values
        features = features.dropna()
        
        # Cluster into regimes
        kmeans = KMeans(n_clusters=2, random_state=42)
        regime_labels = kmeans.fit_predict(features)
        
        # Identify regime changes
        regime_changes = np.where(np.diff(regime_labels) != 0)[0]
        regime_changes = np.concatenate([[0], regime_changes + 1, [len(regime_labels)]])
        
        # Create regime information
        regimes = []
        for i in range(len(regime_changes) - 1):
            start_idx = regime_changes[i]
            end_idx = regime_changes[i + 1]
            
            regime_returns = self.factor_returns.iloc[start_idx:end_idx]
            regime_vols = rolling_vol.iloc[start_idx:end_idx].mean()
            regime_corrs = rolling_corr.iloc[start_idx:end_idx].mean()
            
            regimes.append(RegimeInfo(
                regime_id=regime_labels[start_idx],
                start_date=self.factor_returns.index[start_idx],
                end_date=self.factor_returns.index[end_idx - 1],
                factor_returns=regime_returns,
                factor_volatilities=regime_vols,
                factor_correlations=regime_corrs
            ))
        
        return regimes
    
    def _calculate_persistence(self, series: pd.Series) -> float:
        """
        Calculate AR(1) persistence coefficient.
        
        Args:
            series: Time series to analyze
            
        Returns:
            AR(1) coefficient
        """
        # Run AR(1) regression
        y = series.values[1:]
        X = series.values[:-1].reshape(-1, 1)
        model = stats.linregress(X.flatten(), y)
        
        return model.slope
    
    def _calculate_regime_sensitivity(
        self,
        asset_returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> float:
        """
        Calculate asset's sensitivity to regime changes.
        
        Args:
            asset_returns: Asset return series
            factor_returns: Factor return DataFrame
            
        Returns:
            Regime sensitivity coefficient
        """
        if not self.regimes:
            return 0.0
            
        # Calculate regime-specific betas
        regime_betas = []
        for regime in self.regimes:
            regime_asset_returns = asset_returns[regime.start_date:regime.end_date]
            regime_factor_returns = regime.factor_returns
            
            # Calculate beta in this regime
            beta = np.cov(regime_asset_returns, regime_factor_returns.mean(axis=1))[0, 1] / \
                   np.var(regime_factor_returns.mean(axis=1))
            regime_betas.append(beta)
        
        # Calculate sensitivity as standard deviation of regime betas
        return np.std(regime_betas)
    
    def _calculate_factor_exposures(
        self,
        asset_returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> Dict[str, FactorExposure]:
        """
        Calculate factor exposures for an asset.
        
        Args:
            asset_returns: Asset return series
            factor_returns: Factor return DataFrame
            
        Returns:
            Dictionary of factor exposures
        """
        exposures = {}
        
        for factor in factor_returns.columns:
            # Run regression
            model = stats.linregress(factor_returns[factor], asset_returns)
            
            # Calculate t-statistic
            n = len(asset_returns)
            dof = n - 2  # degrees of freedom
            mse = np.sum((asset_returns - model.intercept - model.slope * factor_returns[factor]) ** 2) / dof
            std_error = np.sqrt(mse / np.sum((factor_returns[factor] - factor_returns[factor].mean()) ** 2))
            t_stat = model.slope / std_error
            
            # Calculate p-value
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof))
            
            # Calculate half-life
            half_life = self._analyze_factor_decay(factor_returns[factor])
            
            # Calculate persistence
            persistence = self._calculate_persistence(factor_returns[factor])
            
            # Calculate regime sensitivity
            regime_sensitivity = self._calculate_regime_sensitivity(asset_returns, factor_returns)
            
            exposures[factor] = FactorExposure(
                factor_name=factor,
                exposure=model.slope,
                t_stat=t_stat,
                p_value=p_value,
                half_life=half_life,
                regime_sensitivity=regime_sensitivity,
                persistence=persistence
            )
        
        return exposures
    
    def fit(self) -> None:
        """
        Fit the factor model to the data.
        """
        # Calculate factor returns using PCA
        pca = PCA(n_components=self.max_factors)
        factor_returns = pd.DataFrame(
            pca.fit_transform(self.returns),
            index=self.returns.index,
            columns=[f"Factor_{i+1}" for i in range(self.max_factors)]
        )
        
        # Detect regimes
        self.regimes = self._detect_regimes()
        
        # Calculate factor exposures for each asset
        factor_exposures = {}
        residual_returns = pd.DataFrame(index=self.returns.index, columns=self.returns.columns)
        
        for asset in self.returns.columns:
            # Calculate exposures
            exposures = self._calculate_factor_exposures(
                self.returns[asset],
                factor_returns
            )
            
            # Filter significant factors
            significant_exposures = {
                factor: exp for factor, exp in exposures.items()
                if (abs(exp.t_stat) >= self.min_t_stat and
                    self.min_half_life <= exp.half_life <= self.max_half_life)
            }
            
            factor_exposures[asset] = significant_exposures
            
            # Calculate residual returns
            predicted_returns = pd.Series(0, index=self.returns.index)
            for factor, exp in significant_exposures.items():
                predicted_returns += exp.exposure * factor_returns[factor]
            
            residual_returns[asset] = self.returns[asset] - predicted_returns
        
        self.factor_returns = factor_returns
        self.factor_exposures = factor_exposures
        self.residual_returns = residual_returns
    
    def predict_returns(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Predict asset returns using factor model.
        
        Args:
            factor_returns: Factor return DataFrame
            
        Returns:
            DataFrame of predicted returns
        """
        if not self.factor_exposures:
            raise ValueError("Model must be fitted before making predictions")
        
        predicted_returns = pd.DataFrame(
            index=factor_returns.index,
            columns=self.returns.columns
        )
        
        for asset in self.returns.columns:
            asset_exposures = self.factor_exposures[asset]
            predicted_returns[asset] = sum(
                exp.exposure * factor_returns[factor]
                for factor, exp in asset_exposures.items()
            )
        
        return predicted_returns
    
    def get_factor_importance(self) -> pd.DataFrame:
        """
        Get factor importance metrics.
        
        Returns:
            DataFrame with factor importance metrics
        """
        if not self.factor_exposures:
            raise ValueError("Model must be fitted before getting factor importance")
        
        # Calculate factor importance metrics
        importance_data = []
        
        for factor in self.factor_returns.columns:
            # Count significant exposures
            n_significant = sum(
                1 for exposures in self.factor_exposures.values()
                if factor in exposures
            )
            
            # Calculate average exposure
            avg_exposure = np.mean([
                exposures[factor].exposure
                for exposures in self.factor_exposures.values()
                if factor in exposures
            ])
            
            # Calculate average half-life
            avg_half_life = np.mean([
                exposures[factor].half_life
                for exposures in self.factor_exposures.values()
                if factor in exposures
            ])
            
            # Calculate average regime sensitivity
            avg_regime_sensitivity = np.mean([
                exposures[factor].regime_sensitivity
                for exposures in self.factor_exposures.values()
                if factor in exposures
            ])
            
            # Calculate average persistence
            avg_persistence = np.mean([
                exposures[factor].persistence
                for exposures in self.factor_exposures.values()
                if factor in exposures
            ])
            
            # Calculate factor return volatility
            factor_vol = self.factor_returns[factor].std() * np.sqrt(252)
            
            importance_data.append({
                'factor': factor,
                'n_significant': n_significant,
                'avg_exposure': avg_exposure,
                'avg_half_life': avg_half_life,
                'avg_regime_sensitivity': avg_regime_sensitivity,
                'avg_persistence': avg_persistence,
                'volatility': factor_vol
            })
        
        return pd.DataFrame(importance_data)
    
    def get_regime_info(self) -> List[RegimeInfo]:
        """
        Get information about detected regimes.
        
        Returns:
            List of regime information
        """
        if not self.regimes:
            raise ValueError("Model must be fitted before getting regime information")
        return self.regimes 