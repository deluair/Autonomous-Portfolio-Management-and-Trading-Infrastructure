# Autonomous Portfolio Management and Trading Infrastructure (APMTI)

A sophisticated investment management system that combines multi-objective portfolio optimization, systematic trading, and electronic market making with integrated ESG considerations.

## Features

- Multi-objective portfolio optimization with ESG constraints
- High-frequency backtesting infrastructure
- Electronic market making algorithms
- Factor model development and alpha decay analysis
- Real-time portfolio risk monitoring
- Automated rebalancing systems

## Project Structure

```
apmti/
├── core/                 # Core system components
│   ├── optimization/     # Portfolio optimization modules
│   ├── trading/         # Trading system components
│   ├── risk/            # Risk management modules
│   └── market_making/   # Market making algorithms
├── data/                # Data management and processing
│   ├── ingestion/       # Data ingestion pipelines
│   ├── processing/      # Data processing modules
│   └── storage/         # Data storage interfaces
├── models/              # ML models and factor models
│   ├── factors/         # Factor model implementations
│   ├── ml/             # Machine learning models
│   └── esg/            # ESG scoring models
├── backtesting/         # Backtesting framework
├── monitoring/          # System monitoring and logging
└── utils/              # Utility functions and helpers
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/apmti.git
cd apmti
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

[Usage instructions will be added as the system develops]

## Development

- Follow PEP 8 style guide
- Run tests with pytest
- Use black for code formatting
- Run mypy for type checking

## License

[License information to be added]

## Factor Model System

The APMTI includes a sophisticated factor model system for risk and alpha analysis:
- **Automatic factor extraction** (PCA-based)
- **Alpha decay analysis** (half-life calculation)
- **Statistical significance filtering** (t-stat, half-life)
- **Residual return analysis**
- **Factor importance metrics**
- **Comprehensive visualization**

### Running the Factor Model Example

To run the factor model example and generate analysis/plots:

```bash
python examples/factor_model_example.py
```

This will output:
- Factor importance table
- Factor exposures summary (first 5 assets)
- Several plots (saved as PNG files):
  - `factor_returns.png`: Cumulative factor returns
  - `factor_importance.png`: Number of significant exposures (if any)
  - `factor_half_lives.png`: Factor decay analysis (if any)
  - `residual_returns.png`: Distribution of residual returns (if any)
  - `factor_exposures.png`: Heatmap of factor exposures (if any)
  - `rolling_importance.png`: Rolling factor importance (if any)

**Note:**
- If you see warnings about "no significant exposures" or missing plots, this is expected with random or unstructured synthetic data. For meaningful results, use real or more structured synthetic data.
- The script is robust to empty/NaN results and will skip plots or print warnings as needed. 