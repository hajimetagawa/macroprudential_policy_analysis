# Macroprudential Policy Analysis

## Overview

This project analyzes the relationship between macroeconomic indicators and macroprudential policy implementation using international panel data. The analysis focuses on identifying which macroeconomic factors influence the probability and intensity of macroprudential policy activation.

## Project Structure

```
macroprudential_policy_analysis/
├── config/                 # Configuration files
│   ├── config.yaml        # Main configuration
│   ├── countries.yaml     # Country definitions and scope
│   └── data_sources.yaml  # Data source configurations
├── src/                   # Source code modules
│   ├── iMaPP_data_loader.py       # iMaPP database loader
│   ├── fetch_api_data.py          # BIS API data fetcher
│   ├── iMaPP_transformer.py       # Data transformation utilities
│   ├── yaml_loader.py             # YAML configuration loader
│   ├── process_datasets.py        # Dataset processing pipeline
│   ├── build_dataset.py           # Dataset merging and construction
│   └── dataset_assessment.py      # Data quality assessment
├── scripts/               # Executable scripts
│   ├── run_pipeline.py    # Main data processing pipeline
│   └── run_analysis.py    # Statistical analysis and visualization
├── data/                  # Data directories (not tracked in git)
│   ├── raw/              # Raw data files
│   ├── processed/        # Processed data files
│   └── dataset/          # Final analysis datasets
├── results/              # Analysis results and visualizations
├── notebooks/            # Jupyter notebooks for exploration
└── logs/                 # Log files
```

## Data Sources

### Primary Data
- **iMaPP Database**: International database of Macroprudential Policies (IMF)
  - Policy activation indicators (binary and count)
  - Country and time coverage: 1990-2023

### Macroeconomic Indicators (BIS APIs)
- Credit gap (actual and trend)
- Residential property price changes
- Debt service ratios (private non-financial sector)
- Central bank policy rates
- Total credit measures
- Effective exchange rates
- Commercial property prices

## Key Features

### Data Processing Pipeline
1. **Data Loading**: Automated loading from Excel files and BIS APIs
2. **Data Transformation**: Standardization and cleaning procedures
3. **Quality Assessment**: Comprehensive data quality checks
4. **Dataset Construction**: Merging multiple data sources with proper alignment

### Statistical Analysis
1. **Logit Models**: Policy activation probability analysis
2. **Count Models**: Policy intensity analysis (Negative Binomial/OLS)
3. **Marginal Effects**: Quantification of economic significance
4. **Fixed Effects**: Country and time controls

### Visualization
- Coefficient plots with statistical significance
- Time series trends of policy activation
- Correlation matrices of macroeconomic indicators
- Distribution comparisons (policy vs. no-policy periods)
- Model performance evaluation (ROC curves)

## Installation and Setup

### Prerequisites
- Python 3.8+
- Required packages (see requirements section)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd macroprudential_policy_analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Python Packages
```
pandas>=1.5.0
numpy>=1.20.0
statsmodels>=0.13.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
pyyaml>=6.0
requests>=2.28.0
openpyxl>=3.0.0
```

## Usage

### 1. Data Processing Pipeline
```bash
cd scripts
python run_pipeline.py
```

This will:
- Download data from BIS APIs
- Load and process iMaPP data
- Merge datasets
- Apply country filtering
- Generate final analysis dataset

### 2. Statistical Analysis
```bash
cd scripts
python run_analysis.py
```

This will:
- Run logit and count models
- Calculate marginal effects
- Generate visualizations
- Save results to `results/` folder

### Configuration

Edit configuration files in `config/`:
- `config.yaml`: Main settings (paths, logging, output options)
- `countries.yaml`: Country scope and classifications
- `data_sources.yaml`: BIS API endpoints and parameters

## Analysis Methodology

### Research Question
**"Which macroeconomic indicators influence the probability and intensity of macroprudential policy implementation?"**

### Statistical Models

#### 1. Policy Activation Probability (Logit Model)
```
P(Policy_t = 1) = Λ(β₀ + β₁X₁ᵢₜ + β₂X₂ᵢₜ + ... + αᵢ + δₜ + εᵢₜ)
```

Where:
- `Λ` is the logistic cumulative distribution function
- `Xᵢₜ` are macroeconomic indicators (standardized)
- `αᵢ` are country fixed effects (optional)
- `δₜ` are year fixed effects
- `εᵢₜ` is the error term

#### 2. Policy Intensity (Count Model)
```
E[Policy_Count_it] = exp(β₀ + β₁X₁ᵢₜ + β₂X₂ᵢₜ + ... + δₜ + εᵢₜ)
```

### Key Variables
- **Dependent Variables**:
  - `obs_bin`: Policy activation (0/1)
  - `obs_agg`: Number of policies activated
- **Independent Variables**:
  - Credit gap measures
  - Property price growth rates
  - Debt service ratios
  - Policy interest rates
  - Exchange rate measures

## Results and Interpretation

The analysis produces:

1. **Coefficient Estimates**: Log-odds effects on policy probability
2. **Marginal Effects**: Percentage point changes in activation probability
3. **Statistical Significance**: P-values and confidence intervals
4. **Model Fit**: Pseudo R² and information criteria
5. **Visualizations**: Charts saved in `results/` folder

## Sample Output Files

After running the analysis, the following files are generated in `results/`:
- `policy_coefficients_and_marginal_effects.png`
- `policy_trends_over_time.png`
- `indicators_correlation_matrix.png`
- `indicators_distribution_comparison.png`
- `model_performance_evaluation.png`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit your changes (`git commit -am 'Add new analysis'`)
4. Push to the branch (`git push origin feature/new-analysis`)
5. Create a Pull Request

## Data Privacy and Usage

- This project uses publicly available data from the IMF and BIS
- Raw data files are not included in the repository due to size constraints
- Users must download data independently using the provided scripts
- Ensure compliance with data source terms of use

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or collaboration opportunities, please open an issue in the GitHub repository.

## Citation

If you use this code or methodology in your research, please cite:
```
[Your Name/Institution] (2024). Macroprudential Policy Analysis: 
An Empirical Study of Policy Response Functions. 
GitHub: https://github.com/[username]/macroprudential_policy_analysis
```

## Acknowledgments

- International Monetary Fund (IMF) for the iMaPP database
- Bank for International Settlements (BIS) for macroeconomic data APIs
- Contributors to the open-source Python ecosystem used in this project