# CO₂ Emissions Analysis Project

A machine learning project for analyzing and predicting CO₂ emissions based on economic and energy indicators.

## 📋 Project Overview

This project implements a Random Forest model for time series forecasting of CO₂ emissions. The analysis focuses on:
- Time series analysis with lagged variables
- Panel data processing (countries × years)
- Feature importance analysis
- Result visualization

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/project_programming_for_ai.git
   cd project_programming_for_ai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the analysis:
   ```bash
   python src/analysis/rf_example.py
   ```

## 🏗️ Project Structure

```
project_programming_for_ai/
├── src/                    # Source code
│   └── analysis/          # Analysis and models
│       ├── preprocessing.py  # Data preprocessing
│       └── rf_example.py     # Random Forest analysis
├── temp_data/             # Intermediate data files
└── results/               # Analysis results
```

## 🔍 Example Usage

### Training the Model
```python
# Run analysis with default parameters
from src.analysis.rf_example import main

main()
```

### Customizing Parameters
You can adjust model parameters in `rf_example.py`:
- `split_year` - train/test split year
- `base_features` - list of features to use
- Random Forest hyperparameters

## 📊 Sample Results

### Feature Importance
![Feature Importance](src/analysis/results/analysis/feature_importance.png)

### Predictions vs Actual
![Predictions](src/analysis/results/analysis/predictions_plot.png)

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
