# CO₂ Emissions Analysis Project (Old Version)

This directory contains the initial version of the project, which focused on analyzing and predicting CO₂ emissions based on economic and energy indicators using Random Forest models.

## Project Overview

This project implements a Random Forest model for time series forecasting of CO₂ emissions. The analysis focuses on:
- Time series analysis with lagged variables
- Panel data processing (countries × years)
- Feature importance analysis
- Result visualization

## Project Structure

```
old_version/
├── src/                # Source code
│   └── diagnostic/     # Diagnostic tools and analysis
└── results/            # Analysis results
    └── analysis/       # Generated plots and visualizations
```

## How to Run

1. Navigate to the project root directory
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the analysis:
   ```bash
   python src/diagnostic/rf_example.py
   ```

---
*This was the initial version of the project, which has since been replaced by the financial market prediction system in the parent directory.*
