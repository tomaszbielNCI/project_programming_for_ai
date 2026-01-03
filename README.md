# Financial Market Prediction with Bayesian Neural Networks

This project implements a Bayesian Neural Network (BNN) for probabilistic forecasting of financial market returns, with a focus on high-frequency data (HFD) analysis. The project was developed as part of the National College of Ireland's Programming for AI module.

## 📋 Project Overview

This project focuses on:
- Processing and analyzing high-frequency financial market data
- Implementing Bayesian Neural Networks for probabilistic forecasting
- Evaluating model performance using calibration metrics and uncertainty estimation
- Visualizing predictions with confidence intervals

## 🏗️ Project Structure

```
project_programming_for_ai/
├── data/                    # Data storage
│   ├── hfd/                # Raw high-frequency data (tick data)
│   │   └── [symbol]/       # Per-instrument HFD data
│   ├── historical/         # Historical market data (1m, 15m, 1h, etc.)
│   └── parsed/             # Processed data files
│       ├── US.100.parquet              # Raw HFD data
│       └── US.100_1min_minimal.parquet # Processed 1-min bars
│
├── old_version/            # Previous version of the project
│   ├── src/               # Old source code
│   └── results/           # Old results
│
├── notebooks/              # Jupyter notebooks for analysis
├── results/                # Model outputs and visualizations
│   ├── bnn_v1/            # V1: Standard 1-minute model results
│   │   ├── models/        # V1 model checkpoints
│   │   └── predictions/   # V1 prediction outputs
│   ├── bnn_v2/            # V2: Enhanced 1-minute model results
│   │   ├── models/        # V2 model checkpoints
│   │   └── predictions/   # V2 prediction outputs
│   └── bnn_hfd/           # HFD-optimized model results
│       ├── models/        # HFD model checkpoints
│       └── predictions/   # HFD prediction outputs
│
└── src/                    # Source code
    ├── cleaning/          # Data cleaning utilities
    ├── diagnostic/        # Diagnostic tools
    ├── ingesting/         # Data ingestion modules
    │   ├── fast_hfd_aggregator.py  # HFD → 1-min aggregation
    │   └── parser_arrow.py         # Raw data parsing
    │
    ├── loaders/           # Data loading utilities
    │   ├── fast_hfd_loader.py     # Optimized HFD loader
    │   └── data_IO.py             # General data I/O
    │
    └── models/            # Model implementations
        └── bnn/           # BNN model code
            # --- 1-Minute Data Models ---
            ├── train_bnn_v2.py            # V2: Initial implementation
            ├── predict_rolling_bnn_v2.py   # V2: Enhanced predictions
            ├── viz_bnn_predictions.py      # V1 visualizations
            │
            # --- HFD Models ---
            ├── train_bnn_hfd.py            # HFD-optimized training
            ├── predict_rolling_bnn_hfd.py   # HFD predictions
            └── viz_bnn_predictions_hfd.py   # HFD visualizations
```

## 🔄 Vertical Slices

### 1. 1-Minute Data Models
- **V1 (Initial Implementation)**
  - Training: `train_bnn_v1.py`
  - Prediction: `predict_rolling_bnn_v1.py`
  - Visualization: `viz_bnn_predictions.py`
  - Results: `results/bnn_v1/`

- **V2 (Enhanced 1-Minute)**
  - Training: `train_bnn_v2.py`
  - Prediction: `predict_rolling_bnn_v2.py`
  - Visualization: `viz_bnn_predictions.py` (shared with V1)
  - Results: `results/bnn_v2/`

### 2. HFD-Optimized Models
- **HFD Implementation**
  - Training: `train_bnn_hfd.py`
  - Prediction: `predict_rolling_bnn_hfd.py`
  - Visualization: `viz_bnn_predictions_hfd.py`
  - Results: `results/bnn_hfd/`

## 🚀 Quick Start

### For 1-Minute Data (V2 - Recommended)
```bash
# Train the model
python src/models/bnn/train_bnn_v2.py

# Generate predictions
python src/models/bnn/predict_rolling_bnn_v2.py

# Visualize results
python src/models/bnn/viz_bnn_predictions.py
```

### For HFD Data
```bash
# Train the HFD-optimized model
python src/models/bnn/train_bnn_hfd.py

# Generate HFD predictions
python src/models/bnn/predict_rolling_bnn_hfd.py

# Visualize HFD results
python src/models/bnn/viz_bnn_predictions_hfd.py
```

## 🔄 Data Processing Pipelines

### 1. High-Frequency Data (HFD) Pipeline
- **Input**: Raw tick data in parquet format
- **Processing**:
  - Aggregated to 1-minute bars using `fast_hfd_aggregator.py`
  - Minimal features: timestamp, mid price, tick count, log returns
- **Output**: `US.100_1min_minimal.parquet`

### 2. 1-Minute Data Pipeline
- **Input**: Pre-processed 1-minute bars
- **Processing**:
  - Feature engineering for BNN model
  - Time-based train/test split
  - Normalization/standardization
- **Output**: Ready-to-use dataset for model training

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required packages (see `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tomaszbielNCI/project_programming_for_ai.git
   cd project_programming_for_ai
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🧠 Model Training

### Training the BNN Model
```bash
python src/models/bnn/train_bnn_hfd.py
```

### Making Predictions
```bash
python src/models/bnn/predict_rolling_bnn_hfd.py
```

### Generating Visualizations
```bash
python src/models/bnn/viz_bnn_predictions_hfd.py
```

## 📊 Key Features

- **Probabilistic Forecasting**: Predicts both the expected return and uncertainty
- **Laplace Calibration**: Implements proper scoring rules for probabilistic forecasts
- **High-Frequency Data Processing**: Specialized tools for handling HFD
- **Comprehensive Visualization**: Generates detailed plots of predictions and uncertainty

## 📝 Project Report Structure (In Progress)

The project report will include the following sections (to be completed):
1. **Abstract**: [To be completed] - Summary of objectives, methods, and results
2. **Introduction**: [To be completed] - Motivation and project objectives
3. **Related Work**: [To be completed] - Critical review of relevant literature
4. **Methodology**: [Partially completed]
   - [✓] Data description and preprocessing
   - [✓] BNN architecture and training
   - [ ] Evaluation metrics (in progress)
5. **Results**: [In progress] - Analysis of model performance
6. **Conclusions**: [To be completed] - Summary of findings and future work

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- Tomasz Biel


## 📚 References



---
*This project was submitted in partial fulfillment of the requirements for the Programming for AI module at the National College of Ireland.*
