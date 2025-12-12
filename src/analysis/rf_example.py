import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import DataIO from loaders
from src.loaders.data_IO import DataIO

# Configure display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set_style("whitegrid")


def get_next_results_dir(base_path='results/analysis'):
    """Generate unique results directory with index."""
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    # Find next available index
    index = 1
    while (base_path / f"run_{index:03d}").exists():
        index += 1

    results_dir = base_path / f"run_{index:03d}"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Created results directory: {results_dir} (run_{index:03d}_{timestamp})")
    return results_dir


def save_results_txt(results_dir, features, y_test, y_pred, metrics, feature_importance):
    """Save all analysis results to TEXT file."""
    results_file = results_dir / "results.countries_of_interest.txt"
    
    # Reset index to ensure we have sequential indices
    y_test = y_test.reset_index(drop=True)

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EMISSIONS PREDICTION - RANDOM FOREST ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Run ID: {results_dir.name}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("MODEL PERFORMANCE:\n")
        f.write("-" * 40 + "\n")
        f.write(f"R² Score:        {metrics['r2']:.4f}\n")
        f.write(f"MSE:             {metrics['mse']:.2f}\n")
        f.write(f"MAE:             {metrics['mae']:.2f}\n\n")

        f.write("FEATURES USED:\n")
        f.write("-" * 40 + "\n")
        for i, feat in enumerate(features, 1):
            f.write(f"{i:2d}. {feat}\n")
        f.write("\n")

        f.write("FEATURE IMPORTANCE:\n")
        f.write("-" * 40 + "\n")
        sorted_importance = sorted(zip(features, feature_importance), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_importance:
            f.write(f"{feat:15s}: {imp:6.4f}\n")
        f.write("\n")

        f.write("PREDICTION SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Test samples:    {len(y_test)}\n")
        f.write(f"Mean actual:     {y_test.mean():.2f}\n")
        f.write(f"Mean predicted:  {y_pred.mean():.2f}\n")
        f.write(f"Mean error:      {(y_test - y_pred).mean():.2f}\n\n")

        f.write("TOP 5 PREDICTIONS (Actual vs Predicted):\n")
        f.write("-" * 40 + "\n")
        errors = np.abs(y_test - y_pred)
        top_indices = np.argsort(errors)[-5:][::-1]
        for i in top_indices:
            f.write(
                f"Sample {i + 1:2d}: Actual={y_test.iloc[i]:6.1f} | Pred={y_pred[i]:6.1f} | Error={errors[i]:5.1f}\n")

    print(f"✅ Saved detailed results: {results_file}")


def load_data():
    """Load data using DataIO class."""
    print("Loading data using DataIO...")
    try:
        data_io = DataIO()
        # Load the data from CSV - adjust the file path as needed
        df = data_io.from_csv('cleaned_data.csv').load()
        print(f"Loaded {len(df)} records")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

'''
def prepare_data(data, include_temp=True):
    """Prepare data for modeling.
    
    Args:
        data: DataFrame with the input data
        include_temp: If False, excludes 'mean_temp' from features
    """
    print("\nPreparing data...")
    # Select features and target
    base_features = ['energy_use', 'population', 'renewable_pct', 'gdp']
    if include_temp:
        features = base_features + ['mean_temp']
    else:
        features = base_features
    
    target = 'emissions'

    # Create a copy to avoid SettingWithCopyWarning
    df = data[features + [target]].copy()

    # Print info about missing values
    print("\nMissing values per column before cleaning:")
    print(df.isnull().sum())

    # Drop rows with any missing values
    df_clean = df.dropna()
    
    print(f"\nOriginal dataset size: {len(df)} records")
    print(f"After removing rows with missing values: {len(df_clean)} records")
    print(f"Dropped {len(df) - len(df_clean)} records due to missing values")

    # Separate features and target
    X = df_clean[features]
    y = df_clean[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Print dataset information
    print(f"\nUsing features: {features}")
    print(f"Final dataset size: {len(X)} complete records")

    return X_train, X_test, y_train, y_test, features
'''
def prepare_data(data, include_temp=True, split_year=2010):
    print("\nPreparing data...")

    base_features = ['energy_use', 'population', 'renewable_pct', 'gdp']
    if include_temp:
        features = base_features + ['mean_temp']
    else:
        features = base_features

    target = 'emissions'

    # enforce year sort
    data = data.sort_values(["country", "year"]).reset_index(drop=True)

    # select only columns needed
    df = data[features + [target, "year"]].copy()

    # drop missing
    df = df.dropna()

    print(f"Total df after cleaning: {len(df)} rows")

    # time-based split
    train_df = df[df["year"] <= split_year]
    test_df  = df[df["year"] > split_year]

    print(f"Train: {len(train_df)} rows ({train_df.year.min()}–{train_df.year.max()})")
    print(f"Test:  {len(test_df)} rows ({test_df.year.min()}–{test_df.year.max()})")

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    return X_train, X_test, y_train, y_test, features


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train and evaluate Random Forest model."""
    print("\nTraining Random Forest model...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred)
    }

    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"Mean Squared Error: {metrics['mse']:.2f}")
    print(f"Mean Absolute Error: {metrics['mae']:.2f}")

    return model, y_pred, metrics


def plot_feature_importance(model, feature_names, output_path):
    """Plot feature importance."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(importance)), importance[indices], align='center')
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Saved feature importance plot: {output_path}")


def plot_predictions(y_test, y_pred, output_path):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Emissions')
    plt.ylabel('Predicted Emissions')
    plt.title('Actual vs Predicted Emissions')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Saved predictions plot: {output_path}")


def main():
    """Main function to run the analysis."""
    print(f"=== Emissions Prediction using Random Forest ===")
    
    # Run analysis with mean_temp
    print("\n" + "="*50)
    print("ANALYSIS WITH MEAN_TEMP")
    print("="*50)
    results_dir = get_next_results_dir()
    print(f"Created results directory: {results_dir}")
    
    # Load and prepare data
    data = load_data()
    X_train, X_test, y_train, y_test, features = prepare_data(data, include_temp=True)

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train model and get predictions
    model, y_pred, metrics = train_random_forest(X_train_s, X_test_s, y_train, y_test)

    # Save results
    save_results_txt(results_dir, features, y_test, y_pred, metrics, 
                    model.feature_importances_)
    
    # Plot feature importance
    plot_feature_importance(model, features, results_dir / 'feature_importance.png')
    
    # Plot predictions
    plot_predictions(y_test, y_pred, results_dir / 'predictions_plot.png')
    
    # Run analysis WITHOUT mean_temp
    print("\n" + "="*50)
    print("ANALYSIS WITHOUT MEAN_TEMP")
    print("="*50)
    results_dir_no_temp = get_next_results_dir()
    print(f"Created results directory: {results_dir_no_temp}")
    
    # Prepare data without mean_temp
    X_train_nt, X_test_nt, y_train_nt, y_test_nt, features_nt = prepare_data(data, include_temp=False)
    
    # Scale features
    scaler_nt = StandardScaler()
    X_train_nt_s = scaler_nt.fit_transform(X_train_nt)
    X_test_nt_s = scaler_nt.transform(X_test_nt)

    # Train model and get predictions
    model_nt, y_pred_nt, metrics_nt = train_random_forest(X_train_nt_s, X_test_nt_s, y_train_nt, y_test_nt)
    
    # Save results
    save_results_txt(results_dir_no_temp, features_nt, y_test_nt, y_pred_nt, metrics_nt,
                    model_nt.feature_importances_)
    
    # Plot feature importance
    plot_feature_importance(model_nt, features_nt, results_dir_no_temp / 'feature_importance.png')
    
    # Plot predictions
    plot_predictions(y_test_nt, y_pred_nt, results_dir_no_temp / 'predictions_plot.png')

    print(f"\n✅ Analysis complete! Results saved in:")
    print(f"- With mean_temp: {results_dir}")
    print(f"- Without mean_temp: {results_dir_no_temp}")
    print(f"📄 Main results files:")
    print(f"  - With mean_temp: {os.path.join(results_dir, 'results.countries_of_interest.txt')}")
    print(f"  - Without mean_temp: {os.path.join(results_dir_no_temp, 'results.countries_of_interest.txt')}")


if __name__ == "__main__":
    main()
