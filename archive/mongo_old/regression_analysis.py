import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA_DIR = Path(r"/mongo/data/development_analysis")
OUT_DIR = Path(__file__).parent / "ml_output"

def list_csvs(p: Path) -> list:
    return [f for f in p.glob("*.csv") if f.is_file()]

def load_csv(fp: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(fp)
        return df
    except Exception:
        try:
            df = pd.read_csv(fp, sep=';')
            return df
        except Exception as e:
            raise e

def basic_info(df: pd.DataFrame) -> dict:
    info = {}
    info["shape"] = df.shape
    info["columns"] = list(df.columns)
    info["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    info["missing_total"] = int(df.isna().sum().sum())
    info["missing_by_col"] = df.isna().sum().to_dict()
    num_df = df.select_dtypes(include=[np.number])
    info["numeric_cols"] = list(num_df.columns)
    info["numeric_count"] = len(num_df.columns)
    if len(num_df.columns) > 0:
        desc = num_df.describe().T
        info["describe"] = desc
    else:
        info["describe"] = pd.DataFrame()
    return info

def select_features_target(df: pd.DataFrame) -> tuple:
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.shape[1] < 2:
        return None, None
    target = num_df.columns[-1]
    features = [c for c in num_df.columns if c != target]
    X = num_df[features]
    y = num_df[target]
    return X, y

def metrics(y_true, y_pred) -> dict:
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
    }

def train_lr(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    return pipe

def train_rf(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

def plot_predictions(y_test, pred_lr, pred_rf, dataset_name: str, out_folder: Path):
    out_folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_test, pred_lr, alpha=0.6)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0].set_title("Linear Regression: Actual vs Predicted")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")

    axes[1].scatter(y_test, pred_rf, alpha=0.6, color='orange')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[1].set_title("Random Forest: Actual vs Predicted")
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")

    plt.tight_layout()
    fp = out_folder / f"{dataset_name}_predictions.png"
    plt.savefig(fp, dpi=200, bbox_inches='tight')
    plt.close()
    return fp

def plot_residuals(y_test, pred_lr, pred_rf, dataset_name: str, out_folder: Path):
    out_folder.mkdir(parents=True, exist_ok=True)
    res_lr = y_test - pred_lr
    res_rf = y_test - pred_rf
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(res_lr, bins=40, alpha=0.8)
    axes[0].set_title("Linear Regression Residuals")
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Count")

    axes[1].hist(res_rf, bins=40, alpha=0.8, color='orange')
    axes[1].set_title("Random Forest Residuals")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    fp = out_folder / f"{dataset_name}_residuals.png"
    plt.savefig(fp, dpi=200, bbox_inches='tight')
    plt.close()
    return fp

def save_feature_importance(model, feature_names: list, dataset_name: str, out_folder: Path):
    if not hasattr(model, "feature_importances_"):
        return None
    out_folder.mkdir(parents=True, exist_ok=True)
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(3, int(len(imp) * 0.3))))
    imp.plot(kind='bar', ax=ax)
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    fp = out_folder / f"{dataset_name}_rf_feature_importance.png"
    plt.savefig(fp, dpi=200, bbox_inches='tight')
    plt.close()
    return fp

def process_dataset(csv_path: Path, report_lines: list):
    name = csv_path.stem
    out_folder = OUT_DIR / name
    df = load_csv(csv_path)
    info = basic_info(df)
    report_lines.append(f"=== {name} ===")
    report_lines.append(f"Path: {csv_path}")
    report_lines.append(f"Shape: {info['shape']}")
    report_lines.append(f"Numeric columns ({info['numeric_count']}): {info['numeric_cols']}")
    report_lines.append(f"Total missing values: {info['missing_total']}")
    if info['numeric_count'] < 2:
        report_lines.append("Not enough numeric columns for regression (need at least 2). Skipping.")
        report_lines.append("")
        return

    X, y = select_features_target(df)
    if X is None:
        report_lines.append("Could not select features/target. Skipping.")
        report_lines.append("")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = train_lr(X_train, y_train)
    rf = train_rf(X_train, y_train)

    pred_lr = lr.predict(X_test)
    pred_rf = rf.predict(X_test)

    m_lr = metrics(y_test, pred_lr)
    m_rf = metrics(y_test, pred_rf)

    report_lines.append("Linear Regression (with StandardScaler)")
    report_lines.append(f"R2: {m_lr['R2']:.4f} | RMSE: {m_lr['RMSE']:.4f} | MAE: {m_lr['MAE']:.4f}")
    report_lines.append("Random Forest (100 trees, max_depth=10)")
    report_lines.append(f"R2: {m_rf['R2']:.4f} | RMSE: {m_rf['RMSE']:.4f} | MAE: {m_rf['MAE']:.4f}")

    p1 = plot_predictions(y_test.values if hasattr(y_test, 'values') else y_test, pred_lr, pred_rf, name, out_folder)
    p2 = plot_residuals(y_test.values if hasattr(y_test, 'values') else y_test, pred_lr, pred_rf, name, out_folder)
    p3 = save_feature_importance(rf, list(X.columns), name, out_folder)

    report_lines.append(f"Saved plots: {p1.name}, {p2.name}{', ' + p3.name if p3 else ''}")
    report_lines.append("")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        sys.exit(1)

    csvs = list_csvs(DATA_DIR)
    if not csvs:
        print(f"No CSV files found in {DATA_DIR}")
        sys.exit(0)

    report_lines = []
    report_lines.append("MACHINE LEARNING REGRESSION SUMMARY")
    report_lines.append("Data folder: " + str(DATA_DIR))
    report_lines.append("")

    for csv in csvs:
        try:
            process_dataset(csv, report_lines)
        except Exception as e:
            report_lines.append(f"=== {csv.stem} ===")
            report_lines.append(f"Error: {e}")
            report_lines.append("")

    report_path = OUT_DIR / "summary.countries_of_interest 1.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("Done.")
    print("Report:", report_path)
    print("Outputs per dataset in:", OUT_DIR)

if __name__ == "__main__":
    main()
