
from pathlib import Path
import numpy as np
from .model_v1 import build_bnn_showcase
from src.loaders.bnn_basic_loader import load_parquet_for_bnn
import joblib

def train_and_evaluate(
    data_path: str,
    window: int = 20,
    target_horizon: int = 1,
    batch_size: int = 128,
    epochs: int = 5,
    validation_split: float = 0.1
):
    """
    Train and evaluate the Bayesian Neural Network model.
    
    Args:
        data_path: Path to the parquet file with financial data
        window: Lookback window size for time series
        target_horizon: Number of steps ahead to predict
        batch_size: Training batch size
        epochs: Number of training epochs
        validation_split: Fraction of data to use for validation
    """
    print("\n" + "="*50)
    print("STARTING BAYESIAN NEURAL NETWORK TRAINING")
    print("="*50)
    print(f"\nConfiguration:")
    print(f"- Data file: {data_path}")
    print(f"- Window size: {window}")
    print(f"- Target horizon: {target_horizon}")
    print(f"- Batch size: {batch_size}")
    print(f"- Epochs: {epochs}")
    print(f"- Validation split: {validation_split}")
    
    # Load and prepare data
    print("\nLoading and preparing data...")
    X, y = load_parquet_for_bnn(
        path=data_path,
        window=window,
        target_horizon=target_horizon
    )
    
    print(f"\nData loaded successfully!")
    print(f"- Input shape: {X.shape}")
    print(f"- Target shape: {y.shape}")
    
    # Build and train the model
    print("\nBuilding the model...")
    model = build_bayesian_model_laplace(
        window=window,
        feature_count=X.shape[2],
        train_size=int(len(X) * (1 - validation_split))
    )
    
    print("\nStarting training...")
    history = model.fit(
        X, y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        verbose=1
    )
    
    # Save training history
    history_file = Path(data_path).parent / 'training_history.pkl'
    joblib.dump(history.history, history_file)
    print(f"\nTraining completed! History saved to: {history_file}")
    
    # Print final metrics
    print("\nFinal Metrics:")
    for key, value in history.history.items():
        print(f"- {key}: {value[-1]:.4f}")
    
    return model, history

def main():
    """Main function to run the BNN training pipeline."""
    # Default configuration - can be modified as needed
    config = {
        'data_path': r"C:\python\project_programming_for_ai\data\parsed\US.100+.parquet",
        'window': 20,
        'target_horizon': 1,
        'batch_size': 128,
        'epochs': 5,
        'validation_split': 0.1
    }
    
    try:
        model, history = train_and_evaluate(**config)
        print("\nTraining completed successfully!")
        return model, history
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()

__all__ = ['build_bnn_showcase', 'train_and_evaluate', 'main']
