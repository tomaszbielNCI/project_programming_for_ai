import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load the history file
with open('C:/python/project_programming_for_ai/src/models/bnn/history.pkl', 'rb') as f:
    history = pickle.load(f)

# Check if history is a Keras History object or a dictionary
if hasattr(history, 'history'):
    history_dict = history.history
else:
    history_dict = history

# Print available keys
print("Available keys in history:", list(history_dict.keys()))

# Convert to DataFrame for better visualization
history_df = pd.DataFrame(history_dict)
print("\nTraining History:")
print(history_df)

# Plot training history
plt.figure(figsize=(12, 4))

# Plot loss if available
if 'loss' in history_dict:
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['loss'], label='Training Loss')
    if 'val_loss' in history_dict:
        plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

# Plot accuracy if available
if 'accuracy' in history_dict:
    plt.subplot(1, 2, 2)
    plt.plot(history_dict['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history_dict:
        plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

plt.tight_layout()
plt.show()