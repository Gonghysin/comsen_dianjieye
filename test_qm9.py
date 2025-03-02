import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_model import predict, smiles_to_graph

# Test prediction function
def test_prediction():
    # Test some SMILES strings
    test_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "C1=CC=C(C=C1)C(=O)O"  # Benzoic acid
    ]
    
    # Use absolute path to locate model file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models', 'best_model.pt')
    
    print(f"Using model path: {model_path}")
    print("Test molecule prediction results:")
    for smiles in test_smiles:
        result = predict(smiles, model_path=model_path)
        if result is not None:
            print(f"SMILES: {smiles}")
            print(f"Predicted value: {result:.4f}")
        else:
            print(f"SMILES: {smiles} - Cannot parse")
    
    return True

def evaluate_qm9_predictions():
    """
    Read SMILES and LUMO values from QM9 dataset, make predictions and evaluate model performance
    """
    # Use absolute path to locate model file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models', 'best_model.pt')
    
    print(f"Using model path: {model_path}")
    print("Evaluating model performance on QM9 dataset...")
    
    # Read QM9 dataset - also using absolute path
    qm9_path = os.path.join(current_dir, 'qm9.csv')
    try:
        df = pd.read_csv(qm9_path)
        # Only use the first 10,000 data points
        df = df.head(10000)
        smiles_list = df['smiles'].tolist()
        lumo_values = -df['lumo'].values * 27.2114  # Convert to eV
    except Exception as e:
        print(f"Failed to read dataset: {e}")
        return False
    
    # Make predictions
    predictions = []
    valid_indices = []
    
    print(f"Starting prediction for {len(smiles_list)} molecules...")
    for i, smiles in enumerate(smiles_list):
        if i % 100 == 0:
            print(f"Processed {i}/{len(smiles_list)} molecules")
        
        try:
            pred = predict(smiles, model_path=model_path)
            if pred is not None:
                predictions.append(pred)
                valid_indices.append(i)
        except Exception as e:
            print(f"Error predicting molecule {smiles}: {e}")
    
    # Get corresponding true values
    true_values = lumo_values[valid_indices]
    
    # Calculate evaluation metrics
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    rse = np.sqrt(np.sum((np.array(predictions) - true_values)**2) / np.sum(true_values**2))
    
    print(f"Evaluation results:")
    print(f"Sample count: {len(predictions)}")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"RSE: {rse:.6f}")
    
    # Plot scatter plot
    plt.figure(figsize=(10, 8))
    
    # Predicted vs true values scatter plot
    plt.subplot(2, 1, 1)
    plt.scatter(true_values, predictions, alpha=0.5)
    
    # Add y=x line
    min_val = min(min(true_values), min(predictions))
    max_val = max(max(true_values), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('True Values (-LUMO)')
    plt.ylabel('Predicted Values')
    plt.title('Prediction Results on QM9 Dataset')
    
    # Plot residuals
    plt.subplot(2, 1, 2)
    residuals = np.array(predictions) - true_values
    plt.scatter(true_values, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True Values (-LUMO)')
    plt.ylabel('Residuals (Predicted - True)')
    plt.title('Residual Distribution')
    
    plt.tight_layout()
    plt.savefig('qm9_prediction_results.png')
    plt.show()
    
    return True

if __name__ == "__main__":
    test_prediction()
    evaluate_qm9_predictions()
