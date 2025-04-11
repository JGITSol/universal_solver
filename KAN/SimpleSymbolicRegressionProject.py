# Core implementation structure
from turtle import width
from kan import KAN
import torch
import numpy as np
import matplotlib.pyplot as plt

# Generate Kepler's Third Law dataset
def generate_kepler_data(n_samples=1000):
    a = np.random.uniform(0.3, 30, n_samples)  # Semi-major axis (AU)
    T = np.sqrt(a**3) + np.random.normal(0, 0.1, n_samples)  # Period (years) with noise
    return torch.tensor(a, dtype=torch.float32).reshape(-1, 1), torch.tensor(T, dtype=torch.float32)

# Training and symbolic extraction pipeline
def symbolic_discovery_pipeline():
    # Generate data
    x_train, y_train = generate_kepler_data()
    
    # Initialize and train KAN
    kan = KAN(width=[1, 3, 1], grid=5, k=3)
    kan.train(x_train, y_train, steps=500)
    
    # Visualize and prune
    kan.visualize()
    kan.sparsify(regularization=1e-3)
    kan.prune(threshold=0.01)
    
    # Extract symbolic formula
    symbolic_formula = kan.to_symbolic()
    print(f"Discovered formula: {symbolic_formula}")
    
    # Evaluate against ground truth
    x_test = torch.linspace(0.3, 30, 100).reshape(-1, 1)
    y_test = torch.sqrt(x_test**3)
    y_pred = kan(x_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, alpha=0.3, label='Training data')
    plt.plot(x_test, y_test, 'r-', label='Ground truth')
    plt.plot(x_test, y_pred.detach(), 'g--', label='KAN prediction')
    plt.legend()
    plt.xlabel('Semi-major axis (AU)')
    plt.ylabel('Orbital period (years)')
    plt.title(f'Discovered formula: {symbolic_formula}')
    plt.savefig('kepler_law_discovery.png')

# Add this line to actually run the function
if __name__ == "__main__":
    symbolic_discovery_pipeline()
