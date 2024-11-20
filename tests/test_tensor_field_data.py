import pytest
import torch
from src.complextensor.tensor_field import TensorField
import os
import numpy as np

OUTPUT_DIR = "test_outputs"

@pytest.fixture(scope="module")
def tensor_field():
    """Fixture to initialize the TensorField once for all tests."""
    return TensorField()

@pytest.fixture(scope="module", autouse=True)
def setup_output_dir():
    """Ensure the output directory exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_tensor_initialization(tensor_field):
    """Test initialization of the TensorField."""
    assert tensor_field.T.real.shape == (4, 4), "Base tensor T has incorrect shape."
    assert torch.allclose(tensor_field.T.imag, torch.zeros_like(tensor_field.T.real)), "Imaginary part is not zero."

def test_tensor_optimization(tensor_field):
    """Test tensor optimization and compute metrics."""
    T_evolved, T_optimized = tensor_field.optimize_tensor()

    # Assert the normalization step worked
    assert T_optimized.abs().norm(p=2) < T_evolved.abs().norm(p=2), "Normalization failed."
    entropy, norm = tensor_field.compute_metrics(T_optimized)

    # Ensure entropy and norm are reasonable
    assert entropy > 0, "Entropy calculation failed."
    assert norm > 0, "Norm calculation failed."

    # Save evolved and optimized tensors for debugging/analysis
    np.save(os.path.join(OUTPUT_DIR, "T_evolved_real.npy"), T_evolved.real.cpu().numpy())
    np.save(os.path.join(OUTPUT_DIR, "T_evolved_imag.npy"), T_evolved.imag.cpu().numpy())
    np.save(os.path.join(OUTPUT_DIR, "T_optimized_real.npy"), T_optimized.real.cpu().numpy())
    np.save(os.path.join(OUTPUT_DIR, "T_optimized_imag.npy"), T_optimized.imag.cpu().numpy())

    print("\n=== Metrics ===")
    print(f"Entropy: {entropy:.6f}")
    print(f"Norm: {norm:.6f}")

def test_tensor_output_data(tensor_field):
    """Generate and output tensor data for further analysis."""
    T_evolved, T_optimized = tensor_field.optimize_tensor()

    # Compute additional metrics
    evolved_entropy, evolved_norm = tensor_field.compute_metrics(T_evolved)
    optimized_entropy, optimized_norm = tensor_field.compute_metrics(T_optimized)

    # Prepare data to save
    output_data = {
        "evolved_entropy": evolved_entropy,
        "evolved_norm": evolved_norm,
        "optimized_entropy": optimized_entropy,
        "optimized_norm": optimized_norm,
    }

    # Save data to a text file
    output_file = os.path.join(OUTPUT_DIR, "tensor_metrics.txt")
    with open(output_file, "w") as f:
        f.write("=== Tensor Metrics ===\n")
        f.write(f"Evolved Entropy: {evolved_entropy:.6f}\n")
        f.write(f"Evolved Norm: {evolved_norm:.6f}\n")
        f.write(f"Optimized Entropy: {optimized_entropy:.6f}\n")
        f.write(f"Optimized Norm: {optimized_norm:.6f}\n")

    # Print confirmation
    print(f"Metrics saved to {output_file}")

    # Assertions
    assert evolved_entropy > 0, "Evolved entropy should be positive."
    assert optimized_entropy > 0, "Optimized entropy should be positive."
    assert evolved_norm > optimized_norm, "Optimization should reduce the norm."
