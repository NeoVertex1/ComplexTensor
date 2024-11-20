import pytest
import torch
from src.complextensor.tensor_field import TensorField

def test_tensor_initialization():
    tensor_field = TensorField()
    assert tensor_field.T.real.shape == (4, 4), "Base tensor T has incorrect shape."
    assert torch.allclose(tensor_field.T.imag, torch.zeros_like(tensor_field.T.real)), "Imaginary part is not zero."

def test_tensor_optimization():
    tensor_field = TensorField()
    T_evolved, T_optimized = tensor_field.optimize_tensor()

    print("=== Debug: T_evolved ===")
    print("Real Part:\n", T_evolved.real.cpu().numpy())
    print("Imaginary Part:\n", T_evolved.imag.cpu().numpy())
    print("Magnitude:\n", T_evolved.abs().cpu().numpy())

    print("=== Debug: T_optimized ===")
    print("Real Part:\n", T_optimized.real.cpu().numpy())
    print("Imaginary Part:\n", T_optimized.imag.cpu().numpy())
    print("Magnitude:\n", T_optimized.abs().cpu().numpy())

    assert T_optimized.abs().norm(p=2) < T_evolved.abs().norm(p=2), "Normalization failed."
    entropy, norm = tensor_field.compute_metrics(T_optimized)
    assert entropy > 0, "Entropy calculation failed."

