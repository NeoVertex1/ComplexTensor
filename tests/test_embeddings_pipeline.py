import numpy as np
import matplotlib.pyplot as plt
import pytest
import torch
from src.complextensor.complex_tensor import ComplexTensor
from src.complextensor.quantum_state_processor import QuantumStateProcessor
from src.complextensor.complex_embeddings import real_to_complex_embeddings, complex_to_real_embeddings, ComplexEmbedding


@pytest.fixture
def load_embeddings():
    """Load embeddings and related data for testing."""
    data = {
        "real_embeddings": np.load("complex_embeddings_outputs/real_embeddings.npy"),
        "reconstructed_real_embeddings": np.load("complex_embeddings_outputs/reconstructed_real_embeddings.npy"),
        "complex_real": np.load("complex_embeddings_outputs/complex_real_part.npy"),
        "complex_imaginary": np.load("complex_embeddings_outputs/complex_imaginary_part.npy"),
        "memory_metrics": np.load("complex_embeddings_outputs/memory_metrics.npy"),
    }
    return data


def test_reconstruction_error(load_embeddings):
    """Validate the reconstruction error between real and reconstructed embeddings."""
    real_embeddings = load_embeddings["real_embeddings"]
    reconstructed_real_embeddings = load_embeddings["reconstructed_real_embeddings"]

    # Normalize embeddings for fair comparison
    real_embeddings = (real_embeddings - real_embeddings.mean(axis=1, keepdims=True)) / real_embeddings.std(axis=1, keepdims=True)
    reconstructed_real_embeddings = (reconstructed_real_embeddings - reconstructed_real_embeddings.mean(axis=1, keepdims=True)) / reconstructed_real_embeddings.std(axis=1, keepdims=True)

    reconstruction_error = np.linalg.norm(real_embeddings - reconstructed_real_embeddings) / np.linalg.norm(real_embeddings)
    print(f"Reconstruction Error: {reconstruction_error:.6f}")

    # Debugging outputs
    np.set_printoptions(precision=4, suppress=True)
    print("Real Embeddings Sample:\n", real_embeddings[0, :10])
    print("Reconstructed Real Embeddings Sample:\n", reconstructed_real_embeddings[0, :10])
    print("Difference:\n", (real_embeddings - reconstructed_real_embeddings)[0, :10])

    assert reconstruction_error < 0.01, "Reconstruction error is too high"


def test_distribution_visualization(load_embeddings):
    """Visualize distributions of real and imaginary components."""
    complex_real = load_embeddings["complex_real"]
    complex_imaginary = load_embeddings["complex_imaginary"]

    plt.figure(figsize=(10, 5))
    plt.hist(complex_real.flatten(), bins=50, alpha=0.5, label="Complex Real Part")
    plt.hist(complex_imaginary.flatten(), bins=50, alpha=0.5, label="Complex Imaginary Part")
    plt.title("Distribution of Complex Embedding Components")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("complex_embedding_distributions.png")
    print("Saved distribution visualization to complex_embedding_distributions.png")


def test_memory_usage_visualization(load_embeddings):
    """Compare memory usage of real and complex embeddings."""
    memory_metrics = load_embeddings["memory_metrics"]

    plt.figure(figsize=(6, 4))
    labels = ["Real Embeddings", "Complex Embeddings"]
    plt.bar(labels, memory_metrics, color=["blue", "green"], alpha=0.7)
    plt.title("Memory Usage Comparison")
    plt.ylabel("Memory (MB)")
    plt.savefig("memory_usage_comparison.png")
    print("Saved memory usage comparison to memory_usage_comparison.png")


def test_real_to_complex_transformations():
    """Test and debug real-to-complex and back-to-real transformations."""
    # Generate random real embeddings
    real_embeddings = torch.rand((16, 1024), dtype=torch.float32)

    # Normalize the embeddings
    real_embeddings = (real_embeddings - real_embeddings.mean(dim=1, keepdim=True)) / real_embeddings.std(dim=1, keepdim=True)

    # Convert to complex embeddings
    complex_embeddings = real_to_complex_embeddings(real_embeddings)

    # Debugging outputs for real-to-complex
    print("Real-to-Complex Transformation:")
    print("Real Part Shape:", complex_embeddings.real.shape)
    print("Imaginary Part Shape:", complex_embeddings.imag.shape)
    print("Real Part Sample:", complex_embeddings.real[0, :10].cpu().numpy())
    print("Imaginary Part Sample:", complex_embeddings.imag[0, :10].cpu().numpy())

    # Convert back to real embeddings
    reconstructed_real_embeddings = complex_to_real_embeddings(complex_embeddings)

    # Debugging outputs for complex-to-real
    print("Complex-to-Real Transformation:")
    print("Reconstructed Real Embeddings Shape:", reconstructed_real_embeddings.shape)
    print("Reconstructed Real Embeddings Sample:", reconstructed_real_embeddings[0, :10].cpu().numpy())

    # Validate reconstruction
    reconstruction_error = torch.norm(real_embeddings - reconstructed_real_embeddings) / torch.norm(real_embeddings)
    print(f"Reconstruction Error: {reconstruction_error:.6f}")

    assert reconstruction_error < 0.01, "Reconstruction error is too high in transformation pipeline"


def test_quantum_features():
    """Test quantum operations using QuantumStateProcessor."""
    qsp = QuantumStateProcessor(n_qubits=3)

    # Create a superposition state
    alpha, beta = 0.6, 0.8
    superposition_state = qsp.create_superposition(alpha, beta)

    # Apply Hadamard transformation
    hadamard_state = qsp.apply_hadamard(superposition_state)

    # Debugging outputs
    print("Superposition State (real):", superposition_state.real.cpu().numpy())
    print("Hadamard State (real):", hadamard_state.real.cpu().numpy())
    print("Hadamard State (imag):", hadamard_state.imag.cpu().numpy())

    # Compute entanglement entropy
    entropy = qsp.get_entanglement_entropy(hadamard_state, partition=1)
    print(f"Superposition State Norm: {torch.norm(superposition_state.real):.6f}")
    print(f"Entanglement Entropy after Hadamard: {entropy:.6f}")

    assert np.isclose(entropy, 0.0, atol=0.01), "Entanglement entropy is incorrect"


def test_memory_scalability():
    """Test memory usage for larger embeddings."""
    # Generate larger embeddings
    large_real_embeddings = torch.rand((64, 2048))  # 64 samples, 2048 dimensions
    large_complex_embeddings = real_to_complex_embeddings(large_real_embeddings)

    # Compute memory usage
    real_mem = large_real_embeddings.element_size() * large_real_embeddings.nelement() / 1e6  # Convert to MB
    complex_mem = (
        large_complex_embeddings.real.element_size() * large_complex_embeddings.real.nelement() +
        large_complex_embeddings.imag.element_size() * large_complex_embeddings.imag.nelement()
    ) / 1e6  # Convert to MB

    print(f"Memory Usage for Larger Real Embeddings: {real_mem:.2f} MB")
    print(f"Memory Usage for Larger Complex Embeddings: {complex_mem:.2f} MB")
    assert complex_mem < real_mem * 1.5, "Complex embeddings use significantly more memory than expected"
