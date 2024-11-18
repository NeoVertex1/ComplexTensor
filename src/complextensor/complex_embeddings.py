import torch
import torch.nn as nn
from src.complextensor.complex_tensor import ComplexTensor
from src.complextensor.quantum_state_processor import QuantumStateProcessor


def real_to_complex_embeddings(embeddings: torch.Tensor, device: torch.device = torch.device("cpu")) -> ComplexTensor:
    """
    Converts real embeddings to complex embeddings by pairing adjacent dimensions.

    Args:
        embeddings (torch.Tensor): Real-valued embeddings of shape (batch_size, embedding_dim).
        device (torch.device): Device for the resulting tensors.

    Returns:
        ComplexTensor: Complex embeddings of shape (batch_size, embedding_dim / 2).
    """
    if embeddings.size(1) % 2 != 0:
        raise ValueError("Embedding dimension must be even to convert to complex embeddings.")

    # Pair adjacent dimensions to form the real and imaginary parts
    real = embeddings[:, ::2].to(device)
    imag = embeddings[:, 1::2].to(device)

    return ComplexTensor(real, imag)


def complex_to_real_embeddings(complex_embeddings: ComplexTensor) -> torch.Tensor:
    """
    Converts complex embeddings back to real embeddings for compatibility.

    Args:
        complex_embeddings (ComplexTensor): Complex embeddings.

    Returns:
            torch.Tensor: Real embeddings of shape (batch_size, embedding_dim).
    """
    real = complex_embeddings.real  # Shape: (batch_size, embedding_dim / 2)
    imag = complex_embeddings.imag  # Shape: (batch_size, embedding_dim / 2)

    # Interleave real and imaginary parts to reconstruct original embeddings
    return torch.stack([real, imag], dim=-1).reshape(real.size(0), -1)


class ComplexEmbedding(nn.Module):
    """
    Complex-valued embedding layer with quantum-enhanced operations.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, n_qubits: int):
        """
        Initializes the embedding layer with quantum state processing capabilities.

        Args:
            num_embeddings (int): Vocabulary size.
            embedding_dim (int): Embedding dimension (must be even).
            n_qubits (int): Number of qubits for quantum operations.
        """
        super(ComplexEmbedding, self).__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("Embedding dimension must be even for complex embeddings.")
        self.embedding_dim = embedding_dim
        self.complex_dim = embedding_dim // 2
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)

        # Initialize Quantum State Processor
        self.qsp = QuantumStateProcessor(n_qubits=n_qubits)

    def forward(self, input_ids: torch.Tensor) -> ComplexTensor:
        """
        Forward pass to retrieve embeddings, convert to complex space, and apply quantum operations.

        Args:
            input_ids (torch.Tensor): Input token IDs.

        Returns:
            ComplexTensor: Quantum-enhanced complex embeddings.
        """
        # Retrieve embeddings
        embeddings = self.embeddings(input_ids)  # Shape: (batch_size, embedding_dim)

        # Convert to complex embeddings
        real = embeddings[:, ::2]  # Pair even-indexed dimensions
        imag = embeddings[:, 1::2]  # Pair odd-indexed dimensions
        complex_embeddings = ComplexTensor(real, imag)

        # Apply quantum superposition
        flattened_real = complex_embeddings.real.flatten().tolist()
        superposition = self.qsp.create_superposition(flattened_real)
        complex_embeddings.real = superposition.real.view_as(real)
        complex_embeddings.imag = superposition.imag.view_as(imag)

        return complex_embeddings

    def entangle_embeddings(self, complex_embeddings: ComplexTensor) -> ComplexTensor:
        """
        Applies quantum entanglement to complex embeddings.

        Args:
            complex_embeddings (ComplexTensor): Input complex embeddings.

        Returns:
            ComplexTensor: Entangled complex embeddings.
        """
        # Use the QuantumStateProcessor to create entanglement
        bell_state = self.qsp.create_bell_state(bell_type=0)

        # Project embeddings onto the entangled Bell state
        entangled_real = complex_embeddings.real + bell_state.real[:complex_embeddings.real.shape[1]]
        entangled_imag = complex_embeddings.imag + bell_state.imag[:complex_embeddings.imag.shape[1]]

        return ComplexTensor(entangled_real, entangled_imag)
