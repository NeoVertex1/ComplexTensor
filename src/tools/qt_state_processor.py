# should be able to handle superposition and entanglement, more testing is needed

import torch
import numpy as np
from complextensor import ComplexTensor
from typing import List

class QuantumStateProcessor:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.state_size = 2 ** n_qubits
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.zero_state = self._create_basis_state(0)
        self.one_state = self._create_basis_state(1)

    def _create_basis_state(self, state: int) -> ComplexTensor:
        real = torch.zeros(self.state_size, device=self.device)
        real[state] = 1.0
        return ComplexTensor(real)

    def create_superposition(self, alpha: float, beta: float) -> ComplexTensor:
        norm_factor = (alpha**2 + beta**2) ** 0.5
        alpha, beta = alpha / norm_factor, beta / norm_factor
        real = torch.tensor([alpha, beta] + [0.0] * (self.state_size - 2), device=self.device)
        return ComplexTensor(real)

    def create_bell_state(self, bell_type: int = 0) -> ComplexTensor:
        real = torch.zeros(self.state_size, device=self.device)
        if bell_type == 0:
            real[0] = 1 / np.sqrt(2)
            real[3] = 1 / np.sqrt(2)
        elif bell_type == 1:
            real[0] = 1 / np.sqrt(2)
            real[3] = -1 / np.sqrt(2)
        elif bell_type == 2:
            real[1] = 1 / np.sqrt(2)
            real[2] = 1 / np.sqrt(2)
        elif bell_type == 3:
            real[1] = 1 / np.sqrt(2)
            real[2] = -1 / np.sqrt(2)
        else:
            raise ValueError("Bell state type must be between 0 and 3")
        return ComplexTensor(real)

    def measure_state(self, state: ComplexTensor, n_samples: int = 100000) -> torch.Tensor:
        probabilities = state.abs().to(self.device)**2
        measurements = torch.multinomial(probabilities, n_samples, replacement=True).to("cpu")
        return measurements

    def get_entanglement_entropy(self, state: ComplexTensor, partition: int) -> float:
        shape = [2] * self.n_qubits
        state_reshaped = state.forward().view(shape).to(self.device)
        rho_A = self._partial_trace(state_reshaped, partition)

        eigenvalues = torch.linalg.eigvalsh(rho_A)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        entropy = -torch.sum(eigenvalues * torch.log2(eigenvalues)).item()
        return entropy

    def _partial_trace(self, state: torch.Tensor, partition: int) -> torch.Tensor:
        n_traced = self.n_qubits - partition
        dims_A = [2] * partition
        dims_B = [2] * n_traced
        state = state.reshape(self._prod(dims_A), self._prod(dims_B))
        rho = torch.mm(state, state.t().conj()).to(self.device)
        return rho

    def _prod(self, iterable):
        result = 1
        for x in iterable:
            result *= x
        return result

    def apply_hadamard(self, state: ComplexTensor) -> ComplexTensor:
        h_matrix = torch.tensor([[1, 1], [1, -1]], dtype=torch.float32, device=self.device) / torch.sqrt(torch.tensor(2.0, dtype=torch.float32))
        full_h_matrix = h_matrix

        for _ in range(self.n_qubits - 1):
            full_h_matrix = torch.kron(full_h_matrix, h_matrix)

        transformed_real = full_h_matrix.to(dtype=torch.float32) @ state.real.to(dtype=torch.float32)
        transformed_imag = full_h_matrix.to(dtype=torch.float32) @ state.imag.to(dtype=torch.float32)

        return ComplexTensor(transformed_real, transformed_imag)
