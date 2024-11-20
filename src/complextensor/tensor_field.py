import torch
from complextensor import ComplexTensor
import logging

class TensorField:
    def __init__(self, α=0.5, β=0.3, ψ=44.8, ξ=3721.8, ε=0.28082, τ=64713.97, π=3.14159, η=0.05):
        """
        Initialize the tensor field and related parameters.
        """
        self.α = α
        self.β = β
        self.constants = {"ψ": ψ, "ξ": ξ, "ε": ε, "τ": τ, "π": π, "η": η}
        self.logger = logging.getLogger("TensorField")
        self.logger.setLevel(logging.DEBUG)

        self.logger.debug("Initializing TensorField.")
        self.T = self.initialize_tensor()
        self.R = self.compute_resonance_matrix()
        self.S = self.compute_stability_matrix()

    def initialize_tensor(self):
        """
        Initialize the base tensor T.
        """
        constants = self.constants
        real = torch.tensor([
            [constants["ψ"], constants["ε"], 0, constants["π"]],
            [constants["ε"], constants["ξ"], constants["τ"], 0],
            [0, constants["τ"], constants["π"], constants["ε"]],
            [constants["π"], 0, constants["ε"], constants["ψ"]]
        ], dtype=torch.float32)

        imag = torch.zeros_like(real)
        self.logger.debug(f"Base tensor T initialized with shape {real.shape}.")
        return ComplexTensor(real, imag)

    def compute_resonance_matrix(self, δ=0.1, ρ=0.2):
        """
        Compute the resonance matrix R.
        """
        real = torch.tensor([
            [0, δ, ρ, 0],
            [δ, 0, 0, ρ],
            [ρ, 0, 0, δ],
            [0, ρ, δ, 0]
        ], dtype=torch.float32)

        imag = torch.zeros_like(real)
        self.logger.debug(f"Resonance matrix R initialized with shape {real.shape}.")
        return ComplexTensor(real, imag)

    def compute_stability_matrix(self):
        """
        Compute the stability matrix S.
        """
        η = self.constants["η"]
        real = torch.eye(4, dtype=torch.float32) * η
        imag = torch.zeros_like(real)
        self.logger.debug(f"Stability matrix S initialized with shape {real.shape}.")
        return ComplexTensor(real, imag)

    def scalar_multiply(self, scalar, tensor):
        """
        Perform scalar multiplication for ComplexTensor.
        """
        self.logger.debug(f"Performing scalar multiplication: scalar={scalar}, tensor shape={tensor.real.shape}.")
        return ComplexTensor(tensor.real * scalar, tensor.imag * scalar)

    def normalize_tensor(self, T):
        """
        Normalize a tensor to stabilize its norm.
        """
        self.logger.debug("Normalizing tensor.")
        norm = T.abs().norm(p=2) + 1e-8
        normalized_tensor = ComplexTensor(T.real / norm, T.imag / norm)
        self.logger.debug("Tensor normalization complete.")
        return normalized_tensor

    def compute_metrics(self, T):
        """
        Compute entropy and norm for the tensor.
        """
        self.logger.debug("Computing metrics for tensor.")
        
        # Use log-sum-exp trick for stable entropy calculation
        entropy = torch.logsumexp(torch.log(T.abs() + 1e-8).flatten(), dim=0).item()
        norm = T.abs().norm(p=2).item()
        
        self.logger.debug(f"Computed metrics - Entropy: {entropy}, Norm: {norm}.")
        return entropy, norm


    def optimize_tensor(self):
        """
        Apply the optimization process to the tensor field.
        """
        self.logger.info("Optimizing tensor.")
        T_evolved = self.T + self.scalar_multiply(self.α, self.R) + self.scalar_multiply(self.β, self.S)
        T_optimized = self.normalize_tensor(T_evolved)
        return T_evolved, T_optimized
