import torch
import numpy as np
import string
import logging
from complextensor import ComplexTensor

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# MISTransform: Morphing Infinity Spiral Transform implementation
class MISTransform:
    """Morphing Infinity Spiral Transform implementation."""
    def __init__(self, alpha: complex = 1.0, beta: complex = 2.0):
        self.alpha = alpha
        self.beta = beta
        logger.debug(f"Initialized MISTransform with alpha={alpha}, beta={beta}")
        
    def complex_power(self, z: ComplexTensor, exponent: float) -> ComplexTensor:
        """Compute z^exponent using exp(log) method."""
        if not isinstance(z, ComplexTensor):
            raise TypeError("Input must be a ComplexTensor")
            
        log_z = ComplexTensor(
            torch.log(z.abs() + 1e-10),
            z.angle()
        )
        exponent_tensor = ComplexTensor(
            torch.full_like(z.real, exponent),
            torch.zeros_like(z.imag)
        )
        return (log_z * exponent_tensor).exp()
    
    def __call__(self, z: ComplexTensor, t: float) -> ComplexTensor:
        """Apply MIS transformation."""
        try:
            power_term = self.complex_power(z, float(self.alpha.real))
            
            log_z = ComplexTensor(
                torch.log(z.abs() + 1e-10),
                z.angle()
            )
            log_z_beta = self.complex_power(log_z, float(self.beta.real))
            
            phase_tensor = ComplexTensor(
                torch.full_like(z.real, t),
                torch.zeros_like(z.imag)
            )
            rotation = (log_z_beta * phase_tensor).imag
            
            rotation_term = ComplexTensor(
                torch.cos(rotation),
                torch.sin(rotation)
            )
            
            return power_term * rotation_term
            
        except Exception as e:
            logger.error(f"Error in MIS transformation: {str(e)}")
            raise

# Step 1: Define critical values for tensor stability
ψ, ξ, τ, ε, π = 44.8, 3721.8, 64713.97, 0.28082, torch.tensor(np.pi, dtype=torch.float32)

# Initialize a 4x4 ComplexTensor T with the given critical values, adding random imaginary part
real_part = torch.tensor([
    [ψ, ε, 0, π],
    [ε, ξ, τ, 0],
    [0, τ, π, ε],
    [π, 0, ε, ψ]
], dtype=torch.float32)

imag_part = torch.randn_like(real_part) * 0.01  # Small random imaginary component for complexity
T = ComplexTensor(real_part, imag_part)

# Step 2: Encoding function for quantum state
def encode_state(T: ComplexTensor, state_vector: torch.Tensor) -> tuple[ComplexTensor, torch.Tensor]:
    """
    Encodes a quantum state vector into the eigenbasis of T.
    """
    eigvals, eigvecs = torch.linalg.eigh(T.real)
    eigvecs_tensor = ComplexTensor(
        torch.tensor(eigvecs, dtype=torch.float32), 
        torch.zeros_like(torch.tensor(eigvecs, dtype=torch.float32))
    )
    
    # Convert state_vector to ComplexTensor for compatibility
    state_vector_complex = ComplexTensor(state_vector, torch.zeros_like(state_vector))
    
    # Encode state by projecting onto the eigenvectors of T
    encoded_state = eigvecs_tensor  # Encoding state as eigenbasis
    
    return encoded_state, eigvals

# Step 3: Function to generate random characters based on the encoded quantum state with MIS transformation
def generate_random_characters(encoded_state: ComplexTensor, transform: MISTransform, length: int = 1024) -> str:
    """
    Generates a sequence of random characters based on the encoded quantum state.
    """
    character_set = string.printable  # All printable ASCII characters
    num_characters = len(character_set)
    char_sequence = []

    for i in range(length):
        # Apply MIS transformation to the encoded state
        transformed_state = transform(encoded_state, t=float(i) / length)

        # Get magnitude and phase for randomness
        magnitudes = transformed_state.abs().cpu().numpy()
        phases = transformed_state.angle().cpu().numpy()

        # Map the combined magnitude and phase values to a character index
        random_value = int(np.sum(magnitudes * np.abs(np.sin(phases))) * 1e5) % num_characters
        char_sequence.append(character_set[random_value])

    return ''.join(char_sequence)

# Example usage
if __name__ == "__main__":
    # Initialize MISTransform with specific parameters for added complexity
    transform = MISTransform(alpha=1.3 + 0.5j, beta=1.7 + 0.8j)
    
    # Generate a random state vector with some variation for testing
    state_vector = torch.randn(4) * 10  # Introduce more variation in the state vector
    encoded_state, _ = encode_state(T, state_vector)
    
    # Generate a sequence of random characters based on the transformed encoded state
    random_characters = generate_random_characters(encoded_state, transform)
    print(f"Generated Quantum Random Character String: {random_characters}")
