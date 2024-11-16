import torch
from src.complextensor.complex_tensor import ComplexTensor


def apply_gate(state: ComplexTensor, gate: torch.Tensor) -> ComplexTensor:
    """
    Applies a quantum gate to a ComplexTensor.

    Args:
        state (ComplexTensor): The quantum state to which the gate is applied.
        gate (torch.Tensor): The quantum gate to apply.

    Returns:
        ComplexTensor: The resulting state after applying the gate.

    Raises:
        ValueError: If the gate dimensions do not match the state dimensions.
    """
    # Ensure the gate is 2D and the state is 1D or 2D
    if gate.ndim != 2:
        raise ValueError("Gate must be a 2D tensor (matrix).")
    if state.real.ndim not in {1, 2}:
        raise ValueError("State must be a 1D or 2D tensor.")

    # Validate that gate's dimensions align with the state's last dimension
    if gate.shape[0] != state.real.shape[-1]:
        print(f"[ERROR] Dimension mismatch:")
        print(f"  Gate shape: {gate.shape}")
        print(f"  State shape: {state.real.shape}")
        raise ValueError(
            f"Gate's first dimension ({gate.shape[0]}) must match the state's last dimension ({state.real.shape[-1]})."
        )

    if gate.shape[1] != state.real.shape[-1]:
        print(f"[ERROR] Invalid gate shape for multiplication:")
        print(f"  Gate shape: {gate.shape}")
        print(f"  State shape: {state.real.shape}")
        raise ValueError(
            f"Gate's second dimension ({gate.shape[1]}) must match the state's last dimension ({state.real.shape[-1]})."
        )

    # Debugging: Print the initial state and gate
    print(f"[INFO] Applying gate to state:")
    print(f"  Gate:\n{gate}")
    print(f"  State Real Part:\n{state.real}")
    print(f"  State Imaginary Part:\n{state.imag}")

    # Apply the gate to both real and imaginary parts
    real_part = gate @ state.real
    imag_part = gate @ state.imag

    # Debugging: Print the resulting parts
    print(f"[INFO] Result after applying gate:")
    print(f"  Result Real Part:\n{real_part}")
    print(f"  Result Imaginary Part:\n{imag_part}")

    return ComplexTensor(real=real_part, imag=imag_part)
