import unittest
import torch
from src.complextensor.complex_tensor import ComplexTensor
from src.complextensor.apply_gate import apply_gate


class TestApplyGate(unittest.TestCase):
    def test_apply_gate_identity(self):
        """
        Test applying the Identity gate to a single-qubit state.
        The state should remain unchanged.
        """
        state = ComplexTensor(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0]))
        identity_gate = torch.eye(2)

        result = apply_gate(state, identity_gate)

        print("[TEST] Identity Gate:")
        print(f"  State Real Part: {state.real}")
        print(f"  State Imaginary Part: {state.imag}")
        print(f"  Result Real Part: {result.real}")
        print(f"  Result Imaginary Part: {result.imag}")

        self.assertTrue(torch.allclose(result.real, state.real), "Real part mismatch")
        self.assertTrue(torch.allclose(result.imag, state.imag), "Imaginary part mismatch")

    def test_apply_gate_hadamard(self):
        """
        Test applying the Hadamard gate to a single-qubit state.
        """
        state = ComplexTensor(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0]))
        hadamard_gate = torch.tensor([[1.0, 1.0], [1.0, -1.0]]) / torch.sqrt(torch.tensor(2.0))

        result = apply_gate(state, hadamard_gate)

        expected_real = torch.tensor([1.0, 1.0]) / torch.sqrt(torch.tensor(2.0))
        expected_imag = torch.tensor([0.0, 0.0])

        print("[TEST] Hadamard Gate:")
        print(f"  State Real Part: {state.real}")
        print(f"  State Imaginary Part: {state.imag}")
        print(f"  Hadamard Gate:\n{hadamard_gate}")
        print(f"  Result Real Part: {result.real}")
        print(f"  Result Imaginary Part: {result.imag}")
        print(f"  Expected Real Part: {expected_real}")
        print(f"  Expected Imaginary Part: {expected_imag}")

        self.assertTrue(torch.allclose(result.real, expected_real, rtol=1e-05, atol=1e-08), "Real part mismatch")
        self.assertTrue(torch.allclose(result.imag, expected_imag, rtol=1e-05, atol=1e-08), "Imaginary part mismatch")

    def test_apply_gate_invalid_dimensions(self):
        """
        Test applying a gate with mismatched dimensions.
        """
        state = ComplexTensor(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0]))
        invalid_gate = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # Invalid gate dimensions

        print("[TEST] Invalid Gate Dimensions:")
        print(f"  State Real Part Shape: {state.real.shape}")
        print(f"  Invalid Gate Shape: {invalid_gate.shape}")

        with self.assertRaises(ValueError) as context:
            apply_gate(state, invalid_gate)

        self.assertIn("must match the state's last dimension", str(context.exception))


if __name__ == '__main__':
    unittest.main()
