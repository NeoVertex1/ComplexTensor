import unittest
import torch
from src.complextensor.complex_tensor import ComplexTensor


class TestTensorProduct(unittest.TestCase):
    def test_tensor_product_simple(self):
        # Define two simple complex tensors
        tensor1 = ComplexTensor(
            real=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            imag=torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        )
        tensor2 = ComplexTensor(
            real=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            imag=torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        )

        # Compute the tensor product
        result = tensor1 @ tensor2

        # Expected result
        expected_real = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        expected_imag = torch.tensor([[0.0, 0.0], [0.0, 0.0]])

        # Assertions
        self.assertTrue(
            torch.allclose(result.real, expected_real, rtol=1e-05, atol=1e-08),
            "Real part mismatch"
        )
        self.assertTrue(
            torch.allclose(result.imag, expected_imag, rtol=1e-05, atol=1e-08),
            "Imaginary part mismatch"
        )

    def test_tensor_product_shapes(self):
        # Define two tensors with compatible shapes
        tensor1 = ComplexTensor(
            real=torch.rand(2, 3),
            imag=torch.rand(2, 3)
        )
        tensor2 = ComplexTensor(
            real=torch.rand(3, 4),
            imag=torch.rand(3, 4)
        )

        # Compute the tensor product
        result = tensor1 @ tensor2

        # Assert shapes
        self.assertEqual(result.real.shape, (2, 4))
        self.assertEqual(result.imag.shape, (2, 4))

    def test_tensor_product_with_imaginary(self):
        # Define two complex tensors
        tensor1 = ComplexTensor(
            real=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            imag=torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        )
        tensor2 = ComplexTensor(
            real=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            imag=torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        )

        # Compute the tensor product
        result = tensor1 @ tensor2

        # Expected result
        expected_real = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
        expected_imag = torch.tensor([[1.0, 2.0], [2.0, 1.0]])  # Corrected expected values

        # Debugging: print results
        print("Tensor 1 Real:\n", tensor1.real)
        print("Tensor 1 Imaginary:\n", tensor1.imag)
        print("Tensor 2 Real:\n", tensor2.real)
        print("Tensor 2 Imaginary:\n", tensor2.imag)
        print("Result Real Part:\n", result.real)
        print("Result Imaginary Part:\n", result.imag)
        print("Expected Real Part:\n", expected_real)
        print("Expected Imaginary Part:\n", expected_imag)

        # Assertions
        self.assertTrue(
            torch.allclose(result.real, expected_real, rtol=1e-05, atol=1e-08),
            "Real part mismatch"
        )
        self.assertTrue(
            torch.allclose(result.imag, expected_imag, rtol=1e-05, atol=1e-08),
            "Imaginary part mismatch"
        )


if __name__ == '__main__':
    unittest.main()
