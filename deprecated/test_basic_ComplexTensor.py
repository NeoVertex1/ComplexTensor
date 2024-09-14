import unittest
import torch
from basic_ComplexTensor import ComplexTensor
import os

class TestComplexTensor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\nRunning ComplexTensor tests:")

    def setUp(self):
        torch.manual_seed(42)

    def test_initialization(self):
        real = torch.randn(2, 3)
        ct = ComplexTensor(real)
        self.assertTrue(torch.allclose(ct.real, real))
        self.assertTrue(torch.allclose(ct.imag, torch.zeros_like(real)))
        print("✓ Initialization")

    def test_forward(self):
        real = torch.randn(2, 3)
        imag = torch.randn(2, 3)
        ct = ComplexTensor(real, imag)
        result = ct.forward()
        self.assertTrue(torch.is_complex(result))
        self.assertTrue(torch.allclose(result.real, real))
        self.assertTrue(torch.allclose(result.imag, imag))
        print("✓ Forward pass")

    def test_addition(self):
        ct1 = ComplexTensor(torch.randn(2, 3), torch.randn(2, 3))
        ct2 = ComplexTensor(torch.randn(2, 3), torch.randn(2, 3))
        result = ct1 + ct2
        self.assertIsInstance(result, ComplexTensor)
        self.assertTrue(torch.allclose(result.real, ct1.real + ct2.real))
        self.assertTrue(torch.allclose(result.imag, ct1.imag + ct2.imag))
        print("✓ Addition")

    def test_multiplication(self):
        ct1 = ComplexTensor(torch.randn(2, 3), torch.randn(2, 3))
        ct2 = ComplexTensor(torch.randn(2, 3), torch.randn(2, 3))
        result = ct1 * ct2
        expected_real = ct1.real * ct2.real - ct1.imag * ct2.imag
        expected_imag = ct1.real * ct2.imag + ct1.imag * ct2.real
        self.assertIsInstance(result, ComplexTensor)
        self.assertTrue(torch.allclose(result.real, expected_real))
        self.assertTrue(torch.allclose(result.imag, expected_imag))
        print("✓ Multiplication")

    def test_gradients(self):
        real = torch.randn(2, 3, requires_grad=True)
        imag = torch.randn(2, 3, requires_grad=True)
        ct = ComplexTensor(real, imag)
        output = ct.forward().abs().sum()
        output.backward()
        
        print(f"Real grad: {real.grad}")
        print(f"Imag grad: {imag.grad}")
        
        self.assertIsNotNone(real.grad, "Real part gradient is None")
        self.assertIsNotNone(imag.grad, "Imaginary part gradient is None")
        self.assertFalse(torch.allclose(real.grad, torch.zeros_like(real.grad)), "Real part gradient is all zeros")
        self.assertFalse(torch.allclose(imag.grad, torch.zeros_like(imag.grad)), "Imaginary part gradient is all zeros")
        print("✓ Gradients")

    def test_serialization(self):
        ct = ComplexTensor(torch.randn(2, 3), torch.randn(2, 3))
        torch.save(ct.state_dict(), 'complex_tensor.pt')
        loaded_ct = ComplexTensor(torch.randn(2, 3))
        loaded_ct.load_state_dict(torch.load('complex_tensor.pt'))
        self.assertTrue(torch.allclose(ct.real, loaded_ct.real))
        self.assertTrue(torch.allclose(ct.imag, loaded_ct.imag))
        os.remove('complex_tensor.pt')
        print("✓ Serialization")

if __name__ == '__main__':
    unittest.main(verbosity=0)