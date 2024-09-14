import unittest
import torch
from src.complextensor.complex_tensor import ComplexTensor, ComplexFunction

class TestComplexFunction(unittest.TestCase):
    def setUp(self):
        self.real = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        self.imag = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

    def test_complex_function_forward(self):
        result = ComplexFunction.apply(self.real, self.imag)
        self.assertTrue(torch.is_complex(result))
        self.assertTrue(torch.allclose(result.real, self.real))
        self.assertTrue(torch.allclose(result.imag, self.imag))

    def test_complex_function_backward(self):
        result = ComplexFunction.apply(self.real, self.imag)
        loss = result.abs().sum()
        loss.backward()

        self.assertIsNotNone(self.real.grad)
        self.assertIsNotNone(self.imag.grad)
        self.assertFalse(torch.allclose(self.real.grad, torch.zeros_like(self.real)))
        self.assertFalse(torch.allclose(self.imag.grad, torch.zeros_like(self.imag)))

class TestComplexTensor(unittest.TestCase):
    def setUp(self):
        self.real = torch.tensor([1.0, 2.0, 3.0])
        self.imag = torch.tensor([4.0, 5.0, 6.0])
        self.ct = ComplexTensor(self.real, self.imag)

    def test_initialization(self):
        self.assertTrue(torch.allclose(self.ct.real, self.real))
        self.assertTrue(torch.allclose(self.ct.imag, self.imag))

    def test_forward(self):
        result = self.ct.forward()
        self.assertTrue(torch.is_complex(result))
        self.assertTrue(torch.allclose(result.real, self.real))
        self.assertTrue(torch.allclose(result.imag, self.imag))

    def test_addition(self):
        other = ComplexTensor(torch.tensor([1.0, 1.0, 1.0]), torch.tensor([2.0, 2.0, 2.0]))
        result = self.ct + other
        self.assertTrue(torch.allclose(result.real, self.real + 1))
        self.assertTrue(torch.allclose(result.imag, self.imag + 2))

    def test_subtraction(self):
        other = ComplexTensor(torch.tensor([1.0, 1.0, 1.0]), torch.tensor([2.0, 2.0, 2.0]))
        result = self.ct - other
        self.assertTrue(torch.allclose(result.real, self.real - 1))
        self.assertTrue(torch.allclose(result.imag, self.imag - 2))

    def test_multiplication(self):
        other = ComplexTensor(torch.tensor([2.0, 2.0, 2.0]), torch.tensor([3.0, 3.0, 3.0]))
        result = self.ct * other
        expected_real = self.real * 2 - self.imag * 3
        expected_imag = self.real * 3 + self.imag * 2
        self.assertTrue(torch.allclose(result.real, expected_real))
        self.assertTrue(torch.allclose(result.imag, expected_imag))

    def test_division(self):
        other = ComplexTensor(torch.tensor([2.0, 2.0, 2.0]), torch.tensor([3.0, 3.0, 3.0]))
        result = self.ct / other
        denominator = 4 + 9  # 2^2 + 3^2
        expected_real = (self.real * 2 + self.imag * 3) / denominator
        expected_imag = (self.imag * 2 - self.real * 3) / denominator
        self.assertTrue(torch.allclose(result.real, expected_real))
        self.assertTrue(torch.allclose(result.imag, expected_imag))

    def test_conjugate(self):
        result = self.ct.conj()
        self.assertTrue(torch.allclose(result.real, self.real))
        self.assertTrue(torch.allclose(result.imag, -self.imag))

    def test_abs(self):
        result = self.ct.abs()
        expected = torch.sqrt(self.real**2 + self.imag**2)
        self.assertTrue(torch.allclose(result, expected))

    def test_angle(self):
        result = self.ct.angle()
        expected = torch.atan2(self.imag, self.real)
        self.assertTrue(torch.allclose(result, expected))

    def test_to_polar(self):
        magnitude, phase = self.ct.to_polar()
        expected_magnitude = torch.sqrt(self.real**2 + self.imag**2)
        expected_phase = torch.atan2(self.imag, self.real)
        self.assertTrue(torch.allclose(magnitude, expected_magnitude))
        self.assertTrue(torch.allclose(phase, expected_phase))

    def test_complex_relu(self):
        result = self.ct.complex_relu()
        self.assertTrue(torch.allclose(result.real, torch.relu(self.real)))
        self.assertTrue(torch.allclose(result.imag, torch.relu(self.imag)))

    def test_complex_sigmoid(self):
        result = self.ct.complex_sigmoid()
        self.assertTrue(torch.allclose(result.real, torch.sigmoid(self.real)))
        self.assertTrue(torch.allclose(result.imag, torch.sigmoid(self.imag)))

    def test_exp(self):
        result = self.ct.exp()
        expected_real = torch.exp(self.real) * torch.cos(self.imag)
        expected_imag = torch.exp(self.real) * torch.sin(self.imag)
        self.assertTrue(torch.allclose(result.real, expected_real))
        self.assertTrue(torch.allclose(result.imag, expected_imag))

    def test_log(self):
        result = self.ct.log()
        expected_real = torch.log(self.ct.abs())
        expected_imag = self.ct.angle()
        self.assertTrue(torch.allclose(result.real, expected_real))
        self.assertTrue(torch.allclose(result.imag, expected_imag))

    def test_sin(self):
        result = self.ct.sin()
        expected_real = torch.sin(self.real) * torch.cosh(self.imag)
        expected_imag = torch.cos(self.real) * torch.sinh(self.imag)
        self.assertTrue(torch.allclose(result.real, expected_real))
        self.assertTrue(torch.allclose(result.imag, expected_imag))

    def test_cos(self):
        result = self.ct.cos()
        expected_real = torch.cos(self.real) * torch.cosh(self.imag)
        expected_imag = -torch.sin(self.real) * torch.sinh(self.imag)
        self.assertTrue(torch.allclose(result.real, expected_real))
        self.assertTrue(torch.allclose(result.imag, expected_imag))

    def test_tan(self):
        result = self.ct.tan()
        sin_ct = self.ct.sin()
        cos_ct = self.ct.cos()
        expected_real = (sin_ct.real * cos_ct.real + sin_ct.imag * cos_ct.imag) / (cos_ct.real**2 + cos_ct.imag**2)
        expected_imag = (sin_ct.imag * cos_ct.real - sin_ct.real * cos_ct.imag) / (cos_ct.real**2 + cos_ct.imag**2)
        self.assertTrue(torch.allclose(result.real, expected_real, atol=1e-5))
        self.assertTrue(torch.allclose(result.imag, expected_imag, atol=1e-5))

    def test_power(self):
        exponent = 2
        result = self.ct.power(exponent)
        expected = self.ct * self.ct
        self.assertTrue(torch.allclose(result.real, expected.real))
        self.assertTrue(torch.allclose(result.imag, expected.imag))

    def test_fft(self):
        result = self.ct.fft()
        expected = torch.fft.fft(self.ct.forward())
        self.assertTrue(torch.allclose(result.real, expected.real))
        self.assertTrue(torch.allclose(result.imag, expected.imag))

    def test_ifft(self):
        result = self.ct.ifft()
        expected = torch.fft.ifft(self.ct.forward())
        self.assertTrue(torch.allclose(result.real, expected.real))
        self.assertTrue(torch.allclose(result.imag, expected.imag))

    def test_gradient_computation(self):
        real = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        imag = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
        ct = ComplexTensor(real, imag)
        
        output = ct.forward().abs().sum()
        output.backward()
        
        self.assertIsNotNone(real.grad)
        self.assertIsNotNone(imag.grad)
        self.assertFalse(torch.allclose(real.grad, torch.zeros_like(real.grad)))
        self.assertFalse(torch.allclose(imag.grad, torch.zeros_like(imag.grad)))

    def test_to_device(self):
        if torch.cuda.is_available():
            ct_gpu = self.ct.to(torch.device('cuda'))
            self.assertEqual(ct_gpu.real.device.type, 'cuda')
            self.assertEqual(ct_gpu.imag.device.type, 'cuda')

    def test_detach(self):
        ct_detached = self.ct.detach()
        self.assertFalse(ct_detached.real.requires_grad)
        self.assertFalse(ct_detached.imag.requires_grad)

    def test_requires_grad(self):
        self.ct.requires_grad_(True)
        self.assertTrue(self.ct.real.requires_grad)
        self.assertTrue(self.ct.imag.requires_grad)

    def test_complex_function_integration(self):
        real = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        imag = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
        ct = ComplexTensor(real, imag)
        
        result = ct.forward()
        self.assertTrue(torch.is_complex(result))
        self.assertTrue(torch.allclose(result.real, real))
        self.assertTrue(torch.allclose(result.imag, imag))

        loss = result.abs().sum()
        loss.backward()

        self.assertIsNotNone(real.grad)
        self.assertIsNotNone(imag.grad)
        self.assertFalse(torch.allclose(real.grad, torch.zeros_like(real)))
        self.assertFalse(torch.allclose(imag.grad, torch.zeros_like(imag)))

if __name__ == '__main__':
    unittest.main()