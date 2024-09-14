import torch
from torch import Tensor
from torch.autograd import Function

class ComplexFunction(Function):
    @staticmethod
    def forward(ctx, real, imag):
        ctx.save_for_backward(real, imag)
        return torch.complex(real, imag)

    @staticmethod
    def backward(ctx, grad_output):
        real, imag = ctx.saved_tensors
        return grad_output.real, grad_output.imag

class ComplexTensor:
    def __init__(self, real: Tensor, imag: Tensor = None):
        if not isinstance(real, Tensor):
            raise TypeError("real must be a Tensor")
        
        self.real = real
        if imag is None:
            self.imag = torch.zeros_like(real)
        else:
            if not isinstance(imag, Tensor):
                raise TypeError("imag must be a Tensor")
            if imag.shape != real.shape:
                raise ValueError("real and imag must have the same shape")
            self.imag = imag

    def forward(self):
        return ComplexFunction.apply(self.real, self.imag)

    def __add__(self, other):
        if not isinstance(other, ComplexTensor):
            raise TypeError("Addition is only supported between ComplexTensor instances")
        return ComplexTensor(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other):
        if not isinstance(other, ComplexTensor):
            raise TypeError("Multiplication is only supported between ComplexTensor instances")
        real_part = self.real * other.real - self.imag * other.imag
        imag_part = self.real * other.imag + self.imag * other.real
        return ComplexTensor(real_part, imag_part)

    def state_dict(self):
        return {'real': self.real, 'imag': self.imag}

    def load_state_dict(self, state_dict):
        self.real = state_dict['real']
        self.imag = state_dict['imag']