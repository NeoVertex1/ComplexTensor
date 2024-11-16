import torch
from .complex_tensor import ComplexTensor


def tensor_product(tensor1: ComplexTensor, tensor2: ComplexTensor) -> ComplexTensor:
    """
    Compute the tensor (Kronecker) product of two ComplexTensor objects.

    Args:
        tensor1 (ComplexTensor): The first complex tensor.
        tensor2 (ComplexTensor): The second complex tensor.

    Returns:
        ComplexTensor: The resulting tensor product as a new ComplexTensor.
    """
    real_part = torch.kron(tensor1.real, tensor2.real) - torch.kron(tensor1.imag, tensor2.imag)
    imag_part = torch.kron(tensor1.real, tensor2.imag) + torch.kron(tensor1.imag, tensor2.real)
    return ComplexTensor(real_part, imag_part)
