import torch
import logging
from torch import Tensor
from torch.autograd import Function
from typing import Tuple, Dict, Optional, Union


# Configure the logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)  # Set to DEBUG to capture all debug messages, set to CRITICAL to reduce messages.

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add the handler to the logger if not already added
if not logger.handlers:
    logger.addHandler(ch)


class ComplexFunction(Function):
    """
    Custom autograd Function to handle complex tensor operations.
    Supports higher-order gradients and per-component gradient manipulation.
    """

    @staticmethod
    def forward(ctx, real: Tensor, imag: Tensor) -> Tensor:
        """
        Forward pass for creating a complex tensor from real and imaginary parts.

        Args:
            real (Tensor): Real part of the complex tensor.
            imag (Tensor): Imaginary part of the complex tensor.

        Returns:
            Tensor: Complex tensor combining real and imaginary parts.
        """
        logger.debug(f"ComplexFunction.forward called with real shape {real.shape}, "
                     f"imag shape {imag.shape}, device {real.device}, dtype {real.dtype}")
        ctx.save_for_backward(real, imag)
        complex_tensor = torch.complex(real, imag)
        logger.debug(f"ComplexFunction.forward output shape {complex_tensor.shape}, "
                     f"device {complex_tensor.device}, dtype {complex_tensor.dtype}")
        return complex_tensor

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Backward pass for ComplexFunction. Computes gradients for real and imaginary parts.

        Args:
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
            Tuple[Tensor, Tensor]: Gradients with respect to real and imaginary parts.
        """
        real, imag = ctx.saved_tensors
        logger.debug(f"ComplexFunction.backward called with grad_output shape {grad_output.shape}, "
                    f"device {grad_output.device}, dtype {grad_output.dtype}")
        
        # Ensure gradients are computed for both real and imaginary parts
        grad_real = grad_output.real
        grad_imag = grad_output.imag
        
        # If grad_output is not complex, use the same gradient for both parts
        if not grad_output.is_complex():
            grad_real = grad_output
            grad_imag = grad_output

        logger.debug(f"ComplexFunction.backward returning grad_real shape {grad_real.shape}, "
                    f"grad_imag shape {grad_imag.shape}")
        return grad_real, grad_imag


class ComplexTensor:
    def __init__(self, real: Tensor, imag: Optional[Tensor] = None) -> None:
        """
        Initializes a ComplexTensor with real and imaginary parts.

        Args:
            real (Tensor): Real part of the complex tensor.
            imag (Tensor, optional): Imaginary part of the complex tensor. 
                                    If not provided, defaults to a tensor of zeros.

        Raises:
            TypeError: If real or imag is not a Tensor.
            ValueError: If real and imag have different shapes.
        """
        logger.debug("Initializing ComplexTensor.")
        if not isinstance(real, Tensor):
            logger.error("Initialization failed: 'real' must be a Tensor.")
            raise TypeError("real must be a Tensor")

        self.real: Tensor = real
        logger.debug(f"Real part initialized with shape {self.real.shape}, "
                    f"device {self.real.device}, dtype {self.real.dtype}, "
                    f"requires_grad={self.real.requires_grad}")

        if imag is None:
            self.imag: Tensor = torch.zeros_like(real, requires_grad=real.requires_grad)
            logger.debug(f"Imaginary part not provided. Initialized to zeros with shape {self.imag.shape}, "
                        f"device {self.imag.device}, dtype {self.imag.dtype}, "
                        f"requires_grad={self.imag.requires_grad}")
        else:
            if not isinstance(imag, Tensor):
                logger.error("Initialization failed: 'imag' must be a Tensor.")
                raise TypeError("imag must be a Tensor")
            if imag.shape != real.shape:
                logger.error("Initialization failed: 'real' and 'imag' must have the same shape.")
                raise ValueError("real and imag have different shapes")
            self.imag: Tensor = imag
            logger.debug(f"Imaginary part initialized with shape {self.imag.shape}, "
                        f"device {self.imag.device}, dtype {self.imag.dtype}, "
                        f"requires_grad={self.imag.requires_grad}")

    def forward(self) -> Tensor:
        """
        Forward pass that combines real and imaginary parts into a complex tensor.

        Returns:
            Tensor: Complex tensor created from real and imaginary parts.
        """
        logger.debug(f"ComplexTensor.forward called with real shape {self.real.shape}, "
                     f"imag shape {self.imag.shape}")
        complex_tensor = ComplexFunction.apply(self.real, self.imag)
        logger.debug(f"ComplexTensor.forward output shape {complex_tensor.shape}")
        return complex_tensor

    def __add__(self, other: 'ComplexTensor') -> 'ComplexTensor':
        """
        Defines addition operation for ComplexTensor.

        Args:
            other (ComplexTensor): Another ComplexTensor instance to add.

        Returns:
            ComplexTensor: Result of the addition.

        Raises:
            TypeError: If other is not a ComplexTensor instance.
        """
        logger.debug(f"Adding ComplexTensor with shape {self.real.shape} to ComplexTensor with shape {other.real.shape}")
        if not isinstance(other, ComplexTensor):
            logger.error("Addition failed: Other operand is not a ComplexTensor instance.")
            raise TypeError("Addition is only supported between ComplexTensor instances")
        try:
            real_sum = self.real + other.real
            imag_sum = self.imag + other.imag
            logger.debug(f"Addition result real shape {real_sum.shape}, imag shape {imag_sum.shape}")
            return ComplexTensor(real_sum, imag_sum)
        except RuntimeError as e:
            logger.error(f"Addition failed due to shape mismatch: {e}")
            raise

    def __sub__(self, other: 'ComplexTensor') -> 'ComplexTensor':
        """
        Defines subtraction operation for ComplexTensor.

        Args:
            other (ComplexTensor): Another ComplexTensor instance to subtract.

        Returns:
            ComplexTensor: Result of the subtraction.

        Raises:
            TypeError: If other is not a ComplexTensor instance.
        """
        logger.debug(f"Subtracting ComplexTensor with shape {other.real.shape} from ComplexTensor with shape {self.real.shape}")
        if not isinstance(other, ComplexTensor):
            logger.error("Subtraction failed: Other operand is not a ComplexTensor instance.")
            raise TypeError("Subtraction is only supported between ComplexTensor instances")
        try:
            real_diff = self.real - other.real
            imag_diff = self.imag - other.imag
            logger.debug(f"Subtraction result real shape {real_diff.shape}, imag shape {imag_diff.shape}")
            return ComplexTensor(real_diff, imag_diff)
        except RuntimeError as e:
            logger.error(f"Subtraction failed due to shape mismatch: {e}")
            raise

    def __mul__(self, other: 'ComplexTensor') -> 'ComplexTensor':
        """
        Defines multiplication operation for ComplexTensor.

        Args:
            other (ComplexTensor): Another ComplexTensor instance to multiply.

        Returns:
            ComplexTensor: Result of the multiplication.

        Raises:
            TypeError: If other is not a ComplexTensor instance.
        """
        logger.debug(f"Multiplying ComplexTensor with shape {self.real.shape} by ComplexTensor with shape {other.real.shape}")
        if not isinstance(other, ComplexTensor):
            logger.error("Multiplication failed: Other operand is not a ComplexTensor instance.")
            raise TypeError("Multiplication is only supported between ComplexTensor instances")
        try:
            real_part = self.real * other.real - self.imag * other.imag
            imag_part = self.real * other.imag + self.imag * other.real
            logger.debug(f"Multiplication result real shape {real_part.shape}, imag shape {imag_part.shape}")
            return ComplexTensor(real_part, imag_part)
        except RuntimeError as e:
            logger.error(f"Multiplication failed due to shape mismatch: {e}")
            raise

    def __matmul__(self, other: 'ComplexTensor') -> 'ComplexTensor':
        """
        Implements matrix multiplication for ComplexTensor.

        Args:
            other (ComplexTensor): The other ComplexTensor to multiply with.

        Returns:
            ComplexTensor: The result of the matrix multiplication.

        Raises:
            TypeError: If other is not a ComplexTensor.
            ValueError: If the shapes of the tensors are not compatible for matrix multiplication.
        """
        logger.debug(f"Performing matrix multiplication between ComplexTensors: "
                    f"self.real shape {self.real.shape}, other.real shape {other.real.shape}")
        
        if not isinstance(other, ComplexTensor):
            logger.error("Matrix multiplication failed: Other operand is not a ComplexTensor instance.")
            raise TypeError("Matrix multiplication is only supported between ComplexTensor instances")

        # Ensure compatibility of shapes for matrix multiplication
        if self.real.shape[-1] != other.real.shape[-2]:
            logger.error(f"Matrix multiplication failed: incompatible shapes "
                        f"{self.real.shape} and {other.real.shape}.")
            raise ValueError("Matrix multiplication requires matching inner dimensions")

        # Compute the real and imaginary parts for the result
        real_part = self.real @ other.real - self.imag @ other.imag
        imag_part = self.real @ other.imag + self.imag @ other.real

        logger.debug(f"Matrix multiplication result: real_part shape {real_part.shape}, "
                    f"imag_part shape {imag_part.shape}")

        # Return a new ComplexTensor with the computed real and imaginary parts
        return ComplexTensor(real=real_part, imag=imag_part)



    def __truediv__(self, other: 'ComplexTensor') -> 'ComplexTensor':
        """
        Defines division operation for ComplexTensor.

        Args:
            other (ComplexTensor): Another ComplexTensor instance to divide by.

        Returns:
            ComplexTensor: Result of the division.

        Raises:
            TypeError: If other is not a ComplexTensor instance.
            ZeroDivisionError: If the denominator is zero.
        """
        logger.debug(f"Dividing ComplexTensor with shape {self.real.shape} by ComplexTensor with shape {other.real.shape}")
        if not isinstance(other, ComplexTensor):
            logger.error("Division failed: Other operand is not a ComplexTensor instance.")
            raise TypeError("Division is only supported between ComplexTensor instances")
        denominator = other.real ** 2 + other.imag ** 2
        logger.debug(f"Denominator shape {denominator.shape}, any zeros: {torch.any(denominator == 0).item()}")
        if torch.any(denominator == 0):
            logger.error("Division failed: Division by zero encountered in denominator.")
            raise ZeroDivisionError("Division by zero encountered in denominator")
        try:
            real = (self.real * other.real + self.imag * other.imag) / denominator
            imag = (self.imag * other.real - self.real * other.imag) / denominator
            logger.debug(f"Division result real shape {real.shape}, imag shape {imag.shape}")
            return ComplexTensor(real, imag)
        except RuntimeError as e:
            logger.error(f"Division failed due to shape mismatch: {e}")
            raise

    def conj(self) -> 'ComplexTensor':
        """
        Returns the complex conjugate of the ComplexTensor.

        Returns:
            ComplexTensor: Conjugate of the original ComplexTensor.
        """
        logger.debug(f"Computing conjugate of ComplexTensor with shape {self.real.shape}")
        conjugate = ComplexTensor(self.real, -self.imag)
        logger.debug(f"Conjugate shape {conjugate.real.shape}, {conjugate.imag.shape}")
        return conjugate

    def abs(self) -> Tensor:
        """
        Returns the magnitude of the ComplexTensor.

        Returns:
            Tensor: Magnitude computed as sqrt(real^2 + imag^2).
        """
        logger.debug(f"Computing magnitude of ComplexTensor with shape {self.real.shape}")
        magnitude = torch.sqrt(self.real ** 2 + self.imag ** 2)
        logger.debug(f"Magnitude shape {magnitude.shape}")
        return magnitude

    def angle(self) -> Tensor:
        """
        Returns the phase (angle) of the ComplexTensor.

        Returns:
            Tensor: Phase computed as arctan(imag / real).
        """
        logger.debug(f"Computing phase of ComplexTensor with shape {self.real.shape}")
        phase = torch.atan2(self.imag, self.real)
        logger.debug(f"Phase shape {phase.shape}")
        return phase

    def to_polar(self) -> Tuple[Tensor, Tensor]:
        """
        Converts the ComplexTensor to its polar representation.

        Returns:
            Tuple[Tensor, Tensor]: Tuple containing magnitude and phase.
        """
        logger.debug(f"Converting ComplexTensor to polar representation with shape {self.real.shape}")
        magnitude = self.abs()
        phase = self.angle()
        logger.debug(f"Polar representation - magnitude shape {magnitude.shape}, phase shape {phase.shape}")
        return (magnitude, phase)

    def complex_relu(self) -> 'ComplexTensor':
        """
        Applies the ReLU activation function to the real and imaginary parts separately.

        Returns:
            ComplexTensor: Result after applying ReLU.
        """
        logger.debug(f"Applying ReLU activation to ComplexTensor with shape {self.real.shape}")
        relu_real = torch.relu(self.real)
        relu_imag = torch.relu(self.imag)
        logger.debug(f"ReLU activation result real shape {relu_real.shape}, imag shape {relu_imag.shape}")
        return ComplexTensor(relu_real, relu_imag)

    def complex_sigmoid(self) -> 'ComplexTensor':
        """
        Applies the Sigmoid activation function to the real and imaginary parts separately.

        Returns:
            ComplexTensor: Result after applying Sigmoid.
        """
        logger.debug(f"Applying Sigmoid activation to ComplexTensor with shape {self.real.shape}")
        sigmoid_real = torch.sigmoid(self.real)
        sigmoid_imag = torch.sigmoid(self.imag)
        logger.debug(f"Sigmoid activation result real shape {sigmoid_real.shape}, imag shape {sigmoid_imag.shape}")
        return ComplexTensor(sigmoid_real, sigmoid_imag)

    def state_dict(self) -> Dict[str, Tensor]:
        """
        Returns the state of the ComplexTensor as a dictionary.

        Returns:
            Dict[str, Tensor]: Dictionary containing real and imaginary parts.
        """
        logger.debug("Creating state_dict for ComplexTensor.")
        return {'real': self.real, 'imag': self.imag}

    def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        """
        Loads the state of the ComplexTensor from a dictionary.

        Args:
            state_dict (Dict[str, Tensor]): Dictionary containing real and imaginary parts.

        Raises:
            KeyError: If 'real' or 'imag' keys are missing in the state_dict.
        """
        logger.debug("Loading state_dict into ComplexTensor.")
        if 'real' not in state_dict or 'imag' not in state_dict:
            logger.error("State dictionary must contain 'real' and 'imag' keys.")
            raise KeyError("State dictionary must contain 'real' and 'imag' keys")
        self.real = state_dict['real']
        self.imag = state_dict['imag']
        logger.debug(f"State loaded. Real shape {self.real.shape}, Imag shape {self.imag.shape}")

    def __repr__(self) -> str:
        """
        Returns the official string representation of the ComplexTensor.

        Returns:
            str: String representation of the ComplexTensor.
        """
        return f"ComplexTensor(real={self.real}, imag={self.imag})"

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> 'ComplexTensor':
        """
        Moves the ComplexTensor to the specified device and dtype.

        Args:
            device (torch.device): The target device.
            dtype (torch.dtype, optional): The desired data type.

        Returns:
            ComplexTensor: ComplexTensor moved to the target device and dtype.
        """
        if dtype:
            logger.debug(f"Moving ComplexTensor to device {device} and dtype {dtype}.")
            real_moved = self.real.to(device=device, dtype=dtype)
            imag_moved = self.imag.to(device=device, dtype=dtype)
            logger.debug(f"Moved ComplexTensor to device {device}, dtype {dtype}.")
            return ComplexTensor(real_moved, imag_moved)
        else:
            logger.debug(f"Moving ComplexTensor to device {device} without changing dtype.")
            real_moved = self.real.to(device=device)
            imag_moved = self.imag.to(device=device)
            logger.debug(f"Moved ComplexTensor to device {device} without changing dtype.")
            return ComplexTensor(real_moved, imag_moved)

    # Advanced Autograd Customization
    def apply_gradient_manipulation(self, grad_scale_real: float = 1.0, grad_scale_imag: float = 1.0) -> 'ComplexTensor':
        """
        Applies gradient scaling to the real and imaginary parts separately.

        Args:
            grad_scale_real (float): Scaling factor for the real part gradients.
            grad_scale_imag (float): Scaling factor for the imaginary part gradients.

        Returns:
            ComplexTensor: ComplexTensor with scaled gradients.
        """
        logger.debug(f"Applying gradient manipulation with scale_real={grad_scale_real}, scale_imag={grad_scale_imag}")
        # Custom autograd Function to scale gradients
        class GradientManipulationFunction(Function):
            @staticmethod
            def forward(ctx, real: Tensor, imag: Tensor, scale_real: float, scale_imag: float) -> Tensor:
                ctx.scale_real = scale_real
                ctx.scale_imag = scale_imag
                return ComplexFunction.apply(real, imag)

            @staticmethod
            def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor, None, None]:
                grad_real = grad_output.real * ctx.scale_real
                grad_imag = grad_output.imag * ctx.scale_imag
                logger.debug(f"GradientManipulationFunction.backward with scale_real={ctx.scale_real}, "
                             f"scale_imag={ctx.scale_imag}")
                return grad_real, grad_imag, None, None

        complex_tensor = GradientManipulationFunction.apply(self.real, self.imag, grad_scale_real, grad_scale_imag)
        logger.debug(f"Gradient manipulation result shape {complex_tensor.shape}")
        return ComplexTensor(complex_tensor.real, complex_tensor.imag)

    # Complex Math Functions
    def exp(self) -> 'ComplexTensor':
        """
        Computes the complex exponential of the ComplexTensor.

        Returns:
            ComplexTensor: Result of the exponential operation.
        """
        logger.debug(f"Computing exponential of ComplexTensor with shape {self.real.shape}")
        exp_real = torch.exp(self.real) * torch.cos(self.imag)
        exp_imag = torch.exp(self.real) * torch.sin(self.imag)
        logger.debug(f"Exponential result shape {exp_real.shape}, {exp_imag.shape}")
        return ComplexTensor(exp_real, exp_imag)

    def log(self) -> 'ComplexTensor':
        """
        Computes the complex logarithm of the ComplexTensor.

        Returns:
            ComplexTensor: Result of the logarithm operation.
        """
        logger.debug(f"Computing logarithm of ComplexTensor with shape {self.real.shape}")
        magnitude = self.abs()
        phase = self.angle()
        log_real = torch.log(magnitude)
        log_imag = phase
        logger.debug(f"Logarithm result shape {log_real.shape}, {log_imag.shape}")
        return ComplexTensor(log_real, log_imag)

    def sin(self) -> 'ComplexTensor':
        """
        Computes the complex sine of the ComplexTensor.

        Returns:
            ComplexTensor: Result of the sine operation.
        """
        logger.debug(f"Computing sine of ComplexTensor with shape {self.real.shape}")
        sin_real = torch.sin(self.real) * torch.cosh(self.imag)
        sin_imag = torch.cos(self.real) * torch.sinh(self.imag)
        logger.debug(f"Sine result shape {sin_real.shape}, {sin_imag.shape}")
        return ComplexTensor(sin_real, sin_imag)

    def cos(self) -> 'ComplexTensor':
        """
        Computes the complex cosine of the ComplexTensor.

        Returns:
            ComplexTensor: Result of the cosine operation.
        """
        logger.debug(f"Computing cosine of ComplexTensor with shape {self.real.shape}")
        cos_real = torch.cos(self.real) * torch.cosh(self.imag)
        cos_imag = -torch.sin(self.real) * torch.sinh(self.imag)
        logger.debug(f"Cosine result shape {cos_real.shape}, {cos_imag.shape}")
        return ComplexTensor(cos_real, cos_imag)

    def tan(self) -> 'ComplexTensor':
        """
        Computes the complex tangent of the ComplexTensor.

        Returns:
            ComplexTensor: Result of the tangent operation.
        """
        logger.debug(f"Computing tangent of ComplexTensor with shape {self.real.shape}")
        sin_ct = self.sin()
        cos_ct = self.cos()
        # To prevent division by zero, add a small epsilon
        epsilon = 1e-12
        cos_real_safe = cos_ct.real + epsilon
        tan_real = sin_ct.real / cos_real_safe
        tan_imag = sin_ct.imag / cos_real_safe
        logger.debug(f"Tangent result shape {tan_real.shape}, {tan_imag.shape}")
        return ComplexTensor(tan_real, tan_imag)

    # Fast Fourier Transform (FFT)
    def fft(self, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> 'ComplexTensor':
        """
        Computes the Fast Fourier Transform of the ComplexTensor.

        Args:
            n (int, optional): Signal length. If n is smaller than the size of the input, the input is cropped.
                                If it is larger, the input is padded with zeros.
            dim (int, optional): Dimension over which to compute the FFT. Default: -1
            norm (str, optional): Normalization mode. Can be 'forward', 'backward', or 'ortho'.

        Returns:
            ComplexTensor: Result of the FFT operation.
        """
        logger.debug(f"Computing FFT of ComplexTensor with shape {self.real.shape} along dim {dim}, n={n}, norm={norm}")
        complex_tensor = self.forward()
        fft_result = torch.fft.fft(complex_tensor, n=n, dim=dim, norm=norm)
        logger.debug(f"FFT result shape {fft_result.shape}")
        return ComplexTensor(fft_result.real, fft_result.imag)

    def ifft(self, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> 'ComplexTensor':
        """
        Computes the Inverse Fast Fourier Transform of the ComplexTensor.

        Args:
            n (int, optional): Signal length. If n is smaller than the size of the input, the input is cropped.
                                If it is larger, the input is padded with zeros.
            dim (int, optional): Dimension over which to compute the IFFT. Default: -1
            norm (str, optional): Normalization mode. Can be 'forward', 'backward', or 'ortho'.

        Returns:
            ComplexTensor: Result of the IFFT operation.
        """
        logger.debug(f"Computing IFFT of ComplexTensor with shape {self.real.shape} along dim {dim}, n={n}, norm={norm}")
        complex_tensor = self.forward()
        ifft_result = torch.fft.ifft(complex_tensor, n=n, dim=dim, norm=norm)
        logger.debug(f"IFFT result shape {ifft_result.shape}")
        return ComplexTensor(ifft_result.real, ifft_result.imag)

    # Advanced Autograd Customization
    def apply_gradient_manipulation(self, grad_scale_real: float = 1.0, grad_scale_imag: float = 1.0) -> 'ComplexTensor':
        """
        Applies gradient scaling to the real and imaginary parts separately.

        Args:
            grad_scale_real (float): Scaling factor for the real part gradients.
            grad_scale_imag (float): Scaling factor for the imaginary part gradients.

        Returns:
            ComplexTensor: ComplexTensor with scaled gradients.
        """
        logger.debug(f"Applying gradient manipulation with scale_real={grad_scale_real}, scale_imag={grad_scale_imag}")
        # Custom autograd Function to scale gradients
        class GradientManipulationFunction(Function):
            @staticmethod
            def forward(ctx, real: Tensor, imag: Tensor, scale_real: float, scale_imag: float) -> Tensor:
                ctx.scale_real = scale_real
                ctx.scale_imag = scale_imag
                return ComplexFunction.apply(real, imag)

            @staticmethod
            def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor, None, None]:
                grad_real = grad_output.real * ctx.scale_real
                grad_imag = grad_output.imag * ctx.scale_imag
                logger.debug(f"GradientManipulationFunction.backward with scale_real={ctx.scale_real}, "
                             f"scale_imag={ctx.scale_imag}")
                return grad_real, grad_imag, None, None

        complex_tensor = GradientManipulationFunction.apply(self.real, self.imag, grad_scale_real, grad_scale_imag)
        logger.debug(f"Gradient manipulation result shape {complex_tensor.shape}")
        return ComplexTensor(complex_tensor.real, complex_tensor.imag)

    # Complex Math Functions
    def exp(self) -> 'ComplexTensor':
        """
        Computes the complex exponential of the ComplexTensor.

        Returns:
            ComplexTensor: Result of the exponential operation.
        """
        logger.debug(f"Computing exponential of ComplexTensor with shape {self.real.shape}")
        exp_real = torch.exp(self.real) * torch.cos(self.imag)
        exp_imag = torch.exp(self.real) * torch.sin(self.imag)
        logger.debug(f"Exponential result shape {exp_real.shape}, {exp_imag.shape}")
        return ComplexTensor(exp_real, exp_imag)

    def log(self) -> 'ComplexTensor':
        """
        Computes the complex logarithm of the ComplexTensor.

        Returns:
            ComplexTensor: Result of the logarithm operation.
        """
        logger.debug(f"Computing logarithm of ComplexTensor with shape {self.real.shape}")
        magnitude = self.abs()
        phase = self.angle()
        log_real = torch.log(magnitude)
        log_imag = phase
        logger.debug(f"Logarithm result shape {log_real.shape}, {log_imag.shape}")
        return ComplexTensor(log_real, log_imag)

    def sin(self) -> 'ComplexTensor':
        """
        Computes the complex sine of the ComplexTensor.

        Returns:
            ComplexTensor: Result of the sine operation.
        """
        logger.debug(f"Computing sine of ComplexTensor with shape {self.real.shape}")
        sin_real = torch.sin(self.real) * torch.cosh(self.imag)
        sin_imag = torch.cos(self.real) * torch.sinh(self.imag)
        logger.debug(f"Sine result shape {sin_real.shape}, {sin_imag.shape}")
        return ComplexTensor(sin_real, sin_imag)

    def cos(self) -> 'ComplexTensor':
        """
        Computes the complex cosine of the ComplexTensor.

        Returns:
            ComplexTensor: Result of the cosine operation.
        """
        logger.debug(f"Computing cosine of ComplexTensor with shape {self.real.shape}")
        cos_real = torch.cos(self.real) * torch.cosh(self.imag)
        cos_imag = -torch.sin(self.real) * torch.sinh(self.imag)
        logger.debug(f"Cosine result shape {cos_real.shape}, {cos_imag.shape}")
        return ComplexTensor(cos_real, cos_imag)

    def tan(self) -> 'ComplexTensor':
        """
        Computes the complex tangent of the ComplexTensor.

        Returns:
            ComplexTensor: Result of the tangent operation.
        """
        logger.debug(f"Computing tangent of ComplexTensor with shape {self.real.shape}")
        sin_ct = self.sin()
        cos_ct = self.cos()
        # To prevent division by zero, add a small epsilon
        epsilon = 1e-12
        cos_real_safe = cos_ct.real + epsilon
        cos_imag_safe = cos_ct.imag + epsilon
        denominator = cos_real_safe**2 + cos_imag_safe**2
        tan_real = (sin_ct.real * cos_real_safe + sin_ct.imag * cos_imag_safe) / denominator
        tan_imag = (sin_ct.imag * cos_real_safe - sin_ct.real * cos_imag_safe) / denominator
        logger.debug(f"Tangent result shape {tan_real.shape}, {tan_imag.shape}")
        return ComplexTensor(tan_real, tan_imag)

    def power(self, exponent: Union[float, 'ComplexTensor']) -> 'ComplexTensor':
        """
        Raises the ComplexTensor to a power.

        Args:
            exponent (Union[float, ComplexTensor]): The exponent to raise the ComplexTensor to.

        Returns:
            ComplexTensor: Result of the power operation.
        """
        logger.debug(f"Computing power of ComplexTensor with shape {self.real.shape}")
        if isinstance(exponent, (int, float)):
            magnitude, angle = self.to_polar()
            new_magnitude = magnitude ** exponent
            new_angle = angle * exponent
            power_real = new_magnitude * torch.cos(new_angle)
            power_imag = new_magnitude * torch.sin(new_angle)
        elif isinstance(exponent, ComplexTensor):
            log_self = self.log()
            product = log_self * exponent
            power = product.exp()
            power_real, power_imag = power.real, power.imag
        else:
            raise TypeError("Exponent must be a scalar or ComplexTensor")
        
        logger.debug(f"Power result shape {power_real.shape}, {power_imag.shape}")
        return ComplexTensor(power_real, power_imag)

    def __pow__(self, exponent: Union[float, 'ComplexTensor']) -> 'ComplexTensor':
        """
        Implements the power operation using the ** operator.

        Args:
            exponent (Union[float, ComplexTensor]): The exponent to raise the ComplexTensor to.

        Returns:
            ComplexTensor: Result of the power operation.
        """
        return self.power(exponent)

    @staticmethod
    def from_polar(magnitude: Tensor, phase: Tensor) -> 'ComplexTensor':
        """
        Creates a ComplexTensor from polar coordinates.

        Args:
            magnitude (Tensor): The magnitude of the complex numbers.
            phase (Tensor): The phase of the complex numbers in radians.

        Returns:
            ComplexTensor: The resulting complex tensor.
        """
        logger.debug(f"Creating ComplexTensor from polar coordinates with magnitude shape {magnitude.shape} and phase shape {phase.shape}")
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        return ComplexTensor(real, imag)

    def __len__(self) -> int:
        """
        Returns the length of the ComplexTensor.

        Returns:
            int: The length of the tensor.
        """
        return len(self.real)

    def __getitem__(self, index) -> 'ComplexTensor':
        """
        Implements indexing for ComplexTensor.

        Args:
            index: The index or slice to access.

        Returns:
            ComplexTensor: A new ComplexTensor with the indexed values.
        """
        return ComplexTensor(self.real[index], self.imag[index])

    def __setitem__(self, index, value: 'ComplexTensor') -> None:
        """
        Implements item assignment for ComplexTensor.

        Args:
            index: The index or slice to assign to.
            value (ComplexTensor): The value to assign.
        """
        if not isinstance(value, ComplexTensor):
            raise TypeError("Can only assign ComplexTensor to indexed positions")
        self.real[index] = value.real
        self.imag[index] = value.imag

    @property
    def shape(self) -> torch.Size:
        """
        Returns the shape of the ComplexTensor.

        Returns:
            torch.Size: The shape of the tensor.
        """
        return self.real.shape

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the data type of the ComplexTensor.

        Returns:
            torch.dtype: The data type of the tensor.
        """
        return self.real.dtype

    @property
    def device(self) -> torch.device:
        """
        Returns the device of the ComplexTensor.

        Returns:
            torch.device: The device of the tensor.
        """
        return self.real.device

    def detach(self) -> 'ComplexTensor':
        """
        Returns a new ComplexTensor detached from the current graph.

        Returns:
            ComplexTensor: A new ComplexTensor with detached tensors.
        """
        return ComplexTensor(self.real.detach(), self.imag.detach())

    def requires_grad_(self, requires_grad: bool = True) -> 'ComplexTensor':
        """
        Change if autograd should record operations on this tensor.

        Args:
            requires_grad (bool): If autograd should record operations on this tensor.

        Returns:
            ComplexTensor: This ComplexTensor.
        """
        self.real.requires_grad_(requires_grad)
        self.imag.requires_grad_(requires_grad)
        return self

# End of ComplexTensor class
