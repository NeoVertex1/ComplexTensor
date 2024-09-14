import torch
import torchaudio
import torch.nn as nn
from complextensor import ComplexTensor  # Assuming ComplexTensor is implemented as discussed

# Load an example audio file from the LibriSpeech dataset
waveform, sample_rate = torchaudio.load('/Users/kinuhero/Music/doomer_voice/doomer2.wav')

# Apply Short-Time Fourier Transform (STFT) to the audio signal
n_fft = 400  # FFT window size
hop_length = 160  # Hop length
stft_result = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, return_complex=True)

# Create ComplexTensor from the STFT result
real_part = stft_result.real
imag_part = stft_result.imag
stft_complex_tensor = ComplexTensor(real_part, imag_part)

# Neural Network for processing the complex-valued STFT data
class ComplexSTFTNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ComplexSTFTNet, self).__init__()
        self.conv1 = ComplexConv1D(in_channels, 32, kernel_size)
        self.conv2 = ComplexConv1D(32, 64, kernel_size)
        self.fc1 = ComplexLinear(64 * 10, 128)
        self.fc2 = ComplexLinear(128, out_channels)

    def forward(self, x: ComplexTensor) -> ComplexTensor:
        x = self.conv1(x).complex_relu()
        x = self.conv2(x).complex_relu()
        x = x.real.view(x.real.size(0), -1)  # Flatten for the linear layers
        x = self.fc1(ComplexTensor(x, torch.zeros_like(x))).complex_relu()
        return self.fc2(x)

# Create the model for classification (e.g., speech recognition, signal classification)
model = ComplexSTFTNet(in_channels=1, out_channels=10, kernel_size=3)

# Forward pass the STFT ComplexTensor through the model
output = model(stft_complex_tensor)

# Example loss and optimization
loss_fn = nn.CrossEntropyLoss()
real_labels = torch.randint(0, 10, (1,))  # Example labels (for classification task)
loss = loss_fn(output.real, real_labels)

# Backward pass and optimization step
loss.backward()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer.step()

print(f"Output (Real): {output.real}")
print(f"Output (Imaginary): {output.imag}")
print(f"Loss: {loss.item()}")
