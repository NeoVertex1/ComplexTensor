import matplotlib.pyplot as plt
import numpy as np

# Data for real and imaginary parts
real_part = np.array([[2.0, -1.0], [-1.0, 2.0]])
imag_part = np.array([[1.0, 2.0], [2.0, 1.0]])

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot Real Part
im1 = axes[0].imshow(real_part, cmap='Blues', interpolation='none')
axes[0].set_title('Real Part')
axes[0].set_xticks(range(real_part.shape[1]))
axes[0].set_yticks(range(real_part.shape[0]))
fig.colorbar(im1, ax=axes[0])

# Plot Imaginary Part
im2 = axes[1].imshow(imag_part, cmap='Reds', interpolation='none')
axes[1].set_title('Imaginary Part')
axes[1].set_xticks(range(imag_part.shape[1]))
axes[1].set_yticks(range(imag_part.shape[0]))
fig.colorbar(im2, ax=axes[1])

# Show the plot
plt.tight_layout()
plt.show()
