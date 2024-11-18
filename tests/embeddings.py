import os
import unittest
import torch
import numpy as np
from src.complextensor.complex_tensor import ComplexTensor
from src.complextensor.complex_embeddings import (
    real_to_complex_embeddings,
    complex_to_real_embeddings,
    ComplexEmbedding,
)


class TestComplexEmbeddingsVerbose(unittest.TestCase):
    def setUp(self):
        """
        Set up test fixtures and output directories. Initialize a sample real embedding tensor for testing.
        """
        self.batch_size = 16
        self.embedding_dim = 1048576  # Must be even for complex embeddings
        self.num_embeddings = 1000

        # Sample real embeddings
        self.real_embeddings = torch.rand(self.batch_size, self.embedding_dim, requires_grad=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Log and output directories
        self.log_file = "complex_embeddings_test.log"
        self.output_dir = "complex_embeddings_outputs"
        os.makedirs(self.output_dir, exist_ok=True)

        # Clear the log file
        with open(self.log_file, "w") as f:
            f.write("Complex Embeddings Test Log\n")

    def log(self, message):
        """Append a message to the log file."""
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

    def save_npy(self, filename, data):
        """Save data to an .npy file in the output directory."""
        filepath = os.path.join(self.output_dir, filename)
        if isinstance(data, torch.Tensor):
            # Detach the tensor if it requires gradients
            data = data.detach().cpu().numpy()
        np.save(filepath, data)

    def test_real_to_complex_conversion(self):
        """Test and log the conversion from real to complex embeddings."""
        self.log("\n--- Real to Complex Conversion ---")
        self.log(f"Input Real Embeddings:\n{self.real_embeddings}")

        complex_embeddings = real_to_complex_embeddings(self.real_embeddings, device=self.device)

        self.log(f"Real Part of Complex Embeddings:\n{complex_embeddings.real}")
        self.log(f"Imaginary Part of Complex Embeddings:\n{complex_embeddings.imag}")

        # Save large outputs
        self.save_npy("real_embeddings.npy", self.real_embeddings)
        self.save_npy("complex_real_part.npy", complex_embeddings.real)
        self.save_npy("complex_imaginary_part.npy", complex_embeddings.imag)

        # Assertions
        self.assertEqual(complex_embeddings.real.shape, (self.batch_size, self.embedding_dim // 2))
        self.assertEqual(complex_embeddings.imag.shape, (self.batch_size, self.embedding_dim // 2))

    def test_complex_to_real_conversion(self):
        """Test and log the conversion from complex embeddings back to real embeddings."""
        self.log("\n--- Complex to Real Conversion ---")
        complex_embeddings = real_to_complex_embeddings(self.real_embeddings, device=self.device)
        self.log(f"Input Complex Embeddings (Real Part):\n{complex_embeddings.real}")
        self.log(f"Input Complex Embeddings (Imaginary Part):\n{complex_embeddings.imag}")

        reconstructed_real_embeddings = complex_to_real_embeddings(complex_embeddings)
        self.log(f"Reconstructed Real Embeddings:\n{reconstructed_real_embeddings}")

        # Save large outputs
        self.save_npy("reconstructed_real_embeddings.npy", reconstructed_real_embeddings)

        # Assertions
        self.assertEqual(reconstructed_real_embeddings.shape, self.real_embeddings.shape)
        self.assertTrue(torch.allclose(reconstructed_real_embeddings, self.real_embeddings, atol=1e-6))

    def test_complex_embedding_layer(self):
        """Test and log the ComplexEmbedding layer."""
        self.log("\n--- Complex Embedding Layer ---")
        embedding_layer = ComplexEmbedding(
            num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim
        ).to(self.device)

        input_ids = torch.randint(0, self.num_embeddings, (self.batch_size,))
        self.log(f"Input Token IDs:\n{input_ids}")

        complex_embeddings = embedding_layer(input_ids)
        self.log(f"Output Complex Embeddings (Real Part):\n{complex_embeddings.real}")
        self.log(f"Output Complex Embeddings (Imaginary Part):\n{complex_embeddings.imag}")

        # Save large outputs
        self.save_npy("complex_embeddings_real.npy", complex_embeddings.real)
        self.save_npy("complex_embeddings_imaginary.npy", complex_embeddings.imag)

        # Assertions
        self.assertEqual(complex_embeddings.real.shape, (self.batch_size, self.embedding_dim // 2))
        self.assertEqual(complex_embeddings.imag.shape, (self.batch_size, self.embedding_dim // 2))

    def test_memory_savings(self):
        """Test and log memory usage for real and complex embeddings."""
        self.log("\n--- Memory Savings ---")
        complex_embeddings = real_to_complex_embeddings(self.real_embeddings, device=self.device)

        real_mem = self.real_embeddings.element_size() * self.real_embeddings.nelement()
        complex_mem = (
            complex_embeddings.real.element_size() * complex_embeddings.real.nelement() +
            complex_embeddings.imag.element_size() * complex_embeddings.imag.nelement()
        )

        self.log(f"Real Embeddings Memory: {real_mem / 1e6:.2f} MB")
        self.log(f"Complex Embeddings Memory: {complex_mem / 1e6:.2f} MB")

        # Save memory metrics
        memory_metrics = {
            "real_memory_mb": real_mem / 1e6,
            "complex_memory_mb": complex_mem / 1e6,
        }
        self.save_npy("memory_metrics.npy", torch.tensor(list(memory_metrics.values())))

        # Assertions
        self.assertTrue(complex_mem <= real_mem)


if __name__ == "__main__":
    unittest.main()
