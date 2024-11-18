import unittest
import torch
from src.complextensor.complex_tensor import ComplexTensor
from src.complextensor.complex_embeddings import (
    real_to_complex_embeddings,
    complex_to_real_embeddings,
    ComplexEmbedding,
)




class TestComplexEmbeddingsVerbose(unittest.TestCase):
    def setUp(self):
        """
        Set up test fixtures. Initialize a sample real embedding tensor for testing.
        """
        self.batch_size = 4
        self.embedding_dim = 8  # Must be even for complex embeddings
        self.num_embeddings = 100

        # Sample real embeddings
        self.real_embeddings = torch.rand(self.batch_size, self.embedding_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_real_to_complex_conversion(self):
        """
        Test the conversion from real embeddings to complex embeddings and print intermediate outputs.
        """
        print("\n--- Real to Complex Conversion ---")
        print(f"Input Real Embeddings:\n{self.real_embeddings}")

        complex_embeddings = real_to_complex_embeddings(self.real_embeddings, device=self.device)

        print(f"Real Part of Complex Embeddings:\n{complex_embeddings.real}")
        print(f"Imaginary Part of Complex Embeddings:\n{complex_embeddings.imag}")

        # Assertions
        self.assertEqual(complex_embeddings.real.shape, (self.batch_size, self.embedding_dim // 2))
        self.assertEqual(complex_embeddings.imag.shape, (self.batch_size, self.embedding_dim // 2))

    def test_complex_to_real_conversion(self):
        """
        Test the conversion from complex embeddings back to real embeddings and print outputs.
        """
        print("\n--- Complex to Real Conversion ---")
        complex_embeddings = real_to_complex_embeddings(self.real_embeddings, device=self.device)
        print(f"Input Complex Embeddings (Real Part):\n{complex_embeddings.real}")
        print(f"Input Complex Embeddings (Imaginary Part):\n{complex_embeddings.imag}")

        reconstructed_real_embeddings = complex_to_real_embeddings(complex_embeddings)
        print(f"Reconstructed Real Embeddings:\n{reconstructed_real_embeddings}")

        # Assertions
        self.assertEqual(reconstructed_real_embeddings.shape, self.real_embeddings.shape)
        self.assertTrue(torch.allclose(reconstructed_real_embeddings, self.real_embeddings, atol=1e-6))

    def test_complex_embedding_layer(self):
        """
        Test the ComplexEmbedding layer and print intermediate outputs.
        """
        print("\n--- Complex Embedding Layer ---")
        embedding_layer = ComplexEmbedding(
            num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim
        ).to(self.device)

        input_ids = torch.randint(0, self.num_embeddings, (self.batch_size,))
        print(f"Input Token IDs:\n{input_ids}")

        complex_embeddings = embedding_layer(input_ids)
        print(f"Output Complex Embeddings (Real Part):\n{complex_embeddings.real}")
        print(f"Output Complex Embeddings (Imaginary Part):\n{complex_embeddings.imag}")

        # Assertions
        self.assertEqual(complex_embeddings.real.shape, (self.batch_size, self.embedding_dim // 2))
        self.assertEqual(complex_embeddings.imag.shape, (self.batch_size, self.embedding_dim // 2))

    def test_memory_savings(self):
        """
        Test and print memory usage for real and complex embeddings.
        """
        print("\n--- Memory Savings ---")
        complex_embeddings = real_to_complex_embeddings(self.real_embeddings, device=self.device)

        real_mem = self.real_embeddings.element_size() * self.real_embeddings.nelement()
        complex_mem = (
            complex_embeddings.real.element_size() * complex_embeddings.real.nelement() +
            complex_embeddings.imag.element_size() * complex_embeddings.imag.nelement()
        )

        print(f"Real Embeddings Memory: {real_mem / 1e6:.2f} MB")
        print(f"Complex Embeddings Memory: {complex_mem / 1e6:.2f} MB")

        # Assertions
        self.assertTrue(complex_mem <= real_mem)


if __name__ == "__main__":
    unittest.main()
