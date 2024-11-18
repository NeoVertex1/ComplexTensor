import unittest
import torch
from complextensor import ComplexTensor
from src.complextensor.complex_embeddings import (
    real_to_complex_embeddings,
    complex_to_real_embeddings,
    ComplexEmbedding,
)


class TestComplexEmbeddings(unittest.TestCase):

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
        Test the conversion from real embeddings to complex embeddings.
        """
        complex_embeddings = real_to_complex_embeddings(self.real_embeddings, device=self.device)

        # Assert shapes
        self.assertEqual(complex_embeddings.real.shape, (self.batch_size, self.embedding_dim // 2))
        self.assertEqual(complex_embeddings.imag.shape, (self.batch_size, self.embedding_dim // 2))

        # Check data consistency
        for i in range(self.embedding_dim // 2):
            self.assertAlmostEqual(
                complex_embeddings.real[0, i].item(), self.real_embeddings[0, 2 * i].item()
            )
            self.assertAlmostEqual(
                complex_embeddings.imag[0, i].item(), self.real_embeddings[0, 2 * i + 1].item()
            )

    def test_complex_to_real_conversion(self):
        """
        Test the conversion from complex embeddings back to real embeddings.
        """
        complex_embeddings = real_to_complex_embeddings(self.real_embeddings, device=self.device)
        reconstructed_real_embeddings = complex_to_real_embeddings(complex_embeddings)

        # Assert shape
        self.assertEqual(reconstructed_real_embeddings.shape, self.real_embeddings.shape)

        # Assert data equality
        self.assertTrue(torch.allclose(reconstructed_real_embeddings, self.real_embeddings))

    def test_complex_embedding_layer(self):
        """
        Test the ComplexEmbedding layer.
        """
        embedding_layer = ComplexEmbedding(
            num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim
        ).to(self.device)

        # Generate random input IDs
        input_ids = torch.randint(0, self.num_embeddings, (self.batch_size,))

        # Forward pass through the layer
        complex_embeddings = embedding_layer(input_ids)

        # Assert shapes
        self.assertEqual(complex_embeddings.real.shape, (self.batch_size, self.embedding_dim // 2))
        self.assertEqual(complex_embeddings.imag.shape, (self.batch_size, self.embedding_dim // 2))

    def test_memory_savings(self):
        """
        Test the memory savings achieved by complex embeddings.
        """
        complex_embeddings = real_to_complex_embeddings(self.real_embeddings, device=self.device)

        # Calculate memory usage
        real_mem = self.real_embeddings.element_size() * self.real_embeddings.nelement()
        complex_mem = (
            complex_embeddings.real.element_size() * complex_embeddings.real.nelement() +
            complex_embeddings.imag.element_size() * complex_embeddings.imag.nelement()
        )

        # Assert complex embeddings use less memory
        self.assertTrue(complex_mem <= real_mem)
        print(f"Real Embeddings Memory: {real_mem / 1e6:.2f} MB")
        print(f"Complex Embeddings Memory: {complex_mem / 1e6:.2f} MB")


if __name__ == "__main__":
    unittest.main()
