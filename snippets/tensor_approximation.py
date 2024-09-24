import torch

def tensor_approximation(U, V, W, sigma):
    """
    tensor approximation formula:
    T ≈ Σᵢ σᵢ U_i ⊗ V_i ⊗ W_i

    its  used for tensor decomposition which is a generalization of matrix 
    factorization to higher-order tensors, it's useful in machine learning 
    for dimensionality reduction, feature extraction, and compressed representation of 
    high-dimensional data.

    math:
    - T is the original tensor we're approximating
    - σᵢ (sigma) are singular values, representing the importance of each component
    - U_i, V_i, W_i are factor matrices, each representing a mode of the tensor
    - ⊗ denotes the outer product

    approximation is achieved by summing over the outer products of the columns of 
    U, V, and W, weighted by their corresponding singular values.

    ML applications:
    1. Recommender systems: To capture multi-way interactions in user-item-context data
    2. Natural Language Processing: For semantic analysis and topic modeling
    3. Computer Vision: For facial recognition and image compression
    4. Time Series Analysis: To capture complex temporal patterns

    Args:
    U (torch.Tensor): Factor matrix for the first mode (shape: [I, R])
    V (torch.Tensor): Factor matrix for the second mode (shape: [J, R])
    W (torch.Tensor): Factor matrix for the third mode (shape: [K, R])
    sigma (torch.Tensor): Vector of singular values (shape: [R])

    Returns:
    torch.Tensor: Approximated tensor T (shape: [I, J, K])

    Where:
    I, J, K are the dimensions of the original tensor
    R is the rank of the approximation (number of components used)
    """
    
    # Ensure inputs are PyTorch tensors
    U, V, W, sigma = map(torch.as_tensor, (U, V, W, sigma))
    
    # Get the shapes of the factor matrices
    I, J, K = U.shape[0], V.shape[0], W.shape[0]
    R = sigma.shape[0]
    
    # Initialize the result tensor
    T = torch.zeros((I, J, K), dtype=U.dtype, device=U.device)
    
    # Compute the approximation
    for r in range(R):
        # Compute the outer product of U[:, r], V[:, r], and W[:, r]
        component = torch.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
        # Weight the component by its singular value and add to the result
        T += sigma[r] * component
    
    return T

if __name__ == "__main__":
    # Generate random factor matrices and singular values
    I, J, K, R = 10, 15, 20, 5
    U = torch.rand(I, R)
    V = torch.rand(J, R)
    W = torch.rand(K, R)
    sigma = torch.rand(R)
    
    # Compute the approximation
    T_approx = tensor_approximation(U, V, W, sigma)
    
    print(f"Approximated tensor shape: {T_approx.shape}")
    
    # Additional verification
    print(f"U shape: {U.shape}")
    print(f"V shape: {V.shape}")
    print(f"W shape: {W.shape}")
    print(f"sigma shape: {sigma.shape}")
    
    # Check if any values are NaN or Inf
    print(f"Contains NaN: {torch.isnan(T_approx).any()}")
    print(f"Contains Inf: {torch.isinf(T_approx).any()}")
    
    # Print some values from the approximated tensor
    print("Some values from T_approx:")
    print(T_approx[0, 0, 0])
    print(T_approx[-1, -1, -1])
