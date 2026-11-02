"""
Dimensionality Reduction from Scratch in NumPy: PCA, Factor Analysis by EM and the Manifold Zoo

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - pca_eig
import numpy as np

def fix_signs(W):
    W = np.asarray(W, dtype=float)
    rows = np.argmax(np.abs(W), axis=0)
    flips = np.sign(W[rows, np.arange(W.shape[1])])
    flips[flips == 0] = 1.0
    return W * flips


def pca_eig(X, L):
    """
    Perform PCA using eigendecomposition of the empirical
    covariance matrix.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Input data, with one example per row.
    L : int
        Number of principal components to retain.

    Returns
    -------
    mu : ndarray of shape (D,)
        Feature-wise mean.

    W : ndarray of shape (D, L)
        Top L eigenvectors, ordered by decreasing eigenvalue,
        with deterministic signs.

    lam : ndarray of shape (D,)
        All eigenvalues in decreasing order.
    """

    # Convert input into a floating-point NumPy array.
    X = np.asarray(X, dtype=float)

    # Get the number of samples (N) and features (D).
    N, D = X.shape

    # Validate the number of principal components.
    if not isinstance(L, (int, np.integer)) or not 0 <= L <= D:
        raise ValueError("L must be an integer between 0 and D.")

    if N == 0:
        raise ValueError("X must contain at least one sample.")

    # Compute the feature-wise mean.
    mu = np.mean(X, axis=0)

    # Center the data by subtracting the mean.
    Xc = X - mu

    # Compute the empirical covariance matrix.
    # Divide by N, not N - 1, as specified in the project.
    Sigma = (Xc.T @ Xc) / N

    # Perform eigendecomposition of the symmetric covariance matrix.
    # eigh is preferred for symmetric matrices.
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)

    # Sort eigenvalues and corresponding eigenvectors
    # in decreasing order of eigenvalue.
    order = np.argsort(eigenvalues)[::-1]

    # Reorder all eigenvalues.
    lam = eigenvalues[order]

    # Clip negative eigenvalues caused by floating-point
    # rounding errors to zero.
    lam = np.maximum(lam, 0.0)

    # Select the top L eigenvectors as columns.
    W = eigenvectors[:, order[:L]]

    # Fix eigenvector signs to ensure deterministic results.
    W = fix_signs(W)

    # Return the mean, top L eigenvectors, and all eigenvalues.
    return mu, W, lam

