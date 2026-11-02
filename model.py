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

# Step 2 - pca_gram
def pca_gram(X, L):
    """
    Perform PCA using eigendecomposition of the N x N Gram matrix.

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
        Top L principal components as orthonormal columns,
        with deterministic signs.

    lam : ndarray of shape (N,)
        All covariance eigenvalues in decreasing order.
    """

    # Convert input into a floating-point NumPy array.
    X = np.asarray(X, dtype=float)

    # Get the number of samples (N) and features (D).
    N, D = X.shape

    # Validate the number of principal components.
    if not isinstance(L, (int, np.integer)) or not 0 <= L <= min(N, D):
        raise ValueError("L must be an integer between 0 and min(N, D).")

    if N == 0 or D == 0:
        raise ValueError("X must contain at least one sample and one feature.")

    # Compute the feature-wise mean.
    mu = np.mean(X, axis=0)

    # Center the data.
    Xc = X - mu

    # Construct the N x N Gram matrix.
    G = Xc @ Xc.T

    # Compute the eigendecomposition of the Gram matrix.
    # Since G is symmetric, use np.linalg.eigh.
    eigenvalues, U = np.linalg.eigh(G)

    # Sort eigenvalues and their corresponding eigenvectors
    # in decreasing order.
    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    U = U[:, order]

    # Clip negative eigenvalues caused by numerical rounding.
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Map the top L eigenvectors from sample space
    # back into the original feature space.
    if L > 0:
        top_eigenvalues = eigenvalues[:L]

        # The selected eigenvalues must be positive to avoid
        # division by zero when mapping eigenvectors.
        if np.any(top_eigenvalues <= 0):
            raise ValueError("L must not exceed the rank of the centered data.")

        # V = Xc.T @ U @ Lambda^(-1/2)
        W = (Xc.T @ U[:, :L]) / np.sqrt(top_eigenvalues)[None, :]
    else:
        W = np.empty((D, 0), dtype=float)

    # Fix the signs of the principal components.
    W = fix_signs(W)

    # Convert Gram-matrix eigenvalues into covariance eigenvalues.
    # The covariance matrix uses division by N.
    lam = eigenvalues / N

    # Return the mean, top L principal components,
    # and all N covariance eigenvalues.
    return mu, W, lam

# Step 3 - pca_svd
def pca_svd(X, L):
    """
    Perform PCA using the thin SVD of the centered data matrix.

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
        First L right singular vectors as columns,
        with deterministic signs.

    lam : ndarray of shape (min(N, D),)
        All covariance eigenvalues in decreasing order.

    Z : ndarray of shape (N, L)
        Principal component scores for the first L components.
    """

    # Convert input into a floating-point NumPy array.
    X = np.asarray(X, dtype=float)

    # Get the number of samples and features.
    N, D = X.shape

    # Validate the number of principal components.
    K = min(N, D)

    if not isinstance(L, (int, np.integer)) or not 0 <= L <= K:
        raise ValueError("L must be an integer between 0 and min(N, D).")

    if N == 0 or D == 0:
        raise ValueError("X must contain at least one sample and one feature.")

    # Compute the feature-wise mean.
    mu = np.mean(X, axis=0)

    # Center the data.
    Xc = X - mu

    # Compute the thin SVD:
    # Xc = U @ diag(s) @ Vt
    #
    # U  : (N, K)
    # s  : (K,)
    # Vt : (K, D)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)

    # Compute all covariance eigenvalues.
    # Eigenvalues are already in decreasing order because
    # singular values returned by SVD are sorted.
    lam = (s ** 2) / N

    # Extract the first L right singular vectors as columns.
    W_raw = Vt[:L, :].T

    # Fix the signs of the principal component directions.
    W = fix_signs(W_raw)

    # Determine the sign flips applied to each column of W.
    # These same sign flips must also be applied to the scores.
    flips = np.sign(np.sum(W_raw * W, axis=0))
    flips[flips == 0] = 1.0

    # Compute principal component scores directly from
    # the left singular vectors and singular values.
    # Z = U[:, :L] @ diag(s[:L])
    Z = U[:, :L] * s[:L]

    # Apply the same sign changes used for W.
    Z = Z * flips[None, :]

    # Return the mean, principal components,
    # covariance eigenvalues, and principal component scores.
    return mu, W, lam, Z

# Step 4 - correlation_pca
def correlation_pca(X, L):
    """
    Perform PCA on standardized data using the SVD-based PCA function.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Input data, with one example per row.
    L : int
        Number of principal components to retain.

    Returns
    -------
    W : ndarray of shape (D, L)
        Principal component directions of standardized data.

    lam : ndarray of shape (min(N, D),)
        Covariance eigenvalues of standardized data.

    Z : ndarray of shape (N, L)
        Principal component scores.

    sd : ndarray of shape (D,)
        Population standard deviations used for scaling.
    """

    # Convert input into a floating-point NumPy array.
    X = np.asarray(X, dtype=float)

    # Compute the population standard deviation of each feature.
    # ddof=0 is the default, as required by the instructions.
    sd = np.std(X, axis=0)

    # Avoid division by zero for constant features.
    # Use 1.0 as the divisor for columns with zero variance.
    sd[sd == 0] = 1.0

    # Scale each feature by its standard deviation.
    # pca_svd() will center the scaled data internally.
    X_scaled = X / sd

    # Perform PCA using the existing SVD-based implementation.
    mu, W, lam, Z = pca_svd(X_scaled, L)

    # Return the principal components, eigenvalues,
    # scores, and feature scaling factors.
    return W, lam, Z, sd

