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

# Step 5 - reconstruction_curves
def reconstruction_curves(X_train, X_test, Ls):
    """
    Compute training and test reconstruction errors,
    eigenvalues, and fraction of variance explained
    for different numbers of PCA components.

    Parameters
    ----------
    X_train : ndarray of shape (N_train, D)
        Training dataset.

    X_test : ndarray of shape (N_test, D)
        Test dataset.

    Ls : list or array
        Numbers of principal components to evaluate.

    Returns
    -------
    dict
        {
            "train": training reconstruction errors,
            "test": test reconstruction errors,
            "eigvals": all covariance eigenvalues,
            "fve": fraction of variance explained
        }
    """

    # Convert datasets into floating-point NumPy arrays.
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    # Convert Ls into a list to support different iterable inputs.
    Ls = list(Ls)

    # Validate the component list.
    if len(Ls) == 0:
        raise ValueError("Ls must contain at least one component count.")

    if any(not isinstance(L, (int, np.integer)) for L in Ls):
        raise ValueError("All values in Ls must be integers.")

    if any(L < 0 for L in Ls):
        raise ValueError("Component counts cannot be negative.")

    # Ensure training and test datasets have the same number of features.
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train and X_test must be 2D arrays.")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Training and test datasets must have the same number of features.")

    # Determine the maximum number of components requested.
    max_L = max(Ls)

    # Fit PCA exactly once on the training dataset.
    # The training mean and principal component basis
    # will also be used for reconstructing the test dataset.
    mu, W_max, lam, _ = pca_svd(X_train, max_L)

    # Initialize lists for the reconstruction errors.
    train_errors = []
    test_errors = []
    fve = []

    # Calculate the total variance from all eigenvalues.
    total_variance = np.sum(lam)

    # Evaluate each requested number of principal components.
    for L in Ls:

        # Select the first L principal component directions.
        W = W_max[:, :L]

        # -----------------------------
        # Training reconstruction error
        # -----------------------------

        # Project training data onto the first L components.
        Z_train = (X_train - mu) @ W

        # Reconstruct the training data.
        X_train_reconstructed = Z_train @ W.T + mu

        # Calculate mean squared reconstruction error.
        # Sum squared errors across features, then average
        # across all training observations.
        train_error = np.mean(
            np.sum((X_train - X_train_reconstructed) ** 2, axis=1)
        )

        train_errors.append(train_error)

        # -----------------------------
        # Test reconstruction error
        # -----------------------------

        # Project test data using the training mean and basis.
        Z_test = (X_test - mu) @ W

        # Reconstruct the test data.
        X_test_reconstructed = Z_test @ W.T + mu

        # Calculate mean squared test reconstruction error.
        test_error = np.mean(
            np.sum((X_test - X_test_reconstructed) ** 2, axis=1)
        )

        test_errors.append(test_error)

        # -----------------------------
        # Fraction of variance explained
        # -----------------------------

        # Calculate cumulative fraction of variance explained.
        if total_variance > 0:
            explained_variance = np.sum(lam[:L]) / total_variance
        else:
            explained_variance = 0.0

        fve.append(explained_variance)

    # Return all requested results.
    return {
        "train": np.array(train_errors),
        "test": np.array(test_errors),
        "eigvals": lam,
        "fve": np.array(fve)
    }

# Step 6 - profile_log_likelihood
def profile_log_likelihood(eigvals):
    """
    Compute the profile log-likelihood for every possible
    change point in a sorted sequence of eigenvalues.

    Parameters
    ----------
    eigvals : ndarray of shape (Lmax,)
        Eigenvalues in decreasing order.

    Returns
    -------
    ell : ndarray of shape (Lmax - 1,)
        Profile log-likelihood scores for each possible split.
        Entry i corresponds to a split after eigenvalue i + 1.
    """

    # Convert eigenvalues into a one-dimensional float array.
    eigvals = np.asarray(eigvals, dtype=float).ravel()

    # Number of eigenvalues.
    Lmax = len(eigvals)

    # At least two eigenvalues are required for a valid split.
    if Lmax < 2:
        return np.empty(0, dtype=float)

    # Store the log-likelihood score for each possible split.
    ell = np.empty(Lmax - 1, dtype=float)

    # Evaluate every possible change point.
    for L in range(1, Lmax):

        # Divide eigenvalues into two groups.
        group1 = eigvals[:L]
        group2 = eigvals[L:]

        # Compute the mean of each group.
        mu1 = np.mean(group1)
        mu2 = np.mean(group2)

        # Compute the within-group squared deviations.
        ss1 = np.sum((group1 - mu1) ** 2)
        ss2 = np.sum((group2 - mu2) ** 2)

        # Compute the pooled variance.
        # Divide by Lmax, as specified in the instructions.
        sigma2 = (ss1 + ss2) / Lmax

        # Prevent division by zero for a perfect two-level curve.
        sigma2 = max(sigma2, 1e-12)

        # Compute the Gaussian log-likelihood for both groups.
        # Both groups share the same pooled variance.
        ll1 = -0.5 * (
            len(group1) * np.log(2.0 * np.pi * sigma2)
            + ss1 / sigma2
        )

        ll2 = -0.5 * (
            len(group2) * np.log(2.0 * np.pi * sigma2)
            + ss2 / sigma2
        )

        # Store the total log-likelihood for this split.
        ell[L - 1] = ll1 + ll2

    # Return all profile log-likelihood scores.
    return ell

# Step 7 - ppca_fit
def ppca_fit(X, L):
    """
    Fit Probabilistic PCA using its closed-form maximum-likelihood solution.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Input dataset, with one observation per row.

    L : int
        Number of latent dimensions.

    Returns
    -------
    mu : ndarray of shape (D,)
        Sample mean.

    W : ndarray of shape (D, L)
        PPCA loading matrix.

    sigma2 : float
        Maximum-likelihood isotropic noise variance.
    """

    # Convert input into a floating-point NumPy array.
    X = np.asarray(X, dtype=float)

    # Get the number of features.
    D = X.shape[1]

    # Fit PCA using the existing SVD-based implementation.
    # U_L contains the top L eigenvectors as columns.
    # lam contains the available covariance eigenvalues.
    mu, U_L, lam, _ = pca_svd(X, L)

    # Estimate the isotropic noise variance.
    # When L < D, average all discarded eigenvalues.
    if L < D:
        # Eigenvalues not returned by thin SVD when N < D
        # correspond to exact zeros and need not be explicitly appended.
        sigma2 = float(np.sum(lam[L:]) / (D - L))
    else:
        # No discarded dimensions remain.
        sigma2 = 0.0

    # Calculate the square-root scaling factors.
    # Clip small negative differences caused by numerical precision.
    scales = np.sqrt(np.maximum(lam[:L] - sigma2, 0.0))

    # Construct the PPCA loading matrix:
    # W = U_L @ sqrt(Lambda_L - sigma2 * I)
    W = U_L * scales[np.newaxis, :]

    # Return the sample mean, loading matrix, and noise variance.
    return mu, W, float(sigma2)

# Step 8 - ppca_logpdf
def ppca_logpdf(X, mu, W, sigma2):
    """
    Compute the PPCA log-density for each observation.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Input observations, with one observation per row.

    mu : ndarray of shape (D,)
        Mean vector of the PPCA model.

    W : ndarray of shape (D, L)
        Loading matrix.

    sigma2 : float
        Positive isotropic noise variance.

    Returns
    -------
    logpdf : ndarray of shape (N,)
        Log probability density of each observation.
    """

    # Convert inputs into floating-point NumPy arrays.
    X = np.asarray(X, dtype=float)
    mu = np.asarray(mu, dtype=float)
    W = np.asarray(W, dtype=float)

    # Get the number of observations, features, and latent dimensions.
    N, D = X.shape
    L = W.shape[1]

    # Validate input dimensions.
    if X.ndim != 2 or W.ndim != 2:
        raise ValueError("X and W must be two-dimensional arrays.")

    if mu.shape != (D,) or W.shape[0] != D:
        raise ValueError("Incompatible dimensions between X, mu, and W.")

    # The noise variance must be strictly positive.
    if not np.isscalar(sigma2) or not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError("sigma2 must be a positive finite scalar.")

    sigma2 = float(sigma2)

    # Center the observations using the model mean.
    Xc = X - mu

    # Construct the smaller L x L matrix:
    # M = W.T @ W + sigma2 * I
    M = W.T @ W + sigma2 * np.eye(L)

    # Compute the log determinant of M.
    # slogdet is more numerically stable than calculating det directly.
    sign, logdet_M = np.linalg.slogdet(M)

    if sign <= 0:
        raise np.linalg.LinAlgError("Matrix M must have a positive determinant.")

    # Apply the matrix determinant lemma:
    # log|C| = (D - L) * log(sigma2) + log|M|
    logdet_C = (D - L) * np.log(sigma2) + logdet_M

    # Project centered observations into the latent space.
    # A has shape (N, L).
    A = Xc @ W

    # Calculate the Mahalanobis distance using the
    # Woodbury matrix identity, without constructing C.
    #
    # C^(-1) = (1 / sigma2) *
    #          (I - W @ M^(-1) @ W.T)
    #
    # Solve M @ B = A.T instead of explicitly calculating M^(-1).
    if L > 0:
        B = np.linalg.solve(M, A.T).T

        # Efficient quadratic form for every observation.
        mahalanobis = (
            np.sum(Xc ** 2, axis=1) - np.sum(A * B, axis=1)
        ) / sigma2
    else:
        # Special case: no latent dimensions.
        mahalanobis = np.sum(Xc ** 2, axis=1) / sigma2

    # Calculate the Gaussian log-density.
    logpdf = -0.5 * (
        D * np.log(2.0 * np.pi)
        + logdet_C
        + mahalanobis
    )

    # Return one log-density value per observation.
    return logpdf

# Step 9 - ppca_posterior
def ppca_posterior(X, mu, W, sigma2):
    """
    Compute the posterior distribution of latent variables
    given observations under the PPCA model.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Input observations, with one observation per row.

    mu : ndarray of shape (D,)
        Mean vector of the PPCA model.

    W : ndarray of shape (D, L)
        PPCA loading matrix.

    sigma2 : float
        Positive isotropic noise variance.

    Returns
    -------
    means : ndarray of shape (N, L)
        Posterior mean of the latent variables for each observation.

    cov : ndarray of shape (L, L)
        Shared posterior covariance matrix of the latent variables.
    """

    # Convert inputs into floating-point NumPy arrays.
    X = np.asarray(X, dtype=float)
    mu = np.asarray(mu, dtype=float)
    W = np.asarray(W, dtype=float)

    # Get the number of observations and latent dimensions.
    N, D = X.shape
    L = W.shape[1]

    # Validate input dimensions.
    if X.ndim != 2 or W.ndim != 2:
        raise ValueError("X and W must be two-dimensional arrays.")

    if mu.shape != (D,) or W.shape[0] != D:
        raise ValueError("Incompatible dimensions between X, mu, and W.")

    # Validate the noise variance.
    if not np.isscalar(sigma2) or not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError("sigma2 must be a positive finite scalar.")

    sigma2 = float(sigma2)

    # Center the observations.
    Xc = X - mu

    # Construct the L x L matrix:
    # M = W.T @ W + sigma2 * I
    M = W.T @ W + sigma2 * np.eye(L)

    # Calculate the posterior covariance:
    # cov = sigma2 * M^(-1)
    #
    # Solve a linear system instead of explicitly computing
    # the matrix inverse.
    if L > 0:
        cov = sigma2 * np.linalg.solve(M, np.eye(L))

        # Calculate W.T @ (X - mu) for all observations.
        projected = Xc @ W

        # Calculate the posterior means:
        # means = M^(-1) @ W.T @ (X - mu)
        means = np.linalg.solve(M, projected.T).T

    else:
        # Handle the case of zero latent dimensions.
        means = np.empty((N, 0), dtype=float)
        cov = np.empty((0, 0), dtype=float)

    # Return posterior means and shared posterior covariance.
    return means, cov

# Step 10 - heldout_loglik_curve
def heldout_loglik_curve(X_train, X_test, Ls):
    """
    Evaluate PPCA using training and held-out log-likelihood
    for different numbers of latent dimensions.

    Parameters
    ----------
    X_train : ndarray of shape (N_train, D)
        Training dataset.

    X_test : ndarray of shape (N_test, D)
        Test dataset.

    Ls : list or array
        Numbers of latent dimensions to evaluate.

    Returns
    -------
    dict
        {
            "train": array of mean training log-likelihoods,
            "test": array of mean test log-likelihoods,
            "best_L": optimal number of latent dimensions
        }
    """

    # Convert datasets into floating-point NumPy arrays.
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    # Convert Ls into a list.
    Ls = list(Ls)

    # Ensure that at least one component count is provided.
    if len(Ls) == 0:
        raise ValueError("Ls must contain at least one component count.")

    # Store the average log-likelihoods.
    train_scores = []
    test_scores = []

    # Evaluate PPCA for every requested latent dimension.
    for L in Ls:

        # Fit PPCA using only the training dataset.
        mu, W, sigma2 = ppca_fit(X_train, L)

        # Calculate the log-likelihood of the training observations.
        train_logpdf = ppca_logpdf(
            X_train, mu, W, sigma2
        )

        # Calculate the log-likelihood of the test observations
        # using the same model fitted on the training dataset.
        test_logpdf = ppca_logpdf(
            X_test, mu, W, sigma2
        )

        # Calculate mean log-likelihood per observation.
        train_scores.append(np.mean(train_logpdf))
        test_scores.append(np.mean(test_logpdf))

    # Find the index with the highest test log-likelihood.
    best_index = int(np.argmax(test_scores))

    # Retrieve the corresponding number of latent dimensions.
    best_L = int(Ls[best_index])

    # Return the evaluation results.
    return {
        "train": np.array(train_scores),
        "test": np.array(test_scores),
        "best_L": best_L
    }

# Step 11 - fa_e_step
def fa_e_step(X, W, mu, psi):
    """
    Perform the E-step of the EM algorithm for Factor Analysis.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Input observations, with one observation per row.

    W : ndarray of shape (D, L)
        Factor loading matrix.

    mu : ndarray of shape (D,)
        Mean vector of the observed data.

    psi : ndarray of shape (D,)
        Diagonal entries of the noise covariance matrix.

    Returns
    -------
    m : ndarray of shape (N, L)
        Posterior means of the latent factors.

    Sigma : ndarray of shape (L, L)
        Shared posterior covariance matrix of the latent factors.
    """

    # Convert inputs into floating-point NumPy arrays.
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    mu = np.asarray(mu, dtype=float)
    psi = np.asarray(psi, dtype=float)

    # Validate input dimensions.
    if X.ndim != 2 or W.ndim != 2:
        raise ValueError("X and W must be two-dimensional arrays.")

    N, D = X.shape
    L = W.shape[1]

    if W.shape[0] != D:
        raise ValueError("W must have D rows.")

    if mu.shape != (D,):
        raise ValueError("mu must have shape (D,).")

    if psi.shape != (D,):
        raise ValueError("psi must have shape (D,).")

    # Every feature-specific noise variance must be positive.
    if np.any(psi <= 0) or not np.all(np.isfinite(psi)):
        raise ValueError("All entries of psi must be positive and finite.")

    # Center the observations.
    Xc = X - mu

    # Handle the special case of zero latent dimensions.
    if L == 0:
        m = np.empty((N, 0), dtype=float)
        Sigma = np.empty((0, 0), dtype=float)
        return m, Sigma

    # Compute Psi^(-1) @ W efficiently.
    # Since Psi is diagonal, divide each row of W by
    # its corresponding noise variance.
    weighted_W = W / psi[:, np.newaxis]

    # Compute:
    # M = I + W.T @ Psi^(-1) @ W
    M = np.eye(L) + W.T @ weighted_W

    # Compute the posterior covariance:
    # Sigma = M^(-1)
    #
    # Use solve() instead of explicitly calculating the inverse.
    Sigma = np.linalg.solve(M, np.eye(L))

    # Compute the posterior means:
    # m_i = Sigma @ W.T @ Psi^(-1) @ (x_i - mu)
    #
    # For row-oriented observations, this becomes:
    # m = (X - mu) @ Psi^(-1) @ W @ Sigma
    m = Xc @ weighted_W @ Sigma

    # Return posterior means and shared posterior covariance.
    return m, Sigma

# Step 12 - fa_m_step
def fa_m_step(X, m, Sigma, psi_min=1e-6):
    """
    Perform the M-step of the EM algorithm for Factor Analysis.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Input observations, with one observation per row.

    m : ndarray of shape (N, L)
        Posterior means of the latent factors from the E-step.

    Sigma : ndarray of shape (L, L)
        Shared posterior covariance matrix from the E-step.

    psi_min : float, default=1e-6
        Minimum allowed value for each noise variance.

    Returns
    -------
    W : ndarray of shape (D, L)
        Updated factor loading matrix.

    mu : ndarray of shape (D,)
        Updated mean vector.

    psi : ndarray of shape (D,)
        Updated diagonal noise variances.
    """

    # Convert inputs into floating-point NumPy arrays.
    X = np.asarray(X, dtype=float)
    m = np.asarray(m, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    # Get the number of observations, features, and latent factors.
    N, D = X.shape
    L = m.shape[1]

    # Validate input dimensions.
    if X.ndim != 2 or m.ndim != 2 or Sigma.ndim != 2:
        raise ValueError("X, m, and Sigma must be two-dimensional arrays.")

    if N == 0:
        raise ValueError("X must contain at least one observation.")

    if m.shape[0] != N:
        raise ValueError("m must have the same number of rows as X.")

    if Sigma.shape != (L, L):
        raise ValueError("Sigma must have shape (L, L).")

    if not np.isscalar(psi_min) or not np.isfinite(psi_min) or psi_min <= 0:
        raise ValueError("psi_min must be a positive finite scalar.")

    # Construct b_i = (m_i, 1) for every observation.
    # B has shape (N, L + 1).
    B = np.column_stack((m, np.ones(N)))

    # Calculate the sum of the augmented second-moment matrices:
    #
    # C_i = [[Sigma + m_i m_i.T, m_i],
    #        [m_i.T,             1  ]]
    #
    # The sum of the top-left blocks is:
    # N * Sigma + m.T @ m.

    C_sum = np.zeros((L + 1, L + 1), dtype=float)

    # Top-left block.
    C_sum[:L, :L] = N * Sigma + m.T @ m

    # Top-right block.
    C_sum[:L, L] = np.sum(m, axis=0)

    # Bottom-left block.
    C_sum[L, :L] = np.sum(m, axis=0)

    # Bottom-right block.
    C_sum[L, L] = N

    # Compute the sum of x_i @ b_i.T:
    # Shape = (D, L + 1).
    A = X.T @ B

    # Estimate the augmented loading matrix:
    #
    # W_tilde = [sum(x_i b_i.T)] @ [sum(C_i)]^(-1)
    #
    # Solve the transposed system instead of explicitly
    # computing the matrix inverse.
    W_tilde = np.linalg.solve(C_sum.T, A.T).T

    # Extract the updated factor loading matrix.
    W = W_tilde[:, :L]

    # Extract the updated mean vector (intercept column).
    mu = W_tilde[:, L]

    # Calculate the fitted conditional means:
    # W_tilde @ b_i
    # Shape = (N, D) after transposing the parameter matrix.
    X_fitted = B @ W_tilde.T

    # Estimate the diagonal noise covariance:
    #
    # psi = (1/N) * diag(sum(
    #     (x_i - W_tilde @ b_i) @ x_i.T
    # ))
    #
    # Element-wise multiplication computes the diagonal
    # without constructing a full D x D matrix.
    psi = np.mean((X - X_fitted) * X, axis=0)

    # Apply the minimum variance floor.
    psi = np.maximum(psi, psi_min)

    # Return the updated factor loadings, mean, and noise variances.
    return W, mu, psi

