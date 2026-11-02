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

# Step 13 - fit_fa
def fa_loglik(X, W, mu, psi):
    """
    Compute the mean per-example log-likelihood under
    the Factor Analysis Gaussian model.

    The covariance matrix is:
        C = W @ W.T + diag(psi)

    Uses the Woodbury identity and matrix determinant lemma
    to avoid constructing or inverting a D x D matrix.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Input observations.

    W : ndarray of shape (D, L)
        Factor loading matrix.

    mu : ndarray of shape (D,)
        Mean vector.

    psi : ndarray of shape (D,)
        Diagonal noise variances.

    Returns
    -------
    float
        Mean log-likelihood per observation.
    """

    # Convert inputs into floating-point NumPy arrays.
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    mu = np.asarray(mu, dtype=float)
    psi = np.asarray(psi, dtype=float)

    # Get the number of observations, features, and latent factors.
    N, D = X.shape
    L = W.shape[1]

    # Validate dimensions.
    if W.shape[0] != D or mu.shape != (D,) or psi.shape != (D,):
        raise ValueError("Incompatible dimensions between X, W, mu, and psi.")

    if np.any(psi <= 0) or not np.all(np.isfinite(psi)):
        raise ValueError("All entries of psi must be positive and finite.")

    # Center the observations.
    Xc = X - mu

    # Compute Psi^(-1) @ W efficiently.
    # Psi is diagonal, so divide each row by its corresponding variance.
    weighted_W = W / psi[:, np.newaxis]

    # Construct the smaller L x L matrix:
    # A = I + W.T @ Psi^(-1) @ W
    A = np.eye(L) + W.T @ weighted_W

    # Compute the log determinant of A.
    sign, logdet_A = np.linalg.slogdet(A)

    if sign <= 0:
        raise np.linalg.LinAlgError("Matrix A must have a positive determinant.")

    # Apply the matrix determinant lemma:
    # log|C| = sum(log(psi)) + log|A|
    logdet_C = np.sum(np.log(psi)) + logdet_A

    # First part of the Mahalanobis distance:
    # x_c.T @ Psi^(-1) @ x_c
    term1 = np.sum((Xc ** 2) / psi, axis=1)

    # Project centered observations into the weighted latent space:
    # T = Xc @ Psi^(-1) @ W
    T = Xc @ weighted_W

    if L > 0:
        # Solve A @ B = T.T instead of explicitly computing A^(-1).
        B = np.linalg.solve(A, T.T).T

        # Second part of the Mahalanobis distance:
        # T @ A^(-1) @ T.T
        term2 = np.sum(T * B, axis=1)
    else:
        term2 = np.zeros(N, dtype=float)

    # Complete Mahalanobis distance using the Woodbury identity.
    mahalanobis = term1 - term2

    # Gaussian log-density for each observation.
    logpdf = -0.5 * (
        D * np.log(2.0 * np.pi)
        + logdet_C
        + mahalanobis
    )

    # Return the mean log-likelihood as a Python float.
    return float(np.mean(logpdf))


def fit_fa(X, L, n_iter=100, seed=0):
    """
    Fit a Factor Analysis model using the EM algorithm.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Input observations.

    L : int
        Number of latent factors.

    n_iter : int, default=100
        Number of EM iterations.

    seed : int, default=0
        Random seed for reproducible initialization.

    Returns
    -------
    W : ndarray of shape (D, L)
        Estimated factor loading matrix.

    mu : ndarray of shape (D,)
        Estimated mean vector.

    psi : ndarray of shape (D,)
        Estimated diagonal noise variances.

    trace : ndarray of shape (n_iter,)
        Mean log-likelihood after each EM iteration.
    """

    # Convert input into a floating-point NumPy array.
    X = np.asarray(X, dtype=float)

    # Get the number of observations and features.
    N, D = X.shape

    # Validate the number of latent factors and iterations.
    if not isinstance(L, (int, np.integer)) or not 0 <= L <= D:
        raise ValueError("L must be an integer between 0 and D.")

    if not isinstance(n_iter, (int, np.integer)) or n_iter < 0:
        raise ValueError("n_iter must be a non-negative integer.")

    if N == 0:
        raise ValueError("X must contain at least one observation.")

    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    # Initialize the mean vector using the sample mean.
    mu = np.mean(X, axis=0)

    # Initialize feature-specific noise variances.
    # Apply a minimum variance floor of 1e-6.
    psi = np.maximum(np.var(X, axis=0), 1e-6)

    # Initialize the factor loading matrix with random values.
    W = (
        rng.standard_normal((D, L))
        * np.sqrt(psi.mean())
    )

    # Allocate the likelihood convergence trace.
    trace = np.empty(n_iter, dtype=float)

    # Run the EM algorithm.
    for iteration in range(n_iter):

        # --------------------------------
        # E-step: Estimate latent factors
        # --------------------------------
        m, Sigma = fa_e_step(X, W, mu, psi)

        # --------------------------------
        # M-step: Update model parameters
        # --------------------------------
        W, mu, psi = fa_m_step(
            X,
            m,
            Sigma,
            psi_min=1e-6
        )

        # --------------------------------
        # Evaluate the updated model
        # --------------------------------
        trace[iteration] = fa_loglik(X, W, mu, psi)

    # Return fitted parameters and likelihood history.
    return W, mu, psi, trace

# Step 14 - varimax
def varimax(W, n_iter=100, tol=1e-8):
    """
    Perform orthogonal Varimax rotation of a factor loading matrix.

    Parameters
    ----------
    W : ndarray of shape (D, L)
        Original factor loading matrix.

    n_iter : int, default=100
        Maximum number of SVD iterations.

    tol : float, default=1e-8
        Relative tolerance for convergence.

    Returns
    -------
    W_rot : ndarray of shape (D, L)
        Rotated factor loading matrix.

    R : ndarray of shape (L, L)
        Orthogonal rotation matrix.
    """

    # Convert the loading matrix into a floating-point NumPy array.
    W = np.asarray(W, dtype=float)

    # Validate the input.
    if W.ndim != 2:
        raise ValueError("W must be a two-dimensional array.")

    if not isinstance(n_iter, (int, np.integer)) or n_iter < 0:
        raise ValueError("n_iter must be a non-negative integer.")

    if not np.isscalar(tol) or not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be a non-negative finite scalar.")

    # Get the number of features and factors.
    D, L = W.shape

    # Initialize the orthogonal rotation matrix.
    R = np.eye(L)

    # Handle an empty factor dimension.
    if L == 0:
        return W.copy(), R

    # Initialize the previous objective value.
    obj = 0.0

    # Perform the iterative Varimax optimization.
    for _ in range(n_iter):

        # Apply the current orthogonal rotation.
        A = W @ R

        # Calculate the matrix used in the SVD update.
        #
        # G = W.T @ (
        #     A**3 - A * sum(A**2, axis=0) / D
        # )
        G = W.T @ (
            A**3
            - A * np.sum(A**2, axis=0) / D
        )

        # Compute the singular value decomposition of G.
        U, s, Vt = np.linalg.svd(G)

        # Update the orthogonal rotation matrix.
        # This maximizes the Varimax objective at each iteration.
        R_new = U @ Vt

        # Calculate the new objective value.
        new_obj = float(np.sum(s))

        # Check for convergence.
        if new_obj < obj * (1.0 + tol):
            break

        # Accept the updated rotation and objective.
        R = R_new
        obj = new_obj

    # Apply the final orthogonal rotation to the loadings.
    W_rot = W @ R

    # Return rotated loadings and the rotation matrix.
    return W_rot, R

# Step 15 - sensor_noise_experiment
def principal_angle(A, B):
    """
    Calculate the largest principal angle between
    the column spaces of two matrices.

    Parameters
    ----------
    A : ndarray
        First basis matrix.

    B : ndarray
        Second basis matrix.

    Returns
    -------
    float
        Largest principal angle in degrees.
    """

    # Obtain orthonormal bases using QR decomposition.
    Qa, _ = np.linalg.qr(np.asarray(A, dtype=float))
    Qb, _ = np.linalg.qr(np.asarray(B, dtype=float))

    # Compute singular values of the cross-product.
    c = np.linalg.svd(Qa.T @ Qb, compute_uv=False)

    # The smallest singular value gives the largest
    # principal angle.
    return float(
        np.degrees(
            np.arccos(np.clip(c.min(), -1.0, 1.0))
        )
    )


def sensor_noise_experiment(n=2000, noise_var=25.0, seed=0):
    """
    Compare PPCA and Factor Analysis on data containing
    one particularly noisy sensor.

    Parameters
    ----------
    n : int
        Number of training and test observations.

    noise_var : float
        Noise variance of the faulty sensor (dimension 0).

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Experiment results containing six Python floats.
    """

    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    # Define the number of observed dimensions and latent factors.
    D = 10
    L = 2

    # Generate the true factor loading matrix.
    W_true = rng.standard_normal((D, L))

    # Define the true feature-specific noise variances.
    psi_true = np.full(D, 0.1)

    # Assign higher noise variance to sensor 0.
    psi_true[0] = noise_var

    # Generate the training dataset.
    X_train = (
        rng.standard_normal((n, L)) @ W_true.T
        + rng.standard_normal((n, D)) * np.sqrt(psi_true)
    )

    # Generate the test dataset using the same true model.
    X_test = (
        rng.standard_normal((n, L)) @ W_true.T
        + rng.standard_normal((n, D)) * np.sqrt(psi_true)
    )

    # Fit Probabilistic PCA on the training dataset.
    mu_ppca, W_ppca, sigma2 = ppca_fit(X_train, L)

    # Fit Factor Analysis using 200 EM iterations.
    W_fa, mu_fa, psi_fa, trace = fit_fa(
        X_train,
        L,
        n_iter=200,
        seed=seed
    )

    # Calculate the largest principal angle for PPCA.
    ppca_angle = principal_angle(W_ppca, W_true)

    # Calculate the largest principal angle for Factor Analysis.
    fa_angle = principal_angle(W_fa, W_true)

    # Evaluate PPCA mean log-likelihood on the test dataset.
    ppca_test = float(
        np.mean(
            ppca_logpdf(
                X_test,
                mu_ppca,
                W_ppca,
                sigma2
            )
        )
    )

    # Evaluate Factor Analysis mean log-likelihood on the test dataset.
    fa_test = float(
        fa_loglik(
            X_test,
            W_fa,
            mu_fa,
            psi_fa
        )
    )

    # Return the experiment results.
    return {
        "ppca_angle": float(ppca_angle),
        "fa_angle": float(fa_angle),
        "ppca_test": float(ppca_test),
        "fa_test": float(fa_test),
        "psi_noisy": float(psi_fa[0]),
        "sigma2": float(sigma2)
    }

# Step 16 - em_pca
def em_pca(X, L, n_iter=50, seed=0):
    """
    Perform PCA using the EM algorithm in the zero-noise limit.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Input data, with one example per row.

    L : int
        Number of principal components.

    n_iter : int
        Number of EM iterations.

    seed : int
        Random seed for initialization.

    Returns
    -------
    mu : ndarray of shape (D,)
        Sample mean.

    W : ndarray of shape (D, L)
        Orthonormal principal components in decreasing
        order of eigenvalues.

    lam : ndarray of shape (L,)
        Principal component eigenvalues in decreasing order.

    errs : ndarray of shape (n_iter,)
        Reconstruction error after each EM iteration.
    """

    # Convert input to a floating-point NumPy array.
    X = np.asarray(X, dtype=float)

    # Get the number of samples and features.
    N, D = X.shape

    # Compute the sample mean and center the data.
    mu = np.mean(X, axis=0)
    Xc = X - mu

    # Transpose so that observations are stored in columns.
    Xt = Xc.T

    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    # Initialize the loading matrix.
    W = rng.standard_normal((D, L))

    # Store reconstruction errors after each iteration.
    errs = []

    # Run the EM algorithm.
    for _ in range(n_iter):

        # -----------------------------
        # E-step
        # -----------------------------
        # Z = (W.T @ W)^(-1) @ W.T @ Xt
        #
        # Solve the linear system instead of explicitly
        # calculating the matrix inverse.
        Z = np.linalg.solve(
            W.T @ W,
            W.T @ Xt
        )

        # -----------------------------
        # M-step
        # -----------------------------
        # W = Xt @ Z.T @ (Z @ Z.T)^(-1)
        #
        # Solve the transposed linear system for stability.
        ZZT = Z @ Z.T

        W = np.linalg.solve(
            ZZT.T,
            (Xt @ Z.T).T
        ).T

        # Calculate reconstruction error after the M-step.
        reconstruction = W @ Z

        error = np.sum((Xt - reconstruction) ** 2) / N

        errs.append(float(error))

    # -----------------------------
    # Final orthonormalization
    # -----------------------------

    # Obtain an orthonormal basis for the learned subspace.
    Q, _ = np.linalg.qr(W)

    # Project the centered data onto the orthonormal basis.
    Zq = Q.T @ Xt

    # Compute the L x L covariance matrix in the projected space.
    Cq = Zq @ Zq.T / N

    # Eigendecompose the projected covariance matrix.
    eigenvalues, eigenvectors = np.linalg.eigh(Cq)

    # Sort eigenvalues in decreasing order.
    order = np.argsort(eigenvalues)[::-1]

    lam = eigenvalues[order]

    # Rotate the orthonormal basis using the sorted eigenvectors.
    W = Q @ eigenvectors[:, order]

    # Fix eigenvector signs for deterministic results.
    W = fix_signs(W)

    # Clip tiny negative eigenvalues caused by floating-point errors.
    lam = np.maximum(lam, 0.0)

    # Return the mean, ordered orthonormal basis,
    # eigenvalues, and reconstruction error history.
    return mu, W, lam, np.asarray(errs)

# Step 17 - mppca_e_step
def mppca_e_step(X, pi, mus, Ws, sigma2s):
    """
    Perform the E-step for a mixture of Probabilistic PCA models.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Input observations.

    pi : ndarray of shape (K,)
        Mixture weights.

    mus : ndarray of shape (K, D)
        Mean vectors of the mixture components.

    Ws : ndarray of shape (K, D, L)
        Loading matrices of the mixture components.

    sigma2s : ndarray of shape (K,)
        Isotropic noise variances of the mixture components.

    Returns
    -------
    resp : ndarray of shape (N, K)
        Responsibilities, where each row sums to one.

    ll : float
        Mean log-likelihood per observation.
    """

    # Convert inputs into floating-point NumPy arrays.
    X = np.asarray(X, dtype=float)
    pi = np.asarray(pi, dtype=float)
    mus = np.asarray(mus, dtype=float)
    Ws = np.asarray(Ws, dtype=float)
    sigma2s = np.asarray(sigma2s, dtype=float)

    # Get the number of observations and features.
    N, D = X.shape

    # Get the number of mixture components.
    K = len(pi)

    # Validate input dimensions.
    if pi.ndim != 1:
        raise ValueError("pi must be a one-dimensional array.")

    if mus.shape != (K, D):
        raise ValueError("mus must have shape (K, D).")

    if Ws.ndim != 3 or Ws.shape[:2] != (K, D):
        raise ValueError("Ws must have shape (K, D, L).")

    if sigma2s.shape != (K,):
        raise ValueError("sigma2s must have shape (K,).")

    if np.any(pi < 0) or not np.all(np.isfinite(pi)):
        raise ValueError("Mixture weights must be non-negative and finite.")

    if np.sum(pi) <= 0:
        raise ValueError("At least one mixture weight must be positive.")

    if np.any(sigma2s <= 0) or not np.all(np.isfinite(sigma2s)):
        raise ValueError("All noise variances must be positive and finite.")

    # Initialize the matrix of log joint probabilities.
    # Each entry represents:
    # log pi_k + log N(x_n | mu_k, W_k W_k.T + sigma2_k I).
    log_joint = np.empty((N, K), dtype=float)

    # Calculate the log joint probability for each component.
    for k in range(K):

        # Handle zero mixture weights without taking log(0).
        if pi[k] == 0:
            log_joint[:, k] = -np.inf
            continue

        # Compute the component's log-density using PPCA.
        component_logpdf = ppca_logpdf(
            X,
            mus[k],
            Ws[k],
            sigma2s[k]
        )

        # Add the logarithm of the mixture weight.
        log_joint[:, k] = np.log(pi[k]) + component_logpdf

    # -----------------------------------------
    # Stable log-sum-exp normalization
    # -----------------------------------------

    # Find the maximum log joint probability for each observation.
    # Keeping dimensions allows broadcasting.
    max_log = np.max(log_joint, axis=1, keepdims=True)

    # Compute the log normalization constant.
    log_norm = max_log + np.log(
        np.sum(
            np.exp(log_joint - max_log),
            axis=1,
            keepdims=True
        )
    )

    # Calculate normalized responsibilities.
    # Every row of resp sums to one.
    resp = np.exp(log_joint - log_norm)

    # Calculate the mean log-likelihood per observation.
    ll = float(np.mean(log_norm))

    # Return responsibilities and mean log-likelihood.
    return resp, ll

# Step 18 - fit_mppca
def curved_data(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, n)
    y = np.sin(1.5 * x) + 0.05 * rng.standard_normal(n)
    return np.column_stack([x + 0.05 * rng.standard_normal(n), y])


def fit_mppca(X, K, L, n_iter=100, seed=0):
    """
    Fit a Mixture of Probabilistic PCA models using EM.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Input observations, with one example per row.

    K : int
        Number of mixture components.

    L : int
        Number of latent dimensions per component.

    n_iter : int, default=100
        Number of EM iterations.

    seed : int, default=0
        Random seed for reproducible initialization.

    Returns
    -------
    pi : ndarray of shape (K,)
        Mixture weights.

    mus : ndarray of shape (K, D)
        Component mean vectors.

    Ws : ndarray of shape (K, D, L)
        Component loading matrices.

    sigma2s : ndarray of shape (K,)
        Component noise variances.

    trace : ndarray of shape (n_iter,)
        Mean log-likelihood after each EM iteration.
    """

    # Convert input into a floating-point NumPy array.
    X = np.asarray(X, dtype=float)

    # Get the number of observations and features.
    N, D = X.shape

    # Validate input parameters.
    if not isinstance(K, (int, np.integer)) or not 1 <= K <= N:
        raise ValueError("K must be an integer between 1 and N.")

    if not isinstance(L, (int, np.integer)) or not 0 <= L < D:
        raise ValueError("L must be an integer between 0 and D - 1.")

    if not isinstance(n_iter, (int, np.integer)) or n_iter < 1:
        raise ValueError("n_iter must be a positive integer.")

    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # 1. K-means initialization
    # --------------------------------------------------

    # Select K distinct observations as initial cluster centers.
    centers = X[rng.choice(N, K, replace=False)].copy()

    # Perform exactly five rounds of K-means.
    for _ in range(5):

        # Compute squared Euclidean distances from every
        # observation to every cluster center.
        distances = np.sum(
            (X[:, None, :] - centers[None, :, :]) ** 2,
            axis=2
        )

        # Assign each observation to its nearest center.
        labels = np.argmin(distances, axis=1)

        # Update each cluster center.
        for k in range(K):

            # Find observations assigned to cluster k.
            cluster_points = X[labels == k]

            # Replace the center with the cluster mean.
            # Keep the old center if the cluster is empty.
            if len(cluster_points) > 0:
                centers[k] = np.mean(cluster_points, axis=0)

    # Assign observations once more using the final centers.
    distances = np.sum(
        (X[:, None, :] - centers[None, :, :]) ** 2,
        axis=2
    )

    labels = np.argmin(distances, axis=1)

    # Construct one-hot initial responsibilities.
    resp = np.zeros((N, K), dtype=float)
    resp[np.arange(N), labels] = 1.0

    # --------------------------------------------------
    # 2. Allocate model parameters
    # --------------------------------------------------

    pi = np.empty(K, dtype=float)
    mus = np.empty((K, D), dtype=float)
    Ws = np.empty((K, D, L), dtype=float)
    sigma2s = np.empty(K, dtype=float)

    # Store the mean log-likelihood at each iteration.
    trace = np.empty(n_iter, dtype=float)

    # --------------------------------------------------
    # 3. EM algorithm
    # --------------------------------------------------

    for iteration in range(n_iter):

        # ==============================================
        # M-step: Update mixture and PPCA parameters
        # ==============================================

        # Calculate effective component sample counts.
        # The floor protects against empty components.
        Nk = np.maximum(
            np.sum(resp, axis=0),
            1e-10
        )

        # Update mixture weights.
        pi = Nk / N

        # Calculate responsibility-weighted component means.
        mus = (resp.T @ X) / Nk[:, None]

        # Fit a separate PPCA model to each component.
        for k in range(K):

            # Center observations around component k's mean.
            Xc = X - mus[k]

            # Calculate the weighted empirical covariance.
            # S_k = sum_n r_nk (x_n-mu_k)(x_n-mu_k).T / Nk
            S = (
                Xc.T @ (resp[:, k, None] * Xc)
            ) / Nk[k]

            # Ensure numerical symmetry.
            S = 0.5 * (S + S.T)

            # Eigendecompose the D x D covariance matrix.
            eigenvalues, eigenvectors = np.linalg.eigh(S)

            # Sort eigenvalues and eigenvectors in decreasing order.
            order = np.argsort(eigenvalues)[::-1]

            eigenvalues = np.maximum(eigenvalues[order], 0.0)
            eigenvectors = eigenvectors[:, order]

            # Estimate the isotropic noise variance from
            # the discarded eigenvalues.
            sigma2s[k] = max(
                float(np.mean(eigenvalues[L:])),
                1e-6
            )

            # Calculate the loading matrix:
            # W_k = U_L @ sqrt(Lambda_L - sigma2 * I)
            scales = np.sqrt(
                np.maximum(
                    eigenvalues[:L] - sigma2s[k],
                    0.0
                )
            )

            Ws[k] = eigenvectors[:, :L] * scales[None, :]

        # ==============================================
        # E-step: Update responsibilities
        # ==============================================

        resp, ll = mppca_e_step(
            X,
            pi,
            mus,
            Ws,
            sigma2s
        )

        # Record the mean log-likelihood.
        trace[iteration] = ll

    # Return the fitted mixture model and likelihood trace.
    return pi, mus, Ws, sigma2s, trace

# Step 19 - swiss_roll
def swiss_roll(n=1000, noise=0.0, seed=0):
    """
    Generate a Swiss roll dataset and its true manifold coordinates.

    Parameters
    ----------
    n : int, default=1000
        Number of samples.

    noise : float, default=0.0
        Standard deviation of Gaussian noise added to the data.

    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n, 3)
        Three-dimensional Swiss roll observations.

    U : ndarray of shape (n, 2)
        True manifold coordinates containing arc length s
        and height h.
    """

    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    # Generate the spiral parameter t.
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.random(n))

    # Generate the height coordinate.
    h = 21.0 * rng.random(n)

    # Construct the three-dimensional Swiss roll.
    X = np.column_stack((
        t * np.cos(t),
        h,
        t * np.sin(t)
    ))

    # Add Gaussian noise only when noise is positive.
    if noise > 0:
        X = X + noise * rng.standard_normal((n, 3))

    # Calculate the true arc length along the spiral.
    s = 0.5 * (
        t * np.sqrt(1.0 + t**2)
        + np.arcsinh(t)
    )

    # Construct the true two-dimensional manifold coordinates.
    # First column: arc length along the spiral.
    # Second column: height.
    U = np.column_stack((s, h))

    return X, U


def unroll_score(Z, U):
    """
    Evaluate how well a low-dimensional embedding recovers
    the true manifold coordinates using Spearman correlation.

    For each column of U, calculate its Spearman rank
    correlation with every column of Z and retain the
    largest absolute correlation.

    Parameters
    ----------
    Z : ndarray of shape (N, d)
        Low-dimensional embedding.

    U : ndarray of shape (N, q)
        True manifold coordinates.

    Returns
    -------
    tuple of floats
        Maximum absolute Spearman correlations,
        one for each column of U.
    """

    # Convert inputs into floating-point NumPy arrays.
    Z = np.asarray(Z, dtype=float)
    U = np.asarray(U, dtype=float)

    # Ensure both inputs are two-dimensional.
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    if U.ndim == 1:
        U = U.reshape(-1, 1)

    # Validate the number of observations.
    if Z.shape[0] != U.shape[0]:
        raise ValueError("Z and U must have the same number of observations.")

    if Z.shape[0] < 2:
        raise ValueError("At least two observations are required.")

    # Compute the ranks of every embedding column.
    # The data contain no ties, so double argsort gives
    # the required ordinal ranks.
    rank_Z = np.argsort(
        np.argsort(Z, axis=0),
        axis=0
    ).astype(float)

    rank_U = np.argsort(
        np.argsort(U, axis=0),
        axis=0
    ).astype(float)

    # Center the ranks.
    rank_Z -= np.mean(rank_Z, axis=0, keepdims=True)
    rank_U -= np.mean(rank_U, axis=0, keepdims=True)

    # Compute the squared norms of the centered ranks.
    norm_Z = np.sum(rank_Z**2, axis=0)
    norm_U = np.sum(rank_U**2, axis=0)

    # Calculate Spearman correlations between every pair
    # of columns of Z and U.
    scores = []

    for j in range(U.shape[1]):

        # Calculate correlations with every embedding column.
        denominator = np.sqrt(norm_U[j] * norm_Z)

        # Avoid division by zero for constant embedding columns.
        correlations = np.divide(
            rank_U[:, j] @ rank_Z,
            denominator,
            out=np.zeros(Z.shape[1], dtype=float),
            where=denominator > 0
        )

        # Keep the largest absolute Spearman correlation.
        best_score = np.max(np.abs(correlations))

        scores.append(float(best_score))

    # Return one Python float for each true manifold coordinate.
    return tuple(scores)

# Step 20 - classical_mds
def pairwise_sqdist(A, B=None):
    A = np.asarray(A, dtype=float)
    B = A if B is None else np.asarray(B, dtype=float)
    d2 = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2.0 * A @ B.T
    return np.clip(d2, 0.0, None)


def classical_mds(D2, L):
    """
    Perform Classical Multidimensional Scaling (MDS)
    using a matrix of squared pairwise distances.

    Parameters
    ----------
    D2 : ndarray of shape (N, N)
        Symmetric matrix of squared dissimilarities.

    L : int
        Number of embedding dimensions.

    Returns
    -------
    Z : ndarray of shape (N, L)
        Low-dimensional embedding.

    lam : ndarray of shape (N,)
        All eigenvalues in decreasing order.
        Negative eigenvalues are not clipped.
    """

    # Convert the squared distance matrix into a float array.
    D2 = np.asarray(D2, dtype=float)

    # Get the number of observations.
    N = D2.shape[0]

    # Validate the input dimensions.
    if D2.ndim != 2 or D2.shape[0] != D2.shape[1]:
        raise ValueError("D2 must be a square matrix.")

    if not isinstance(L, (int, np.integer)) or not 0 <= L <= N:
        raise ValueError("L must be an integer between 0 and N.")

    # Construct the centering matrix:
    # H = I - (1/N) * 11.T
    H = np.eye(N) - np.ones((N, N)) / N

    # Double-center the squared distance matrix.
    # K = -0.5 * H @ D2 @ H
    K = -0.5 * H @ D2 @ H

    # Ensure numerical symmetry before eigendecomposition.
    K = 0.5 * (K + K.T)

    # Eigendecompose the centered matrix.
    # eigh is appropriate because K is symmetric.
    eigenvalues, eigenvectors = np.linalg.eigh(K)

    # Sort eigenvalues and corresponding eigenvectors
    # in decreasing order.
    order = np.argsort(eigenvalues)[::-1]

    lam = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Construct the embedding using the first L eigenvectors.
    # Clip negative eigenvalues only for the square root.
    # The returned lam remains unmodified.
    scales = np.sqrt(np.maximum(lam[:L], 0.0))

    Z = eigenvectors[:, :L] * scales[None, :]

    # Fix the signs of the embedding coordinates.
    Z = fix_signs(Z)

    # Return the embedding and all original eigenvalues.
    return Z, lam

# Step 21 - trustworthiness
def pairwise_sqdist(A, B=None):
    A = np.asarray(A, dtype=float)
    B = A if B is None else np.asarray(B, dtype=float)

    d2 = (
        (A ** 2).sum(axis=1)[:, None]
        + (B ** 2).sum(axis=1)[None, :]
        - 2.0 * A @ B.T
    )

    return np.clip(d2, 0.0, None)


def knn_indices(D2, k):
    """
    Return the indices of the k nearest neighbours
    of every point, excluding the point itself.

    Parameters
    ----------
    D2 : ndarray of shape (N, N)
        Squared pairwise distance matrix.

    k : int
        Number of neighbours.

    Returns
    -------
    ndarray of shape (N, k)
        Indices of nearest neighbours.
    """

    D2 = np.array(D2, dtype=float)
    np.fill_diagonal(D2, np.inf)

    return np.argsort(D2, axis=1, kind="stable")[:, :k]


def trustworthiness(D2, Z, k=5):
    """
    Calculate the trustworthiness of a low-dimensional embedding.

    Parameters
    ----------
    D2 : ndarray of shape (N, N)
        Squared distances in the original space.

    Z : ndarray of shape (N, L)
        Low-dimensional embedding.

    k : int, default=5
        Number of nearest neighbours.

    Returns
    -------
    float
        Trustworthiness score of the embedding.
    """

    # Convert inputs into floating-point NumPy arrays.
    D2 = np.asarray(D2, dtype=float)
    Z = np.asarray(Z, dtype=float)

    # Get the number of observations.
    N = D2.shape[0]

    # Validate input dimensions.
    if D2.ndim != 2 or D2.shape != (N, N):
        raise ValueError("D2 must be a square distance matrix.")

    if Z.ndim != 2 or Z.shape[0] != N:
        raise ValueError("Z must have the same number of observations as D2.")

    if not isinstance(k, (int, np.integer)) or not 1 <= k < N / 2:
        raise ValueError("k must be an integer satisfying 1 <= k < N/2.")

    # -----------------------------------------
    # 1. Calculate original-space neighbour ranks
    # -----------------------------------------

    # Stable sorting of original-space distances.
    # The diagonal is zero, so each point receives rank 0
    # and its nearest other point receives rank 1.
    sorted_indices = np.argsort(D2, axis=1, kind="stable")

    # Convert sorted indices into rank positions.
    ranks = np.empty_like(sorted_indices)

    ranks[
        np.arange(N)[:, None],
        sorted_indices
    ] = np.arange(N)[None, :]

    # -----------------------------------------
    # 2. Find neighbours in the embedding space
    # -----------------------------------------

    # Calculate squared pairwise distances in the embedding.
    D2_embedding = pairwise_sqdist(Z)

    # Obtain the k nearest neighbours of each embedded point.
    embedding_neighbors = knn_indices(D2_embedding, k)

    # -----------------------------------------
    # 3. Calculate the trustworthiness penalty
    # -----------------------------------------

    # Retrieve the original-space ranks of embedding neighbours.
    neighbor_ranks = ranks[
        np.arange(N)[:, None],
        embedding_neighbors
    ]

    # Only penalize neighbours that were not among
    # the original k nearest neighbours.
    penalties = np.maximum(neighbor_ranks - k, 0)

    # Sum all neighbourhood violations.
    total_penalty = np.sum(penalties)

    # -----------------------------------------
    # 4. Normalize the trustworthiness score
    # -----------------------------------------

    denominator = N * k * (2 * N - 3 * k - 1)

    score = 1.0 - (
        2.0 * total_penalty / denominator
    )

    # Return a Python float.
    return float(score)


def knn_accuracy(Z, y, k=5):
    """
    Calculate leave-one-out k-nearest-neighbour classification
    accuracy using a low-dimensional embedding.

    Parameters
    ----------
    Z : ndarray of shape (N, L)
        Low-dimensional embedding.

    y : ndarray of shape (N,)
        Integer class labels.

    k : int, default=5
        Number of neighbours used for classification.

    Returns
    -------
    float
        Leave-one-out classification accuracy.
    """

    # Convert inputs into NumPy arrays.
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y)

    # Ensure Z is two-dimensional.
    if Z.ndim != 2:
        raise ValueError("Z must be a two-dimensional array.")

    # Validate the labels.
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")

    N = Z.shape[0]

    if len(y) != N:
        raise ValueError("Z and y must contain the same number of observations.")

    if not isinstance(k, (int, np.integer)) or not 1 <= k < N:
        raise ValueError("k must be an integer satisfying 1 <= k < N.")

    # -----------------------------------------
    # 1. Calculate embedding-space distances
    # -----------------------------------------

    D2 = pairwise_sqdist(Z)

    # Obtain k nearest neighbours, excluding each point itself.
    neighbors = knn_indices(D2, k)

    # Store the number of correct predictions.
    correct = 0

    # -----------------------------------------
    # 2. Leave-one-out majority-vote classification
    # -----------------------------------------

    for i in range(N):

        # Get the labels of the k nearest neighbours.
        neighbor_labels = y[neighbors[i]]

        # Count the occurrences of each label.
        labels, counts = np.unique(
            neighbor_labels,
            return_counts=True
        )

        # Find the largest vote count.
        max_count = np.max(counts)

        # Resolve ties by choosing the smallest label.
        predicted_label = np.min(
            labels[counts == max_count]
        )

        # Check whether the predicted label is correct.
        if predicted_label == y[i]:
            correct += 1

    # -----------------------------------------
    # 3. Calculate classification accuracy
    # -----------------------------------------

    accuracy = correct / N

    return float(accuracy)

# Step 22 - knn_geodesics
def knn_geodesics(D2, k):
    """
    Compute approximate geodesic distances using a symmetric
    k-nearest-neighbour graph and all-pairs shortest paths.

    Parameters
    ----------
    D2 : ndarray of shape (N, N)
        Squared Euclidean distance matrix.

    k : int
        Number of nearest neighbours for graph construction.

    Returns
    -------
    G : ndarray of shape (N, N)
        Geodesic distance matrix.

        - Zero on the diagonal.
        - Shortest-path distance for connected points.
        - np.inf for disconnected points.
    """

    # Convert the squared distance matrix into a float array.
    D2 = np.asarray(D2, dtype=float)

    # Get the number of observations.
    N = D2.shape[0]

    # Validate the distance matrix.
    if D2.ndim != 2 or D2.shape != (N, N):
        raise ValueError("D2 must be a square matrix.")

    if not isinstance(k, (int, np.integer)) or not 1 <= k < N:
        raise ValueError("k must be an integer between 1 and N - 1.")

    # -------------------------------------------------
    # 1. Construct the symmetric k-nearest-neighbour graph
    # -------------------------------------------------

    # Find the k nearest neighbours of every observation.
    neighbors = knn_indices(D2, k)

    # Initialize the graph with infinite distances.
    # Infinite distance means that no direct edge exists.
    G = np.full((N, N), np.inf, dtype=float)

    # Set the diagonal to zero.
    np.fill_diagonal(G, 0.0)

    # Generate the row indices corresponding to neighbours.
    rows = np.repeat(np.arange(N), k)

    # Flatten the neighbour indices.
    cols = neighbors.ravel()

    # Convert squared Euclidean distances into Euclidean distances.
    edge_lengths = np.sqrt(
        np.maximum(D2[rows, cols], 0.0)
    )

    # Add directed nearest-neighbour edges.
    G[rows, cols] = edge_lengths

    # Make the graph symmetric.
    # An undirected edge exists if either endpoint selects the other.
    G[cols, rows] = edge_lengths

    # -------------------------------------------------
    # 2. Calculate all-pairs shortest paths
    # -------------------------------------------------

    # Use the vectorized Floyd-Warshall algorithm.
    #
    # For every intermediate point m:
    # G[i, j] = min(G[i, j], G[i, m] + G[m, j])
    #
    # This naturally preserves np.inf between disconnected
    # components.
    for m in range(N):
        G = np.minimum(
            G,
            G[:, m:m+1] + G[m:m+1, :]
        )

    # Ensure the diagonal remains exactly zero.
    np.fill_diagonal(G, 0.0)

    return G

# Step 23 - isomap
def isomap(X, k=10, L=2):
    """
    Perform Isomap dimensionality reduction.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Input observations.

    k : int, default=10
        Number of nearest neighbours used to construct
        the neighbourhood graph.

    L : int, default=2
        Number of embedding dimensions.

    Returns
    -------
    Z : ndarray of shape (len(keep), L)
        Low-dimensional Isomap embedding.

    keep : ndarray of shape (len(keep),)
        Sorted indices of observations belonging to
        the largest connected component.

    lam : ndarray
        Eigenvalues returned by Classical MDS.
    """

    # Convert input into a floating-point NumPy array.
    X = np.asarray(X, dtype=float)

    # Get the number of observations.
    N = X.shape[0]

    # Compute squared Euclidean distances.
    D2 = pairwise_sqdist(X)

    # Compute approximate geodesic distances using
    # the symmetric k-nearest-neighbour graph.
    G = knn_geodesics(D2, k)

    # -------------------------------------------------
    # Identify connected components
    # -------------------------------------------------

    # For each row, find the first finite entry.
    # All points belonging to the same connected component
    # have the same first finite column index.
    component_labels = np.argmax(
        np.isfinite(G),
        axis=1
    )

    # Count the number of observations in each component.
    unique_labels, counts = np.unique(
        component_labels,
        return_counts=True
    )

    # Select the largest connected component.
    # np.argmax selects the first component in case of a tie.
    largest_component = unique_labels[np.argmax(counts)]

    # Obtain the sorted original indices of the selected points.
    keep = np.flatnonzero(
        component_labels == largest_component
    )

    # -------------------------------------------------
    # Apply Classical MDS to the geodesic distances
    # -------------------------------------------------

    # Extract the geodesic distances within the largest
    # connected component.
    G_kept = G[np.ix_(keep, keep)]

    # Classical MDS expects squared distances.
    D2_geodesic = G_kept ** 2

    # Compute the low-dimensional Isomap embedding.
    Z, lam = classical_mds(D2_geodesic, L)

    # Return embedding, retained indices, and eigenvalues.
    return Z, keep, lam


def short_circuit_count(D2, U, k, gap=15.0):
    """
    Count directed nearest-neighbour edges that connect
    points far apart in their true manifold coordinates.

    Parameters
    ----------
    D2 : ndarray of shape (N, N)
        Squared Euclidean distances in the observed space.

    U : ndarray of shape (N, q)
        True manifold coordinates.

    k : int
        Number of nearest neighbours.

    gap : float, default=15.0
        Minimum manifold distance for an edge to be
        considered a short-circuit edge.

    Returns
    -------
    int
        Number of directed nearest-neighbour edges whose
        endpoints are more than gap apart in U.
    """

    # Convert inputs into floating-point NumPy arrays.
    D2 = np.asarray(D2, dtype=float)
    U = np.asarray(U, dtype=float)

    # Get the number of observations.
    N = D2.shape[0]

    # Validate input dimensions.
    if D2.ndim != 2 or D2.shape != (N, N):
        raise ValueError("D2 must be a square distance matrix.")

    if U.ndim != 2 or U.shape[0] != N:
        raise ValueError("U must have the same number of observations as D2.")

    # Validate the number of neighbours.
    if not isinstance(k, (int, np.integer)) or not 1 <= k < N:
        raise ValueError("k must be an integer between 1 and N - 1.")

    # Validate the manifold distance threshold.
    if not np.isscalar(gap) or not np.isfinite(gap) or gap < 0:
        raise ValueError("gap must be a non-negative finite number.")

    # Find the k nearest neighbours in the observed space.
    # Each row contains directed nearest-neighbour connections.
    neighbors = knn_indices(D2, k)

    # Calculate true manifold distances between each point
    # and its selected nearest neighbours.
    manifold_distances = np.linalg.norm(
        U[:, None, :] - U[neighbors],
        axis=2
    )

    # Count directed edges whose manifold distance exceeds gap.
    count = np.count_nonzero(manifold_distances > gap)

    # Return a Python integer.
    return int(count)

# Step 24 - kernel_pca
def fix_signs(W):
    W = np.asarray(W, dtype=float)
    rows = np.argmax(np.abs(W), axis=0)
    flips = np.sign(W[rows, np.arange(W.shape[1])])
    flips[flips == 0] = 1.0
    return W * flips


def pairwise_sqdist(A, B=None):
    A = np.asarray(A, dtype=float)
    B = A if B is None else np.asarray(B, dtype=float)

    d2 = (
        (A ** 2).sum(axis=1)[:, None]
        + (B ** 2).sum(axis=1)[None, :]
        - 2.0 * A @ B.T
    )

    return np.clip(d2, 0.0, None)


def rbf_kernel(A, B, sigma2):
    """
    Compute the RBF (Gaussian) kernel between rows of A and B.
    """
    return np.exp(-pairwise_sqdist(A, B) / (2.0 * sigma2))


def kernel_pca(K, L):
    """
    Perform Kernel PCA using a precomputed training Gram matrix.

    Parameters
    ----------
    K : ndarray of shape (N, N)
        Training kernel (Gram) matrix.

    L : int
        Number of kernel principal components.

    Returns
    -------
    dict
        {
            "coef": kernel projection coefficients, shape (N, L),
            "eigvals": all centered eigenvalues, shape (N,),
            "Z": training embedding, shape (N, L),
            "col_mean": training kernel column means, shape (N,),
            "all_mean": overall training kernel mean, Python float
        }
    """

    # Convert the kernel matrix into a floating-point NumPy array.
    K = np.asarray(K, dtype=float)

    # Get the number of training observations.
    N = K.shape[0]

    # Validate the input matrix.
    if K.ndim != 2 or K.shape != (N, N):
        raise ValueError("K must be a square Gram matrix.")

    if not np.all(np.isfinite(K)):
        raise ValueError("K must contain only finite values.")

    if not isinstance(L, (int, np.integer)) or not 0 <= L <= N:
        raise ValueError("L must be an integer between 0 and N.")

    # Ensure numerical symmetry.
    K = 0.5 * (K + K.T)

    # Store the original training kernel centering statistics.
    col_mean = K.mean(axis=0)
    all_mean = float(K.mean())

    # Construct the centering matrix.
    H = np.eye(N) - np.ones((N, N)) / N

    # Double-center the kernel matrix.
    K_centered = H @ K @ H

    # Ensure numerical symmetry after centering.
    K_centered = 0.5 * (K_centered + K_centered.T)

    # Eigendecompose the centered kernel matrix.
    eigenvalues, eigenvectors = np.linalg.eigh(K_centered)

    # Sort eigenvalues and eigenvectors in decreasing order.
    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Clip negative eigenvalues caused by numerical errors.
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Select the top L eigenvectors.
    U_L = eigenvectors[:, :L]

    # Fix eigenvector signs for reproducibility.
    U_L = fix_signs(U_L)

    # Calculate Lambda^(-1/2).
    # Handle zero eigenvalues safely.
    inv_sqrt = np.zeros(L, dtype=float)

    positive = eigenvalues[:L] > 0.0

    inv_sqrt[positive] = (
        1.0 / np.sqrt(eigenvalues[:L][positive])
    )

    # Calculate the kernel PCA coefficients:
    # coef = U_L @ Lambda_L^(-1/2)
    coef = U_L * inv_sqrt[None, :]

    # Calculate the training embedding.
    Z = K_centered @ coef

    # Return the fitted kernel PCA model.
    return {
        "coef": coef,
        "eigvals": eigenvalues,
        "Z": Z,
        "col_mean": col_mean,
        "all_mean": all_mean
    }


def kpca_transform(k_star, model):
    """
    Project new observations into an existing Kernel PCA space.

    Parameters
    ----------
    k_star : ndarray of shape (M, N)
        Kernel matrix between M new observations
        and N training observations.

    model : dict
        Model returned by kernel_pca().

    Returns
    -------
    ndarray of shape (M, L)
        Kernel PCA embedding of the new observations.
    """

    # Convert the new-point kernel matrix into a float array.
    k_star = np.asarray(k_star, dtype=float)

    # Retrieve training centering statistics and coefficients.
    coef = model["coef"]
    col_mean = model["col_mean"]
    all_mean = model["all_mean"]

    # Validate dimensions.
    if k_star.ndim != 2:
        raise ValueError("k_star must be a two-dimensional array.")

    if k_star.shape[1] != coef.shape[0]:
        raise ValueError(
            "k_star must have one column per training observation."
        )

    if col_mean.shape != (k_star.shape[1],):
        raise ValueError("Invalid training column means.")

    # Compute the mean kernel value for each new observation.
    row_mean = k_star.mean(axis=1, keepdims=True)

    # Center the new-point kernel matrix consistently
    # with the original training kernel matrix.
    K_centered = (
        k_star
        - row_mean
        - col_mean[None, :]
        + all_mean
    )

    # Project the centered kernel matrix into the
    # learned Kernel PCA space.
    return K_centered @ coef

# Step 25 - lle
def lle(X, k=10, L=2, reg=1e-3):
    """
    Perform Locally Linear Embedding (LLE).

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Input dataset, with one observation per row.

    k : int, default=10
        Number of nearest neighbours.

    L : int, default=2
        Number of embedding dimensions.

    reg : float, default=1e-3
        Regularization parameter for local reconstruction.

    Returns
    -------
    Z : ndarray of shape (N, L)
        Low-dimensional LLE embedding.

    W : ndarray of shape (N, N)
        Sparse reconstruction weight matrix.
    """

    # Convert input into a floating-point NumPy array.
    X = np.asarray(X, dtype=float)

    # Get the number of observations and features.
    N, D = X.shape

    # Validate input dimensions and parameters.
    if X.ndim != 2 or N < 2:
        raise ValueError("X must contain at least two observations.")

    if not isinstance(k, (int, np.integer)) or not 1 <= k < N:
        raise ValueError("k must be an integer between 1 and N - 1.")

    if not isinstance(L, (int, np.integer)) or not 0 <= L < N:
        raise ValueError("L must be an integer between 0 and N - 1.")

    if not np.isfinite(reg) or reg < 0:
        raise ValueError("reg must be non-negative and finite.")

    # -------------------------------------------------
    # Step 1: Find k nearest neighbours of every point.
    # -------------------------------------------------

    D2 = pairwise_sqdist(X)

    idx = knn_indices(D2, k)

    # -------------------------------------------------
    # Step 2: Calculate local reconstruction weights.
    # -------------------------------------------------

    # Initialize the N x N reconstruction weight matrix.
    W = np.zeros((N, N), dtype=float)

    # Each observation is reconstructed independently
    # from its k nearest neighbours.
    for i in range(N):

        # Get the indices of the neighbours of observation i.
        neighbors = idx[i]

        # Calculate neighbour difference vectors:
        # A = X[neighbors] - X[i]
        A = X[neighbors] - X[i]

        # Construct the local covariance matrix:
        # C = A @ A.T
        C = A @ A.T

        # Regularize the local covariance matrix.
        # The trace-based regularization helps handle
        # rank-deficient local neighbourhoods.
        C += (
            reg * np.trace(C) + 1e-12
        ) * np.eye(k)

        # Solve C @ w = 1.
        # This is more numerically stable than explicitly
        # calculating the inverse of C.
        w = np.linalg.solve(C, np.ones(k))

        # Normalize the weights so that they sum to one.
        w /= np.sum(w)

        # Store the weights in their corresponding positions.
        W[i, neighbors] = w

    # -------------------------------------------------
    # Step 3: Construct the LLE optimization matrix.
    # -------------------------------------------------

    # Compute (I - W).
    M = np.eye(N) - W

    # Construct the symmetric matrix:
    # M.T @ M = (I - W).T @ (I - W)
    M = M.T @ M

    # Remove small numerical asymmetries.
    M = 0.5 * (M + M.T)

    # -------------------------------------------------
    # Step 4: Eigendecomposition.
    # -------------------------------------------------

    # Compute eigenvalues and eigenvectors.
    # eigh returns eigenvalues in ascending order.
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    # Skip the smallest eigenvector, which corresponds
    # to the constant embedding direction.
    #
    # Select the next L eigenvectors.
    Z = eigenvectors[:, 1:L + 1]

    # -------------------------------------------------
    # Step 5: Fix embedding signs.
    # -------------------------------------------------

    # Make the embedding deterministic by fixing
    # the sign of each eigenvector.
    Z = fix_signs(Z)

    # Return the embedding and reconstruction weights.
    return Z, W

# Step 26 - laplacian_eigenmaps
def fix_signs(W):
    W = np.asarray(W, dtype=float)
    rows = np.argmax(np.abs(W), axis=0)
    flips = np.sign(W[rows, np.arange(W.shape[1])])
    flips[flips == 0] = 1.0
    return W * flips


def pairwise_sqdist(A, B=None):
    A = np.asarray(A, dtype=float)
    B = A if B is None else np.asarray(B, dtype=float)

    d2 = (
        (A ** 2).sum(axis=1)[:, None]
        + (B ** 2).sum(axis=1)[None, :]
        - 2.0 * A @ B.T
    )

    return np.clip(d2, 0.0, None)


def knn_indices(D2, k):
    D2 = np.array(D2, dtype=float, copy=True)
    np.fill_diagonal(D2, np.inf)
    return np.argsort(D2, axis=1, kind="stable")[:, :k]


def graph_weights(X, k, sigma):
    """
    Construct a symmetric k-nearest-neighbour graph
    with heat-kernel edge weights.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Input observations.

    k : int
        Number of nearest neighbours.

    sigma : float
        Heat-kernel bandwidth parameter.

    Returns
    -------
    W : ndarray of shape (N, N)
        Symmetric graph weight matrix with zeros on the diagonal.
    """

    # Convert input into a floating-point NumPy array.
    X = np.asarray(X, dtype=float)

    # Get the number of observations.
    N = X.shape[0]

    # Validate the input.
    if X.ndim != 2 or N < 2:
        raise ValueError("X must contain at least two observations.")

    if not isinstance(k, (int, np.integer)) or not 1 <= k < N:
        raise ValueError("k must be an integer between 1 and N - 1.")

    if not np.isscalar(sigma) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a positive finite scalar.")

    # Compute squared Euclidean distances.
    D2 = pairwise_sqdist(X)

    # Find the k nearest neighbours of every observation.
    neighbors = knn_indices(D2, k)

    # Construct the directed nearest-neighbour adjacency matrix.
    adjacency = np.zeros((N, N), dtype=bool)

    adjacency[np.arange(N)[:, None], neighbors] = True

    # Symmetrize the graph.
    # An edge exists if either endpoint selects the other.
    adjacency = adjacency | adjacency.T

    # Calculate the heat-kernel weights:
    #
    # W_ij = exp(-||x_i - x_j||^2 / (2 * sigma^2))
    W = np.exp(
        -D2 / (2.0 * sigma**2)
    )

    # Retain weights only for neighbouring observations.
    W *= adjacency

    # Explicitly set the diagonal to zero.
    np.fill_diagonal(W, 0.0)

    return W


def dirichlet_energy(W, f):
    """
    Calculate the Dirichlet energy f.T @ L @ f,
    where L is the unnormalized graph Laplacian.

    Parameters
    ----------
    W : ndarray of shape (N, N)
        Symmetric graph weight matrix.

    f : ndarray of shape (N,)
        Function values defined on graph vertices.

    Returns
    -------
    float
        Dirichlet energy.
    """

    # Convert inputs into floating-point NumPy arrays.
    W = np.asarray(W, dtype=float)
    f = np.asarray(f, dtype=float)

    # Ensure f is a one-dimensional vector.
    f = f.ravel()

    # Validate dimensions.
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix.")

    if f.shape != (W.shape[0],):
        raise ValueError("f must contain one value per graph vertex.")

    # Calculate the degree vector.
    degrees = np.sum(W, axis=1)

    # Apply the graph Laplacian:
    # L = D - W
    Lf = degrees * f - W @ f

    # Calculate f.T @ L @ f.
    energy = f @ Lf

    # Return a Python float.
    return float(energy)


def laplacian_eigenmaps(X, k=10, sigma=1.0, L=2):
    """
    Perform dimensionality reduction using Laplacian Eigenmaps.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Input observations.

    k : int, default=10
        Number of nearest neighbours.

    sigma : float, default=1.0
        Heat-kernel bandwidth parameter.

    L : int, default=2
        Number of embedding dimensions.

    Returns
    -------
    Z : ndarray of shape (N, L)
        Low-dimensional embedding.

    lam : ndarray of shape (N,)
        All eigenvalues in increasing order,
        with numerical negative values clipped to zero.

    W : ndarray of shape (N, N)
        Symmetric heat-kernel graph weight matrix.
    """

    # Convert input into a floating-point NumPy array.
    X = np.asarray(X, dtype=float)

    # Get the number of observations.
    N = X.shape[0]

    # Validate the input.
    if X.ndim != 2 or N < 2:
        raise ValueError("X must contain at least two observations.")

    if not isinstance(L, (int, np.integer)) or not 0 <= L < N:
        raise ValueError("L must be an integer between 0 and N - 1.")

    # -------------------------------------------------
    # Step 1: Construct the heat-kernel graph.
    # -------------------------------------------------

    W = graph_weights(X, k, sigma)

    # -------------------------------------------------
    # Step 2: Construct the graph Laplacian.
    # -------------------------------------------------

    # Compute the degree vector.
    degrees = np.sum(W, axis=1)

    # Construct the degree matrix implicitly:
    # D = diag(degrees)
    #
    # The unnormalized graph Laplacian is:
    # L_graph = D - W
    L_graph = np.diag(degrees) - W

    # -------------------------------------------------
    # Step 3: Construct the symmetric normalized Laplacian.
    # -------------------------------------------------

    # Calculate D^(-1/2).
    #
    # A zero-degree vertex receives an inverse degree of zero
    # to avoid division-by-zero errors.
    inv_sqrt_degrees = np.divide(
        1.0,
        np.sqrt(degrees),
        out=np.zeros_like(degrees),
        where=degrees > 0
    )

    # Compute:
    # L_sym = D^(-1/2) @ L_graph @ D^(-1/2)
    L_sym = (
        inv_sqrt_degrees[:, None]
        * L_graph
        * inv_sqrt_degrees[None, :]
    )

    # Ensure numerical symmetry.
    L_sym = 0.5 * (L_sym + L_sym.T)

    # -------------------------------------------------
    # Step 4: Eigendecomposition.
    # -------------------------------------------------

    # Compute eigenvalues and eigenvectors.
    # eigh returns eigenvalues in increasing order.
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)

    # Clip rounding-level negative eigenvalues to zero.
    lam = np.maximum(eigenvalues, 0.0)

    # -------------------------------------------------
    # Step 5: Obtain the generalized eigenvectors.
    # -------------------------------------------------

    # Convert the symmetric eigenvectors into generalized
    # eigenvectors using:
    #
    # z = D^(-1/2) @ u
    #
    # Skip the first eigenvector corresponding to lambda = 0.
    U_L = eigenvectors[:, 1:L + 1]

    Z = (
        inv_sqrt_degrees[:, None]
        * U_L
    )

    # Fix the signs of the embedding coordinates.
    Z = fix_signs(Z)

    # Return the embedding, all eigenvalues, and graph weights.
    return Z, lam, W

# Step 27 - conditional_p
def conditional_p(D2, perplexity=30.0, tol=1e-5, max_iter=50):
    """
    Compute conditional neighbourhood probabilities for SNE
    using binary search to calibrate the Gaussian bandwidth
    of every observation.

    Parameters
    ----------
    D2 : ndarray of shape (N, N)
        Squared pairwise distance matrix.

    perplexity : float, default=30.0
        Target perplexity of each conditional distribution.

    tol : float, default=1e-5
        Convergence tolerance for entropy.

    max_iter : int, default=50
        Maximum number of binary search iterations per observation.

    Returns
    -------
    P : ndarray of shape (N, N)
        Conditional probability matrix with zero diagonal.

    sigmas : ndarray of shape (N,)
        Gaussian bandwidth for each observation.
    """

    # Convert the distance matrix into a floating-point array.
    D2 = np.asarray(D2, dtype=float)

    # Get the number of observations.
    N = D2.shape[0]

    # Validate the distance matrix.
    if D2.ndim != 2 or D2.shape != (N, N):
        raise ValueError("D2 must be a square matrix.")

    if N < 2:
        raise ValueError("At least two observations are required.")

    if not np.isfinite(perplexity) or not 0 < perplexity <= N - 1:
        raise ValueError("perplexity must be in the range (0, N-1].")

    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be non-negative and finite.")

    if not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError("max_iter must be a non-negative integer.")

    # Allocate the conditional probability matrix and bandwidths.
    P = np.zeros((N, N), dtype=float)
    sigmas = np.empty(N, dtype=float)

    # Target entropy in bits:
    # H(P_i) = log2(perplexity)
    target_entropy = np.log2(perplexity)

    # Calibrate the bandwidth independently for every observation.
    for i in range(N):

        # Extract distances from observation i to all other points.
        distances = D2[i].copy()

        # Exclude the observation itself.
        distances[i] = np.inf

        # Identify valid neighbour distances.
        valid = np.isfinite(distances)

        if not np.any(valid):
            raise ValueError("Each observation must have at least one finite neighbour.")

        # Subtract the smallest distance for numerical stability.
        # This ensures that at least one exponential remains equal to 1.
        min_dist = np.min(distances[valid])

        shifted_distances = distances - min_dist

        # Initialize beta and its search bounds.
        beta = 1.0
        beta_min = 0.0
        beta_max = np.inf

        # -----------------------------------------
        # Binary search for the target perplexity
        # -----------------------------------------

        for _ in range(max_iter):

            # Compute Gaussian probabilities in exponential form.
            exp_values = np.zeros(N, dtype=float)

            exp_values[valid] = np.exp(
                -beta * shifted_distances[valid]
            )

            # Normalize to obtain conditional probabilities.
            p = exp_values / np.sum(exp_values)

            # Calculate entropy in bits.
            positive = p > 0

            entropy = -np.sum(
                p[positive] * np.log2(p[positive])
            )

            # Check whether the target entropy is reached.
            if abs(entropy - target_entropy) < tol:
                break

            # If entropy is too high, increase beta.
            # A larger beta produces a narrower distribution.
            if entropy > target_entropy:

                beta_min = beta

                # No upper bound has been established yet.
                if np.isinf(beta_max):
                    beta *= 2.0

                else:
                    # Bisect between the established bounds.
                    beta = 0.5 * (beta_min + beta_max)

            else:

                # Entropy is too low: decrease beta.
                beta_max = beta

                if beta_min == 0.0:
                    beta *= 0.5

                else:
                    # Bisect between the established bounds.
                    beta = 0.5 * (beta_min + beta_max)

        # Recalculate probabilities using the final beta.
        # This also handles max_iter=0 correctly.
        exp_values = np.zeros(N, dtype=float)

        exp_values[valid] = np.exp(
            -beta * shifted_distances[valid]
        )

        P[i] = exp_values / np.sum(exp_values)

        # Explicitly enforce a zero diagonal.
        P[i, i] = 0.0

        # Convert beta into the Gaussian bandwidth:
        # beta = 1 / (2 * sigma_i^2)
        sigmas[i] = np.sqrt(1.0 / (2.0 * beta))

    return P, sigmas

# Step 28 - tsne_kl_grad
def tsne_kl_grad(P, Z, kernel="t"):
    """
    Compute the symmetric SNE / t-SNE KL-divergence
    objective and its gradient.

    Parameters
    ----------
    P : ndarray of shape (N, N)
        Symmetric joint probability distribution in the
        original high-dimensional space.

    Z : ndarray of shape (N, 2)
        Low-dimensional embedding.

    kernel : str, default="t"
        Kernel used in the embedding space:
        "t"    : Student-t kernel.
        "gauss": Gaussian kernel.

    Returns
    -------
    kl : float
        KL-divergence objective.

    grad : ndarray of shape (N, 2)
        Gradient of the KL-divergence with respect to Z.
    """

    # Convert inputs into floating-point NumPy arrays.
    P = np.asarray(P, dtype=float)
    Z = np.asarray(Z, dtype=float)

    # Get the number of observations.
    N = Z.shape[0]

    # Validate input dimensions.
    if Z.ndim != 2 or Z.shape[1] != 2:
        raise ValueError("Z must have shape (N, 2).")

    if P.shape != (N, N):
        raise ValueError("P must have shape (N, N).")

    if not np.all(np.isfinite(P)) or np.any(P < 0):
        raise ValueError("P must contain finite, non-negative probabilities.")

    if kernel not in ("t", "gauss"):
        raise ValueError("kernel must be either 't' or 'gauss'.")

    if N < 2:
        raise ValueError("At least two observations are required.")

    # -------------------------------------------------
    # Step 1: Calculate pairwise squared distances
    # in the low-dimensional embedding.
    # -------------------------------------------------

    D2 = pairwise_sqdist(Z)

    # -------------------------------------------------
    # Step 2: Calculate the unnormalized similarities.
    # -------------------------------------------------

    if kernel == "t":

        # Student-t kernel with one degree of freedom:
        #
        # num_ij = 1 / (1 + ||z_i - z_j||^2)
        num = 1.0 / (1.0 + D2)

    else:

        # Gaussian kernel:
        #
        # num_ij = exp(-||z_i - z_j||^2)
        #
        # Subtract the smallest off-diagonal distance
        # for numerical stability. This does not change
        # the normalized probability distribution.
        off_diagonal = ~np.eye(N, dtype=bool)

        min_dist = np.min(D2[off_diagonal])

        num = np.exp(-(D2 - min_dist))

    # Exclude self-similarities.
    np.fill_diagonal(num, 0.0)

    # -------------------------------------------------
    # Step 3: Normalize the similarities.
    # -------------------------------------------------

    # Normalize over all ordered pairs i != j.
    Q = num / np.sum(num)

    # -------------------------------------------------
    # Step 4: Calculate the KL-divergence objective.
    # -------------------------------------------------

    # Only entries where P is positive contribute.
    mask = P > 0

    # Floor Q inside the logarithm to avoid log(0).
    Q_safe = np.maximum(Q[mask], 1e-300)

    # Symmetric KL-divergence:
    #
    # KL(P || Q) = sum_ij P_ij * log(P_ij / Q_ij)
    kl = float(
        np.sum(
            P[mask] * np.log(P[mask] / Q_safe)
        )
    )

    # -------------------------------------------------
    # Step 5: Calculate the gradient.
    # -------------------------------------------------

    # Difference between original-space and
    # embedding-space joint probabilities.
    PQ_diff = P - Q

    if kernel == "t":

        # Student-t gradient:
        #
        # grad_i = 4 * sum_j (
        #   (p_ij - q_ij) * (z_i - z_j)
        #   / (1 + ||z_i - z_j||^2)
        # )

        weights = PQ_diff / (1.0 + D2)

    else:

        # Gaussian SNE gradient:
        #
        # grad_i = 4 * sum_j (
        #   (p_ij - q_ij) * (z_i - z_j)
        # )

        weights = PQ_diff

    # -------------------------------------------------
    # Step 6: Vectorize the gradient calculation.
    # -------------------------------------------------

    # sum_j w_ij * (z_i - z_j)
    #
    # Equivalent to:
    # sum_j(w_ij) * z_i - sum_j(w_ij * z_j)
    grad = 4.0 * (
        np.sum(weights, axis=1, keepdims=True) * Z
        - weights @ Z
    )

    # Return the KL objective and its gradient.
    return kl, grad

# Step 29 - fit_tsne
def fit_tsne(
    X,
    perplexity=30.0,
    n_iter=500,
    lr=None,
    seed=0,
    kernel="t",
    exaggeration=12.0,
    exag_iters=100
):
    """
    Fit a t-SNE or Gaussian SNE embedding using
    momentum-based gradient descent.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Input observations.

    perplexity : float, default=30.0
        Target perplexity for conditional probabilities.

    n_iter : int, default=500
        Number of optimization iterations.

    lr : float or None, default=None
        Learning rate. If None, uses max(N / 8, 50).

    seed : int, default=0
        Random seed for reproducibility.

    kernel : str, default="t"
        Embedding kernel: "t" or "gauss".

    exaggeration : float, default=12.0
        Early exaggeration factor.

    exag_iters : int, default=100
        Number of early exaggeration iterations.

    Returns
    -------
    Z : ndarray of shape (N, 2)
        Final two-dimensional embedding.

    trace : ndarray of shape (n_iter,)
        True KL-divergence at every iteration.
    """

    # Convert input into a floating-point NumPy array.
    X = np.asarray(X, dtype=float)

    # Get the number of observations.
    N = X.shape[0]

    # Validate the input.
    if X.ndim != 2 or N < 2:
        raise ValueError("X must contain at least two observations.")

    if not isinstance(n_iter, (int, np.integer)) or n_iter < 0:
        raise ValueError("n_iter must be a non-negative integer.")

    if not isinstance(exag_iters, (int, np.integer)) or exag_iters < 0:
        raise ValueError("exag_iters must be a non-negative integer.")

    if not np.isfinite(exaggeration) or exaggeration <= 0:
        raise ValueError("exaggeration must be positive and finite.")

    if lr is not None and (not np.isfinite(lr) or lr <= 0):
        raise ValueError("lr must be positive and finite.")

    # -------------------------------------------------
    # Step 1: Calculate high-dimensional probabilities.
    # -------------------------------------------------

    # Calculate conditional probabilities using the
    # target perplexity and squared Euclidean distances.
    P_cond, _ = conditional_p(
        pairwise_sqdist(X),
        perplexity
    )

    # Symmetrize the conditional probabilities.
    #
    # p_ij = (p_j|i + p_i|j) / (2N)
    P = (P_cond + P_cond.T) / (2.0 * N)

    # -------------------------------------------------
    # Step 2: Set the learning rate.
    # -------------------------------------------------

    if lr is None:
        lr = max(N / 8.0, 50.0)

    # -------------------------------------------------
    # Step 3: Initialize the low-dimensional embedding.
    # -------------------------------------------------

    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    # Start with small random coordinates.
    Z = 1e-4 * rng.standard_normal((N, 2))

    # Initialize velocity to zero.
    v = np.zeros_like(Z)

    # Allocate the KL-divergence history.
    trace = np.empty(n_iter, dtype=float)

    # -------------------------------------------------
    # Step 4: Gradient descent optimization.
    # -------------------------------------------------

    for it in range(n_iter):

        # Determine the early exaggeration factor.
        boost = exaggeration if it < exag_iters else 1.0

        # Calculate the KL objective and gradient.
        # During early exaggeration, use boosted P.
        kl, g = tsne_kl_grad(
            P * boost,
            Z,
            kernel=kernel
        )

        # Record the TRUE KL-divergence of P.
        # Correct for the early exaggeration factor.
        trace[it] = kl / boost - np.log(boost)

        # Select the momentum coefficient.
        momentum = 0.5 if it < exag_iters else 0.8

        # Update the velocity using momentum.
        v = momentum * v - lr * g

        # Update the embedding coordinates.
        Z = Z + v

        # Re-center the embedding after every iteration.
        Z = Z - np.mean(Z, axis=0, keepdims=True)

    # Return the final embedding and KL-divergence history.
    return Z, trace

# Step 30 - embedding_report
def embedding_report(X, y, k=10, perplexity=30.0, n_iter=500, seed=0):
    """
    Compare six dimensionality-reduction methods on the same dataset.

    Methods:
        1. PCA
        2. RBF Kernel PCA
        3. Isomap
        4. Locally Linear Embedding (LLE)
        5. Laplacian Eigenmaps
        6. t-SNE

    Each embedding is evaluated using trustworthiness and
    leave-one-out 5-NN classification accuracy.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        Input dataset, with one observation per row.

    y : ndarray of shape (N,)
        Class labels.

    k : int, default=10
        Number of neighbours for manifold learning methods
        and trustworthiness evaluation.

    perplexity : float, default=30.0
        Target perplexity for t-SNE.

    n_iter : int, default=500
        Number of t-SNE optimization iterations.

    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    dict
        A dictionary containing the trustworthiness and
        5-NN accuracy for each embedding method.
    """

    # Convert inputs into NumPy arrays.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    # Validate input dimensions.
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")

    N = X.shape[0]

    if y.ndim != 1 or len(y) != N:
        raise ValueError("y must contain one label per observation.")

    if X.shape[1] < 1 or N < 6:
        raise ValueError("X must have at least one feature and six observations.")

    # -------------------------------------------------
    # Step 1: Compute pairwise squared Euclidean distances.
    # -------------------------------------------------

    D2 = pairwise_sqdist(X)

    # Calculate sigma2 as the median of all nonzero
    # entries in the original squared distance matrix.
    nonzero_distances = D2[D2 > 0]

    if nonzero_distances.size == 0:
        raise ValueError("X must contain at least two distinct observations.")

    sigma2 = float(np.median(nonzero_distances))

    # -------------------------------------------------
    # Step 2: PCA embedding.
    # -------------------------------------------------

    # Use the principal component scores from SVD-based PCA.
    Z_pca = pca_svd(X, 2)[3]

    # -------------------------------------------------
    # Step 3: RBF Kernel PCA embedding.
    # -------------------------------------------------

    # Construct the RBF kernel using the median squared distance.
    K = rbf_kernel(X, X, sigma2)

    # Fit Kernel PCA and extract the training embedding.
    Z_kpca = kernel_pca(K, 2)["Z"]

    # -------------------------------------------------
    # Step 4: Isomap embedding.
    # -------------------------------------------------

    # Compute the Isomap embedding and retain the indices
    # of points belonging to the largest connected component.
    Z_isomap, keep, _ = isomap(X, k, 2)

    # Restrict the original distance matrix and labels
    # to the retained Isomap observations.
    D2_isomap = D2[np.ix_(keep, keep)]
    y_isomap = y[keep]

    # -------------------------------------------------
    # Step 5: Locally Linear Embedding.
    # -------------------------------------------------

    # Extract the LLE embedding.
    Z_lle = lle(X, k, 2)[0]

    # -------------------------------------------------
    # Step 6: Laplacian Eigenmaps.
    # -------------------------------------------------

    # Use sqrt(sigma2) as the heat-kernel bandwidth.
    Z_laplacian = laplacian_eigenmaps(
        X,
        k,
        np.sqrt(sigma2),
        2
    )[0]

    # -------------------------------------------------
    # Step 7: t-SNE embedding.
    # -------------------------------------------------

    # Fit t-SNE using the requested perplexity and iterations.
    Z_tsne = fit_tsne(
        X,
        perplexity=perplexity,
        n_iter=n_iter,
        seed=seed
    )[0]

    # -------------------------------------------------
    # Step 8: Collect all embeddings.
    # -------------------------------------------------

    embeddings = {
        "pca": Z_pca,
        "kpca_rbf": Z_kpca,
        "lle": Z_lle,
        "laplacian": Z_laplacian,
        "tsne": Z_tsne
    }

    # -------------------------------------------------
    # Step 9: Evaluate the embeddings.
    # -------------------------------------------------

    report = {}

    # Evaluate PCA, Kernel PCA, LLE, Laplacian Eigenmaps,
    # and t-SNE using all original observations.
    for name, Z in embeddings.items():

        report[name] = {
            "trust": float(
                trustworthiness(D2, Z, k)
            ),
            "knn": float(
                knn_accuracy(Z, y, 5)
            )
        }

    # Evaluate Isomap separately because it may discard
    # observations outside the largest connected component.
    report["isomap"] = {
        "trust": float(
            trustworthiness(D2_isomap, Z_isomap, k)
        ),
        "knn": float(
            knn_accuracy(Z_isomap, y_isomap, 5)
        )
    }

    # Return the complete comparison report.
    return report

