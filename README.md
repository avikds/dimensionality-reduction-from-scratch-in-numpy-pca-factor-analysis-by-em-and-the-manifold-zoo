# Dimensionality Reduction from Scratch in NumPy: PCA, Factor Analysis by EM and the Manifold Zoo

Build the dimensionality-reduction methods of chapter 20 of Murphy's Probabilistic Machine Learning in NumPy and put each one to a measured test. Compute PCA three ways (covariance, the Gram trick on 4096-pixel faces, the SVD) and find out why test reconstruction error can never pick the number of components. Turn PCA into a density with probabilistic PCA and score held-out faces, where the U-shaped curve finally appears. Fit factor analysis by EM, see that two seeds agree on the likelihood and disagree on every loading, recover the structure with varimax, and watch one noisy sensor capture PCA while factor analysis sets it aside. Run EM for PCA and test the claim that it needs only a few iterations, then cover a curved dataset with a mixture of PPCA models. Next comes the manifold zoo on the Swiss roll and the digits, scored against true sheet coordinates and by trustworthiness: classical MDS (which is PCA), Isomap and its short-circuit failure, kernel PCA, locally linear embedding, Laplacian eigenmaps and t-SNE with perplexity calibration. Along the way you reproduce Figures 20.12, 20.34, 20.35 and 20.42 and find which of the book's claims hold on the data in front of you (Murphy, PML Book 1, sections 20.1, 20.2 and 20.4).

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** pca_eig
- [x] **2.** pca_gram
- [x] **3.** pca_svd
- [x] **4.** correlation_pca
- [x] **5.** reconstruction_curves
- [x] **6.** profile_log_likelihood
- [x] **7.** ppca_fit
- [x] **8.** ppca_logpdf
- [x] **9.** ppca_posterior
- [x] **10.** heldout_loglik_curve
- [x] **11.** fa_e_step
- [x] **12.** fa_m_step
- [x] **13.** fit_fa
- [x] **14.** varimax
- [x] **15.** sensor_noise_experiment
- [x] **16.** em_pca
- [x] **17.** mppca_e_step

---

Built on Deep-ML.
