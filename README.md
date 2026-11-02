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
- [x] **18.** fit_mppca
- [x] **19.** swiss_roll
- [x] **20.** classical_mds
- [x] **21.** trustworthiness
- [x] **22.** knn_geodesics
- [x] **23.** isomap
- [x] **24.** kernel_pca
- [x] **25.** lle
- [x] **26.** laplacian_eigenmaps
- [x] **27.** conditional_p
- [x] **28.** tsne_kl_grad
- [x] **29.** fit_tsne
- [x] **30.** embedding_report

## Results

```
Part 1: PCA three ways
  digits: eig vs svd max |diff| 4.4e-15, eig vs Gram on 50 images 3.1e-15
  digits: 2 components explain 0.285 of the variance, rank 61 of 64
  faces (D=4096, N=400): 3 eigenfaces explain 0.458, 50 explain 0.874
  face recognition, 1-NN over 10 splits: pixels 0.926, 20 eigenfaces 0.924
  wine: raw PC1 0.998 of the variance (proline), 5-NN on 2 PCs 0.680
  wine: standardized PC1 0.362, 5-NN on 2 PCs 0.961
Part 2: how many components
  digits L=[1, 5, 10, 20, 40, 60]
    train error [1020.1, 546.1, 313.5, 125.5, 14.0, 0.0]
    test error  [1032.6, 552.2, 321.4, 134.7, 15.2, 0.0]
  profile likelihood: digits L*=4, faces L*=2
Part 3: probabilistic PCA on 300 training and 100 test faces
  L=[1, 10, 30, 50, 60, 80, 120, 200, 280]
    train log-lik [2880, 4460, 5753, 6498, 6810, 7376, 8425, 10654, 15068]
    test log-lik  [2660, 4207, 5063, 5308, 5309, 5183, 4118, -4766, -150131]   best L=60
    test recon    [65.0, 30.0, 18.4, 14.7, 13.7, 12.1, 10.3, 8.4, 7.4]
  10 unseen people: best L=20
Part 4: factor analysis on dequantized digits, 497 test images
  diagonal Gaussian -150.36
  L= 2 FA -145.40  PPCA -177.70
  L=10 FA -136.32  PPCA -160.64
  L=50 FA -131.13  PPCA -131.72
  two EM seeds: log-lik differs by 5.3e-15, largest loading difference 2.08
  loadings under 0.2: EM 0.15, varimax 0.67, true 0.67
  one bad sensor, seed 0: subspace angle PPCA 83.0 deg, FA 1.0 deg, fitted psi_0 25.3
  one bad sensor, seed 1: subspace angle PPCA 65.5 deg, FA 3.9 deg, fitted psi_0 23.9
Part 5: EM for PCA and mixtures
  digits L= 2: EM iterations to within 1 degree: 29
  digits L= 5: EM iterations to within 1 degree: 30
  digits L=10: EM iterations to within 1 degree: 17
  digits L=20: EM iterations to within 1 degree: 235
  large eigengap, D=500, L=10: 3  (the book reports a mean of 3.6)
  mixture of PPCA K= 1: train -2.07, test -2.11
  mixture of PPCA K= 3: train -0.63, test -0.67
  mixture of PPCA K=10: train -0.14, test -0.20
  mixture of PPCA K=20: train -0.07, test -0.23
Parts 6-7: Swiss roll, 1000 points, (arc-length, height) Spearman and trustworthiness against the true sheet
  pca                      arc 0.224  height 0.211  trust 0.978
  isomap k=10              arc 1.000  height 0.995  trust 1.000
  lle k=10                 arc 1.000  height 0.508  trust 0.994
  laplacian k=10 sigma=1   arc 0.999  height 0.026  trust 0.950
  tsne perp=30             arc 0.278  height 0.104  trust 0.999
  isomap noise=0.5 seed=1 k= 5: false edges   0 of 5000, arc 0.999, height 0.947
  isomap noise=0.5 seed=1 k=10: false edges   5 of 10000, arc 0.892, height 0.327
  isomap noise=0.5 seed=2 k=15: false edges  10 of 15000, arc 0.499, height 0.176
  isomap noise=1.0 seed=0 k=10: false edges  48 of 10000, arc 0.277, height 0.114
  noise=0.5, 3 seeds, mean arc-length score: isomap 0.96, lle 0.79, laplacian sigma=1 1.00
  noise=1.0, 3 seeds, mean arc-length score: isomap 0.24, lle 0.10, laplacian sigma=1 0.95
  Figure 20.35 kernel PCA eigenvalues: [22.89, 21.44, 4.44, 4.06] (book 22.56, 20.94, 4.65, 3.99)
Part 8: t-SNE
  600 digits, t-SNE:          KL 0.735, map std 8.65, trust 0.968, 5-NN 0.960
  600 digits, symmetric SNE:  KL 1.093, map std 1.76, trust 0.944, 5-NN 0.923
  three clusters, perplexity   2.0: far gap / near gap 1.32 (true 4.00)
  three clusters, perplexity   5.0: far gap / near gap 1.15 (true 4.00)
  three clusters, perplexity  30.0: far gap / near gap 1.00 (true 4.00)
  three clusters, perplexity 100.0: far gap / near gap 3.95 (true 4.00)
Part 9: all 1797 digits, 64-D to 2-D (k = 10)
  pca        trustworthiness 0.830   5-NN accuracy 0.635
  kpca_rbf   trustworthiness 0.828   5-NN accuracy 0.638
  lle        trustworthiness 0.913   5-NN accuracy 0.914
  laplacian  trustworthiness 0.927   5-NN accuracy 0.919
  tsne       trustworthiness 0.992   5-NN accuracy 0.991
  isomap     trustworthiness 0.837   5-NN accuracy 0.727
```
