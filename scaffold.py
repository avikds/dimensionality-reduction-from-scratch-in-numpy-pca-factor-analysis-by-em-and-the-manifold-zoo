"""
Dimensionality Reduction from Scratch in NumPy: PCA, Factor Analysis by EM and the Manifold Zoo scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Dimensionality Reduction from Scratch (Murphy, Probabilistic Machine Learning: An Introduction, chapter 20).
PCA three ways and choosing L (20.1), probabilistic PCA and factor analysis (20.2), EM for PCA and mixtures (20.2.3, 20.2.6),
then the manifold zoo (20.4) scored on the Swiss roll and on the digits."""
import numpy as np
import dml_data
from sklearn.datasets import load_wine


def main():
    d = dml_data.load("sklearn-digits")
    X, y = np.asarray(d.data, dtype=float), np.asarray(d.target)
    of = dml_data.load("olivetti-faces")
    F, fy = np.asarray(of.data, dtype=float), np.asarray(of.target)

    # --- Part 1: PCA three ways ---
    print("Part 1: PCA three ways")
    _, W1, lam = pca_eig(X, 5)
    print(f"  digits: eig vs svd max |diff| {np.abs(W1 - pca_svd(X, 5)[1]).max():.1e}, eig vs Gram on 50 images {np.abs(pca_eig(X[:50], 5)[1] - pca_gram(X[:50], 5)[1]).max():.1e}")
    print(f"  digits: 2 components explain {lam[:2].sum() / lam.sum():.3f} of the variance, rank {(lam > 1e-9).sum()} of 64")
    _, _, lf = pca_gram(F, 50)
    print(f"  faces (D=4096, N=400): 3 eigenfaces explain {lf[:3].sum() / lf.sum():.3f}, 50 explain {lf[:50].sum() / lf.sum():.3f}")
    acc = {"pixels": [], 20: []}
    for seed in range(10):
        rng = np.random.default_rng(seed)
        tr = np.ones(400, bool)
        for p in range(40):
            tr[rng.choice(np.flatnonzero(fy == p), 2, replace=False)] = False
        acc["pixels"].append(knn_1(F[~tr], F[tr], fy[tr], fy[~tr]))
        mu, W, _ = pca_gram(F[tr], 20)
        acc[20].append(knn_1((F[~tr] - mu) @ W, (F[tr] - mu) @ W, fy[tr], fy[~tr]))
    print(f"  face recognition, 1-NN over 10 splits: pixels {np.mean(acc['pixels']):.3f}, 20 eigenfaces {np.mean(acc[20]):.3f}")
    w = load_wine()
    _, W0, lam0, Z0 = pca_svd(w.data, 2)
    Wc, lamc, Zc, _ = correlation_pca(w.data, 2)
    print(f"  wine: raw PC1 {lam0[0] / lam0.sum():.3f} of the variance ({w.feature_names[int(np.argmax(np.abs(W0[:, 0])))]}), 5-NN on 2 PCs {knn_accuracy(Z0, w.target, 5):.3f}")
    print(f"  wine: standardized PC1 {lamc[0] / lamc.sum():.3f}, 5-NN on 2 PCs {knn_accuracy(Zc, w.target, 5):.3f}")

    # --- Part 2: how many components ---
    print("Part 2: how many components")
    p = np.random.default_rng(0).permutation(len(X))
    Ls = [1, 5, 10, 20, 40, 60]
    rc = reconstruction_curves(X[p[:1300]], X[p[1300:]], Ls)
    print(f"  digits L={Ls}")
    print(f"    train error {np.round(rc['train'], 1).tolist()}")
    print(f"    test error  {np.round(rc['test'], 1).tolist()}")
    print(f"  profile likelihood: digits L*={int(np.argmax(profile_log_likelihood(lam))) + 1}, faces L*={int(np.argmax(profile_log_likelihood(lf[:50]))) + 1}")

    # --- Part 3: probabilistic PCA ---
    print("Part 3: probabilistic PCA on 300 training and 100 test faces")
    pf = np.random.default_rng(0).permutation(400)
    Ls = [1, 10, 30, 50, 60, 80, 120, 200, 280]
    h = heldout_loglik_curve(F[pf[:300]], F[pf[300:]], Ls)
    rc = reconstruction_curves(F[pf[:300]], F[pf[300:]], Ls)
    print(f"  L={Ls}")
    print(f"    train log-lik {np.round(h['train']).astype(int).tolist()}")
    print(f"    test log-lik  {np.round(h['test']).astype(int).tolist()}   best L={h['best_L']}")
    print(f"    test recon    {np.round(rc['test'], 1).tolist()}")
    print(f"  10 unseen people: best L={heldout_loglik_curve(F[fy < 30], F[fy >= 30], [5, 10, 20, 30, 40, 60, 100])['best_L']}")

    # --- Part 4: factor analysis ---
    print("Part 4: factor analysis on dequantized digits, 497 test images")
    rng = np.random.default_rng(0)
    Xj = X + rng.random(X.shape)
    pj = rng.permutation(len(X))
    tr, te = Xj[pj[:1300]], Xj[pj[1300:]]
    v = tr.var(axis=0)
    xc = te - tr.mean(axis=0)
    print(f"  diagonal Gaussian {(-0.5 * (64 * np.log(2 * np.pi) + np.log(v).sum() + (xc ** 2 / v).sum(axis=1))).mean():.2f}")
    for L in (2, 10, 50):
        Wf, mf, pf_, _ = fit_fa(tr, L, n_iter=200)
        print(f"  L={L:2d} FA {fa_loglik(te, Wf, mf, pf_):.2f}  PPCA {ppca_logpdf(te, *ppca_fit(tr, L)).mean():.2f}")
    rng = np.random.default_rng(3)
    W_true = np.zeros((9, 3))
    W_true[:3, 0] = [1.5, 1.2, 1.0]
    W_true[3:6, 1] = [1.4, 1.1, 0.9]
    W_true[6:, 2] = [1.3, 1.0, 0.8]
    Xs = rng.standard_normal((3000, 3)) @ W_true.T + rng.standard_normal((3000, 9)) * np.sqrt(0.2 + 0.3 * rng.random(9))
    Wa, ma, pa, ta = fit_fa(Xs, 3, n_iter=300, seed=0)
    Wb, _, _, tb = fit_fa(Xs, 3, n_iter=300, seed=1)
    print(f"  two EM seeds: log-lik differs by {abs(ta[-1] - tb[-1]):.1e}, largest loading difference {np.abs(Wa - Wb).max():.2f}")
    print(f"  loadings under 0.2: EM {(np.abs(Wa) < 0.2).mean():.2f}, varimax {(np.abs(varimax(Wa)[0]) < 0.2).mean():.2f}, true {(W_true == 0).mean():.2f}")
    for s in (0, 1):
        r = sensor_noise_experiment(2000, 25.0, s)
        print(f"  one bad sensor, seed {s}: subspace angle PPCA {r['ppca_angle']:.1f} deg, FA {r['fa_angle']:.1f} deg, fitted psi_0 {r['psi_noisy']:.1f}")

    # --- Part 5: EM for PCA and mixtures ---
    print("Part 5: EM for PCA and mixtures")
    for L in (2, 5, 10, 20):
        U = pca_svd(X, L)[1]
        print(f"  digits L={L:2d}: EM iterations to within 1 degree: {iterations_to_angle(X, L, U, 1.0)}")
    Xg = np.random.default_rng(0).standard_normal((1000, 500)) * np.concatenate([10.0 * (1 + 0.5 * np.arange(10)), np.ones(490)])
    print(f"  large eigengap, D=500, L=10: {iterations_to_angle(Xg, 10, pca_svd(Xg, 10)[1], 1.0)}  (the book reports a mean of 3.6)")
    Cu, Cv = curved_data(1000, 0), curved_data(1000, 1)
    for K in (1, 3, 10, 20):
        pi, mus, Ws, ss, trace = fit_mppca(Cu, K, 1, n_iter=100, seed=0)
        print(f"  mixture of PPCA K={K:2d}: train {trace[-1]:.2f}, test {mppca_e_step(Cv, pi, mus, Ws, ss)[1]:.2f}")

    # --- Parts 6, 7: the manifold zoo on the Swiss roll ---
    print("Parts 6-7: Swiss roll, 1000 points, (arc-length, height) Spearman and trustworthiness against the true sheet")
    Xr, U = swiss_roll(1000, 0.0, 0)
    D2 = pairwise_sqdist(U)
    Zi = isomap(Xr, 10, 2)
    row("pca", pca_svd(Xr, 2)[3], U, D2)
    row("isomap k=10", Zi[0], U, D2)
    row("lle k=10", lle(Xr, 10, 2)[0], U, D2)
    row("laplacian k=10 sigma=1", laplacian_eigenmaps(Xr, 10, 1.0, 2)[0], U, D2)
    row("tsne perp=30", fit_tsne(Xr, 30.0, 500, seed=0)[0], U, D2)
    for noise, seed, k in ((0.5, 1, 5), (0.5, 1, 10), (0.5, 2, 15), (1.0, 0, 10)):
        Xn, Un = swiss_roll(1000, noise, seed)
        Zn, keep, _ = isomap(Xn, k, 2)
        s = unroll_score(Zn, Un[keep])
        print(f"  isomap noise={noise} seed={seed} k={k:2d}: false edges {short_circuit_count(pairwise_sqdist(Xn), Un, k):3d} of {1000 * k}, arc {s[0]:.3f}, height {s[1]:.3f}")
    for noise in (0.5, 1.0):
        a, b, c = [], [], []
        for seed in range(3):
            Xn, Un = swiss_roll(1000, noise, seed)
            Zn, keep, _ = isomap(Xn, 10, 2)
            a.append(unroll_score(Zn, Un[keep])[0])
            b.append(unroll_score(lle(Xn, 10, 2)[0], Un)[0])
            c.append(unroll_score(laplacian_eigenmaps(Xn, 10, 1.0, 2)[0], Un)[0])
        print(f"  noise={noise}, 3 seeds, mean arc-length score: isomap {np.mean(a):.2f}, lle {np.mean(b):.2f}, laplacian sigma=1 {np.mean(c):.2f}")
    rng = np.random.default_rng(0)
    Xk = np.concatenate([np.array(c_) + 0.1 * rng.standard_normal((30, 2)) for c_ in ([-0.5, -0.2], [0.0, 0.6], [0.5, 0.0])])
    print(f"  Figure 20.35 kernel PCA eigenvalues: {np.round(kernel_pca(rbf_kernel(Xk, Xk, 0.05), 2)['eigvals'][:4], 2).tolist()} (book 22.56, 20.94, 4.65, 3.99)")

    # --- Part 8: t-SNE ---
    print("Part 8: t-SNE")
    ps = np.random.default_rng(0).permutation(len(X))[:600]
    Xs6, ys6 = X[ps], y[ps]
    D6 = pairwise_sqdist(Xs6)
    Zt, tr_t = fit_tsne(Xs6, 30.0, 500, seed=0)
    Zg, tr_g = fit_tsne(Xs6, 30.0, 500, lr=10.0, seed=0, kernel="gauss")
    print(f"  600 digits, t-SNE:          KL {tr_t[-1]:.3f}, map std {Zt.std():.2f}, trust {trustworthiness(D6, Zt, 10):.3f}, 5-NN {knn_accuracy(Zt, ys6):.3f}")
    print(f"  600 digits, symmetric SNE:  KL {tr_g[-1]:.3f}, map std {Zg.std():.2f}, trust {trustworthiness(D6, Zg, 10):.3f}, 5-NN {knn_accuracy(Zg, ys6):.3f}")
    rng = np.random.default_rng(0)
    A = np.concatenate([np.array(c_) + rng.standard_normal((50, 2)) for c_ in ([0.0, 0.0], [10.0, 0.0], [50.0, 0.0])])
    lab = np.repeat([0, 1, 2], 50)
    for perp in (2.0, 5.0, 30.0, 100.0):
        Zc2 = fit_tsne(A, perp, 750, seed=0)[0]
        c_ = np.stack([Zc2[lab == i].mean(axis=0) for i in range(3)])
        print(f"  three clusters, perplexity {perp:5.1f}: far gap / near gap {np.linalg.norm(c_[1] - c_[2]) / np.linalg.norm(c_[0] - c_[1]):.2f} (true 4.00)")

    # --- Part 9: which map kept what ---
    print("Part 9: all 1797 digits, 64-D to 2-D (k = 10)")
    rep = embedding_report(X, y, 10, 30.0, 500, 0)
    for name, r in rep.items():
        print(f"  {name:10s} trustworthiness {r['trust']:.3f}   5-NN accuracy {r['knn']:.3f}")


def knn_1(A, B, yb, ya):
    return float((yb[np.argmin(pairwise_sqdist(A, B), axis=1)] == ya).mean())


def iterations_to_angle(X, L, U, tol, cap=500):
    Xt = (X - X.mean(axis=0)).T
    W = np.random.default_rng(0).standard_normal((X.shape[1], L))
    for i in range(1, cap + 1):
        Z = np.linalg.solve(W.T @ W, W.T @ Xt)
        W = np.linalg.solve(Z @ Z.T, Z @ Xt.T).T
        if principal_angle(W, U) < tol:
            return i
    return f"more than {cap}"


def row(name, Z, U, D2):
    s = unroll_score(Z, U)
    print(f"  {name:24s} arc {s[0]:.3f}  height {s[1]:.3f}  trust {trustworthiness(D2, Z, 10):.3f}")


if __name__ == "__main__":
    main()

