"""Re-implementation of the pipeline described in the manuscript (Sections 3-4, Table 4)."""
import numpy as np, pandas as pd, warnings
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, roc_auc_score, mean_absolute_error, brier_score_loss
from scipy.optimize import minimize_scalar
warnings.filterwarnings("ignore")

import os
DATA = os.environ.get("APC_DATA", "data")
INTENSITIES = [0.0, 0.25, 0.5, 0.75, 1.0]
GRID = np.round(np.arange(0.10, 0.9001, 0.02), 2)  # 41 values


def load(name):
    if name == "bc":
        d = load_breast_cancer(); return d.data, 1 - d.target   # malignant = 1
    if name == "diab":
        d = load_diabetes(); return d.data, d.target
    if name == "logs":
        df = pd.read_csv(f"{DATA}/delaney-processed.csv")
        cols = ['Minimum Degree', 'Molecular Weight', 'Number of H-Bond Donors', 'Number of Rings',
                'Number of Rotatable Bonds', 'Polar Surface Area']
        return df[cols].values.astype(float), df['measured log solubility in mols per litre'].values


def sigmoid(z): return 1 / (1 + np.exp(-z))
def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))


def ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    for b in range(n_bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


# ---------------- drift ----------------
def drift_clf(Xs, y, kind, iota, rng):
    """returns Xd, yd, keep (indices into the test batch)"""
    n = len(y); keep = np.arange(n); Xd = Xs.copy(); yd = y.copy()
    if kind == "covariate" and iota > 0:
        Xd = Xs + rng.normal(iota, 0.3 * iota, Xs.shape)
    elif kind == "concept" and iota > 0:
        k = int(round(0.4 * iota * n)); idx = rng.choice(n, k, replace=False); yd[idx] = 1 - yd[idx]
    elif kind == "prior" and iota > 0:
        pos = np.where(y == 1)[0]; k = int(round(0.7 * iota * len(pos)))
        drop = rng.choice(pos, k, replace=False); keep = np.setdiff1d(np.arange(n), drop)
        Xd, yd = Xs[keep], y[keep]
    return Xd, yd, keep


def drift_reg(Xs, y, kind, iota, rng, sigma_tr):
    Xd = Xs.copy(); yd = y.copy()
    if kind == "covariate" and iota > 0:
        Xd = Xs + rng.normal(iota, 0.3 * iota, Xs.shape)
    elif kind == "concept" and iota > 0:
        yd = y + rng.normal(0, 2 * iota * sigma_tr, len(y))
    return Xd, yd


# ---------------- classification correctors ----------------
def ost(yw, pw, grid=GRID):
    if len(np.unique(yw)) < 2: return 0.5
    f = np.array([f1_score(yw, (pw >= t).astype(int), zero_division=0) for t in grid])
    best = np.flatnonzero(np.isclose(f, f.max()))
    return grid[best[np.argmin(np.abs(grid[best] - 0.5))]]


def gsp(p, t, iota):
    a = 0.25 + 0.15 * iota
    return np.clip(p + a * (t - 0.5) * (1 - 2 * np.abs(p - 0.5)), 0, 1)


def ost_uncapped(yw, zw):
    if len(np.unique(yw)) < 2: return 0.0
    s = np.sort(np.unique(zw)); cands = np.concatenate([[s[0] - 1], (s[:-1] + s[1:]) / 2, [s[-1] + 1]])
    f = np.array([f1_score(yw, (zw >= t).astype(int), zero_division=0) for t in cands])
    best = np.flatnonzero(np.isclose(f, f.max()))
    return cands[best[np.argmin(np.abs(cands[best]))]]


def platt(yw, zw, ze):
    if len(np.unique(yw)) < 2: return sigmoid(ze)
    m = LogisticRegression(C=1.0, max_iter=1000).fit(zw[:, None], yw)
    return m.predict_proba(ze[:, None])[:, 1]


def intercept(yw, zw, ze):
    if len(np.unique(yw)) < 2: return sigmoid(ze)
    nll = lambda c: -np.sum(yw * np.log(sigmoid(zw + c) + 1e-12) + (1 - yw) * np.log(1 - sigmoid(zw + c) + 1e-12))
    c = minimize_scalar(nll, bounds=(-30, 30), method="bounded").x
    return sigmoid(ze + c)


def isotonic(yw, pw, pe):
    if len(np.unique(yw)) < 2: return pe
    return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(pw, yw).predict(pe)


def temperature(yw, zw, ze):
    if len(np.unique(yw)) < 2: return sigmoid(ze)
    nll = lambda T: -np.sum(yw * np.log(sigmoid(zw / T) + 1e-12) + (1 - yw) * np.log(1 - sigmoid(zw / T) + 1e-12))
    T = minimize_scalar(nll, bounds=(0.05, 10), method="bounded").x
    return sigmoid(ze / T)


def em_prior(p, prior_src, iters=100):
    pi = prior_src
    for _ in range(iters):
        w1, w0 = pi / prior_src, (1 - pi) / (1 - prior_src)
        q = w1 * p / (w1 * p + w0 * (1 - p)); pi = np.clip(q.mean(), 0.001, 0.999)
    w1, w0 = pi / prior_src, (1 - pi) / (1 - prior_src)
    return w1 * p / (w1 * p + w0 * (1 - p))


def restandardise(Xbatch):
    sd = Xbatch.std(0); sd[sd == 0] = 1
    return (Xbatch - Xbatch.mean(0)) / sd


# ---------------- intensity proxy ----------------
def iota_proxy(Xbatch_std):
    """Data-driven proxy: average displacement of the standardised feature means of the
    unlabelled drifted batch (the simulated covariate shift adds mean iota to each feature)."""
    return float(np.clip(np.mean(Xbatch_std.mean(0)), 0, 1))


def boot_ci(v, B=2000, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(v)
    m = rng.choice(v, (B, len(v)), replace=True).mean(1)
    return np.percentile(m, [2.5, 97.5])
