# =============================================================================
# Adaptive post-hoc correction under distribution shift -- unified pipeline
# IJIES Paper ID 20266614
#
# ONE script produces EVERY number, table and figure of the manuscript.
#   python experiments.py            (about 10-15 min on a laptop CPU)
# Outputs: results/*.csv, figures/*.png
#
# Protocol (identical for all datasets / methods):
#   * R = 30 repetitions. Repetition r uses seed r for (i) the 80/20 split,
#     (ii) the drift realisation, (iii) the order of the drifted batch and
#     hence the calibration window.
#   * The drifted test batch is split ONCE into a calibration window (first
#     n_c = max(10, floor(0.15 n)) rows) and an evaluation subset (the rest).
#     Clean, drifted and corrected scores are ALWAYS computed on the same
#     evaluation rows. Window rows are never used for scoring.
#   * No hyper-parameter is tuned on evaluation rows. All settings are fixed
#     a priori (Table of settings in the paper) or tuned on training folds.
# =============================================================================
import os, warnings, hashlib, json
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import wilcoxon, ks_2samp
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              RandomForestRegressor, GradientBoostingRegressor)
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (f1_score, roc_auc_score, brier_score_loss, accuracy_score,
                             mean_absolute_error, mean_squared_error, r2_score)
warnings.filterwarnings("ignore")

R = 30
INTENSITIES = [0.0, 0.25, 0.50, 0.75, 1.00]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
EPS = 1e-6

# ----------------------------------------------------------------------------- data
def load_bc():
    d = load_breast_cancer()
    return d.data, 1 - d.target          # positive class (1) = malignant

def load_diab():
    d = load_diabetes(); return d.data, d.target

LOGS_COLS = ['Minimum Degree', 'Molecular Weight', 'Number of H-Bond Donors',
             'Number of Rings', 'Number of Rotatable Bonds', 'Polar Surface Area']
LOGS_FILE = os.path.join(HERE, "data", "delaney-processed.csv")
def load_logs():
    df = pd.read_csv(LOGS_FILE)
    return df[LOGS_COLS].values.astype(float), df['measured log solubility in mols per litre'].values

# ----------------------------------------------------------------------------- drift
def covariate_shift(X, y, i, rng):
    return X + rng.normal(i, 0.3 * i, size=X.shape), y.copy(), np.arange(len(X))

def concept_clf(X, y, i, rng):
    y2 = y.copy(); k = int(0.4 * i * len(y))
    if k: idx = rng.choice(len(y), k, replace=False); y2[idx] = 1 - y2[idx]
    return X.copy(), y2, np.arange(len(X))

def prior_clf(X, y, i, rng):
    pos = np.where(y == 1)[0]; k = int(0.7 * i * len(pos))
    keep = np.ones(len(y), bool)
    if k: keep[rng.choice(pos, k, replace=False)] = False
    return X[keep], y[keep], np.where(keep)[0]

def concept_reg(X, y, i, rng, sigma):
    return X.copy(), y + rng.normal(0, 2 * sigma * i, size=y.shape) if i > 0 else y.copy(), np.arange(len(X))

# ----------------------------------------------------------------------------- helpers
def logit(p):
    p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def sigm(z): return 1 / (1 + np.exp(-z))

def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    idx = np.minimum((p * bins).astype(int), bins - 1)
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e

GRID = np.round(np.arange(0.10, 0.901, 0.02), 2)          # 41 thresholds
def ost_threshold(y, p):
    """F1-maximising threshold on the window; ties -> threshold closest to 0.5."""
    if y.sum() == 0 or y.sum() == len(y): return 0.5
    f = np.array([f1_score(y, (p >= t).astype(int), zero_division=0) for t in GRID])
    best = np.where(f >= f.max() - 1e-12)[0]
    return float(GRID[best[np.argmin(np.abs(GRID[best] - 0.5))]])

def gsp(p, t, i):
    a = 0.25 + 0.15 * i
    return np.clip(p + a * (t - 0.5) * (1 - 2 * np.abs(p - 0.5)), 0, 1)

# ----------------------------------------------------------------------------- classification methods
# every method returns (hard prediction, probability) on the evaluation rows
def m_uncorrected(c): return (c['p_ev'] >= 0.5).astype(int), c['p_ev']

def m_proposed(c):
    t = ost_threshold(c['y_w'], c['p_w']); p = gsp(c['p_ev'], t, c['i'])
    return (p >= t).astype(int), p

def m_ost_only(c):
    t = ost_threshold(c['y_w'], c['p_w']); return (c['p_ev'] >= t).astype(int), c['p_ev']

def m_ost_unbounded(c):
    """Ablation: same F1 search, but candidate thresholds are the mid-points of the sorted
    window scores in logit space, i.e. no [0.1, 0.9] cap."""
    y, z = c['y_w'], np.sort(logit(c['p_w']))
    if y.sum() == 0 or y.sum() == len(y): return m_uncorrected(c)
    cand = np.r_[z[0] - 1, (z[1:] + z[:-1]) / 2, z[-1] + 1, 0.0]; zw = logit(c['p_w'])
    f = np.array([f1_score(y, (zw >= t).astype(int), zero_division=0) for t in cand])
    best = np.where(f >= f.max() - 1e-12)[0]; t = cand[best[np.argmin(np.abs(cand[best]))]]
    return (logit(c['p_ev']) >= t).astype(int), c['p_ev']

def _two_classes(y): return 0 < y.sum() < len(y)

def m_platt(c):
    if not _two_classes(c['y_w']): return m_uncorrected(c)
    lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000).fit(logit(c['p_w'])[:, None], c['y_w'])
    p = lr.predict_proba(logit(c['p_ev'])[:, None])[:, 1]; return (p >= 0.5).astype(int), p

def m_intercept(c):
    """Recalibration-in-the-large: one free parameter b, logit' = logit + b (maximum likelihood)."""
    if not _two_classes(c['y_w']): return m_uncorrected(c)
    z = logit(c['p_w']); y = c['y_w']
    nll = lambda b: -np.mean(y * np.log(sigm(z + b) + EPS) + (1 - y) * np.log(1 - sigm(z + b) + EPS))
    b = minimize_scalar(nll, bounds=(-30, 30), method='bounded').x
    p = sigm(logit(c['p_ev']) + b); return (p >= 0.5).astype(int), p

def m_isotonic(c):
    if not _two_classes(c['y_w']): return m_uncorrected(c)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(c['p_w'], c['y_w'])
    p = iso.predict(c['p_ev']); return (p >= 0.5).astype(int), p

def m_temperature(c):
    if not _two_classes(c['y_w']): return m_uncorrected(c)
    z = logit(c['p_w']); y = c['y_w']
    nll = lambda T: -np.mean(y * np.log(sigm(z / T) + EPS) + (1 - y) * np.log(1 - sigm(z / T) + EPS))
    T = minimize_scalar(nll, bounds=(0.05, 10), method='bounded').x
    p = sigm(logit(c['p_ev']) / T); return (p >= 0.5).astype(int), p

def m_retrain(c):
    if not _two_classes(c['y_w']): return m_uncorrected(c)
    m = LogisticRegression(C=1.0, max_iter=2000).fit(c['X_w'], c['y_w'])
    p = m.predict_proba(c['X_ev'])[:, 1]; return (p >= 0.5).astype(int), p

# ---- unlabelled-target methods (see all drifted features, no target label) ----
def m_restandardise(c):
    """Feature re-standardisation with target statistics (test-time normalisation)."""
    Xall = np.vstack([c['X_w'], c['X_ev']])
    mu, sd = Xall.mean(0), Xall.std(0) + EPS
    p = c['model'].predict_proba((c['X_ev'] - mu) / sd)[:, 1]; return (p >= 0.5).astype(int), p

def m_mlls_em(c, iters=100):
    """Saerens et al. (2002) EM re-estimation of the class prior."""
    pall = np.concatenate([c['p_w'], c['p_ev']]); ps = c['prior_s']; pt = ps
    for _ in range(iters):
        num = pall * pt / ps; den = num + (1 - pall) * (1 - pt) / (1 - ps)
        pt = float(np.clip((num / den).mean(), 1e-3, 1 - 1e-3))
    num = c['p_ev'] * pt / ps; p = num / (num + (1 - c['p_ev']) * (1 - pt) / (1 - ps))
    return (p >= 0.5).astype(int), p

def m_adaptable_ldh(c, alpha=0.1, bs=64, ql=0.25, qh=0.75):
    """Label-distribution handler of AdapTable (Kim et al., 2024), Eqs. 3-6, processed in
    batches of 64. The GNN shift-aware calibrator is not used (no per-sample learned
    temperature for a linear base model); the margin is computed from the raw output."""
    ps = np.array([1 - c['prior_s'], c['prior_s']])
    rho = ps.max() / ps.min(); T = 1.5 * rho / (rho - 1 + 1e-6)
    z_all = logit(np.concatenate([c['p_w'], c['p_ev']]))
    oe = np.array([0.5, 0.5]); out = np.zeros(len(z_all))
    for s in range(0, len(z_all), bs):
        z = z_all[s:s + bs]; p1 = sigm(z); P = np.c_[1 - p1, p1]
        delta = 1 / (np.abs(P[:, 1] - P[:, 0]) + EPS)
        Ti = np.ones(len(z)); Ti[delta >= np.quantile(delta, qh)] = T; Ti[delta <= np.quantile(delta, ql)] = 1 / T
        p1c = sigm(z / Ti); Pc = np.c_[1 - p1c, p1c]
        de = P / ps; de /= de.sum(1, keepdims=True)
        pt = (1 - alpha) * de.mean(0) + alpha * oe
        adj = Pc * pt / ps; adj /= adj.sum(1, keepdims=True)
        Pbar = (Pc + adj) / 2
        oe = (1 - alpha) * Pbar.mean(0) + alpha * oe
        out[s:s + bs] = Pbar[:, 1]
    p = out[len(c['p_w']):]; return (p >= 0.5).astype(int), p

def _bbse_weights(c):
    ys, hs = c['y_src'], (c['p_src'] >= 0.5).astype(int)
    C = np.zeros((2, 2))
    for a in (0, 1):
        for b in (0, 1): C[a, b] = np.mean((hs == a) & (ys == b))
    ht = (np.concatenate([c['p_w'], c['p_ev']]) >= 0.5).astype(int)
    mu = np.array([np.mean(ht == 0), np.mean(ht == 1)])
    try: w = np.linalg.solve(C, mu)
    except np.linalg.LinAlgError: w = np.ones(2)
    return np.clip(w, 0.05, 20)

def m_lascal(c, bins=15):
    """LaSCal (Popordanoska et al., 2024): temperature chosen to minimise the importance-weighted
    class-wise L2 calibration-error estimator on UNLABELLED target scores; labelled source scores
    come from out-of-fold training predictions; weights from BBSE; binning kernel, equal-mass bins."""
    w = _bbse_weights(c); zs = logit(c['p_src']); ys = c['y_src']
    zt = logit(np.concatenate([c['p_w'], c['p_ev']])); n, m = len(zs), len(zt)
    def est(T):
        tot = 0.0
        for cls in (0, 1):
            s = sigm(zs / T) if cls else 1 - sigm(zs / T); t = sigm(zt / T) if cls else 1 - sigm(zt / T)
            yc = (ys == cls).astype(float)
            edges = np.quantile(t, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -1, 2
            bt = np.clip(np.searchsorted(edges, t, side='right') - 1, 0, bins - 1)
            bs_ = np.clip(np.searchsorted(edges, s, side='right') - 1, 0, bins - 1)
            for b in range(bins):
                mt = bt == b
                if mt.sum() < 2: continue
                num = w[cls] * yc[bs_ == b].sum() / n; den = (mt.sum() - 1) / (m - 1)
                tot += np.sum((num / den - t[mt]) ** 2)
        return tot / (2 * m)
    Ts = np.exp(np.linspace(np.log(0.25), np.log(4), 41)); T = Ts[np.argmin([est(T) for T in Ts])]
    p = sigm(logit(c['p_ev']) / T); return (p >= 0.5).astype(int), p

# ---- model-update method (unlabelled target, parameters ARE changed) ----
def m_ftat(c, lrs=(1e-3, 1e-2, 1e-1), bs=32, alpha=0.1, ent_frac=0.4, beta=0.1):
    """FTAT-style test-time adaptation (after Zhou et al., AAAI 2025), re-implemented for a
    logistic model: weighted entropy minimisation on unlabelled batches (local-consistency
    weights), an ensemble of 3 learning rates weighted by 1 - batch entropy, and the
    confident label-distribution optimiser. Hyper-parameters are ours (not given in the paper)."""
    w0, b0 = c['model'].coef_[0].copy(), float(c['model'].intercept_[0])
    P0 = np.array([1 - c['prior_s'], c['prior_s']]); Phat = P0.copy()
    params = [[w0.copy(), b0] for _ in lrs]
    Xall = np.vstack([c['X_w'], c['X_ev']]); out = np.zeros(len(Xall))
    H = lambda p: -(p * np.log(p + EPS) + (1 - p) * np.log(1 - p + EPS))
    for s in range(0, len(Xall), bs):
        X = Xall[s:s + bs]; preds, losses = [], []
        D = np.linalg.norm(X[:, None] - X[None], axis=2); nb = D < D[np.triu_indices(len(X), 1)].mean() if len(X) > 1 else D < 1
        for k, lr in enumerate(lrs):
            w, b = params[k]; z = X @ w + b; p = sigm(z)
            preds.append(p); losses.append(H(p).mean() / np.log(2))
            nbmean = (nb * p[None]).sum(1) / nb.sum(1)
            W = np.abs(2 * p - 1) * (np.abs(p - nbmean) < beta)
            g = -(z * p * (1 - p)) * W                       # d entropy / d z, weighted
            params[k] = [w - lr * (g[:, None] * X).mean(0), b - lr * g.mean()]
        wts = 1 - np.array(losses); wts = wts / wts.sum() if wts.sum() > 0 else np.ones(len(lrs)) / len(lrs)
        p = np.sum([a * q for a, q in zip(wts, preds)], axis=0)
        P = np.c_[1 - p, p] * Phat / P0; P /= P.sum(1, keepdims=True)
        conf = H(P[:, 1]) < ent_frac * np.log(2)
        if conf.sum() >= 2:
            Pt = P[conf].mean(0); Cm = np.eye(2)
            for k in (0, 1):
                mk = P.argmax(1) == k
                if mk.any(): Cm[k] = P[mk].mean(0)
            try: deb = np.clip(np.linalg.solve(Cm.T, Pt), 1e-3, None); deb /= deb.sum()
            except np.linalg.LinAlgError: deb = Pt
            Phat = (1 - alpha) * Phat + alpha * deb; Phat /= Phat.sum()
        out[s:s + bs] = P[:, 1]
    p = out[len(c['X_w']):]; return (p >= 0.5).astype(int), p

CLF_METHODS = [  # (name, function, information setting)
    ('Uncorrected', m_uncorrected, '-'),
    ('OST+GSP (proposed)', m_proposed, 'L'), ('OST only', m_ost_only, 'L'), ('OST, uncapped grid', m_ost_unbounded, 'L'),
    ('Platt scaling', m_platt, 'L'), ('Intercept shift', m_intercept, 'L'),
    ('Isotonic regression', m_isotonic, 'L'), ('Temperature scaling', m_temperature, 'L'),
    ('Retrain on window', m_retrain, 'L'),
    ('Re-standardisation', m_restandardise, 'U'), ('MLLS-EM', m_mlls_em, 'U'),
    ('AdapTable-LDH', m_adaptable_ldh, 'U'), ('LaSCal-TS', m_lascal, 'U'),
    ('FTAT-style TTA', m_ftat, 'M')]

# ----------------------------------------------------------------------------- classification runs
def clf_context(seed):
    X, y = load_bc()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    model = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    p_src = cross_val_predict(LogisticRegression(max_iter=2000), Xtr, ytr, method='predict_proba',
                              cv=StratifiedKFold(5, shuffle=True, random_state=seed))[:, 1]
    return dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, model=model, p_src=p_src)

def run_clf(window_frac=0.15, intensities=INTENSITIES, tag='main'):
    rows = []
    for seed in range(R):
        b = clf_context(seed); model = b['model']
        for dname, dfn in [('covariate', covariate_shift), ('concept', concept_clf), ('prior', prior_clf)]:
            for i in intensities:
                rng = np.random.default_rng(1000 * seed + int(100 * i))
                Xd, yd, oi = dfn(b['Xte'], b['yte'], i, rng)
                n_c = max(10, int(window_frac * len(Xd)))
                w, ev = np.arange(n_c), np.arange(n_c, len(Xd))
                c = dict(model=model, i=i, X_w=Xd[w], y_w=yd[w], X_ev=Xd[ev], y_ev=yd[ev],
                         p_w=model.predict_proba(Xd[w])[:, 1], p_ev=model.predict_proba(Xd[ev])[:, 1],
                         prior_s=b['ytr'].mean(), p_src=b['p_src'], y_src=b['ytr'])
                p_clean = model.predict_proba(b['Xte'][oi[ev]])[:, 1]; y_clean = b['yte'][oi[ev]]
                f1_clean = f1_score(y_clean, (p_clean >= 0.5).astype(int), zero_division=0)
                auc_clean = roc_auc_score(y_clean, p_clean)
                f1_d = f1_score(c['y_ev'], (c['p_ev'] >= 0.5).astype(int), zero_division=0)
                for name, fn, info in CLF_METHODS:
                    pred, p = fn(c)
                    rows.append(dict(seed=seed, drift=dname, intensity=i, method=name, info=info,
                                     n_window=n_c, n_eval=len(ev), window_pos=int(c['y_w'].sum()),
                                     f1_clean=f1_clean, auc_clean=auc_clean, f1_drift=f1_d,
                                     f1=f1_score(c['y_ev'], pred, zero_division=0),
                                     auc=roc_auc_score(c['y_ev'], p) if _two_classes(c['y_ev']) else np.nan,
                                     ece=ece(c['y_ev'], p), brier=brier_score_loss(c['y_ev'], p),
                                     t_star=ost_threshold(c['y_w'], c['p_w']) if name == 'OST only' else np.nan))
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/clf_{tag}_raw.csv", index=False); return df

# ----------------------------------------------------------------------------- regression
def reg_context(X, y, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    model = Ridge(alpha=1.0).fit(Xtr, ytr)
    res = ytr - model.predict(Xtr); hard = np.abs(res) > np.percentile(np.abs(res), 55)
    corrector = Ridge(alpha=1.0).fit(Xtr[hard], res[hard])
    return dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, model=model, corrector=corrector)

def reg_methods(c):
    f_w, f_ev = c['model'].predict(c['X_w']), c['model'].predict(c['X_ev'])
    out = {'Uncorrected': f_ev,
           'Residual Ridge (proposed)': f_ev + np.clip(0.9 * c['i'], 0, 0.75) * c['corrector'].predict(c['X_ev']),
           'Mean-bias correction': f_ev + (c['y_w'] - f_w).mean()}
    A = np.c_[np.ones(len(f_w)), f_w]; coef = np.linalg.lstsq(A, c['y_w'], rcond=None)[0]
    out['Linear recalibration'] = coef[0] + coef[1] * f_ev
    out['Retrain on window'] = Ridge(alpha=10.0).fit(c['X_w'], c['y_w']).predict(c['X_ev'])
    Xall = np.vstack([c['X_w'], c['X_ev']])
    out['Re-standardisation'] = c['model'].predict((c['X_ev'] - Xall.mean(0)) / (Xall.std(0) + EPS))
    return out
REG_INFO = {'Uncorrected': '-', 'Residual Ridge (proposed)': 'S', 'Mean-bias correction': 'L',
            'Linear recalibration': 'L', 'Retrain on window': 'L', 'Re-standardisation': 'U'}

def run_reg(name, X, y, window_frac=0.15, intensities=INTENSITIES, tag='main'):
    rows = []
    for seed in range(R):
        b = reg_context(X, y, seed); sigma = b['ytr'].std()
        for dname in ('covariate', 'concept'):
            for i in intensities:
                rng = np.random.default_rng(1000 * seed + int(100 * i))
                if dname == 'covariate': Xd, yd, oi = covariate_shift(b['Xte'], b['yte'], i, rng)
                else: Xd, yd, oi = concept_reg(b['Xte'], b['yte'], i, rng, sigma)
                n_c = max(10, int(window_frac * len(Xd))); w, ev = np.arange(n_c), np.arange(n_c, len(Xd))
                c = dict(model=b['model'], corrector=b['corrector'], i=i, X_w=Xd[w], y_w=yd[w], X_ev=Xd[ev], y_ev=yd[ev])
                mae_clean = mean_absolute_error(b['yte'][ev], b['model'].predict(b['Xte'][ev]))
                preds = reg_methods(c); mae_d = mean_absolute_error(c['y_ev'], preds['Uncorrected'])
                for m, p in preds.items():
                    rows.append(dict(dataset=name, seed=seed, drift=dname, intensity=i, method=m, info=REG_INFO[m],
                                     n_window=n_c, n_eval=len(ev), mae_clean=mae_clean, mae_drift=mae_d,
                                     mae=mean_absolute_error(c['y_ev'], p)))
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/reg_{name}_{tag}_raw.csv", index=False); return df

# ----------------------------------------------------------------------------- statistics
def ci95(x, B=2000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    bs = rng.choice(x, (B, len(x))).mean(1); return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def recovery_ratio(clean, drift, corr, higher=True, B=2000, seed=0):
    """Pooled normalised recovery ratio = mean gain / mean loss, bootstrap CI over repetitions.
    Returned as 'n.d.' when the mean drift-induced loss is not clearly positive."""
    clean, drift, corr = map(np.asarray, (clean, drift, corr)); s = 1 if higher else -1
    loss, gain = s * (clean - drift), s * (corr - drift)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(loss), (B, len(loss)))
    lb = loss[idx].mean(1); gb = gain[idx].mean(1)
    lo_loss = np.percentile(lb, 2.5)
    if lo_loss <= 0: return np.nan, np.nan, np.nan
    r = gb / lb; return gain.mean() / loss.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5)

def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); adj = np.empty(len(p)); run = 0
    for k, j in enumerate(order):
        run = max(run, (len(p) - k) * p[j]); adj[j] = min(1, run)
    return adj

def summarise_clf(df, tag='main'):
    rows = []
    for (d, i), g in df.groupby(['drift', 'intensity']):
        piv = g.pivot(index='seed', columns='method', values='f1'); ref = piv['OST+GSP (proposed)']
        ranks = piv.drop(columns='Uncorrected').rank(axis=1, ascending=False)
        pv = {}
        for m in piv.columns:
            if m == 'OST+GSP (proposed)': continue
            diff = ref - piv[m]; pv[m] = wilcoxon(ref, piv[m]).pvalue if np.any(diff != 0) else 1.0
        adj = dict(zip(pv, holm(list(pv.values()))))
        for m, _, info in CLF_METHODS:
            gm = g[g.method == m].sort_values('seed')
            f, lo, hi = ci95(gm.f1); dm, dlo, dhi = ci95(gm.f1.values - gm.f1_drift.values)
            rr, rlo, rhi = recovery_ratio(gm.f1_clean, gm.f1_drift, gm.f1)
            pm, plo, phi = ci95(ref.values - gm.f1.values) if m != 'OST+GSP (proposed)' else (0, 0, 0)
            rows.append(dict(drift=d, intensity=i, method=m, info=info, f1_clean=gm.f1_clean.mean(),
                             f1_drift=gm.f1_drift.mean(), f1=f, f1_lo=lo, f1_hi=hi, gain_pp=100 * dm,
                             gain_lo=100 * dlo, gain_hi=100 * dhi, nrr=rr, nrr_lo=rlo, nrr_hi=rhi,
                             ece=gm.ece.mean(), brier=gm.brier.mean(), auc=gm.auc.mean(),
                             prop_minus_m_pp=100 * pm, pm_lo=100 * plo, pm_hi=100 * phi,
                             p_holm=adj.get(m, np.nan),
                             mean_rank=ranks[m].mean() if m in ranks else np.nan,
                             rank_sd=ranks[m].std() if m in ranks else np.nan))
    s = pd.DataFrame(rows); s.to_csv(f"{OUT}/clf_{tag}_summary.csv", index=False); return s

def summarise_reg(df, name, tag='main'):
    rows = []
    for (d, i), g in df.groupby(['drift', 'intensity']):
        piv = g.pivot(index='seed', columns='method', values='mae'); ref = piv['Residual Ridge (proposed)']
        ranks = piv.drop(columns='Uncorrected').rank(axis=1)
        pv = {m: (wilcoxon(ref, piv[m]).pvalue if np.any(ref != piv[m]) else 1.0) for m in piv.columns if m != 'Residual Ridge (proposed)'}
        adj = dict(zip(pv, holm(list(pv.values()))))
        for m in REG_INFO:
            gm = g[g.method == m].sort_values('seed'); f, lo, hi = ci95(gm.mae)
            rr, rlo, rhi = recovery_ratio(gm.mae_clean, gm.mae_drift, gm.mae, higher=False)
            dm, dlo, dhi = ci95(gm.mae_drift.values - gm.mae.values)
            rows.append(dict(dataset=name, drift=d, intensity=i, method=m, info=REG_INFO[m], mae_clean=gm.mae_clean.mean(),
                             mae_drift=gm.mae_drift.mean(), mae=f, mae_lo=lo, mae_hi=hi, reduction=dm, red_lo=dlo, red_hi=dhi,
                             nrr=rr, nrr_lo=rlo, nrr_hi=rhi, p_holm=adj.get(m, np.nan),
                             mean_rank=ranks[m].mean() if m in ranks else np.nan))
    s = pd.DataFrame(rows); s.to_csv(f"{OUT}/reg_{name}_{tag}_summary.csv", index=False); return s

# ----------------------------------------------------------------------------- baselines table (Table 2)
def baseline_table():
    rows = []
    for seed in range(R):
        X, y = load_bc(); Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        for n, m in [('Logistic Regression', LogisticRegression(max_iter=2000)), ('SVM (RBF)', SVC(probability=True, random_state=seed)),
                     ('Random Forest', RandomForestClassifier(100, random_state=seed)), ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=seed))]:
            m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]; h = (p >= 0.5).astype(int)
            rows.append(dict(dataset='Breast Cancer', model=n, seed=seed, m1=accuracy_score(yte, h), m2=f1_score(yte, h), m3=roc_auc_score(yte, p)))
        for dn, (X, y) in [('Diabetes', load_diab()), ('LogS', load_logs())]:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
            sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            for n, m in [('Ridge', Ridge(alpha=1.0)), ('Random Forest', RandomForestRegressor(100, random_state=seed)), ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=seed))]:
                m.fit(Xtr, ytr); p = m.predict(Xte)
                rows.append(dict(dataset=dn, model=n, seed=seed, m1=mean_absolute_error(yte, p), m2=np.sqrt(mean_squared_error(yte, p)), m3=r2_score(yte, p)))
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/baseline_models_raw.csv", index=False)
    s = df.groupby(['dataset', 'model'])[['m1', 'm2', 'm3']].agg(['mean', 'std']); s.to_csv(f"{OUT}/baseline_models_summary.csv"); return s

# ----------------------------------------------------------------------------- AUC vs F1 mechanism (same rows, same probabilities)
def auc_f1_check(df):
    g = df[(df.method == 'Uncorrected') & (df.drift == 'covariate')]
    s = g.groupby('intensity').agg(auc=('auc', 'mean'), f1=('f1', 'mean'), ece=('ece', 'mean')).reset_index()
    t = df[(df.method == 'OST only') & (df.drift == 'covariate')].groupby('intensity').t_star.mean().values
    s['mean_t_star'] = t
    rows = []
    for seed in range(R):
        b = clf_context(seed); w = b['model'].coef_[0]
        z = b['Xte'] @ w + b['model'].intercept_[0]
        rows.append(dict(seed=seed, sum_w=w.sum(), norm_w=np.linalg.norm(w), sd_logit=z.std()))
    m = pd.DataFrame(rows).mean()
    s['pred_logit_shift'] = s.intensity * m.sum_w; s['pred_noise_sd'] = 0.3 * s.intensity * m.norm_w; s['clean_logit_sd'] = m.sd_logit
    s.to_csv(f"{OUT}/auc_f1_mechanism.csv", index=False); return s

if __name__ == '__main__':
    print(hashlib.sha256(open(LOGS_FILE, 'rb').read()).hexdigest(), 'delaney-processed.csv')
    print(baseline_table().round(4))
    dc = run_clf(); sc_ = summarise_clf(dc); print(auc_f1_check(dc).round(3))
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
    print(sc_[sc_.intensity == 0.5][['drift', 'method', 'info', 'f1_clean', 'f1_drift', 'f1', 'f1_lo', 'f1_hi', 'gain_pp', 'nrr', 'ece', 'brier', 'p_holm', 'mean_rank']].round(3).to_string())
    for nm, (X, y) in [('Diabetes', load_diab()), ('LogS', load_logs())]:
        s = summarise_reg(run_reg(nm, X, y), nm)
        print(s[s.intensity == 0.5][['drift', 'method', 'mae_clean', 'mae_drift', 'mae', 'mae_lo', 'mae_hi', 'nrr', 'p_holm', 'mean_rank']].round(3).to_string())
