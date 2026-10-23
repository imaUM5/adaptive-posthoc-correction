import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from core import *
from sklearn.ensemble import RandomForestClassifier


def make_model(kind, seed):
    if kind == "lr": return LogisticRegression(C=1.0, max_iter=5000)
    if kind == "rf": return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1)


def scores(model, X):
    p = model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"): z = model.decision_function(X)
    else: z = logit(p, 1e-3)
    return p, z


def run_rep(X, y, r, kind, iota, base="lr", iota_mode="oracle", iota_bias=0.0, iota_sd=0.0,
            window_label_noise=0.0, delay=0, methods="all"):
    """One repetition. iota_mode: oracle | biased | proxy. delay: number of drifted rows that pass
    before window labels become usable (their rows are scored uncorrected and excluded from window)."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=r)
    sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    m = make_model(base, r).fit(Xtr_s, ytr)
    rng = np.random.default_rng(r)
    Xd, yd, keep = drift_clf(Xte_s, yte, kind, iota, rng)
    order = rng.permutation(len(yd))
    Xd, yd, keep = Xd[order], yd[order], keep[order]
    Xclean, yclean = Xte_s[keep], yte[keep]
    n = len(yd); nc = max(10, int(np.floor(0.15 * n)))
    w = np.arange(delay, delay + nc); e = np.arange(delay + nc, n)
    pd_, zd = scores(m, Xd); pc, zc = scores(m, Xclean)
    yw = yd[w].copy()
    if window_label_noise > 0:
        nrng = np.random.default_rng(10_000 + r); k = int(round(window_label_noise * len(yw)))
        idx = nrng.choice(len(yw), k, replace=False); yw[idx] = 1 - yw[idx]
    # intensity used by GSP
    if iota_mode == "oracle": iu = iota
    elif iota_mode == "biased":
        nrng = np.random.default_rng(20_000 + r)
        iu = float(np.clip(iota * (1 + iota_bias) + nrng.normal(0, iota_sd), 0, 1))
    elif iota_mode == "proxy": iu = iota_proxy(Xd)
    ye = yd[e]
    out = {"rep": r, "kind": kind, "iota": iota, "iota_used": iu, "n_eval": len(e), "n_win": len(w),
           "win_pos": int(yw.sum())}
    out["clean"] = f1_score(yclean[e], (pc[e] >= .5).astype(int), zero_division=0)
    res = {}
    res["Uncorrected"] = pd_[e]
    t = ost(yw, pd_[w]); g = gsp(pd_[e], t, iu)
    res["OST+GSP"] = (g, t); out["t_star"] = t
    if methods == "all" or "ost" in methods:
        res["OST only"] = (pd_[e], t)
        tu = ost_uncapped(yw, zd[w]); res["OST uncapped"] = (sigmoid(zd[e]), sigmoid(tu))
    if methods == "all":
        res["Platt"] = platt(yw, zd[w], zd[e])
        res["Intercept"] = intercept(yw, zd[w], zd[e])
        res["Isotonic"] = isotonic(yw, pd_[w], pd_[e])
        res["Temperature"] = temperature(yw, zd[w], zd[e])
        if len(np.unique(yw)) < 2: res["Retrain"] = pd_[e]
        else:
            mr = (LogisticRegression(C=1.0, max_iter=2000) if base == "lr" else make_model(base, r)).fit(Xd[w], yw)
            res["Retrain"] = mr.predict_proba(Xd[e])[:, 1]
        Xr = restandardise(Xd); res["Re-standardisation"] = m.predict_proba(Xr[e])[:, 1]
        res["MLLS-EM"] = em_prior(pd_, ytr.mean())[e]
    for k, v in res.items():
        p, thr = (v if isinstance(v, tuple) else (v, 0.5))
        yh = (p >= thr).astype(int)
        if delay > 0:
            pre = np.arange(0, delay)
            out[k] = f1_score(np.r_[yd[pre], ye], np.r_[(pd_[pre] >= .5).astype(int), yh], zero_division=0)
        else:
            out[k] = f1_score(ye, yh, zero_division=0)
        out[k + "|ece"] = ece(ye, np.clip(p, 0, 1)); out[k + "|brier"] = brier_score_loss(ye, np.clip(p, 0, 1))
    out["auc_drift"] = roc_auc_score(ye, pd_[e]) if len(np.unique(ye)) > 1 else np.nan
    return out


def run(kind, iota, reps=30, **kw):
    X, y = load("bc")
    return pd.DataFrame([run_rep(X, y, r, kind, iota, **kw) for r in range(reps)])


if __name__ == "__main__":
    rows = []
    for kind in ["covariate", "concept", "prior"]:
        df = run(kind, 0.5)
        print(kind, "clean %.3f" % df.clean.mean(), {k: round(df[k].mean(), 3) for k in
              ["Uncorrected", "OST+GSP", "OST only", "OST uncapped", "Platt", "Intercept", "Isotonic", "Temperature", "Retrain", "Re-standardisation", "MLLS-EM"]})
        print("  ece", {k: round(df[k + "|ece"].mean(), 3) for k in ["Uncorrected", "OST+GSP", "Platt"]}, "t*", df.t_star.mean().round(2))
