"""Detector calibration, family-wise false alarms, and independent (without-replacement) streams."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from core import *
from sklearn.model_selection import cross_val_predict
from scipy.stats import ks_2samp, beta
OUT = "out"


def ph_path(e, delta=0.05):
    e = np.asarray(e, float); mbar = np.cumsum(e) / np.arange(1, len(e) + 1)
    m = np.cumsum(e - mbar - delta); return m - np.minimum.accumulate(np.minimum(m, 0))


def ewma_path(e, z0, lam=0.2, burn=10):
    z = np.empty(len(e)); c = z0
    for i, v in enumerate(e): c = lam * v + (1 - lam) * c; z[i] = c
    z[:burn] = -np.inf; return z


def ks_path(s, ref, win=30, step=5):
    """two-sample KS statistic between the reference scores and the last `win` stream scores, every `step` rows"""
    out = np.full(len(s), -np.inf); ref = np.sort(ref); nr = len(ref)
    for t in range(win, len(s) + 1, step):
        w = np.sort(s[t - win:t]); allv = np.concatenate([ref, w])
        cdf1 = np.searchsorted(ref, allv, side="right") / nr; cdf2 = np.searchsorted(w, allv, side="right") / win
        out[t - 1] = np.max(np.abs(cdf1 - cdf2))
    return out


def first_alarm(path, thr):
    idx = np.flatnonzero(path > thr); return idx[0] if len(idx) else None


def cp_ci(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def setup(ds, r, test_size=0.2):
    X, y = load(ds)
    strat = y if ds == "bc" else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=strat, random_state=r)
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
    if ds == "bc":
        m = LogisticRegression(max_iter=5000).fit(A, ytr)
        oof_p = cross_val_predict(LogisticRegression(max_iter=5000), A, ytr, cv=5, method="predict_proba")[:, 1]
        oof_err = ((oof_p >= .5) != ytr).astype(float); ref = m.predict_proba(A)[:, 1]
        err = lambda X_, y_: (m.predict(X_) != y_).astype(float); score = lambda X_: m.predict_proba(X_)[:, 1]
        oof_score = oof_p
    else:
        m = Ridge(1.0).fit(A, ytr); oof = cross_val_predict(Ridge(1.0), A, ytr, cv=5)
        scale = np.percentile(np.abs(oof - ytr), 99)
        oof_err = np.clip(np.abs(oof - ytr) / scale, 0, 1); ref = m.predict(A)
        err = lambda X_, y_: np.clip(np.abs(m.predict(X_) - y_) / scale, 0, 1); score = lambda X_: m.predict(X_)
        oof_score = oof
    return dict(A=A, B=B, ytr=ytr, yte=yte, m=m, oof_err=oof_err, ref=ref, err=err, score=score, oof_score=oof_score,
                sigma=ytr.std())


def calibrate(S, L, rng, n_streams=200):
    mx = {"PH": [], "EWMA": [], "KS": []}
    for _ in range(n_streams):
        i = rng.integers(0, len(S["oof_err"]), L)
        e = S["oof_err"][i]; s = S["oof_score"][i]
        mx["PH"].append(ph_path(e).max()); mx["EWMA"].append(ewma_path(e, S["oof_err"].mean()).max())
        mx["KS"].append(ks_path(s, S["ref"]).max())
    return {k: np.percentile(v, 95) for k, v in mx.items()}


def drifted(S, ds, kind, iota, rng):
    B, y = S["B"], S["yte"]
    if ds == "bc": return drift_clf(B, y, kind, iota, rng)[:2]
    return drift_reg(B, y, kind, iota, rng, S["sigma"])


def detect(S, thr, e, s, onset):
    res = {}
    paths = {"PH": ph_path(e), "EWMA": ewma_path(e, S["oof_err"].mean()), "KS": ks_path(s, S["ref"])}
    for k, p in paths.items():
        a = first_alarm(p, thr[k]); res[k] = a
    return res


rows, fa_rows = [], []
conds = [("bc", "covariate", .5), ("bc", "concept", .5), ("bc", "prior", .5), ("diab", "covariate", .5),
         ("diab", "covariate", 1.0), ("diab", "concept", .5)]
for r in range(30):
    for ds in ["bc", "diab"]:
        S = setup(ds, r); rng = np.random.default_rng(1000 + r)
        thr = calibrate(S, 300, rng)
        # (i) many drift-free streams from the test set (with replacement), plus fresh OOF streams
        for j in range(50):
            i = rng.integers(0, len(S["yte"]), 300)
            res = detect(S, thr, S["err"](S["B"][i], S["yte"][i]), S["score"](S["B"][i]), 150)
            i2 = rng.integers(0, len(S["oof_err"]), 300)
            res2 = detect(S, thr, S["oof_err"][i2], S["oof_score"][i2], 150)
            for k in res:
                fa_rows.append(dict(ds=ds, rep=r, src="test", det=k, fa150=res[k] is not None and res[k] < 150,
                                    fa300=res[k] is not None))
                fa_rows.append(dict(ds=ds, rep=r, src="oof-fresh", det=k, fa150=res2[k] is not None and res2[k] < 150,
                                    fa300=res2[k] is not None))
        # (ii) original design: resampled 150 + 150
        for (d0, kind, iota) in conds:
            if d0 != ds: continue
            Xd, yd = drifted(S, ds, kind, iota, np.random.default_rng(r))
            i = rng.integers(0, len(S["yte"]), 150); k2 = rng.integers(0, len(yd), 150)
            e = np.r_[S["err"](S["B"][i], S["yte"][i]), S["err"](Xd[k2], yd[k2])]
            s = np.r_[S["score"](S["B"][i]), S["score"](Xd[k2])]
            for det, a in detect(S, thr, e, s, 150).items():
                rows.append(dict(design="resampled", ds=ds, kind=kind, iota=iota, rep=r, det=det,
                                 fa=a is not None and a < 150, det_ok=a is not None and a >= 150,
                                 delay=(a - 150) if (a is not None and a >= 150) else np.nan))
    # (iii) independent streams: 50/50 split, every stream row used once (no replacement)
    for ds in ["bc", "diab"]:
        S = setup(ds, r, test_size=0.5); rng = np.random.default_rng(2000 + r)
        n = len(S["yte"]); half = n // 2; thr = calibrate(S, 2 * half, rng)
        for (d0, kind, iota) in conds:
            if d0 != ds: continue
            perm = np.random.default_rng(3000 + r).permutation(n); c, d = perm[:half], perm[half:2 * half]
            S2 = dict(S); S2["B"], S2["yte"] = S["B"][d], S["yte"][d]
            Xd, yd = drifted(S2, ds, kind, iota, np.random.default_rng(r))
            e = np.r_[S["err"](S["B"][c], S["yte"][c]), S["err"](Xd, yd)]
            s = np.r_[S["score"](S["B"][c]), S["score"](Xd)]
            for det, a in detect(S, thr, e, s, half).items():
                rows.append(dict(design="independent", ds=ds, kind=kind, iota=iota, rep=r, det=det,
                                 fa=a is not None and a < half, det_ok=a is not None and a >= half,
                                 delay=(a - half) if (a is not None and a >= half) else np.nan, seg=half))
    print("rep", r, flush=True)

df = pd.DataFrame(rows); df.to_csv(f"{OUT}/C_detect_streams.csv", index=False)
fa = pd.DataFrame(fa_rows); fa.to_csv(f"{OUT}/C_false_alarms.csv", index=False)
print(df.groupby(["design", "ds", "kind", "iota", "det"])[["fa", "det_ok", "delay"]].mean().round(3))
g = fa.groupby(["ds", "src", "det"])[["fa150", "fa300"]].agg(["mean", "sum", "count"])
print(g)
