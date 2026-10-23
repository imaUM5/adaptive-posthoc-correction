"""New analyses on the synthetic benchmarks requested in the second review."""
import sys, json; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from core import *
import run_clf as C, run_reg as R
from scipy import stats
OUT = "out"; import os; os.makedirs(OUT, exist_ok=True)

# ---------- A. drift-intensity sensitivity ----------
settings = [("oracle", 0, 0), ("biased", -0.5, 0), ("biased", -0.25, 0), ("biased", 0.25, 0), ("biased", 0.5, 0),
            ("biased", 0, 0.1), ("biased", 0, 0.25), ("proxy", 0, 0)]
label = lambda m, b, s: "oracle" if m == "oracle" else ("proxy" if m == "proxy" else (f"bias {int(b*100):+d}%" if s == 0 else f"noise SD {s}"))
rows = []
for kind in ["covariate", "concept", "prior"]:
    for iota in [0.25, 0.5, 1.0]:
        for m, b, s in settings:
            df = C.run(kind, iota, methods="gsp", iota_mode=m, iota_bias=b, iota_sd=s)
            rows.append(dict(task="BC", kind=kind, iota=iota, setting=label(m, b, s), value=df["OST+GSP"].mean(),
                             unc=df["Uncorrected"].mean(), iota_used=df.iota_used.mean(), iota_used_sd=df.iota_used.std()))
for ds in ["diab", "logs"]:
    for kind in ["covariate", "concept"]:
        for iota in [0.25, 0.5, 1.0]:
            for m, b, s in settings:
                df = R.run(ds, kind, iota, iota_mode=m, iota_bias=b, iota_sd=s)
                rows.append(dict(task=ds, kind=kind, iota=iota, setting=label(m, b, s), value=df["Residual Ridge"].mean(),
                                 unc=df["Uncorrected"].mean(), mb=df["Mean-bias"].mean(), iota_used=df.iota_used.mean(),
                                 iota_used_sd=df.iota_used.std()))
pd.DataFrame(rows).to_csv(f"{OUT}/A_iota_sensitivity.csv", index=False)
print("A done")

# ---------- B. dependence-aware inference ----------
def nb_test(d, n_train, n_test):
    """Nadeau-Bengio corrected resampled t-test on J paired differences."""
    J = len(d); m = d.mean(); v = d.var(ddof=1)
    if v == 0: return 1.0
    t = m / np.sqrt((1 / J + n_test / n_train) * v)
    return 2 * stats.t.sf(abs(t), J - 1)

def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for i, k in enumerate(o):
        run = max(run, min(1, (len(p) - i) * p[k])); adj[k] = run
    return adj

brows = []
clf_methods = ["Uncorrected", "OST only", "OST uncapped", "Platt", "Intercept", "Isotonic", "Temperature", "Retrain",
               "Re-standardisation", "MLLS-EM"]
for kind in ["covariate", "concept", "prior"]:
    df = C.run(kind, 0.5); df.to_csv(f"{OUT}/B_clf_{kind}.csv", index=False)
    ntr, nte = 455, df.n_eval.iloc[0] + df.n_win.iloc[0]
    pw, pn = [], []
    for mth in clf_methods:
        d = (df[mth] - df["OST+GSP"]).values
        pw.append(stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0); pn.append(nb_test(d, ntr, nte))
    pw, pn = holm(pw), holm(pn)
    for i, mth in enumerate(clf_methods):
        d = (df[mth] - df["OST+GSP"]).values
        brows.append(dict(task="BC", kind=kind, method=mth, mean=df[mth].mean(), ref=df["OST+GSP"].mean(), diff=d.mean(),
                          p_wilcoxon_holm=pw[i], p_nb_holm=pn[i], sd_split=d.std(ddof=1)))
reg_methods = ["Uncorrected", "Mean-bias", "Linear recal.", "Retrain", "Re-standardisation"]
for ds, ntr in [("diab", 353), ("logs", 902)]:
    for kind in ["covariate", "concept"]:
        df = R.run(ds, kind, 0.5); df.to_csv(f"{OUT}/B_{ds}_{kind}.csv", index=False)
        nte = {"diab": 89, "logs": 226}[ds]
        pw, pn = [], []
        for mth in reg_methods:
            d = (df[mth] - df["Residual Ridge"]).values
            pw.append(stats.wilcoxon(d).pvalue); pn.append(nb_test(d, ntr, nte))
        pw, pn = holm(pw), holm(pn)
        for i, mth in enumerate(reg_methods):
            d = (df[mth] - df["Residual Ridge"]).values
            brows.append(dict(task=ds, kind=kind, method=mth, mean=df[mth].mean(), ref=df["Residual Ridge"].mean(), diff=d.mean(),
                              p_wilcoxon_holm=pw[i], p_nb_holm=pn[i], sd_split=d.std(ddof=1)))
pd.DataFrame(brows).to_csv(f"{OUT}/B_inference.csv", index=False)
print("B done")

# ---------- D. matched-row prior shift ----------
drows = []
X, y = load("bc")
for iota in INTENSITIES:
    for r in range(30):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=r)
        sc = StandardScaler().fit(Xtr); m = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), ytr)
        Xs = sc.transform(Xte); rng = np.random.default_rng(r)
        Xd, yd, keep = drift_clf(Xs, yte, "prior", iota, rng)
        order = rng.permutation(len(yd)); keep = keep[order]; n = len(keep); nc = max(10, int(np.floor(0.15 * n)))
        ev = keep[nc:]
        pc = m.predict(Xs[ev]); pdft = m.predict(Xd[order][nc:])
        from sklearn.metrics import precision_score, recall_score
        drows.append(dict(iota=iota, rep=r, n_eval=len(ev), n_pos=int(yte[ev].sum()),
                          f1_clean_matched=f1_score(yte[ev], pc), f1_shifted_matched=f1_score(yd[order][nc:], pdft),
                          precision=precision_score(yte[ev], pc), recall=recall_score(yte[ev], pc),
                          fp=int(((pc == 1) & (yte[ev] == 0)).sum())))
pd.DataFrame(drows).to_csv(f"{OUT}/D_prior_matched.csv", index=False)
print("D done")

# ---------- E. non-linear base models ----------
erows = []
for kind in ["covariate", "concept", "prior"]:
    df = C.run(kind, 0.5, base="rf"); df.to_csv(f"{OUT}/E_bc_rf_{kind}.csv", index=False)
    for mth in ["Uncorrected", "OST+GSP", "OST only", "OST uncapped", "Platt", "Intercept", "Isotonic", "Retrain", "Re-standardisation", "MLLS-EM"]:
        erows.append(dict(task="BC", base="RF", kind=kind, method=mth, mean=df[mth].mean(), clean=df.clean.mean(),
                          lo=boot_ci(df[mth])[0], hi=boot_ci(df[mth])[1]))
for ds in ["diab", "logs"]:
    for kind in ["covariate", "concept"]:
        df = R.run(ds, kind, 0.5, base="hgb"); df.to_csv(f"{OUT}/E_{ds}_hgb_{kind}.csv", index=False)
        for mth in ["Uncorrected", "Residual Ridge", "Mean-bias", "Linear recal.", "Retrain", "Re-standardisation"]:
            erows.append(dict(task=ds, base="HGB", kind=kind, method=mth, mean=df[mth].mean(), clean=df.clean.mean(),
                              lo=boot_ci(df[mth])[0], hi=boot_ci(df[mth])[1]))
pd.DataFrame(erows).to_csv(f"{OUT}/E_nonlinear.csv", index=False)
print("E done")

# ---------- G. delayed and noisy labels ----------
grows = []
for kind in ["covariate", "concept"]:
    for noise in [0, 0.1, 0.2]:
        for delay in [0, 17, 34]:
            df = C.run(kind, 0.5, window_label_noise=noise, delay=delay)
            for mth in ["Uncorrected", "OST+GSP", "Platt", "Intercept", "Isotonic", "Retrain"]:
                grows.append(dict(task="BC", kind=kind, noise=noise, delay=delay, method=mth, mean=df[mth].mean()))
for kind in ["covariate", "concept"]:
    for noise in [0, 0.25, 0.5]:
        for delay in [0, 13, 26]:
            df = R.run("diab", kind, 0.5, window_noise_sd=noise, delay=delay)
            for mth in ["Uncorrected", "Residual Ridge", "Mean-bias", "Linear recal.", "Retrain"]:
                grows.append(dict(task="diab", kind=kind, noise=noise, delay=delay, method=mth, mean=df[mth].mean()))
pd.DataFrame(grows).to_csv(f"{OUT}/G_delay_noise.csv", index=False)
print("G done")
