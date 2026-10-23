"""Naturally shifted data: multi-site heart disease (clinical, classification) and
Beijing PM2.5 2010-2014 (environmental, regression, temporal)."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from core import *
import pyreadr
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from scipy import stats
OUT = "out"

# ---------------- heart disease ----------------
def load_site(s):
    d = list(pyreadr.read_r(f"{DATA}/heart_disease_{s}.rda").values())[0].copy()
    for c in ["age", "trestbps", "chol", "thalach", "oldpeak", "fbs", "num"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d.loc[d.chol == 0, "chol"] = np.nan
    out = pd.DataFrame({"age": d.age, "male": (d.sex.astype(str) == "Male").astype(float), "trestbps": d.trestbps,
                        "chol": d.chol, "fbs": d.fbs.astype(float), "thalach": d.thalach,
                        "exang": d.exang.astype(str).map({"Yes": 1.0, "No": 0.0}), "oldpeak": d.oldpeak})
    for v in ["typical angina", "atypical angina", "non-anginal pain"]:
        out["cp_" + v[:4]] = (d.cp.astype(str) == v).astype(float)
    for v in ["normal", "ST-T wave abnormality"]:
        out["ecg_" + v[:4]] = (d.restecg.astype(str) == v).astype(float)
    return out, (d.num > 0).astype(int).values

sites = {s: load_site(s) for s in ["cl", "hu", "ch", "va"]}
for s, (Xs_, ys_) in sites.items(): print(s, Xs_.shape, "prevalence %.2f" % ys_.mean(), "missing %.2f" % Xs_.isna().mean().mean())

Xsrc, ysrc = sites["cl"]; med = Xsrc.median()
sc = StandardScaler().fit(Xsrc.fillna(med))
prep = lambda X_: sc.transform(X_.fillna(med))
A = prep(Xsrc)
bases = {"LR": lambda: LogisticRegression(C=1.0, max_iter=5000),
         "RF": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=3, random_state=0, n_jobs=-1)}
hrows = []
for bname, mk in bases.items():
    m = mk().fit(A, ysrc)
    oof = cross_val_predict(mk(), A, ysrc, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    src_f1 = f1_score(ysrc, oof >= .5)
    for tname in ["hu", "ch", "va"]:
        Xt, yt = sites[tname]; B = prep(Xt)
        p_all = m.predict_proba(B)[:, 1]; z_all = m.decision_function(B) if bname == "LR" else logit(p_all, 1e-3)
        iota_hat = iota_proxy(B)
        for r in range(30):
            o = np.random.default_rng(r).permutation(len(yt)); n = len(yt); nc = max(10, int(np.floor(0.15 * n)))
            w, e = o[:nc], o[nc:]; yw, ye = yt[w], yt[e]; p, z = p_all, z_all
            t = ost(yw, p[w]); res = {"Uncorrected": (p[e], .5),
                                       "OST+GSP (proxy)": (gsp(p[e], t, iota_hat), t),
                                       "OST+GSP (iota=1)": (gsp(p[e], t, 1.0), t),
                                       "OST only": (p[e], t)}
            tu = ost_uncapped(yw, z[w]); res["OST uncapped"] = (sigmoid(z[e]), sigmoid(tu))
            res["Platt"] = (platt(yw, z[w], z[e]), .5); res["Intercept"] = (intercept(yw, z[w], z[e]), .5)
            res["Isotonic"] = (isotonic(yw, p[w], p[e]), .5)
            res["Retrain"] = ((mk().fit(B[w], yw).predict_proba(B[e])[:, 1] if len(set(yw)) > 1 else p[e]), .5)
            res["Re-standardisation"] = (m.predict_proba(restandardise(B))[:, 1][e], .5)
            res["MLLS-EM"] = (em_prior(p, ysrc.mean())[e], .5)
            row = dict(base=bname, target=tname, rep=r, src_cv_f1=src_f1, iota_hat=iota_hat, n=n, nc=nc,
                       prev=yt.mean(), auc=roc_auc_score(ye, p[e]))
            for k, (pp, th) in res.items():
                row[k] = f1_score(ye, (pp >= th).astype(int), zero_division=0)
                from sklearn.metrics import balanced_accuracy_score
                row[k + "|bacc"] = balanced_accuracy_score(ye, (pp >= th).astype(int))
            hrows.append(row)
H = pd.DataFrame(hrows); H.to_csv(f"{OUT}/F_heart.csv", index=False)
print(H.groupby(["base", "target"]).mean(numeric_only=True).T.round(3).to_string())

# ---------------- Beijing PM2.5 ----------------
df = pd.read_csv(f"{DATA}/pollution.csv")
df["date"] = pd.to_datetime(df[["year", "month", "day"]])
g = df.groupby("date")
day = pd.DataFrame({"pm": g["pm2.5"].mean(), "DEWP": g.DEWP.mean(), "TEMP": g.TEMP.mean(), "PRES": g.PRES.mean(),
                    "Iws": g.Iws.max(), "Is": g.Is.max(), "Ir": g.Ir.max(),
                    "NW": g.cbwd.apply(lambda s: (s == "NW").mean()), "NE": g.cbwd.apply(lambda s: (s == "NE").mean()),
                    "SE": g.cbwd.apply(lambda s: (s == "SE").mean()),
                    "n_obs": g["pm2.5"].count()}).reset_index()
day = day[day.n_obs >= 18].dropna().reset_index(drop=True)
day["msin"] = np.sin(2 * np.pi * day.date.dt.month / 12); day["mcos"] = np.cos(2 * np.pi * day.date.dt.month / 12)
day["dewp_temp"] = day.TEMP - day.DEWP
feats = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir", "NW", "NE", "SE", "msin", "mcos", "dewp_temp"]
src = day[day.date.dt.year <= 2011]; tgt = day[day.date.dt.year >= 2012]
print("PM2.5 days: source", len(src), "target", len(tgt))
scp = StandardScaler().fit(src[feats]); A = scp.transform(src[feats]); ya = src.pm.values
regs = {"Ridge": lambda: Ridge(1.0), "HGB": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                                                    min_samples_leaf=10, random_state=0)}
prows = []
for bname, mk in regs.items():
    f = mk().fit(A, ya)
    oof = cross_val_predict(mk(), A, ya, cv=5); src_mae = mean_absolute_error(ya, oof)
    rtr = ya - f.predict(A); Hm = np.abs(rtr) > np.percentile(np.abs(rtr), 55); Rc = Ridge(1.0).fit(A[Hm], rtr[Hm])
    for (yr, mo), blk in tgt.groupby([tgt.date.dt.year, tgt.date.dt.month]):
        blk = blk.sort_values("date"); B = scp.transform(blk[feats]); yb = blk.pm.values; n = len(yb)
        nc = max(10, int(np.floor(0.15 * n))); w, e = np.arange(nc), np.arange(nc, n)
        fd = f.predict(B); ih = iota_proxy(B)
        res = {"Uncorrected": fd[e], "Residual Ridge (proxy)": fd[e] + min(0.9 * ih, .75) * Rc.predict(B[e]),
               "Residual Ridge (iota=1)": fd[e] + 0.75 * Rc.predict(B[e]),
               "Mean-bias": fd[e] + np.mean(yb[w] - fd[w]),
               "Linear recal.": LinearRegression().fit(fd[w][:, None], yb[w]).predict(fd[e][:, None]),
               "Retrain": Ridge(10.0).fit(B[w], yb[w]).predict(B[e]),
               "Re-standardisation": f.predict(restandardise(B))[e]}
        row = dict(base=bname, year=yr, month=mo, n=n, nc=nc, iota_hat=ih, src_cv_mae=src_mae)
        for k, v in res.items(): row[k] = mean_absolute_error(yb[e], v)
        prows.append(row)
P = pd.DataFrame(prows); P.to_csv(f"{OUT}/F_pm25.csv", index=False)
print(P.groupby("base").mean(numeric_only=True).T.round(2).to_string())
for bname in regs:
    sub = P[P.base == bname]
    for k in ["Uncorrected", "Mean-bias", "Linear recal.", "Retrain", "Re-standardisation", "Residual Ridge (iota=1)"]:
        d = sub[k] - sub["Residual Ridge (proxy)"]
        print(bname, k, "diff %.2f" % d.mean(), "p=%.4f" % stats.wilcoxon(d).pvalue if (d != 0).any() else "")
