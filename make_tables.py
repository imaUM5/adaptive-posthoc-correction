"""Builds the values of Tables 10-20 (out/tables.json) and the dataset-level summary (out/H_dataset_level.csv)
from the CSV files written by analyses_synth.py, analyses_detect.py and analyses_natural.py."""
import json, numpy as np, pandas as pd
from scipy.stats import beta
O = "out/"
f3 = lambda v: f"{v:.3f}"; f2 = lambda v: f"{v:.2f}"
# dataset-level summary (Table 20)
rows = []
def add(ds, shift, base, unc, stud, simp, higher=True, metric="F1"):
    g = lambda m: ((m - unc) / unc * 100 if higher else (unc - m) / unc * 100)
    rows.append(dict(ds=ds, shift=shift, base=base, metric=metric, unc=unc, stud=stud, simp=simp, gs=g(stud), gp=g(simp)))
for k in ["covariate", "concept", "prior"]:
    b = pd.read_csv(f"{O}B_clf_{k}.csv"); add("Breast Cancer", k, "LR", b.Uncorrected.mean(), b["OST+GSP"].mean(), b.Platt.mean())
    e = pd.read_csv(f"{O}E_bc_rf_{k}.csv"); add("Breast Cancer", k, "RF", e.Uncorrected.mean(), e["OST+GSP"].mean(), e.Platt.mean())
for ds, nm in [("diab", "Diabetes"), ("logs", "LogS")]:
    for k in ["covariate", "concept"]:
        b = pd.read_csv(f"{O}B_{ds}_{k}.csv"); add(nm, k, "Ridge", b.Uncorrected.mean(), b["Residual Ridge"].mean(), b["Mean-bias"].mean(), False, "MAE")
        e = pd.read_csv(f"{O}E_{ds}_hgb_{k}.csv"); add(nm, k, "HGB", e.Uncorrected.mean(), e["Residual Ridge"].mean(), e["Mean-bias"].mean(), False, "MAE")
H = pd.read_csv(f"{O}F_heart.csv")
for (b, t), s in H.groupby(["base", "target"]):
    add("Heart " + {"hu": "Hungary", "ch": "Switzerland", "va": "Long Beach"}[t], "natural", b, s["Uncorrected|bacc"].mean(),
        s["OST+GSP (proxy)|bacc"].mean(), s["Platt|bacc"].mean(), True, "BAcc")
P = pd.read_csv(f"{O}F_pm25.csv")
for b, s in P.groupby("base"):
    add("Beijing PM2.5", "natural", b, s.Uncorrected.mean(), s["Residual Ridge (proxy)"].mean(), s["Mean-bias"].mean(), False, "MAE")
HL = pd.DataFrame(rows); HL.to_csv(f"{O}H_dataset_level.csv", index=False)

T = {}
D = pd.read_csv(f"{O}D_prior_matched.csv").groupby("iota").mean()
T["t10"] = [["iota", "Eval. rows", "Positives", "FP", "F1 clean", "F1 shifted", "Precision", "Recall"]] + [
    [f"{i:.2f}", f"{r.n_eval:.0f}", f"{r.n_pos:.1f}", f"{r.fp:.2f}", f3(r.f1_clean_matched), f3(r.f1_shifted_matched),
     f3(r.precision), f3(r.recall)] for i, r in D.iterrows()]
fa = pd.read_csv(f"{O}C_false_alarms.csv")
def cp(k, n): return (beta.ppf(.025, k, n - k + 1) if k > 0 else 0, beta.ppf(.975, k + 1, n - k) if k < n else 1)
def fmt(s, col):
    k = s[col].sum(); n = len(s); lo, hi = cp(k, n); return f"{100*k/n:.1f} [{100*lo:.1f}, {100*hi:.1f}]"
r_ = [["Data", "Detector", "OOF streams, 300 rows", "Test streams, 300 rows", "Test streams, first 150 rows"]]
for ds, nm in [("bc", "Breast Cancer"), ("diab", "Diabetes")]:
    for det, dn in [("PH", "Page-Hinkley"), ("EWMA", "EWMA"), ("KS", "KS (label-free)")]:
        a = fa[(fa.ds == ds) & (fa.det == det)]
        r_.append([nm, dn, fmt(a[a.src == "oof-fresh"], "fa300"), fmt(a[a.src == "test"], "fa300"), fmt(a[a.src == "test"], "fa150")])
T["t11"] = r_
C = pd.read_csv(f"{O}C_detect_streams.csv")
r_ = [["Condition", "Design", "Page-Hinkley", "EWMA", "KS"]]
for ds, k, i, nm in [("bc", "covariate", .5, "BC covariate 0.5"), ("bc", "concept", .5, "BC concept 0.5"), ("bc", "prior", .5, "BC prior 0.5"),
                     ("diab", "covariate", .5, "Diabetes covariate 0.5"), ("diab", "covariate", 1.0, "Diabetes covariate 1.0"),
                     ("diab", "concept", .5, "Diabetes concept 0.5")]:
    for des in ["resampled", "independent"]:
        row = [nm if des == "resampled" else "", des]
        for det in ["PH", "EWMA", "KS"]:
            s = C[(C.design == des) & (C.ds == ds) & (C.kind == k) & (C.iota == i) & (C.det == det)]
            row.append(f"{100*s.det_ok.mean():.0f}/{100*s.fa.mean():.0f}")
        r_.append(row)
T["t12"] = r_
A = pd.read_csv(f"{O}A_iota_sensitivity.csv"); A = A[A.iota == 0.5]
g = lambda t, k, s: A[(A.task == t) & (A.kind == k) & (A.setting == s)].iloc[0]
r_ = [["iota used", "BC cov. F1", "BC con. F1", "Diab. cov. MAE", "LogS cov. MAE"],
      ["Uncorrected", f3(g("BC", "covariate", "oracle").unc), f3(g("BC", "concept", "oracle").unc), f2(g("diab", "covariate", "oracle").unc), f3(g("logs", "covariate", "oracle").unc)]]
for s in ["oracle", "bias -50%", "bias -25%", "bias +25%", "bias +50%", "noise SD 0.1", "noise SD 0.25", "proxy"]:
    r_.append([s, f3(g("BC", "covariate", s).value), f3(g("BC", "concept", s).value), f2(g("diab", "covariate", s).value), f3(g("logs", "covariate", s).value)])
r_.append(["Mean-bias (no iota)", "-", "-", f2(g("diab", "covariate", "oracle").mb), f3(g("logs", "covariate", "oracle").mb)])
T["t13"] = r_
G = pd.read_csv(f"{O}G_delay_noise.csv"); Gb = G[G.task == "BC"]
r_ = [["Method", "Cov. on time", "Cov. delay 34", "Cov. 20% noise", "Con. on time", "Con. delay 34", "Con. 20% noise"]]
for m in ["Uncorrected", "OST+GSP", "Platt", "Intercept", "Isotonic", "Retrain"]:
    v = lambda k, no, de: f3(Gb[(Gb.kind == k) & (Gb.noise == no) & (Gb.delay == de) & (Gb.method == m)]["mean"].iloc[0])
    r_.append([m, v("covariate", 0, 0), v("covariate", 0, 34), v("covariate", .2, 0), v("concept", 0, 0), v("concept", 0, 34), v("concept", .2, 0)])
T["t14"] = r_
E = pd.read_csv(f"{O}E_nonlinear.csv"); eb = E[E.task == "BC"]
r_ = [["Method", "Covariate F1", "Concept F1", "Prior F1"], ["Clean (same rows)"] + [f3(eb[eb.kind == k].clean.iloc[0]) for k in ["covariate", "concept", "prior"]]]
for m in ["Uncorrected", "OST+GSP", "OST only", "OST uncapped", "Platt", "Intercept", "Isotonic", "Retrain", "Re-standardisation", "MLLS-EM"]:
    r_.append([m] + [f3(eb[(eb.kind == k) & (eb.method == m)]["mean"].iloc[0]) for k in ["covariate", "concept", "prior"]])
T["t15"] = r_
er = E[E.base == "HGB"]
cell = lambda t, k, m, f: f(er[(er.task == t) & (er.kind == k) & (er.method == m)]["mean"].iloc[0])
r_ = [["Method", "Diab. cov.", "Diab. con.", "LogS cov.", "LogS con."]]
for m in ["Uncorrected", "Residual Ridge", "Mean-bias", "Linear recal.", "Retrain", "Re-standardisation"]:
    r_.append([m, cell("diab", "covariate", m, f2), cell("diab", "concept", m, f2), cell("logs", "covariate", m, f3), cell("logs", "concept", m, f3)])
T["t16"] = r_
Hl = H[H.base == "LR"]
r_ = [["Method", "HU F1", "HU BAcc", "CH F1", "CH BAcc", "LB F1", "LB BAcc"]]
for m in ["Uncorrected", "OST+GSP (proxy)", "OST only", "OST uncapped", "Platt", "Intercept", "Isotonic", "Retrain", "Re-standardisation", "MLLS-EM"]:
    row = [m]
    for t in ["hu", "ch", "va"]:
        s = Hl[Hl.target == t]; row += [f3(s[m].mean()), f3(s[m + "|bacc"].mean())]
    r_.append(row)
T["t17"] = r_
r_ = [["Method", "Ridge MAE", "Ridge: months better", "HGB MAE", "HGB: months better"]]
for m in ["Uncorrected", "Residual Ridge (proxy)", "Residual Ridge (iota=1)", "Mean-bias", "Linear recal.", "Retrain", "Re-standardisation"]:
    row = [m]
    for b in ["Ridge", "HGB"]:
        s = P[P.base == b]; row += [f2(s[m].mean()), "-" if m == "Uncorrected" else f"{int((s[m] < s.Uncorrected - 1e-9).sum())}/36"]
    r_.append(row)
T["t18"] = r_
T["t19_inference"] = pd.read_csv(f"{O}B_inference.csv").round(4).values.tolist()
T["t20"] = [["Data and shift", "Base", "Metric", "Studied: gain %", "Simple: gain %"]] + [
    [f"{r.ds}, {r['shift']}", r.base, r.metric, f"{r.gs:+.1f}", f"{r.gp:+.1f}"] for _, r in HL.iterrows()]
json.dump(T, open(f"{O}tables.json", "w"), indent=0)
for k, v in T.items():
    print(k); [print("  ", x) for x in v]
