import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from core import *
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor


def make_reg(base, seed):
    if base == "ridge": return Ridge(alpha=1.0)
    if base == "rf": return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1)
    if base == "hgb": return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed)


def run_rep(X, y, r, kind, iota, base="ridge", iota_mode="oracle", iota_bias=0.0, iota_sd=0.0,
            window_noise_sd=0.0, delay=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=r)
    sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    f = make_reg(base, r).fit(Xtr_s, ytr)
    rng = np.random.default_rng(r)
    Xd, yd = drift_reg(Xte_s, yte, kind, iota, rng, ytr.std())
    order = rng.permutation(len(yd)); Xd, yd, Xc, yc = Xd[order], yd[order], Xte_s[order], yte[order]
    n = len(yd); nc = max(10, int(np.floor(0.15 * n)))
    w = np.arange(delay, delay + nc); e = np.arange(delay + nc, n); pre = np.arange(0, delay)
    # residual corrector fitted offline on hard training samples
    rtr = ytr - f.predict(Xtr_s); H = np.abs(rtr) > np.percentile(np.abs(rtr), 55)
    R = Ridge(alpha=1.0).fit(Xtr_s[H], rtr[H])
    if iota_mode == "oracle": iu = iota
    elif iota_mode == "biased":
        nrng = np.random.default_rng(20_000 + r); iu = float(np.clip(iota * (1 + iota_bias) + nrng.normal(0, iota_sd), 0, 1))
    elif iota_mode == "proxy": iu = iota_proxy(Xd)
    elif iota_mode == "fixed": iu = iota_bias
    wgt = min(0.9 * iu, 0.75)
    fd = f.predict(Xd); yw = yd[w].copy()
    if window_noise_sd > 0:
        yw = yw + np.random.default_rng(10_000 + r).normal(0, window_noise_sd * ytr.std(), len(yw))
    res = {"Uncorrected": fd[e], "Residual Ridge": fd[e] + wgt * R.predict(Xd[e]),
           "Mean-bias": fd[e] + np.mean(yw - fd[w])}
    lr = LinearRegression().fit(fd[w][:, None], yw); res["Linear recal."] = lr.predict(fd[e][:, None])
    res["Retrain"] = Ridge(alpha=10.0).fit(Xd[w], yw).predict(Xd[e])  # replacement model is a Ridge for every base (Table 4)
    res["Re-standardisation"] = f.predict(restandardise(Xd))[e]
    out = {"rep": r, "kind": kind, "iota": iota, "iota_used": iu,
           "clean": mean_absolute_error(yc[e], f.predict(Xc[e]))}
    for k, v in res.items():
        if delay > 0:  # rows that pass before labels arrive are scored uncorrected
            out[k] = mean_absolute_error(np.r_[yd[pre], yd[e]], np.r_[fd[pre], v])
        else:
            out[k] = mean_absolute_error(yd[e], v)
    return out


def run(ds, kind, iota, reps=30, **kw):
    X, y = load(ds)
    return pd.DataFrame([run_rep(X, y, r, kind, iota, **kw) for r in range(reps)])


if __name__ == "__main__":
    for ds in ["diab", "logs"]:
        for kind in ["covariate", "concept"]:
            df = run(ds, kind, 0.5)
            print(ds, kind, "clean %.3f" % df.clean.mean(), {k: round(df[k].mean(), 3) for k in
                  ["Uncorrected", "Residual Ridge", "Mean-bias", "Linear recal.", "Retrain", "Re-standardisation"]})
