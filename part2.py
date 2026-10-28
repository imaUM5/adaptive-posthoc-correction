# Part 2: window-size sensitivity, recovery-ratio stability, end-to-end detector-corrector pipeline
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, mean_absolute_error
import experiments as E
from experiments import *

# ---------------------------------------------------------------- A. window-size sensitivity (iota = 0.5)
def window_sensitivity():
    out = []
    for frac in (0.05, 0.10, 0.15, 0.30, 0.50):
        df = E.run_clf(window_frac=frac, intensities=[0.5], tag=f'win{int(frac*100)}')
        g = df.groupby(['drift', 'method']).agg(f1=('f1', 'mean'), sd=('f1', 'std'), n_window=('n_window', 'mean')).reset_index()
        g['frac'] = frac; out.append(g)
    pd.concat(out).to_csv(f"{OUT}/window_sensitivity.csv", index=False)

# ---------------------------------------------------------------- B. recovery-ratio stability
def nrr_stability():
    df = pd.read_csv(f"{OUT}/clf_main_raw.csv"); g = df[(df.method == 'OST+GSP (proposed)') & (df.intensity > 0)]
    rows = []
    for (d, i), q in g.groupby(['drift', 'intensity']):
        loss = (q.f1_clean - q.f1_drift).values; gain = (q.f1 - q.f1_drift).values
        per = gain / np.maximum(loss, 1e-3)                       # original per-run definition (Eq. with eps floor)
        pooled, lo, hi = E.recovery_ratio(q.f1_clean, q.f1_drift, q.f1)
        rows.append(dict(drift=d, intensity=i, mean_loss=loss.mean(), sd_loss=loss.std(ddof=1), mean_gain=gain.mean(), sd_gain=gain.std(ddof=1),
                         per_run_mean=per.mean(), per_run_sd=per.std(ddof=1), per_run_min=per.min(), per_run_max=per.max(),
                         frac_loss_below_0p02=(loss < 0.02).mean(), pooled=pooled, pooled_lo=lo, pooled_hi=hi,
                         delta_method_sd=np.sqrt(gain.var(ddof=1) / max(loss.mean(), 1e-9) ** 2 + gain.mean() ** 2 * loss.var(ddof=1) / max(loss.mean(), 1e-9) ** 4),
                         ece_change=(q.ece.mean() - df[(df.method == 'Uncorrected') & (df.drift == d) & (df.intensity == i)].ece.mean()),
                         brier_change=(q.brier.mean() - df[(df.method == 'Uncorrected') & (df.drift == d) & (df.intensity == i)].brier.mean())))
    pd.DataFrame(rows).to_csv(f"{OUT}/nrr_stability.csv", index=False)

# ---------------------------------------------------------------- C. detectors
def ph_stat(e, delta=0.05):
    """Standard Page-Hinkley statistic (increase test) m_t - min_{s<=t} m_s."""
    t = np.arange(1, len(e) + 1); mean = np.cumsum(e) / t; m = np.cumsum(e - mean - delta)
    return m - np.minimum.accumulate(np.minimum(m, 0))

def ewma_stat(e, mu0, sd0, lam=0.2):
    """EWMA control chart on the error stream (ECDD-style, Ross et al., 2012):
    standardised distance of the EWMA from the in-control mean mu0 (training out-of-fold errors)."""
    z = mu0; out = np.zeros(len(e))
    for t, x in enumerate(e, 1):
        z = (1 - lam) * z + lam * x
        out[t - 1] = (z - mu0) / (sd0 * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * t))) + 1e-12)
    return out

def ks_stat(v, ref, win=30, step=5):
    """Label-free: -log10 p of a two-sample KS test between reference model scores and the
    latest `win` stream scores, evaluated every `step` instances."""
    out = np.zeros(len(v))
    for t in range(win, len(v) + 1, step): out[t - 1] = -np.log10(max(ks_2samp(ref, v[t - win:t]).pvalue, 1e-300))
    return out

DETECTORS = ['Page-Hinkley', 'EWMA chart', 'KS on scores (label-free)']
L0 = L1 = 150
def stat(dn, err, score, ref, mu0=0, sd0=1):
    return ph_stat(err) if dn == 'Page-Hinkley' else ewma_stat(err, mu0, sd0) if dn == 'EWMA chart' else ks_stat(score, ref)
def first_alarm(st, thr):
    idx = np.where(st > thr)[0]; return int(idx[0]) if len(idx) else None

def run_detection(task):
    rows = []
    for seed in range(R):
        rng = np.random.default_rng(seed)
        if task == 'BC':
            b = E.clf_context(seed); model = b['model']
            err_src = ((b['p_src'] >= 0.5).astype(int) != b['ytr']).astype(float); sc_src = logit(b['p_src'])
            drifts = [('covariate', covariate_shift), ('concept', concept_clf), ('prior', prior_clf)]
            scale = lambda e: e
        else:
            X, y = E.load_diab(); b = E.reg_context(X, y, seed); model = b['model']
            oof = cross_val_predict(Ridge(alpha=1.0), b['Xtr'], b['ytr'], cv=KFold(5, shuffle=True, random_state=seed))
            ae = np.abs(oof - b['ytr']); cap = np.percentile(ae, 99) * 2; err_src = np.clip(ae / cap, 0, 1); sc_src = oof
            drifts = [('covariate', covariate_shift), ('concept', None)]
            scale = lambda e: np.clip(e / cap, 0, 1)
        # tune on clean streams resampled from out-of-fold training errors / scores
        # thresholds: 95th percentile of the maximum statistic over 200 clean streams resampled from
        # out-of-fold TRAINING errors / scores (target false-alarm rate 5% per 300-instance stream)
        mu0, sd0 = err_src.mean(), err_src.std() + 1e-9
        tr_streams = [rng.choice(len(err_src), L0 + L1) for _ in range(200)]
        par = {dn: float(np.percentile([stat(dn, err_src[q], sc_src[q], sc_src, mu0, sd0).max() for q in tr_streams], 95)) for dn in DETECTORS}
        for dname, dfn in drifts:
            for i in (0.0, 0.25, 0.5, 1.0):
                r2 = np.random.default_rng(1000 * seed + int(100 * i) + 7)
                if dname == 'concept' and task != 'BC': Xd, yd, _ = concept_reg(b['Xte'], b['yte'], i, r2, b['ytr'].std())
                else: Xd, yd, _ = dfn(b['Xte'], b['yte'], i, r2)
                a = r2.choice(len(b['Xte']), L0); d = r2.choice(len(Xd), L1)
                Xs = np.vstack([b['Xte'][a], Xd[d]]); ys = np.r_[b['yte'][a], yd[d]]
                if task == 'BC':
                    p = model.predict_proba(Xs)[:, 1]; err = ((p >= 0.5).astype(int) != ys).astype(float); score = logit(p)
                else:
                    p = model.predict(Xs); err = scale(np.abs(p - ys)); score = p
                triggers = {'No correction': None, 'Oracle trigger': L0}
                for dn in DETECTORS: triggers[dn] = first_alarm(stat(dn, err, score, sc_src, mu0, sd0), par[dn])
                n_c = 22
                for dn, tau in triggers.items():
                    # downstream utility on the drifted segment [L0, L0+L1)
                    pred = (p >= 0.5).astype(int) if task == 'BC' else p.copy()
                    if tau is not None and tau + 1 + n_c < L0 + L1:
                        w = np.arange(tau + 1, tau + 1 + n_c); rest = np.arange(tau + 1 + n_c, L0 + L1)
                        if task == 'BC':
                            c = dict(y_w=ys[w], p_w=p[w], p_ev=p[rest], i=i)
                            pred_p = pred.copy(); pred_p[rest] = E.m_platt(c)[0]
                            pred_o = pred.copy(); pred_o[rest] = E.m_proposed(c)[0]
                        else:
                            pred_p = pred.copy(); pred_p[rest] = p[rest] + (ys[w] - p[w]).mean()
                            pred_o = pred.copy(); pred_o[rest] = p[rest] + np.clip(0.9 * i, 0, 0.75) * b['corrector'].predict(Xs[rest])
                    else: pred_p = pred_o = pred
                    seg = np.arange(L0, L0 + L1); pre = np.arange(0, L0)
                    met = (lambda yy, pp: f1_score(yy, pp, zero_division=0)) if task == 'BC' else mean_absolute_error
                    rows.append(dict(task=task, seed=seed, drift=dname, intensity=i, detector=dn, alarm=tau is not None,
                                     false_alarm=(tau is not None and tau < L0), detected=(tau is not None and tau >= L0),
                                     delay=(tau - L0) if (tau is not None and tau >= L0) else np.nan,
                                     util_simple=met(ys[seg], pred_p[seg]), util_proposed=met(ys[seg], pred_o[seg]),
                                     util_pre=met(ys[pre], pred_p[pre]), param=par.get(dn, np.nan)))
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/detection_{task}_raw.csv", index=False)
    s = df.groupby(['task', 'drift', 'intensity', 'detector']).agg(false_alarm=('false_alarm', 'mean'), detected=('detected', 'mean'),
            delay=('delay', 'mean'), util_simple=('util_simple', 'mean'), util_proposed=('util_proposed', 'mean'), util_pre=('util_pre', 'mean'), param=('param', 'median')).reset_index()
    s['missed'] = 1 - s.detected - s.false_alarm
    s.to_csv(f"{OUT}/detection_{task}_summary.csv", index=False); return s

if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
    nrr_stability(); print(pd.read_csv(f"{OUT}/nrr_stability.csv").round(3).to_string())
    for t in ('BC', 'Diabetes'): print(run_detection(t).round(3).to_string())
    window_sensitivity()
    w = pd.read_csv(f"{OUT}/window_sensitivity.csv"); print(w.pivot_table(index=['drift', 'method'], columns='frac', values='f1').round(3).to_string())
