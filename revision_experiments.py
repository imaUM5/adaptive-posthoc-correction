# ==========================================================================
# Adaptive Post-Hoc Correction — Revision Experiments (v2)
# Paper ID 20264087 (IJIES)
#
# CHANGES FROM v1 (per author-requested corrections, do not remove this log):
#   - Clean / drifted / corrected metrics are now computed on EXACTLY the
#     same evaluation subset (same sample identities), for every drift type
#     including prior-probability shift, where dropped rows previously
#     broke this correspondence. CRS is computed from these three matched
#     values, not from mismatched subsets.
#   - Corrected PROBABILITIES are preserved for every method (not just hard
#     0/1 predictions), so ECE and Brier score can be computed before vs.
#     after correction.
#   - ECE and Brier score are now actually calculated (previously only a
#     function existed but was never called) and exported to Table 5.
#   - Page-Hinkley is now actually run on an error stream for every
#     (drift_type, intensity) condition and its detection outcome is
#     exported to page_hinkley_detection.csv. If you do not run this cell,
#     do not claim in the manuscript that Page-Hinkley was experimentally
#     evaluated by this script.
#   - Calibration methods are skipped (with a logged reason) whenever the
#     calibration window contains only one class, instead of silently
#     crashing or producing a meaningless fit.
#   - The LogS loader no longer uses the model's own predicted solubility
#     as an input feature (that was a target leak) and no longer invents a
#     "molecular weight / 12" descriptor. It now returns only the 6
#     verified physicochemical columns present in the public Delaney/ESOL
#     mirror. This is NOT the same as your original manuscript's "seven
#     physicochemical descriptors" — see the LOGS DATASET section below,
#     you must plug in your own original feature set for a fully
#     apples-to-apples LogS comparison.
#   - Table 4 and Table 5 are exported as their own clearly-named CSVs.
#
# WHAT THIS SCRIPT PRODUCES:
#   table4_classification.csv   -- Breast Cancer: uncorrected vs. 5 methods
#   table4_regression.csv       -- Diabetes (+ LogS if you supply data):
#                                   uncorrected vs. 3 correction methods
#   table5_crs_ece_brier.csv    -- CRS, ECE, Brier before/after, proposed
#                                   method only, classification, ι > 0
#   page_hinkley_detection.csv  -- detection index (or "not detected") per
#                                   drift_type / intensity
#   fig_baseline_comparison_breastcancer.png
#
# HOW TO USE: paste into Google Colab, Run all. Copy the CSV contents
# (or attach the CSVs) into Table 4 / Table 5 of the manuscript. Do not
# type numbers from memory — copy them directly from the CSV/printed
# output of an actual run.
# ==========================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (f1_score, mean_absolute_error,
                              brier_score_loss, log_loss)
from scipy.optimize import minimize_scalar

RNG = 42
np.random.seed(RNG)


def _calibrated_prefit(model, method):
    """Compatibility shim: CalibratedClassifierCV(cv='prefit') was removed
    in sklearn >= 1.6 in favor of wrapping the fitted estimator in
    sklearn.frozen.FrozenEstimator. Works on old and new sklearn."""
    try:
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(FrozenEstimator(model), method=method)
    except ImportError:
        return CalibratedClassifierCV(model, method=method, cv='prefit')


# --------------------------------------------------------------------------
# 1. DATA LOADING
# --------------------------------------------------------------------------

def load_breast_cancer_data():
    data = load_breast_cancer()
    return data.data, data.target


def load_diabetes_data():
    data = load_diabetes()
    return data.data, data.target


def load_logs_data():
    """Delaney ESOL solubility dataset.

    IMPORTANT / CORRECTED: this loader returns ONLY the 6 physicochemical
    descriptor columns that are genuinely present in the public mirror
    (Minimum Degree, Molecular Weight, Number of H-Bond Donors, Number of
    Rings, Number of Rotatable Bonds, Polar Surface Area). It deliberately
    does NOT include the dataset's own "ESOL predicted log solubility"
    column as a feature (using a model's predicted target as an input
    feature is a target leak), and does NOT invent a synthetic 7th
    descriptor.

    Your manuscript states the LogS task used seven physicochemical
    descriptors. This loader only recovers six verified ones. If your
    original preprocessing script computed a 7th real descriptor (e.g. an
    RDKit-computed logP or TPSA), load your own CSV/feature matrix instead
    of calling this function, and pass your X (n_samples x 7) and y
    directly into run_regression_experiment(X, y, 'LogS'). Do not
    substitute an invented feature to force the count to 7.
    """
    urls = [
        "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv",
        "https://raw.githubusercontent.com/GLambard/Molecules_Dataset_Collection/master/latest/delaney-processed.csv",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            feat_cols = ['Minimum Degree', 'Molecular Weight',
                         'Number of H-Bond Donors', 'Number of Rings',
                         'Number of Rotatable Bonds', 'Polar Surface Area']
            missing = [c for c in feat_cols if c not in df.columns]
            if missing:
                continue
            X = df[feat_cols].values
            y = df['measured log solubility in mols per litre'].values
            print(f"[load_logs_data] Loaded {X.shape[0]} samples x "
                  f"{X.shape[1]} verified descriptors from {url}. "
                  f"NOTE: this is 6 descriptors, not the 7 used in your "
                  f"manuscript -- substitute your original loader for an "
                  f"exact match.")
            return X, y
        except Exception:
            continue
    print("[load_logs_data] Could not fetch either public mirror. "
          "Load your original LogS CSV and pass its X, y directly into "
          "run_regression_experiment(X, y, 'LogS') instead of calling "
          "this function.")
    return None, None


# --------------------------------------------------------------------------
# 2. DRIFT SIMULATION
#    Each function now returns (Xd, yd, orig_idx) where orig_idx[i] is the
#    row of the ORIGINAL (undrifted) X_test/y_test that drifted row i
#    corresponds to. This lets us later compute "clean" performance on
#    exactly the same sample identities as the drifted/corrected eval set.
# --------------------------------------------------------------------------

def apply_covariate_shift(X, y, intensity, rng):
    noise = rng.normal(loc=intensity, scale=0.3 * intensity, size=X.shape)
    return X + noise, y.copy(), np.arange(len(X))


def apply_concept_drift_clf(X, y, intensity, rng):
    y2 = y.copy()
    n_flip = int(0.4 * intensity * len(y))
    if n_flip > 0:
        idx = rng.choice(len(y), size=n_flip, replace=False)
        y2[idx] = 1 - y2[idx]
    return X.copy(), y2, np.arange(len(X))


def apply_prior_shift_clf(X, y, intensity, rng, positive_label=1):
    pos_idx = np.where(y == positive_label)[0]
    n_drop = int(0.7 * intensity * len(pos_idx))
    if n_drop == 0:
        return X.copy(), y.copy(), np.arange(len(X))
    drop_idx = rng.choice(pos_idx, size=n_drop, replace=False)
    keep_mask = np.ones(len(y), dtype=bool)
    keep_mask[drop_idx] = False
    kept_idx = np.where(keep_mask)[0]
    return X[keep_mask], y[keep_mask], kept_idx


def apply_concept_drift_reg(X, y, intensity, rng):
    sigma = y.std()
    noise = rng.normal(0, 2 * sigma * intensity, size=y.shape)
    return X.copy(), y + noise, np.arange(len(X))


INTENSITIES = [0.0, 0.25, 0.50, 0.75, 1.00]

# --------------------------------------------------------------------------
# 3. PROPOSED FRAMEWORK — Page-Hinkley + OST + GSP + adaptive residual Ridge
# --------------------------------------------------------------------------

# Manuscript's Section 4.2 Page-Hinkley parameters
PH_PARAMS = {
    'classification': dict(delta=0.005, alpha=0.9999, lam=30),
    'diabetes': dict(delta=1.0, alpha=0.9999, lam=100),
    'logs': dict(delta=0.05, alpha=0.9999, lam=50),
}


def page_hinkley(errors, delta, alpha=0.9999, lam=30):
    """Returns the index in `errors` at which drift is first signaled, or
    None if never signaled."""
    mu_hat, ph, ph_min = 0.0, 0.0, 0.0
    n = 0
    for i, e in enumerate(errors):
        n += 1
        mu_hat += (e - mu_hat) / n
        ph = alpha * ph + (e - mu_hat - delta)
        ph_min = min(ph_min, ph)
        if ph - ph_min > lam:
            return i
    return None


def ost_threshold(y_calib, p_calib):
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.1, 0.91, 0.02):
        f1 = f1_score(y_calib, (p_calib >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def gsp_correction(p, t_star, intensity):
    alpha_gsp = 0.25 + 0.15 * intensity
    p_gsp = p + alpha_gsp * (t_star - 0.5) * (1 - 2 * np.abs(p - 0.5))
    return np.clip(p_gsp, 0.0, 1.0)  # keep valid probabilities for ECE/Brier


def proposed_correction_reg(model, X_train, y_train, X_test, intensity):
    y_pred_train = model.predict(X_train)
    resid = np.abs(y_pred_train - y_train)
    q55 = np.percentile(resid, 55)
    hard_idx = np.where(resid > q55)[0]
    corrector = Ridge().fit(X_train[hard_idx], (y_train - y_pred_train)[hard_idx])
    w = np.clip(0.9 * intensity, 0, 0.75)
    return model.predict(X_test) + w * corrector.predict(X_test)


# --------------------------------------------------------------------------
# 4. BASELINE METHODS — all now return (hard_pred, probabilities)
# --------------------------------------------------------------------------

def baseline_platt(model, X_calib, y_calib, X_eval):
    clf = _calibrated_prefit(model, method='sigmoid')
    clf.fit(X_calib, y_calib)
    p = clf.predict_proba(X_eval)[:, 1]
    return (p >= 0.5).astype(int), p


def baseline_isotonic(model, X_calib, y_calib, X_eval):
    clf = _calibrated_prefit(model, method='isotonic')
    clf.fit(X_calib, y_calib)
    p = clf.predict_proba(X_eval)[:, 1]
    return (p >= 0.5).astype(int), p


def baseline_temperature_scaling(model, X_calib, y_calib, X_eval):
    def to_logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    logits_calib = to_logit(model.predict_proba(X_calib)[:, 1])

    def nll(T):
        p = 1 / (1 + np.exp(-logits_calib / T))
        return log_loss(y_calib, p, labels=[0, 1])

    res = minimize_scalar(nll, bounds=(0.05, 10), method='bounded')
    T = res.x
    logits_eval = to_logit(model.predict_proba(X_eval)[:, 1])
    p_eval = 1 / (1 + np.exp(-logits_eval / T))
    return (p_eval >= 0.5).astype(int), p_eval


def baseline_threshold_tuning(model, X_calib, y_calib, X_eval):
    """OST's threshold step WITHOUT GSP smoothing. Probabilities are
    unchanged (only the decision threshold moves), so 'after' ECE/Brier
    for this baseline is identical to 'before' by construction -- this is
    expected and should be reported as such, not treated as a bug."""
    p_calib = model.predict_proba(X_calib)[:, 1]
    t_star = ost_threshold(y_calib, p_calib)
    p_eval = model.predict_proba(X_eval)[:, 1]
    return (p_eval >= t_star).astype(int), p_eval


def baseline_retrain_on_window(model_class, X_calib, y_calib, X_eval, **kwargs):
    m = model_class(**kwargs).fit(X_calib, y_calib)
    p_eval = m.predict_proba(X_eval)[:, 1]
    return (p_eval >= 0.5).astype(int), p_eval


# --------------------------------------------------------------------------
# 5. CRS + calibration-quality metrics
# --------------------------------------------------------------------------

def compute_crs(metric_before_drift, metric_drifted, metric_corrected,
                 higher_is_better=True, eps=1e-3):
    if higher_is_better:
        delta = metric_before_drift - metric_drifted
        gain = metric_corrected - metric_drifted
    else:
        delta = metric_drifted - metric_before_drift
        gain = metric_drifted - metric_corrected
    return gain / max(delta, eps)


def expected_calibration_error(y_true, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = p[mask].mean()
        ece += (mask.sum() / len(p)) * abs(acc - conf)
    return ece


# --------------------------------------------------------------------------
# 6. CLASSIFICATION EXPERIMENT (Breast Cancer) — Table 4 (clf) + Table 5
# --------------------------------------------------------------------------

def run_classification_experiment():
    X, y = load_breast_cancer_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RNG, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    base_model = LogisticRegression(max_iter=2000).fit(X_train_s, y_train)

    table4_rows = []
    ph_rows = []
    skipped = []

    for drift_type, drift_fn in [
        ('covariate', apply_covariate_shift),
        ('concept', apply_concept_drift_clf),
        ('prior', apply_prior_shift_clf),
    ]:
        for ι in INTENSITIES:
            rng = np.random.default_rng(RNG)
            Xd, yd, orig_idx = drift_fn(X_test_s, y_test, ι, rng)

            # --- Page-Hinkley on the full drifted stream (error indicator) ---
            p_stream = base_model.predict_proba(Xd)[:, 1]
            err_stream = (yd != (p_stream >= 0.5).astype(int)).astype(float)
            ph_idx = page_hinkley(err_stream, **PH_PARAMS['classification'])
            ph_rows.append({
                'drift_type': drift_type, 'intensity': ι,
                'stream_length': len(err_stream),
                'detected': ph_idx is not None,
                'detection_index': ph_idx if ph_idx is not None else '',
            })

            if len(np.unique(yd)) < 2 or len(yd) < 20:
                skipped.append((drift_type, ι, 'fewer than 2 classes or <20 samples after drift'))
                continue

            n_calib = max(10, int(0.15 * len(Xd)))
            calib_local = np.arange(n_calib)
            eval_local = np.arange(n_calib, len(Xd))
            if len(eval_local) < 5:
                skipped.append((drift_type, ι, 'eval subset too small'))
                continue

            X_calib, y_calib = Xd[calib_local], yd[calib_local]
            X_eval, y_eval = Xd[eval_local], yd[eval_local]
            orig_idx_eval = orig_idx[eval_local]

            if len(np.unique(y_calib)) < 2:
                skipped.append((drift_type, ι, 'calibration window has only one class -- calibration-based methods skipped'))
                continue

            # "clean" reference computed on the EXACT SAME sample identities
            # as the drifted/corrected evaluation subset
            X_clean_eval = X_test_s[orig_idx_eval]
            y_clean_eval = y_test[orig_idx_eval]
            p_clean_eval = base_model.predict_proba(X_clean_eval)[:, 1]
            f1_clean = f1_score(y_clean_eval, (p_clean_eval >= 0.5).astype(int), zero_division=0)

            p_drift_eval = base_model.predict_proba(X_eval)[:, 1]
            f1_drift = f1_score(y_eval, (p_drift_eval >= 0.5).astype(int), zero_division=0)
            ece_before = expected_calibration_error(y_eval, p_drift_eval)
            brier_before = brier_score_loss(y_eval, p_drift_eval)

            methods = {'Uncorrected (drifted)': ((p_drift_eval >= 0.5).astype(int), p_drift_eval)}

            t_star = ost_threshold(y_calib, base_model.predict_proba(X_calib)[:, 1])
            p_gsp = gsp_correction(p_drift_eval, t_star, ι)
            methods['Proposed (OST+GSP)'] = ((p_gsp >= t_star).astype(int), p_gsp)

            methods['Platt scaling'] = baseline_platt(base_model, X_calib, y_calib, X_eval)
            methods['Isotonic regression'] = baseline_isotonic(base_model, X_calib, y_calib, X_eval)
            methods['Temperature scaling'] = baseline_temperature_scaling(base_model, X_calib, y_calib, X_eval)
            methods['Threshold tuning (no GSP)'] = baseline_threshold_tuning(base_model, X_calib, y_calib, X_eval)
            methods['Retrain-on-window'] = baseline_retrain_on_window(
                LogisticRegression, X_calib, y_calib, X_eval, max_iter=2000)

            for name, (pred, p_corr) in methods.items():
                f1_corr = f1_score(y_eval, pred, zero_division=0)
                crs = compute_crs(f1_clean, f1_drift, f1_corr, higher_is_better=True)
                ece_after = expected_calibration_error(y_eval, p_corr)
                brier_after = brier_score_loss(y_eval, p_corr)
                table4_rows.append({
                    'drift_type': drift_type, 'intensity': ι, 'method': name,
                    'f1_clean': round(f1_clean, 4), 'f1_drifted': round(f1_drift, 4),
                    'f1_corrected': round(f1_corr, 4),
                    'abs_pp': round((f1_corr - f1_drift) * 100, 2),
                    'rel_pct': round((f1_corr - f1_drift) / max(f1_drift, 1e-6) * 100, 2),
                    'CRS': round(crs, 3),
                    'ECE_before': round(ece_before, 4), 'ECE_after': round(ece_after, 4),
                    'Brier_before': round(brier_before, 4), 'Brier_after': round(brier_after, 4),
                    'n_eval': len(y_eval),
                })

    return pd.DataFrame(table4_rows), pd.DataFrame(ph_rows), skipped


# --------------------------------------------------------------------------
# 7. REGRESSION EXPERIMENT — Table 4 (regression)
# --------------------------------------------------------------------------

def run_regression_experiment(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RNG)
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    base_model = Ridge().fit(X_train_s, y_train)

    ph_key = 'diabetes' if dataset_name.lower().startswith('diab') else 'logs'
    rows = []
    ph_rows = []

    for drift_type, drift_fn in [
        ('covariate', apply_covariate_shift),
        ('concept', apply_concept_drift_reg),
    ]:
        for ι in INTENSITIES:
            rng = np.random.default_rng(RNG)
            Xd, yd, orig_idx = drift_fn(X_test_s, y_test, ι, rng)

            pred_stream = base_model.predict(Xd)
            err_stream = np.abs(pred_stream - yd)
            ph_idx = page_hinkley(err_stream, **PH_PARAMS[ph_key])
            ph_rows.append({
                'dataset': dataset_name, 'drift_type': drift_type, 'intensity': ι,
                'stream_length': len(err_stream), 'detected': ph_idx is not None,
                'detection_index': ph_idx if ph_idx is not None else '',
            })

            n_calib = max(10, int(0.15 * len(Xd)))
            calib_local = np.arange(n_calib)
            eval_local = np.arange(n_calib, len(Xd))
            if len(eval_local) < 5:
                continue
            X_calib, y_calib = Xd[calib_local], yd[calib_local]
            X_eval, y_eval = Xd[eval_local], yd[eval_local]
            orig_idx_eval = orig_idx[eval_local]

            X_clean_eval = X_test_s[orig_idx_eval]
            y_clean_eval = y_test[orig_idx_eval]
            mae_clean = mean_absolute_error(y_clean_eval, base_model.predict(X_clean_eval))

            pred_drift_eval = base_model.predict(X_eval)
            mae_drift = mean_absolute_error(y_eval, pred_drift_eval)

            pred_proposed = proposed_correction_reg(base_model, X_train_s, y_train, X_eval, ι)
            mae_proposed = mean_absolute_error(y_eval, pred_proposed)

            if len(X_calib) >= 5:
                retrain_model = Ridge(alpha=10.0).fit(X_calib, y_calib)
                mae_retrain = mean_absolute_error(y_eval, retrain_model.predict(X_eval))
            else:
                mae_retrain = np.nan

            bias = (y_calib - base_model.predict(X_calib)).mean()
            mae_bias_corr = mean_absolute_error(y_eval, base_model.predict(X_eval) + bias)

            for name, mae_corr in [
                ('Uncorrected (drifted)', mae_drift),
                ('Mean-bias correction', mae_bias_corr),
                ('Retrain-on-window', mae_retrain),
                ('Proposed (adaptive Ridge)', mae_proposed),
            ]:
                if np.isnan(mae_corr):
                    continue
                crs = compute_crs(mae_clean, mae_drift, mae_corr, higher_is_better=False)
                rows.append({
                    'dataset': dataset_name, 'drift_type': drift_type, 'intensity': ι,
                    'method': name, 'mae_clean': round(mae_clean, 3),
                    'mae_drifted': round(mae_drift, 3), 'mae_corrected': round(mae_corr, 3),
                    'abs_change_pct_of_drifted': round((mae_drift - mae_corr) / mae_drift * 100, 2),
                    'CRS': round(crs, 3), 'n_eval': len(y_eval),
                })
    return pd.DataFrame(rows), pd.DataFrame(ph_rows)


# --------------------------------------------------------------------------
# 8. RUN EVERYTHING
# --------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== Classification (Breast Cancer): Table 4 + Table 5 + Page-Hinkley ===")
    df_table4_clf, df_ph_clf, skipped = run_classification_experiment()
    df_table4_clf.to_csv('table4_classification.csv', index=False)
    if skipped:
        print("Skipped conditions (logged, not silently dropped):")
        for s in skipped:
            print("  ", s)

    print("\n=== Regression (Diabetes): Table 4 ===")
    X_diab, y_diab = load_diabetes_data()
    df_table4_reg, df_ph_diab = run_regression_experiment(X_diab, y_diab, 'Diabetes')

    print("\n=== Regression (LogS): Table 4 ===")
    X_logs, y_logs = load_logs_data()
    df_ph_logs = pd.DataFrame()
    if X_logs is not None:
        df_table4_logs, df_ph_logs = run_regression_experiment(X_logs, y_logs, 'LogS')
        df_table4_reg = pd.concat([df_table4_reg, df_table4_logs], ignore_index=True)
    else:
        print("[INSERT VERIFIED SCRIPT OUTPUT] -- LogS Table 4 rows require your "
              "original 7-descriptor dataset; re-run with your own loader.")

    df_table4_reg.to_csv('table4_regression.csv', index=False)

    # --- Table 5: CRS + ECE + Brier, proposed method only, ι > 0 ---
    table5 = df_table4_clf[
        (df_table4_clf['method'] == 'Proposed (OST+GSP)') & (df_table4_clf['intensity'] > 0)
    ][['drift_type', 'intensity', 'f1_clean', 'f1_drifted', 'f1_corrected',
       'abs_pp', 'rel_pct', 'CRS', 'ECE_before', 'ECE_after',
       'Brier_before', 'Brier_after']]
    table5.to_csv('table5_crs_ece_brier.csv', index=False)

    # --- Page-Hinkley detection log (classification + regression) ---
    df_ph_all = pd.concat([
        df_ph_clf.assign(dataset='BreastCancer'),
        df_ph_diab,
        df_ph_logs if not df_ph_logs.empty else pd.DataFrame(),
    ], ignore_index=True)
    df_ph_all.to_csv('page_hinkley_detection.csv', index=False)

    print("\n=== Table 4 (classification) ===")
    print(df_table4_clf.to_string(index=False))
    print("\n=== Table 4 (regression) ===")
    print(df_table4_reg.to_string(index=False))
    print("\n=== Table 5 (CRS / ECE / Brier, proposed method, ι>0) ===")
    print(table5.to_string(index=False))
    print("\n=== Page-Hinkley detection log ===")
    print(df_ph_all.to_string(index=False))

    plt.figure(figsize=(7, 4.5))
    sub = df_table4_clf[df_table4_clf['drift_type'] == 'covariate']
    for method in sub['method'].unique():
        s = sub[sub['method'] == method].sort_values('intensity')
        plt.plot(s['intensity'], s['f1_corrected'], marker='o', label=method)
    plt.xlabel('Drift intensity ι')
    plt.ylabel('F1 (corrected)')
    plt.title('Breast Cancer — covariate shift: proposed vs. baselines')
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig('fig_baseline_comparison_breastcancer.png', dpi=150)

    print("\nSaved: table4_classification.csv, table4_regression.csv, "
          "table5_crs_ece_brier.csv, page_hinkley_detection.csv, "
          "fig_baseline_comparison_breastcancer.png")
