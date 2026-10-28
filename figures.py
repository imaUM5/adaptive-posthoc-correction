# Generates every figure of the manuscript from results/*.csv
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import experiments as E
plt.rcParams.update({'font.size': 10, 'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'axes.axisbelow': True,
                     'grid.color': '#e6e6e6', 'grid.linewidth': 0.6, 'lines.linewidth': 1.6, 'lines.markersize': 5,
                     'legend.frameon': False, 'figure.dpi': 200, 'savefig.bbox': 'tight', 'axes.titlesize': 10})
C = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'verm': '#D55E00', 'purple': '#CC79A7', 'sky': '#56B4E9', 'grey': '#666666', 'black': '#000000'}
F = 'figures/'; R = 'results/'

def fig1():
    Xb, yb = E.load_bc(); Xd, yd = E.load_diab(); Xl, yl = E.load_logs()
    from sklearn.datasets import load_breast_cancer, load_diabetes
    nb = load_breast_cancer().feature_names; nd = load_diabetes().feature_names
    fig, ax = plt.subplots(3, 3, figsize=(9, 8.6))
    ax[0, 0].bar(['Benign', 'Malignant'], [(yb == 0).sum(), (yb == 1).sum()], color=[C['blue'], C['verm']], width=0.55)
    for k, v in enumerate([(yb == 0).sum(), (yb == 1).sum()]): ax[0, 0].text(k, v + 5, str(v), ha='center')
    ax[0, 0].set_ylabel('Samples'); ax[0, 0].set_title('Breast Cancer: class counts')
    cb = np.array([np.corrcoef(Xb[:, j], yb)[0, 1] for j in range(Xb.shape[1])]); top = np.argsort(-np.abs(cb))[:5]
    Z = (Xb - Xb.mean(0)) / Xb.std(0)
    bp = ax[0, 1].boxplot([Z[yb == c, j] for j in top for c in (0, 1)], positions=[k + o for k in range(5) for o in (-0.18, 0.18)], widths=0.3, patch_artist=True, showfliers=False)
    for k, b in enumerate(bp['boxes']): b.set_facecolor(C['blue'] if k % 2 == 0 else C['verm']); b.set_alpha(0.8)
    for m in bp['medians']: m.set_color('black')
    ax[0, 1].set_xticks(range(5)); ax[0, 1].set_xticklabels([nb[j].replace(' ', '\n') for j in top], fontsize=5); ax[0, 1].set_ylabel('z-score')
    ax[0, 1].set_title('Top-5 features by class'); ax[0, 1].legend([bp['boxes'][0], bp['boxes'][1]], ['Benign', 'Malignant'], fontsize=6)
    top8 = np.argsort(-np.abs(cb))[:8]; im = ax[0, 2].imshow(np.corrcoef(Xb[:, top8].T), cmap='Blues', vmin=0, vmax=1)
    ax[0, 2].set_xticks(range(8)); ax[0, 2].set_yticks(range(8)); ax[0, 2].set_xticklabels([nb[j] for j in top8], rotation=90, fontsize=5); ax[0, 2].set_yticklabels([nb[j] for j in top8], fontsize=5)
    ax[0, 2].grid(False); ax[0, 2].set_title('Feature correlation (|r|)'); fig.colorbar(im, ax=ax[0, 2], fraction=0.046)
    ax[1, 0].hist(yd, bins=25, color=C['green'], edgecolor='white'); ax[1, 0].axvline(yd.mean(), color='black', ls='--', lw=1); ax[1, 0].set_xlabel('Disease progression'); ax[1, 0].set_ylabel('Patients'); ax[1, 0].set_title(f'Diabetes target (mean {yd.mean():.0f})')
    cd = np.array([np.corrcoef(Xd[:, j], yd)[0, 1] for j in range(10)]); j = np.argmax(np.abs(cd))
    ax[1, 1].scatter(Xd[:, j], yd, s=6, color=C['green'], alpha=0.6); ax[1, 1].set_xlabel(f'{nd[j]} (standardised)'); ax[1, 1].set_ylabel('Disease progression'); ax[1, 1].set_title(f'Strongest predictor (r = {cd[j]:.2f})')
    o = np.argsort(cd); ax[1, 2].barh([nd[k] for k in o], cd[o], color=C['green']); ax[1, 2].axvline(0, color='black', lw=0.8); ax[1, 2].set_xlabel('Pearson r with target'); ax[1, 2].set_title('Diabetes: feature-target correlation')
    ax[2, 0].hist(yl, bins=30, color=C['purple'], edgecolor='white'); ax[2, 0].axvline(yl.mean(), color='black', ls='--', lw=1); ax[2, 0].set_xlabel('LogS (log mol/L)'); ax[2, 0].set_ylabel('Molecules'); ax[2, 0].set_title(f'LogS target (mean {yl.mean():.2f})')
    ax[2, 1].scatter(Xl[:, 1], yl, s=5, color=C['purple'], alpha=0.5); ax[2, 1].set_xlabel('Molecular weight (g/mol)'); ax[2, 1].set_ylabel('LogS'); ax[2, 1].set_title('Molecular weight vs LogS')
    cl = np.array([np.corrcoef(Xl[:, j], yl)[0, 1] for j in range(6)]); o = np.argsort(cl); short = ['Min. degree', 'Mol. weight', 'H-bond donors', 'Rings', 'Rot. bonds', 'Polar surf. area']
    ax[2, 2].barh([short[k] for k in o], cl[o], color=C['purple']); ax[2, 2].axvline(0, color='black', lw=0.8); ax[2, 2].set_xlabel('Pearson r with target'); ax[2, 2].set_title('LogS: feature-target correlation')
    fig.tight_layout(); fig.savefig(F + 'fig1_eda.png'); plt.close(fig)

def fig2():
    df = pd.read_csv(R + 'baseline_models_raw.csv'); fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
    for a, (ds, col, lab) in zip(ax, [('Breast Cancer', 'm2', 'F1 (malignant)'), ('Diabetes', 'm1', 'MAE'), ('LogS', 'm1', 'MAE (log mol/L)')]):
        g = df[df.dataset == ds].groupby('model')[col].agg(['mean', 'std']).sort_values('mean', ascending=(ds != 'Breast Cancer'))
        a.bar(range(len(g)), g['mean'], yerr=g['std'], color=C['blue'], width=0.55, capsize=3, error_kw=dict(lw=1))
        a.set_xticks(range(len(g))); a.set_xticklabels([m.replace(' ', '\n') for m in g.index], fontsize=6.5); a.set_ylabel(lab); a.set_title(ds)
        if ds == 'Breast Cancer': a.set_ylim(0.85, 1.0)
        for k, v in enumerate(g['mean']): a.text(k, v + g['std'].iloc[k] + (0.003 if ds == 'Breast Cancer' else 0.01 * v), f'{v:.3f}' if v < 5 else f'{v:.1f}', ha='center', fontsize=6.5)
    fig.tight_layout(); fig.savefig(F + 'fig2_baselines.png'); plt.close(fig)

def fig3():
    s = pd.read_csv(R + 'clf_main_summary.csv'); u = s[s.method == 'Uncorrected']; fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
    st = {'covariate': (C['blue'], 'o', '-'), 'concept': (C['verm'], 's', '--'), 'prior': (C['green'], '^', ':')}
    for d, (c, m, l) in st.items():
        g = u[u.drift == d]; ax[0].plot(g.intensity, g.f1, color=c, marker=m, ls=l, label=d); ax[1].plot(g.intensity, g.auc, color=c, marker=m, ls=l, label=d)
    ax[0].set_title('Breast Cancer: F1, uncorrected'); ax[1].set_title('Breast Cancer: AUC, uncorrected'); ax[0].set_ylabel('F1'); ax[1].set_ylabel('AUC'); ax[1].set_ylim(0.5, 1.02); ax[0].set_ylim(0.45, 1.02); ax[0].legend(fontsize=6.5)
    for n, c, m in [('Diabetes', C['green'], 'o'), ('LogS', C['purple'], 's')]:
        r = pd.read_csv(R + f'reg_{n}_main_summary.csv'); r = r[r.method == 'Uncorrected']
        for d, l in [('covariate', '-'), ('concept', '--')]:
            g = r[r.drift == d]; ax[2].plot(g.intensity, g.mae / g.mae_clean, color=c, marker=m, ls=l, label=f'{n}, {d}')
    ax[2].set_title('Regression: MAE relative to clean'); ax[2].set_ylabel('MAE / clean MAE'); ax[2].legend(fontsize=6)
    for a in ax: a.set_xlabel('Drift intensity ι'); a.set_xticks(E.INTENSITIES)
    fig.tight_layout(); fig.savefig(F + 'fig3_degradation.png'); plt.close(fig)

SEL = [('Uncorrected', C['grey'], 'x', ':'), ('OST+GSP (proposed)', C['verm'], 'o', '-'), ('OST only', C['orange'], 'v', '-'), ('OST, uncapped grid', C['purple'], 'D', '--'),
       ('Platt scaling', C['blue'], 's', '--'), ('Retrain on window', C['sky'], '^', '-.'), ('Re-standardisation', C['green'], 'P', '-.')]
def fig4():
    s = pd.read_csv(R + 'clf_main_summary.csv'); fig, ax = plt.subplots(1, 3, figsize=(9, 3.6), sharey=True)
    for a, d, t in zip(ax, ['covariate', 'concept', 'prior'], ['Covariate shift', 'Concept drift', 'Prior-probability shift']):
        for m, c, mk, l in SEL:
            g = s[(s.drift == d) & (s.method == m)]; a.plot(g.intensity, g.f1, color=c, marker=mk, ls=l, label=m.replace('(proposed)', '(studied)')); a.fill_between(g.intensity, g.f1_lo, g.f1_hi, color=c, alpha=0.12, lw=0)
        a.set_title(t); a.set_xlabel('Drift intensity ι'); a.set_xticks(E.INTENSITIES)
    ax[0].set_ylabel('F1 after correction'); h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); fig.savefig(F + 'fig4_classification.png'); plt.close(fig)

def fig5():
    sel = [('Uncorrected', C['grey'], 'x', ':'), ('Residual Ridge (proposed)', C['verm'], 'o', '-'), ('Mean-bias correction', C['blue'], 's', '--'), ('Linear recalibration', C['purple'], 'D', '--'),
           ('Retrain on window', C['sky'], '^', '-.'), ('Re-standardisation', C['green'], 'P', '-.')]
    fig, ax = plt.subplots(1, 4, figsize=(9, 3.4)); k = 0
    for n in ('Diabetes', 'LogS'):
        r = pd.read_csv(R + f'reg_{n}_main_summary.csv')
        for d in ('covariate', 'concept'):
            a = ax[k]
            for m, c, mk, l in sel:
                g = r[(r.drift == d) & (r.method == m)]; a.plot(g.intensity, g.mae, color=c, marker=mk, ls=l, label=m.replace('(proposed)', '(studied)'), ms=4); a.fill_between(g.intensity, g.mae_lo, g.mae_hi, color=c, alpha=0.12, lw=0)
            a.set_title(f"{n}, {d}"); a.set_xlabel('ι'); a.set_xticks([0, .5, 1]); k += 1
    ax[0].set_ylabel('MAE after correction'); h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout(); fig.savefig(F + 'fig5_regression.png'); plt.close(fig)

def fig6():
    d = pd.read_csv(R + 'detection_BC_summary.csv'); dets = [('Page-Hinkley', C['blue'], 'o', '-'), ('EWMA chart', C['orange'], 's', '--'), ('KS on scores (label-free)', C['green'], '^', ':')]
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.3), sharey=True)
    for a, dr, t in zip(ax, ['covariate', 'concept', 'prior'], ['Covariate shift', 'Concept drift', 'Prior-probability shift']):
        for n, c, m, l in dets:
            g = d[(d.drift == dr) & (d.detector == n)]; a.plot(g.intensity, g.detected.where(g.intensity > 0, g.detected + g.false_alarm), color=c, marker=m, ls=l, label=n)
        a.set_title(t); a.set_xlabel('Drift intensity ι'); a.set_xticks([0, .25, .5, 1])
    ax[0].set_ylabel('Streams with an alarm\n(ι = 0: false alarms)'); ax[0].legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(F + 'fig6_detection.png'); plt.close(fig)

def fig7():
    df = pd.read_csv(R + 'clf_main_raw.csv'); g = df[(df.method == 'OST+GSP (proposed)') & (df.intensity > 0)]; fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for d, c, m in [('covariate', C['blue'], 'o'), ('concept', C['verm'], 's'), ('prior', C['green'], '^')]:
        q = g[g.drift == d]; loss = q.f1_clean - q.f1_drift; r = (q.f1 - q.f1_drift) / np.maximum(loss, 1e-3)
        ax[0].scatter(loss, np.clip(r, -10, 10), s=10, color=c, marker=m, alpha=0.6, label=d)
    ax[0].set_xlabel('Drift-induced F1 loss Δ (per run)'); ax[0].set_ylabel('Per-run recovery ratio (clipped to ±10)'); ax[0].set_title('The ratio explodes when Δ is small'); ax[0].legend(fontsize=6.5)
    w = pd.read_csv(R + 'window_sensitivity.csv'); w = w[w.drift == 'covariate']
    for m, c, mk, l in SEL:
        q = w[w.method == m]; ax[1].plot(q.n_window, q.f1, color=c, marker=mk, ls=l, label=m.replace('(proposed)', '(studied)'))
    ax[1].set_xlabel('Labelled window size n_c'); ax[1].set_ylabel('F1 after correction'); ax[1].set_title('Window-size sensitivity (covariate, ι = 0.5)'); ax[1].legend(fontsize=5.5, ncol=2)
    fig.tight_layout(); fig.savefig(F + 'fig7_ratio_window.png'); plt.close(fig)

if __name__ == '__main__':
    import sys
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        try: f()
        except Exception as e: print(f.__name__, 'skipped:', e)
