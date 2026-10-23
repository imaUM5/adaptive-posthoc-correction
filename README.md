# Adaptive Post-Hoc Correction Under Distribution Shift

Code for "Adaptive Post-Hoc Correction of ML Predictions Under Distribution Shift:
A Controlled Mechanistic Study on Clinical and Environmental Tabular Data" (IJIES, Paper ID 20266614).

## Contents
| Files | Produces |
|---|---|
| experiments.py, part2.py, figures.py | Tables 1–9 and all figures |
| core.py, run_clf.py, run_reg.py, analyses_synth.py, analyses_detect.py, analyses_natural.py, make_tables.py | Tables 10–20 (see README_revision.md) |

## Reproducing
pip install -r requirements.txt
Seeds 0–29 define the 30 repetitions. Breast Cancer and Diabetes are loaded from scikit-learn;
the LogS, heart disease and Beijing PM2.5 files and their sources are listed in README_revision.md.
