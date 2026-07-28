# Adaptive Post-Hoc Correction — Baseline Comparison & CRS Validation

Supporting code for the revision of "Adaptive Post-Hoc Correction of ML
Predictions Under Distribution Shift: A Unified Framework for Clinical and
Environmental Decision Support" (IJIES, Paper ID 20264087).

This repository contains the script used to produce **Table 4** (comparison
against post-hoc calibration/adaptation baselines) and **Table 5** (CRS, ECE,
Brier score) in the revised manuscript, in response to reviewer requests for
reproducible, implemented baselines rather than descriptions alone.

## What's here

| File | Description |
|---|---|
| `revision_experiments.py` | Main script. Implements the proposed OST+GSP / adaptive-Ridge framework and five baselines, runs all drift conditions, and exports the CSVs below. |
| `requirements.txt` | Python dependencies. |
| `table4_classification.csv` | Breast Cancer: F1 clean/drifted/corrected, pp, % rel., CRS, ECE, Brier — all methods, all drift types, all five intensities. |
| `table4_regression.csv` | Diabetes (+ LogS if you supply your own loader): MAE clean/drifted/corrected, CRS — all methods, all drift types, all five intensities. |
| `table5_crs_ece_brier.csv` | CRS/ECE/Brier for the proposed method only, ι > 0 (classification). |
| `page_hinkley_detection.csv` | Page-Hinkley detection outcome (fired / not fired, and at what index) for every dataset, drift type, and intensity. |

## Reproducing the results

```bash
pip install -r requirements.txt
python revision_experiments.py
```

Random seed is fixed at `RNG = 42` throughout (data split, drift injection,
calibration window selection). Re-running the script reproduces every number
in the CSVs and in the manuscript's Table 4 / Table 5 exactly.

Runtime: under a minute on a standard CPU (no GPU required, no deep learning).

## Known limitations (documented, not hidden)

- **LogS dataset**: `load_logs_data()` only recovers 6 verified physicochemical
  descriptors from the public Delaney/ESOL mirror, not the 7 used in the
  original manuscript, and deliberately excludes the dataset's own predicted-
  solubility column as a feature (using a model's own prediction as an input
  is a target leak). To reproduce LogS results matching the original
  manuscript, replace `load_logs_data()` with your own loader/feature matrix
  and call `run_regression_experiment(X, y, 'LogS')` with it.
- **Page-Hinkley parameters**: with the manuscript's stated parameters
  (δ=0.005, Λ=30 for classification; δ=1.0, Λ=100 for diabetes), the detector
  never fires within the tested stream length for Breast Cancer, and fires
  even at zero drift intensity for Diabetes (false positive). See
  `page_hinkley_detection.csv`. These parameters should be re-tuned before
  the detector is used to gate correction in a deployed system.
- **Retrain-on-window baseline** can be numerically unstable for regression
  with very small calibration windows relative to the feature count; this is
  reported as a finding (naive retraining is unreliable at this data scale),
  not treated as a bug to be hidden.

## Citation

If you use this code, please cite the associated manuscript (citation details
to be added upon acceptance).
