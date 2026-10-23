# Revision 2 scripts (Tables 10–20)

These scripts produce every number in Tables 10–20 of the revised manuscript
(IJIES Paper ID 20266614). Tables 1–9 and all figures come from the main pipeline
(`experiments.py`, `part2.py`, `figures.py`).

## Data
Put the following files in `./data` (or set `APC_DATA=/path/to/data`):

| File | Source |
|---|---|
| `delaney-processed.csv` | https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv (SHA-256 8c06a76f…602f189) |
| `heart_disease_cl.rda`, `heart_disease_hu.rda`, `heart_disease_ch.rda`, `heart_disease_va.rda` | UCI Heart Disease (processed Cleveland, Hungarian, Switzerland, VA files), as packaged in https://github.com/coatless-rpkg/ucidata (`data/`) |
| `pollution.csv` | UCI Beijing PM2.5 (PRSA_data_2010.1.1-2014.12.31), mirror: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv |

Breast Cancer and Diabetes are loaded from scikit-learn.

## Run
```bash
pip install numpy pandas scipy scikit-learn pyreadr
python analyses_synth.py     # A: drift-intensity sensitivity (Table 13), B: overlap-corrected tests (Table 19),
                             # D: matched-row prior shift (Table 10), E: non-linear models (Tables 15-16),
                             # G: delayed / noisy labels (Table 14)
python analyses_detect.py    # detector false alarms and independent streams (Tables 11-12)
python analyses_natural.py   # heart disease and Beijing PM2.5 (Tables 17-18)
python make_tables.py        # assembles Tables 10-20 and the dataset-level summary (Table 20)
```
Seeds 0–29 define the 30 repetitions. Outputs are written to `./out`.
Runtime: about 30–60 minutes on two CPU cores.
