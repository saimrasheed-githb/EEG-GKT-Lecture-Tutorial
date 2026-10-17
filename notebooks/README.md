# EEG-based concealed information test on the Gao et al. (2014) data — notebooks

Eight notebooks reproduce every result on the "Results on the Gao data" slides (88–98) of the public deck v1.7.
All notebooks import `gkt_utils.py`, so the folds, the training task, the decision rule and the metrics are identical for all methods.

| Notebook | Content | Run time (2-core CPU) |
|---|---|---|
| `00_Data_Preparation` | channel order from the raw .cnt header; file audit (9 files excluded); 0–1 s crop, 500 → 128 Hz; ±100 µV rejection → `work/dataset.npz` (23,388 epochs) | long on first run (text parsing), seconds afterwards |
| `01_Classical_Pz_Test` | mean Pz amplitude 300–800 ms, no ML | < 1 min |
| `02_LDA_SVM` | 96 window means, shrinkage LDA and linear SVM, leave-one-pair-out | ~2 min |
| `03_EEGNet` | EEGNet (1,361 parameters) | seconds with saved weights; ~35 min to retrain |
| `04_CNN_LSTM` | EEGNet-style front end + LSTM (6,929 parameters) | seconds with saved weights; ~30 min to retrain |
| `05_Comparison_All_Methods` | results table with Wilson CIs, every person × every method, error overlap | < 1 min |
| `06_Channel_and_Region_Analysis` | single-electrode and region LDA; person-by-person check of identical decisions | ~5 min |
| `07_EEGNet_Attribution` | input × gradient on held-out target, probe and irrelevant epochs | ~1 min |

## Setup
1. Download the Dryad release (doi:10.5061/dryad.2qc64) and point `GAO_DIR` to its folder `14 electrodes-30 subjects`
   (default `./data/14 electrodes-30 subjects/`), e.g. `export GAO_DIR=".../14 electrodes-30 subjects/"`.
2. `pip install -r requirements.txt`
3. Run the notebooks in order. Notebook 00 must run first; 05 needs 01–04; 07 needs the EEGNet weights.

## Saved weights
`work/weights/` holds the trained weights of all 15 folds for EEGNet and CNN-LSTM. Notebooks 03 and 04 load them when present
(`TRAIN = False`) and reproduce the reported scores exactly; set `TRAIN = True` to retrain from scratch
(seed 0 in every fold; results can differ slightly across PyTorch versions and hardware).
Executed outputs in these notebooks were produced with Python 3, PyTorch 2.x, scikit-learn 1.9.

## Conventions
* Rows of each released epoch: Oz, Pz, P4, C4, P3, Cz, C3, Fz, F4, F3, Fp2, Fp1, HEOG, VEOG — read from the one raw recording
  (lying/subject12/session4/xiaowushuang1.cnt) and assumed for all other files.
* `pid` = subject number, +100 for the lying folder; codes L01–L15 (knowledgeable), H01–H15 (control).
* `stim`: 0 irrelevant, 1 probe, 2 target. Models learn target vs irrelevant; probes are never used in training.
* A person is called knowledgeable if their probe scores exceed their irrelevant scores (one-sided permutation test, 2,000 shuffles, p < 0.05).

## Scope
One public dataset, 30 people, leave-one-pair-out validation. Scores measure how target-like probe epochs are — recognition-related
responses — not deception or guilt. Attribution maps show model sensitivity, not cortical sources.
