# EEG-Based GKT: reproducibility package (Public edition v1.7)

Code and frozen results behind the "Results on the Gao data" slides (88–98) of the public deck.
Data: Gao et al., public Dryad release, doi:10.5061/dryad.2qc64 (not included; download separately).

## Channel order
Rows of every released epoch are taken as **Oz, Pz, P4, C4, P3, Cz, C3, Fz, F4, F3, Fp2, Fp1, HEOG, VEOG**,
read from the header of the one raw recording in the release (lying/subject12/session4/xiaowushuang1.cnt).
The same order is assumed for all other exported files. Earlier code (GKT_Gao_ML_Code.zip) used a different,
assumed order; that package is superseded. Only the classical Pz baseline and the channel-level analyses depend
on the order; LDA, SVM, EEGNet and CNN-LSTM use all 12 rows and are unaffected.

## Run order
| Step | Script | Output |
|---|---|---|
| 1 | `prep.py` (set `GAO_DIR` to the "14 electrodes-30 subjects" folder; uses `chunks.json` for the file list) | `cache/*.npz`, `manifest.json` |
| 2 | `build_dataset.py` | `dataset.npz`: 23,388 epochs, 12 × 128 (0–1 s, 128 Hz), > ±100 µV rejected |
| 3 | `run_classic.py` | `scores_classic.npz` (LDA, SVM, classical Pz test) |
| 4 | `run_dl.py EEGNet` and `run_dl.py CNN-LSTM` | `scores_*.npz`, model weights per fold |
| 5 | `evaluate.py "Pz-bootstrap (no ML)" LDA SVM EEGNet CNN-LSTM` | main results JSON (slide 89) |
| 6 | `channel_wise_true.py` | `channel_wise.json` (slides 93–95) |
| 7 | `person_match.py` | `person_match.json`: decisions per participant (slide 95 claim) |
| 8 | `saliency2.py`, `sal_probe_fig.py` | `saliency_probe.json`, probe-epoch relevance figure (slide 92) |
| 9 | `grid_fig.py`, `channel_fig.py` | figures for slides 90 and 93 |

`plan.json` records the analysis plan fixed before results (folds, training task, models, decision rule, seeds).

## Frozen results (folder `results/`)
- `results_Pz-bootstrap_LDA_SVM_EEGNet_CNN-LSTM.json` – summary and per-person scores/decisions, five methods
- `channel_wise.json`, `person_match.json`, `saliency_probe.json`
- `file_manifest.json`, `GKT_Dryad_File_Manifest.csv` – every expected file, its checksum and the exclusion reason (9 files excluded)

Re-running steps 3 and 5 on the cached dataset reproduces the frozen main-results JSON exactly (checked).

## Scope
One public dataset, 30 people, leave-one-pair-out validation. Scores measure how target-like probe epochs are
(recognition-related responses); they do not measure deception or guilt. Attribution maps show model sensitivity,
not cortical sources.
