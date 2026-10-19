# EEG-Based Guilty Knowledge Testing — Lecture Tutorial

**Methods, Evidence, Limitations, and Future Directions**

📌 Using this material? Please cite it — see **[How to Cite](How%20to%20Cite.md)**.
*From psychophysiological recognition to brain-based concealed-information detection*

Saim Rasheed · Department of Information Technology, Faculty of Computing and Information Technology, King Abdulaziz University
Public edition v1.7 · October 2026

---

## What this tutorial covers

A 119-slide lecture tutorial on the Guilty Knowledge Test (GKT), also called the Concealed Information Test (CIT), measured with EEG and the P300 event-related potential. It works through one real public dataset (Gao et al., 2014) and covers everything from the recorded signal to classical statistics and deep learning.

Central message: **EEG-based GKT does not read lies.** It measures neural responses that may be consistent with *recognition* of concealed information. Recognition ≠ deception ≠ guilt.

| Section | Slides | Question |
|---|---|---|
| Opening | 1–5 | Fundamental question, recognition vs deception vs guilt, roadmap, how to read the deck |
| I | 6–16 | What is EEG-based GKT, and why is it needed? |
| II | 17–35 | How does it work? EEG, preprocessing, stimuli, ERPs, the P300, decision rules, a worked bootstrap example |
| III | 36–62 | One public P300-CIT study: mock crime, files, data audit, time zero, first look at the data |
| IV | 63–98 | Computational methods: features, LDA/SVM, convolution step by step, EEGNet, CNN-LSTM, validation, **results on the Gao data** and their interpretation |
| V | 99–105 | Limitations: memory, countermeasures, variability, threat matrix, ethics and law |
| VI | 106–111 | Future directions: better experiments, cross-dataset transfer, a translation ladder |
| Conclusion | 112–114 | Take-home messages |
| References | 115–119 | 59 numbered references (12 marked † as further reading) and 3 supplementary architecture papers |

Every slide states why it is there. Speaker notes are included in the PowerPoint file.

## Repository layout

```
slides/          lecture tutorial (PPTX with speaker notes, and PDF)
notebooks/       8 Jupyter notebooks reproducing every result on slides 88–98
figures/         full-resolution figures for online reading
reproducibility/ analysis plan, file manifest, frozen results, per-person decisions
```

## Notebooks

| Notebook | Content |
|---|---|
| `00_Data_Preparation` | Channel order from the raw .cnt header; file audit (9 files excluded); 0–1 s crop, 500 → 128 Hz; ±100 µV rejection (23,388 epochs) |
| `01_Classical_Pz_Test` | Mean Pz amplitude 300–800 ms, no machine learning |
| `02_LDA_SVM` | 96 window-mean features; shrinkage LDA and linear SVM |
| `03_EEGNet` | EEGNet, 1,361 parameters |
| `04_CNN_LSTM` | EEGNet-style front end + LSTM, 6,929 parameters |
| `05_Comparison_All_Methods` | Results with Wilson CIs; every person × every method |
| `06_Channel_and_Region_Analysis` | Single-electrode and region models |
| `07_EEGNet_Attribution` | Input × gradient on held-out target, probe and irrelevant epochs |

Validation is leave-one-pair-out: 15 folds, each holding out one knowledgeable and one control person. Models learn target vs irrelevant epochs, and probes are never used in training. A person is called knowledgeable when their probe scores exceed their irrelevant scores (one-sided permutation test, p < 0.05).

| Method | Trial AUC | Person AUC | Detected | Controls cleared | Balanced accuracy |
|---|---|---|---|---|---|
| Classical Pz test (no ML) | 0.62 | 0.87 | 12/15 | 13/15 | 0.83 |
| LDA | 0.90 | 0.94 | 14/15 | 13/15 | 0.90 |
| SVM | 0.90 | 0.94 | 14/15 | 12/15 | 0.87 |
| EEGNet | 0.91 | 0.92 | 14/15 | 13/15 | 0.90 |
| CNN-LSTM | 0.91 | 0.94 | 14/15 | 14/15 | 0.93 |

One dataset, 30 people: these results are recognition evidence under laboratory conditions. They do not show deception, guilt or transfer to other settings.

## Data

The EEG data are **not** included. Download them from Dryad: Gao, J. et al. (2015), doi:[10.5061/dryad.2qc64](https://doi.org/10.5061/dryad.2qc64).
Original article: Gao, J. et al. (2014), *PLOS ONE* 9(11): e109700, doi:[10.1371/journal.pone.0109700](https://doi.org/10.1371/journal.pone.0109700).

Channel names follow the header of the one raw recording in the release (`lying/subject12/session4/xiaowushuang1.cnt`); the same order is assumed for all other files.

## How to cite

See **[How to Cite](How%20to%20Cite.md)** for APA, IEEE and BibTeX entries, including the original Gao et al. data. GitHub's **"Cite this repository"** button (right sidebar) gives the same citation.

> Rasheed, S. (2026). *EEG-Based Guilty Knowledge Testing: Methods, Evidence, Limitations, and Future Directions* (Lecture tutorial, public edition v1.7). King Abdulaziz University. https://github.com/saimrasheed-githb/EEG-GKT-Lecture-Tutorial

## License

MIT License — see [LICENSE](LICENSE).
