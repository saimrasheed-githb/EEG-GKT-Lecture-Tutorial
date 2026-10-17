"""Shared helpers for the Gao et al. (2014) P300-CIT notebooks.

Every notebook imports this file, so the folds, the decision rule and the metrics
are identical for all five methods.
"""
import os, json, numpy as np
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------- paths
# Folder that contains "honest" and "lying" (the "14 electrodes-30 subjects" folder of the Dryad release)
GAO_DIR = os.environ.get("GAO_DIR", "./data/14 electrodes-30 subjects/")
WORK = os.environ.get("GKT_WORK", "./work/")          # cache, dataset, scores, weights
os.makedirs(WORK, exist_ok=True)
P = lambda name: os.path.join(WORK, name)

# ---------------------------------------------------------------- channels
# Row order of the released epochs, read from the header of the one raw recording
# in the release (lying/subject12/session4/xiaowushuang1.cnt). Assumed identical for all files.
CHANNELS = ["Oz", "Pz", "P4", "C4", "P3", "Cz", "C3", "Fz", "F4", "F3", "Fp2", "Fp1", "HEOG", "VEOG"]
EEG = CHANNELS[:12]
PZ = EEG.index("Pz")                                    # = 1
FRONT_TO_BACK = [EEG.index(c) for c in ["Fp1", "Fp2", "F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4", "Oz"]]

# ---------------------------------------------------------------- epoch timing
FS_RAW, N_RAW, ONSET = 500, 800, 150                   # released epochs: -300 ... +1298 ms, onset at sample 151
FS, T = 128, 128                                        # model input: 0-998 ms resampled to 128 Hz
TIME_MS = np.arange(T) / FS * 1000
STIM_CODE = {"irrelevant": 0, "probe": 1, "target": 2}

# ---------------------------------------------------------------- data
def load_dataset(path=None):
    d = np.load(path or P("dataset.npz"))
    return d["X"], d["group"], d["pid"], d["session"], d["stim"]

def code(pid):
    """L01..L15 = lying folder (knowledgeable), H01..H15 = honest folder (control)."""
    return ("L" if pid > 100 else "H") + f"{pid % 100:02d}"

# ---------------------------------------------------------------- validation plan
PAIRS = [(100 + i, i) for i in range(1, 16)]            # fold k holds out L_k and H_k

def inner_val(k):
    """Inner validation people for early stopping: the next two pairs (cyclic), never the test pair."""
    vp, j = [], k
    while len(vp) < 4:
        j = (j % 15) + 1
        if j == k: continue
        vp += [100 + j, j]
    return set(vp)

def window_means(X, n_win=8):
    """12 x 128 epoch -> 12 x 8 window means (16 samples = 125 ms each) -> 96 features."""
    n, c, t = X.shape
    return X.reshape(n, c, n_win, t // n_win).mean(3).reshape(n, -1)

# ---------------------------------------------------------------- person-level decision
def within_auc(sp, si):
    """P(probe score > irrelevant score); ties count 0.5."""
    from scipy.stats import mannwhitneyu
    return mannwhitneyu(sp, si, alternative="two-sided").statistic / (len(sp) * len(si))

def perm_p(sp, si, n=2000, rng=None):
    """One-sided permutation test of mean(probe) - mean(irrelevant) within one person."""
    rng = rng or np.random.default_rng(0)
    obs = sp.mean() - si.mean(); allv = np.concatenate([sp, si]).copy(); k = len(sp); c = 0
    for _ in range(n):
        rng.shuffle(allv); c += (allv[:k].mean() - allv[k:].mean()) >= obs
    return (c + 1) / (n + 1)

def wilson(k, n, z=1.959964):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)

def evaluate(scores, PID, STIM, n_perm_group=10000):
    """Trial AUC (target vs irrelevant), per-person AUC and decision, person AUC with label-permutation p,
    confusion matrix and Wilson CIs. Same rule for every method."""
    ok = ~np.isnan(scores)
    ti = ok & (STIM != 1)
    trial_auc = roc_auc_score(STIM[ti] == 2, scores[ti])
    rows = []
    for p in sorted(set(PID.tolist())):
        m = ok & (PID == p); sp = scores[m & (STIM == 1)]; si = scores[m & (STIM == 0)]
        pv = perm_p(sp, si, rng=np.random.default_rng(p))
        rows.append(dict(pid=int(p), code=code(p), group=int(p > 100), n_probe=len(sp), n_irr=len(si),
                         auc=float(within_auc(sp, si)), p=float(pv), decision=int(pv < 0.05)))
    y = np.array([r["group"] for r in rows]); a = np.array([r["auc"] for r in rows]); d = np.array([r["decision"] for r in rows])
    pauc = roc_auc_score(y, a)
    rng = np.random.default_rng(0)
    null = [roc_auc_score(rng.permutation(y), a) for _ in range(n_perm_group)]
    pperm = (np.sum(np.array(null) >= pauc) + 1) / (n_perm_group + 1)
    tp = int(((d == 1) & (y == 1)).sum()); fn = 15 - tp; fp = int(((d == 1) & (y == 0)).sum()); tn = 15 - fp
    sens, spec = tp / 15, tn / 15
    summ = dict(trial_auc_target_vs_irrelevant=float(trial_auc), person_auc=float(pauc), person_auc_perm_p=float(pperm),
                TP=tp, FN=fn, TN=tn, FP=fp, sensitivity=sens, sens_ci=list(wilson(tp, 15)),
                specificity=spec, spec_ci=list(wilson(tn, 15)), balanced_accuracy=(sens + spec) / 2, n_persons=len(rows))
    return summ, rows

def show(summ, name=""):
    print(f"{name:24s} trial AUC {summ['trial_auc_target_vs_irrelevant']:.3f} | person AUC {summ['person_auc']:.3f} "
          f"(perm p = {summ['person_auc_perm_p']:.4f}) | detected {summ['TP']}/15 | cleared {summ['TN']}/15 | "
          f"BA {summ['balanced_accuracy']:.3f}")
