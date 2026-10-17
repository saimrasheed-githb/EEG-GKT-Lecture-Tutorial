import numpy as np
D=np.load("dataset.npz")
X,G,PID,SES,STIM=D["X"],D["group"],D["pid"],D["session"],D["stim"]
PIDS=sorted(set(PID.tolist()))
PAIRS=[(100+i,i) for i in range(1,16)]          # fold k holds out lying subject k and honest subject k
def inner_val(train_pids,k):
    # deterministic inner validation: the next two pairs (cyclic) among training persons
    vp=[]; j=k
    while len(vp)<4:
        j=(j%15)+1
        if j==k: continue
        vp+= [100+j, j]
    return set(vp)
def within_auc(sp,si):
    # P(probe score > irrelevant score), ties 0.5
    from scipy.stats import mannwhitneyu
    return mannwhitneyu(sp,si,alternative="two-sided").statistic/(len(sp)*len(si))
def perm_p(sp,si,n=2000,rng=None):
    rng=rng or np.random.default_rng(0)
    obs=sp.mean()-si.mean(); allv=np.concatenate([sp,si]); k=len(sp); c=0
    for _ in range(n):
        rng.shuffle(allv); c+= (allv[:k].mean()-allv[k:].mean())>=obs
    return (c+1)/(n+1)
