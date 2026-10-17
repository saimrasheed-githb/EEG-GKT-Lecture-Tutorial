import numpy as np, json
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from common import *
def feats(x): return x.reshape(len(x),12,8,16).mean(3).reshape(len(x),-1)
F=feats(X)
out={m:np.full(len(X),np.nan) for m in ("LDA","SVM")}
for k,(pl,ph) in enumerate(PAIRS,1):
    te=np.isin(PID,[pl,ph]); tr=(~te)&(STIM!=1)
    y=(STIM[tr]==2).astype(int)
    sc=StandardScaler().fit(F[tr]); A=sc.transform(F[tr]); B=sc.transform(F[te])
    lda=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto",priors=[0.5,0.5]).fit(A,y)
    out["LDA"][te]=lda.decision_function(B)
    svm=LinearSVC(C=1.0,class_weight="balanced",max_iter=20000,random_state=0).fit(A,y)
    out["SVM"][te]=svm.decision_function(B)
    print("fold",k,flush=True)
# classic Pz baseline: mean amplitude 300-800 ms at Pz (row index 1; row order from the raw .cnt header: Oz Pz P4 C4 P3 Cz C3 Fz F4 F3 Fp2 Fp1); raw uV, no training
t=np.arange(128)/128*1000
w=(t>=300)&(t<=800)
out["Pz-bootstrap (no ML)"]=X[:,1][:,w].mean(1)
np.savez("scores_classic.npz",**out)
print("done")
