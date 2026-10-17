# Channel-wise and region-wise LDA, same fixed leave-one-pair-out plan as the main analysis
import numpy as np, json
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from common import *
CH=["Oz","Pz","P4","C4","P3","Cz","C3","Fz","F4","F3","Fp2","Fp1"]  # row order verified from the raw .cnt header
def run(chs):
    F=X[:,chs].reshape(len(X),len(chs),8,16).mean(3).reshape(len(X),-1); s=np.full(len(X),np.nan)
    for k,(pl,ph) in enumerate(PAIRS,1):
        te=np.isin(PID,[pl,ph]); tr=(~te)&(STIM!=1); y=(STIM[tr]==2).astype(int)
        sc=StandardScaler().fit(F[tr]); m=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto",priors=[.5,.5]).fit(sc.transform(F[tr]),y)
        s[te]=m.decision_function(sc.transform(F[te]))
    ti=STIM!=1; y=[];a=[];d=[]
    for p in PIDS:
        mm=PID==p; sp=s[mm&(STIM==1)]; si=s[mm&(STIM==0)]
        y.append(int(p>100)); a.append(within_auc(sp,si)); d.append(perm_p(sp,si,rng=np.random.default_rng(p))<0.05)
    y=np.array(y);a=np.array(a);d=np.array(d)
    TP=int((d&(y==1)).sum());FN=15-TP;FP=int((d&(y==0)).sum());TN=15-FP
    prec=TP/(TP+FP) if TP+FP else float("nan"); sen=TP/15; spe=TN/15
    return dict(trial_auc=roc_auc_score(STIM[ti]==2,s[ti]),person_auc=roc_auc_score(y,a),TP=TP,FN=FN,TN=TN,FP=FP,
                accuracy=(TP+TN)/30,precision=prec,sensitivity=sen,specificity=spe,
                f1=(2*prec*sen/(prec+sen) if TP else 0.0),bal_acc=(sen+spe)/2)
out={}
sets={c:[i] for i,c in enumerate(CH)}
sets.update({"Frontal (Fp1 Fp2 F3 Fz F4)":[7,8,9,10,11],"Frontal without Fp1/Fp2 (F3 Fz F4)":[7,8,9],"Central (C3 Cz C4)":[3,5,6],
             "Parietal-occipital (P3 Pz P4 Oz)":[0,1,2,4],"All 12 without Fp1/Fp2":list(range(10)),"All 12 channels":list(range(12))})
for n,c in sets.items():
    out[n]=run(c); r=out[n]
    print(f"{n:36s} tAUC {r['trial_auc']:.3f} pAUC {r['person_auc']:.3f} TP{r['TP']} FP{r['FP']} acc {r['accuracy']:.2f} prec {r['precision']:.2f} sens {r['sensitivity']:.2f} spec {r['specificity']:.2f} F1 {r['f1']:.2f}",flush=True)
json.dump(out,open("channel_wise.json","w"),indent=1)
