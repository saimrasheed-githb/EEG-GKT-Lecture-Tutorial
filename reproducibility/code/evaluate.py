import numpy as np, json, sys
from sklearn.metrics import roc_auc_score
def proportion_confint(k,n,method="wilson",z=1.959964):
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return c-h,c+h
from common import *
def load(name):
    if name in ("LDA","SVM","Pz-bootstrap (no ML)"): return np.load("scores_classic.npz")[name]
    return np.load(f"scores_{name}.npz")["scores"]
res={}; per={}
for name in sys.argv[1:]:
    s=load(name); ok=~np.isnan(s)
    ti=ok&(STIM!=1)
    trial_auc=roc_auc_score(STIM[ti]==2,s[ti]) if name!="Pz-bootstrap (no ML)" else roc_auc_score(STIM[ti]==2,s[ti])
    rows=[]
    for p in PIDS:
        m=ok&(PID==p); sp=s[m&(STIM==1)]; si=s[m&(STIM==0)]
        if len(sp)==0 or len(si)==0: continue
        a=within_auc(sp,si); pv=perm_p(sp,si,rng=np.random.default_rng(p))
        rows.append(dict(pid=int(p),code=("L" if p>100 else "H")+f"{p%100:02d}",group=int(p>100),n_probe=len(sp),n_irr=len(si),auc=float(a),p=float(pv),decision=int(pv<0.05)))
    y=np.array([r["group"] for r in rows]); a=np.array([r["auc"] for r in rows]); d=np.array([r["decision"] for r in rows])
    pauc=roc_auc_score(y,a)
    rng=np.random.default_rng(0); null=[roc_auc_score(rng.permutation(y),a) for _ in range(10000)]
    pperm=(np.sum(np.array(null)>=pauc)+1)/10001
    tp=int(((d==1)&(y==1)).sum()); fn=int(((d==0)&(y==1)).sum()); tn=int(((d==0)&(y==0)).sum()); fp=int(((d==1)&(y==0)).sum())
    sens=tp/(tp+fn); spec=tn/(tn+fp)
    ci=lambda k,n: [float(v) for v in proportion_confint(k,n,method="wilson")]
    res[name]=dict(trial_auc_target_vs_irrelevant=float(trial_auc),person_auc=float(pauc),person_auc_perm_p=float(pperm),
        TP=tp,FN=fn,TN=tn,FP=fp,sensitivity=sens,sens_ci=ci(tp,tp+fn),specificity=spec,spec_ci=ci(tn,tn+fp),balanced_accuracy=(sens+spec)/2,n_persons=len(rows))
    per[name]=rows
    print(f"{name:22s} trialAUC {trial_auc:.3f} personAUC {pauc:.3f} (p={pperm:.4f}) sens {tp}/{tp+fn} spec {tn}/{tn+fp} BA {(sens+spec)/2:.3f}")
json.dump(dict(summary=res,persons=per),open("results_"+"_".join(n.split()[0] for n in sys.argv[1:])+".json","w"),indent=1)
