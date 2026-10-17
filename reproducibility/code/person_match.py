# Participant-by-participant comparison of decisions for selected electrode sets (same plan as channel_wise_true.py)
import numpy as np, json
exec(open("channel_wise_true.py").read().split("out={}")[0])
def decisions(chs):
    F=X[:,chs].reshape(len(X),len(chs),8,16).mean(3).reshape(len(X),-1); s=np.full(len(X),np.nan)
    for k,(pl,ph) in enumerate(PAIRS,1):
        te=np.isin(PID,[pl,ph]); tr=(~te)&(STIM!=1); y=(STIM[tr]==2).astype(int)
        sc=StandardScaler().fit(F[tr]); m=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto",priors=[.5,.5]).fit(sc.transform(F[tr]),y)
        s[te]=m.decision_function(sc.transform(F[te]))
    return {("L" if p>100 else "H")+f"{p%100:02d}":bool(perm_p(s[(PID==p)&(STIM==1)],s[(PID==p)&(STIM==0)],rng=np.random.default_rng(p))<0.05) for p in PIDS}
S={"Parietal-occipital":[0,1,2,4],"All 12 without Fp1/Fp2":list(range(10)),"All 12":list(range(12))}
D={k:decisions(v) for k,v in S.items()}
main=json.load(open("results_Pz-bootstrap_LDA_SVM_EEGNet_CNN-LSTM.json"))["persons"]["LDA"]
D["LDA main analysis"]={r["code"]:bool(r["decision"]) for r in main}
codes=sorted(D["All 12"])
for k,v in D.items():
    err=[c for c in codes if v[c]!=(c[0]=="L")]
    diff=[c for c in codes if v[c]!=D["All 12"][c]]
    print(f"{k:26s} errors {err}  differs from All 12: {diff}")
json.dump(D,open("person_match.json","w"),indent=1)
