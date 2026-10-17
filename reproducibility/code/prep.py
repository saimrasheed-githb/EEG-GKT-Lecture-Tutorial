# Convert staged Dryad text files into per-file cached arrays (run repeatedly; skips done files)
import json, os, hashlib, numpy as np, pandas as pd
from scipy.signal import resample_poly
UP=os.environ.get("GAO_DIR","./14 electrodes-30 subjects/")   # the "14 electrodes-30 subjects" folder of the Dryad release
os.makedirs("cache",exist_ok=True)
ch=json.load(open("chunks.json")); paths=[p for c in ch for p in c]
man=json.load(open("manifest.json")) if os.path.exists("manifest.json") else {}
for p in paths:
    rel=p.split("14 electrodes-30 subjects\\")[1].replace("\\","/")
    g,subj,sess,stim,_=rel.split("/")
    key=f"{g}/{subj}/{sess}/{stim}"
    if key in man: continue
    f=UP+rel
    if not os.path.exists(f): continue
    raw=open(f,"rb").read(); md5=hashlib.md5(raw).hexdigest()
    rec=dict(group=g,subject=int(subj[7:]),session=int(sess[7:]),stim=stim,file=rel,bytes=len(raw),md5=md5)
    if len(raw)==0:
        rec.update(status="empty"); man[key]=rec; continue
    try: X=np.loadtxt(f,dtype=np.float32)
    except ValueError:
        rec.update(status="malformed"); man[key]=rec; json.dump(man,open("manifest.json","w")); continue
    rec["rows"],rec["cols"]=int(X.shape[0]),int(X.shape[1])
    if X.shape[0]!=14 or X.shape[1]%800!=0:
        rec.update(status="not_epoched"); man[key]=rec; continue
    E=X.reshape(14,-1,800).transpose(1,0,2)
    base=np.abs(E[:,:,:150].mean(2)).max(1)
    rec["n_epochs"]=int(E.shape[0]); rec["baseline_ok_frac"]=float((base<0.05).mean())
    if rec["baseline_ok_frac"]<0.9:
        rec.update(status="no_baseline"); man[key]=rec; continue
    maxabs=np.abs(E[:,:12,:]).max((1,2))
    X128=resample_poly(E[:,:12,150:650].astype(np.float64),32,125,axis=2).astype(np.float32)  # 0-998 ms -> 128 samples
    np.savez_compressed(f"cache/{g}_{subj}_{sess}_{stim}.npz",X=X128,maxabs=maxabs,eogmax=np.abs(E[:,12:,:]).max((1,2)))
    rec.update(status="ok"); man[key]=rec
    json.dump(man,open("manifest.json","w"))
json.dump(man,open("manifest.json","w"))
print(len(man), sum(r["status"]=="ok" for r in man.values()))
