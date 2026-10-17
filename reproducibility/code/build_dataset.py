import json, collections, numpy as np
m=json.load(open("manifest.json"))
d=collections.defaultdict(list)
for k,r in m.items(): d[r["md5"]].append(k)
dups={k for ks in d.values() if len(ks)>1 for k in ks}
Xs=[];rows=[]
for k,r in sorted(m.items()):
    if r["status"]!="ok" or k in dups: continue
    z=np.load(f"cache/{r['group']}_subject{r['subject']}_session{r['session']}_{r['stim']}.npz")
    keep=z["maxabs"]<=100; Xs.append(z["X"][keep])
    rows+=[(1 if r["group"]=="lying" else 0, r["subject"]+(100 if r["group"]=="lying" else 0), r["session"], {"irrelevant":0,"probe":1,"target":2}[r["stim"]])]*int(keep.sum())
L=np.array(rows,dtype=np.int16)
np.savez("dataset.npz",X=np.concatenate(Xs).astype(np.float32),group=L[:,0],pid=L[:,1],session=L[:,2],stim=L[:,3])
