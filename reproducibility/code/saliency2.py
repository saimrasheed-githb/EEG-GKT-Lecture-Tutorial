import numpy as np, torch, json, sys
sys.argv=["x","EEGNet"]
exec(open("run_dl.py").read().split("Net={")[0].replace("MODEL=sys.argv[1]",""))
CH=["Oz","Pz","P4","C4","P3","Cz","C3","Fz","F4","F3","Fp2","Fp1"]   # true row order (raw .cnt header)
def rel(net,x):
    xt=torch.tensor(x,dtype=torch.float32,requires_grad=True); net(xt).sum().backward()
    return np.abs((xt.grad*xt).detach().numpy()).mean(0)          # 12 x 128, mean |input x gradient|
R={}   # (pid, cond) -> 12x128
for k,(pl,ph) in enumerate(PAIRS,1):
    te=np.isin(PID,[pl,ph]); trp=[p for p in PIDS if p not in (pl,ph)]; vp=inner_val(trp,k)
    tr=(~te)&(STIM!=1)&~np.isin(PID,list(vp))
    mu=X[tr].mean((0,2),keepdims=True); sd=X[tr].std((0,2),keepdims=True)
    net=EEGNet(); net.load_state_dict(torch.load(f"EEGNet_fold{k}.pt")); net.eval()
    for p in (pl,ph):
        for c,name in ((2,"target"),(1,"probe"),(0,"irrelevant")):
            m=(PID==p)&(STIM==c); R[(p,name)]=rel(net,(X[m]-mu)/sd)
tm=np.arange(128)/128*1000; win=(tm>=300)&(tm<=800)
out={}
for grp,gl in (("knowledgeable",[100+i for i in range(1,16)]),("control",list(range(1,16)))):
    for name in ("target","probe","irrelevant"):
        maps=np.stack([R[(p,name)]/R[(p,name)].sum() for p in gl])   # each person normalised to sum 1
        chp=maps.sum(2)                       # person x channel share
        frac=maps[:,:,win].sum((1,2))         # person share in 300-800 ms
        pk=tm[maps.sum(1).argmax(1)]
        top=[CH[i] for i in np.argsort(-chp.mean(0))[:4]]
        out[f"{grp}/{name}"]=dict(channel_share_mean=chp.mean(0).tolist(),channel_share_se=(chp.std(0,ddof=1)/np.sqrt(15)).tolist(),
            frac300_800_mean=float(frac.mean()),frac300_800_range=[float(frac.min()),float(frac.max())],
            peak_ms_median=float(np.median(pk)),top4=top,
            n_people_top_posterior=int(sum(np.argmax(r) in (0,1,2,4) for r in chp)),
            time_profile=maps.sum(1).mean(0).tolist())
        print(grp,name,top,"frac %.2f [%.2f-%.2f]"%(frac.mean(),frac.min(),frac.max()),"peak med",np.median(pk),"top-ch posterior in",out[f"{grp}/{name}"]["n_people_top_posterior"],"/15")
json.dump(out,open("saliency_probe.json","w"),indent=1)
