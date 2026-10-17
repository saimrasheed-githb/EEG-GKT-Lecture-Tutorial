import sys, time, numpy as np, torch, torch.nn as nn
from common import *
torch.set_num_threads(2)
MODEL=sys.argv[1]
class EEGNet(nn.Module):
    def __init__(s,C=12,T=128,F1=8,D=2,F2=16,k=64,p=0.5):
        super().__init__()
        s.b1=nn.Sequential(nn.Conv2d(1,F1,(1,k),padding=(0,k//2),bias=False),nn.BatchNorm2d(F1),
            nn.Conv2d(F1,F1*D,(C,1),groups=F1,bias=False),nn.BatchNorm2d(F1*D),nn.ELU(),nn.AvgPool2d((1,4)),nn.Dropout(p))
        s.b2=nn.Sequential(nn.Conv2d(F1*D,F1*D,(1,16),padding=(0,8),groups=F1*D,bias=False),nn.Conv2d(F1*D,F2,1,bias=False),
            nn.BatchNorm2d(F2),nn.ELU(),nn.AvgPool2d((1,8)),nn.Dropout(p))
        with torch.no_grad(): n=s.b2(s.b1(torch.zeros(1,1,C,T))).numel()
        s.fc=nn.Linear(n,1)
    def forward(s,x): return s.fc(s.b2(s.b1(x.unsqueeze(1))).flatten(1)).squeeze(1)
class CNNLSTM(nn.Module):
    def __init__(s,C=12,F1=8,D=2,k=32,p=0.5,H=32):
        super().__init__()
        s.cnn=nn.Sequential(nn.Conv2d(1,F1,(1,k),padding=(0,k//2),bias=False),nn.BatchNorm2d(F1),
            nn.Conv2d(F1,F1*D,(C,1),groups=F1,bias=False),nn.BatchNorm2d(F1*D),nn.ELU(),nn.AvgPool2d((1,4)),nn.Dropout(p))
        s.lstm=nn.LSTM(F1*D,H,batch_first=True); s.fc=nn.Linear(H,1); s.dp=nn.Dropout(p)
    def forward(s,x):
        z=s.cnn(x.unsqueeze(1)).squeeze(2).transpose(1,2)   # batch x time x features
        o,(h,c)=s.lstm(z); return s.fc(s.dp(h[-1])).squeeze(1)
Net={"EEGNet":EEGNet,"CNN-LSTM":CNNLSTM}[MODEL]
scores=np.full(len(X),np.nan); log=[]
t0=time.time()
for k,(pl,ph) in enumerate(PAIRS,1):
    torch.manual_seed(0); np.random.seed(0)
    te=np.isin(PID,[pl,ph]); trp=[p for p in PIDS if p not in (pl,ph)]
    vp=inner_val(trp,k)
    base=(~te)&(STIM!=1)
    va=base&np.isin(PID,list(vp)); tr=base&~np.isin(PID,list(vp))
    mu=X[tr].mean((0,2),keepdims=True); sd=X[tr].std((0,2),keepdims=True)
    f=lambda m: torch.tensor((X[m]-mu)/sd,dtype=torch.float32)
    Xtr,Xva,Xte=f(tr),f(va),f(te)
    ytr=torch.tensor((STIM[tr]==2).astype(np.float32)); yva=torch.tensor((STIM[va]==2).astype(np.float32))
    pw=torch.tensor((ytr==0).sum()/(ytr==1).sum())
    lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
    net=Net(); opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    best=(1e9,None,0); bad=0; g=torch.Generator().manual_seed(0)
    for ep in range(30):
        net.train(); perm=torch.randperm(len(Xtr),generator=g)
        for i in range(0,len(perm),64):
            b=perm[i:i+64]; opt.zero_grad(); l=lossf(net(Xtr[b]),ytr[b]); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            trl=float(np.mean([lossf(net(Xtr[i:i+2048]),ytr[i:i+2048]).item() for i in range(0,len(Xtr),2048)]))
            vl=lossf(net(Xva),yva).item()
        log.append((k,ep,trl,vl))
        if vl<best[0]-1e-4: best=(vl,{kk:v.clone() for kk,v in net.state_dict().items()},ep); bad=0
        else:
            bad+=1
            if bad>=5: break
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad(): scores[te]=torch.cat([net(Xte[i:i+2048]) for i in range(0,len(Xte),2048)]).numpy()
    if k==1 and MODEL=="EEGNet": torch.save(net.state_dict(),"eegnet_fold1.pt")
    torch.save(net.state_dict(),f"{MODEL}_fold{k}.pt")
    print(f"{MODEL} fold {k} best_epoch {best[2]} val_loss {best[0]:.3f} elapsed {time.time()-t0:.0f}s",flush=True)
np.savez(f"scores_{MODEL}.npz",scores=scores,log=np.array(log))
print("done",flush=True)
