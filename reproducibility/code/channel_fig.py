import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Circle, Polygon
for f in fm.findSystemFonts():
    if "Carlito" in f: fm.fontManager.addfont(f)
plt.rcParams.update({"font.family":"Carlito","font.size":12})
R=json.load(open("channel_wise.json"))
POS={"Fp1":(-0.3,0.82),"Fp2":(0.3,0.82),"F3":(-0.42,0.42),"Fz":(0,0.40),"F4":(0.42,0.42),"C3":(-0.52,0),"Cz":(0,0),"C4":(0.52,0),
     "P3":(-0.42,-0.42),"Pz":(0,-0.40),"P4":(0.42,-0.42),"Oz":(0,-0.82)}
cmap=plt.get_cmap("YlGnBu")
fig,axs=plt.subplots(1,2,figsize=(11,5.3))
for ax,(key,title,lo,hi) in zip(axs,[("trial_auc","Trial AUC (target vs irrelevant epochs)",0.6,0.85),("person_auc","Person AUC (knowledgeable vs control)",0.7,0.96)]):
    ax.add_patch(Circle((0,0),1,fill=False,lw=2,color="#1D2433"))
    ax.add_patch(Polygon([(-0.1,0.99),(0,1.12),(0.1,0.99)],closed=True,fill=False,lw=2,color="#1D2433"))
    for c,(x,y) in POS.items():
        v=R[c][key]; col=cmap((v-lo)/(hi-lo))
        ax.add_patch(Circle((x,y),0.15,color=col,ec="#1D2433",lw=1))
        tc="white" if (v-lo)/(hi-lo)>0.55 else "#1D2433"
        ax.text(x,y+0.04,c,ha="center",va="center",fontsize=14,fontweight="bold",color=tc)
        ax.text(x,y-0.065,f"{v:.2f}",ha="center",va="center",fontsize=13,color=tc)
    ax.set_xlim(-1.15,1.15); ax.set_ylim(-1.1,1.2); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title,fontsize=16,fontweight="bold",color="#14213D")
    sm=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(lo,hi)); cb=fig.colorbar(sm,ax=ax,fraction=0.04,pad=0.02); cb.ax.tick_params(labelsize=12)
fig.text(0.5,0.1,"nose at top · single-channel LDA, 8 time-window features per channel · channel order from the raw .cnt header",ha="center",fontsize=13,color="#6B7585",style="italic")
fig.savefig("res_channels_map.png",dpi=200,bbox_inches="tight"); print("ok")
