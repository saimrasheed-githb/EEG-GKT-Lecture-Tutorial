import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if "Carlito" in f: fm.fontManager.addfont(f)
plt.rcParams.update({"font.family":"Carlito","font.size":12})
R=json.load(open("results_Pz-bootstrap_LDA_SVM_EEGNet_CNN-LSTM.json"))
M=["Pz-bootstrap (no ML)","LDA","SVM","EEGNet","CNN-LSTM"]
codes=[f"L{i:02d}" for i in range(1,16)]+[f"H{i:02d}" for i in range(1,16)]
A=np.zeros((len(M),30)); D=np.zeros((len(M),30),int); Gt=np.array([1]*15+[0]*15)
for i,m in enumerate(M):
    d={r["code"]:r for r in R["persons"][m]}
    for j,c in enumerate(codes): A[i,j]=d[c]["auc"]; D[i,j]=d[c]["decision"]
fig,ax=plt.subplots(figsize=(13,3.6))
im=ax.imshow(A,cmap="RdBu_r",vmin=0.2,vmax=0.8,aspect="auto")
for i in range(len(M)):
    for j in range(30):
        err=D[i,j]!=Gt[j]
        ax.text(j,i,"✕" if err else ("●" if D[i,j] else "○"),ha="center",va="center",fontsize=11 if err else 9,
                color="black" if err else ("white" if abs(A[i,j]-0.5)>0.2 else "#1D2433"),fontweight="bold" if err else "normal",family="DejaVu Sans")
ax.set_xticks(range(30)); ax.set_xticklabels(codes,rotation=90,fontsize=10.5)
ax.set_yticks(range(len(M))); ax.set_yticklabels(["Pz (no ML)","LDA","SVM","EEGNet","CNN-LSTM"],fontsize=12)
ax.axvline(14.5,color="black",lw=2)
ax.text(7,-0.85,"Knowledgeable (lying folder)",ha="center",fontsize=12.5,color="#0F8B8D",fontweight="bold")
ax.text(22,-0.85,"Control (honest folder)",ha="center",fontsize=12.5,color="#14213D",fontweight="bold")
cb=fig.colorbar(im,ax=ax,fraction=0.02,pad=0.01); cb.set_label("within-person AUC (0.5 = no difference)",fontsize=10.5)
fig.text(0.01,-0.08,"●  called knowledgeable    ○  called not knowledgeable    ✕  decision error (wrong for the true group)",fontsize=11.5,family="DejaVu Sans")
fig.savefig("res_grid.png",dpi=200,bbox_inches="tight")
print("ok")
