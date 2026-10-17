import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if "Carlito" in f: fm.fontManager.addfont(f)
plt.rcParams.update({"font.family":"Carlito","font.size":16})
d=json.load(open("saliency_probe.json"))
CH=["Oz","Pz","P4","C4","P3","Cz","C3","Fz","F4","F3","Fp2","Fp1"]
order=[11,10,9,7,8,6,5,3,4,1,2,0]   # front to back: Fp1 Fp2 F3 Fz F4 C3 Cz C4 P3 Pz P4 Oz
conds=[("knowledgeable/probe","Probe epochs · knowledgeable (15)","#C0392B"),("control/probe","Probe epochs · control (15)","#1F4E79"),("knowledgeable/irrelevant","Irrelevant epochs · knowledgeable (15)","#8A94A6")]
fig,(a,b)=plt.subplots(1,2,figsize=(10.5,4.9),gridspec_kw=dict(width_ratios=[1.25,1],wspace=0.3))
y=np.arange(12); h=0.26
for j,(k,lab,c) in enumerate(conds):
    m=np.array(d[k]["channel_share_mean"])[order]*100; se=np.array(d[k]["channel_share_se"])[order]*100
    a.barh(y+(j-1)*h,m,h,xerr=se,color=c,label=lab,error_kw=dict(lw=0.8,capsize=1.5,ecolor="#333"))
a.set_yticks(y); a.set_yticklabels([CH[i] for i in order]); a.invert_yaxis(); a.set_xlabel("% of relevance per channel (mean ± SE)")
a.spines[["top","right"]].set_visible(False); fig.legend(*a.get_legend_handles_labels(),fontsize=14,frameon=False,loc="upper center",ncol=3,bbox_to_anchor=(0.5,1.08))
tm=np.arange(128)/128*1000
for k,lab,c in conds: b.plot(tm,np.array(d[k]["time_profile"])*100,color=c,lw=2)
b.axvspan(300,800,color="#0F8B8D",alpha=0.08); b.text(550,b.get_ylim()[1]*0.97 if False else 0,"",ha="center")
b.set_xlabel("ms from stimulus onset"); b.set_ylabel("% of relevance per sample"); b.spines[["top","right"]].set_visible(False)
b.text(550,0.15,"300–800 ms",ha="center",fontsize=11.5,color="#0F8B8D")
fig.text(0.01,-0.05,"Held-out people only · each person normalised to 100% · channel names from the raw .cnt header (one file checked)",fontsize=11.5,color="#6B7585")
fig.savefig("res_sal_probe.png",dpi=200,bbox_inches="tight")
for k in d: print(k, "Fp1+Fp2 %.1f%%"%(100*(d[k]["channel_share_mean"][10]+d[k]["channel_share_mean"][11])), "parietal-occ %.1f%%"%(100*sum(d[k]["channel_share_mean"][i] for i in (0,1,2,4))))
