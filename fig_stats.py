import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':'#52514e','axes.linewidth':0.8,'xtick.color':'#52514e','ytick.color':'#52514e','savefig.dpi':600,'pdf.fonttype':42})
INK='#1f2328'; Q='#57606a'; BLUE='#2a78d6'; ORA='#eb6834'; GRY='#9aa0a6'; GRID='#e5e5e5'
COL={'LightGBM':'#1baf7a','XGBoost':'#eb6834','AdaBoost':'#6250d6','GBM':'#eda100','CF-MIL-Boost':'#2a78d6'}
R={t:pd.DataFrame([json.loads(l) for l in open(f'mil/{t}_results.jsonl')]) for t in ['bin','cat']}
def vals(t,lvl,arm,met='mcc'):
    d=R[t]; d=d[(d.level==lvl)&(d.arm==arm)].sort_values('rep'); return d[met].values
# ---------- Fig 2: evaluation artefacts ----------
D=pd.read_parquet('ctx_flows.parquet',columns=['cid','cat','t'])
cap=D.groupby('cid').agg(cat=('cat','first'),dur=('t','max'))
names={1:'Benign',0:'Adware',2:'Ransomware',4:'Scareware',3:'SMS malware'}
fig,axs=plt.subplots(1,3,figsize=(7.4,2.6),gridspec_kw={'wspace':0.5,'width_ratios':[1.35,1,1]})
ax=axs[0]; order=[1,0,2,4,3]
data=[cap[cap.cat==c].dur.values/60 for c in order]
bp=ax.boxplot(data,orientation='horizontal',widths=0.55,patch_artist=True,showfliers=False,medianprops=dict(color=INK,lw=1.2),whiskerprops=dict(color=Q),capprops=dict(color=Q))
for i,b in enumerate(bp['boxes']): b.set(facecolor=GRY if i==0 else ORA,alpha=0.55,edgecolor=Q)
ax.set_yticks(range(1,6)); ax.set_yticklabels([names[c] for c in order]); ax.invert_yaxis()
ax.axvline(5,color=BLUE,ls='--',lw=1); ax.text(5.4,4.55,'300 s',color=BLUE,fontsize=6.8,va='center')
ax.set_xscale('log'); ax.set_xlabel('Capture length (min, log scale)'); ax.set_title('(a) Capture length by class',fontsize=8,loc='left',color=INK)
ax.set_xlim(2,260); from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
ax.xaxis.set_major_locator(FixedLocator([2,5,10,30,60,180])); ax.xaxis.set_major_formatter(FixedFormatter(['2','5','10','30','60','180'])); ax.xaxis.set_minor_locator(NullLocator())
for i,c in enumerate(order): ax.text(140, i+1, f'{np.median(cap[cap.cat==c].dur)/60:.0f}',fontsize=6.3,va='center',color=Q,ha='center')
ax.text(140,0.62,'median',fontsize=6.3,color=Q,va='center',ha='center')
ax.grid(axis='x',color=GRID,lw=0.6); ax.set_axisbelow(True)
# (b) time-only AUC
ax=axs[1]; ctl=vals('bin','window','CONTROL_time_index_only','auc')
lab=['Flow time\n(full)','Flow time\n(300 s)','Window\nindex']
RS=json.load(open('repro_stats.json'))  # produced by repro_stats.py
v=[RS['auc_elapsed_time_full'],RS['auc_elapsed_time_300s'],ctl.mean()]; e=[0,0,stats.t.ppf(.975,len(ctl)-1)*ctl.std(ddof=1)/np.sqrt(len(ctl))]
ax.bar(range(3),v,0.6,color=[ORA,BLUE,BLUE],alpha=0.85); ax.errorbar(2,v[2],e[2],color=INK,capsize=2,lw=0.8)
ax.axhline(0.5,color=Q,ls=':',lw=0.9); ax.text(-0.45,0.51,'chance',fontsize=6.3,color=Q,ha='left')
for i,x in enumerate(v): ax.text(i,x+(0.02 if i<2 else 0.035),f'{x:.3f}',ha='center',fontsize=6.8)
ax.set_xticks(range(3)); ax.set_xticklabels(lab,fontsize=6.3); ax.set_ylim(0.4,1.0); ax.set_ylabel('ROC-AUC of time alone')
ax.set_title('(b) Capture-length shortcut',fontsize=8,loc='left',color=INK); ax.grid(axis='y',color=GRID,lw=0.6); ax.set_axisbelow(True)
# (c) random vs grouped
ax=axs[2]; g=vals('bin','flow','lgb_flow'); r=vals('bin','flow','lgb_flow_RANDOMsplit')
for i in range(len(g)): ax.plot([0,1],[g[i],r[i]],color=GRY,lw=0.7,zorder=1)
ax.scatter(np.zeros(len(g)),g,s=14,color=BLUE,zorder=2); ax.scatter(np.ones(len(r)),r,s=14,color=ORA,zorder=2)
ax.set_xticks([0,1]); ax.set_xticklabels(['Grouped\nby capture','Random\nflow split'],fontsize=6.6); ax.set_xlim(-0.4,1.4)
ax.set_ylabel('Flow-level MCC (LightGBM)'); ax.set_title('(c) Split-induced leakage',fontsize=8,loc='left',color=INK)
ax.text(1.0,0.15,f'mean +{(r.mean()/g.mean()-1)*100:.0f}%',ha='center',fontsize=7,color=ORA,weight='bold')
ax.grid(axis='y',color=GRID,lw=0.6); ax.set_axisbelow(True)
fig.savefig('paperfigs/Fig2_artefacts.png',bbox_inches='tight'); fig.savefig('paperfigs/Fig2_artefacts.pdf',bbox_inches='tight')
print('median durations', {names[c]: float(np.median(cap[cap.cat==c].dur)) for c in order}, 'ratio', r.mean()/g.mean())
# ---------- Fig 5: forest plot of corrected differences ----------
def nb(d,ratio=0.25):
    n=len(d); m=d.mean(); v=d.var(ddof=1); se=np.sqrt((1/n+ratio)*v); tt=m/se; p=2*stats.t.sf(abs(tt),n-1)
    h=stats.t.ppf(.975,n-1)*se; return m,m-h,m+h,p
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),sharey=True,gridspec_kw={'wspace':0.08})
bases=['LightGBM','XGBoost','AdaBoost','GBM']; rows=[]
for ax,t,tt in [(axs[0],'bin','(a) Binary detection (10 repetitions)'),(axs[1],'cat','(b) Five-class categorisation (5 repetitions)')]:
    y=0; yt=[]; yl=[]
    for lvl,ln in [('window','Window'),('capture','Capture')]:
        for b in bases:
            d=(vals(t,lvl,'HYBRID_scores+raw')-vals(t,lvl,f'win_{b}'))*100
            m,lo,hi,p=nb(d); rows.append((t,lvl,b,m,lo,hi,p))
            ax.plot([lo,hi],[y,y],color=COL[b],lw=1.6); ax.scatter([m],[y],s=22,color=COL[b],zorder=3,marker='D' if lvl=='capture' else 'o')
            ax.text(49,y,('p<0.001' if p<0.001 else f'p={p:.3f}'),fontsize=6.3,va='center',color=Q)
            yt.append(y); yl.append(f'{ln} vs {b}'); y+=1
        y+=0.6
    ax.axvline(0,color=INK,lw=0.8); ax.set_yticks(yt); ax.set_yticklabels(yl,fontsize=6.8)
    ax.set_xlim(-5,60); ax.set_xlabel('ΔMCC, CF-MIL-Boost − baseline (pp)'); ax.set_title(tt,fontsize=8,loc='left')
    ax.grid(axis='x',color=GRID,lw=0.6); ax.set_axisbelow(True)
axs[0].invert_yaxis()
fig.savefig('paperfigs/Fig5_forest.png',bbox_inches='tight'); fig.savefig('paperfigs/Fig5_forest.pdf',bbox_inches='tight')
pd.DataFrame(rows,columns=['task','level','base','d','lo','hi','p']).to_csv('paperfigs/forest_ci.csv',index=False)
print(pd.DataFrame(rows,columns=['task','level','base','d','lo','hi','p']).round(4).to_string())
# ---------- Fig 7: ablation ----------
arms=[('CF-MIL-Boost','HYBRID_scores+raw',BLUE),('Scores only','HYBRID_scores_only','#7fb0ea'),('Raw stats only\n(window LightGBM)','win_LightGBM','#1baf7a'),
      ('MIL-LR on XGBoost\nflow scores','xgb_flow+MIL','#eb6834'),('MIL-LR on LightGBM\nflow scores','lgb_flow+MIL','#f2a07a'),
      ('Mean vote\n(LightGBM flows)','lgb_flow+meanvote','#9aa0a6'),('Host context\n+ MIL-LR','lgb_ctx+MIL','#6250d6')]
fig,axs=plt.subplots(1,2,figsize=(7.2,2.7),sharey=True,gridspec_kw={'wspace':0.06})
for ax,t,tt in [(axs[0],'bin','(a) Binary'),(axs[1],'cat','(b) Five-class')]:
    for li,lvl in enumerate(['window','capture']):
        for ai,(n,a,c) in enumerate(arms):
            v=vals(t,lvl,a); x=li*(len(arms)+1.2)+ai
            ax.bar(x,v.mean(),0.82,color=c,label=n if li==0 and t=='bin' else None)
            ax.errorbar(x,v.mean(),stats.t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)),color=INK,lw=0.7,capsize=1.2)
            if ai==0: ax.text(x,v.mean()+0.04,f'{v.mean():.3f}',ha='center',fontsize=6.3)
    ax.set_xticks([(len(arms)-1)/2,(len(arms)+1.2)+(len(arms)-1)/2]); ax.set_xticklabels(['Window level','Capture level'])
    ax.set_title(tt,fontsize=8,loc='left'); ax.set_ylim(0,0.9); ax.grid(axis='y',color=GRID,lw=0.6); ax.set_axisbelow(True)
axs[0].set_ylabel('MCC (mean, 95% CI)')
fig.legend(loc='lower center',ncol=4,frameon=False,fontsize=6.6,bbox_to_anchor=(0.5,-0.2),handlelength=1)
fig.savefig('paperfigs/Fig7_ablation.png',bbox_inches='tight'); fig.savefig('paperfigs/Fig7_ablation.pdf',bbox_inches='tight')
