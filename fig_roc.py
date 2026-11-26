import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':'#52514e','axes.linewidth':0.8,'xtick.color':'#52514e','ytick.color':'#52514e','savefig.dpi':600,'pdf.fonttype':42})
COL={'LightGBM':'#1baf7a','XGBoost':'#eb6834','CF-MIL-Boost':'#2a78d6'}
L=lambda t,a: np.load(f'mil/pred_{t}_r0_{a}.npz')
arms=[('CF-MIL-Boost','HYBRID_scores+raw'),('LightGBM','win_LightGBM'),('XGBoost','win_XGBoost')]
fig,axs=plt.subplots(2,2,figsize=(6.6,6.2),gridspec_kw={'wspace':0.35,'hspace':0.45}); axs=axs.ravel()
for ax,lvl,tt in [(axs[0],'window','(a) Binary, window'),(axs[1],'capture','(b) Binary, capture')]:
    for n,a in arms:
        z=L('bin',a); p,y,c=z['p'],z['y'],z['cap']
        if lvl=='capture':
            d=pd.DataFrame({'p':p,'y':y,'c':c}).groupby('c').agg(p=('p','mean'),y=('y','first')); p,y=d.p.values,d.y.values
        f,t,_=roc_curve(y,p); ax.plot(f,t,color=COL[n],lw=1.3 if n=='CF-MIL-Boost' else 1.0,label=f'{n} ({roc_auc_score(y,p):.3f})')
    ax.plot([0,1],[0,1],':',color='#9aa0a6',lw=0.8); ax.set_xlabel('False-positive rate'); ax.set_ylabel('True-positive rate')
    ax.set_title(tt,fontsize=8,loc='left'); ax.legend(fontsize=7,frameon=False,loc='lower right',handlelength=1.2); ax.set_aspect('equal')
names={1:'Benign',0:'Adware',2:'Ransom.',4:'Scare.',3:'SMS'}; order=[1,0,2,4,3]
rec={}
for ax,(n,a),tt in [(axs[2],arms[0],'(c) Five-class, window: CF-MIL-Boost'),(axs[3],arms[2],'(d) Five-class, window: XGBoost')]:
    z=L('cat',a); y=z['y']; pr=z['p'].argmax(1)
    cm=confusion_matrix(y,pr,labels=order).astype(float); cm=cm/cm.sum(1,keepdims=True); rec[n]=np.diag(cm)
    ax.imshow(cm,cmap='Blues',vmin=0,vmax=1)
    for i in range(5):
        for j in range(5): ax.text(j,i,f'{cm[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if cm[i,j]>0.55 else '#1f2328')
    ax.set_xticks(range(5)); ax.set_xticklabels(['Ben.','Adw.','Ran.','Sca.','SMS'],fontsize=7)
    ax.set_yticks(range(5)); ax.set_yticklabels([names[k] for k in order],fontsize=7)
    ax.set_xlabel('Predicted',fontsize=7); ax.set_ylabel('True (windows)',fontsize=7); ax.set_title(tt,fontsize=8,loc='left')
    for s in ax.spines.values(): s.set_visible(False)
fig.savefig('paperfigs/Fig6_roc_cm.png',bbox_inches='tight'); fig.savefig('paperfigs/Fig6_roc_cm.pdf',bbox_inches='tight')
print({k:np.round(v,3) for k,v in rec.items()}, [names[k] for k in order])
for n,a in arms:
    z=L('cat',a); y=z['y']; pr=z['p'].argmax(1); cm=confusion_matrix(y,pr,labels=order); print(n,'malware->benign share', round(cm[1:,0].sum()/cm[1:].sum(),3))
