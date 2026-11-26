import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'savefig.dpi':600,'pdf.fonttype':42})
INK='#1f2328'; Q='#57606a'; EDGE='#6e7781'
BLUE='#2a78d6'; BLUEL='#e3eefb'; ORA='#e8833a'; ORAL='#fdeee2'; GRN='#1baf7a'; GRNL='#e1f5ee'; GRY='#f3f4f6'
fig=plt.figure(figsize=(7.2,3.55)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,720); ax.set_ylim(0,350); ax.axis('off')
def box(x,y,w,h,fc='white',ec=EDGE,lw=0.9,r=6,ls='-'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',fc=fc,ec=ec,lw=lw,ls=ls))
def arr(x1,y1,x2,y2,c=EDGE,lw=1.0,cs='arc3'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=8,color=c,lw=lw,connectionstyle=cs,shrinkA=0,shrinkB=0))
def T(x,y,s,**k):
    k.setdefault('ha','center'); k.setdefault('va','center'); k.setdefault('color',INK); ax.text(x,y,s,**k)
# column headers
for x,s in [(62,'(a) Input'),(178,'(b) Bags'),(318,'(c) Stage 1'),(468,'(d) Window vector'),(590,'(e) Stage 2'),(677,'(f) Decision')]:
    T(x,336,s,fontsize=7.5,weight='bold',color=Q)
# (a) capture timeline
box(8,128,108,192,fc=GRY,ec=GRY)
T(62,305,'Capture $c$',weight='bold'); T(62,292,'one app run, first 300 s',fontsize=6.5,color=Q)
rng=np.random.RandomState(3)
for j in range(5):
    y0=270-j*24
    ax.add_patch(Rectangle((24,y0-8),84,16,fc='white',ec=EDGE,lw=0.6))
    for xx in 24+84*rng.rand(rng.randint(5,11)): ax.plot([xx,xx],[y0-5,y0+5],color=BLUE,lw=0.9)
    T(16,y0,f'{j}',fontsize=5.5,color=Q)
T(62,146,'flows $x_i$ at start\ntimes $t_i$ (80 features)',fontsize=6.0,color=Q)
arr(118,230,140,230)
# (b) bags
box(142,128,74,192,fc=GRY,ec=GRY)
T(179,305,'60-s windows',weight='bold'); T(179,292,'bags $B_{c,j}$',fontsize=6.5,color=Q)
for j in range(5):
    y0=270-j*24
    box(157,y0-8,44,16,fc='white',ec=EDGE,lw=0.6,r=3)
    for k in range(5): ax.add_patch(Rectangle((161+k*8,y0-4),5,8,fc=BLUE,ec='none',alpha=0.75))
T(179,146,'bag label =\ncapture label',fontsize=6.0,color=Q)
arr(218,230,236,230)
# (c) stage 1
box(238,128,166,192,fc=BLUEL,ec=BLUE,lw=1.3)
T(321,305,'Flow-level LightGBM $f$',weight='bold')
T(321,292,'instance score $\\tilde{s}_i$ for every flow',fontsize=6.5,color=Q)
T(321,274,'Training captures: 3-fold',fontsize=6.3); T(321,263,'cross-fitting by capture',fontsize=6.3)
for k in range(3):
    y0=244-k*17
    for m in range(3):
        sc=(m==k)
        ax.add_patch(FancyBboxPatch((252+m*34,y0-6),30,12,boxstyle='round,pad=0,rounding_size=2',fc=ORA if sc else 'white',ec=ORA if sc else BLUE,lw=0.7))
        T(267+m*34,y0,f'$T_{m+1}$',fontsize=5.8,color='white' if sc else INK)
    T(358,y0,f'$\\rightarrow f^{{(-{k+1})}}$',fontsize=6.2,ha='left')
ax.add_patch(Rectangle((256,191),9,7,fc='white',ec=BLUE,lw=0.7)); T(268,194.5,'used to fit',fontsize=5.8,ha='left')
ax.add_patch(Rectangle((318,191),9,7,fc=ORA,ec=ORA,lw=0.7)); T(330,194.5,'held out, scored',fontsize=5.8,ha='left')
ax.plot([248,394],[180,180],color=BLUE,lw=0.5,ls=(0,(2,2)))
T(321,163,'Validation/test captures: scored\nby $f$ fitted on all training captures',fontsize=6.1)
T(321,140,'$\\leq$150 flows per capture; early stopping',fontsize=5.4,color=Q)
arr(406,230,420,230)
# (d) window vector
box(422,236,96,84,fc=ORAL,ec=ORA,lw=1.0)
T(470,306,'Score statistics',weight='bold',fontsize=7)
T(470,272,'mean, max, median,\n90th pct. and std of $\\tilde{s}$\nper class\n(5 or 25 features)',fontsize=6.0)
box(422,146,96,82,fc=GRNL,ec=GRN,lw=1.0)
T(470,214,'Raw statistics',weight='bold',fontsize=7)
T(470,180,'mean, std, max of\nsigned-log features\n+ log flow count\n(241 features)',fontsize=6.0)
T(470,133,'$\\mathbf{z}_{c,j}$ = [score, raw]',fontsize=6.4)
arr(520,230,540,230)
# (e) stage 2
box(542,146,96,174,fc=BLUEL,ec=BLUE,lw=1.3)
T(590,305,'Window-level',weight='bold'); T(590,293,'LightGBM $h$',weight='bold')
T(590,262,'class-balanced\nweights',fontsize=6.2)
T(590,222,'6 configurations,\nearly stopping,\nselected by\nvalidation MCC',fontsize=6.2)
T(590,170,'$h(\\mathbf{z}_{c,j})$: window\nprobabilities',fontsize=6.2,color=Q)
arr(640,245,656,245)
# (f) decisions
box(658,262,56,58,fc='white',ec=EDGE)
T(686,305,'Window',weight='bold',fontsize=7); T(686,282,'alert after\n1 min',fontsize=6.0,color=Q)
box(658,146,56,106,fc='white',ec=EDGE)
T(686,238,'Capture',weight='bold',fontsize=7); T(686,198,'$\\hat p(c)$:\nmean of\nwindow\nprobabilities;\n$\\hat y=\\arg\\max_k$',fontsize=5.9)
arr(686,262,686,253)
# bottom band
box(8,8,706,104,fc='white',ec=EDGE,lw=0.7,ls=(0,(3,2)))
T(20,98,'Leakage-free protocol (applied identically to the proposed model and to every baseline)',ha='left',weight='bold',fontsize=7)
cols=[('Grouped split by capture','captures, never flows, are split:\n20% test; the rest into fit 70%,\nvalidation 15%, aggregation 15%'),
      ('300-s horizon','removes the capture-length\nshortcut (elapsed-time flow\nAUC 0.877 → 0.494)'),
      ('Time-only control','window position as the only\nscore, every repetition\n(AUC 0.498 ± 0.002)'),
      ('Identical baselines','same windows, captures,\nraw statistics, weights and\nselection rule (MCC)')]
for i,(h,b) in enumerate(cols):
    x=20+i*176
    T(x,78,h,ha='left',weight='bold',fontsize=6.6,color=BLUE)
    T(x,44,b,ha='left',fontsize=6.1)
fig.savefig('paperfigs/Fig1_pipeline.png'); fig.savefig('paperfigs/Fig1_pipeline.pdf')
