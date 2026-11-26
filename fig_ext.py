"""Figures for the revision experiments: generalisation (Fig. 8) and SHAP attribution (Fig. 9)."""
import json, glob, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'axes.linewidth': 0.8, 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'savefig.dpi': 600, 'pdf.fonttype': 42})
COL = {'win_LightGBM': '#1baf7a', 'win_XGBoost': '#eb6834', 'CF-MIL-Boost': '#2a78d6'}
LAB = {'win_LightGBM': 'LightGBM (window)', 'win_XGBoost': 'XGBoost (window)', 'CF-MIL-Boost': 'CF-MIL-Boost'}
GRID = '#e5e5e5'
def load(tag):
    d = pd.DataFrame([json.loads(l) for l in open(f'ext/{tag}.jsonl')])
    return d.drop_duplicates([k for k in ['level', 'arm', 'rep', 'H', 'W'] if k in d.columns], keep='last')
# ---------- Fig 8: generalisation ----------
sets = [('main_bin', 'Binary\nstrat.'), ('lfo_bin', 'Binary\nLFO'),
        ('loco_bin', 'Binary\nLOCO'), ('main_cat10', '5-class\nstrat.'),
        ('lfo_cat', '5-class\nLFO')]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True, gridspec_kw={'wspace': 0.06})
for ax, lvl, tt in [(axs[0], 'window', '(a) Window level'), (axs[1], 'capture', '(b) Capture level')]:
    for i, (tag, name) in enumerate(sets):
        d = load(tag); d = d[d.level == lvl]
        for j, arm in enumerate(['win_LightGBM', 'win_XGBoost', 'CF-MIL-Boost']):
            v = d[d.arm == arm].mcc.values
            if len(v) == 0: continue
            x = i + (j - 1) * 0.27
            ax.bar(x, v.mean(), 0.25, color=COL[arm], label=LAB[arm] if (i == 0 and lvl == 'window') else None)
            ax.errorbar(x, v.mean(), stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)), color='#1f2328', lw=0.7, capsize=1.2)
    ax.set_xticks(range(len(sets))); ax.set_xticklabels([s[1] for s in sets], fontsize=7)
    ax.set_title(tt, fontsize=8, loc='left'); ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_ylim(0, 0.9)
axs[0].set_ylabel('MCC (mean, 95% CI)')
fig.legend(loc='lower center', ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.14))
fig.savefig('paperfigs/Fig8_generalisation.png', bbox_inches='tight'); fig.savefig('paperfigs/Fig8_generalisation.pdf', bbox_inches='tight')
# ---------- Fig 9: SHAP ----------
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={'wspace': 1.15})
pretty = {'mean': 'mean', 'max': 'max', 'median': 'median', 'q90': '90th pct.', 'std': 'std'}
CLS = {0: 'adware', 1: 'benign', 2: 'ransomware', 3: 'SMS', 4: 'scareware'}
for ax, t, tt in [(axs[0], 'bin', '(a) Binary detection'), (axs[1], 'cat', '(b) Five-class categorisation')]:
    fs = sorted(glob.glob(f'ext/pred/shap_main_{t}_r*.npz'))
    M = np.vstack([np.load(f)['mabs'] for f in fs]); names = np.load(fs[0])['names']; nS = int(np.load(fs[0])['n_score'])
    M = M / M.sum(1, keepdims=True)   # share of total attribution per repetition
    mean = M.mean(0); top = np.argsort(-mean)[:12][::-1]
    lab = []
    for i in top:
        n = str(names[i])
        if i < nS:
            s, k = n.rsplit('_', 1); lab.append(f'score {pretty[s]}' + ('' if t == 'bin' else f' ({CLS[int(k)]})'))
        elif n == 'log_count': lab.append('log flow count')
        else:
            s, f = n.split(':', 1); lab.append(f'{s} of {f}')
    ax.barh(range(len(top)), mean[top] * 100, color=['#2a78d6' if i < nS else '#9aa0a6' for i in top],
            xerr=M[:, top].std(0, ddof=1) * 100, error_kw=dict(lw=0.6, capsize=1))
    ax.set_yticks(range(len(top))); ax.set_yticklabels(lab, fontsize=6.4)
    share = M[:, :nS].sum(1)
    ax.set_title(f'{tt}\nscore statistics: {share.mean()*100:.0f}% of attribution ({nS} of {M.shape[1]} features)', fontsize=7.5, loc='left')
    ax.set_xlabel('Share of mean |SHAP| (%)'); ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.savefig('paperfigs/Fig9_shap.png', bbox_inches='tight'); fig.savefig('paperfigs/Fig9_shap.pdf', bbox_inches='tight')
