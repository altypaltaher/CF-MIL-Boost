import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#52514e', 'axes.linewidth': 0.8, 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'savefig.dpi': 600, 'pdf.fonttype': 42})
import sys; TASK = sys.argv[1] if len(sys.argv) > 1 else 'bin'
D = pd.read_json(f'mil/{TASK}_results.jsonl', lines=True).drop_duplicates(['level', 'arm', 'rep'], keep='last')
COL = {'CF-MIL-Boost': '#2a78d6', 'XGBoost': '#eb6834', 'LightGBM': '#1baf7a', 'GBM': '#eda100', 'AdaBoost': '#6250d6'}
arms = {'flow': {'LightGBM': 'lgb_flow', 'XGBoost': 'xgb_flow', 'AdaBoost': 'ada_flow', 'GBM': 'gbm_flow'},
        'window': {'LightGBM': 'win_LightGBM', 'XGBoost': 'win_XGBoost', 'AdaBoost': 'win_AdaBoost', 'GBM': 'win_GBM', 'CF-MIL-Boost': 'HYBRID_scores+raw'},
        'capture': {'LightGBM': 'win_LightGBM', 'XGBoost': 'win_XGBoost', 'AdaBoost': 'win_AdaBoost', 'GBM': 'win_GBM', 'CF-MIL-Boost': 'HYBRID_scores+raw'}}
titles = {'flow': 'Single\nflow', 'window': '1-min\nwindow', 'capture': 'App capture\n(first 5 min)'}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={'wspace': 0.32})
for ax, met, lab in [(axs[0], 'mcc', 'MCC'), (axs[1], 'auc', 'ROC-AUC')]:
    names = ['LightGBM', 'XGBoost', 'AdaBoost', 'GBM', 'CF-MIL-Boost']; bw = 0.16
    for li, lvl in enumerate(['flow', 'window', 'capture']):
        for ni, n in enumerate(names):
            if n not in arms[lvl]: continue
            v = D[(D.level == lvl) & (D.arm == arms[lvl][n])][met].values
            m = v.mean(); ci = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            x = li + (ni - 2) * bw
            ax.bar(x, m, bw - 0.025, color=COL[n], label=n if li == 1 else None)
            ax.errorbar(x, m, ci, color='#0b0b0b', lw=0.8, capsize=1.5)
            if n == 'CF-MIL-Boost': ax.text(x, m + ci + 0.015, f'{m:.2f}', ha='center', va='bottom', fontsize=7, color='#0b0b0b')
    ax.set_xticks(range(3)); ax.set_xticklabels([titles[l] for l in ['flow', 'window', 'capture']], fontsize=7.5)
    ax.set_ylabel(lab); ax.set_ylim(0 if met == 'mcc' else 0.5, 1.0); ax.yaxis.grid(True, color='#e4e3df', lw=0.6); ax.set_axisbelow(True)
axs[0].legend(ncol=5, loc='upper center', bbox_to_anchor=(1.1, -0.2), frameon=False, fontsize=7.5, handlelength=1)
fig.savefig(f'figs/MIL_levels_{TASK}.png', bbox_inches='tight'); fig.savefig(f'figs/MIL_levels_{TASK}.pdf', bbox_inches='tight')
