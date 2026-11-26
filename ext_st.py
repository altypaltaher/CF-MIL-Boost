"""Summaries of the revision experiments (ext/*.jsonl, ext/pred/*.npz) -> ext/summary.json and printed tables."""
import json, glob, os, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, precision_recall_fscore_support, confusion_matrix
W = os.environ.get('CFMIL_WORK', os.path.dirname(os.path.abspath(__file__))) + '/'
E = W + 'ext/'; OUT = {}
def load(tag):
    f = E + f'{tag}.jsonl'
    if not os.path.exists(f): return pd.DataFrame()
    d = pd.DataFrame([json.loads(l) for l in open(f)])
    return d.drop_duplicates(['level', 'arm', 'rep', 'H', 'W'], keep='last')
def nb(d, ratio):
    n = len(d); m = d.mean(); v = d.var(ddof=1)
    if n < 2 or v == 0: return m, np.nan, np.nan, np.nan
    se = np.sqrt((1 / n + ratio) * v); t = m / se; p = 2 * stats.t.sf(abs(t), n - 1); h = stats.t.ppf(.975, n - 1) * se
    return m, m - h, m + h, p
def table(d, ratio, label):
    rows = []
    for lvl in ('window', 'capture'):
        x = d[d.level == lvl]
        for arm in ('CF-MIL-Boost', 'win_LightGBM', 'win_XGBoost'):
            y = x[x.arm == arm].sort_values('rep')
            if len(y) == 0: continue
            rows.append(dict(set=label, level=lvl, arm=arm, n=len(y), mcc=y.mcc.mean(), mcc_sd=y.mcc.std(ddof=1), f1m=y.f1m.mean(), auc=y.auc.mean(), auc_sd=y.auc.std(ddof=1)))
        cf = x[x.arm == 'CF-MIL-Boost'].set_index('rep')
        for b in ('win_LightGBM', 'win_XGBoost'):
            bb = x[x.arm == b].set_index('rep')
            r = cf.index.intersection(bb.index)
            if len(r) < 2: continue
            dd = (cf.loc[r, 'mcc'] - bb.loc[r, 'mcc']).values
            m, lo, hi, p = nb(dd, ratio)
            rows.append(dict(set=label, level=lvl, arm=f'delta_vs_{b}', n=len(r), mcc=m, lo=lo, hi=hi, p=p, wins=int((dd > 0).sum())))
    return rows
R = []
for tag, ratio in [('main_bin', 0.25), ('main_cat', 0.25), ('lfo_bin', 0.25), ('lfo_cat', 0.25), ('loco_bin', 1 / 3)]:
    d = load(tag)
    if len(d): R += table(d, ratio, tag)
T = pd.DataFrame(R); OUT['tables'] = T.to_dict('records')
pd.set_option('display.width', 220); print(T.round(4).to_string())
# LOCO per held-out category
d = load('loco_bin')
if len(d):
    print('\nLOCO per fold'); print(d[d.arm != 'CONTROL_time_index_only'].pivot_table(index=['rep', 'level'], columns='arm', values=['mcc', 'auc']).round(3))
# controls
for tag in ('main_bin', 'lfo_bin', 'loco_bin', 'sens_bin', 'nested_bin'):
    d = load(tag)
    if len(d) and (d.arm == 'CONTROL_time_index_only').any():
        c = d[d.arm == 'CONTROL_time_index_only'].groupby(['H', 'W']).auc.agg(['mean', 'std', 'count'])
        print('\ncontrol', tag); print(c.round(4)); OUT[f'control_{tag}'] = c.reset_index().to_dict('records')
# nested vs fixed stage-1 configuration (same splits)
for t in ('bin', 'cat'):
    n, m = load(f'nested_{t}'), load(f'main_{t}')
    if len(n) and len(m):
        rows = []
        for lvl in ('window', 'capture'):
            a = n[(n.level == lvl) & (n.arm == 'CF-MIL-Boost')].set_index('rep').mcc
            b = m[(m.level == lvl) & (m.arm == 'CF-MIL-Boost')].set_index('rep').mcc
            c = n[(n.level == lvl) & (n.arm == 'win_LightGBM')].set_index('rep').mcc
            r = a.index.intersection(b.index)
            dd = (a.loc[r] - c.loc[r]).values; mm, lo, hi, p = nb(dd, 0.25)
            rows.append(dict(task=t, level=lvl, n=len(r), nested=a.loc[r].mean(), nested_sd=a.loc[r].std(ddof=1), fixed=b.loc[r].mean(), lgb=c.loc[r].mean(),
                             d_vs_lgb=mm, lo=lo, hi=hi, p=p, wins=int((dd > 0).sum())))
        print('\nnested', t); print(pd.DataFrame(rows).round(4)); OUT[f'nested_{t}'] = rows
        print('selected stage-1 configs:', n[(n.arm == 'CF-MIL-Boost') & (n.level == 'window')].s1cfg.value_counts().to_dict())
# sensitivity
s = load('sens_bin'); m = load('main_bin')
if len(s):
    s = pd.concat([s, m[m.rep.isin(s.rep.unique()) & m.arm.isin(['CF-MIL-Boost', 'win_LightGBM', 'CONTROL_time_index_only'])]])
    piv = s.pivot_table(index=['H', 'W', 'level'], columns='arm', values='mcc', aggfunc='mean')
    ctl = s[s.arm == 'CONTROL_time_index_only'].groupby(['H', 'W']).auc.mean()
    print('\nsensitivity (3 reps)'); print(piv.round(3)); print(ctl.round(3))
    OUT['sens'] = piv.reset_index().to_dict('records'); OUT['sens_ctl'] = ctl.reset_index().to_dict('records')
# timing
for t in ('bin', 'cat'):
    d = load(f'main_{t}')
    if len(d):
        x = d[(d.level == 'window') & d.arm.isin(['CF-MIL-Boost', 'win_LightGBM', 'win_XGBoost'])].groupby('arm')[['train_s', 'infer_ms_per_window']].agg(['mean', 'std'])
        print('\ntiming', t); print(x.round(3)); OUT[f'timing_{t}'] = {a: dict(train_s=float(x.loc[a, ('train_s', 'mean')]), infer=float(x.loc[a, ('infer_ms_per_window', 'mean')])) for a in x.index}
dd = W + 'ext/deep_bin.jsonl'
for t in ('bin', 'cat'):
    f = W + f'ext/deep_{t}.jsonl'
    if os.path.exists(f):
        d = pd.DataFrame([json.loads(l) for l in open(f)]).drop_duplicates(['level', 'arm', 'rep'], keep='last')
        x = d.groupby(['level', 'arm'])[['mcc', 'f1m', 'auc', 'train_s', 'infer_ms_per_window']].agg(['mean', 'std', 'count'])
        print('\ndeep', t); print(x.round(3)); OUT[f'deep_{t}'] = d.to_dict('records')
        m = load(f'main_{t}')
        rows = []
        for lvl in ('window', 'capture'):
            cf = m[(m.level == lvl) & (m.arm == 'CF-MIL-Boost')].set_index('rep').mcc
            for arm in d.arm.unique():
                b = d[(d.level == lvl) & (d.arm == arm)].set_index('rep').mcc; r = cf.index.intersection(b.index)
                if len(r) < 2: continue
                df_ = (cf.loc[r] - b.loc[r]).values; mm, lo, hi, p = nb(df_, 0.25)
                rows.append(dict(task=t, level=lvl, arm=arm, n=len(r), d=mm, lo=lo, hi=hi, p=p, wins=int((df_ > 0).sum())))
        print(pd.DataFrame(rows).round(4)); OUT[f'deep_delta_{t}'] = rows
# TPR at fixed FPR and per-class metrics from saved predictions (main protocol, all repetitions)
def tpr_at(y, p, f):
    fpr, tpr, _ = roc_curve(y, p); return float(np.interp(f, fpr, tpr))
for t in ('bin', 'cat'):
    fs = sorted(glob.glob(E + f'pred/main_{t}_r*.npz'))
    if not fs: continue
    rows = []; pcs = []
    for f in fs:
        z = np.load(f); r = int(f.split('_r')[-1][:-4])
        for arm in ('CF-MIL-Boost', 'win_LightGBM', 'win_XGBoost'):
            if f'{arm}__pw' not in z: continue
            for lvl, pk, yk in (('window', 'pw', 'yw'), ('capture', 'pc', 'yc')):
                p, y = z[f'{arm}__{pk}'], z[f'{arm}__{yk}']
                if t == 'bin':
                    pred = (p >= .5).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
                    rows.append(dict(rep=r, arm=arm, level=lvl, tpr1=tpr_at(y, p, .01), tpr5=tpr_at(y, p, .05), prec=tp / max(tp + fp, 1), rec=tp / (tp + fn), fpr=fp / (fp + tn)))
                else:
                    pred = p.argmax(1); pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=range(5), zero_division=0)
                    for k in range(5): pcs.append(dict(rep=r, arm=arm, level=lvl, cls=k, prec=pr[k], rec=rc[k], f1=f1[k]))
    if rows:
        x = pd.DataFrame(rows).groupby(['level', 'arm'])[['tpr1', 'tpr5', 'prec', 'rec', 'fpr']].agg(['mean', 'std'])
        print('\nTPR@FPR', t, len(fs), 'reps'); print(x.round(3)); OUT[f'tprfpr_{t}'] = pd.DataFrame(rows).to_dict('records')
    if pcs:
        x = pd.DataFrame(pcs).groupby(['level', 'cls', 'arm'])[['prec', 'rec', 'f1']].mean()
        print('\nper-class', t, len(fs), 'reps'); print(x.round(3)); OUT[f'perclass_{t}'] = pd.DataFrame(pcs).to_dict('records')
# SHAP: share of |contribution| carried by score statistics vs raw statistics
for t in ('bin', 'cat'):
    fs = sorted(glob.glob(E + f'pred/shap_main_{t}_r*.npz'))
    if not fs: continue
    M = np.vstack([np.load(f)['mabs'] for f in fs]); names = np.load(fs[0])['names']; nS = int(np.load(fs[0])['n_score'])
    share = M[:, :nS].sum(1) / M.sum(1)
    mean = M.mean(0); top = np.argsort(-mean)[:15]
    print(f'\nSHAP {t}: score-statistics share {share.mean():.3f} ± {share.std(ddof=1):.3f} over {len(fs)} reps; score feats {nS} of {M.shape[1]}')
    for i in top: print(f'  {names[i]:45s} {mean[i]:.4f}')
    OUT[f'shap_{t}'] = dict(share=float(share.mean()), share_sd=float(share.std(ddof=1)), n=len(fs), n_score=nS, n_feat=int(M.shape[1]),
                            top=[(str(names[i]), float(mean[i])) for i in top])
json.dump(OUT, open(E + 'summary.json', 'w'), default=float, indent=1)
