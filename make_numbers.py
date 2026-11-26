"""Collect every number quoted in the revised paper from the logged results -> numbers.json (and tables.json)."""
import json, numpy as np, pandas as pd, sys
from scipy import stats
import os; W = os.environ.get('CFMIL_WORK', os.path.dirname(os.path.abspath(__file__))) + '/'
def jl(f, keys=('level', 'arm', 'rep')):
    d = pd.DataFrame([json.loads(l) for l in open(f)])
    return d.drop_duplicates([k for k in keys if k in d.columns], keep='last')
def nb(d, ratio):
    d = np.asarray(d, float); n = len(d); m = d.mean(); v = d.var(ddof=1)
    se = np.sqrt((1 / n + ratio) * v); t = m / se; p = 2 * stats.t.sf(abs(t), n - 1); h = stats.t.ppf(.975, n - 1) * se
    return m, m - h, m + h, p
f3 = lambda x: f'{x:.3f}'
fp = lambda p: '<0.001' if p < 0.001 else f'{p:.3f}'
N = {}; T = {}
# ---------------- main protocol (ext runs reproduce mil_run exactly; deep from deep_*.jsonl) ----------------
mb = jl(W + 'ext/main_bin.jsonl')
# five-class, 10 repetitions: reps 0-4 from mil_run.py (identical code path), reps 5-9 from ext_run.py
_m = jl(W + 'mil/cat_results.jsonl'); _m['arm'] = _m.arm.replace({'HYBRID_scores+raw': 'CF-MIL-Boost'})
_m = _m[_m.arm.isin(['CF-MIL-Boost', 'win_LightGBM', 'win_XGBoost']) & (_m.rep < 5)]
mc = pd.concat([_m, jl(W + 'ext/main_cat.jsonl')]).drop_duplicates(['level', 'arm', 'rep'], keep='last')
mc[['level', 'arm', 'rep', 'mcc', 'f1m', 'auc']].to_json(W + 'ext/main_cat10.jsonl', orient='records', lines=True)
db = jl(W + 'ext/deep_bin.jsonl'); dc = jl(W + 'ext/deep_cat.jsonl')
mil = {t: jl(W + f'mil/{t}_results.jsonl') for t in ('bin', 'cat')}
for t in ('bin', 'cat'): mil[t]['arm'] = mil[t].arm.replace({'HYBRID_scores+raw': 'CF-MIL-Boost'})
def arm(d, lvl, a, reps=None, met='mcc'):
    x = d[(d.level == lvl) & (d.arm == a)].set_index('rep')[met].sort_index()
    return x if reps is None else x.loc[[r for r in reps if r in x.index]]
def msd(x): return f'{x.mean():.3f} ± {x.std(ddof=1):.3f}'
# Table rows for deep baselines (mean ± sd) -- binary 10 reps, five-class reps available
for t, d, reps in (('bin', db, range(10)), ('cat', dc, range(5))):
    for a in ('AttnMIL', 'BiGRU'):
        for lvl in ('window', 'capture'):
            r = [msd(arm(d, lvl, a, reps, m)) for m in ('mcc', 'f1m', 'auc')]
            T[f'{t}_{a}_{lvl}'] = r
        N[f'{t}_{a}_epochs'] = f"{d[d.arm == a].epochs.mean():.0f}"
        N[f'{t}_{a}_train'] = f"{d[(d.arm == a) & (d.level == 'window')].train_s.median():.0f}"  # median: one run overlapped with another job
# five-class 10 repetitions (CF, LGB, XGB)
for lvl, k in (('window', 'w'), ('capture', 'c')):
    cf = arm(mc, lvl, 'CF-MIL-Boost'); lg = arm(mc, lvl, 'win_LightGBM'); xg = arm(mc, lvl, 'win_XGBoost')
    best = max(lg.mean(), xg.mean())
    N[f'c10_{k}_cf'] = f3(cf.mean()); N[f'c10_{k}_best'] = f3(best); N[f'c10_{k}_n'] = len(cf)
    for name, b in (('LightGBM', lg), ('XGBoost', xg)):
        T[f'cat10_{lvl}_{name}'] = [msd(arm(mc, lvl, f'win_{name}', None, m)) for m in ('mcc', 'f1m', 'auc')]
    T[f'cat10_{lvl}_CF'] = [msd(arm(mc, lvl, 'CF-MIL-Boost', None, m)) for m in ('mcc', 'f1m', 'auc')]
# significance table rows: (task, level, baseline, dMCC, p, dAUC, p, W/L, n)
sig = []
def addsig(task, lvl, name, cfd, bd, ratio=0.25):
    r = cfd['mcc'].index.intersection(bd['mcc'].index)
    dm = (cfd['mcc'].loc[r] - bd['mcc'].loc[r]).values * 100; da = (cfd['auc'].loc[r] - bd['auc'].loc[r]).values * 100
    m1, lo1, hi1, p1 = nb(dm, ratio); m2, _, _, p2 = nb(da, ratio)
    wil = stats.wilcoxon(dm).pvalue if len(r) >= 10 else np.nan
    sig.append(dict(task=task, level=lvl, base=name, n=len(r), dmcc=m1, lo=lo1, hi=hi1, pm=p1, dauc=m2, pa=p2, wins=int((dm > 0).sum()), wil=wil))
for lvl in ('window', 'capture'):
    cfb = {m: arm(mil['bin'], lvl, 'CF-MIL-Boost', None, m) for m in ('mcc', 'auc')}
    for b in ('LightGBM', 'XGBoost', 'AdaBoost', 'GBM'):
        addsig('Binary', lvl, b, cfb, {m: arm(mil['bin'], lvl, f'win_{b}', None, m) for m in ('mcc', 'auc')})
    for b in ('AttnMIL', 'BiGRU'):
        addsig('Binary', lvl, b, cfb, {m: arm(db, lvl, b, None, m) for m in ('mcc', 'auc')})
    cfc10 = {m: arm(mc, lvl, 'CF-MIL-Boost', None, m) for m in ('mcc', 'auc')}
    cfc5 = {m: arm(mil['cat'], lvl, 'CF-MIL-Boost', None, m) for m in ('mcc', 'auc')}
    for b in ('LightGBM', 'XGBoost'):
        addsig('Five-class', lvl, b, cfc10, {m: arm(mc, lvl, f'win_{b}', None, m) for m in ('mcc', 'auc')})
    for b in ('AdaBoost', 'GBM'):
        addsig('Five-class', lvl, b, cfc5, {m: arm(mil['cat'], lvl, f'win_{b}', None, m) for m in ('mcc', 'auc')})
    for b in ('AttnMIL', 'BiGRU'):
        addsig('Five-class', lvl, b, cfc10, {m: arm(dc, lvl, b, None, m) for m in ('mcc', 'auc')})
S = pd.DataFrame(sig); T['sig'] = S.to_dict('records'); print(S.round(4).to_string())
N['n_sig_tests'] = int(2 * len(S)); N['n_sig_sig'] = int((S.pm < 0.05).sum() + (S.pa < 0.05).sum())
N['n_pairs'] = int(S.n.sum()); N['n_wins'] = int(S.wins.sum())
N['sig_fail'] = [f"{r.task} {r.level} {'MCC' if r.pm >= 0.05 else 'AUC'} vs {r.base} (p = {fp(r.pm if r.pm >= 0.05 else r.pa)})"
                 for r in S.itertuples() for _ in [0] if (r.pm >= 0.05 or r.pa >= 0.05)]
c5 = S[(S.task == 'Five-class') & S.base.isin(['LightGBM', 'XGBoost', 'AdaBoost', 'GBM']) & (S.level == 'window')]
N['c10_dmin'] = f"{c5.dmcc.min():.1f}"; N['c5_dmax'] = f"{c5.dmcc.max():.1f}"
cc = S[(S.task == 'Five-class') & S.base.isin(['LightGBM', 'XGBoost', 'AdaBoost', 'GBM']) & (S.level == 'capture')]
N['c_cap_dmin'] = f"{cc.dmcc.min():.1f}"; N['c_cap_dmax'] = f"{cc.dmcc.max():.1f}"
# ---------------- generalisation ----------------
gen = []
for tag, ratio, label in (('lfo_bin', 0.25, 'Binary, unseen families'), ('loco_bin', 1 / 3, 'Binary, unseen category'), ('lfo_cat', 0.25, 'Five-class, unseen families')):
    d = jl(W + f'ext/{tag}.jsonl')
    for lvl in ('window', 'capture'):
        cf = arm(d, lvl, 'CF-MIL-Boost'); row = dict(set=label, level=lvl, n=len(cf), cf=msd(cf))
        for b in ('LightGBM', 'XGBoost'):
            x = arm(d, lvl, f'win_{b}'); r = cf.index.intersection(x.index); dd = (cf.loc[r] - x.loc[r]).values * 100
            m, lo, hi, p = nb(dd, ratio)
            row[b] = msd(x); row[f'd_{b}'] = f'{m:+.1f}'; row[f'p_{b}'] = fp(p); row[f'w_{b}'] = f'{int((dd > 0).sum())}/{len(r)}'
        gen.append(row)
        best = max(arm(d, lvl, 'win_LightGBM').mean(), arm(d, lvl, 'win_XGBoost').mean())
        k = {'lfo_bin': 'lfo_b', 'loco_bin': 'loco_b', 'lfo_cat': 'lfo_c'}[tag] + ('_w' if lvl == 'window' else '_c')
        N[f'{k}_cf'] = f3(cf.mean()); N[f'{k}_best'] = f3(best)
    if tag == 'loco_bin':
        cats = {0: 'adware', 1: 'ransomware', 2: 'scareware', 3: 'SMS malware'}
        per = d[(d.level == 'window') & d.arm.isin(['CF-MIL-Boost', 'win_LightGBM', 'win_XGBoost'])].pivot_table(index='rep', columns='arm', values='mcc')
        T['loco_per'] = {cats[r]: {a: float(per.loc[r, a]) for a in per.columns} for r in per.index}
    ctl = d[d.arm == 'CONTROL_time_index_only'].auc if (d.arm == 'CONTROL_time_index_only').any() else None
    if ctl is not None: N[f'{tag}_ctl'] = f'{ctl.mean():.3f}'
T['gen'] = gen; print(pd.DataFrame(gen).to_string())
# ---------------- operating points (binary) and per-class (five-class) ----------------
sm = json.load(open(W + 'ext/summary.json'))
op = pd.DataFrame(sm['tprfpr_bin']).groupby(['level', 'arm'])[['tpr1', 'tpr5', 'prec', 'rec', 'fpr']].agg(['mean', 'std'])
T['op'] = {f'{l}|{a}': {m: f"{op.loc[(l, a), (m, 'mean')]:.3f} ± {op.loc[(l, a), (m, 'std')]:.3f}" for m in ('tpr1', 'tpr5', 'prec', 'rec', 'fpr')} for l, a in op.index}
pc = pd.DataFrame(sm['perclass_cat']).groupby(['level', 'cls', 'arm'])[['prec', 'rec', 'f1']].mean()
T['pc'] = {f'{l}|{c}|{a}': {m: f'{pc.loc[(l, c, a), m]:.3f}' for m in ('prec', 'rec', 'f1')} for l, c, a in pc.index}
N['n_pc_reps'] = len(pd.DataFrame(sm['perclass_cat']).rep.unique())
# ---------------- SHAP ----------------
for t in ('bin', 'cat'):
    N[f'shap_{t}_share'] = f"{100 * sm[f'shap_{t}']['share']:.0f}"; N[f'shap_{t}_sd'] = f"{100 * sm[f'shap_{t}']['share_sd']:.0f}"
    N[f'shap_{t}_nS'] = sm[f'shap_{t}']['n_score']; N[f'shap_{t}_nF'] = sm[f'shap_{t}']['n_feat']
# ---------------- timing ----------------
for t in ('bin', 'cat'):
    tm = sm[f'timing_{t}']
    for a, k in (('CF-MIL-Boost', 'cf'), ('win_LightGBM', 'lgb'), ('win_XGBoost', 'xgb')):
        N[f'time_{t}_{k}_train'] = f"{tm[a]['train_s']:.0f}"; N[f'time_{t}_{k}_inf'] = f"{tm[a]['infer']:.2f}"
# ---------------- nested and sensitivity ----------------
nest = sm.get('nested_bin')
if nest:
    for r in nest:
        k = 'w' if r['level'] == 'window' else 'c'
        N[f'nest_{k}'] = f"{r['nested']:.3f}"; N[f'nest_{k}_fixed'] = f"{r['fixed']:.3f}"; N[f'nest_{k}_lgb'] = f"{r['lgb']:.3f}"
        N[f'nest_{k}_d'] = f"{100 * r['d_vs_lgb']:.1f}"; N[f'nest_{k}_p'] = fp(r['p']); N[f'nest_{k}_wins'] = f"{r['wins']}/{r['n']}"
    nb_ = jl(W + 'ext/nested_bin.jsonl'); N['nest_n'] = int(nb_.rep.nunique())
    N['nest_cfgs'] = nb_[(nb_.arm == 'CF-MIL-Boost') & (nb_.level == 'window')].s1cfg.tolist()
if 'sens' in sm:
    T['sens'] = sm['sens']; T['sens_ctl'] = sm['sens_ctl']
json.dump(N, open(W + 'numbers.json', 'w'), indent=1, default=str)
json.dump(T, open(W + 'tables.json', 'w'), indent=1, default=str)
print(json.dumps(N, indent=1, default=str))
