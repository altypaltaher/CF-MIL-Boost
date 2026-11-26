"""Extended experiments for CF-MIL-Boost (revision).

usage: python3 ext_run.py --task bin|cat --mode main|lfo|loco --rep R [--H 300] [--W 60]
                          [--s1 fixed|nested] [--base lgb,xgb] [--tag NAME]

  main : stratified capture split, identical to mil_run.py (seed 1000+rep); reproduces stage D3/D1
  lfo  : leave-family-out, 5 folds; malware families of every category are assigned to folds, benign
         captures at random; validation captures come from families not used for fitting
  loco : leave-one-malware-category-out (binary only), 4 folds: test = one malware category + 25% of
         benign captures; training uses the other three categories
  --s1 nested : the flow-level (stage-1) configuration is selected inside the repetition, among the same
         4 sampled configurations given to the flow-level LightGBM baseline (validation MCC)

Writes one JSON line per (arm, level) to ext/<tag>.jsonl and test predictions to ext/pred/.
"""
import sys, os, json, time, argparse, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics import matthews_corrcoef as MCC, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import lightgbm as lgb, xgboost as xgb

ap = argparse.ArgumentParser()
ap.add_argument('--task', default='bin'); ap.add_argument('--mode', default='main'); ap.add_argument('--rep', type=int, default=0)
ap.add_argument('--H', type=int, default=300); ap.add_argument('--W', type=int, default=60)
ap.add_argument('--s1', default='fixed'); ap.add_argument('--base', default='lgb,xgb'); ap.add_argument('--tag', default=None)
ap.add_argument('--shap', type=int, default=0)
A = ap.parse_args()
W_ = os.environ.get('CFMIL_WORK', os.path.dirname(os.path.abspath(__file__))) + '/'
OUT = W_ + 'ext/'; os.makedirs(OUT + 'pred', exist_ok=True)
TASK, MODE, REP, HORIZON, WIN = A.task, A.mode, A.rep, A.H, A.W
TAG = A.tag or f'{MODE}_{TASK}'
CAP = 150
SEED = {'main': 1000, 'lfo': 2000, 'loco': 3000}[MODE] + REP

D = pd.read_parquet(W_ + 'ctx_flows.parquet', filters=[('t', '<', HORIZON)]).reset_index(drop=True)
FEAT = [c for c in D.columns if c not in ('t', 'cid', 'cat', 'fam') and not c.startswith('ctx_')]
CTX = [c for c in D.columns if c.startswith('ctx_')]
X = np.array(D[FEAT + CTX].to_numpy(np.float32), copy=True); X[~np.isfinite(X)] = np.nan
cid = np.array(D.cid); cat = np.array(D.cat); t = np.array(D.t); fam = np.array(D.fam).astype(str)
y = (cat != 1).astype(int) if TASK == 'bin' else cat
K = len(np.unique(y)); del D
iF = np.arange(len(FEAT))
win = cid.astype(np.int64) * 100000 + (t // WIN).astype(np.int64)

# ---------------- splits ----------------
caps = pd.DataFrame({'cid': cid, 'cat': cat, 'fam': fam}).drop_duplicates('cid').reset_index(drop=True)
def grouped_val(tr, seed, frac=0.15):
    """validation captures drawn from whole families (malware) / random captures (benign)"""
    r = np.random.RandomState(seed); vfam = []
    for c, g in tr[tr.cat != 1].groupby('cat'):
        f = np.array(sorted(g.fam.unique())); r.shuffle(f); vfam += list(f[:max(1, int(round(frac * len(f))))])
    ben = tr[tr.cat == 1].cid.values.copy(); r.shuffle(ben)
    vmask = tr.fam.isin(vfam) & (tr.cat != 1) | tr.cid.isin(ben[:int(round(frac * len(ben)))])
    val = tr[vmask]; rest = tr[~vmask]
    fit, agg = train_test_split(rest, test_size=0.15 / 0.85, stratify=rest.cat, random_state=seed)
    return fit, val, agg
if MODE == 'main':
    tr, te = train_test_split(caps, test_size=0.2, stratify=caps.cat, random_state=SEED)
    fit, rest = train_test_split(tr, test_size=0.3, stratify=tr.cat, random_state=SEED)
    val, agg = train_test_split(rest, test_size=0.5, stratify=rest.cat, random_state=SEED)
elif MODE == 'lfo':
    r = np.random.RandomState(777); fold = {}
    for c, g in caps[caps.cat != 1].groupby('cat'):
        f = np.array(sorted(g.fam.unique())); r.shuffle(f)
        for i, ff in enumerate(f): fold[ff] = i % 5
    ben = caps[caps.cat == 1].cid.values.copy(); r.shuffle(ben); bfold = {b: i % 5 for i, b in enumerate(ben)}
    fk = np.array([bfold[c] if k == 1 else fold[f] for c, k, f in zip(caps.cid, caps.cat, caps.fam)])
    te, tr = caps[fk == REP], caps[fk != REP]
    fit, val, agg = grouped_val(tr, SEED)
elif MODE == 'loco':
    assert TASK == 'bin'
    mcat = [0, 2, 4, 3][REP]  # Adware, Ransomware, Scareware, SMSmalware
    r = np.random.RandomState(888); ben = caps[caps.cat == 1].cid.values.copy(); r.shuffle(ben)
    bte = ben[np.arange(len(ben)) % 4 == REP]
    temask = (caps.cat == mcat) | caps.cid.isin(bte)
    te = caps[temask]; tr = caps[~temask & ((caps.cat == 1) | (caps.cat != mcat))]
    fit, val, agg = grouped_val(tr, SEED)
S = {k: np.isin(cid, v.cid.values) for k, v in dict(fit=fit, val=val, agg=agg, te=te).items()}
rng = np.random.RandomState(SEED)
def capped(mask, cap):
    idx = np.flatnonzero(mask); g = pd.Series(idx).groupby(cid[idx])
    return np.sort(np.concatenate([v.values if len(v) <= cap else rng.choice(v.values, cap, replace=False) for _, v in g]))
I_fit = capped(S['fit'], CAP); I_val = capped(S['val'], CAP)
I_te = np.flatnonzero(S['te']); I_val_all = np.flatnonzero(S['val']); I_agg = np.flatnonzero(S['agg'])

def log(rec):
    rec.update(task=TASK, mode=MODE, rep=REP, H=HORIZON, W=WIN, s1=A.s1)
    with open(OUT + f'{TAG}.jsonl', 'a') as fh: fh.write(json.dumps(rec, default=float) + '\n')
    print(rec, flush=True)
def metr(yt, p, th=0.5):
    pred = (p >= th).astype(int) if p.ndim == 1 else p.argmax(1)
    auc = roc_auc_score(yt, p) if p.ndim == 1 else roc_auc_score(yt, p, multi_class='ovr', average='macro')
    return dict(mcc=MCC(yt, pred), f1m=f1_score(yt, pred, average='macro'), auc=auc)
def proba(m, Xs):
    p = m.predict_proba(Xs); return p[:, 1] if K == 2 else p

# ---------------- stage 1: cross-fitted flow scores ----------------
FCFG = dict(num_leaves=63, learning_rate=0.05, min_child_samples=100, colsample_bytree=0.6, reg_lambda=1)
t_s1 = time.time()
if A.s1 == 'nested':
    grid = dict(num_leaves=[31, 63, 127], learning_rate=[0.05, 0.1], min_child_samples=[20, 100],
                colsample_bytree=[0.6, 0.8], reg_lambda=[0, 1])
    best = None
    for c in ParameterSampler(grid, 4, random_state=SEED):
        m = lgb.LGBMClassifier(n_estimators=2000, subsample=0.8, subsample_freq=1, n_jobs=2, verbose=-1, random_state=SEED, **c)
        m.fit(X[I_fit][:, iF], y[I_fit], eval_set=[(X[I_val][:, iF], y[I_val])], callbacks=[lgb.early_stopping(50, verbose=False)])
        s = metr(y[I_val], proba(m, X[I_val][:, iF]))['mcc']
        if best is None or s > best[0]: best = (s, c)
    FCFG = best[1]
T = pd.concat([fit, agg]); folds = np.arange(len(T)) % 3; rng.shuffle(folds)
def flow_model(Itr):
    m = lgb.LGBMClassifier(n_estimators=2000, subsample=0.8, subsample_freq=1, n_jobs=2, verbose=-1, random_state=SEED, **FCFG)
    m.fit(X[Itr][:, iF], y[Itr], eval_set=[(X[I_val][:, iF], y[I_val])], callbacks=[lgb.early_stopping(50, verbose=False)]); return m
I_T = np.flatnonzero(np.isin(cid, T.cid.values)); oof = np.zeros(len(I_T), np.float32) if K == 2 else np.zeros((len(I_T), K), np.float32)
for k in range(3):
    trc, hoc = T.cid.values[folds != k], T.cid.values[folds == k]
    m = flow_model(capped(np.isin(cid, trc), CAP)); h = np.isin(cid[I_T], hoc)
    oof[h] = proba(m, X[I_T[h]][:, iF])
FM = flow_model(capped(np.isin(cid, T.cid.values), CAP))
t_s1 = time.time() - t_s1
p_val = proba(FM, X[I_val_all][:, iF])

def raw_part(I):
    u, inv, n = np.unique(win[I], return_inverse=True, return_counts=True)
    A_ = np.sign(X[I][:, iF].astype(np.float64)); A_ = np.nan_to_num(A_ * np.log1p(np.abs(np.nan_to_num(X[I][:, iF].astype(np.float64)))))
    s1 = np.zeros((len(u), A_.shape[1])); s2 = np.zeros_like(s1); mx = np.full_like(s1, -np.inf)
    np.add.at(s1, inv, A_); np.add.at(s2, inv, A_ * A_); np.maximum.at(mx, inv, A_)
    mu = s1 / n[:, None]; sd = np.sqrt(np.maximum(s2 / n[:, None] - mu * mu, 0))
    yl = pd.Series(y[I]).groupby(inv).first().values
    return np.hstack([mu, sd, mx, np.log1p(n)[:, None]]).astype(np.float32), u, inv, yl
def score_part(p, inv):
    P = pd.DataFrame(p if p.ndim > 1 else p[:, None]); g = P.groupby(inv)
    return np.hstack([g.mean().values, g.max().values, g.quantile(.5).values, g.quantile(.9).values, g.std().fillna(0).values]).astype(np.float32)
def wfeat(I, p):
    R, u, inv, yl = raw_part(I); return R, score_part(p, inv), yl, u // 100000, u
Rtr, Str, ytr, _, _ = wfeat(I_T, oof); Rva, Sva, yva, _, _ = wfeat(I_val_all, p_val)
sw = compute_sample_weight('balanced', ytr)

def capture(pt, yt, cte):
    cp = pd.DataFrame(pt if pt.ndim > 1 else pt[:, None]).groupby(cte).mean().values
    cy = pd.Series(yt).groupby(cte).first().values
    return (cp[:, 0] if K == 2 else cp), cy, np.unique(cte)
PRED = {}
def report(arm, pt, yt, cte, u, extra):
    log(dict(level='window', arm=arm, n_units=len(yt), **metr(yt, pt), **extra))
    cp, cy, cu = capture(pt, yt, cte)
    log(dict(level='capture', arm=arm, n_units=len(cy), **metr(cy, cp), **extra))
    PRED[arm] = dict(pw=pt, yw=yt, capw=cte, widx=(u % 100000), pc=cp, yc=cy, cid=cu)

# ---------------- stage 2: CF-MIL-Boost ----------------
gl = [dict(num_leaves=nl, learning_rate=0.05, min_child_samples=mc) for nl in (15, 31, 63) for mc in (10, 30)]
t2 = time.time(); best = None
A_tr, A_va = np.hstack([Str, Rtr]), np.hstack([Sva, Rva])
for c in gl:
    mm = lgb.LGBMClassifier(n_estimators=2000, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=2, verbose=-1, random_state=SEED, **c)
    mm.fit(A_tr, ytr, sample_weight=sw, eval_set=[(A_va, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
    s = metr(yva, proba(mm, A_va))['mcc']
    if best is None or s > best[0]: best = (s, mm, c)
t2 = time.time() - t2; M2 = best[1]
# inference timing on the test captures: flow scoring + window features + window model
ti = time.time()
p_te = proba(FM, X[I_te][:, iF]); Rte, Ste, yte, cte, ute = wfeat(I_te, p_te); pt = proba(M2, np.hstack([Ste, Rte]))
ti = time.time() - ti
report('CF-MIL-Boost', pt, yte, cte, ute, dict(cfg=json.dumps(best[2]), s1cfg=json.dumps(FCFG), train_s=round(t_s1 + t2, 1),
       infer_ms_per_window=1000 * ti / len(yte), n_test_flows=len(I_te)))
if A.shap:
    nS = Ste.shape[1]; Xs = np.hstack([Ste, Rte])
    C = M2.predict(Xs, pred_contrib=True)
    C = np.abs(C).reshape(len(Xs), -1, Xs.shape[1] + 1).sum(1) if K > 2 else np.abs(C)
    names = [f'{s}_{k}' for s in ('mean', 'max', 'median', 'q90', 'std') for k in range(nS // 5)] + \
            [f'{s}:{f}' for s in ('mean', 'std', 'max') for f in FEAT] + ['log_count']
    np.savez_compressed(OUT + f'pred/shap_{TAG}_r{REP}.npz', mabs=C[:, :-1].mean(0), names=np.array(names), n_score=nS)

# ---------------- time-only control ----------------
if K == 2: log(dict(level='window', arm='CONTROL_time_index_only', auc=roc_auc_score(yte, (ute % 100000).astype(float))))

# ---------------- window-level baselines on raw statistics ----------------
# same training captures (fit+agg) and validation windows; row order as in mil_run.py stage D1
_f = raw_part(np.flatnonzero(S['fit'])); _a = raw_part(I_agg)
Rtrb = np.vstack([_f[0], _a[0]]); ytrb = np.concatenate([_f[3], _a[3]]); swb = compute_sample_weight('balanced', ytrb); Rvab = Rva
for b in A.base.split(','):
    t0 = time.time(); best = None
    if b == 'lgb':
        models = [(c, lgb.LGBMClassifier(n_estimators=2000, n_jobs=2, verbose=-1, random_state=SEED, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, **c)) for c in gl]
    else:
        models = [(dict(d=d, lr=0.05), xgb.XGBClassifier(n_estimators=2000, max_depth=d, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=2, random_state=SEED, early_stopping_rounds=50)) for d in (3, 4, 6, 8)]
    for c, m in models:
        m.fit(Rtrb, ytrb, sample_weight=swb, eval_set=[(Rvab, yva)], **({'callbacks': [lgb.early_stopping(50, verbose=False)]} if b == 'lgb' else {'verbose': False}))
        s = metr(yva, proba(m, Rvab))['mcc']
        if best is None or s > best[0]: best = (s, m, c)
    tb = time.time() - t0
    ti = time.time(); Rte2, _, _, _ = raw_part(I_te); pb = proba(best[1], Rte2); ti = time.time() - ti
    report({'lgb': 'win_LightGBM', 'xgb': 'win_XGBoost'}[b], pb, yte, cte, ute,
           dict(cfg=json.dumps(best[2]), train_s=round(tb, 1), infer_ms_per_window=1000 * ti / len(yte)))
np.savez_compressed(OUT + f'pred/{TAG}_r{REP}.npz', **{f'{a}__{k}': v for a, d in PRED.items() for k, v in d.items()})
