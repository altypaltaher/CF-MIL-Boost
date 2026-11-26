"""Session-level (multi-instance) detection with host-context features on CICAndMal2017.
Leakage-free protocol: captures (one app run each) are split, never flows.
usage: python3 mil_run.py <task: bin|cat> <rep> <stage: A|B|C|D|E>
  A  flow models: LightGBM tuned on flow-only and flow+context features
  B  flow models: XGBoost tuned (flow-only, flow+context)
  C  flow models: AdaBoost, GBM tuned (flow-only; smaller per-capture cap, see CAP_SMALL)
  D  window/capture level: base classifiers on window aggregates + MIL aggregators; metrics
  E  leakage check: LightGBM flow-only with a random FLOW split (old protocol)
"""
import sys, os, json, time, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics import matthews_corrcoef as MCC, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb, xgboost as xgb

W = os.environ.get('CFMIL_WORK', os.path.dirname(os.path.abspath(__file__))) + '/'; OUT = W + 'mil/'; os.makedirs(OUT, exist_ok=True)
TASK, REP, STAGE = sys.argv[1], int(sys.argv[2]), sys.argv[3]
WIN = 60           # decision window, seconds
HORIZON = 300      # only the first 5 min of every capture: malware captures run ~45 min vs ~12 min benign,
                   # so later flows let any time-correlated feature predict the label (collection artifact)
CAP = 150          # max training flows sampled per capture (all models, flow level)
CAP_SMALL = 50     # AdaBoost / GBM flow-level budget
SEED = 1000 + REP
D = pd.read_parquet(W + 'ctx_flows.parquet', filters=[('t', '<', HORIZON)]).reset_index(drop=True)
FEAT = [c for c in D.columns if c not in ('t', 'cid', 'cat', 'fam') and not c.startswith('ctx_')]
CTX = [c for c in D.columns if c.startswith('ctx_')]
X = np.array(D[FEAT + CTX].to_numpy(np.float32), copy=True); X[~np.isfinite(X)] = np.nan
cid = np.array(D.cid); cat = np.array(D.cat); t = np.array(D.t)
y = (cat != 1).astype(int) if TASK == 'bin' else cat
K = len(np.unique(y)); del D
iF = np.arange(len(FEAT)); iFC = np.arange(len(FEAT) + len(CTX))
win = cid.astype(np.int64) * 100000 + (t // WIN).astype(np.int64)   # device-window id

# ---------------- capture split (stratified by category) ----------------
caps = pd.DataFrame({'cid': cid, 'cat': cat}).drop_duplicates('cid')
tr, te = train_test_split(caps, test_size=0.2, stratify=caps.cat, random_state=SEED)
fit, rest = train_test_split(tr, test_size=0.3, stratify=tr.cat, random_state=SEED)
val, agg = train_test_split(rest, test_size=0.5, stratify=rest.cat, random_state=SEED)
S = {k: np.isin(cid, v.cid.values) for k, v in dict(fit=fit, val=val, agg=agg, te=te).items()}
rng = np.random.RandomState(SEED)
def capped(mask, cap):
    idx = np.flatnonzero(mask); g = pd.Series(idx).groupby(cid[idx])
    return np.sort(np.concatenate([v.values if len(v) <= cap else rng.choice(v.values, cap, replace=False) for _, v in g]))
I_fit = capped(S['fit'], CAP); I_val = capped(S['val'], CAP)
I_te = np.flatnonzero(S['te']); I_val_all = np.flatnonzero(S['val']); I_agg = np.flatnonzero(S['agg'])

def save(name, **arrs): np.savez_compressed(OUT + f'{TASK}_r{REP}_{name}.npz', **arrs)
def log(rec):
    rec.update(task=TASK, rep=REP)
    with open(OUT + f'{TASK}_results.jsonl', 'a') as fh: fh.write(json.dumps(rec, default=float) + '\n')
    print(rec, flush=True)
def metr(yt, p, th=0.5):
    pred = (p >= th).astype(int) if p.ndim == 1 else p.argmax(1)
    auc = roc_auc_score(yt, p) if p.ndim == 1 else roc_auc_score(yt, p, multi_class='ovr', average='macro')
    return dict(mcc=MCC(yt, pred), f1m=f1_score(yt, pred, average='macro'), auc=auc)
def proba(m, Xs):
    p = m.predict_proba(Xs); return p[:, 1] if K == 2 else p

# ---------------- tuned flow models ----------------
def tune_lgb(cols, Itr, Iva, n=4):
    best = None
    grid = dict(num_leaves=[31, 63, 127], learning_rate=[0.05, 0.1], min_child_samples=[20, 100],
                colsample_bytree=[0.6, 0.8], reg_lambda=[0, 1])
    for c in ParameterSampler(grid, n, random_state=SEED):
        m = lgb.LGBMClassifier(n_estimators=2000, subsample=0.8, subsample_freq=1, n_jobs=2, verbose=-1, random_state=SEED, **c)
        m.fit(X[Itr][:, cols], y[Itr], eval_set=[(X[Iva][:, cols], y[Iva])], callbacks=[lgb.early_stopping(50, verbose=False)])
        s = metr(y[Iva], proba(m, X[Iva][:, cols]))['mcc']
        if best is None or s > best[0]: best = (s, m, c)
    return best
def tune_xgb(cols, Itr, Iva, n=4):
    best = None
    grid = dict(max_depth=[6, 8, 10], learning_rate=[0.05, 0.1], min_child_weight=[1, 5], colsample_bytree=[0.6, 0.8])
    for c in ParameterSampler(grid, n, random_state=SEED):
        m = xgb.XGBClassifier(n_estimators=2000, subsample=0.8, tree_method='hist', n_jobs=2, random_state=SEED,
                              early_stopping_rounds=50, eval_metric='logloss' if K == 2 else 'mlogloss', **c)
        m.fit(X[Itr][:, cols], y[Itr], eval_set=[(X[Iva][:, cols], y[Iva])], verbose=False)
        s = metr(y[Iva], proba(m, X[Iva][:, cols]))['mcc']
        if best is None or s > best[0]: best = (s, m, c)
    return best
def impute(A, med): A = A.copy(); r, c = np.where(np.isnan(A)); A[r, c] = med[c]; return A
def tune_sk(kind, cols, Itr, Iva):
    med = np.nanmedian(X[Itr][:, cols], 0); med[np.isnan(med)] = 0
    Xa, Xv = impute(X[Itr][:, cols], med), impute(X[Iva][:, cols], med)
    best = None
    cfgs = [dict(d=1, lr=0.5), dict(d=2, lr=1.0)] if kind == 'ada' else [dict(d=3, lr=0.1, ss=0.8), dict(d=5, lr=0.1, ss=0.8)]
    for c in cfgs:
        if kind == 'ada':
            m = AdaBoostClassifier(DecisionTreeClassifier(max_depth=c['d']), n_estimators=200, learning_rate=c['lr'], random_state=SEED)
        else:
            m = GradientBoostingClassifier(n_estimators=200, max_depth=c['d'], learning_rate=c['lr'], subsample=c['ss'], max_features=0.5, random_state=SEED)
        m.fit(Xa, y[Itr])
        sc = [MCC(y[Iva], pv) for pv in m.staged_predict(Xv)]; k = int(np.argmax(sc))
        if best is None or sc[k] > best[0]: best = (sc[k], m, dict(c, n=k + 1), med)
    return best
def staged_proba(m, A, n):
    for i, p in enumerate(m.staged_predict_proba(A)):
        if i + 1 == n: return p[:, 1] if K == 2 else p

def run_flow(name, kind, cols):
    t0 = time.time()
    if kind == 'lgb': s, m, c = tune_lgb(cols, I_fit, I_val); pr = lambda I: proba(m, X[I][:, cols])
    elif kind == 'xgb': s, m, c = tune_xgb(cols, I_fit, I_val); pr = lambda I: proba(m, X[I][:, cols])
    else:
        Is = capped(S['fit'], CAP_SMALL); s, m, c, med = tune_sk(kind, cols, Is, I_val)
        pr = lambda I: staged_proba(m, impute(X[I][:, cols], med), c['n'])
    pte, pval, pagg = pr(I_te), pr(I_val_all), pr(I_agg)
    save(name, te=pte.astype(np.float32), val=pval.astype(np.float32), agg=pagg.astype(np.float32))
    log(dict(level='flow', arm=name, cfg=json.dumps(c, default=str), secs=round(time.time() - t0), **metr(y[I_te], pte)))

if STAGE == 'A':
    run_flow('lgb_flow', 'lgb', iF); run_flow('lgb_ctx', 'lgb', iFC)
elif STAGE == 'B':
    run_flow('xgb_flow', 'xgb', iF); run_flow('xgb_ctx', 'xgb', iFC)
elif STAGE == 'C':
    run_flow('ada_flow', 'ada', iF); run_flow('gbm_flow', 'gbm', iF)
elif STAGE == 'E':
    # old protocol: random flow split, same training size, same tuning
    n_tr = len(I_fit); allidx = rng.permutation(len(y))
    Ite = allidx[:len(y) // 5]; Ifit = allidx[len(y) // 5:len(y) // 5 + n_tr]; Iva = allidx[len(y) // 5 + n_tr:len(y) // 5 + n_tr + len(I_val)]
    s, m, c = tune_lgb(iF, Ifit, Iva)
    log(dict(level='flow', arm='lgb_flow_RANDOMsplit', cfg=json.dumps(c), **metr(y[Ite], proba(m, X[Ite][:, iF]))))
elif STAGE in ('D1', 'D2'):
    # ---------- window units ----------
    def wtab(I):
        w = win[I]; u, inv = np.unique(w, return_inverse=True)
        return u, inv, np.bincount(inv, minlength=len(u))
    def wlabel(I, inv, n): return np.bincount(inv, weights=y[I], minlength=len(n)) / n if K == 2 else \
        pd.Series(y[I]).groupby(inv).first().values
    # raw window aggregates (for window-level base classifiers): mean & std of 80 flow features + log count
    def raw_agg(I):
        u, inv, n = wtab(I); A = np.nan_to_num(X[I][:, iF].astype(np.float64), nan=0.0)
        A = np.sign(A) * np.log1p(np.abs(A))  # robust scale before averaging
        s1 = np.zeros((len(u), A.shape[1])); s2 = np.zeros_like(s1)
        mx = np.full_like(s1, -np.inf)
        np.add.at(s1, inv, A); np.add.at(s2, inv, A * A); np.maximum.at(mx, inv, A)
        mu = s1 / n[:, None]; sd = np.sqrt(np.maximum(s2 / n[:, None] - mu * mu, 0))
        return np.hstack([mu, sd, mx, np.log1p(n)[:, None]]).astype(np.float32), u, inv, n
    def score_agg(p, inv, n):
        # MIL features from flow scores per window (binary: P(malicious); multiclass: per-class probs)
        P = p[:, None] if p.ndim == 1 else p
        df = pd.DataFrame(P).groupby(inv)
        feats = [df.mean().values, df.max().values, df.quantile(0.5).values, df.quantile(0.9).values,
                 df.apply(lambda g: np.sort(g.values, 0)[-5:].mean(0)).apply(pd.Series).values if P.shape[1] > 1 else
                 df[0].apply(lambda g: np.sort(g.values)[-5:].mean()).values[:, None],
                 (df.apply(lambda g: (g.values.argmax(1) if P.shape[1] > 1 else (g.values[:, 0] >= .5)).mean())).values[:, None],
                 np.log1p(n)[:, None]]
        return np.hstack(feats).astype(np.float32)
    Wd = {}
    for k, I in dict(fit=np.flatnonzero(S['fit']), val=I_val_all, agg=I_agg, te=I_te).items():
        R, u, inv, n = raw_agg(I); yl = wlabel(I, inv, n); yl = (yl >= 0.5).astype(int) if K == 2 else yl.astype(int)
        Wd[k] = dict(R=R, u=u, inv=inv, n=n, y=yl, cap=(u // 100000))
    # shortcut control: does the window's position in time alone predict the label?
    wi = (Wd['te']['u'] % 100000).astype(float)
    if STAGE == 'D1': log(dict(level='window', arm='CONTROL_time_index_only', auc=roc_auc_score(Wd['te']['y'], wi) if K == 2 else np.nan))
    def wlog(arm, p, extra={}, th=0.5):
        yt = Wd['te']['y']; m = metr(yt, p, th); log(dict(level='window', arm=arm, n_units=len(yt), **m, **extra))
        # capture level: mean of window probabilities within capture
        cp = pd.DataFrame(p if p.ndim > 1 else p[:, None]).groupby(Wd['te']['cap']).mean().values
        cy = pd.Series(yt).groupby(Wd['te']['cap']).first().values
        log(dict(level='capture', arm=arm, n_units=len(cy), **metr(cy, cp[:, 0] if K == 2 else cp, th), **extra))
    # (1) window-level base classifiers on raw aggregates: train on fit+agg windows, select on val windows
    Rtr = np.vstack([Wd['fit']['R'], Wd['agg']['R']]); ytr = np.concatenate([Wd['fit']['y'], Wd['agg']['y']])
    Rva, yva, Rte = Wd['val']['R'], Wd['val']['y'], Wd['te']['R']
    from sklearn.utils.class_weight import compute_sample_weight
    sw = compute_sample_weight('balanced', ytr)
    def sel(models):
        best = None
        for c, m in models:
            if hasattr(m, 'staged_predict'):
                m.fit(Rtr, ytr, sample_weight=sw); sc = [MCC(yva, q) for q in m.staged_predict(Rva)]; k = int(np.argmax(sc))
                pt = staged_proba(m, Rte, k + 1); s = sc[k]; c = dict(c, n=k + 1)
            else:
                m.fit(Rtr, ytr, sample_weight=sw, eval_set=[(Rva, yva)], **({'callbacks': [lgb.early_stopping(50, verbose=False)]} if isinstance(m, lgb.LGBMClassifier) else {'verbose': False}))
                pv = proba(m, Rva); s = metr(yva, pv)['mcc']; pt = proba(m, Rte)
            if best is None or s > best[0]: best = (s, pt, c)
        return best
    gl = [dict(num_leaves=nl, learning_rate=lr, min_child_samples=mc) for nl in (15, 31, 63) for lr in (0.05,) for mc in (10, 30)]
    base = {
        'LightGBM': [(c, lgb.LGBMClassifier(n_estimators=2000, n_jobs=2, verbose=-1, random_state=SEED, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, **c)) for c in gl],
        'XGBoost': [(dict(d=d, lr=0.05), xgb.XGBClassifier(n_estimators=2000, max_depth=d, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=2, random_state=SEED, early_stopping_rounds=50)) for d in (3, 4, 6, 8)],
        'AdaBoost': [(dict(d=d, lr=lr), AdaBoostClassifier(DecisionTreeClassifier(max_depth=d), n_estimators=300, learning_rate=lr, random_state=SEED)) for d, lr in ((1, 1.0), (2, 0.5), (3, 0.5))],
        'GBM': [(dict(d=d), GradientBoostingClassifier(n_estimators=300, max_depth=d, learning_rate=0.1, subsample=0.8, max_features=0.3, random_state=SEED)) for d in (3, 5)],
    }
    for name, models in (base.items() if STAGE == 'D1' else []):
        t0 = time.time(); s, pt, c = sel(models); wlog(f'win_{name}', pt, dict(cfg=json.dumps(c), secs=round(time.time() - t0)))
    # (2) flow-level base classifiers aggregated by mean probability (the 'naive' aggregation)
    # (3) MIL: second-stage logistic regression on score aggregates, fitted on the 'agg' captures only
    for arm in (['lgb_flow', 'lgb_ctx', 'xgb_flow', 'xgb_ctx', 'ada_flow', 'gbm_flow'] if STAGE == 'D2' else []):
        f = OUT + f'{TASK}_r{REP}_{arm}.npz'
        if not os.path.exists(f): continue
        Z = np.load(f); pte, pagg = Z['te'], Z['agg']
        Fte = score_agg(pte, Wd['te']['inv'], Wd['te']['n']); Fag = score_agg(pagg, Wd['agg']['inv'], Wd['agg']['n'])
        if K == 2:
            ths = np.linspace(0.02, 0.98, 49); ma = Fag[:, 0]
            th = ths[int(np.argmax([MCC(Wd['agg']['y'], (ma >= h).astype(int)) for h in ths]))]
            wlog(f'{arm}+meanvote', Fte[:, 0], dict(th=th), th=th)
        else:
            wlog(f'{arm}+meanvote', Fte[:, :K])
        lr_ = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced').fit(Fag, Wd['agg']['y'])
        wlog(f'{arm}+MIL', proba(lr_, Fte))
    # (4) proposed full: MIL score aggregates + raw window aggregates -> LightGBM, trained on 'agg' windows
    f = OUT + f'{TASK}_r{REP}_lgb_ctx.npz'
    if STAGE == 'D2' and os.path.exists(f):
        Z = np.load(f)
        Fte = np.hstack([score_agg(Z['te'], Wd['te']['inv'], Wd['te']['n']), Wd['te']['R']])
        Fag = np.hstack([score_agg(Z['agg'], Wd['agg']['inv'], Wd['agg']['n']), Wd['agg']['R']])
        Fva = np.hstack([score_agg(Z['val'], Wd['val']['inv'], Wd['val']['n']), Wd['val']['R']])
        m = lgb.LGBMClassifier(n_estimators=2000, num_leaves=15, learning_rate=0.05, min_child_samples=10, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, n_jobs=2, verbose=-1, random_state=SEED)
        m.fit(Fag, Wd['agg']['y'], sample_weight=compute_sample_weight('balanced', Wd['agg']['y']), eval_set=[(Fva, Wd['val']['y'])], callbacks=[lgb.early_stopping(50, verbose=False)])
        wlog('lgb_ctx+MIL-stack', proba(m, Fte))

if STAGE == 'D3':
    # Proposed hybrid: cross-fitted flow-level LightGBM scores (MIL instance scores) + raw window aggregates
    # -> window-level LightGBM. Same training captures (fit+agg), same validation captures, same tuning grid
    # as the window-level baselines in D1; the only difference is the added instance-score features.
    from sklearn.utils.class_weight import compute_sample_weight
    FCFG = dict(num_leaves=63, learning_rate=0.05, min_child_samples=100, colsample_bytree=0.6, reg_lambda=1)
    T = pd.concat([fit, agg]); folds = np.arange(len(T)) % 3; rng.shuffle(folds)
    def flow_model(Itr):
        m = lgb.LGBMClassifier(n_estimators=2000, subsample=0.8, subsample_freq=1, n_jobs=2, verbose=-1, random_state=SEED, **FCFG)
        m.fit(X[Itr][:, iF], y[Itr], eval_set=[(X[I_val][:, iF], y[I_val])], callbacks=[lgb.early_stopping(50, verbose=False)]); return m
    I_T = np.flatnonzero(np.isin(cid, T.cid.values)); oof = np.zeros(len(I_T), np.float32) if K == 2 else np.zeros((len(I_T), K), np.float32)
    for k in range(3):
        trc, hoc = T.cid.values[folds != k], T.cid.values[folds == k]
        m = flow_model(capped(np.isin(cid, trc), CAP)); h = np.isin(cid[I_T], hoc)
        oof[h] = proba(m, X[I_T[h]][:, iF])
    m = flow_model(capped(np.isin(cid, T.cid.values), CAP))
    p_val, p_te = proba(m, X[I_val_all][:, iF]), proba(m, X[I_te][:, iF])
    def wfeat(I, p):
        u, inv, n = np.unique(win[I], return_inverse=True, return_counts=True)
        A = np.sign(X[I][:, iF].astype(np.float64)); A = np.nan_to_num(A * np.log1p(np.abs(np.nan_to_num(X[I][:, iF].astype(np.float64)))))
        s1 = np.zeros((len(u), A.shape[1])); s2 = np.zeros_like(s1); mx = np.full_like(s1, -np.inf)
        np.add.at(s1, inv, A); np.add.at(s2, inv, A * A); np.maximum.at(mx, inv, A)
        mu = s1 / n[:, None]; sd = np.sqrt(np.maximum(s2 / n[:, None] - mu * mu, 0))
        P = pd.DataFrame(p if p.ndim > 1 else p[:, None]); g = P.groupby(inv)
        sa = np.hstack([g.mean().values, g.max().values, g.quantile(.5).values, g.quantile(.9).values, g.std().fillna(0).values])
        yl = pd.Series(y[I]).groupby(inv).first().values
        return np.hstack([mu, sd, mx, np.log1p(n)[:, None]]).astype(np.float32), sa.astype(np.float32), yl, u // 100000
    Rtr, Str, ytr, _ = wfeat(I_T, oof); Rva, Sva, yva, _ = wfeat(I_val_all, p_val); Rte, Ste, yte, cte = wfeat(I_te, p_te)
    sw = compute_sample_weight('balanced', ytr)
    gl = [dict(num_leaves=nl, learning_rate=0.05, min_child_samples=mc) for nl in (15, 31, 63) for mc in (10, 30)]
    for arm, (A_tr, A_va, A_te) in {'HYBRID_scores+raw': (np.hstack([Str, Rtr]), np.hstack([Sva, Rva]), np.hstack([Ste, Rte])),
                                     'HYBRID_scores_only': (Str, Sva, Ste)}.items():
        best = None
        for c in gl:
            mm = lgb.LGBMClassifier(n_estimators=2000, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=2, verbose=-1, random_state=SEED, **c)
            mm.fit(A_tr, ytr, sample_weight=sw, eval_set=[(A_va, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
            s = metr(yva, proba(mm, A_va))['mcc']
            if best is None or s > best[0]: best = (s, proba(mm, A_te), c)
        pt = best[1]; log(dict(level='window', arm=arm, n_units=len(yte), cfg=json.dumps(best[2]), **metr(yte, pt)))
        cp = pd.DataFrame(pt if pt.ndim > 1 else pt[:, None]).groupby(cte).mean().values
        cy = pd.Series(yte).groupby(cte).first().values
        log(dict(level='capture', arm=arm, n_units=len(cy), **metr(cy, cp[:, 0] if K == 2 else cp)))
