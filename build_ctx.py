"""Build per-flow table from CICAndMal2017 per-capture CSVs, with capture IDs and causal
host-context features (computed only from earlier flows of the same device in the same capture).
Output: ctx_flows.parquet"""
import pandas as pd, numpy as np, glob, os, sys, time
from multiprocessing import Pool
# CICAndMal2017 CSV root: <ROOT>/<Category>/<Family>/<capture>.csv (set CFMIL_DATA)
ROOT = os.environ.get('CFMIL_DATA', 'data/CICAndMal2017')
W = os.environ.get('CFMIL_WORK', os.path.dirname(os.path.abspath(__file__))) + '/'
FEAT = [l.strip() for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'features.txt')) if l.strip()]
CATS = ['Adware', 'Benign', 'Ransomware', 'SMSmalware', 'Scareware']
META = ['Source IP', 'Destination IP', 'Timestamp']

def one(args):
    cid, f = args
    cat, fam = f.split('/')[-3], f.split('/')[-2]
    d = pd.read_csv(f, skipinitialspace=True, low_memory=False)
    d.columns = [c.strip() for c in d.columns]
    if len(d) == 0: return None
    d = d[[c for c in FEAT if c in d.columns] + META].copy()
    for c in FEAT:
        if c not in d: d[c] = np.nan
    t = pd.to_datetime(d['Timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    if t.isna().mean() > 0.5:
        t = pd.to_datetime(d['Timestamp'], dayfirst=True, errors='coerce')
    d = d[t.notna()].copy(); t = t[t.notna()]
    loc = pd.concat([d['Source IP'], d['Destination IP']])
    loc = loc[loc.astype(str).str.startswith('10.42.0.')]
    dev = loc.value_counts().index[0] if len(loc) else d['Source IP'].mode()[0]
    out = d['Source IP'] == dev
    d['remote'] = np.where(out, d['Destination IP'], d['Source IP'])
    d['remote'] = d['remote'].astype(str)
    d['rport'] = np.where(out, d['Destination Port'], d['Source Port'])
    d['t'] = (t - t.min()).dt.total_seconds().astype(np.float32)
    d = d.sort_values('t', kind='stable').reset_index(drop=True)
    d['ts'] = pd.to_datetime(d['t'], unit='s')
    out = (d['Source IP'] == dev).values
    # indicators
    num = lambda c: pd.to_numeric(d[c], errors='coerce').fillna(0).astype(np.float64)
    dur = num('Flow Duration'); byts = num('Total Length of Fwd Packets') + num('Total Length of Bwd Packets')
    ind = pd.DataFrame({
        'one': 1.0,
        'newip': (~d['remote'].duplicated()).astype(float),
        'newport': (~d['rport'].duplicated()).astype(float),
        'dns': (d['rport'] == 53).astype(float),
        'web': d['rport'].isin([80, 443]).astype(float),
        'short': (dur < 1000).astype(float),
        'rst': (num('RST Flag Count') > 0).astype(float),
        'lbytes': np.log1p(byts.clip(lower=0)),
        'ldur': np.log1p(dur.clip(lower=0)),
        'inbound': (~out).astype(float),
    }, index=d.index)
    ind.index = d['ts']
    ctx = {}
    for w in (10, 60, 300):
        r = ind.rolling(f'{w}s').sum()
        n = r['one'].values
        ctx[f'ctx_n{w}'] = n
        if w in (60, 300):
            for k in ['newip', 'newport']: ctx[f'ctx_{k}{w}'] = r[k].values
            for k in ['dns', 'web', 'short', 'rst', 'lbytes', 'ldur', 'inbound']: ctx[f'ctx_{k}{w}'] = r[k].values / n
    ctx['ctx_burst'] = ctx['ctx_n10'] / (ctx['ctx_n60'] / 6.0)
    # flows to the same remote IP in the past 60 s (repetition / beaconing)
    ctx['ctx_samerem60'] = _same_remote(d['t'].values, d['remote'].values)
    # gap since previous flow to same remote (beacon periodicity)
    prev = d.groupby('remote')['t'].diff().fillna(-1).values
    ctx['ctx_gap_same'] = prev.astype(np.float32)
    o = d[FEAT].apply(pd.to_numeric, errors='coerce').astype(np.float32)
    for k, v in ctx.items(): o[k] = np.asarray(v, dtype=np.float32)
    o['t'] = d['t'].values; o['cid'] = cid; o['cat'] = CATS.index(cat); o['fam'] = fam
    return o

def _same_remote(t, rem):
    res = np.zeros(len(t), np.float32)
    s = pd.Series(np.arange(len(t)))
    for r, idx in s.groupby(rem).groups.items():
        idx = np.asarray(idx); tt = t[idx]
        lo = np.searchsorted(tt, tt - 60, side='left')
        res[idx] = np.arange(len(idx)) - lo + 1
    return res

if __name__ == '__main__':
    fs = sorted(glob.glob(ROOT + '/*/*/*.csv'))
    print(len(fs), 'captures'); t0 = time.time()
    with Pool(2) as p: parts = p.map(one, list(enumerate(fs)), chunksize=8)
    parts = [x for x in parts if x is not None]
    D = pd.concat(parts, ignore_index=True)
    D['fam'] = D['fam'].astype('category')
    cap = pd.DataFrame({'cid': range(len(fs)), 'file': fs})
    cap.to_csv(W + 'ctx_captures.csv', index=False)
    D.to_parquet(W + 'ctx_flows.parquet')
    print(D.shape, D.groupby('cat').cid.nunique().to_dict(), D.groupby('cat').size().to_dict(), f'{time.time()-t0:.0f}s')
