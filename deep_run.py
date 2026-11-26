"""Deep multiple-instance baselines on the same capture splits as mil_run.py / ext_run.py (main mode).

usage: python3 deep_run.py <bin|cat> <rep> <attn|gru>
  attn : gated attention-based deep MIL (Ilse et al., ICML 2018) over the flows of a window
  gru  : bidirectional GRU over the time-ordered flows of a window (sequence model)
Each window is represented by at most L=64 of its flows (random subset, time order kept); features are
signed-log transformed and standardised with training statistics. Class-balanced cross-entropy,
Adam (learning rate 1e-3 or 3e-4, selected on validation MCC), early stopping (patience 10, at most 80 epochs) on validation MCC.
The log flow count of the window is appended to the pooled representation, as for the boosting models. Capture decision = mean of window probabilities.
"""
import sys, os, json, time, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef as MCC, f1_score, roc_auc_score
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '2')))
W_ = os.environ.get('CFMIL_WORK', os.path.dirname(os.path.abspath(__file__))) + '/'
OUT = W_ + 'ext/'; os.makedirs(OUT + 'pred', exist_ok=True)
TASK, REP, ARCH = sys.argv[1], int(sys.argv[2]), sys.argv[3]
HORIZON, WIN, L = 300, 60, 64
SEED = 1000 + REP
torch.manual_seed(SEED); np.random.seed(SEED)

import pyarrow.parquet as pq
cols = [c for c in pq.read_schema(W_ + 'ctx_flows.parquet').names if not c.startswith('ctx_') and c != 'fam']
D = pd.read_parquet(W_ + 'ctx_flows.parquet', columns=cols, filters=[('t', '<', HORIZON)]).reset_index(drop=True)
FEAT = [c for c in D.columns if c not in ('t', 'cid', 'cat')]
X = np.nan_to_num(D[FEAT].to_numpy(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
X = (np.sign(X) * np.log1p(np.abs(X))).astype(np.float32)
cid = np.array(D.cid); cat = np.array(D.cat); t = np.array(D.t); del D
y = (cat != 1).astype(int) if TASK == 'bin' else cat
K = len(np.unique(y))
win = cid.astype(np.int64) * 100000 + (t // WIN).astype(np.int64)
caps = pd.DataFrame({'cid': cid, 'cat': cat}).drop_duplicates('cid')
tr, te = train_test_split(caps, test_size=0.2, stratify=caps.cat, random_state=SEED)
fit, rest = train_test_split(tr, test_size=0.3, stratify=tr.cat, random_state=SEED)
val, agg = train_test_split(rest, test_size=0.5, stratify=rest.cat, random_state=SEED)
rng = np.random.RandomState(SEED)

def bags(capset):
    I = np.flatnonzero(np.isin(cid, capset.cid.values))
    I = I[np.lexsort((t[I], win[I]))]
    u, start, n = np.unique(win[I], return_index=True, return_counts=True)
    B = np.zeros((len(u), L, X.shape[1]), np.float32); M = np.zeros((len(u), L), bool); Y = np.zeros(len(u), int)
    NC = np.log1p(n).astype(np.float32)[:, None]
    for k, (s, c) in enumerate(zip(start, n)):
        idx = I[s:s + c]
        if c > L: idx = idx[np.sort(rng.choice(c, L, replace=False))]
        B[k, :len(idx)] = X[idx]; M[k, :len(idx)] = True; Y[k] = y[idx[0]]
    return B, M, Y, u // 100000, NC
Btr, Mtr, Ytr, _, Ntr = bags(pd.concat([fit, agg])); Bva, Mva, Yva, _, Nva = bags(val); Bte, Mte, Yte, Cte, Nte = bags(te)
nm, ns = Ntr.mean(), Ntr.std() + 1e-6
Ntr, Nva, Nte = (Ntr - nm) / ns, (Nva - nm) / ns, (Nte - nm) / ns
mu = Btr[Mtr].mean(0); sd = Btr[Mtr].std(0) + 1e-6
for B, M in ((Btr, Mtr), (Bva, Mva), (Bte, Mte)): B[M] = (B[M] - mu) / sd

class Attn(nn.Module):
    def __init__(s, d, k):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU())
        s.V = nn.Linear(64, 32); s.U = nn.Linear(64, 32); s.w = nn.Linear(32, 1); s.c = nn.Linear(65, k)
    def forward(s, x, m, nc):
        h = s.f(x); a = s.w(torch.tanh(s.V(h)) * torch.sigmoid(s.U(h))).squeeze(-1)
        a = a.masked_fill(~m, -1e9).softmax(1); return s.c(torch.cat([(a.unsqueeze(-1) * h).sum(1), nc], 1))
class GRU(nn.Module):
    def __init__(s, d, k):
        super().__init__(); s.p = nn.Linear(d, 64); s.g = nn.GRU(64, 64, batch_first=True, bidirectional=True); s.c = nn.Linear(129, k)
    def forward(s, x, m, nc):
        o, _ = s.g(torch.relu(s.p(x))); mf = m.unsqueeze(-1).float()
        return s.c(torch.cat([(o * mf).sum(1) / mf.sum(1).clamp(min=1), nc], 1))
cw = torch.tensor(len(Ytr) / (K * np.bincount(Ytr, minlength=K)), dtype=torch.float32)
lossf = nn.CrossEntropyLoss(weight=cw)
T = lambda a: torch.from_numpy(a)
def predict(net, B, M, Nc):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(B), 512): out.append(net(T(B[i:i + 512]), T(M[i:i + 512]), T(Nc[i:i + 512])).softmax(1).numpy())
    P = np.vstack(out); return P[:, 1] if K == 2 else P
def mcc(yt, p): return MCC(yt, (p >= .5).astype(int) if p.ndim == 1 else p.argmax(1))
t0 = time.time(); sel = None
for lr in (1e-3, 3e-4):   # learning rate selected on validation MCC
    torch.manual_seed(SEED); net = (Attn if ARCH == 'attn' else GRU)(X.shape[1], K)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4); best = (-2, None, 0); bad = 0
    for ep in range(80):
        net.train(); perm = rng.permutation(len(Ytr))
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]; opt.zero_grad()
            l = lossf(net(T(Btr[b]), T(Mtr[b]), T(Ntr[b])), T(Ytr[b])); l.backward(); opt.step()
        s_ = mcc(Yva, predict(net, Bva, Mva, Nva))
        if s_ > best[0]: best = (s_, {k: v.clone() for k, v in net.state_dict().items()}, ep); bad = 0
        else:
            bad += 1
            if bad >= 10: break
    net.load_state_dict(best[1])
    if sel is None or best[0] > sel[0]: sel = (best[0], net, best[2], lr)
net = sel[1]; best = (sel[0], None, sel[2]); tt = time.time() - t0
ti = time.time(); pt = predict(net, Bte, Mte, Nte); ti = time.time() - ti
def metr(yt, p):
    pred = (p >= .5).astype(int) if p.ndim == 1 else p.argmax(1)
    auc = roc_auc_score(yt, p) if p.ndim == 1 else roc_auc_score(yt, p, multi_class='ovr', average='macro')
    return dict(mcc=MCC(yt, pred), f1m=f1_score(yt, pred, average='macro'), auc=auc)
arm = {'attn': 'AttnMIL', 'gru': 'BiGRU'}[ARCH]
recs = [dict(level='window', arm=arm, n_units=len(Yte), **metr(Yte, pt))]
cp = pd.DataFrame(pt if pt.ndim > 1 else pt[:, None]).groupby(Cte).mean().values
cy = pd.Series(Yte).groupby(Cte).first().values
recs.append(dict(level='capture', arm=arm, n_units=len(cy), **metr(cy, cp[:, 0] if K == 2 else cp)))
for r in recs:
    r.update(task=TASK, mode='main', rep=REP, epochs=best[2] + 1, lr=sel[3], train_s=round(tt, 1), infer_ms_per_window=1000 * ti / len(Yte))
    with open(OUT + f'deep_{TASK}.jsonl', 'a') as fh: fh.write(json.dumps(r, default=float) + '\n')
    print(r, flush=True)
