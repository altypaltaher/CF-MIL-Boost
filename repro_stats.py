"""Dataset-level statistics reported in the paper, computed from ctx_flows.parquet (output of build_ctx.py).

  1. ROC-AUC of elapsed capture time alone as a flow score (binary label), on full captures and within 300 s
  2. Capture-length statistics by class (median, seconds) and captures that cover the full 300 s
  3. Label conflicts: distinct 80-feature flow vectors that occur with more than one five-class label
Reads the file in batches to keep memory low. Writes repro_stats.json.
"""
import os, json, numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
W = os.environ.get('CFMIL_WORK', os.path.dirname(os.path.abspath(__file__))) + '/'
F = W + 'ctx_flows.parquet'
FEAT = [l.strip() for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'features.txt')) if l.strip()]
pf = pq.ParquetFile(F)
T, C, ID, H = [], [], [], []
for b in pf.iter_batches(columns=FEAT + ['t', 'cat', 'cid'], batch_size=200_000):
    d = b.to_pandas()
    T.append(d.t.to_numpy(np.float32)); C.append(d.cat.to_numpy()); ID.append(d.cid.to_numpy())
    Xf = d[FEAT].astype(np.float32); Xf = Xf.where(np.isfinite(Xf))   # inf -> missing, as in the models
    H.append(pd.util.hash_pandas_object(Xf, index=False).to_numpy())
t = np.concatenate(T); cat = np.concatenate(C); cid = np.concatenate(ID); h = np.concatenate(H)
ybin = (cat != 1).astype(int)   # category code 1 = Benign
names = {0: 'Adware', 1: 'Benign', 2: 'Ransomware', 3: 'SMSmalware', 4: 'Scareware'}
R = {}
R['n_flows'] = int(len(t)); R['n_captures'] = int(len(np.unique(cid)))
R['auc_elapsed_time_full'] = float(roc_auc_score(ybin, t))
m = t < 300; R['auc_elapsed_time_300s'] = float(roc_auc_score(ybin[m], t[m])); R['n_flows_300s'] = int(m.sum())
cap = pd.DataFrame({'cid': cid, 'cat': cat, 't': t}).groupby('cid').agg(cat=('cat', 'first'), dur=('t', 'max'))
R['median_capture_length_s'] = {names[k]: float(v) for k, v in cap.groupby('cat').dur.median().items()}
R['captures_covering_300s'] = int((cap.dur >= 300).sum())   # capture lasts at least 300 s
R['captures_total'] = int(len(cap))
g = pd.DataFrame({'h': h, 'c': cat}).drop_duplicates()
nlab = g.groupby('h').c.nunique()
R['distinct_flow_vectors'] = int(len(nlab))
R['distinct_vectors_with_conflicting_labels_5class'] = int((nlab > 1).sum())
R['flows_with_conflicting_vector_5class'] = int(np.isin(h, nlab.index[nlab > 1].to_numpy()).sum())
gb = pd.DataFrame({'h': h, 'c': ybin}).drop_duplicates().groupby('h').c.nunique()
R['distinct_vectors_with_conflicting_labels_binary'] = int((gb > 1).sum())
json.dump(R, open(W + 'repro_stats.json', 'w'), indent=1); print(json.dumps(R, indent=1))
