import pandas as pd, numpy as np, sys
from scipy import stats
TASK = sys.argv[1] if len(sys.argv) > 1 else 'bin'
D = pd.read_json(f'mil/{TASK}_results.jsonl', lines=True)
D = D.drop_duplicates(['level', 'arm', 'rep'], keep='last')
pd.set_option('display.width', 250)
S = D.groupby(['level', 'arm'])[['mcc', 'f1m', 'auc']].agg(['mean', 'std', 'count']).round(4)
print(S.to_string())
P = 'HYBRID_scores+raw'; R = 0.25  # n_test/n_train (20% / 80% of captures)
def nb(d):  # Nadeau-Bengio corrected resampled t-test
    J = len(d); m = d.mean(); v = d.var(ddof=1)
    t = m / np.sqrt((1 / J + R) * v) if v > 0 else np.inf
    return 2 * stats.t.sf(abs(t), J - 1)
rows = []
for lvl in ['window', 'capture']:
    base = ['win_LightGBM', 'win_XGBoost', 'win_AdaBoost', 'win_GBM', 'lgb_flow+meanvote', 'xgb_flow+meanvote',
            'ada_flow+meanvote', 'gbm_flow+meanvote', 'lgb_flow+MIL', 'xgb_flow+MIL', 'lgb_ctx+MIL', 'HYBRID_scores_only']
    for met in ['mcc', 'auc', 'f1m']:
        T = D[D.level == lvl].pivot(index='rep', columns='arm', values=met)
        for b in base:
            if b not in T or P not in T: continue
            d = (T[P] - T[b]).dropna()
            rows.append(dict(level=lvl, metric=met, baseline=b, n=len(d), proposed=T[P].mean(), base=T[b].mean(),
                             diff_pp=d.mean() * 100, W=(d > 0).sum(), L=(d < 0).sum(), p_NB=nb(d.values),
                             p_wil=stats.wilcoxon(d).pvalue if len(d) > 5 and d.abs().sum() > 0 else np.nan))
O = pd.DataFrame(rows); O.to_csv(f'mil/{TASK}_pairwise.csv', index=False)
print(O.round(4).to_string())
