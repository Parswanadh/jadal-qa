# Reproduces the LOCO baseline: 26 GPU per-pixel features + class-balanced logistic (C=0.3), leave-one-clip-out. Expect pooled AUROC ~0.81 (good-vs-semantic-bad 0.80, good-vs-quality-bad 0.85; pallet/pallet2 at 0.50). Run: OMP_NUM_THREADS=1 python3 baseline_logistic_loco.py
import json, os, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
B=os.environ.get('JADAL_BUNDLE','/workspace/jadal_pod_bundle')
F=json.load(open(os.path.join(B,'caches','gpu','features.json')))['windows']
M={i['id']:i for i in json.load(open(os.path.join(B,'labels','manifest_gold.json')))['items']}
ids=[k for k in M if k in F and F[k].get('full')]
names=sorted(F[ids[0]]['full'].keys())
X=np.array([[F[k]['full'][n] for n in names] for k in ids],float); X=np.nan_to_num(X)
y=np.array([M[k]['label'] for k in ids]); clip=np.array([M[k]['clip'] for k in ids]); reason=np.array([M[k]['reason'] for k in ids])
print("n",len(ids),"features",len(names),"bad by reason:",{r:int((reason==r).sum()) for r in set(reason) if r})
print("bad rate per clip:",{c:f"{int((y[clip==c]==0).sum())}/{int((clip==c).sum())}" for c in sorted(set(clip))})
def loco(Xm, mask=None):
    p=np.zeros(len(y))
    for c in set(clip):
        tr=clip!=c; te=clip==c
        sc=StandardScaler().fit(Xm[tr]); m=LogisticRegression(max_iter=3000,C=0.3,class_weight='balanced').fit(sc.transform(Xm[tr]),y[tr])
        p[te]=m.predict_proba(sc.transform(Xm[te]))[:,1]
    return p
def auc(p,sel): return roc_auc_score(y[sel],p[sel])
# within-clip z-scoring (unsupervised: uses only that clip's own feature distribution, no labels)
Xz=X.copy()
for c in set(clip):
    s=clip==c; mu=X[s].mean(0); sd=X[s].std(0)+1e-9; Xz[s]=(X[s]-mu)/sd
good=(y==1); sem=good|np.isin(reason,['NO_ACTIVE_WORK','HANDS_NOT_VISIBLE']); qual=good|np.isin(reason,['TOO_DARK','DUPLICATE','UNDECODABLE'])
for tag,Xm in [("raw",X),("within-clip z",Xz)]:
    p=loco(Xm)
    print(f"{tag:14s} all AUROC {auc(p,np.ones(len(y),bool)):.3f} | good-vs-SEMANTIC-bad {auc(p,sem):.3f} | good-vs-QUALITY-bad {auc(p,qual):.3f}",
          " | per-clip:",{c:round(roc_auc_score(y[clip==c],p[clip==c]),2) for c in sorted(set(clip)) if 0<y[clip==c].sum()<(clip==c).sum()})
# best single features (pooled AUROC, orientation-free) on semantic-bad
rows=[]
for j,n in enumerate(names):
    a=roc_auc_score(y[sem],X[sem][:,j]); rows.append((max(a,1-a),n,a))
rows.sort(reverse=True); print("top single features (semantic-bad vs good, pooled AUROC):",[(n,round(a,2)) for _,n,a in rows[:5]])
