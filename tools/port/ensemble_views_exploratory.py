import json, os, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
B=os.environ.get('JADAL_BUNDLE','/workspace/jadal_pod_bundle')
M={i['id']:i for i in json.load(open(os.path.join(B,'labels','manifest_gold.json')))['items']}
F=json.load(open(os.path.join(B,'caches','gpu','features.json')))['windows']
z=np.load(os.path.join(B,'caches','frames10k','segments.npz'),allow_pickle=True)
ids=[i for i in z['ids']]; clip=np.array([M[i]['clip'] for i in ids]); y=np.array([M[i]['label'] for i in ids])
E=z['F'].astype(np.float32); cnt=z['counts']
Xe=np.stack([E[k,:max(1,int(cnt[k]))].mean(0) for k in range(len(ids))]); Xe/=np.linalg.norm(Xe,axis=1,keepdims=True)
names=sorted(F[ids[0]]['full']); Xg=np.nan_to_num(np.array([[F[i]['full'][n] for n in names] for i in ids],float))
def loco(X,C):
    p=np.zeros(len(y))
    for c in set(clip):
        tr=clip!=c; te=clip==c; sc=StandardScaler().fit(X[tr])
        m=LogisticRegression(max_iter=4000,C=C,class_weight='balanced').fit(sc.transform(X[tr]),y[tr]); p[te]=m.predict_proba(sc.transform(X[te]))[:,1]
    return p
pe=loco(Xe,0.01); pg=loco(Xg,0.3)
def rk(p): # rank within each held-out clip is NOT used (would leak clip prior); use global rank of OOF probs
    return rankdata(p)/len(p)
ens=(rk(pe)+rk(pg))/2; ensp=(pe+pg)/2
rng=np.random.default_rng(0); clips=sorted(set(clip))
def boot(p,B=1000):
    out=[]
    for _ in range(B):
        cs=rng.choice(clips,len(clips)); idx=np.concatenate([np.where(clip==c)[0] for c in cs])
        if len(set(y[idx]))<2: continue
        out.append(roc_auc_score(y[idx],p[idx]))
    return np.percentile(out,[2.5,97.5])
def bal(p,t=0.5,sel=None):
    sel=np.arange(len(y)) if sel is None else sel
    pr=(p[sel]>t).astype(int); yy=y[sel]; return ((pr[yy==1]==1).mean()+(pr[yy==0]==0).mean())/2
print(f"n={len(y)} clips={len(clips)}")
for tag,p in [("EG2 mean-embedding logistic (C=0.01)",pe),("26 per-pixel feats logistic (C=0.3)",pg),("prob-average of both",ensp),("rank-average of both",ens)]:
    lo,hi=boot(p); print(f"{tag:40s} AUROC {roc_auc_score(y,p):.3f} [{lo:.3f},{hi:.3f}]  bal.acc@0.5 {bal(p):.3f}" if tag.startswith(('EG2','26','prob')) else f"{tag:40s} AUROC {roc_auc_score(y,p):.3f} [{lo:.3f},{hi:.3f}]")
print("per-clip AUROC (prob-average):",{c:round(roc_auc_score(y[clip==c],ensp[clip==c]),2) for c in clips if 0<y[clip==c].sum()<(clip==c).sum()})
conf=np.abs(ensp-0.5); order=np.argsort(-conf)
print("selective (prob-average, threshold 0.5): coverage -> balanced acc, good wrongly rejected")
for cov in (1.0,0.8,0.6,0.5):
    k=int(cov*len(y)); s=order[:k]; fr=((ensp[s]<=0.5)&(y[s]==1)).sum()
    print(f"  {cov:4.0%}: {bal(ensp,0.5,s):.3f}  {fr}/{int((y[s]==1).sum())}")
