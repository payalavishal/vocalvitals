"""Validation-only feature/model experiments on original real VocalVitals windows."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, RandomForestRegressor, VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def extract(X):
    out=[]
    fs=16000
    freq=np.fft.rfftfreq(X.shape[-1],1/fs)
    bands=[(0,250),(250,500),(500,1000),(1000,2000),(2000,4000),(4000,8000)]
    for wi,w in enumerate(X):
        feats=[]
        for ch in w:
            z=np.asarray(ch,dtype=np.float64)
            d=np.diff(z)
            spec=np.abs(np.fft.rfft(z-z.mean()))**2
            total=spec.sum()+1e-12
            ps=spec/total
            feats += [
                z.mean(),z.std(),np.sqrt(np.mean(z*z)),np.max(np.abs(z)),
                np.mean(np.abs(z)),np.median(np.abs(z)),np.percentile(np.abs(z),90),
                np.percentile(np.abs(z),99),np.sqrt(np.mean(d*d)),
                np.mean(np.abs(d)),np.mean(np.diff(np.signbit(z))),
                skew(z),kurtosis(z),np.sum(z*z),
                np.sum(freq*spec)/total,
                -np.sum(ps*np.log(ps+1e-12)),
                np.exp(np.mean(np.log(spec+1e-12)))/(np.mean(spec)+1e-12),
                np.argmax(spec)*fs/X.shape[-1],
            ]
            for lo,hi in bands:
                m=(freq>=lo)&(freq<hi)
                feats.append(float(spec[m].sum()/total) if m.any() else 0.)
        out.append(feats)
    return np.asarray(out,dtype=np.float32)

def metrics(y,p,t):
    pred=(p>=t).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"threshold":float(t),"accuracy":float((pred==y).mean()),
      "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
      "f1":float(f1_score(y,pred,zero_division=0)),
      "specificity":float(tn/(tn+fp)) if tn+fp else 0.,
      "recall":float(tp/(tp+fn)) if tp+fn else 0.,
      "confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]]}

def main():
    base=Path(".")
    with np.load(base/"data/processed/swallow_binary_windows.npz",allow_pickle=False) as d:
        X=d["X"].astype(np.float32); y=d["y"].astype(int)
    m=json.loads((base/"artifacts/reports/participant_split.json").read_text())
    tr=np.asarray(m["splits"]["train"]["indices"]); va=np.asarray(m["splits"]["validation"]["indices"])
    print("Real dataset only. Train/validation only; test indices are not loaded.")
    print("X",X.shape,"train",len(tr),"val",len(va),"class counts",np.bincount(y))
    t0=time.time(); F=extract(X); print("features",F.shape,"seconds",round(time.time()-t0,1),flush=True)
    models={
      "extra_trees":ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,max_features=.8,class_weight="balanced",n_jobs=-1,random_state=42),
      "random_forest":RandomForestClassifier(n_estimators=500,min_samples_leaf=2,max_features="sqrt",class_weight="balanced_subsample",n_jobs=-1,random_state=42),
      "hist_gradient":HistGradientBoostingClassifier(max_iter=250,learning_rate=.05,max_leaf_nodes=15,l2_regularization=2.,random_state=42),
      "svm_rbf":make_pipeline(SimpleImputer(),StandardScaler(),SVC(C=2.,kernel="rbf",class_weight="balanced",probability=True,random_state=42)),
      "logistic":make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression(C=.1,class_weight="balanced",max_iter=2000)),
      "lda":make_pipeline(SimpleImputer(),StandardScaler(),LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto")),
    }
    report={"note":"Validation-only; original real four-channel windows; no synthetic data; test untouched.","models":{}}
    for name,model in models.items():
      start=time.time()
      model.fit(F[tr],y[tr])
      p=model.predict_proba(F[va])[:,1]
      auc=float(roc_auc_score(y[va],p))
      sweep=[metrics(y[va],p,t) for t in np.arange(.1,.91,.05)]
      best=max(sweep,key=lambda z:z["balanced_accuracy"])
      report["models"][name]={"auc":auc,"best_validation_balanced_accuracy":best,"threshold_sweep":sweep,"seconds":round(time.time()-start,1)}
      print(name,"AUC",round(auc,4),"best_bal_acc",round(best["balanced_accuracy"],4),"thr",best["threshold"],"spec",round(best["specificity"],4),"f1",round(best["f1"],4),flush=True)
    out=base/"artifacts/reports/real_feature_experiments.json"
    out.write_text(json.dumps(report,indent=2))
    print("Saved",out)
if __name__=="__main__": main()
