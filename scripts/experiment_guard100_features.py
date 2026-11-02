"""Handcrafted feature baselines on corrected guard100 real windows."""
import json,time
from pathlib import Path
import numpy as np
from scipy.stats import skew,kurtosis
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier,RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,confusion_matrix,balanced_accuracy_score,f1_score
def extract(X):
 out=[];freq=np.fft.rfftfreq(X.shape[-1],1/16000);bands=[(0,250),(250,500),(500,1000),(1000,2000),(2000,4000),(4000,8000)]
 for w in X:
  row=[]
  for z in w:
   z=z.astype(np.float64);d=np.diff(z);sp=np.abs(np.fft.rfft(z-z.mean()))**2;tot=sp.sum()+1e-12;ps=sp/tot
   row += [z.mean(),z.std(),np.sqrt(np.mean(z*z)),np.max(np.abs(z)),np.mean(np.abs(z)),np.median(np.abs(z)),np.percentile(np.abs(z),90),np.percentile(np.abs(z),99),np.sqrt(np.mean(d*d)),np.mean(np.abs(d)),np.mean(np.diff(np.signbit(z))),skew(z),kurtosis(z),np.sum(z*z),np.sum(freq*sp)/tot,-np.sum(ps*np.log(ps+1e-12)),np.exp(np.mean(np.log(sp+1e-12)))/(np.mean(sp)+1e-12),np.argmax(sp)*16000/X.shape[-1]]
   for lo,hi in bands:
    m=(freq>=lo)&(freq<hi);row.append(float(sp[m].sum()/tot) if m.any() else 0.)
  out.append(row)
 return np.asarray(out,dtype=np.float32)
def met(y,p,t):
 z=(p>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
 return {"threshold":float(t),"balanced_accuracy":float(balanced_accuracy_score(y,z)),"accuracy":float((z==y).mean()),"f1":float(f1_score(y,z,zero_division=0)),"specificity":float(tn/(tn+fp)),"recall":float(tp/(tp+fn)),"confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]]}
def main():
 with np.load("data/interim/guard100/raw_windows.npz",allow_pickle=False) as d:X=d["X"].astype(np.float32)
 with np.load("data/interim/guard100/swallow_binary_windows.npz",allow_pickle=False) as d:y=d["y"].astype(int)
 s=json.loads(Path("data/interim/guard100/participant_split.json").read_text());tr=np.asarray(s["splits"]["train"]["indices"]);va=np.asarray(s["splits"]["validation"]["indices"])
 scale=np.quantile(np.abs(X[tr]),.995,axis=(0,2)).astype(np.float32);X=np.clip(X/scale[None,:,None],-8,8)
 F=extract(X);print("features",F.shape,"train",len(tr),"val",len(va),flush=True)
 models={
 "extra_trees":ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,max_features=.8,class_weight="balanced",n_jobs=-1,random_state=42),
 "random_forest":RandomForestClassifier(n_estimators=500,min_samples_leaf=2,max_features="sqrt",class_weight="balanced_subsample",n_jobs=-1,random_state=42),
 "hist_gradient":HistGradientBoostingClassifier(max_iter=250,learning_rate=.05,max_leaf_nodes=15,l2_regularization=2.,random_state=42),
 "svm_rbf":make_pipeline(SimpleImputer(),StandardScaler(),SVC(C=2,kernel="rbf",class_weight="balanced",probability=True,random_state=42)),
 "logistic":make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression(C=.1,class_weight="balanced",max_iter=2000))}
 rep={"note":"Corrected guard100 dataset, participant-wise train/validation, real data only, test untouched.","models":{}}
 for name,model in models.items():
  t=time.time();model.fit(F[tr],y[tr]);p=model.predict_proba(F[va])[:,1];sweep=[met(y[va],p,x) for x in np.arange(.1,.91,.05)];best=max(sweep,key=lambda z:z["balanced_accuracy"])
  rep["models"][name]={"auc":float(roc_auc_score(y[va],p)),"best":best,"seconds":round(time.time()-t,1)}
  print(name,"AUC",round(rep["models"][name]["auc"],4),"best",best,flush=True)
 Path("artifacts/reports/guard100_feature_experiments.json").write_text(json.dumps(rep,indent=2))
if __name__=="__main__":main()
