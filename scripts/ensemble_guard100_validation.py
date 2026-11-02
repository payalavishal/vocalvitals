"""Validation-only blend check for guard100 spectrogram CNN + RF."""
import json
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,confusion_matrix,balanced_accuracy_score,accuracy_score,f1_score
from experiment_guard100_spectrogram import Net,spec,pred
from experiment_guard100_features import extract
def score(y,p,t):
 z=(p>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
 return {"threshold":float(t),"accuracy":float(accuracy_score(y,z)),"balanced_accuracy":float(balanced_accuracy_score(y,z)),"f1":float(f1_score(y,z,zero_division=0)),"specificity":float(tn/(tn+fp)),"recall":float(tp/(tp+fn)),"cm":[[int(tn),int(fp)],[int(fn),int(tp)]]}
def main():
 dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 with np.load("data/interim/guard100/raw_windows.npz",allow_pickle=False) as d:X=d["X"].astype(np.float32)
 with np.load("data/interim/guard100/swallow_binary_windows.npz",allow_pickle=False) as d:y=d["y"].astype(int)
 s=json.loads(Path("data/interim/guard100/participant_split.json").read_text());tr=np.asarray(s["splits"]["train"]["indices"]);va=np.asarray(s["splits"]["validation"]["indices"])
 ck=torch.load("artifacts/checkpoints/guard100_spectrogram_best.pt",map_location=dev,weights_only=True)
 scale=np.asarray(ck["scale"],np.float32);X=np.clip(X/scale[None,:,None],-8,8)
 m=Net().to(dev);m.load_state_dict(ck["model_state_dict"])
 vl=DataLoader(TensorDataset(torch.from_numpy(X[va]),torch.from_numpy(y[va].astype(np.float32))),batch_size=64,shuffle=False)
 yy,pc=pred(m,vl,dev)
 F=extract(X)
 rf=RandomForestClassifier(n_estimators=700,min_samples_leaf=2,max_features="sqrt",class_weight="balanced_subsample",n_jobs=-1,random_state=42)
 rf.fit(F[tr],y[tr]);pr=rf.predict_proba(F[va])[:,1]
 results={}
 for name,p in [("spectrogram_cnn",pc),("random_forest",pr),("blend_50_50",.5*pc+.5*pr),("blend_70cnn_30rf",.7*pc+.3*pr),("blend_30cnn_70rf",.3*pc+.7*pr)]:
  sweep=[score(yy,p,t) for t in np.arange(.1,.91,.025)]
  results[name]={"auc":float(roc_auc_score(yy,p)),"best_accuracy":max(sweep,key=lambda z:z["accuracy"]),"best_balanced_accuracy":max(sweep,key=lambda z:z["balanced_accuracy"])}
  print(name,"AUC",round(results[name]["auc"],4),"best_acc",results[name]["best_accuracy"],"best_bal",results[name]["best_balanced_accuracy"],flush=True)
 np.savez_compressed("artifacts/reports/guard100_validation_probabilities.npz",y=yy,cnn=pc,rf=pr)
 Path("artifacts/reports/guard100_ensemble_validation.json").write_text(json.dumps({"note":"Validation only; test untouched.","results":results},indent=2))
if __name__=="__main__":main()
