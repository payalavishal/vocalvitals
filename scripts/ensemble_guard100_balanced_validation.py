"""Validation-only ensembles on balanced, overlap-corrected real dataset."""
import json
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score,confusion_matrix,balanced_accuracy_score,accuracy_score,f1_score
from experiment_guard100_balanced_spectrogram import Net as SpecNet,pred as spec_pred
from experiment_guard100_balanced_1dcnn import Net as RawNet,predict as raw_pred
from experiment_guard100_balanced_features import extract
def score(y,p,t):
 z=(p>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
 return {"threshold":float(t),"accuracy":float(accuracy_score(y,z)),"balanced_accuracy":float(balanced_accuracy_score(y,z)),"f1":float(f1_score(y,z,zero_division=0)),"specificity":float(tn/(tn+fp)),"recall":float(tp/(tp+fn)),"cm":[[int(tn),int(fp)],[int(fn),int(tp)]]}
def report(name,y,p):
 sw=[score(y,p,t) for t in np.arange(.1,.91,.025)]
 return {"auc":float(roc_auc_score(y,p)),"best_accuracy":max(sw,key=lambda x:x["accuracy"]),"best_balanced_accuracy":max(sw,key=lambda x:x["balanced_accuracy"])}
def main():
 dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 with np.load("data/interim/guard100_balanced/raw_windows.npz",allow_pickle=False) as d:X=d["X"].astype(np.float32)
 with np.load("data/interim/guard100_balanced/swallow_binary_windows.npz",allow_pickle=False) as d:y=d["y"].astype(int)
 s=json.loads(Path("data/interim/guard100_balanced/participant_split.json").read_text());tr=np.asarray(s["splits"]["train"]["indices"]);va=np.asarray(s["splits"]["validation"]["indices"])
 cck=torch.load("artifacts/checkpoints/guard100_balanced_spectrogram_best.pt",map_location=dev,weights_only=True)
 rck=torch.load("artifacts/checkpoints/guard100_balanced_1dcnn_best.pt",map_location=dev,weights_only=True)
 Xc=np.clip(X/np.asarray(cck["scale"],np.float32)[None,:,None],-8,8)
 Xr=np.clip(X/np.asarray(rck["scale"],np.float32)[None,:,None],-8,8)
 m=SpecNet().to(dev);m.load_state_dict(cck["model_state_dict"])
 vl=DataLoader(TensorDataset(torch.from_numpy(Xc[va]),torch.from_numpy(y[va].astype(np.float32))),batch_size=64,shuffle=False)
 yy,pc=spec_pred(m,vl,dev)
 mr=RawNet().to(dev);mr.load_state_dict(rck["model_state_dict"])
 vlr=DataLoader(TensorDataset(torch.from_numpy(Xr[va]),torch.from_numpy(y[va].astype(np.float32))),batch_size=96,shuffle=False)
 yr,pd=raw_pred(mr,vlr,dev);assert np.array_equal(yy,yr)
 F=extract(Xc)
 rf=RandomForestClassifier(n_estimators=700,min_samples_leaf=2,max_features="sqrt",class_weight="balanced_subsample",n_jobs=-1,random_state=42)
 rf.fit(F[tr],y[tr]);pr=rf.predict_proba(F[va])[:,1]
 svm=make_pipeline(SimpleImputer(),StandardScaler(),SVC(C=2,kernel="rbf",class_weight="balanced",probability=True,random_state=42))
 svm.fit(F[tr],y[tr]);ps=svm.predict_proba(F[va])[:,1]
 probs={"spectrogram_cnn":pc,"raw_1dcnn":pd,"random_forest":pr,"svm_rbf":ps}
 combos={**probs,
 "cnn_svm_50_50":.5*pc+.5*ps,
 "cnn_rf_50_50":.5*pc+.5*pr,
 "cnn_1d_50_50":.5*pc+.5*pd,
 "cnn_rf_svm_equal":(pc+pr+ps)/3,
 "cnn_rf_1d_equal":(pc+pr+pd)/3,
 "cnn_svm_1d_equal":(pc+ps+pd)/3,
 "all_equal":(pc+pr+pd+ps)/4,
 "cnn60_svm20_rf10_1d10":.6*pc+.2*ps+.1*pr+.1*pd}
 results={}
 for name,p in combos.items():
  results[name]=report(name,yy,p)
  print(name,"AUC",round(results[name]["auc"],4),"best_bal",results[name]["best_balanced_accuracy"],"best_acc",results[name]["best_accuracy"],flush=True)
 np.savez_compressed("artifacts/reports/guard100_balanced_validation_probabilities.npz",y=yy,**probs)
 Path("artifacts/reports/guard100_balanced_ensemble_validation.json").write_text(json.dumps({"note":"Validation only; participant-disjoint; real data only; test untouched.","results":results},indent=2))
if __name__=="__main__":main()
