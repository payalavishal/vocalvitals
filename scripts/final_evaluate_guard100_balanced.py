"""One-time final test of validation-selected five-model ensemble. Do not tune on test."""
import json,joblib
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score,accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from experiment_guard100_balanced_spectrogram import Net as Spec256,pred as pred256
from experiment_guard100_balanced_spec512 import Net as Spec512,pred as pred512
from experiment_guard100_balanced_1dcnn import Net as Raw1D,predict as pred1d
from experiment_guard100_balanced_features import extract

def metrics(y,p,threshold=.5):
 z=(p>=threshold).astype(int);tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
 return {"threshold":threshold,"accuracy":float(accuracy_score(y,z)),"balanced_accuracy":float(balanced_accuracy_score(y,z)),
 "precision":float(precision_score(y,z,zero_division=0)),"recall":float(recall_score(y,z,zero_division=0)),
 "specificity":float(tn/(tn+fp)) if tn+fp else 0.,"f1":float(f1_score(y,z,zero_division=0)),
 "auc":float(roc_auc_score(y,p)),"confusion_matrix_labels":["non_swallow","swallow"],
 "confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]]}

def main():
 dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 root=Path(".")
 with np.load("data/interim/guard100_balanced/raw_windows.npz",allow_pickle=False) as d:X=d["X"].astype(np.float32)
 with np.load("data/interim/guard100_balanced/swallow_binary_windows.npz",allow_pickle=False) as d:y=d["y"].astype(np.int64)
 split=json.loads(Path("data/interim/guard100_balanced/participant_split.json").read_text())
 tr=np.asarray(split["splits"]["train"]["indices"]);va=np.asarray(split["splits"]["validation"]["indices"]);te=np.asarray(split["splits"]["test"]["indices"])
 print("Training final components using train split only; validation selects threshold; test used once at end.",flush=True)
 # Load trained deep models; each scaler was computed on training participants only.
 c256=torch.load("artifacts/checkpoints/guard100_balanced_spectrogram_best.pt",map_location=dev,weights_only=True)
 c512=torch.load("artifacts/checkpoints/guard100_balanced_spec512_best.pt",map_location=dev,weights_only=True)
 c1d=torch.load("artifacts/checkpoints/guard100_balanced_1dcnn_best.pt",map_location=dev,weights_only=True)
 X256=np.clip(X/np.asarray(c256["scale"],np.float32)[None,:,None],-8,8)
 X512=np.clip(X/np.asarray(c512["scale"],np.float32)[None,:,None],-8,8)
 X1d=np.clip(X/np.asarray(c1d["scale"],np.float32)[None,:,None],-8,8)
 def dl(arr,idx,bs):
  return DataLoader(TensorDataset(torch.from_numpy(arr[idx]),torch.from_numpy(y[idx].astype(np.float32))),batch_size=bs,shuffle=False)
 m256=Spec256().to(dev);m256.load_state_dict(c256["model_state_dict"])
 m512=Spec512().to(dev);m512.load_state_dict(c512["model_state_dict"])
 m1d=Raw1D().to(dev);m1d.load_state_dict(c1d["model_state_dict"])
 _,v256=pred256(m256,dl(X256,va,64),dev);_,t256=pred256(m256,dl(X256,te,64),dev)
 _,v512=pred512(m512,dl(X512,va,64),dev);_,t512=pred512(m512,dl(X512,te,64),dev)
 _,v1d=pred1d(m1d,dl(X1d,va,96),dev);_,t1d=pred1d(m1d,dl(X1d,te,96),dev)
 # Fit classical components only on training participants.
 F=extract(X256)
 rf=RandomForestClassifier(n_estimators=700,min_samples_leaf=2,max_features="sqrt",class_weight="balanced_subsample",n_jobs=-1,random_state=42)
 svm=make_pipeline(SimpleImputer(),StandardScaler(),SVC(C=2,kernel="rbf",class_weight="balanced",probability=True,random_state=42))
 rf.fit(F[tr],y[tr]);svm.fit(F[tr],y[tr])
 vrf=rf.predict_proba(F[va])[:,1];trf=rf.predict_proba(F[te])[:,1]
 vsvm=svm.predict_proba(F[va])[:,1];tsvm=svm.predict_proba(F[te])[:,1]
 joblib.dump(rf,"artifacts/checkpoints/guard100_balanced_random_forest.joblib")
 joblib.dump(svm,"artifacts/checkpoints/guard100_balanced_svm.joblib")
 # Fixed weights and threshold selected before test evaluation.
 vp=.4*v256+.2*v512+.2*vsvm+.1*vrf+.1*v1d
 tp=.4*t256+.2*t512+.2*tsvm+.1*trf+.1*t1d
 vm=metrics(y[va],vp,.5)
 print("VALIDATION",json.dumps(vm),flush=True)
 # First and only use of the new held-out test labels.
 tm=metrics(y[te],tp,.5)
 majority=max(float(np.mean(y[te]==0)),float(np.mean(y[te]==1)))
 report={"note":"Final participant-disjoint held-out test evaluated once. Original signals.zip and labels.zip only. No synthetic data. Five-model ensemble weights and threshold were selected on validation before test evaluation.",
 "dataset":{"window_count":int(len(y)),"train_windows":int(len(tr)),"validation_windows":int(len(va)),"test_windows":int(len(te)),"train_participants":len(split["splits"]["train"]["participants"]),"validation_participants":len(split["splits"]["validation"]["participants"]),"test_participants":len(split["splits"]["test"]["participants"]),"class_counts":{"non_swallow":int((y==0).sum()),"swallow":int((y==1).sum())}},
 "ensemble":{"weights":{"spectrogram_cnn_256":.4,"spectrogram_cnn_512":.2,"svm_rbf":.2,"random_forest":.1,"time_domain_1dcnn":.1},"threshold":.5},
 "validation_metrics":vm,"test_metrics":tm,"test_majority_class_baseline_accuracy":majority,
 "components":{"spectrogram_cnn_256_auc_validation":float(roc_auc_score(y[va],v256)),"spectrogram_cnn_512_auc_validation":float(roc_auc_score(y[va],v512)),"svm_auc_validation":float(roc_auc_score(y[va],vsvm)),"random_forest_auc_validation":float(roc_auc_score(y[va],vrf)),"time_domain_1dcnn_auc_validation":float(roc_auc_score(y[va],v1d))}}
 Path("artifacts/reports/final_guard100_balanced_test_metrics.json").write_text(json.dumps(report,indent=2))
 np.savez_compressed("artifacts/reports/final_guard100_balanced_predictions.npz",y_test=y[te],p_test=tp,y_validation=y[va],p_validation=vp)
 print("FINAL TEST",json.dumps(tm),flush=True)
 print("Saved artifacts/reports/final_guard100_balanced_test_metrics.json",flush=True)
if __name__=="__main__":main()
