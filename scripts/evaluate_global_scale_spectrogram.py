import json
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import roc_auc_score,confusion_matrix,balanced_accuracy_score,f1_score
from experiment_global_scale_spectrogram import Net,spec,pred,metric
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
with np.load("data/interim/swallow_windows_raw_scale_audit.npz",allow_pickle=False) as d:X=d["X"].astype(np.float32)
with np.load("data/processed/swallow_binary_windows.npz",allow_pickle=False) as d:y=d["y"].astype(np.int64)
s=json.loads(Path("artifacts/reports/participant_split.json").read_text());va=np.asarray(s["splits"]["validation"]["indices"])
ck=torch.load("artifacts/checkpoints/global_scale_spectrogram_best.pt",map_location=dev,weights_only=False)
scale=np.asarray(ck["scale"],dtype=np.float32);X=np.clip(X/scale[None,:,None],-8,8)
loader=DataLoader(TensorDataset(torch.from_numpy(X[va]),torch.from_numpy(y[va].astype(np.float32))),batch_size=64,shuffle=False)
m=Net().to(dev);m.load_state_dict(ck["model_state_dict"]);yy,pp=pred(m,loader,dev)
sw=[metric(yy,pp,t) for t in np.arange(.1,.91,.025)];best=max(sw,key=lambda z:z["balanced_accuracy"])
report={"note":"Validation-only, participant-disjoint, real data only; test untouched.","auc":float(roc_auc_score(yy,pp)),"checkpoint_epoch":int(ck["epoch"]),"best_validation_balanced_accuracy":best,"threshold_sweep":sw}
Path("artifacts/reports/global_scale_spectrogram_experiment.json").write_text(json.dumps(report,indent=2))
print(json.dumps({"auc":report["auc"],"epoch":report["checkpoint_epoch"],"best":best},indent=2))
