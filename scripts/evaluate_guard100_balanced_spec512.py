import json
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset
from experiment_guard100_balanced_spec512 import Net,pred
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
with np.load("data/interim/guard100_balanced/raw_windows.npz",allow_pickle=False) as d:X=d["X"].astype(np.float32)
with np.load("data/interim/guard100_balanced/swallow_binary_windows.npz",allow_pickle=False) as d:y=d["y"].astype(int)
s=json.loads(Path("data/interim/guard100_balanced/participant_split.json").read_text());va=np.asarray(s["splits"]["validation"]["indices"])
ck=torch.load("artifacts/checkpoints/guard100_balanced_spec512_best.pt",map_location=dev,weights_only=True)
X=np.clip(X/np.asarray(ck["scale"],np.float32)[None,:,None],-8,8)
loader=DataLoader(TensorDataset(torch.from_numpy(X[va]),torch.from_numpy(y[va].astype(np.float32))),batch_size=64,shuffle=False)
m=Net().to(dev);m.load_state_dict(ck["model_state_dict"]);yy,p=pred(m,loader,dev)
a=np.load("artifacts/reports/guard100_balanced_validation_probabilities.npz")
assert np.array_equal(yy,a["y"])
np.savez_compressed("artifacts/reports/guard100_balanced_validation_probabilities.npz",y=yy,cnn=a["spectrogram_cnn"],raw1d=a["raw_1dcnn"],rf=a["random_forest"],svm=a["svm_rbf"],spec512=p)
print("spec512 validation AUC",__import__("sklearn.metrics",fromlist=["roc_auc_score"]).roc_auc_score(yy,p))
