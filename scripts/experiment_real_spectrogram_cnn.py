"""Four-channel log-spectrogram CNN; real-data train/validation only."""
from __future__ import annotations
import json, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score, f1_score

def seed_all(seed=123):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True

class SpecCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def block(a,b):
            return nn.Sequential(
                nn.Conv2d(a,b,3,padding=1,bias=False),
                nn.BatchNorm2d(b),nn.GELU(),
                nn.Conv2d(b,b,3,padding=1,bias=False),
                nn.BatchNorm2d(b),nn.GELU(),
                nn.MaxPool2d(2))
        self.features=nn.Sequential(block(4,24),block(24,48),block(48,96),
             nn.AdaptiveAvgPool2d((1,1)))
        self.head=nn.Sequential(nn.Flatten(),nn.Dropout(.35),nn.Linear(96,48),
             nn.GELU(),nn.Dropout(.25),nn.Linear(48,1))
    def forward(self,x): return self.head(self.features(x)).squeeze(1)

def make_spec(x,device):
    # x: (B, 4, 3200). Hann-window STFT preserves channel identity.
    b,c,n=x.shape
    z=x.reshape(b*c,n)
    s=torch.stft(z,n_fft=256,hop_length=64,win_length=256,
        window=torch.hann_window(256,device=device),center=True,
        return_complex=True)
    s=torch.log1p(torch.abs(s)).reshape(b,c,s.shape[-2],s.shape[-1])
    return s

def score(y,p,t):
    pred=(p>=t).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"threshold":float(t),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
      "accuracy":float((pred==y).mean()),"f1":float(f1_score(y,pred,zero_division=0)),
      "specificity":float(tn/(tn+fp)) if tn+fp else 0.,
      "recall":float(tp/(tp+fn)) if tp+fn else 0.,
      "confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]]}

@torch.no_grad()
def predict(model,loader,device):
    model.eval(); ps=[]; ys=[]
    for xb,yb in loader:
        xb=make_spec(xb.to(device),device)
        ps.append(torch.sigmoid(model(xb)).cpu().numpy()); ys.append(yb.numpy())
    return np.concatenate(ys),np.concatenate(ps)

def main():
    seed_all()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load("data/processed/swallow_binary_windows.npz",allow_pickle=False) as d:
        X=d["X"].astype(np.float32); y=d["y"].astype(np.int64)
    split=json.loads(Path("artifacts/reports/participant_split.json").read_text())
    tr=np.asarray(split["splits"]["train"]["indices"]); va=np.asarray(split["splits"]["validation"]["indices"])
    # No test indices are read.
    train=TensorDataset(torch.from_numpy(X[tr]),torch.from_numpy(y[tr].astype(np.float32)))
    valid=TensorDataset(torch.from_numpy(X[va]),torch.from_numpy(y[va].astype(np.float32)))
    train_loader=DataLoader(train,batch_size=48,shuffle=True,num_workers=0,pin_memory=device.type=="cuda")
    val_loader=DataLoader(valid,batch_size=64,shuffle=False,num_workers=0)
    model=SpecCNN().to(device)
    npos=int((y[tr]==1).sum()); nneg=int((y[tr]==0).sum())
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(nneg/npos,device=device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=2e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=.5,patience=3,min_lr=1e-5)
    best_auc=-1.; best_epoch=0; patience=9; stale=0; history=[]
    print("Device",device,"train",len(tr),"validation",len(va),"real data only",flush=True)
    for epoch in range(1,41):
        model.train(); losses=[]
        for xb,yb in train_loader:
            xb=make_spec(xb.to(device),device); yb=yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(xb); loss=criterion(logits,yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2.0); optimizer.step()
            losses.append(float(loss.item()))
        vy,vp=predict(model,val_loader,device)
        auc=float(roc_auc_score(vy,vp)); scheduler.step(auc)
        row={"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_auc":auc,
             "learning_rate":optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(f"epoch={epoch:02d} loss={row['train_loss']:.4f} val_auc={auc:.4f} lr={row['learning_rate']:.6g}",flush=True)
        if auc>best_auc+1e-4:
            best_auc=auc; best_epoch=epoch; stale=0
            torch.save({"model_state_dict":model.state_dict(),"epoch":epoch,
                "validation_auc":auc,"architecture":"4-channel log-STFT SpecCNN",
                "source":"original signals.zip + labels.zip","test_used":False},
                "artifacts/checkpoints/real_spectrogram_best.pt")
        else:
            stale+=1
            if stale>=patience:
                print("Early stopping",flush=True); break
    # Reload best validation-AUC checkpoint and report threshold sweep.
    ck=torch.load("artifacts/checkpoints/real_spectrogram_best.pt",map_location=device,weights_only=True)
    model.load_state_dict(ck["model_state_dict"])
    vy,vp=predict(model,val_loader,device)
    sweep=[score(vy,vp,t) for t in np.arange(.1,.91,.025)]
    best=max(sweep,key=lambda r:r["balanced_accuracy"])
    report={"note":"Validation-only, participant-disjoint; real data only; held-out test untouched.",
      "validation_auc":float(roc_auc_score(vy,vp)),"best_epoch":best_epoch,
      "best_validation_balanced_accuracy":best,"threshold_sweep":sweep,"history":history}
    Path("artifacts/reports/real_spectrogram_experiment.json").write_text(json.dumps(report,indent=2))
    print("BEST",json.dumps({"auc":report["validation_auc"],"best":best,"epoch":best_epoch}),flush=True)
    print("Saved artifacts/reports/real_spectrogram_experiment.json",flush=True)
if __name__=="__main__": main()
