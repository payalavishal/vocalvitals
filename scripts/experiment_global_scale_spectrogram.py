"""Spectrogram CNN with train-only global channel scaling; real data only."""
from __future__ import annotations
import json,random
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import roc_auc_score,confusion_matrix,balanced_accuracy_score,f1_score

def seed_all():
 random.seed(321);np.random.seed(321);torch.manual_seed(321)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(321);torch.backends.cudnn.benchmark=True
class Net(nn.Module):
 def __init__(self):
  super().__init__()
  def b(a,z):return nn.Sequential(nn.Conv2d(a,z,3,padding=1,bias=False),nn.BatchNorm2d(z),nn.GELU(),nn.Conv2d(z,z,3,padding=1,bias=False),nn.BatchNorm2d(z),nn.GELU(),nn.MaxPool2d(2))
  self.f=nn.Sequential(b(4,24),b(24,48),b(48,96),nn.AdaptiveAvgPool2d((1,1)))
  self.h=nn.Sequential(nn.Flatten(),nn.Dropout(.35),nn.Linear(96,48),nn.GELU(),nn.Dropout(.25),nn.Linear(48,1))
 def forward(self,x):return self.h(self.f(x)).squeeze(1)
def spec(x,dev):
 b,c,n=x.shape;z=x.reshape(b*c,n)
 s=torch.stft(z,n_fft=256,hop_length=64,win_length=256,window=torch.hann_window(256,device=dev),center=True,return_complex=True)
 return torch.log1p(torch.abs(s)).reshape(b,c,s.shape[-2],s.shape[-1])
def pred(m,loader,dev):
 m.eval();P=[];Y=[]
 with torch.no_grad():
  for x,y in loader:P.append(torch.sigmoid(m(spec(x.to(dev),dev))).cpu().numpy());Y.append(y.numpy())
 return np.concatenate(Y),np.concatenate(P)
def metric(y,p,t):
 z=(p>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
 return {"threshold":float(t),"accuracy":float((z==y).mean()),"balanced_accuracy":float(balanced_accuracy_score(y,z)),"f1":float(f1_score(y,z,zero_division=0)),"specificity":float(tn/(tn+fp)),"recall":float(tp/(tp+fn)),"confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]]}
def main():
 seed_all();dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 with np.load("data/interim/swallow_windows_raw_scale_audit.npz",allow_pickle=False) as d:X=d["X"].astype(np.float32)
 with np.load("data/processed/swallow_binary_windows.npz",allow_pickle=False) as d:y=d["y"].astype(np.int64)
 s=json.loads(Path("artifacts/reports/participant_split.json").read_text());tr=np.asarray(s["splits"]["train"]["indices"]);va=np.asarray(s["splits"]["validation"]["indices"])
 # Scaling statistics come only from training participants.
 scale=np.quantile(np.abs(X[tr]),.995,axis=(0,2)).astype(np.float32)
 X=np.clip(X/scale[None,:,None],-8,8)
 print("TRAIN-ONLY p99.5 channel scales",scale.tolist(),"scaled quantiles",np.quantile(np.abs(X[tr]),[.5,.9,.99,.999]).tolist(),flush=True)
 tl=DataLoader(TensorDataset(torch.from_numpy(X[tr]),torch.from_numpy(y[tr].astype(np.float32))),batch_size=48,shuffle=True)
 vl=DataLoader(TensorDataset(torch.from_numpy(X[va]),torch.from_numpy(y[va].astype(np.float32))),batch_size=64,shuffle=False)
 m=Net().to(dev);pos=int((y[tr]==1).sum());neg=int((y[tr]==0).sum())
 lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/pos,device=dev))
 opt=torch.optim.AdamW(m.parameters(),lr=7e-4,weight_decay=2e-4)
 sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=.5,patience=3,min_lr=1e-5)
 best=-1;stale=0;hist=[]
 for ep in range(1,41):
  m.train();ls=[]
  for xb,yb in tl:
   xb=spec(xb.to(dev),dev);yb=yb.to(dev);opt.zero_grad(set_to_none=True);z=m(xb);l=lossfn(z,yb);l.backward();nn.utils.clip_grad_norm_(m.parameters(),2.);opt.step();ls.append(float(l.item()))
  yy,pp=pred(m,vl,dev);auc=float(roc_auc_score(yy,pp));sch.step(auc)
  hist.append({"epoch":ep,"loss":float(np.mean(ls)),"validation_auc":auc})
  print(f"epoch={ep:02d} loss={np.mean(ls):.4f} val_auc={auc:.4f}",flush=True)
  if auc>best+1e-4:
   best=auc;stale=0;bestep=ep
   torch.save({"model_state_dict":m.state_dict(),"scale":scale,"epoch":ep,"validation_auc":auc,"test_used":False},"artifacts/checkpoints/global_scale_spectrogram_best.pt")
  else:
   stale+=1
   if stale>=9:print("early stop",flush=True);break
 ck=torch.load("artifacts/checkpoints/global_scale_spectrogram_best.pt",map_location=dev,weights_only=True);m.load_state_dict(ck["model_state_dict"])
 yy,pp=pred(m,vl,dev);sw=[metric(yy,pp,t) for t in np.arange(.1,.91,.025)];bestm=max(sw,key=lambda q:q["balanced_accuracy"])
 report={"note":"Original real signals and labels; train-only channel scales; participant-disjoint validation; test untouched.","channel_scales":scale.tolist(),"auc":float(roc_auc_score(yy,pp)),"best_epoch":bestep,"best_validation_balanced_accuracy":bestm,"threshold_sweep":sw,"history":hist}
 Path("artifacts/reports/global_scale_spectrogram_experiment.json").write_text(json.dumps(report,indent=2))
 print("BEST",json.dumps({"auc":report["auc"],"epoch":bestep,"metric":bestm}),flush=True)
if __name__=="__main__":main()
