"""Time-domain 1D CNN on corrected real guard100_balanced windows."""
import json,random
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import roc_auc_score,confusion_matrix,balanced_accuracy_score,accuracy_score,f1_score
def seed():
 random.seed(789);np.random.seed(789);torch.manual_seed(789)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(789);torch.backends.cudnn.benchmark=True
class Net(nn.Module):
 def __init__(self):
  super().__init__()
  self.f=nn.Sequential(
   nn.Conv1d(4,32,31,padding=15,bias=False),nn.BatchNorm1d(32),nn.GELU(),nn.MaxPool1d(2),
   nn.Conv1d(32,64,15,padding=7,bias=False),nn.BatchNorm1d(64),nn.GELU(),nn.MaxPool1d(2),
   nn.Conv1d(64,128,9,padding=4,bias=False),nn.BatchNorm1d(128),nn.GELU(),nn.MaxPool1d(2),
   nn.Conv1d(128,128,5,padding=2,bias=False),nn.BatchNorm1d(128),nn.GELU(),nn.AdaptiveAvgPool1d(1))
  self.h=nn.Sequential(nn.Flatten(),nn.Dropout(.35),nn.Linear(128,64),nn.GELU(),nn.Dropout(.25),nn.Linear(64,1))
 def forward(self,x):return self.h(self.f(x)).squeeze(1)
@torch.no_grad()
def predict(m,loader,dev):
 m.eval();P=[];Y=[]
 for x,y in loader:P.append(torch.sigmoid(m(x.to(dev))).cpu().numpy());Y.append(y.numpy())
 return np.concatenate(Y),np.concatenate(P)
def score(y,p,t):
 z=(p>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
 return {"threshold":float(t),"accuracy":float(accuracy_score(y,z)),"balanced_accuracy":float(balanced_accuracy_score(y,z)),"f1":float(f1_score(y,z,zero_division=0)),"specificity":float(tn/(tn+fp)),"recall":float(tp/(tp+fn)),"cm":[[int(tn),int(fp)],[int(fn),int(tp)]]}
def main():
 seed();dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 with np.load("data/interim/guard100_balanced/raw_windows.npz",allow_pickle=False) as d:X=d["X"].astype(np.float32)
 with np.load("data/interim/guard100_balanced/swallow_binary_windows.npz",allow_pickle=False) as d:y=d["y"].astype(np.int64)
 s=json.loads(Path("data/interim/guard100_balanced/participant_split.json").read_text());tr=np.asarray(s["splits"]["train"]["indices"]);va=np.asarray(s["splits"]["validation"]["indices"])
 scale=np.quantile(np.abs(X[tr]),.995,axis=(0,2)).astype(np.float32);X=np.clip(X/scale[None,:,None],-8,8)
 tl=DataLoader(TensorDataset(torch.from_numpy(X[tr]),torch.from_numpy(y[tr].astype(np.float32))),batch_size=64,shuffle=True)
 vl=DataLoader(TensorDataset(torch.from_numpy(X[va]),torch.from_numpy(y[va].astype(np.float32))),batch_size=96,shuffle=False)
 m=Net().to(dev);pos=int((y[tr]==1).sum());neg=int((y[tr]==0).sum());lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/pos,device=dev))
 opt=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=3e-4);sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=.5,patience=4,min_lr=1e-5)
 best=-1;stale=0;hist=[]
 for ep in range(1,61):
  m.train();ls=[]
  for xb,yb in tl:
   xb=xb.to(dev);yb=yb.to(dev);opt.zero_grad(set_to_none=True);z=m(xb);l=lossfn(z,yb);l.backward();nn.utils.clip_grad_norm_(m.parameters(),2.);opt.step();ls.append(float(l.item()))
  yy,pp=predict(m,vl,dev);auc=float(roc_auc_score(yy,pp));sch.step(auc);hist.append({"epoch":ep,"loss":float(np.mean(ls)),"validation_auc":auc})
  print(f"epoch={ep:02d} loss={np.mean(ls):.4f} val_auc={auc:.4f}",flush=True)
  if auc>best+1e-4:
   best=auc;bestep=ep;stale=0
   torch.save({"model_state_dict":m.state_dict(),"scale":scale.tolist(),"epoch":ep,"validation_auc":auc,"test_used":False},"artifacts/checkpoints/guard100_balanced_1dcnn_best.pt")
  else:
   stale+=1
   if stale>=11:print("early stop",flush=True);break
 ck=torch.load("artifacts/checkpoints/guard100_balanced_1dcnn_best.pt",map_location=dev,weights_only=True);m.load_state_dict(ck["model_state_dict"]);yy,pp=predict(m,vl,dev)
 sw=[score(yy,pp,t) for t in np.arange(.1,.91,.025)];bestm=max(sw,key=lambda q:q["balanced_accuracy"])
 np.savez_compressed("artifacts/reports/guard100_balanced_1dcnn_validation_probabilities.npz",y=yy,p=pp)
 rep={"auc":float(roc_auc_score(yy,pp)),"best_epoch":bestep,"best_validation_balanced_accuracy":bestm,"threshold_sweep":sw,"history":hist,"note":"Corrected guard100_balanced real dataset; participant-disjoint; test untouched."}
 Path("artifacts/reports/guard100_balanced_1dcnn_experiment.json").write_text(json.dumps(rep,indent=2))
 print("BEST",json.dumps({"auc":rep["auc"],"epoch":bestep,"metric":bestm}),flush=True)
if __name__=="__main__":main()

