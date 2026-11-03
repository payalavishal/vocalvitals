"""Five-fold participant-group CV on development participants only; excludes prior test participants."""
import json,copy,random
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score,accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
from scipy.stats import skew,kurtosis

def seed(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(s);torch.backends.cudnn.benchmark=True
class SpecNet(nn.Module):
 def __init__(self):
  super().__init__()
  def b(a,z):return nn.Sequential(nn.Conv2d(a,z,3,padding=1,bias=False),nn.BatchNorm2d(z),nn.GELU(),nn.Conv2d(z,z,3,padding=1,bias=False),nn.BatchNorm2d(z),nn.GELU(),nn.MaxPool2d(2))
  self.f=nn.Sequential(b(4,24),b(24,48),b(48,96),nn.AdaptiveAvgPool2d((1,1)))
  self.h=nn.Sequential(nn.Flatten(),nn.Dropout(.35),nn.Linear(96,48),nn.GELU(),nn.Dropout(.25),nn.Linear(48,1))
 def forward(self,x):return self.h(self.f(x)).squeeze(1)
class RawNet(nn.Module):
 def __init__(self):
  super().__init__()
  self.f=nn.Sequential(nn.Conv1d(4,32,31,padding=15,bias=False),nn.BatchNorm1d(32),nn.GELU(),nn.MaxPool1d(2),
   nn.Conv1d(32,64,15,padding=7,bias=False),nn.BatchNorm1d(64),nn.GELU(),nn.MaxPool1d(2),
   nn.Conv1d(64,128,9,padding=4,bias=False),nn.BatchNorm1d(128),nn.GELU(),nn.MaxPool1d(2),
   nn.Conv1d(128,128,5,padding=2,bias=False),nn.BatchNorm1d(128),nn.GELU(),nn.AdaptiveAvgPool1d(1))
  self.h=nn.Sequential(nn.Flatten(),nn.Dropout(.35),nn.Linear(128,64),nn.GELU(),nn.Dropout(.25),nn.Linear(64,1))
 def forward(self,x):return self.h(self.f(x)).squeeze(1)
def stft(x,dev):
 b,c,n=x.shape;z=x.reshape(b*c,n)
 s=torch.stft(z,n_fft=256,hop_length=64,win_length=256,window=torch.hann_window(256,device=dev),center=True,return_complex=True)
 return torch.log1p(torch.abs(s)).reshape(b,c,s.shape[-2],s.shape[-1])
@torch.no_grad()
def predict(model,loader,dev,kind):
 model.eval();P=[];Y=[]
 for x,y in loader:
  x=x.to(dev)
  if kind=="spec":x=stft(x,dev)
  P.append(torch.sigmoid(model(x)).cpu().numpy());Y.append(y.numpy())
 return np.concatenate(Y),np.concatenate(P)
def train_net(model,kind,X,y,tr,va,dev,seedval,maxep=24,patience=6):
 seed(seedval)
 tl=DataLoader(TensorDataset(torch.from_numpy(X[tr]),torch.from_numpy(y[tr].astype(np.float32))),batch_size=48 if kind=="spec" else 64,shuffle=True)
 vl=DataLoader(TensorDataset(torch.from_numpy(X[va]),torch.from_numpy(y[va].astype(np.float32))),batch_size=64 if kind=="spec" else 96,shuffle=False)
 model=model.to(dev);pos=int((y[tr]==1).sum());neg=int((y[tr]==0).sum())
 lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/pos,device=dev))
 opt=torch.optim.AdamW(model.parameters(),lr=7e-4 if kind=="spec" else 8e-4,weight_decay=2e-4)
 best=-1;stale=0;best_state=None;bestep=0
 for ep in range(1,maxep+1):
  model.train()
  for xb,yb in tl:
   xb=xb.to(dev);yb=yb.to(dev)
   if kind=="spec":xb=stft(xb,dev)
   opt.zero_grad(set_to_none=True);loss=lossfn(model(xb),yb);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
  yy,pp=predict(model,vl,dev,kind);auc=float(roc_auc_score(yy,pp))
  if auc>best+1e-4:
   best=auc;bestep=ep;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  else:
   stale+=1
   if stale>=patience:break
 model.load_state_dict(best_state)
 yy,pp=predict(model,vl,dev,kind)
 return pp,best,bestep
def features(X):
 out=[];freq=np.fft.rfftfreq(X.shape[-1],1/16000);bands=[(0,250),(250,500),(500,1000),(1000,2000),(2000,4000),(4000,8000)]
 for w in X:
  row=[]
  for z in w:
   z=z.astype(np.float64);d=np.diff(z);sp=np.abs(np.fft.rfft(z-z.mean()))**2;tot=sp.sum()+1e-12;ps=sp/tot
   row += [z.mean(),z.std(),np.sqrt(np.mean(z*z)),np.max(np.abs(z)),np.mean(np.abs(z)),np.median(np.abs(z)),np.percentile(np.abs(z),90),np.percentile(np.abs(z),99),np.sqrt(np.mean(d*d)),np.mean(np.abs(d)),np.mean(np.diff(np.signbit(z))),skew(z),kurtosis(z),np.sum(z*z),np.sum(freq*sp)/tot,-np.sum(ps*np.log(ps+1e-12)),np.exp(np.mean(np.log(sp+1e-12)))/(np.mean(sp)+1e-12),np.argmax(sp)*16000/X.shape[-1]]
   for lo,hi in bands:
    m=(freq>=lo)&(freq<hi);row.append(float(sp[m].sum()/tot) if m.any() else 0.)
  out.append(row)
 return np.asarray(out,np.float32)
def metric(y,p,t):
 z=(p>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
 return {"threshold":float(t),"accuracy":float(accuracy_score(y,z)),"balanced_accuracy":float(balanced_accuracy_score(y,z)),"f1":float(f1_score(y,z,zero_division=0)),"specificity":float(tn/(tn+fp)),"recall":float(tp/(tp+fn)),"cm":[[int(tn),int(fp)],[int(fn),int(tp)]]}
def main():
 dev=torch.device("cuda" if torch.cuda.is_available() else "cpu");seed(2026)
 with np.load("data/interim/guard100_balanced/raw_windows.npz",allow_pickle=False) as d:X=d["X"].astype(np.float32)
 with np.load("data/interim/guard100_balanced/swallow_binary_windows.npz",allow_pickle=False) as d:y=d["y"].astype(int);groups=d["participant_id"].astype(str)
 split=json.loads(Path("data/interim/guard100_balanced/participant_split.json").read_text())
 dev_idx=np.concatenate([split["splits"]["train"]["indices"],split["splits"]["validation"]["indices"]]).astype(int)
 excluded=set(split["splits"]["test"]["participants"])
 assert not (set(groups[dev_idx]) & excluded)
 F=features(X)
 cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=2026)
 oof={k:np.full(len(y),np.nan,dtype=np.float32) for k in ["spec","raw1d","svm","blend"]}
 folds=[];print("5-fold group CV on",len(dev_idx),"development windows; prior test participants excluded.",flush=True)
 for fi,(a,b) in enumerate(cv.split(np.zeros(len(dev_idx)),y[dev_idx],groups[dev_idx]),1):
  tr=dev_idx[a];va=dev_idx[b]
  # fold-specific train-only channel scaling
  scale=np.quantile(np.abs(X[tr]),.995,axis=(0,2)).astype(np.float32);Xs=np.clip(X/scale[None,:,None],-8,8)
  seed(1000+fi); ps,ascore,es=train_net(SpecNet(),"spec",Xs,y,tr,va,dev,1000+fi)
  seed(2000+fi); p1,a1,e1=train_net(RawNet(),"raw",Xs,y,tr,va,dev,2000+fi)
  svm=make_pipeline(SimpleImputer(),StandardScaler(),SVC(C=2,kernel="rbf",class_weight="balanced",probability=True,random_state=42+fi))
  svm.fit(F[tr],y[tr]);pv=svm.predict_proba(F[va])[:,1]
  blend=(ps+p1+pv)/3
  oof["spec"][va]=ps;oof["raw1d"][va]=p1;oof["svm"][va]=pv;oof["blend"][va]=blend
  fm={"fold":fi,"train_windows":len(tr),"validation_windows":len(va),"train_participants":len(set(groups[tr])),"validation_participants":len(set(groups[va])),"spec_auc":ascore,"spec_best_epoch":es,"raw1d_auc":a1,"raw1d_best_epoch":e1,"svm_auc":float(roc_auc_score(y[va],pv)),"blend_auc":float(roc_auc_score(y[va],blend))}
  folds.append(fm);print("FOLD",json.dumps(fm),flush=True)
 valid=dev_idx
 results={}
 for name in oof:
  p=oof[name][valid];yy=y[valid]
  sw=[metric(yy,p,t) for t in np.arange(.25,.751,.025)]
  results[name]={"auc":float(roc_auc_score(yy,p)),"best_accuracy":max(sw,key=lambda q:q["accuracy"]),"best_balanced_accuracy":max(sw,key=lambda q:q["balanced_accuracy"]),"at_0_5":metric(yy,p,.5)}
 report={"note":"Out-of-fold participant-group evaluation on development participants only. Previously held-out test participants were excluded and not accessed.","folds":folds,"oof_metrics":results}
 Path("artifacts/reports/guard100_balanced_dev_group_cv.json").write_text(json.dumps(report,indent=2))
 np.savez_compressed("artifacts/reports/guard100_balanced_dev_oof.npz",indices=valid,y=y[valid],**{k:v[valid] for k,v in oof.items()})
 print("OOF",json.dumps(results),flush=True)
if __name__=="__main__":main()

