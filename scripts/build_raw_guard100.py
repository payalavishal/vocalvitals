import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.signal import resample_poly
meta=json.loads(Path("data/interim/guard100/swallow_binary_metadata.json").read_text());records=meta["records"]
signals={p.stem.lower():p for p in Path("data/raw/signals").glob("*.npy")}
by=defaultdict(list)
for i,r in enumerate(records):by[r["recording_id"].lower()].append((i,r))
X=np.zeros((len(records),4,3200),np.float32)
for k,(stem,items) in enumerate(by.items()):
 raw=np.load(signals[stem],mmap_mode="r",allow_pickle=False)
 sig=resample_poly(np.asarray(raw,dtype=np.float32),160,441,axis=0)
 for idx,r in items:
  center=(float(r["start_ms"])+float(r["end_ms"]))/2;c=int(round(center*16));a=max(0,c-1600);b=min(len(sig),c+1600)
  if b>a:X[idx,:,a-(c-1600):a-(c-1600)+(b-a)]=sig[a:b].T
 if (k+1)%250==0:print(k+1,"/",len(by),flush=True)
out=Path("data/interim/guard100/raw_windows.npz");np.savez_compressed(out,X=X)
print("saved",out,"shape",X.shape)
