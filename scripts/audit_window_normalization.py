"""Recreate the exact saved windows from original recordings without per-window peak normalization."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.signal import resample_poly

ROOT=Path(".")
meta=json.loads((ROOT/"data/processed/swallow_binary_metadata.json").read_text())
records=meta["records"]
signals={p.stem.lower():p for p in (ROOT/"data/raw/signals").glob("*.npy")}
by_record=defaultdict(list)
for i,r in enumerate(records): by_record[r["recording_id"].lower()].append((i,r))
X=np.zeros((len(records),4,3200),np.float32)
for k,(stem,items) in enumerate(by_record.items()):
    p=signals.get(stem)
    if p is None: raise FileNotFoundError(stem)
    raw=np.load(p,mmap_mode="r",allow_pickle=False)
    sig=resample_poly(np.asarray(raw,dtype=np.float32),160,441,axis=0)
    for idx,r in items:
        center=(float(r["start_ms"])+float(r["end_ms"]))/2
        c=int(round(center*16))
        start=c-1600; end=start+3200
        a=max(0,start); b=min(len(sig),end)
        if b>a: X[idx,:,a-start:b-start]=sig[a:b].T
    if (k+1)%200==0: print("recordings",k+1,"/",len(by_record),flush=True)
print("shape",X.shape,"finite",bool(np.isfinite(X).all()))
with np.load(ROOT/"data/processed/swallow_binary_windows.npz",allow_pickle=False) as d:
    Xnorm=d["X"]
print("saved peak-normalized shape",Xnorm.shape)
for c in range(4):
    v=X[:,c,:]
    print("channel",c,"raw quantiles",np.quantile(np.abs(v),[.5,.9,.95,.99,.995,.999,1]).tolist(),
          "std",float(v.std()),"mean",float(v.mean()),
          "window_peak_median",float(np.median(np.max(np.abs(v),axis=1))))
# Save exact-index aligned raw windows for train-only global scaling experiments.
np.savez_compressed(ROOT/"data/interim/swallow_windows_raw_scale_audit.npz",X=X)
print("saved data/interim/swallow_windows_raw_scale_audit.npz")
