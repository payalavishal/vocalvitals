import csv
from pathlib import Path
import numpy as np
dur=[];files=0
for p in Path("data/raw/labels").glob("*.csv"):
 files+=1
 with p.open(newline="",encoding="utf-8-sig") as f:
  for r in csv.DictReader(f):
   try:
    d=float(r["duration"]);a=float(r["start"]);b=float(r["end"])
    if d>0 and b>a:dur.append(d)
   except:pass
x=np.asarray(dur)
print("files",files,"events",len(x))
print("duration_ms percentiles",dict(zip(["min","p10","p25","median","p75","p90","p95","max"],np.percentile(x,[0,10,25,50,75,90,95,100]).tolist())))
print("mean_ms",float(x.mean()))
print("under 200ms",int((x<=200).sum()),"200-500",int(((x>200)&(x<=500)).sum()),"500-1000",int(((x>500)&(x<=1000)).sum()),"over 1000",int((x>1000).sum()))
