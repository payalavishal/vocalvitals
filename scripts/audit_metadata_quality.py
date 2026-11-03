import json,collections,importlib.util
from pathlib import Path
import numpy as np
m=json.loads(Path("data/processed/swallow_binary_metadata.json").read_text());r=m["records"]
by=collections.defaultdict(list);pids=collections.defaultdict(set);counts=collections.Counter()
for x in r:
 by[x["recording_id"]].append(x);pids[x["recording_id"]].add(x["participant_id"]);counts[x["label"]]+=1
overlap_pairs=0;max_overlap=0;overlap_windows=0
for stem,rows in by.items():
 pos=sorted([(x["start_ms"],x["end_ms"]) for x in rows if x["label"]==1])
 for i,(a,b) in enumerate(pos):
  for c,d in pos[i+1:]:
   if c>=b:break
   if min(b,d)>max(a,c):overlap_pairs+=1;overlap_windows+=1
 print("",end="")
print("labels",dict(counts),"recordings_with_windows",len(by))
print("recordings_with_multiple_participant_ids",sum(len(v)>1 for v in pids.values()))
print("positive overlapping window pairs",overlap_pairs)
for mod in ["xgboost","lightgbm","catboost","tsfresh","aeon","sktime","torchaudio","librosa"]:
 print("package",mod,bool(importlib.util.find_spec(mod)))
