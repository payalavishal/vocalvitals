import json,collections
from pathlib import Path
m=json.loads(Path("data/processed/swallow_binary_metadata.json").read_text());by=collections.defaultdict(lambda:{0:[],1:[]})
for r in m["records"]:by[r["recording_id"]][r["label"]].append((r["start_ms"],r["end_ms"]))
bad=[];pairs=0
for stem,v in by.items():
 pos=sorted(v[1]);neg=sorted(v[0]);i=0
 for a,b in neg:
  while i<len(pos) and pos[i][1]<=a:i+=1
  j=i
  while j<len(pos) and pos[j][0]<b:
   if min(b,pos[j][1])>max(a,pos[j][0]):pairs+=1;bad.append((stem,a,b,pos[j][0],pos[j][1]))
   j+=1
print("negative windows overlapping positive windows",len(set((x[0],x[1],x[2]) for x in bad)))
print("overlap pairs",pairs,"examples",bad[:10])
print("fraction of negative windows",len(set((x[0],x[1],x[2]) for x in bad))/sum(len(v[0]) for v in by.values()))
