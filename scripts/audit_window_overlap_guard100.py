import json,collections
from pathlib import Path
m=json.loads(Path("data/interim/guard100/swallow_binary_metadata.json").read_text());by=collections.defaultdict(lambda:{0:[],1:[]})
for r in m["records"]:by[r["recording_id"]][r["label"]].append((r["start_ms"],r["end_ms"]))
bad=set();pairs=0
for stem,v in by.items():
 pos=sorted(v[1]);neg=sorted(v[0]);i=0
 for a,b in neg:
  while i<len(pos) and pos[i][1]<=a:i+=1
  j=i
  while j<len(pos) and pos[j][0]<b:
   if min(b,pos[j][1])>max(a,pos[j][0]):pairs+=1;bad.add((stem,a,b))
   j+=1
print("guard",m["event_guard_ms"],"negative_count",len([r for r in m["records"] if r["label"]==0]))
print("overlapping_negative_windows",len(bad),"overlap_pairs",pairs)
print("fraction",len(bad)/sum(len(v[0]) for v in by.values()))
