#!/usr/bin/env python3
"""Verify the public SWEET.TV two-user supplement with Python stdlib only."""
import csv
from collections import defaultdict
from pathlib import Path
root=Path(__file__).resolve().parent

def records(name):
 with (root/name).open(encoding='utf-8-sig',newline='') as f:
  return list(csv.DictReader(f))

def ap5(items,targets):
 if not targets:return 0.0
 hits=0;total=0.0
 for k,item in enumerate(items,1):
  if item in targets:
   hits+=1; total+=hits/k
 return total/min(5,len(targets))

targets={r['user_id']:set(map(int,r['target_ids'].split())) for r in records('two_user_validation_targets.csv')}
rows=records('two_user_clean_validation_top5.csv')
assert len(rows)==60, len(rows)
by=defaultdict(list)
for r in rows:
 by[(r['user_id'],r['method'])].append(r)
assert len(by)==12
for (user,method),v in by.items():
 v.sort(key=lambda x:int(x['rank']))
 pred=[int(x['tv_show_id']) for x in v]
 assert len(pred)==len(set(pred))==5
 assert [int(x['rank']) for x in v]==[1,2,3,4,5]
 assert all(int(x['held_out_target_hit']) == (int(x['tv_show_id']) in targets[user]) for x in v)
 score=ap5(pred,targets[user])
 assert abs(score-float(v[0]['ap_at_5'])) < 1e-8, (user,method,score)
 print(f'PASS {user} / {method}: AP@5={score:.6f}')

scores=records('validation_per_user_ap6.csv')
means={r['method']:float(r['mean_ap5_recomputed']) for r in records('validation_metrics_recomputed.csv')}
assert len(scores)==1260 and len({r['user_id'] for r in scores})==1260
for method,reported in means.items():
 computed=sum(float(row[method]) for row in scores)/len(scores)
 assert abs(computed-reported) < 1e-9, (method,computed,reported)
 print(f'PASS {method}: MAP@5={computed:.9f}')
print('All evidence checks passed.')
