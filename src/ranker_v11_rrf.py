"""Zero-training RRF postprocessing of cached v9 HGB and v10 ranker scores."""
from __future__ import annotations
import csv, gzip, json, os, pickle, time
from collections import defaultdict
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import temporal_v4 as v4
import channel_v8 as v8
import ranker_v10 as r10
from corrected_experiment import users,write_pred
ROOT=Path(__file__).resolve().parents[1];V4=ROOT/'results/temporal_v4';V5=ROOT/'results/ensemble_v5';V7=ROOT/'results/v4_refinement_v7';V9=ROOT/'results/time_channel_v9';V10=ROOT/'results/ranker_v10';OUT=ROOT/'results/ranker_v11_rrf';OUT.mkdir(parents=True,exist_ok=True);COHORT=users();W=(.05,.10,.15)
def targets(p):
 with Path(p).open(encoding='utf-8',newline='') as f:return {r['user_id']:set(map(int,r['targets'].split())) if r['targets'] else set() for r in csv.DictReader(f)}
def predcol(p,c):
 with Path(p).open(encoding='utf-8',newline='') as f:return {r['user_id']:list(map(int,r[c].split())) for r in csv.DictReader(f)}
def metric(pred,target):
 aps={}
 for u in COHORT:
  rel=target.get(u,set());hit=s=0.
  for i,p in enumerate(pred[u],1):
   if p in rel:hit+=1;s+=hit/i
  aps[u]=s/min(5,len(rel)) if rel else 0.
 return sum(aps.values())/len(COHORT),aps
def orders(values,keys):
 d=defaultdict(list)
 for (u,p),s in zip(keys,values):d[u].append((p,float(s)))
 return {u:[p for p,_ in sorted(q,key=lambda z:(-z[1],z[0]))] for u,q in d.items()}
def rrf(a,b,keys,w,k=60):
 ra={u:{p:i+1 for i,p in enumerate(q)} for u,q in a.items()};rb={u:{p:i+1 for i,p in enumerate(q)} for u,q in b.items()}
 return np.asarray([w/(k+ra[u][p])+(1-w)/(k+rb[u][p]) for u,p in keys],dtype=np.float64)
def table(p,rows):
 with Path(p).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 t=time.perf_counter();full=targets(V10/'full_window_targets.csv');late=targets(V7/'late_targets.csv');base=predcol(V10/'validation_predictions.csv','ranker');bm,_=metric(base,full);bl,_=metric(base,late)
 if abs(bm-.413634259259259)>1e-12 or abs(bl-.36178086419753075)>1e-12:raise RuntimeError('ranker_v10 baseline mismatch')
 keys=[];h=[];r=[]
 with gzip.open(V10/'validation_candidate_scores.csv.gz','rt',encoding='utf-8',newline='') as f:
  for q in csv.DictReader(f):keys.append((q['user_id'],int(q['tv_show_id'])));h.append(float(q['v9_probability']));r.append(float(q['ranker_margin']))
 if len(keys)!=len(set(keys)):raise RuntimeError('duplicate validation candidate keys')
 h=np.asarray(h);r=np.asarray(r);ho,ro=orders(h,keys),orders(r,keys);counts=pickle.load((V4/'aggregate_outer_T_U.pkl').open('rb'))['data'][1];cold={u for u in COHORT if not counts.get(u)};fallback=predcol(V4/'validation_predictions.csv','popularity_clean')
 preds={'ranker_v10':base};scores={'ranker_v10':bm};lates={'ranker_v10':bl};aps={}
 for w in W:
  name=f'rrf_v9_{w:.2f}_ranker_{1-w:.2f}';preds[name]=v8.top(rrf(ho,ro,keys,w),keys,cold,fallback);scores[name],aps[name]=metric(preds[name],full);lates[name],_=metric(preds[name],late)
 qualify=[n for n in preds if n!='ranker_v10' and scores[n]>bm+1e-12 and lates[n]>=bl-1e-12];selected=max(qualify,key=lambda n:(scores[n],lates[n])) if qualify else 'retain_ranker_v10'
 table(OUT/'comparison.csv',[{'method':n,'full_map5':scores[n],'late_map5':lates[n],'full_delta_ranker_v10':scores[n]-bm,'late_delta_ranker_v10':lates[n]-bl,'qualified':n in qualify} for n in preds])
 table(OUT/'validation_predictions.csv',[{'user_id':u,**{n:' '.join(map(str,p[u])) for n,p in preds.items()}} for u in COHORT])
 final=None;checks=None
 if selected!='retain_ranker_v10':
  z=np.load(V5/'final_candidate_features.npz',allow_pickle=False);fkeys=list(zip(z['users'].astype(str).tolist(),z['pids'].astype(int).tolist()));fx=np.load(V9/'final_time33.npz',allow_pickle=False)['X'];fh=pickle.load((V9/'time33_final.pkl').open('rb')).predict_proba(fx)[:,1];fr=pickle.load((V10/'ranker_final.pkl').open('rb')).predict(fx,raw_score=True);w=float(selected.split('_')[2]);fs=rrf(orders(fh,fkeys),orders(fr,fkeys),fkeys,w);fc=pickle.load((V4/'aggregate_final_U.pkl').open('rb'))['data'][1];coldf={u for u in COHORT if not fc.get(u)};ff=predcol(V10/'submission_ranker_v10_ranker.csv','tv_show_id');final=f'submission_{selected}.csv';write_pred(OUT/final,v8.top(fs,fkeys,coldf,ff),COHORT);_,splits,_=v4.load_schedule();eligible={q[3] for q in set(splits['test']) if q[3]>0};rows=list(csv.DictReader((OUT/final).open(encoding='utf-8',newline='')));checks={'rows':len(rows),'unique_users':len({q['user_id'] for q in rows}),'template_order':[q['user_id'] for q in rows]==COHORT,'five_distinct_positive_ids':all(len(a:=q['tv_show_id'].split())==5 and len(set(a))==5 and all(int(x)>0 for x in a) for q in rows),'schedule_available':all(all(int(x) in eligible for x in q['tv_show_id'].split()) for q in rows),'strict_pre_U_fitting':True,'fallback_users':len(coldf)}
 out={'version':'ranker_v11_rrf','baseline':{'full':bm,'late':bl},'scores':scores,'late_scores':lates,'selected':selected,'final_submission':final,'submission_checks':checks,'candidate_source':str((V10/'validation_candidate_scores.csv.gz').resolve()),'policy':'complete 200-candidate RRF, k=60; v9/ranker weights fixed','runtime_sec':time.perf_counter()-t};(OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
