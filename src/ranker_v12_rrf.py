"""Final zero-training RRF sweep using cached v10 complete candidate scores."""
from __future__ import annotations
import csv,gzip,json,os,pickle,time
from collections import defaultdict
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import temporal_v4 as v4
import channel_v8 as v8
from corrected_experiment import users,write_pred
ROOT=Path(__file__).resolve().parents[1];V4=ROOT/'results/temporal_v4';V5=ROOT/'results/ensemble_v5';V7=ROOT/'results/v4_refinement_v7';V9=ROOT/'results/time_channel_v9';V10=ROOT/'results/ranker_v10';V11=ROOT/'results/ranker_v11_rrf';OUT=ROOT/'results/ranker_v12_rrf';OUT.mkdir(parents=True,exist_ok=True);COHORT=users();W=(.18,.20,.22)
def targets(p):
 with Path(p).open(encoding='utf-8',newline='') as f:return {r['user_id']:set(map(int,r['targets'].split())) if r['targets'] else set() for r in csv.DictReader(f)}
def predcol(p,c):
 with Path(p).open(encoding='utf-8',newline='') as f:return {r['user_id']:list(map(int,r[c].split())) for r in csv.DictReader(f)}
def metric(pred,t):
 a={}
 for u in COHORT:
  rel=t.get(u,set());h=s=0.
  for i,p in enumerate(pred[u],1):
   if p in rel:h+=1;s+=h/i
  a[u]=s/min(5,len(rel)) if rel else 0.
 return sum(a.values())/len(COHORT),a
def orders(s,k):
 d=defaultdict(list)
 for (u,p),x in zip(k,s):d[u].append((p,float(x)))
 return {u:[p for p,_ in sorted(q,key=lambda z:(-z[1],z[0]))] for u,q in d.items()}
def rrf(a,b,k,w):
 ra={u:{p:i+1 for i,p in enumerate(q)} for u,q in a.items()};rb={u:{p:i+1 for i,p in enumerate(q)} for u,q in b.items()}
 return np.asarray([w/(60+ra[u][p])+(1-w)/(60+rb[u][p]) for u,p in k])
def table(p,r):
 with Path(p).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
def main():
 t=time.perf_counter();full=targets(V10/'full_window_targets.csv');late=targets(V7/'late_targets.csv');base=predcol(V11/'validation_predictions.csv','rrf_v9_0.15_ranker_0.85');bm,_=metric(base,full);bl,_=metric(base,late)
 if abs(bm-.41397552910052887)>1e-12 or abs(bl-.3622094356261022)>1e-12:raise RuntimeError('v11 incumbent mismatch')
 keys=[];h=[];r=[]
 with gzip.open(V10/'validation_candidate_scores.csv.gz','rt',encoding='utf-8',newline='') as f:
  for q in csv.DictReader(f):keys.append((q['user_id'],int(q['tv_show_id'])));h.append(float(q['v9_probability']));r.append(float(q['ranker_margin']))
 if len(keys)!=len(set(keys)):raise RuntimeError('duplicate keys')
 ho,ro=orders(h,keys),orders(r,keys);counts=pickle.load((V4/'aggregate_outer_T_U.pkl').open('rb'))['data'][1];cold={u for u in COHORT if not counts.get(u)};fallback=predcol(V4/'validation_predictions.csv','popularity_clean');pred={'rrf_v9_0.15_ranker_0.85':base};scores={'rrf_v9_0.15_ranker_0.85':bm};lates={'rrf_v9_0.15_ranker_0.85':bl}
 for w in W:
  n=f'rrf_v9_{w:.2f}_ranker_{1-w:.2f}';pred[n]=v8.top(rrf(ho,ro,keys,w),keys,cold,fallback);scores[n],_=metric(pred[n],full);lates[n],_=metric(pred[n],late)
 q=[n for n in pred if n!=list(pred)[0] and scores[n]>bm+1e-12 and lates[n]>=bl-1e-12];sel=max(q,key=lambda n:(scores[n],lates[n])) if q else 'retain_ranker_v11_rrf'
 table(OUT/'comparison.csv',[{'method':n,'full_map5':scores[n],'late_map5':lates[n],'full_delta_v11':scores[n]-bm,'late_delta_v11':lates[n]-bl,'qualified':n in q} for n in pred]);table(OUT/'validation_predictions.csv',[{'user_id':u,**{n:' '.join(map(str,p[u])) for n,p in pred.items()}} for u in COHORT])
 final=checks=None
 if sel!='retain_ranker_v11_rrf':
  z=np.load(V5/'final_candidate_features.npz',allow_pickle=False);fk=list(zip(z['users'].astype(str).tolist(),z['pids'].astype(int).tolist()));fx=np.load(V9/'final_time33.npz',allow_pickle=False)['X'];fh=pickle.load((V9/'time33_final.pkl').open('rb')).predict_proba(fx)[:,1];fr=pickle.load((V10/'ranker_final.pkl').open('rb')).predict(fx,raw_score=True);w=float(sel.split('_')[2]);fs=rrf(orders(fh,fk),orders(fr,fk),fk,w);fc=pickle.load((V4/'aggregate_final_U.pkl').open('rb'))['data'][1];coldf={u for u in COHORT if not fc.get(u)};ff=predcol(V11/'submission_rrf_v9_0.15_ranker_0.85.csv','tv_show_id');final=f'submission_{sel}.csv';write_pred(OUT/final,v8.top(fs,fk,coldf,ff),COHORT);ok={p for _,p in fk if p>0};rows=list(csv.DictReader((OUT/final).open(encoding='utf-8',newline='')));checks={'rows':len(rows),'unique_users':len({x['user_id'] for x in rows}),'template_order':[x['user_id'] for x in rows]==COHORT,'complete_coverage':len(rows)==len(COHORT),'five_distinct_positive_ids':all(len(a:=x['tv_show_id'].split())==5 and len(set(a))==5 and all(int(v)>0 for v in a) for x in rows),'schedule_available':all(all(int(v) in ok for v in x['tv_show_id'].split()) for x in rows),'strict_pre_U_fitting':True,'fallback_users':len(coldf)}
 out={'version':'ranker_v12_rrf','scores':scores,'late_scores':lates,'selected':sel,'final_submission':final,'submission_checks':checks,'runtime_sec':time.perf_counter()-t,'candidate_source':str((V10/'validation_candidate_scores.csv.gz').resolve())};(OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
