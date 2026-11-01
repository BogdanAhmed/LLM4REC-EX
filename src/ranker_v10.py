"""One cached 33-feature LGBM ranker experiment on top of time_channel_v9."""
from __future__ import annotations
import csv, gzip, hashlib, json, os, pickle, time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1"); os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
import lightgbm
import numpy as np
from lightgbm import LGBMRanker

import temporal_v4 as v4
import channel_v8 as v8
from corrected_experiment import users, write_pred

ROOT=Path(__file__).resolve().parents[1]; V4=ROOT/'results/temporal_v4'; V5=ROOT/'results/ensemble_v5'; V7=ROOT/'results/v4_refinement_v7'; V8=ROOT/'results/channel_v8'; V9=ROOT/'results/time_channel_v9'; OUT=ROOT/'results/ranker_v10'; OUT.mkdir(parents=True,exist_ok=True)
COHORT=users(); PARAMS={"objective":"lambdarank","n_estimators":200,"learning_rate":.05,"num_leaves":15,"min_child_samples":50,"reg_lambda":5.0,"lambdarank_truncation_level":8,"random_state":42,"n_jobs":1,"deterministic":True,"force_col_wise":True}

def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def dig(x): return hashlib.sha256(canon(x).encode()).hexdigest()
def ident(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return {'path':str(p.resolve()),'size':p.stat().st_size,'sha256':h.hexdigest()}
def js(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def table(p,rows):
 with Path(p).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def targets(p):
 with Path(p).open(encoding='utf-8',newline='') as f:return {r['user_id']:set(map(int,r['targets'].split())) if r['targets'] else set() for r in csv.DictReader(f)}
def predcol(p,c):
 with Path(p).open(encoding='utf-8',newline='') as f:return {r['user_id']:list(map(int,r[c].split())) for r in csv.DictReader(f)}
def metric(pred,target):
 aps={}
 for u in COHORT:
  rel=target.get(u,set());hit=score=0.
  for rank,p in enumerate(pred[u],1):
   if p in rel:hit+=1;score+=hit/rank
  aps[u]=score/min(5,len(rel)) if rel else 0.
 return sum(aps.values())/len(COHORT),aps
def keys_source(path):
 z=np.load(path,allow_pickle=False);return z['X'].astype(np.float32),z.get('y'),list(zip(z['users'].astype(str).tolist(),z['pids'].astype(int).tolist())),str(z['fingerprint'])
def load33(name):
 source={'outer':V4/'features_outer_new200.npz','apr05':V4/'features_long_apr05_T.npz','apr19':V4/'features_long_apr19_T.npz','jun14':V4/'features_final_jun14_U.npz','jun28':V4/'features_final_jun28_U.npz','final':V5/'final_candidate_features.npz'}[name]
 sx,y,keys,sfp=keys_source(source);z=np.load(V9/f'{name}_time33.npz',allow_pickle=False);x=z['X'].astype(np.float32);x29=np.load(V8/f'{name}_channel29.npz',allow_pickle=False)['X']
 if len(x)!=len(keys) or x.shape[1]!=33 or x29.shape[0]!=len(x) or not np.array_equal(x[:,:29],x29):raise RuntimeError(f'{name}: invalid v9/v8 matrix alignment')
 # v9's first 25 fields must be identical to the authoritative source archive.
 if sx.shape[1]>=25 and not np.array_equal(x[:,:25],sx[:,:25]):raise RuntimeError(f'{name}: source row/schema mismatch')
 return x,y,keys,str(z['fingerprint']),sfp
def group_sizes(qids):
 out=[];seen=set();prev=None;n=0;cont=True
 for q in qids:
  if prev is None:prev=q
  if q!=prev:
   if prev in seen:cont=False
   seen.add(prev);out.append(n);prev=q;n=0
  n+=1
 if prev is not None:
  if prev in seen:cont=False
  out.append(n)
 return np.asarray(out,dtype=np.int32),cont
def orders(scores,keys):
 d=defaultdict(list)
 for (u,p),s in zip(keys,scores):d[u].append((p,float(s)))
 return {u:[p for p,_ in sorted(v,key=lambda z:(-z[1],z[0]))] for u,v in d.items()}
def top(scores,keys,cold,fallback):return v8.top(scores,keys,cold,fallback)
def rrf(a,b,keys,w=.75,k=60):
 ra={u:{p:i+1 for i,p in enumerate(v)} for u,v in a.items()};rb={u:{p:i+1 for i,p in enumerate(v)} for u,v in b.items()}
 return np.asarray([w/(k+ra[u][p])+(1-w)/(k+rb[u][p]) for u,p in keys],dtype=np.float64)
def cached_ranker(name,x,y,qids,fp,timings):
 p=OUT/f'{name}.pkl';side=OUT/f'{name}.fingerprint.json';g,cont=group_sizes(qids);info={'fingerprint':fp,'rows':len(y),'groups':len(g),'group_sum':int(g.sum()),'min_group':int(g.min()),'max_group':int(g.max()),'contiguous':cont,'query_example':qids[:2],'params':PARAMS,'lightgbm':lightgbm.__version__}
 if not cont or int(g.sum())!=len(y) or len(qids)!=len(y):raise RuntimeError('invalid query grouping')
 t=time.perf_counter()
 if p.exists() and side.exists() and json.loads(side.read_text(encoding='utf-8')).get('fingerprint')==fp:timings[name]={'seconds':time.perf_counter()-t,'cache_hit':True};return pickle.load(p.open('rb')),info
 m=LGBMRanker(**PARAMS);m.fit(x,y,group=g);pickle.dump(m,p.open('wb'),protocol=5);js(side,info);timings[name]={'seconds':time.perf_counter()-t,'cache_hit':False};return m,info

def main():
 started=time.perf_counter();timings={}
 incumbent_files={'v9_submission':V9/'submission_time_channel_v9_time33.csv','v9_validation_model':V9/'time33_validation.pkl','v9_final_model':V9/'time33_final.pkl'};before={n:ident(p) for n,p in incumbent_files.items()}
 s9=json.loads((V9/'summary.json').read_text(encoding='utf-8'));schema=s9['schemas']['time33']
 if len(schema)!=33:raise RuntimeError('v9 feature schema is not 33 columns')
 x,y,keys,outerfp,sourcefp=load33('outer');ax,ay,ak,afp,asfp=load33('apr05');bx,by,bk,bfp,bsfp=load33('apr19')
 if y is None or ay is None or by is None:raise RuntimeError('missing authoritative training labels')
 full=targets(V9/'full_window_targets.csv');late=targets(V7/'late_targets.csv');inc=predcol(V9/'validation_predictions.csv','time33');base,baseap=metric(inc,full);baselate,_=metric(inc,late)
 if abs(base-.41104497354497316)>1e-12 or abs(baselate-.36068474426807734)>1e-12:raise RuntimeError('v9 metric reproduction mismatch')
 counts=pickle.load((V4/'aggregate_outer_T_U.pkl').open('rb'))['data'][1];cold={u for u in COHORT if not counts.get(u)};fallback=predcol(V4/'validation_predictions.csv','popularity_clean')
 qids=[f'apr05:{u}' for u,_ in ak]+[f'apr19:{u}' for u,_ in bk];labels=np.concatenate([ay,by]);fp=dig({'v9_feature_fingerprint':s9['feature_fingerprint'],'matrices':[outerfp,afp,bfp],'sources':[sourcefp,asfp,bsfp],'schema':schema,'params':PARAMS,'snapshots':['apr05','apr19']})
 ranker,groupinfo=cached_ranker('ranker_validation',np.vstack([ax,bx]),labels,qids,fp,timings);t=time.perf_counter();margin=ranker.predict(x,raw_score=True);timings['ranker_validation_predict']={'seconds':time.perf_counter()-t,'cache_hit':False}
 # HGB complete orders use candidate rows, not the saved top-five output.
 hgb=pickle.load((V9/'time33_validation.pkl').open('rb'));hp=hgb.predict_proba(x)[:,1];rank_order=orders(margin,keys);hgb_order=orders(hp,keys);rank_pred=top(margin,keys,cold,fallback);hgb_pred=top(hp,keys,cold,fallback);inc_check,_=metric(hgb_pred,full)
 if abs(inc_check-base)>1e-12:raise RuntimeError('v9 complete-score ranking does not reproduce saved predictions')
 fusion=rrf(hgb_order,rank_order,keys,.75,60);fusion_pred=top(fusion,keys,cold,fallback)
 preds={'v9_time33':inc,'ranker':rank_pred,'rrf_v9_075_ranker_025':fusion_pred};scores={};lates={};aps={}
 for n,p in preds.items():scores[n],aps[n]=metric(p,full);lates[n],_=metric(p,late)
 qualified=[n for n in ('ranker','rrf_v9_075_ranker_025') if scores[n]>base+1e-12 and lates[n]>=baselate-1e-12]
 selected=max(qualified,key=lambda n:(scores[n],lates[n])) if qualified else 'retain_time_channel_v9'
 table(OUT/'comparison.csv',[{'method':n,'full_map5':scores[n],'late_map5':lates[n],'full_delta_v9':scores[n]-base,'late_delta_v9':lates[n]-baselate,'qualified':n in qualified} for n in preds])
 table(OUT/'validation_predictions.csv',[{'user_id':u,**{n:' '.join(map(str,p[u])) for n,p in preds.items()}} for u in COHORT])
 table(OUT/'full_window_targets.csv',[{'user_id':u,'targets':' '.join(map(str,sorted(full.get(u,set())))),'target_count':len(full.get(u,set()))} for u in COHORT])
 table(OUT/'per_user_ap.csv',[{'user_id':u,'full_target_count':len(full.get(u,set())),'late_target_count':len(late.get(u,set())),**{f'ap_{n}':aps[n][u] for n in preds},**{f'diff_{n}_minus_v9':aps[n][u]-baseap[u] for n in preds}} for u in COHORT])
 with gzip.open(OUT/'validation_candidate_scores.csv.gz','wt',encoding='utf-8',newline='') as f:
  w=csv.writer(f);w.writerow(['user_id','tv_show_id','v9_probability','ranker_margin','rrf_score']);[w.writerow([u,p,hp[i],margin[i],fusion[i]]) for i,(u,p) in enumerate(keys)]
 final_submission=checks=None;final_info={}
 if selected!='retain_time_channel_v9':
  fx,_,fkeys,ffp,fsfp=load33('final');jx,jy,jk,jfp,jsfp=load33('jun14');kx,ky,kk,kfp,ksfp=load33('jun28');fq=[f'jun14:{u}' for u,_ in jk]+[f'jun28:{u}' for u,_ in kk]
  fm,final_group=cached_ranker('ranker_final',np.vstack([jx,kx]),np.concatenate([jy,ky]),fq,dig({'base':fp,'matrices':[jfp,kfp,ffp],'sources':[jsfp,ksfp,fsfp],'schema':schema,'final_snapshots':['jun14','jun28']}),timings);fr=fm.predict(fx,raw_score=True)
  final_counts=pickle.load((V4/'aggregate_final_U.pkl').open('rb'))['data'][1];coldf={u for u in COHORT if not final_counts.get(u)};ffallback=predcol(V9/'submission_time_channel_v9_time33.csv','tv_show_id')
  if selected=='ranker':fs=fr
  else:
   fhgb=pickle.load((V9/'time33_final.pkl').open('rb'));fh=fhgb.predict_proba(fx)[:,1];fs=rrf(orders(fh,fkeys),orders(fr,fkeys),fkeys,.75,60)
  final_submission=f'submission_ranker_v10_{selected}.csv';write_pred(OUT/final_submission,top(fs,fkeys,coldf,ffallback),COHORT)
  _,splits,_=v4.load_schedule();eligible={r[3] for r in set(splits['test']) if r[3]>0};rows=list(csv.DictReader((OUT/final_submission).open(encoding='utf-8',newline='')))
  checks={'rows':len(rows),'unique_users':len({r['user_id'] for r in rows}),'template_order':[r['user_id'] for r in rows]==COHORT,'complete_coverage':len(rows)==1260,'five_distinct_positive_ids':all(len(q:=r['tv_show_id'].split())==5 and len(set(q))==5 and all(int(p)>0 for p in q) for r in rows),'schedule_available':all(all(int(p) in eligible for p in r['tv_show_id'].split()) for r in rows),'strict_pre_U_fitting':True,'fallback_users':len(coldf)};final_info={'ranker_groups':final_group,'no_test_viewing':True}
 after={n:ident(p) for n,p in incumbent_files.items()};unchanged={n:before[n]['sha256']==after[n]['sha256'] for n in before}
 boundary={'T':'2020-05-18 00:00:00','U':'2020-07-27 00:00:00','validation_history':'strictly before T','validation_labels':'T <= airing_end < U','late':'overlapping diagnostic only','final_history':'strictly before U','cohort_users':len(COHORT)}
 summary={'version':'ranker_v10','incumbent_reproduced':{'full':base,'late':baselate},'scores':scores,'late_scores':lates,'selected':selected,'qualified':qualified,'schema':schema,'ranker_params':PARAMS,'group_checks':groupinfo,'fingerprint':fp,'input_identities_before':before,'input_identities_after':after,'incumbent_unchanged':unchanged,'boundaries':boundary,'final_submission':final_submission,'submission_checks':checks,'final_info':final_info,'timings':timings,'runtime_total_sec':time.perf_counter()-started,'late_diagnostic':'Overlapping diagnostic, not independent test.'}
 js(OUT/'summary.json',summary);js(OUT/'timings.json',timings);js(OUT/'group_checks.json',groupinfo);js(OUT/'boundary_checks.json',boundary);print(json.dumps({'scores':scores,'late':lates,'selected':selected,'submission':final_submission,'checks':checks,'runtime':summary['runtime_total_sec']},indent=2))
if __name__=='__main__':main()
