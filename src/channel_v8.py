"""Channel preference refinement using only cached v4 candidate matrices."""
from __future__ import annotations
import csv, gzip, hashlib, json, os, pickle, time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("LOKY_MAX_CPU_COUNT","1")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import temporal_v4 as v4
from corrected_experiment import users, write_pred

ROOT=Path(__file__).resolve().parents[1]; V2=ROOT/'results/corrected_v2'; V4=ROOT/'results/temporal_v4'; V5=ROOT/'results/ensemble_v5'; V7=ROOT/'results/v4_refinement_v7'; OUT=ROOT/'results/channel_v8'; RAW=ROOT/'data/raw/dataset11-30.csv'; OUT.mkdir(parents=True,exist_ok=True)
COHORT=users(); T=datetime(2020,5,18); U=datetime(2020,7,27); CUTS={'apr05':datetime(2020,4,5),'apr19':datetime(2020,4,19),'outer':T,'jun14':datetime(2020,6,14),'jun28':datetime(2020,6,28),'final':U}; FEATURES=v4.FEATURES_V4+['channel_pref_mean','channel_pref_per_day','channel_pref_max','log_channel_support']
def cj(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def dig(x):return hashlib.sha256(cj(x).encode()).hexdigest()
def js(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf-8')
def ident(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return {'path':str(p.resolve()),'size':p.stat().st_size,'sha256':h.hexdigest()}
def load(name):
 z=np.load(V4/name,allow_pickle=False);return z['X'],z.get('y'),list(zip(z['users'].astype(str).tolist(),z['pids'].astype(int).tolist())),str(z['fingerprint'])
def read_targets(p):
 with Path(p).open(encoding='utf-8',newline='') as f:return {r['user_id']:set(map(int,r['targets'].split())) if r['targets'] else set() for r in csv.DictReader(f)}
def pred_col(p,c):
 with Path(p).open(encoding='utf-8',newline='') as f:return {r['user_id']:list(map(int,r[c].split())) for r in csv.DictReader(f)}
def score(pred,targets):
 a={}
 for u in COHORT:
  rel=targets.get(u,set());h=s=0.
  for r,p in enumerate(pred[u],1):
   if p in rel:h+=1;s+=h/r
  a[u]=s/min(5,len(rel)) if rel else 0.
 return sum(a.values())/1260,a
def orders(scores,keys):
 d=defaultdict(list)
 for (u,p),s in zip(keys,scores):d[u].append((p,float(s)))
 return {u:[p for p,_ in sorted(v,key=lambda q:(-q[1],q[0]))] for u,v in d.items()}
def top(scores,keys,cold,fall):
 o=orders(scores,keys);return {u:list(fall[u]) if u in cold else o[u][:5] for u in COHORT}
def table(p,rows):
 with Path(p).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def profiles(base,times):
 path=OUT/'channel_profiles.pkl'; fp=dig({'base':base,'raw':ident(RAW),'cuts':{k:v.isoformat() for k,v in CUTS.items()}});t=time.perf_counter()
 if path.exists():
  q=pickle.load(path.open('rb'))
  if q['fingerprint']==fp:times['profiles']={'seconds':time.perf_counter()-t,'cache_hit':True};return q
 acc={k:defaultdict(Counter) for k in CUTS};supp={k:defaultdict(float) for k in CUTS};cross={k:0 for k in CUTS};rawchs=set()
 for c in pd.read_csv(RAW,dtype={'user_id':str,'vsetv_id':'int64'},parse_dates=['start_time','stop_time'],chunksize=300000):
  rawchs.update(map(int,c.vsetv_id.unique()))
  for n,cut in CUTS.items():
   s=c.loc[c.start_time<cut,['user_id','vsetv_id','start_time','stop_time']].copy()
   if s.empty:continue
   cross[n]+=int((s.stop_time>cut).sum());s['w']=(s.stop_time.where(s.stop_time<cut,cut)-s.start_time).dt.total_seconds().clip(0,7200)
   for (u,ch),w in s.groupby(['user_id','vsetv_id'],sort=False).w.sum().items():acc[n][u][int(ch)]+=float(w)
   for u,w in s.groupby('user_id',sort=False).w.sum().items():supp[n][u]+=float(w)
 prof={n:{u:({ch:w/supp[n][u] for ch,w in d.items()},supp[n][u]) for u,d in acc[n].items() if supp[n][u]>0} for n in CUTS}
 sch,_,_=v4.load_schedule();schchs={r[0] for r in sch}; mapping={'rule':'vsetv_id equals channel_id','raw_vsetv_ids':len(rawchs),'schedule_channel_ids':len(schchs),'shared_ids':len(rawchs&schchs),'raw_without_schedule':sorted(rawchs-schchs),'schedule_without_raw_count':len(schchs-rawchs)}
 q={'fingerprint':fp,'profiles':prof,'mapping':mapping,'crossing_sessions':cross,'boundary_fixture':{'session':'2020-05-17 23:00 to 2020-05-18 02:00','cutoff':'2020-05-18 00:00','contribution_seconds':3600,'cap_seconds':3600,'start_at_cutoff_excluded':True}};pickle.dump(q,path.open('wb'),protocol=5);js(OUT/'channel_profile_checks.json',{k:q[k] for k in ('mapping','crossing_sessions','boundary_fixture')});times['profiles']={'seconds':time.perf_counter()-t,'cache_hit':False};return q
def airings(start,end):
 r,_,_=v4.load_schedule();d=defaultdict(Counter)
 for ch,st,_,p in set(r):
  if p>0 and start<=st<end:d[p][ch]+=1
 return d,max((end-start).total_seconds()/86400,1e-9)
def add_channel(x,keys,prof,future,days,name,base,times):
 path=OUT/f'{name}_channel29.npz';fp=dig({'base':base,'name':name,'schema':FEATURES});t=time.perf_counter()
 if path.exists():
  z=np.load(path,allow_pickle=False)
  if str(z['fingerprint'])==fp:times[name]={'seconds':time.perf_counter()-t,'cache_hit':True};return z['X']
 e=np.zeros((len(x),4),np.float32)
 for i,(u,p) in enumerate(keys):
  pref,total=prof.get(u,({},0.));cs=future.get(p,{});n=sum(cs.values());w=sum(pref.get(ch,0)*c for ch,c in cs.items())
  if n:e[i,:3]=(w/n,w/days,max((pref.get(ch,0) for ch in cs),default=0))
  e[i,3]=np.log1p(total)
 z=np.hstack([x,e]).astype(np.float32);np.savez_compressed(path,fingerprint=fp,X=z);js(path.with_suffix('.fingerprint.json'),{'fingerprint':fp,'features':FEATURES});times[name]={'seconds':time.perf_counter()-t,'cache_hit':False};return z
def fit(name,x,y,fp,times):
 p=OUT/f'{name}.pkl';s=OUT/f'{name}.fingerprint.json';t=time.perf_counter()
 if p.exists() and s.exists() and json.loads(s.read_text()).get('fingerprint')==fp:times[name]={'seconds':time.perf_counter()-t,'cache_hit':True};return pickle.load(p.open('rb'))
 m=HistGradientBoostingClassifier(**v4.HGB_PARAMS);m.fit(x,y);pickle.dump(m,p.open('wb'),protocol=5);js(s,{'fingerprint':fp,'features':FEATURES,'params':v4.HGB_PARAMS});times[name]={'seconds':time.perf_counter()-t,'cache_hit':False};return m
def verify_late():
 targets=read_targets(V7/'late_targets.csv');pred=list(csv.DictReader((V7/'validation_predictions.csv').open(encoding='utf-8',newline='')));rows=[]
 for r in pred:
  u=r['user_id'];
  def ap(c):
   rel=targets[u];h=s=0.
   for i,p in enumerate(map(int,r[c].split()),1):
    if p in rel:h+=1;s+=h/i
   return s/min(5,len(rel)) if rel else 0.
  a,b=ap('v4'),ap('ranker');rows.append({'user_id':u,'late_target_count':len(targets[u]),'ap5_v4':a,'ap5_ranker':b,'ranker_minus_v4':b-a})
 table(OUT/'late_metric_verification.csv',rows);out={'users':len(rows),'empty_targets':sum(not targets[r['user_id']] for r in pred),'v4_map5':sum(r['ap5_v4'] for r in rows)/1260,'ranker_map5':sum(r['ap5_ranker'] for r in rows)/1260};out['matches_expected']=abs(out['v4_map5']-.35464594356261)<1e-12 and abs(out['ranker_map5']-.3532940917107583)<1e-12;js(OUT/'late_metric_verification.json',out);return out
def main():
 t0=time.perf_counter();times={};lateverify=verify_late();ids={n:ident(p) for n,p in {'outer':V4/'features_outer_new200.npz','a':V4/'features_long_apr05_T.npz','b':V4/'features_long_apr19_T.npz','j':V4/'features_final_jun14_U.npz','k':V4/'features_final_jun28_U.npz','v4m':V4/'reranker_validation_v4.pkl','v4final':V4/'reranker_final_v4.pkl','sub':V4/'submission_temporal_v4_new200.csv','finalx':V5/'final_candidate_features.npz'}.items()};base=dig({'v4':json.loads((V4/'summary.json').read_text())['fingerprint'],'ids':ids,'schema':FEATURES});pc=profiles(base,times)['profiles']
 targets=read_targets(V4/'validation_targets.csv');late=read_targets(V7/'late_targets.csv');actual=pred_col(V4/'validation_predictions.csv','new_reranker_200_clean');control,ap0=score(actual,targets)
 if abs(control-.40403130511463825)>1e-12:raise RuntimeError('v4 control mismatch')
 x,y,keys,of=load('features_outer_new200.npz');fo,do=airings(T,U);xo=add_channel(x,keys,pc['outer'],fo,do,'outer',base,times);counts=pickle.load((V4/'aggregate_outer_T_U.pkl').open('rb'))['data'][1];cold={u for u in COHORT if not counts.get(u)};fallback=pred_col(V4/'validation_predictions.csv','popularity_clean')
 a,ay,ak,af=load('features_long_apr05_T.npz');b,by,bk,bf=load('features_long_apr19_T.npz');fa,da=airings(CUTS['apr05'],T);fb,db=airings(CUTS['apr19'],T);ac=add_channel(a,ak,pc['apr05'],fa,da,'apr05',base,times);bc=add_channel(b,bk,pc['apr19'],fb,db,'apr19',base,times);m=fit('channel_hgb_validation',np.vstack([ac,bc]),np.concatenate([ay,by]),dig({'base':base,'snapshots':[af,bf]}),times)
 p4=pickle.load((V4/'reranker_validation_v4.pkl').open('rb')).predict_proba(x)[:,1];p=m.predict_proba(xo)[:,1];pred={'v4':actual,'channel_hgb':top(p,keys,cold,fallback),'blend_v4_075_channel_025':top(.75*p4+.25*p,keys,cold,fallback)};scores={n:score(q,targets)[0] for n,q in pred.items()};aps={n:score(q,targets)[1] for n,q in pred.items()};lates={n:score(q,late)[0] for n,q in pred.items()};qual=[n for n in pred if n!='v4' and scores[n]>scores['v4']+1e-12 and lates[n]>=lates['v4']-1e-12];sel=max(qual,key=lambda n:(scores[n],n)) if qual else 'retain_temporal_v4'
 table(OUT/'comparison.csv',[{'method':n,'full_map5':scores[n],'late_map5':lates[n],'full_delta':scores[n]-scores['v4'],'late_delta':lates[n]-lates['v4']} for n in pred]);table(OUT/'full_window_targets.csv',[{'user_id':u,'targets':' '.join(map(str,sorted(targets[u]))),'target_count':len(targets[u])} for u in COHORT]);table(OUT/'validation_predictions.csv',[{'user_id':u,**{n:' '.join(map(str,q[u])) for n,q in pred.items()}} for u in COHORT]);table(OUT/'per_user_ap.csv',[{'user_id':u,'full_target_count':len(targets[u]),'late_target_count':len(late[u]),**{f'ap_{n}':aps[n][u] for n in pred},**{f'diff_{n}_minus_v4':aps[n][u]-aps['v4'][u] for n in pred}} for u in COHORT])
 with gzip.open(OUT/'validation_candidate_scores.csv.gz','wt',encoding='utf-8',newline='') as f:
  w=csv.writer(f);w.writerow(['user_id','tv_show_id','v4_probability','channel_hgb_probability','blend_probability']);[w.writerow([u,pid,p4[i],p[i],.75*p4[i]+.25*p[i]]) for i,(u,pid) in enumerate(keys)]
 final=None;checks=None
 if sel!='retain_temporal_v4':
  z=np.load(V5/'final_candidate_features.npz',allow_pickle=False);fx=z['X'];fk=list(zip(z['users'].astype(str).tolist(),z['pids'].astype(int).tolist()));te=max(r[2] for r in v4.load_schedule()[1]['test'])+timedelta(seconds=1);ff,fd=airings(U,te);fxc=add_channel(fx,fk,pc['final'],ff,fd,'final',base,times);j,jy,jk,jf=load('features_final_jun14_U.npz');k,ky,kk,kf=load('features_final_jun28_U.npz');fj,dj=airings(CUTS['jun14'],U);fk2,dk=airings(CUTS['jun28'],U);jc=add_channel(j,jk,pc['jun14'],fj,dj,'jun14',base,times);kc=add_channel(k,kk,pc['jun28'],fk2,dk,'jun28',base,times);fm=fit('channel_hgb_final',np.vstack([jc,kc]),np.concatenate([jy,ky]),dig({'base':base,'final':[jf,kf]}),times);pcf=fm.predict_proba(fxc)[:,1];pbf=pickle.load((V4/'reranker_final_v4.pkl').open('rb')).predict_proba(fx)[:,1];fs=pcf if sel=='channel_hgb' else .75*pbf+.25*pcf;fc=pickle.load((V4/'aggregate_final_U.pkl').open('rb'))['data'][1];coldf={u for u in COHORT if not fc.get(u)}
  with (V4/'submission_temporal_v4_new200.csv').open(encoding='utf-8',newline='') as f:fall={r['user_id']:list(map(int,r['tv_show_id'].split())) for r in csv.DictReader(f)}
  final=f'submission_channel_v8_{sel.replace(".","p")}.csv';write_pred(OUT/final,top(fs,fk,coldf,fall),COHORT);_,sp,_=v4.load_schedule();can={r[3] for r in set(sp['test']) if r[3]>0};rs=list(csv.DictReader((OUT/final).open(encoding='utf-8',newline='')));checks={'rows':len(rs),'template_order':[r['user_id'] for r in rs]==COHORT,'complete_coverage':len(rs)==1260,'five_distinct_positive_ids':all(len(q:=r['tv_show_id'].split())==5 and len(set(q))==5 and all(int(v)>0 for v in q) for r in rs),'schedule_available':all(all(int(v) in can for v in r['tv_show_id'].split()) for r in rs),'strict_pre_U_fitting':True}
 out={'version':'channel_v8','late_verification':lateverify,'scores':scores,'late_scores':lates,'selected':sel,'qualified':qual,'schema':FEATURES,'fingerprint':base,'input_identities':ids,'final_submission':final,'submission_checks':checks,'runtime_total_sec':time.perf_counter()-t0,'timings':times,'late_diagnostic':'Overlapping diagnostic, not independent test.'};js(OUT/'summary.json',out);js(OUT/'timings.json',times);print(json.dumps({'scores':scores,'late':lates,'selected':sel,'submission':final,'checks':checks,'runtime':out['runtime_total_sec']},indent=2))
if __name__=='__main__':main()
