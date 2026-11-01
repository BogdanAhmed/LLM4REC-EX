"""Bounded v3 tuning on corrected SWEET.TV interaction caches."""
from __future__ import annotations
import csv, gzip, json, math, os, pickle, time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT","1")

import numpy as np
import scipy
import sklearn
from scipy.sparse import csr_matrix, diags, save_npz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from baseline import RAW, SCHEDULES
from corrected_experiment import build_graph, fill, map5, pop_order, schedule_ids, users, write_pred

ROOT=Path(__file__).resolve().parents[1]; V2=ROOT/"results/corrected_v2"; OLD=ROOT/"results/baseline"; OUT=ROOT/"results/tuning_v3"; OUT.mkdir(parents=True,exist_ok=True)
TRAIN_CACHE=V2/"train_interactions.csv.gz"; VAL_CACHE=V2/"val_interactions.csv.gz"
FEATURES=["count30","count50","count80","log_duration","relative_frequency","recent14","days_since_last","frequency_rank","ppr_score","ppr_rank","cbf_score","cbf_rank","log_popularity","history_exposure_day","future_exposure_day","exposure_ratio","category_affinity","source_frequency","source_ppr","source_cbf","source_popularity"]

def read_template(): return users()

def load_schedule():
    rows={}; meta={}
    for split,path in SCHEDULES.items():
        values=[]
        with path.open(encoding="utf-8-sig",newline="") as f:
            for r in csv.DictReader(f):
                pid=int(r["tv_show_id"])
                if not pid: continue
                start=datetime.strptime(r["start_time"],"%d.%m.%Y %H:%M:%S")
                values.append((start,pid))
                if pid not in meta:
                    fields=[r.get(k,"") for k in ("tv_show_title","tv_show_category","tv_show_genre_1","tv_show_genre_2","tv_show_genre_3","director","actors","year_of_production")]
                    meta[pid]={"title":r["tv_show_title"],"category":r["tv_show_category"],"doc":" ".join(x for x in fields if x)}
        rows[split]=values
    return rows,meta

def scan_interactions(paths,history_end,label_start=None,label_end=None):
    stats=defaultdict(dict); labels=defaultdict(Counter); recent_cut=datetime.min if history_end==datetime.min else history_end-timedelta(days=14)
    for path in paths:
        with gzip.open(path,"rt",encoding="utf-8",newline="") as f:
            for r in csv.DictReader(f):
                end=datetime.fromisoformat(r["airing_end"]); frac=float(r["overlap_frac"]); u=r["user_id"]; pid=int(r["tv_show_id"])
                if end < history_end and frac>=.30:
                    a=stats[u].setdefault(pid,[0,0,0,0.0,None,0])
                    a[0]+=1
                    if frac>=.50:
                        a[1]+=1; a[3]+=float(r["overlap_sec"]); a[4]=end if a[4] is None or end>a[4] else a[4]
                        if end>=recent_cut: a[5]+=1
                    if frac>=.80: a[2]+=1
                if label_start is not None and label_start<=end<label_end and frac>=.80: labels[u][pid]+=1
    counts=defaultdict(Counter)
    for u,items in stats.items():
        for pid,a in items.items():
            if a[1]: counts[u][pid]=a[1]
    targets={u:set(sorted(c,key=lambda p:(-c[p],p))[:5]) for u,c in labels.items()}
    return stats,counts,targets

def schedule_window(schedule_rows,start,end):
    c=Counter(pid for dt,pid in schedule_rows if start<=dt<end); days=max((end-start).total_seconds()/86400,1e-9)
    return set(c),c,days

def exposure(hist_counts,hist_days,future_counts,future_days,smoothing=.1,clip=(.5,2.0)):
    all_ids=set(hist_counts)|set(future_counts); ratio={}; hr={}; fr={}
    for p in all_ids:
        h=hist_counts[p]/hist_days; f=future_counts[p]/future_days; hr[p]=h; fr[p]=f; ratio[p]=min(clip[1],max(clip[0],(f+smoothing)/(h+smoothing)))
    return hr,fr,ratio

def top_ids(score_row,candidate_ids,n=100,positive=False):
    if len(candidate_ids)==0:return []
    valid=np.flatnonzero(score_row>0) if positive else np.arange(len(candidate_ids))
    if not len(valid): return []
    take=valid if len(valid)<=n else valid[np.argpartition(-score_row[valid],n-1)[:n]]
    return [int(candidate_ids[i]) for i in take[np.lexsort((candidate_ids[take],-score_row[take]))]]

def ppr_scores(graph,counts,pred_users,candidates,alpha,cache_path=None,tol=1e-8,max_iter=100):
    if cache_path is not None and cache_path.exists():
        z=np.load(cache_path)
        if "converged_users" in z and float(z["alpha"])==alpha and list(z["users"].astype(str))==list(pred_users) and np.array_equal(z["candidate_ids"],np.array(sorted(candidates),dtype=np.int64)):
            settings={"alpha":alpha,"users_run":int(z["users_run"]),"converged_users":int(z["converged_users"]),"mean_iterations":float(z["mean_iterations"]),"max_final_delta":float(z["max_final_delta"]),"cached":True}
            return z["candidate_ids"],z["scores"],settings
    cand=np.array(sorted(candidates),dtype=np.int64); cidx=np.array([graph["pos"][int(p)] for p in cand]); scores=np.zeros((len(pred_users),len(cand)),dtype=np.float32); iterations=[]; converged=0; max_delta=0.0
    for i,u in enumerate(pred_users):
        r=np.zeros(len(graph["ids"]),dtype=np.float64)
        for p,n in counts.get(u,{}).items(): r[graph["pos"][p]]=n
        if not r.sum(): continue
        r/=r.sum(); state=r.copy()
        for it in range(1,max_iter+1):
            q=alpha*(graph["P"].T.dot(state)+state[graph["dangling"]].sum()*r)+(1-alpha)*r; delta=float(np.abs(q-state).sum()); state=q
            if delta<tol: converged+=1; break
        iterations.append(it); max_delta=max(max_delta,delta); scores[i]=state[cidx]
    settings={"alpha":alpha,"users_run":len(iterations),"converged_users":converged,"mean_iterations":float(np.mean(iterations)) if iterations else 0.0,"max_final_delta":max_delta,"cached":False}
    if cache_path is not None: np.savez_compressed(cache_path,users=np.array(pred_users),candidate_ids=cand,scores=scores,alpha=alpha,users_run=settings["users_run"],converged_users=converged,mean_iterations=settings["mean_iterations"],max_final_delta=max_delta)
    return cand,scores,settings

def cbf_scores(stats,counts,pred_users,candidates,fit_ids,meta):
    fit=sorted(p for p in fit_ids if p in meta); cand=np.array(sorted(candidates),dtype=np.int64); vec=TfidfVectorizer(lowercase=True,ngram_range=(1,1),min_df=1,max_features=100000); X=vec.fit_transform([meta[p]["doc"] or "unknown" for p in fit]); pos={p:i for i,p in enumerate(fit)}
    rr=[];cc=[];dd=[]
    for i,u in enumerate(pred_users):
        for p,n in counts.get(u,{}).items():
            if p in pos: rr.append(i);cc.append(pos[p]);dd.append(n)
    U=csr_matrix((dd,(rr,cc)),shape=(len(pred_users),len(fit))); profiles=U@X; C=vec.transform([meta.get(int(p),{"doc":"unknown"})["doc"] or "unknown" for p in cand]); S=(profiles@C.T).tocsr()
    return cand,S,{"features":X.shape[1],"fit_items":len(fit)}

def ranked_from_matrix(users_,cand,scores,n=None,positive=False):
    out={}
    for i,u in enumerate(users_): out[u]=top_ids(scores[i],cand,n or len(cand),positive)
    return out

def sparse_row(S,i,ncols):
    row=S.getrow(i); out=np.zeros(ncols,dtype=np.float32); out[row.indices]=row.data.astype(np.float32); return out

def frequency_rank(counts,u,candidates): return [p for p,_ in sorted(((p,n) for p,n in counts.get(u,{}).items() if p in candidates),key=lambda x:(-x[1],x[0]))]

def rrf(a,b,users_,candidates,weight,k=60):
    out={}
    for u in users_:
        s=defaultdict(float)
        for rank,p in enumerate(a.get(u,[])[:200],1): s[p]+=weight/(k+rank)
        for rank,p in enumerate(b.get(u,[])[:200],1): s[p]+=(1-weight)/(k+rank)
        out[u]=[p for p,_ in sorted(s.items(),key=lambda x:(-x[1],x[0])) if p in candidates]
    return out

def build_sources(stats,counts,pred_users,candidates,meta,fit_ids,graph,alpha,pop,hist_exp,fut_exp,ratio,cache_tag=None,precomputed=None):
    cand,ppr,pset=precomputed if precomputed is not None else ppr_scores(graph,counts,pred_users,candidates,alpha,OUT/f"ppr_scores_{cache_tag}.npz" if cache_tag else None); ccand,cbf,cset=cbf_scores(stats,counts,pred_users,candidates,fit_ids,meta); assert np.array_equal(cand,ccand)
    pop_rank={p:i+1 for i,p in enumerate(pop)}; pop_counts=Counter(p for values in counts.values() for p in values); user_index={u:i for i,u in enumerate(pred_users)}; cat_aff={}
    for u,items in counts.items():
        c=Counter(); total=sum(items.values())
        for p,n in items.items(): c[meta.get(p,{}).get("category","")]+=n
        cat_aff[u]={k:v/total for k,v in c.items()} if total else {}
    return {"cand":cand,"ppr":ppr,"ppr_settings":pset,"cbf":cbf,"cbf_settings":cset,"pop":pop,"pop_rank":pop_rank,"pop_counts":pop_counts,"user_index":user_index,"cat_aff":cat_aff,"hist_exp":hist_exp,"fut_exp":fut_exp,"ratio":ratio}

def candidates_and_features(stats,counts,targets,pred_users,sources,meta,history_end,max_candidates=100):
    cand=sources["cand"]; rows=[]; keys=[]; labels=[]; candidate_lists={}; covered=total=0
    for u in pred_users:
        i=sources["user_index"][u]; prow=sources["ppr"][i]; crow=sparse_row(sources["cbf"],i,len(cand)); ppr_top=top_ids(prow,cand,50,True); cbf_top=top_ids(crow,cand,30,True); freq=frequency_rank(counts,u,set(map(int,cand)))[:30]; pop=sources["pop"][:30]
        chosen=[]
        for seq in (freq,ppr_top,cbf_top,pop):
            for p in seq:
                if p not in chosen: chosen.append(p)
        for p in top_ids(prow,cand,100,True):
            if len(chosen)>=max_candidates: break
            if p not in chosen: chosen.append(p)
        for p in sources["pop"]:
            if len(chosen)>=50: break
            if p not in chosen: chosen.append(p)
        chosen=chosen[:max_candidates]; candidate_lists[u]=chosen; rel=targets.get(u,set()); covered+=len(rel&set(chosen)); total+=len(rel)
        fmap={p:r+1 for r,p in enumerate(freq)}; pmap={p:r+1 for r,p in enumerate(top_ids(prow,cand,100,True))}; cmap={p:r+1 for r,p in enumerate(top_ids(crow,cand,100,True))}; cpos={int(p):j for j,p in enumerate(cand)}; user_total=sum(counts.get(u,{}).values()) or 1
        for p in chosen:
            a=stats.get(u,{}).get(p,[0,0,0,0.0,None,0]); last=a[4]; category=meta.get(p,{}).get("category","")
            feat=[a[0],a[1],a[2],math.log1p(a[3]),a[1]/user_total,a[5],(history_end-last).total_seconds()/86400 if last else 999.0,fmap.get(p,999),float(prow[cpos[p]]),pmap.get(p,999),float(crow[cpos[p]]),cmap.get(p,999),math.log1p(sources["pop_counts"].get(p,0)),sources["hist_exp"].get(p,0),sources["fut_exp"].get(p,0),sources["ratio"].get(p,1),sources["cat_aff"].get(u,{}).get(category,0),float(p in freq),float(p in ppr_top),float(p in cbf_top),float(p in pop)]
            rows.append(feat); keys.append((u,p)); labels.append(int(p in rel))
    coverage=covered/total if total else 0.0
    return np.asarray(rows,dtype=np.float32),np.asarray(labels,dtype=np.uint8),keys,candidate_lists,coverage

def make_snapshot(name,history_end,label_start,label_end,cache_paths,schedule_rows,meta,alpha):
    cache=OUT/f"snapshot_{name}.npz"
    if cache.exists():
        z=np.load(cache,allow_pickle=True); info=json.loads(str(z["info"])); info["cached"]=True; return z["X"],z["y"],info
    t=time.perf_counter(); stats,counts,targets=scan_interactions(cache_paths,history_end,label_start,label_end); pred_users=sorted(set(targets)&set(counts)); candidates,future_counts,future_days=schedule_window(schedule_rows,label_start,label_end); fit_ids,hist_counts,hist_days=schedule_window(schedule_rows,min(dt for dt,_ in schedule_rows),history_end); hr,fr,ratio=exposure(hist_counts,hist_days,future_counts,future_days); pop,_=pop_order(counts,candidates); graph=build_graph(counts,candidates); sources=build_sources(stats,counts,pred_users,candidates,meta,fit_ids,graph,alpha,pop,hr,fr,ratio,None); X,y,keys,_,coverage=candidates_and_features(stats,counts,targets,pred_users,sources,meta,history_end)
    info={"name":name,"history_end":history_end.isoformat(sep=" "),"label_start":label_start.isoformat(sep=" "),"label_end":label_end.isoformat(sep=" "),"users":len(pred_users),"rows":len(y),"positives":int(y.sum()),"candidate_coverage":coverage,"runtime_sec":time.perf_counter()-t,"cached":False}; np.savez_compressed(cache,X=X,y=y,info=json.dumps(info)); return X,y,info

def evaluate_ranking(name,ranked,output_users,metric_users,targets,pop,candidates,rows,settings,runtime):
    pred,_=fill(output_users,ranked,pop,candidates); score=map5(pred,targets,metric_users); fallback=sum(len([p for p in ranked.get(u,[]) if p in candidates])<5 for u in metric_users); rows.append({"method":name,"map5":score,"runtime_sec":runtime,"fallback_users":fallback,"settings":json.dumps(settings,sort_keys=True)}); return pred,score

def rank_model(model,X,keys,users_):
    prob=model.predict_proba(X)[:,1]; grouped=defaultdict(list)
    for (u,p),s in zip(keys,prob): grouped[u].append((p,float(s)))
    return {u:[p for p,_ in sorted(grouped.get(u,[]),key=lambda x:(-x[1],x[0]))] for u in users_},prob

def main():
    started=time.perf_counter(); schedule,meta=load_schedule(); cohort=read_template(); case_info=json.loads((V2/"corrected_summary.json").read_text(encoding="utf-8")); case_users=case_info["case_users"]; model_users=list(dict.fromkeys(cohort+case_users)); bounds=case_info["date_boundaries"]; train_start=datetime.fromisoformat(bounds["train"]["start"]); train_end=datetime.fromisoformat(bounds["train"]["end"]); val_start=datetime.fromisoformat(bounds["val"]["start"]); val_end=datetime.fromisoformat(bounds["val"]["end"])
    stats,counts,_=scan_interactions([TRAIN_CACHE],train_end); _,_,targets=scan_interactions([VAL_CACHE],datetime.min,val_start,val_end); val_candidates,val_sched,val_days=schedule_window(schedule["val"],val_start,val_end); fit_ids,hist_sched,hist_days=schedule_window(schedule["train"],train_start,train_end); hr,fr,ratio=exposure(hist_sched,hist_days,val_sched,val_days); pop,pop_counts=pop_order(counts,val_candidates); graph=build_graph(counts,val_candidates); save_npz(OUT/"outer_transition.npz",graph["P"]); (OUT/"outer_graph_ids.json").write_text(json.dumps(graph["ids"]),encoding="utf-8"); rows=[]; predictions={}; score_mats={}; ppr_rankings={}
    t=time.perf_counter(); pop_pred,_=fill(model_users,{},pop,val_candidates); pop_score=map5(pop_pred,targets,cohort); rows.append({"method":"popularity_corrected","map5":pop_score,"runtime_sec":time.perf_counter()-t,"fallback_users":len(cohort),"settings":json.dumps({"recomputed_from":"corrected_v2 train cache"})}); predictions["popularity_corrected"]=pop_pred
    freq={u:frequency_rank(counts,u,val_candidates) for u in model_users}; pred,score=evaluate_ranking("frequency",freq,model_users,cohort,targets,pop,val_candidates,rows,{"threshold":.5},0.0); predictions["frequency"]=pred
    ccand,CS,cset=cbf_scores(stats,counts,model_users,val_candidates,fit_ids,meta); cbf_rank={u:top_ids(sparse_row(CS,i,len(ccand)),ccand,len(ccand),True) for i,u in enumerate(model_users)}; pred,score=evaluate_ranking("cbf",cbf_rank,model_users,cohort,targets,pop,val_candidates,rows,cset,0.0); predictions["cbf"]=pred
    for alpha in (.75,.85,.90,.95):
        t=time.perf_counter(); cand,S,setting=ppr_scores(graph,counts,model_users,val_candidates,alpha,OUT/f"ppr_outer_a{alpha:.2f}.npz"); score_mats[alpha]=S; rank={u:top_ids(S[i],cand,len(cand),True) for i,u in enumerate(model_users)}; ppr_rankings[alpha]=rank; pred,score=evaluate_ranking(f"ppr_a{alpha:.2f}",rank,model_users,cohort,targets,pop,val_candidates,rows,setting,time.perf_counter()-t); predictions[f"ppr_a{alpha:.2f}"]=pred
    best_alpha=max((.75,.85,.90,.95),key=lambda a:next(r["map5"] for r in rows if r["method"]==f"ppr_a{a:.2f}")); base_rank=ppr_rankings[best_alpha]; base_scores=score_mats[best_alpha]
    for w in (.25,.50,.75):
        t=time.perf_counter(); rank=rrf(freq,base_rank,model_users,val_candidates,w); pred,score=evaluate_ranking(f"fusion_freq{w:.2f}_ppr",rank,model_users,cohort,targets,pop,val_candidates,rows,{"frequency_weight":w,"ppr_alpha":best_alpha,"rrf_k":60},time.perf_counter()-t); predictions[f"fusion_freq{w:.2f}_ppr"]=pred
    cand=np.array(sorted(val_candidates),dtype=np.int64); ratio_vec=np.array([ratio.get(int(p),1.0) for p in cand],dtype=np.float32)
    for strength in (.25,.50,1.0):
        t=time.perf_counter(); adjusted=base_scores*np.power(ratio_vec[None,:],strength); rank={u:top_ids(adjusted[i],cand,len(cand),True) for i,u in enumerate(model_users)}; name=f"ppr_exposure_{strength:.2f}"; pred,score=evaluate_ranking(name,rank,model_users,cohort,targets,pop,val_candidates,rows,{"ppr_alpha":best_alpha,"strength":strength,"smoothing_airings_per_day":.1,"clip":[.5,2.0]},time.perf_counter()-t); predictions[name]=pred
    outer_sources=build_sources(stats,counts,model_users,val_candidates,meta,fit_ids,graph,best_alpha,pop,hr,fr,ratio,"outer_best",(cand,base_scores,{"alpha":best_alpha,"reused_outer_scores":True})); Xouter,youter,outer_keys,outer_candidates,_=candidates_and_features(stats,counts,targets,model_users,outer_sources,meta,train_end); outer_total=sum(len(targets.get(u,set())) for u in cohort); outer_covered=sum(len(targets.get(u,set())&set(outer_candidates[u])) for u in cohort); outer_coverage=outer_covered/outer_total if outer_total else 0.0
    snapshots=[("s1",datetime(2020,4,5),datetime(2020,4,5),datetime(2020,4,19)),("s2",datetime(2020,4,19),datetime(2020,4,19),datetime(2020,5,3))]; Xparts=[];yparts=[];snap_info=[]
    for name,h,l0,l1 in snapshots:
        X,y,info=make_snapshot(name,h,l0,l1,[TRAIN_CACHE],schedule["train"],meta,best_alpha); Xparts.append(X);yparts.append(y);snap_info.append(info)
    Xtrain=np.vstack(Xparts); ytrain=np.concatenate(yparts); model_path=OUT/"reranker_validation.pkl"; t=time.perf_counter()
    if model_path.exists(): model=pickle.load(model_path.open("rb")); model_cached=True
    else: model=HistGradientBoostingClassifier(max_iter=120,learning_rate=.08,max_leaf_nodes=31,min_samples_leaf=30,l2_regularization=1.0,class_weight="balanced",random_state=42); model.fit(Xtrain,ytrain); pickle.dump(model,model_path.open("wb")); model_cached=False
    fit_runtime=time.perf_counter()-t; rerank,prob=rank_model(model,Xouter,outer_keys,model_users)
    for u in model_users:
        if not counts.get(u): rerank[u]=[]
    pred,rerank_score=evaluate_ranking("reranker",rerank,model_users,cohort,targets,pop,val_candidates,rows,{"model":"HistGradientBoostingClassifier","sklearn":sklearn.__version__,"training_snapshots":2,"candidate_limit":100,"model_cached":model_cached},fit_runtime); predictions["reranker"]=pred
    for w in (.50,.75):
        t=time.perf_counter(); rank=rrf(rerank,base_rank,model_users,val_candidates,w); name=f"reranker_ppr_{w:.2f}"; pred,score=evaluate_ranking(name,rank,model_users,cohort,targets,pop,val_candidates,rows,{"reranker_weight":w,"ppr_alpha":best_alpha,"rrf_k":60},time.perf_counter()-t); predictions[name]=pred
    baseline=.37907341269841294; selected=max(rows,key=lambda r:(r["map5"],r["method"])); improved=selected["map5"]>baseline+1e-12
    # Refit only after freezing the outer-validation winner.
    final_submission=None; submission_checks=None; final_refit={}
    if improved:
        test_start=datetime.fromisoformat(bounds["test"]["start"]); test_end=datetime.fromisoformat(bounds["test"]["end"]); all_stats,all_counts,_=scan_interactions([TRAIN_CACHE,VAL_CACHE],test_start); test_candidates,test_sched,test_days=schedule_window(schedule["test"],test_start,test_end); all_start=min(train_start,val_start); all_fit,all_hist_sched,all_hist_days=schedule_window(schedule["train"]+schedule["val"],all_start,test_start); ahr,afr,aratio=exposure(all_hist_sched,all_hist_days,test_sched,test_days); final_pop,_=pop_order(all_counts,test_candidates); final_graph=build_graph(all_counts,test_candidates); save_npz(OUT/"final_transition.npz",final_graph["P"])
        if selected["method"].startswith("reranker"):
            X3,y3,info3=make_snapshot("s3_refit",datetime(2020,5,18,12),datetime(2020,5,18,12),datetime(2020,6,15),[TRAIN_CACHE,VAL_CACHE],schedule["train"]+schedule["val"],meta,best_alpha); final_model_path=OUT/"reranker_final.pkl"
            if final_model_path.exists(): final_model=pickle.load(final_model_path.open("rb")); final_cached=True
            else: final_model=HistGradientBoostingClassifier(max_iter=120,learning_rate=.08,max_leaf_nodes=31,min_samples_leaf=30,l2_regularization=1.0,class_weight="balanced",random_state=42); final_model.fit(np.vstack([Xtrain,X3]),np.concatenate([ytrain,y3])); pickle.dump(final_model,final_model_path.open("wb")); final_cached=False
            final_sources=build_sources(all_stats,all_counts,cohort,test_candidates,meta,all_fit,final_graph,best_alpha,final_pop,ahr,afr,aratio,"final"); Xtest,_,test_keys,_,_=candidates_and_features(all_stats,all_counts,{},cohort,final_sources,meta,test_start); final_rank,_=rank_model(final_model,Xtest,test_keys,cohort)
            for u in cohort:
                if not all_counts.get(u): final_rank[u]=[]
            if selected["method"].startswith("reranker_ppr"):
                weight=float(selected["method"].rsplit("_",1)[1]); ppr_final={u:top_ids(final_sources["ppr"][i],final_sources["cand"],len(final_sources["cand"]),True) for i,u in enumerate(cohort)}; final_rank=rrf(final_rank,ppr_final,cohort,test_candidates,weight)
            final_refit={"added_snapshot":info3,"model":"HistGradientBoostingClassifier","model_cached":final_cached}
        else:
            alpha=best_alpha; fcand,FS,_=ppr_scores(final_graph,all_counts,cohort,test_candidates,alpha,OUT/"ppr_final.npz"); ppr_final={u:top_ids(FS[i],fcand,len(fcand),True) for i,u in enumerate(cohort)}
            if selected["method"].startswith("fusion_"): weight=float(selected["method"].split("freq")[1].split("_")[0]); ffreq={u:frequency_rank(all_counts,u,test_candidates) for u in cohort}; final_rank=rrf(ffreq,ppr_final,cohort,test_candidates,weight)
            elif selected["method"].startswith("ppr_exposure_"): strength=float(selected["method"].rsplit("_",1)[1]); rv=np.array([aratio.get(int(p),1.0) for p in fcand]); adj=FS*np.power(rv[None,:],strength); final_rank={u:top_ids(adj[i],fcand,len(fcand),True) for i,u in enumerate(cohort)}
            else: final_rank=ppr_final
        final_pred,_=fill(cohort,final_rank,final_pop,test_candidates); safe=selected["method"].replace(".","p"); final_submission=f"submission_tuning_v3_{safe}.csv"; write_pred(OUT/final_submission,final_pred,cohort); rs=list(csv.DictReader((OUT/final_submission).open(encoding="utf-8",newline=""))); submission_checks={"rows":len(rs),"template_order":[r["user_id"] for r in rs]==cohort,"complete":len(rs)==len(cohort),"five_distinct_positive_available":all(len((v:=r["tv_show_id"].split()))==5 and len(set(v))==5 and all(int(x)>0 and int(x) in test_candidates for x in v) for r in rs),"no_test_views_in_fit":True}
    cases=[]
    for u in case_users:
        for name,pred in predictions.items(): cases.append({"user_id":u,"method":name,"items":[{"tv_show_id":p,"title":meta.get(p,{}).get("title"),"category":meta.get(p,{}).get("category"),"hit":p in targets.get(u,set())} for p in pred[u]]})
    with (OUT/"comparison.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["method","map5","runtime_sec","fallback_users","settings"]);w.writeheader();w.writerows(rows)
    summary={"version":"tuning_v3","libraries":{"python":"3.11","numpy":np.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__},"protocol":{"cohort":len(cohort),"empty_targets":sum(not targets.get(u) for u in cohort),"target_definition_unchanged":True,"repeats_allowed":True,"candidate_availability":"positive IDs in prediction schedule","fallback":"training-only corrected popularity"},"boundaries":{"train_start":train_start.isoformat(sep=" "),"train_end":train_end.isoformat(sep=" "),"validation_start":val_start.isoformat(sep=" "),"validation_end":val_end.isoformat(sep=" ")},"popularity_fix":{"old_inherited_map5":.14163910934744253,"corrected_recomputed_map5":pop_score},"ppr_graph":{"users":graph["users"],"nodes":graph["nodes"],"edges":graph["edges"]},"best_ppr_alpha":best_alpha,"snapshots":snap_info,"candidate_coverage":{"outer_validation":outer_coverage,"snapshots":{x["name"]:x["candidate_coverage"] for x in snap_info}},"configurations_added":12,"comparison":rows,"selected":selected,"improved_over_corrected_v2":improved,"final_refit":final_refit,"final_submission":final_submission,"submission_checks":submission_checks,"runtime_total_sec":time.perf_counter()-started}
    summary["unsuccessful_runs"]=[{"stage":"validation reranker fit","failure":"sandbox denied scikit-learn thread-pool handle creation","correction":"OMP_NUM_THREADS=1 and LOKY_MAX_CPU_COUNT=1"},{"stage":"case-user export","failure":"heavy case user is outside the 1,260-user cohort","correction":"separate diagnostic prediction users from metric users"},{"stage":"cold-start audit","failure":"zero PPR scores were ranked as candidates","correction":"filter zero scores and apply common corrected-popularity fallback"}]
    summary["runtime_note"]="Final per-method PPR/model times are cache-load plus ranking times where settings say cached=true; snapshot construction times are original measured runtimes. Failed-attempt wall time was not retained."
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); (OUT/"case_users.json").write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding="utf-8"); (OUT/"candidate_coverage.json").write_text(json.dumps(summary["candidate_coverage"],indent=2),encoding="utf-8"); (OUT/"submission_checks.json").write_text(json.dumps(submission_checks,indent=2),encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
