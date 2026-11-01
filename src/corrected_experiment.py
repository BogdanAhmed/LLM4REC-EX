"""Corrected v2 experiment: fixed interval join, cached interactions, and PPR sweep."""
from __future__ import annotations
import csv, gzip, json, random, time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, diags

from baseline import RAW, SCHEDULES, schedule_index, join_views, find_overlaps
from methods import read_cache, read_metadata, cbf

ROOT = Path(__file__).resolve().parents[1]; OLD = ROOT/"results/baseline"; OUT = ROOT/"results/corrected_v2"; OUT.mkdir(parents=True, exist_ok=True)

def users():
    with (RAW/"submission.csv").open(encoding="utf-8-sig", newline="") as f: return [r["user_id"] for r in csv.DictReader(f)]

def fast_rows(rows, vs, ve):
    rows = sorted(rows); starts = [r[0] for r in rows]; pref=[]; hi=float("-inf")
    for r in rows: hi=max(hi,r[1]); pref.append(hi)
    return [rows[i] for i in find_overlaps(rows,starts,pref,vs,ve)]

def brute_rows(rows, vs, ve): return [r for r in rows if r[0] < ve and r[1] > vs]

def randomized_checks():
    rng=random.Random(19); checks=[]
    four=[(0,1800,1),(1800,3600,2),(3600,5400,3),(5400,7200,4)]
    checks.append({"name":"two_hour_four_programmes","fast":len(fast_rows(four,0,7200)),"expected":4})
    checks.append({"name":"touching_boundaries","fast_ids":[r[2] for r in fast_rows(four,1800,3600)],"expected_ids":[2]})
    checks.append({"name":"id0_gap","fast_ids":[r[2] for r in fast_rows([(0,100,0),(100,200,5),(200,300,0)],50,250)],"expected_join_ids":[0,5,0]})
    for _ in range(200):
        rows=[]
        for i in range(rng.randint(1,20)):
            a=rng.randint(0,100); rows.append((a,a+rng.randint(1,40),rng.randint(0,8)))
        vs=rng.randint(0,100); ve=vs+rng.randint(1,50)
        f=[r for r in fast_rows(rows,vs,ve)]; b=sorted(brute_rows(rows,vs,ve))
        if f != b: raise AssertionError((rows,vs,ve,f,b))
    checks.append({"name":"randomized_fast_vs_bruteforce","cases":200,"passed":True})
    return checks

def combine_counts(a,b):
    out=defaultdict(Counter)
    for source in (a,b):
        for u,c in source.items(): out[u].update(c)
    return out

def combine_events(a,b):
    out=defaultdict(list)
    for source in (a,b):
        for u,e in source.items(): out[u].extend(e)
    return out

def pop_order(counts,candidates):
    c=Counter(p for vals in counts.values() for p in vals)
    return [p for p,_ in sorted(c.items(),key=lambda x:(-x[1],x[0])) if p in candidates],c

def fill(users_, ranked, pop, candidates):
    fallback=0; result={}; pop=[p for p in pop if p in candidates]+sorted(candidates-set(pop))
    for u in users_:
        own=list(dict.fromkeys([p for p in ranked.get(u,[]) if p in candidates])); fallback+=len(own)<5; ownset=set(own)
        result[u]=(own+[p for p in pop if p not in ownset])[:5]
    return result,fallback

def map5(pred,targets,cohort):
    vals=[]
    for u in cohort:
        rel=targets.get(u,set()); hits=score=0.0
        for rank,p in enumerate(pred[u],1):
            if p in rel: hits+=1; score+=hits/rank
        vals.append(score/min(5,len(rel)) if rel else 0.0)
    return sum(vals)/len(vals)

def build_graph(counts,candidates):
    ids=sorted(set(p for c in counts.values() for p in c)|set(candidates)); pos={p:i for i,p in enumerate(ids)}; us=sorted(counts)
    rows=[]; cols=[]; data=[]
    for i,u in enumerate(us):
        for p in counts[u]: rows.append(i); cols.append(pos[p]); data.append(1.0)
    B=csr_matrix((data,(rows,cols)),shape=(len(us),len(ids))); W=(B.T@B).tocsr(); W.setdiag(0); W.eliminate_zeros(); sums=np.asarray(W.sum(axis=1)).ravel(); P=diags(np.divide(1,sums,out=np.zeros_like(sums),where=sums>0))@W
    return {"ids":ids,"pos":pos,"P":P,"dangling":sums==0,"nodes":len(ids),"edges":int(W.nnz),"users":len(us)}

def infer(graph,counts,users_,candidates,alpha,tol=1e-8,max_iter=100):
    ids,pos,P,dangling=graph["ids"],graph["pos"],graph["P"],graph["dangling"]; ranked={}; converged=0; iterations=[]; final_delta=0.0; scores={}
    for u in users_:
        r=np.zeros(len(ids))
        for item,n in counts.get(u,{}).items(): r[pos[item]]=n
        if not r.sum(): continue
        r/=r.sum(); p=r.copy()
        for it in range(1,max_iter+1):
            q=alpha*(P.T.dot(p)+p[dangling].sum()*r)+(1-alpha)*r; delta=float(np.abs(q-p).sum()); p=q
            if delta<tol: converged+=1; break
        iterations.append(it); final_delta=max(final_delta,delta); scores[u]=p
        ranked[u]=[ids[j] for j in np.argsort(-p,kind="stable") if p[j]>0 and ids[j] in candidates]
    return ranked,{"alpha":alpha,"tol":tol,"max_iter":max_iter,"converged_users":converged,"users_run":len(iterations),"mean_iterations":sum(iterations)/len(iterations) if iterations else 0,"max_final_delta":final_delta},scores

def write_pred(path,pred,cohort):
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(["user_id","tv_show_id"])
        for u in cohort: w.writerow([u," ".join(map(str,pred[u]))])

def schedule_ids(path):
    out=set()
    with path.open(encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            if int(r["tv_show_id"]): out.add(int(r["tv_show_id"]))
    return out

def main():
    started=time.perf_counter(); fixture_checks=randomized_checks(); cohort=users(); case_summary=json.loads((OLD/"baseline_summary.json").read_text(encoding="utf-8")); case_users=[case_summary["case_users"][k]["user_id"] for k in ("heavy","light")]; model_users=list(dict.fromkeys(cohort+case_users))
    bounds={}
    for split in ("train","val","test"):
        _,_,_,b=schedule_index(SCHEDULES[split]); bounds[split] = {"start":b[0].isoformat(sep=" "),"end":datetime.fromtimestamp(b[1]).isoformat(sep=" ")}
    for split in ("train","val"):
        join_views(SCHEDULES[split],OUT/f"{split}_interactions.csv.gz",datetime.fromisoformat(bounds["train"]["end"]).timestamp()) if split=="train" else join_views(SCHEDULES[split],OUT/f"{split}_interactions.csv.gz")
    train,train_events,train_latest=read_cache(OUT/"train_interactions.csv.gz",.50); val80,_,_=read_cache(OUT/"val_interactions.csv.gz",.80); val50,val_events,val_latest=read_cache(OUT/"val_interactions.csv.gz",.50)
    targets={u:set(sorted(c,key=lambda p:(-c[p],p))[:5]) for u,c in val80.items()}; val_candidates=schedule_ids(SCHEDULES["val"]); test_candidates=schedule_ids(SCHEDULES["test"]); meta=read_metadata([SCHEDULES["train"],SCHEDULES["val"],SCHEDULES["test"]]); train_meta=read_metadata([SCHEDULES["train"]]); pop_val,_=pop_order(train,val_candidates); rows=[]; all_rank={}
    def add(name,ranked,settings,runtime):
        pred,fb=fill(model_users,ranked,pop_val,val_candidates); cohort_pred={u:pred[u] for u in cohort}; score=map5(cohort_pred,targets,cohort); all_rank[name]=(ranked,pred); write_pred(OUT/f"validation_{name}.csv",cohort_pred,cohort); rows.append({"method":name,"map5":score,"runtime_sec":runtime,"fallback_users":sum(len([p for p in ranked.get(u,[]) if p in val_candidates])<5 for u in cohort),"cold_start_users":sum(not train.get(u) for u in cohort),"settings":json.dumps(settings,sort_keys=True)})
    t=time.perf_counter(); freq={u:[p for p,_ in sorted(c.items(),key=lambda x:(-x[1],x[0]))] for u,c in train.items()}; add("frequency",freq,{"decay":"none"},time.perf_counter()-t)
    t=time.perf_counter(); cbf_rank,cbf_set=cbf(train,model_users,meta,val_candidates,set(train_meta)); add("cbf",cbf_rank,cbf_set,time.perf_counter()-t)
    graph=build_graph(train,val_candidates); inv_user=next((u for u in cohort if train.get(u)),None); inv1=infer(graph,train,[inv_user],val_candidates,.5)[2][inv_user]; inv2=infer(graph,train,model_users,val_candidates,.5)[2][inv_user]; invariance=bool(np.allclose(inv1,inv2,atol=1e-12))
    ppr_reference=np.array([1.,0.]); refP=np.array([[0.,1.],[1.,0.]])
    for _ in range(100): ppr_reference=.5*refP.T.dot(ppr_reference)+.5*np.array([1.,0.])
    assert np.allclose(ppr_reference,[2/3,1/3],atol=1e-8)
    ppr_ranks={}
    for alpha in (.15,.35,.50,.75):
        t=time.perf_counter(); rank,setts,_=infer(graph,train,model_users,val_candidates,alpha); setts.update({"graph_nodes":graph["nodes"],"graph_edges":graph["edges"],"graph_users":graph["users"]}); add(f"ppr_a{alpha:.2f}",rank,setts,time.perf_counter()-t); ppr_ranks[alpha]=rank
    base=json.loads((OLD/"baseline_summary.json").read_text(encoding="utf-8")); rows.insert(0,{"method":"popularity","map5":base["metrics"]["popularity"]["map5"],"runtime_sec":0.0,"fallback_users":len(cohort),"cold_start_users":sum(not train.get(u) for u in cohort),"settings":"cached baseline popularity"}); selected=max((r for r in rows if r["method"]!="popularity"),key=lambda r:(r["map5"],r["method"]))["method"]; selected_alpha=float(selected[-4:]) if selected.startswith("ppr_a") else None
    final_train=combine_counts(train,val50); final_events=combine_events(train_events,val_events); final_candidates=test_candidates; final_pop,_=pop_order(final_train,final_candidates); final_graph=build_graph(final_train,final_candidates)
    if selected.startswith("ppr_a"): final_rank,final_ppr_set,_=infer(final_graph,final_train,model_users,final_candidates,selected_alpha)
    elif selected=="cbf": final_rank,_=cbf(final_train,model_users,meta,final_candidates,set(train_meta))
    else: final_rank=freq
    final_pred,_=fill(cohort,final_rank,final_pop,final_candidates); final_name=f"submission_corrected_v2_{selected}.csv"; write_pred(OUT/final_name,final_pred,cohort)
    cases=[]
    for u in case_users:
        for name,(ranked,pred) in all_rank.items():
            items=pred[u]; cases.append({"user_id":u,"method":name,"items":[{"tv_show_id":p,"title":meta.get(p,{}).get("title"),"category":meta.get(p,{}).get("category"),"hit":p in targets.get(u,set())} for p in items]})
    submission_checks={"file":str(OUT/final_name),"rows":0,"template_rows":len(cohort),"user_order_matches":False,"five_distinct_positive_available_ids":True}
    with (OUT/final_name).open(encoding="utf-8",newline="") as f:
        rs=list(csv.DictReader(f)); submission_checks["rows"]=len(rs); submission_checks["user_order_matches"]=[r["user_id"] for r in rs]==cohort
        for r in rs:
            v=r["tv_show_id"].split(); submission_checks["five_distinct_positive_available_ids"] &= len(v)==5 and len(set(v))==5 and all(int(x)>0 and int(x) in final_candidates for x in v)
    old=json.loads((OLD/"baseline_summary.json").read_text(encoding="utf-8")); summary={"version":"corrected_v2","date_boundaries":bounds,"before_after":{"old_join_rows":old["join"],"old_issue":"last-airing-only backward scan","new_issue":"prefix-max end pruning plus full candidate-range inspection","corrected_cache_rows":{"train":sum(1 for _ in gzip.open(OUT/"train_interactions.csv.gz","rt",encoding="utf-8"))-1,"val":sum(1 for _ in gzip.open(OUT/"val_interactions.csv.gz","rt",encoding="utf-8"))-1},"fixes":["repaired interval candidate search","rebuilt both caches and validation targets","separated fixed graph construction from user inference","rebuilt final fit from corrected weeks 11–30 caches"]},"protocol":{"cohort":len(cohort),"empty_targets":sum(not targets.get(u) for u in cohort),"train_fraction":.50,"target_fraction":.80,"target_approximation":True,"repeats_allowed":True,"no_test_views_in_fit":True},"randomized_checks":fixture_checks,"ppr_graph_invariance":invariance,"ppr_reference":ppr_reference.tolist(),"graph":{"validation_nodes":graph["nodes"],"validation_edges":graph["edges"],"final_nodes":final_graph["nodes"],"final_edges":final_graph["edges"]},"selected_method":selected,"methods":rows,"case_users":case_users,"final_submission":final_name,"runtime_total_sec":time.perf_counter()-started}
    with (OUT/"comparison.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["method","map5","runtime_sec","fallback_users","cold_start_users","settings"]); w.writeheader(); w.writerows(rows)
    (OUT/"case_users.json").write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding="utf-8"); (OUT/"submission_checks.json").write_text(json.dumps(submission_checks,indent=2),encoding="utf-8"); (OUT/"corrected_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
