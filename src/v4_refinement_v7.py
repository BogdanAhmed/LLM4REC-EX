"""One bounded v4 refinement: cached LGBM ranker and 29-feature HGB."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import pickle
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import lightgbm
import numpy as np
from lightgbm import LGBMRanker
from sklearn.ensemble import HistGradientBoostingClassifier

import temporal_v4 as v4
from corrected_experiment import users, write_pred

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results/corrected_v2"
V4 = ROOT / "results/temporal_v4"
V5 = ROOT / "results/ensemble_v5"
OUT = ROOT / "results/v4_refinement_v7"
OUT.mkdir(parents=True, exist_ok=True)
COHORT = users(); T = datetime(2020, 5, 18); U = datetime(2020, 7, 27); LATE = datetime(2020, 6, 15)
RANKER_PARAMS = {"objective": "lambdarank", "n_estimators": 200, "learning_rate": .05,
                 "num_leaves": 15, "min_child_samples": 50, "reg_lambda": 5.0,
                 "lambdarank_truncation_level": 8, "seed": 42, "n_jobs": 1}
FEATURES_29 = v4.FEATURES_V4 + ["recent_rate_ratio", "exposure_adjusted_personal_rate", "completion_ratio", "normalized_cbf"]

def canonical(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
def digest(x): return hashlib.sha256(canonical(x).encode()).hexdigest()
def save_json(p, x): Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding="utf-8")
def identity(p):
    p=Path(p); h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return {"path":str(p.resolve()),"size":p.stat().st_size,"sha256":h.hexdigest()}

def load_matrix(name):
    z=np.load(V4/name,allow_pickle=False); keys=list(zip(z["users"].astype(str).tolist(),z["pids"].astype(int).tolist()))
    return z["X"],z.get("y"),keys,str(z["fingerprint"])

def targets_from_csv(path):
    with Path(path).open(encoding="utf-8",newline="") as f:
        return {r["user_id"]:set(map(int,r["targets"].split())) if r["targets"] else set() for r in csv.DictReader(f)}

def predictions_column(name):
    with (V4/"validation_predictions.csv").open(encoding="utf-8",newline="") as f:
        return {r["user_id"]:list(map(int,r[name].split())) for r in csv.DictReader(f)}

def metrics(pred, targets):
    aps={}
    for u in COHORT:
        rel=targets.get(u,set()); hit=score=0.0
        for rank,p in enumerate(pred[u],1):
            if p in rel: hit+=1;score+=hit/rank
        aps[u]=score/min(5,len(rel)) if rel else 0.0
    return sum(aps.values())/len(COHORT),aps

def groups(query_ids):
    out=[]; prev=None; n=0; seen=set(); contiguous=True
    for q in query_ids:
        if prev is None: prev=q
        if q != prev:
            if prev in seen: contiguous=False
            seen.add(prev);out.append(n);prev=q;n=0
        n+=1
    if prev is not None:
        if prev in seen: contiguous=False
        out.append(n)
    return np.asarray(out,dtype=np.int32),contiguous

def derive29(x, query_ids, name, base_fp, timings):
    path=OUT/f"{name}_x29.npz"; fp=digest({"base":base_fp,"name":name,"schema":FEATURES_29,"formula":"v7_derived4"})
    t=time.perf_counter()
    if path.exists():
        z=np.load(path,allow_pickle=False)
        if str(z["fingerprint"])==fp:
            timings[f"derive_{name}"]={"seconds":time.perf_counter()-t,"cache_hit":True};return z["X"]
    extra=np.zeros((len(x),4),dtype=np.float32)
    count50=x[:,1]; recent14=x[:,5]; cbf=x[:,10]; exposure=x[:,15]; count80=x[:,2]; personal=x[:,21]; days=x[:,24]
    positive=count50>0
    extra[positive,0]=recent14[positive]*days[positive]/(14.0*count50[positive])
    extra[:,1]=personal*exposure
    extra[positive,2]=count80[positive]/count50[positive]
    start=0
    for end in range(1,len(x)+1):
        if end==len(x) or query_ids[end]!=query_ids[start]:
            mx=float(cbf[start:end].max())
            if mx!=0: extra[start:end,3]=cbf[start:end]/mx
            start=end
    out=np.hstack([x,extra]).astype(np.float32,copy=False)
    np.savez_compressed(path,fingerprint=fp,X=out)
    save_json(path.with_suffix(".fingerprint.json"),{"fingerprint":fp,"features":FEATURES_29,"source_rows":len(x)})
    timings[f"derive_{name}"]={"seconds":time.perf_counter()-t,"cache_hit":False};return out

def full_orders(scores, keys):
    d=defaultdict(list)
    for (u,p),s in zip(keys,scores):d[u].append((p,float(s)))
    return {u:[p for p,_ in sorted(v,key=lambda z:(-z[1],z[0]))] for u,v in d.items()}

def top5(scores, keys, cold, fallback):
    order=full_orders(scores,keys)
    return {u:list(fallback[u]) if u in cold else order[u][:5] for u in COHORT},order

def rrf(order_a,order_b,keys,w,k=60):
    ra={u:{p:i+1 for i,p in enumerate(v)} for u,v in order_a.items()};rb={u:{p:i+1 for i,p in enumerate(v)} for u,v in order_b.items()}
    return np.asarray([w/(k+ra[u][p])+(1-w)/(k+rb[u][p]) for u,p in keys],dtype=np.float64)

def ranker_cached(name,x,y,qids,fp,timings):
    path=OUT/f"{name}.pkl"; side=OUT/f"{name}.fingerprint.json"; g,cont=groups(qids)
    info={"fingerprint":fp,"rows":len(y),"groups":len(g),"group_sum":int(g.sum()),"min_group":int(g.min()),"max_group":int(g.max()),"contiguous":cont,"params":RANKER_PARAMS,"lightgbm":lightgbm.__version__}
    if not cont or int(g.sum())!=len(y):raise RuntimeError("invalid ranker query grouping")
    t=time.perf_counter()
    if path.exists() and side.exists() and json.loads(side.read_text(encoding="utf-8")).get("fingerprint")==fp:
        timings[name]={"seconds":time.perf_counter()-t,"cache_hit":True};return pickle.load(path.open("rb")),info
    model=LGBMRanker(**RANKER_PARAMS);model.fit(x,y,group=g)
    pickle.dump(model,path.open("wb"),protocol=5);save_json(side,info)
    timings[name]={"seconds":time.perf_counter()-t,"cache_hit":False};return model,info

def hgb_cached(name,x,y,fp,timings):
    path=OUT/f"{name}.pkl"; side=OUT/f"{name}.fingerprint.json";t=time.perf_counter()
    if path.exists() and side.exists() and json.loads(side.read_text(encoding="utf-8")).get("fingerprint")==fp:
        timings[name]={"seconds":time.perf_counter()-t,"cache_hit":True};return pickle.load(path.open("rb"))
    m=HistGradientBoostingClassifier(**v4.HGB_PARAMS);m.fit(x,y);pickle.dump(m,path.open("wb"),protocol=5)
    save_json(side,{"fingerprint":fp,"features":FEATURES_29,"params":v4.HGB_PARAMS,"rows":len(y),"positives":int(y.sum())})
    timings[name]={"seconds":time.perf_counter()-t,"cache_hit":False};return m

def late_targets(base_fp,timings):
    path=OUT/"late_targets.csv"; fp=digest({"base":base_fp,"start":LATE.isoformat(),"end":U.isoformat(),"threshold":.8,"top_k":5})
    t=time.perf_counter()
    if path.exists() and (OUT/"late_targets.fingerprint.json").exists() and json.loads((OUT/"late_targets.fingerprint.json").read_text(encoding="utf-8")).get("fingerprint")==fp:
        timings["late_targets"]={"seconds":time.perf_counter()-t,"cache_hit":True};return targets_from_csv(path)
    c=defaultdict(Counter)
    # June 15 onward has no train/validation schedule overlap; this streams only corrected interaction cache labels.
    with gzip.open(V2/"val_interactions.csv.gz","rt",encoding="utf-8",newline="") as f:
        for r in csv.DictReader(f):
            end=datetime.fromisoformat(r["airing_end"])
            if LATE<=end<U and float(r["overlap_frac"])>=.8:c[r["user_id"]][int(r["tv_show_id"])]+=1
    targets={u:set(sorted(c.get(u,{}),key=lambda p:(-c[u][p],p))[:5]) for u in COHORT}
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["user_id","targets","target_count"]);w.writeheader()
        for u in COHORT:w.writerow({"user_id":u,"targets":" ".join(map(str,sorted(targets[u]))),"target_count":len(targets[u])})
    save_json(OUT/"late_targets.fingerprint.json",{"fingerprint":fp,"rule":"overlapping late-period diagnostic only"})
    timings["late_targets"]={"seconds":time.perf_counter()-t,"cache_hit":False};return targets

def write_table(path,rows):
    with Path(path).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    started=time.perf_counter();timings={}
    ids={n:identity(p) for n,p in {"v4_summary":V4/"summary.json","outer25":V4/"features_outer_new200.npz","apr05":V4/"features_long_apr05_T.npz","apr19":V4/"features_long_apr19_T.npz","jun14":V4/"features_final_jun14_U.npz","jun28":V4/"features_final_jun28_U.npz","v4_validation":V4/"reranker_validation_v4.pkl","v4_final":V4/"reranker_final_v4.pkl","v4_submission":V4/"submission_temporal_v4_new200.csv","v5_final_features":V5/"final_candidate_features.npz"}.items()}
    s4=json.loads((V4/"summary.json").read_text(encoding="utf-8"));base_fp=digest({"v4":s4["fingerprint"],"inputs":ids,"ranker":RANKER_PARAMS,"schema":FEATURES_29,"lightgbm":lightgbm.__version__})
    targets=targets_from_csv(V4/"validation_targets.csv");actual=predictions_column("new_reranker_200_clean");reproduced,_=metrics(actual,targets)
    if abs(reproduced-.40403130511463825)>1e-12:raise RuntimeError("v4 prediction reproduction mismatch")
    x,y,keys,outer_fp=load_matrix("features_outer_new200.npz");qa=[u for u,_ in keys];x29=derive29(x,qa,"outer",base_fp,timings)
    cold={u for u in COHORT if not pickle.load((V4/"aggregate_outer_T_U.pkl").open("rb"))["data"][1].get(u)};fallback=predictions_column("popularity_clean")
    v4m=pickle.load((V4/"reranker_validation_v4.pkl").open("rb")); p4=v4m.predict_proba(x)[:,1]
    pred={"v4":actual};scores={"v4":reproduced};aps={};_,aps["v4"]=metrics(actual,targets)
    ax,ay,ak,afp=load_matrix("features_long_apr05_T.npz");bx,by,bk,bfp=load_matrix("features_long_apr19_T.npz")
    aq=[f"apr05:{u}" for u,_ in ak];bq=[f"apr19:{u}" for u,_ in bk]
    ranker,group_info=ranker_cached("ranker_validation",np.vstack([ax,bx]),np.concatenate([ay,by]),aq+bq,digest({"base":base_fp,"inputs":[afp,bfp],"schema":"v4_25"}),timings)
    rank_margin=ranker.predict(x,raw_score=True);pred["ranker"],rank_order=top5(rank_margin,keys,cold,fallback);scores["ranker"],aps["ranker"]=metrics(pred["ranker"],targets)
    ax29=derive29(ax,aq,"apr05",base_fp,timings);bx29=derive29(bx,bq,"apr19",base_fp,timings)
    hgb29=hgb_cached("hgb29_validation",np.vstack([ax29,bx29]),np.concatenate([ay,by]),digest({"base":base_fp,"inputs":[afp,bfp],"schema":FEATURES_29}),timings)
    p29=hgb29.predict_proba(x29)[:,1];pred["hgb29"],hgb_order=top5(p29,keys,cold,fallback);scores["hgb29"],aps["hgb29"]=metrics(pred["hgb29"],targets)
    blend=.5*p29+.5*p4;pred["prob_hgb29_v4_0.50"],_=top5(blend,keys,cold,fallback);scores["prob_hgb29_v4_0.50"],aps["prob_hgb29_v4_0.50"]=metrics(pred["prob_hgb29_v4_0.50"],targets)
    rrf_score=rrf(hgb_order,rank_order,keys,.75);pred["rrf_v4_0.75_ranker"],_=top5(rrf_score,keys,cold,fallback);scores["rrf_v4_0.75_ranker"],aps["rrf_v4_0.75_ranker"]=metrics(pred["rrf_v4_0.75_ranker"],targets)
    late=late_targets(base_fp,timings);late_scores={n:metrics(p,late)[0] for n,p in pred.items()}
    comparison=[{"method":n,"map5":scores[n],"late_map5":late_scores[n],"full_delta":scores[n]-scores["v4"],"late_delta":late_scores[n]-late_scores["v4"]} for n in scores]
    qualified=[n for n in scores if n!="v4" and scores[n]>scores["v4"]+1e-12 and late_scores[n]>=late_scores["v4"]-1e-12]
    selected=max(qualified,key=lambda n:(scores[n],n)) if qualified else "retain_temporal_v4"
    # Export validation evidence before conditional final fit.
    write_table(OUT/"comparison.csv",comparison);write_table(OUT/"validation_targets.csv",[{"user_id":u,"targets":" ".join(map(str,sorted(targets[u]))),"target_count":len(targets[u])} for u in COHORT])
    write_table(OUT/"validation_predictions.csv",[{"user_id":u,**{n:" ".join(map(str,p[u])) for n,p in pred.items()}} for u in COHORT])
    rows=[]
    for u in COHORT:
        r={"user_id":u,"full_target_count":len(targets[u]),"late_target_count":len(late[u])}
        for n in aps:r[f"ap_{n}"]=aps[n][u];r[f"diff_{n}_minus_v4"]=aps[n][u]-aps["v4"][u]
        rows.append(r)
    write_table(OUT/"per_user_ap.csv",rows)
    with gzip.open(OUT/"validation_candidate_scores.csv.gz","wt",encoding="utf-8",newline="") as f:
        w=csv.writer(f);w.writerow(["user_id","tv_show_id","v4_probability","ranker_margin","hgb29_probability","prob_blend","rrf"])
        for i,(u,p) in enumerate(keys):w.writerow([u,p,p4[i],rank_margin[i],p29[i],blend[i],rrf_score[i]])
    final_submission=None;checks=None;final_info={}
    if selected!="retain_temporal_v4":
        z=np.load(V5/"final_candidate_features.npz",allow_pickle=False);fx=z["X"];fkeys=list(zip(z["users"].astype(str).tolist(),z["pids"].astype(int).tolist()));fq=[u for u,_ in fkeys]
        final_counts=pickle.load((V4/"aggregate_final_U.pkl").open("rb"))["data"][1];coldf={u for u in COHORT if not final_counts.get(u)}
        with (V4/"submission_temporal_v4_new200.csv").open(encoding="utf-8",newline="") as f:ffallback={r["user_id"]:list(map(int,r["tv_show_id"].split())) for r in csv.DictReader(f)}
        final4=pickle.load((V4/"reranker_final_v4.pkl").open("rb"));fp4=final4.predict_proba(fx)[:,1];final_hgb_order=full_orders(fp4,fkeys)
        if selected in ("hgb29","prob_hgb29_v4_0.50"):
            jx,jy,jk,jfp=load_matrix("features_final_jun14_U.npz");kx,ky,kk,kfp=load_matrix("features_final_jun28_U.npz");jq=[f"jun14:{u}" for u,_ in jk];kq=[f"jun28:{u}" for u,_ in kk]
            j29=derive29(jx,jq,"jun14",base_fp,timings);k29=derive29(kx,kq,"jun28",base_fp,timings)
            fhgb=hgb_cached("hgb29_final",np.vstack([j29,k29]),np.concatenate([jy,ky]),digest({"base":base_fp,"inputs":[jfp,kfp],"schema":FEATURES_29}),timings);f29=derive29(fx,fq,"final",base_fp,timings);fp29=fhgb.predict_proba(f29)[:,1]
            final_scores=fp29 if selected=="hgb29" else .5*fp29+.5*fp4;final_info={"new_component":"hgb29_final","blend":selected.startswith("prob")}
        elif selected in ("ranker","rrf_v4_0.75_ranker"):
            jx,jy,jk,jfp=load_matrix("features_final_jun14_U.npz");kx,ky,kk,kfp=load_matrix("features_final_jun28_U.npz");jq=[f"jun14:{u}" for u,_ in jk];kq=[f"jun28:{u}" for u,_ in kk]
            frank,_=ranker_cached("ranker_final",np.vstack([jx,kx]),np.concatenate([jy,ky]),jq+kq,digest({"base":base_fp,"inputs":[jfp,kfp],"schema":"v4_25"}),timings);fr=frank.predict(fx,raw_score=True)
            final_scores=fr if selected=="ranker" else rrf(final_hgb_order,full_orders(fr,fkeys),fkeys,.75);final_info={"new_component":"ranker_final","blend":selected.startswith("rrf")}
        else:raise RuntimeError("unknown selected method")
        final_pred,_=top5(final_scores,fkeys,coldf,ffallback);final_submission=f"submission_v4_refinement_v7_{selected.replace('.','p')}.csv";write_pred(OUT/final_submission,final_pred,COHORT)
        _,splits,_=v4.load_schedule();candidates={r[3] for r in set(splits["test"]) if r[3]>0}
        with (OUT/final_submission).open(encoding="utf-8",newline="") as f:srows=list(csv.DictReader(f))
        checks={"rows":len(srows),"template_order":[r["user_id"] for r in srows]==COHORT,"complete_coverage":len(srows)==len(COHORT),"five_distinct_positive_ids":all(len(q:=r["tv_show_id"].split())==5 and len(set(q))==5 and all(int(p)>0 for p in q) for r in srows),"schedule_available":all(all(int(p) in candidates for p in r["tv_show_id"].split()) for r in srows),"strict_pre_U_fitting":True}
    summary={"version":"v4_refinement_v7","v4_reproduced_map5":reproduced,"scores":scores,"late_scores":late_scores,"selected":selected,"qualified":qualified,"ranker_groups":group_info,"schema":FEATURES_29,"ranker_params":RANKER_PARAMS,"fingerprint":base_fp,"input_identities":ids,"libraries":{"lightgbm":lightgbm.__version__,"numpy":np.__version__},"final_submission":final_submission,"final_info":final_info,"submission_checks":checks,"runtime_total_sec":time.perf_counter()-started,"timings":timings,"late_diagnostic":"Overlapping 2020-06-15 <= airing_end < U diagnostic; not an independent test."}
    save_json(OUT/"summary.json",summary);save_json(OUT/"timings.json",timings);save_json(OUT/"ranker_group_checks.json",group_info)
    print(json.dumps({"reproduced":reproduced,"scores":scores,"late":late_scores,"selected":selected,"submission":final_submission,"checks":checks,"runtime":summary["runtime_total_sec"]},indent=2))

if __name__=="__main__":main()
