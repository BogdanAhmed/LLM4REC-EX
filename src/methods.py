"""Cached-interaction evaluation for CBF, PPR, recency and rank fusion."""
from __future__ import annotations
import csv, gzip, json, math, time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]; RAW = ROOT/"data/raw"; BASE = ROOT/"results/baseline"; OUT = ROOT/"results/methods"; OUT.mkdir(parents=True, exist_ok=True)
FMT, SCHFMT = "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M:%S"

def read_template():
    with (RAW/"submission.csv").open(encoding="utf-8-sig", newline="") as f: return [r["user_id"] for r in csv.DictReader(f)]

def read_cache(path, fraction):
    counts, events, latest = defaultdict(Counter), defaultdict(list), None
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if float(r["overlap_frac"]) < fraction: continue
            pid, end = int(r["tv_show_id"]), datetime.fromisoformat(r["airing_end"])
            counts[r["user_id"]][pid] += 1; events[r["user_id"]].append((pid, end)); latest = end if latest is None or end > latest else latest
    return counts, events, latest

def read_metadata(paths):
    meta = {}
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                pid = int(r["tv_show_id"])
                if pid and pid not in meta:
                    fields = [r.get(k, "") for k in ("tv_show_title","tv_show_category","tv_show_genre_1","tv_show_genre_2","tv_show_genre_3","director","actors","year_of_production")]
                    meta[pid] = {"title":r["tv_show_title"], "category":r["tv_show_category"], "doc":" ".join(x for x in fields if x)}
    return meta

def ordered_popularity(train, candidates):
    c = Counter(p for values in train.values() for p in values)
    return [p for p, _ in sorted(c.items(), key=lambda x: (-x[1], x[0])) if p in candidates], c

def fill(users, ranked, popularity, candidates):
    result, fallback = {}, 0; pop = [p for p in popularity if p in candidates] + sorted(candidates-set(popularity))
    for u in users:
        own = [p for p in ranked.get(u, []) if p in candidates]
        own = list(dict.fromkeys(own)); own_set = set(own); fallback += len(own) < 5
        result[u] = (own + [p for p in pop if p not in own_set])[:5]
    return result, fallback

def map5(pred, targets, users):
    vals = []
    for u in users:
        rel, hits, score = targets.get(u, set()), 0, 0.0
        for rank, pid in enumerate(pred[u], 1):
            if pid in rel: hits += 1; score += hits/rank
        vals.append(score/min(5, len(rel)) if rel else 0.0)
    return sum(vals)/len(vals)

def save_pred(path, pred, users):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["user_id","tv_show_id"])
        for u in users: w.writerow([u, " ".join(map(str, pred[u]))])

def rank_scores(scores, candidates):
    return {u:[p for p, s in sorted(v.items(), key=lambda x: (-x[1], x[0])) if s > 0 and p in candidates] for u, v in scores.items()}

def recency_scores(events, cutoff, half_life):
    out = defaultdict(Counter); ln2 = math.log(2)
    for u, rows in events.items():
        for pid, end in rows: out[u][pid] += math.exp(-ln2*max(0, (cutoff-end).total_seconds()/86400)/half_life)
    return out

def cbf(train, users, meta, candidates, fit_ids):
    train_ids = sorted(p for p in fit_ids if p in meta); docs = [meta[p]["doc"] or "unknown" for p in train_ids]
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1,1), min_df=1, max_features=100000); X = vec.fit_transform(docs); pos = {p:i for i,p in enumerate(train_ids)}
    rows, cols, data = [], [], []
    for i, u in enumerate(users):
        for p, n in train.get(u, {}).items():
            if p in pos: rows.append(i); cols.append(pos[p]); data.append(n)
    U = csr_matrix((data,(rows,cols)), shape=(len(users),len(train_ids))); profiles = U @ X
    cand_ids = sorted(candidates); C = vec.transform([meta.get(p, {"doc":"unknown"})["doc"] or "unknown" for p in cand_ids]); scores = profiles @ C.T
    ranked = {}
    for i, u in enumerate(users):
        vals = np.asarray(scores.getrow(i).toarray()).ravel(); ranked[u] = [cand_ids[j] for j in np.argsort(-vals, kind="stable") if vals[j] > 0]
    return ranked, {"features":X.shape[1],"train_items":len(train_ids)}

def ppr(train, users, candidates, alpha=.5, tol=1e-8, max_iter=100):
    ids = sorted(set(p for c in train.values() for p in c) | candidates); pos = {p:i for i,p in enumerate(ids)}
    rows, cols, data = [], [], []
    for i, u in enumerate(users):
        for p in train.get(u, {}): rows.append(i); cols.append(pos[p]); data.append(1.0)
    B = csr_matrix((data,(rows,cols)), shape=(len(users),len(ids))); W = (B.T @ B).tocsr(); W.setdiag(0); W.eliminate_zeros(); sums = np.asarray(W.sum(axis=1)).ravel(); P = diags(np.divide(1, sums, out=np.zeros_like(sums), where=sums>0)) @ W; dangling = sums == 0
    ranked, converged, iterations, max_delta = {}, 0, [], 0.0
    for i, u in enumerate(users):
        r = np.zeros(len(ids));
        for p, n in train.get(u, {}).items(): r[pos[p]] = n
        if r.sum() == 0: continue
        r /= r.sum(); p = r.copy()
        for it in range(1, max_iter+1):
            q = alpha*(P.T.dot(p) + p[dangling].sum()*r) + (1-alpha)*r; delta = float(np.abs(q-p).sum()); p = q
            if delta < tol: converged += 1; break
        iterations.append(it); max_delta = max(max_delta, delta); ranked[u] = [ids[j] for j in np.argsort(-p, kind="stable") if p[j] > 0 and ids[j] in candidates]
    return ranked, {"alpha":alpha,"tol":tol,"max_iter":max_iter,"converged_users":converged,"users_run":len(iterations),"mean_iterations":sum(iterations)/len(iterations) if iterations else 0,"max_final_delta":max_delta,"nodes":len(ids),"edges":int(W.nnz)}

def fuse(a, b, candidates, weight=.5, k=60):
    out = {}
    for u in set(a)|set(b):
        score = defaultdict(float)
        for rank, p in enumerate(a.get(u, []), 1): score[p] += weight/(k+rank)
        for rank, p in enumerate(b.get(u, []), 1): score[p] += (1-weight)/(k+rank)
        out[u] = [p for p, _ in sorted(score.items(), key=lambda x: (-x[1], x[0])) if p in candidates]
    return out

def main():
    started = time.perf_counter(); users = read_template(); bs=json.loads((BASE/"baseline_summary.json").read_text(encoding="utf-8")); case_users=[bs["case_users"][k]["user_id"] for k in ("heavy","light")]; model_users=list(dict.fromkeys(users+case_users)); train, train_events, cutoff = read_cache(BASE/"train_interactions.csv.gz", .50); val_raw, _, _ = read_cache(BASE/"val_interactions.csv.gz", .80)
    targets = {u:set(sorted(c, key=lambda x:(-c[x],x))[:5]) for u,c in val_raw.items()}; meta = read_metadata([RAW/"export_arh_11-20-final.csv",RAW/"export_arh_21-30-final.csv",RAW/"export_arh_31-42-final.csv"]); val_candidates = {p for p in meta if p}; test_candidates = val_candidates
    # Candidate sets are restricted by schedule windows, not by training history.
    def ids(path):
        out=set()
        with path.open(encoding="utf-8-sig",newline="") as f:
            for r in csv.DictReader(f):
                if int(r["tv_show_id"]): out.add(int(r["tv_show_id"]))
        return out
    val_candidates, test_candidates = ids(RAW/"export_arh_21-30-final.csv"), ids(RAW/"export_arh_31-42-final.csv")
    pop_val, counts = ordered_popularity(train, val_candidates); rows=[]; val_rank={}; test_rank={}
    methods = {}
    def add(name, ranked, settings, runtime):
        all_pred, _ = fill(model_users, ranked, pop_val, val_candidates); pv={u:all_pred[u] for u in users}; fbv=sum(len([p for p in ranked.get(u,[]) if p in val_candidates])<5 for u in users); score=map5(pv,targets,users); methods[name]={"validation":pv,"case_predictions":all_pred,"score":score,"fallback_users":fbv,"settings":settings,"runtime_sec":runtime}; save_pred(OUT/f"validation_{name}.csv",pv,users); rows.append({"method":name,"map5":score,"runtime_sec":runtime,"fallback_users":fbv,"settings":json.dumps(settings,sort_keys=True)})
    t=time.perf_counter(); add("frequency", {u:list(sorted(c,key=lambda p:(-c[p],p))) for u,c in train.items()}, {"decay":"none"}, time.perf_counter()-t)
    for h in (14,30,60):
        t=time.perf_counter(); add(f"recency_{h}d", rank_scores(recency_scores(train_events,cutoff,h),val_candidates), {"half_life_days":h}, time.perf_counter()-t)
    train_meta = read_metadata([RAW/"export_arh_11-20-final.csv"])
    t=time.perf_counter(); cbf_rank, cbf_settings=cbf(train,model_users,meta,val_candidates,set(train_meta)); add("cbf",cbf_rank,cbf_settings,time.perf_counter()-t)
    t=time.perf_counter(); ppr_rank, ppr_settings=ppr(train,model_users,val_candidates); add("ppr",ppr_rank,ppr_settings,time.perf_counter()-t)
    for name, other in (("fusion_cbf",cbf_rank),("fusion_ppr",ppr_rank)):
        t=time.perf_counter(); add(name,fuse({u:list(sorted(c,key=lambda p:(-c[p],p))) for u,c in train.items()},other,val_candidates),{"rule":"equal-weight reciprocal-rank fusion","k":60,"components":["frequency",name[7:]]},time.perf_counter()-t)
    base_summary=json.loads((BASE/"baseline_summary.json").read_text(encoding="utf-8")); rows.insert(0,{"method":"popularity","map5":base_summary["metrics"]["popularity"]["map5"],"runtime_sec":0.0,"fallback_users":len(users),"settings":"{\"source\":\"cached baseline\"}"})
    selected=min((r for r in rows if r["method"]!="popularity"),key=lambda r:(-r["map5"],r["method"]))["method"]
    freq_rank={u:list(sorted(c,key=lambda p:(-c[p],p))) for u,c in train.items()}; fusion_cbf=fuse(freq_rank,cbf_rank,val_candidates); fusion_ppr=fuse(freq_rank,ppr_rank,val_candidates)
    if selected=="frequency": selected_rank=freq_rank
    elif selected=="cbf": selected_rank=cbf_rank
    elif selected=="ppr": selected_rank=ppr_rank
    elif selected=="fusion_cbf": selected_rank=fusion_cbf
    elif selected=="fusion_ppr": selected_rank=fusion_ppr
    else: selected_rank=rank_scores(recency_scores(train_events,cutoff,int(selected.split('_')[1][:-1])),val_candidates)
    if selected=="ppr": selected_test,_=ppr(train,model_users,test_candidates)
    elif selected=="cbf": selected_test,_=cbf(train,model_users,meta,test_candidates,set(train_meta))
    elif selected.startswith("recency_"): selected_test=rank_scores(recency_scores(train_events,cutoff,int(selected.split('_')[1][:-1])),test_candidates)
    elif selected=="frequency": selected_test=freq_rank
    elif selected=="fusion_cbf": selected_test=fuse(freq_rank,cbf(train,model_users,meta,test_candidates,set(train_meta))[0],test_candidates)
    else: selected_test=fuse(freq_rank,ppr(train,model_users,test_candidates)[0],test_candidates)
    pop_test, _ = ordered_popularity(train,test_candidates); test_pred,_=fill(users,selected_test,pop_test,test_candidates); save_pred(OUT/f"submission_selected_{selected}.csv",test_pred,users)
    # Save the same heavy/light users used by the baseline audit.
    case=[]
    for u in case_users:
        for name,m in methods.items():
            ids5=m["case_predictions"][u]; case.append({"user_id":u,"method":name,"items":[{"tv_show_id":p,"title":meta.get(p,{}).get("title"),"category":meta.get(p,{}).get("category"),"hit":p in targets.get(u,set())} for p in ids5]})
    with (OUT/"comparison.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["method","map5","runtime_sec","fallback_users","settings"]); w.writeheader(); w.writerows(rows)
    ref=np.array([1.0,0.0]); refP=np.array([[0.0,1.0],[1.0,0.0]])
    for _ in range(100): ref=.5*refP.T.dot(ref)+.5*np.array([1.0,0.0])
    assert np.allclose(ref,[2/3,1/3],atol=1e-8)
    summary={"protocol":{"cohort":len(users),"empty_targets":sum(not targets.get(u) for u in users),"train_fraction":.50,"target_fraction":.80,"approximation":"validation target is each user's five most frequent programmes among >=0.80 overlap airings","repeats_allowed":True},"selected_method":selected,"ppr_reference":{"alpha":.5,"restart":[1,0],"stationary":ref.tolist(),"check_passed":True},"methods":rows,"case_users":case_users,"runtime_total_sec":time.perf_counter()-started}
    (OUT/"methods_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); (OUT/"case_users.json").write_text(json.dumps(case,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
