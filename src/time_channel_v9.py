"""Bounded time-and-channel refinement of the frozen channel_v8 candidates.

It streams raw sessions once (or loads a fingerprinted checkpoint), never joins
views to programmes, and leaves prior experiment directories untouched.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import pickle
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import temporal_v4 as v4
import channel_v8 as v8
from corrected_experiment import users, write_pred

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/dataset11-30.csv"
V4, V5, V7, V8 = (ROOT / "results/temporal_v4", ROOT / "results/ensemble_v5",
                  ROOT / "results/v4_refinement_v7", ROOT / "results/channel_v8")
OUT = ROOT / "results/time_channel_v9"
OUT.mkdir(parents=True, exist_ok=True)
T, U = datetime(2020, 5, 18), datetime(2020, 7, 27)
CUTS = {"apr05": datetime(2020, 4, 5), "apr19": datetime(2020, 4, 19),
        "outer": T, "jun14": datetime(2020, 6, 14), "jun28": datetime(2020, 6, 28),
        "final": U}
COHORT = users()
V8_FEATURES = list(v4.FEATURES_V4) + ["channel_pref_mean", "channel_pref_per_day",
    "channel_pref_max", "log_channel_support"]
TIME_FEATURES = ["time_pref_mean", "channel_time_pref_mean",
                 "channel_time_pref_per_day", "channel_time_interaction"]
RECENT_FEATURES = ["recent28_channel_pref_mean", "log_recent28_channel_support"]


def canonical(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
def digest(x): return hashlib.sha256(canonical(x).encode()).hexdigest()
def save_json(path, x): Path(path).write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding="utf-8")
def identity(path):
    path = Path(path); h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": h.hexdigest()}
def load_npz(path):
    z = np.load(path, allow_pickle=False)
    keys = list(zip(z["users"].astype(str).tolist(), z["pids"].astype(int).tolist()))
    return z["X"].astype(np.float32), z.get("y"), keys, str(z["fingerprint"])
def read_targets(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return {r["user_id"]: set(map(int, r["targets"].split())) if r["targets"] else set()
                for r in csv.DictReader(f)}
def read_predictions(path, column):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return {r["user_id"]: list(map(int, r[column].split())) for r in csv.DictReader(f)}
def ap_scores(pred, targets):
    out = {}
    for u in COHORT:
        hits = score = 0.0; rel = targets.get(u, set())
        for rank, p in enumerate(pred[u], 1):
            if p in rel: hits += 1; score += hits / rank
        out[u] = score / min(5, len(rel)) if rel else 0.0
    return out
def map5(pred, targets):
    a = ap_scores(pred, targets); return sum(a.values()) / len(COHORT), a
def table(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def bin_index(ts): return ts.weekday() * 6 + ts.hour // 4
def next_bin(ts):
    base = ts.replace(hour=(ts.hour // 4) * 4, minute=0, second=0, microsecond=0)
    return base + timedelta(hours=4)
def fragments(start, end, weight):
    """Allocate one capped-session weight proportionally over actual bin seconds."""
    total = (end - start).total_seconds()
    if total <= 0 or weight <= 0: return
    cur = start
    while cur < end:
        nxt = min(end, next_bin(cur)); seconds = (nxt - cur).total_seconds()
        yield bin_index(cur), weight * seconds / total
        cur = nxt


def needed_users():
    found = set(COHORT)
    for name in ("outer", "apr05", "apr19", "jun14", "jun28", "final"):
        _, _, keys, _ = features29(name)
        found.update(u for u, _ in keys)
    return found


def profile_cache(base, timings):
    path = OUT / "time_channel_profiles.pkl"
    users_needed = needed_users()
    fp_data = {"version": "time_channel_v9_profiles", "raw": identity(RAW),
               "cutoffs": {k: v.isoformat() for k, v in CUTS.items()}, "bins": "weekday*6+hour//4",
               "cap": 7200, "recent_days": 28, "needed_users_digest": digest(sorted(users_needed))}
    fp = digest(fp_data); started = time.perf_counter()
    if path.exists():
        q = pickle.load(path.open("rb"))
        if q.get("fingerprint") == fp:
            timings["profiles"] = {"seconds": time.perf_counter()-started, "cache_hit": True}
            return q
    # all[c][u] = [J channel->42-vector, H 42-vector, M]; recent stores raw channel seconds.
    allp = {n: {} for n in CUTS}; recent = {n: {} for n in CUTS}; rows_seen = kept = 0
    for chunk_no, d in enumerate(pd.read_csv(RAW, dtype={"user_id": str, "vsetv_id": "int64"},
                                              parse_dates=["start_time", "stop_time"], chunksize=25000), 1):
        rows_seen += len(d); d = d[d["user_id"].isin(users_needed)]
        kept += len(d)
        for r in d.itertuples(index=False):
            u, ch, st, en = str(r.user_id), int(r.vsetv_id), r.start_time.to_pydatetime(), r.stop_time.to_pydatetime()
            if en <= st: continue
            for name, cut in CUTS.items():
                if st >= cut: continue
                end = min(en, cut); duration = (end-st).total_seconds()
                if duration <= 0: continue
                weight = min(duration, 7200.0)
                entry = allp[name].get(u)
                if entry is None:
                    entry = [defaultdict(lambda: np.zeros(42, dtype=np.float64)), np.zeros(42, dtype=np.float64), 0.0]
                    allp[name][u] = entry
                j, h, _ = entry
                for b, w in fragments(st, end, weight): j[ch][b] += w; h[b] += w
                entry[2] += weight
                rs = max(st, cut - timedelta(days=28))
                if rs < end:
                    rw = min((end-rs).total_seconds(), 7200.0)
                    if rw > 0:
                        rc = recent[name].setdefault(u, Counter()); rc[ch] += rw
        print(f"profile chunks={chunk_no} rows={rows_seen} retained={kept}", flush=True)
    # Convert defaultdicts so cache loading does not depend on local lambdas.
    packed = {}
    for name, d in allp.items():
        packed[name] = {u: ({int(ch): np.asarray(a, dtype=np.float32) for ch, a in j.items()},
                            np.asarray(h, dtype=np.float32), float(m)) for u, (j, h, m) in d.items()}
    # Exact conservation / boundary fixtures and reconciliation against v8 channel marginals.
    fixture = list(fragments(datetime(2020, 5, 17, 23), datetime(2020, 5, 18, 2), 3600.0))
    v8p = pickle.load((V8 / "channel_profiles.pkl").open("rb"))["profiles"]
    reconcile = {}
    for name in CUTS:
        max_support = max_pref = 0.0; compared = 0
        for u, (j, h, m) in packed[name].items():
            old = v8p.get(name, {}).get(u)
            if not old: continue
            prefs, support = old; compared += 1; max_support = max(max_support, abs(m-support))
            for ch in set(prefs) | set(j): max_pref = max(max_pref, abs(float(prefs.get(ch, 0.0)) - float(j.get(ch, np.zeros(42)).sum()/m)))
        reconcile[name] = {"users_compared": compared, "max_support_abs_error": max_support,
                           "max_channel_pref_abs_error": max_pref, "matches_v8": max_support < 1e-5 and max_pref < 1e-7}
    q = {"fingerprint": fp, "fingerprint_data": fp_data, "profiles": packed, "recent": recent,
         "checks": {"fixture": {"cross_midnight_bins": [b for b,_ in fixture], "weight_sum": sum(w for _,w in fixture),
                                    "expected_weight": 3600.0, "post_cutoff_data_excluded_by_start_lt_cutoff": True},
                    "reconciliation_v8": reconcile,
                    "sessions_seen": rows_seen, "sessions_for_needed_users": kept}}
    pickle.dump(q, path.open("wb"), protocol=5); save_json(OUT / "profile_checks.json", q["checks"])
    timings["profiles"] = {"seconds": time.perf_counter()-started, "cache_hit": False}
    return q


def schedule_exposure(start, end):
    rows, _, _ = v4.load_schedule(); out = defaultdict(lambda: defaultdict(lambda: np.zeros(42, dtype=np.float64)))
    for ch, st, en, pid in set(rows):
        if pid <= 0 or not (start <= st < end): continue  # v8 eligibility convention
        left, right = max(st, start), min(en, end)
        duration = (right-left).total_seconds()
        if duration <= 0: continue
        for b, w in fragments(left, right, 1.0): out[pid][ch][b] += w
    packed = {int(p): {int(c): np.asarray(a, dtype=np.float32) for c,a in d.items()} for p,d in out.items()}
    return packed, max((end-start).total_seconds()/86400.0, 1e-9)


def features29(name):
    # channel_v8 cached only its derived X matrix; authoritative keys/labels
    # remain in the source candidate archives it augmented.
    source = {
        "outer": V4 / "features_outer_new200.npz",
        "apr05": V4 / "features_long_apr05_T.npz",
        "apr19": V4 / "features_long_apr19_T.npz",
        "jun14": V4 / "features_final_jun14_U.npz",
        "jun28": V4 / "features_final_jun28_U.npz",
        "final": V5 / "final_candidate_features.npz",
    }[name]
    _, y, keys, source_fp = load_npz(source)
    z = np.load(V8 / f"{name}_channel29.npz", allow_pickle=False)
    x = z["X"].astype(np.float32)
    if len(x) != len(keys) or x.shape[1] != 29:
        raise RuntimeError(f"v8 {name} matrix/key mismatch")
    return x, y, keys, str(z["fingerprint"])

def add_features(x, keys, profile, exposure, days, schema, base, name, timings, output_name=None):
    path = OUT / f"{output_name or name}_{schema}.npz"; fp_data = {"base": base, "name": name, "schema": schema,
        "features": V8_FEATURES + TIME_FEATURES + (RECENT_FEATURES if schema == "time35" else []),
        "rows_digest": digest(keys)}; fp = digest(fp_data); started = time.perf_counter()
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z["fingerprint"]) == fp:
            timings[f"features_{name}_{schema}"] = {"seconds": time.perf_counter()-started, "cache_hit": True}; return z["X"]
    allp, recent = profile["profiles"], profile["recent"]
    cut = {"outer": "outer", "apr05": "apr05", "apr19": "apr19", "jun14": "jun14", "jun28": "jun28", "final": "final"}[name]
    timef = np.zeros((len(keys), 4), dtype=np.float32); recf = np.zeros((len(keys), 2), dtype=np.float32)
    for i, (u, pid) in enumerate(keys):
        entry = allp[cut].get(u); sched = exposure.get(pid)
        if entry is not None and sched:
            j, h, m = entry
            if m > 0:
                pt = (h + 3600.0/42.0) / (m + 3600.0)
                numerator = interaction = n = 0.0
                pc = {ch: float(a.sum()/m) for ch,a in j.items()}
                for ch, e in sched.items():
                    cnt = float(e.sum()); n += cnt
                    pch = pc.get(ch, 0.0); ja = j.get(ch, np.zeros(42, dtype=np.float32))
                    a = pt * ((ja + 3600.0*pch) / (h + 3600.0))
                    numerator += float(np.dot(e, a)); interaction += float(np.dot(e, a - pch*pt))
                if n: timef[i] = (float(np.dot(sum(sched.values()), pt)/n), numerator/n, numerator/days, interaction/n)
        rc = recent[cut].get(u)
        if rc and sched:
            support = float(sum(rc.values())); dist = {ch: w/support for ch,w in rc.items()} if support else {}
        elif entry is not None and sched and entry[2] > 0: # requested all-history fallback, but support stays zero
            j, _, m = entry; dist = {ch: float(a.sum()/m) for ch,a in j.items()}; support = 0.0
        else: dist = {}; support = 0.0
        if sched:
            n = sum(float(e.sum()) for e in sched.values())
            if n: recf[i,0] = sum(float(e.sum())*dist.get(ch,0.0) for ch,e in sched.items()) / n
            recf[i,1] = np.log1p(support)
    out = np.hstack([x, timef] + ([recf] if schema == "time35" else [])).astype(np.float32)
    np.savez_compressed(path, fingerprint=fp, X=out); save_json(path.with_suffix(".fingerprint.json"), fp_data)
    timings[f"features_{name}_{schema}"] = {"seconds": time.perf_counter()-started, "cache_hit": False}; return out


def fit(name, x, y, fp_data, features, timings):
    path = OUT / f"{name}.pkl"; side = path.with_suffix(".fingerprint.json"); fp = digest(fp_data); started = time.perf_counter()
    if path.exists() and side.exists() and json.loads(side.read_text()).get("fingerprint") == fp:
        timings[name] = {"seconds": time.perf_counter()-started, "cache_hit": True}; return pickle.load(path.open("rb"))
    model = HistGradientBoostingClassifier(**v4.HGB_PARAMS); model.fit(x, y)
    pickle.dump(model, path.open("wb"), protocol=5); save_json(side, {"fingerprint": fp, "features": features, "settings": v4.HGB_PARAMS, "rows": len(y), "positives": int(y.sum())})
    timings[name] = {"seconds": time.perf_counter()-started, "cache_hit": False}; return model


def prediction_from_scores(scores, keys, cold, fallback): return v8.top(scores, keys, cold, fallback)


def main():
    wall = time.perf_counter(); timings = {}
    ids_before = {n: identity(p) for n,p in {"v8_submission": V8/"submission_channel_v8_channel_hgb.csv",
        "v8_validation_model": V8/"channel_hgb_validation.pkl", "v8_final_model": V8/"channel_hgb_final.pkl"}.items()}
    target = read_targets(V4 / "validation_targets.csv"); late = read_targets(V7 / "late_targets.csv")
    incumbent = read_predictions(V8 / "validation_predictions.csv", "channel_hgb")
    control_full, control_ap = map5(incumbent, target); control_late, _ = map5(incumbent, late)
    if abs(control_full - .40946318342151666) > 1e-12 or abs(control_late - .35924184303350976) > 1e-12: raise RuntimeError("v8 metric reproduction mismatch")
    x0, y0, keys0, fp0 = features29("outer")
    if x0.shape[1] != 29 or V8_FEATURES != json.loads((V8/"summary.json").read_text())["schema"]: raise RuntimeError("v8 schema mismatch")
    # Exact original-feature identity: the first 25 columns came directly from v4.
    raw25, _, key25, _ = load_npz(V4 / "features_outer_new200.npz")
    if keys0 != key25 or not np.array_equal(x0[:, :25], raw25): raise RuntimeError("v8 original feature values/keys mismatch")
    base = digest({"v8_inputs": ids_before, "v8_fp": json.loads((V8/"summary.json").read_text())["fingerprint"],
                   "feature_schemas": {"time33": V8_FEATURES+TIME_FEATURES, "time35": V8_FEATURES+TIME_FEATURES+RECENT_FEATURES}})
    profiles = profile_cache(base, timings)
    # Stop before fitting if time-bin profiles are not numerically the documented v8 marginals.
    bad = [n for n,q in profiles["checks"]["reconciliation_v8"].items() if not q["matches_v8"]]
    if bad: raise RuntimeError(f"time profiles do not reconcile with v8: {bad}")
    specs = {"outer": (T,U), "apr05": (CUTS["apr05"],T), "apr19": (CUTS["apr19"],T),
             "jun14": (CUTS["jun14"],U), "jun28": (CUTS["jun28"],U)}
    exps = {n: schedule_exposure(*w) for n,w in specs.items()}
    x33o = add_features(x0, keys0, profiles, exps["outer"][0], exps["outer"][1], "time33", base, "outer", timings)
    x35o = add_features(x0, keys0, profiles, exps["outer"][0], exps["outer"][1], "time35", base, "outer", timings)
    parts = {}
    for n in ("apr05", "apr19"):
        x,y,k,fp = features29(n); parts[n] = (add_features(x,k,profiles,exps[n][0],exps[n][1],"time33",base,n,timings),
                                               add_features(x,k,profiles,exps[n][0],exps[n][1],"time35",base,n,timings), y, fp)
    train33 = np.vstack([parts["apr05"][0], parts["apr19"][0]]); train35 = np.vstack([parts["apr05"][1], parts["apr19"][1]])
    labels = np.concatenate([parts["apr05"][2], parts["apr19"][2]])
    m33 = fit("time33_validation", train33, labels, {"base":base,"snapshots":[parts["apr05"][3],parts["apr19"][3]],"schema":"time33"}, V8_FEATURES+TIME_FEATURES, timings)
    m35 = fit("time35_validation", train35, labels, {"base":base,"snapshots":[parts["apr05"][3],parts["apr19"][3]],"schema":"time35"}, V8_FEATURES+TIME_FEATURES+RECENT_FEATURES, timings)
    counts = pickle.load((V4/"aggregate_outer_T_U.pkl").open("rb"))["data"][1]; cold = {u for u in COHORT if not counts.get(u)}
    fallback = read_predictions(V4/"validation_predictions.csv", "popularity_clean")
    p33, p35 = m33.predict_proba(x33o)[:,1], m35.predict_proba(x35o)[:,1]
    preds = {"channel_v8": incumbent, "time33": prediction_from_scores(p33,keys0,cold,fallback), "time35": prediction_from_scores(p35,keys0,cold,fallback)}
    scores, aps, late_scores = {}, {}, {}
    for n,p in preds.items(): scores[n],aps[n] = map5(p,target); late_scores[n],_ = map5(p,late)
    qualified = [n for n in ("time33","time35") if scores[n] > control_full+1e-12 and late_scores[n] >= control_late-1e-12]
    selected = max(qualified, key=lambda n:(scores[n], late_scores[n], -len((V8_FEATURES+TIME_FEATURES) if n=="time33" else (V8_FEATURES+TIME_FEATURES+RECENT_FEATURES)))) if qualified else "retain_channel_v8"
    table(OUT/"comparison.csv", [{"method":n,"full_map5":scores[n],"late_map5":late_scores[n],"full_delta_v8":scores[n]-control_full,"late_delta_v8":late_scores[n]-control_late,"qualified":n in qualified} for n in preds])
    table(OUT/"validation_predictions.csv", [{"user_id":u, **{n:" ".join(map(str,p[u])) for n,p in preds.items()}} for u in COHORT])
    table(OUT/"per_user_ap.csv", [{"user_id":u,"full_target_count":len(target.get(u,set())),"late_target_count":len(late.get(u,set())), **{f"ap_{n}":aps[n][u] for n in preds}, **{f"diff_{n}_minus_v8":aps[n][u]-control_ap[u] for n in preds}} for u in COHORT])
    table(OUT/"full_window_targets.csv", [{"user_id":u,"targets":" ".join(map(str,sorted(target.get(u,set())))),"target_count":len(target.get(u,set()))} for u in COHORT])
    with gzip.open(OUT/"validation_candidate_scores.csv.gz","wt",encoding="utf-8",newline="") as f:
        w=csv.writer(f);w.writerow(["user_id","tv_show_id","time33_probability","time35_probability"])
        for (u,p),a,b in zip(keys0,p33,p35):w.writerow([u,p,a,b])
    final_submission = checks = None
    if selected != "retain_channel_v8":
        # Build final exposure only after selection; no future viewing is read.
        rows, split, _ = v4.load_schedule(); test_end=max(r[2] for r in split["test"])+timedelta(seconds=1)
        final_exp, final_days = schedule_exposure(U,test_end); fx,_,fkeys,_ = features29("final")
        schema = "time33" if selected=="time33" else "time35"
        fxf = add_features(fx,fkeys,profiles,final_exp,final_days,schema,base,"final",timings)
        final_parts=[]; final_y=[]; final_fps=[]
        for n in ("jun14","jun28"):
            x,y,k,fp=features29(n); final_parts.append(add_features(x,k,profiles,exps[n][0],exps[n][1],schema,base,n,timings));final_y.append(y);final_fps.append(fp)
        fm=fit(f"{selected}_final",np.vstack(final_parts),np.concatenate(final_y),{"base":base,"snapshots":final_fps,"schema":schema},V8_FEATURES+TIME_FEATURES+(RECENT_FEATURES if schema=="time35" else []),timings)
        fs=fm.predict_proba(fxf)[:,1]; final_counts=pickle.load((V4/"aggregate_final_U.pkl").open("rb"))["data"][1]; cold_final={u for u in COHORT if not final_counts.get(u)}
        fallback_final=read_predictions(V8/"submission_channel_v8_channel_hgb.csv","tv_show_id")
        final_submission=f"submission_time_channel_v9_{selected}.csv"; write_pred(OUT/final_submission,prediction_from_scores(fs,fkeys,cold_final,fallback_final),COHORT)
        eligible={r[3] for r in set(split["test"]) if r[3]>0}; submitted=list(csv.DictReader((OUT/final_submission).open(encoding="utf-8",newline="")))
        checks={"rows":len(submitted),"template_order":[r["user_id"] for r in submitted]==COHORT,"complete_coverage":len(submitted)==1260,"five_distinct_positive_ids":all(len(a:=r["tv_show_id"].split())==5 and len(set(a))==5 and all(int(x)>0 for x in a) for r in submitted),"schedule_available":all(all(int(x) in eligible for x in r["tv_show_id"].split()) for r in submitted),"strict_pre_U_fitting":True,"fallback_users":len(cold_final)}
        save_json(OUT/"submission_checks.json",checks)
    ids_after={n:identity(Path(x["path"])) for n,x in ids_before.items()}
    unchanged={n:ids_after[n]["sha256"]==x["sha256"] for n,x in ids_before.items()}
    boundaries={"T":T.isoformat(sep=" "),"U":U.isoformat(sep=" "),"validation_history":"strictly before T","validation_labels":"T <= airing_end < U","late":"2020-06-15 <= airing_end < U (overlapping diagnostic)","final_history":"strictly before U","all_1260_users":len(COHORT)==1260}
    summary={"version":"time_channel_v9","incumbent_reproduced":{"full":control_full,"late":control_late},"scores":scores,"late_scores":late_scores,"qualified":qualified,"selected":selected,"schemas":{"time33":V8_FEATURES+TIME_FEATURES,"time35":V8_FEATURES+TIME_FEATURES+RECENT_FEATURES},"feature_fingerprint":base,"input_identities_before":ids_before,"input_identities_after":ids_after,"incumbent_unchanged":unchanged,"boundaries":boundaries,"final_submission":final_submission,"submission_checks":checks,"timings":timings,"runtime_total_sec":time.perf_counter()-wall}
    save_json(OUT/"summary.json",summary);save_json(OUT/"timings.json",timings);save_json(OUT/"boundary_checks.json",boundaries)
    print(json.dumps({"scores":scores,"late":late_scores,"selected":selected,"submission":final_submission,"checks":checks,"runtime":summary["runtime_total_sec"]},indent=2))

if __name__ == "__main__": main()
