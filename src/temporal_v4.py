"""Clean-boundary SWEET.TV evaluation and two bounded reranker experiments.

This runner consumes corrected_v2 interaction caches only.  It never performs the
view/schedule interval join and never writes outside results/temporal_v4.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import scipy
import sklearn
from scipy.sparse import csr_matrix, diags, load_npz, save_npz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from baseline import RAW, SCHEDULES
from corrected_experiment import build_graph, fill, pop_order, users, write_pred

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results/corrected_v2"
V3 = ROOT / "results/tuning_v3"
OUT = ROOT / "results/temporal_v4"
OUT.mkdir(parents=True, exist_ok=True)
TRAIN_CACHE = V2 / "train_interactions.csv.gz"
VAL_CACHE = V2 / "val_interactions.csv.gz"
T = datetime(2020, 5, 18)
U = datetime(2020, 7, 27)
ORIGIN = datetime(2020, 3, 8, 23)
ALPHA = 0.75

FEATURES_V3 = [
    "count30", "count50", "count80", "log_duration", "relative_frequency",
    "recent14", "days_since_last", "frequency_rank", "ppr_score", "ppr_rank",
    "cbf_score", "cbf_rank", "log_popularity", "history_exposure_day",
    "future_exposure_day", "exposure_ratio", "category_affinity",
    "source_frequency", "source_ppr", "source_cbf", "source_popularity",
]
FEATURES_V4 = FEATURES_V3 + [
    "count50_per_history_day", "watched_seconds_per_history_day",
    "recent14_over_count50", "history_length_days",
]
HGB_PARAMS = {
    "max_iter": 120, "learning_rate": 0.08, "max_leaf_nodes": 31,
    "min_samples_leaf": 30, "l2_regularization": 1.0,
    "class_weight": "balanced", "random_state": 42,
}
CANDIDATE_POLICY = {
    "v3_100": "freq30,ppr50,cbf30,pop30; then ppr100; v3 pop floor; truncate100",
    "nested_200": "preserve v3_100; deterministic round-robin tails frequency,ppr,cbf,pop",
    "relevance_blind": True,
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_identity(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    st = path.stat()
    return {"path": str(path.resolve()), "size": st.st_size, "sha256": h.hexdigest()}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def timed(timings, name, fn):
    started = time.perf_counter()
    value, cache_hit = fn()
    timings[name] = {"seconds": time.perf_counter() - started, "cache_hit": bool(cache_hit)}
    return value


def load_schedule():
    all_rows = []
    split_rows = {}
    meta = {}
    for split, path in SCHEDULES.items():
        rows = []
        with path.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                pid = int(r["tv_show_id"])
                start = datetime.strptime(r["start_time"], "%d.%m.%Y %H:%M:%S")
                end = start + timedelta(seconds=float(r["duration"]))
                key = (int(r["channel_id"]), start, end, pid)
                rows.append(key)
                if pid > 0 and pid not in meta:
                    fields = [r.get(k, "") for k in (
                        "tv_show_title", "tv_show_category", "tv_show_genre_1",
                        "tv_show_genre_2", "tv_show_genre_3", "director", "actors",
                        "year_of_production",
                    )]
                    meta[pid] = {
                        "title": r["tv_show_title"], "category": r["tv_show_category"],
                        "doc": " ".join(x for x in fields if x),
                    }
        split_rows[split] = rows
        all_rows.extend(rows)
    dedup = sorted(set(all_rows), key=lambda x: (x[1], x[0], x[2], x[3]))
    return dedup, split_rows, meta


def schedule_window(rows, start, end):
    # Preserve v3 candidate/exposure semantics: schedule windows use airing start.
    # Interaction history and labels independently use the mandated airing_end cuts.
    selected = [r for r in set(rows) if r[3] > 0 and start <= r[1] < end]
    counts = Counter(r[3] for r in selected)
    days = max((end - start).total_seconds() / 86400.0, 1e-9)
    return set(counts), counts, days


def exposure(hist_counts, hist_days, future_counts, future_days, smoothing=0.1, clip=(0.5, 2.0)):
    hr, fr, ratio = {}, {}, {}
    for p in set(hist_counts) | set(future_counts):
        h = hist_counts[p] / hist_days
        f = future_counts[p] / future_days
        hr[p], fr[p] = h, f
        ratio[p] = min(clip[1], max(clip[0], (f + smoothing) / (h + smoothing)))
    return hr, fr, ratio


def scan_interactions(history_end, label_start=None, label_end=None):
    """Aggregate corrected caches, deduplicating their known T-boundary overlap."""
    stats = defaultdict(dict)
    labels = defaultdict(Counter)
    first = {}
    recent_cut = history_end - timedelta(days=14)
    deferred = {}
    observed = {
        "history_max_airing_end": None, "label_min_airing_end": None,
        "label_max_airing_end": None, "cross_cache_duplicates": 0,
    }

    def consume(r):
        end = datetime.fromisoformat(r["airing_end"])
        frac = float(r["overlap_frac"])
        u, p = r["user_id"], int(r["tv_show_id"])
        if end < history_end and frac >= 0.30:
            a = stats[u].setdefault(p, [0, 0, 0, 0.0, None, 0])
            a[0] += 1
            first[u] = min(first.get(u, end), end)
            observed["history_max_airing_end"] = max(observed["history_max_airing_end"] or end, end)
            if frac >= 0.50:
                a[1] += 1
                a[3] += float(r["overlap_sec"])
                a[4] = end if a[4] is None or end > a[4] else a[4]
                if end >= recent_cut:
                    a[5] += 1
            if frac >= 0.80:
                a[2] += 1
        if label_start is not None and label_start <= end < label_end and frac >= 0.80:
            labels[u][p] += 1
            observed["label_min_airing_end"] = min(observed["label_min_airing_end"] or end, end)
            observed["label_max_airing_end"] = max(observed["label_max_airing_end"] or end, end)

    for path in (TRAIN_CACHE, VAL_CACHE):
        source = path.name
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                end = datetime.fromisoformat(r["airing_end"])
                relevant = end < history_end or (
                    label_start is not None and label_start <= end < label_end
                )
                if not relevant:
                    continue
                # Only the 12-hour overlap of the two schedule exports can repeat.
                if T <= end < T + timedelta(hours=12):
                    key = (r["user_id"], r["tv_show_id"], r["airing_start"], r["airing_end"])
                    old = deferred.get(key)
                    if old is not None:
                        observed["cross_cache_duplicates"] += 1
                    if old is None or float(r["overlap_sec"]) > float(old["overlap_sec"]):
                        deferred[key] = dict(r, _source=source)
                else:
                    consume(r)
    for r in deferred.values():
        consume(r)
    counts = defaultdict(Counter)
    for u, items in stats.items():
        for p, a in items.items():
            if a[1]:
                counts[u][p] = a[1]
    targets = {u: set(sorted(c, key=lambda p: (-c[p], p))[:5]) for u, c in labels.items()}
    for key in ("history_max_airing_end", "label_min_airing_end", "label_max_airing_end"):
        if observed[key] is not None:
            observed[key] = observed[key].isoformat(sep=" ")
    return stats, counts, targets, first, observed


def aggregate_cached(name, history_end, label_start, label_end, base_fp):
    path = OUT / f"aggregate_{name}.pkl"
    fp_data = {
        "base": base_fp, "history_end": history_end.isoformat(),
        "label_start": label_start.isoformat() if label_start else None,
        "label_end": label_end.isoformat() if label_end else None,
        "dedupe": "max-overlap per user-airing across corrected caches in T+12h overlap",
    }
    fp = digest(fp_data)
    if path.exists():
        payload = pickle.load(path.open("rb"))
        if payload.get("fingerprint") == fp:
            return payload["data"], True
    data = scan_interactions(history_end, label_start, label_end)
    pickle.dump({"fingerprint": fp, "fingerprint_data": fp_data, "data": data}, path.open("wb"), protocol=5)
    return data, False


def top_ids(score_row, candidate_ids, n, positive=False):
    valid = np.flatnonzero(score_row > 0) if positive else np.arange(len(candidate_ids))
    if not len(valid):
        return []
    take = valid if len(valid) <= n else valid[np.argpartition(-score_row[valid], n - 1)[:n]]
    ordered = take[np.lexsort((candidate_ids[take], -score_row[take]))]
    return [int(candidate_ids[i]) for i in ordered]


def sparse_row(matrix, i, ncols):
    row = matrix.getrow(i)
    out = np.zeros(ncols, dtype=np.float32)
    out[row.indices] = row.data.astype(np.float32)
    return out


def frequency_rank(counts, u, candidates):
    return [p for p, _ in sorted(
        ((p, n) for p, n in counts.get(u, {}).items() if p in candidates),
        key=lambda x: (-x[1], x[0]),
    )]


def ppr_scores(graph, counts, pred_users, candidates, fp_data, cache_name):
    path = OUT / f"{cache_name}.npz"
    fp = digest(fp_data)
    cand = np.array(sorted(candidates), dtype=np.int64)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z["fingerprint"]) == fp:
            settings = json.loads(str(z["settings"]))
            settings["cache_hit"] = True
            return z["candidate_ids"], z["scores"], settings, True
    cidx = np.array([graph["pos"][int(p)] for p in cand])
    scores = np.zeros((len(pred_users), len(cand)), dtype=np.float32)
    iterations, converged, max_delta = [], 0, 0.0
    for i, u in enumerate(pred_users):
        restart = np.zeros(len(graph["ids"]), dtype=np.float64)
        for p, n in counts.get(u, {}).items():
            restart[graph["pos"][p]] = n
        if not restart.sum():
            continue
        restart /= restart.sum()
        state = restart.copy()
        for iteration in range(1, 101):
            nxt = ALPHA * (
                graph["P"].T.dot(state) + state[graph["dangling"]].sum() * restart
            ) + (1 - ALPHA) * restart
            delta = float(np.abs(nxt - state).sum())
            state = nxt
            if delta < 1e-8:
                converged += 1
                break
        iterations.append(iteration)
        max_delta = max(max_delta, delta)
        scores[i] = state[cidx]
    settings = {
        "alpha": ALPHA, "users_run": len(iterations), "converged_users": converged,
        "mean_iterations": float(np.mean(iterations)) if iterations else 0.0,
        "max_final_delta": max_delta, "cache_hit": False,
    }
    np.savez_compressed(path, fingerprint=fp, candidate_ids=cand, scores=scores,
                        settings=canonical(settings))
    return cand, scores, settings, False


def cbf_scores(stats, counts, pred_users, candidates, fit_ids, meta):
    fit = sorted(p for p in fit_ids if p in meta)
    cand = np.array(sorted(candidates), dtype=np.int64)
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), min_df=1, max_features=100000)
    x = vec.fit_transform([meta[p]["doc"] or "unknown" for p in fit])
    pos = {p: i for i, p in enumerate(fit)}
    rr, cc, dd = [], [], []
    for i, u in enumerate(pred_users):
        for p, n in counts.get(u, {}).items():
            if p in pos:
                rr.append(i); cc.append(pos[p]); dd.append(n)
    user_items = csr_matrix((dd, (rr, cc)), shape=(len(pred_users), len(fit)))
    profiles = user_items @ x
    candidate_x = vec.transform([meta.get(int(p), {"doc": "unknown"})["doc"] or "unknown" for p in cand])
    return cand, (profiles @ candidate_x.T).tocsr(), {"features": x.shape[1], "fit_items": len(fit)}


def source_bundle(stats, counts, pred_users, candidates, meta, fit_ids, schedule_rows,
                  history_start, history_end, future_start, future_end, fp_data, cache_tag):
    _, hist_sched, hist_days = schedule_window(schedule_rows, history_start, history_end)
    _, future_sched, future_days = schedule_window(schedule_rows, future_start, future_end)
    hr, fr, ratio = exposure(hist_sched, hist_days, future_sched, future_days)
    pop, pop_counts = pop_order(counts, candidates)
    graph = build_graph(counts, candidates)
    cand, ppr, pset, ppr_hit = ppr_scores(
        graph, counts, pred_users, candidates,
        {**fp_data, "stage": cache_tag, "alpha": ALPHA, "users": pred_users,
         "candidates": sorted(candidates)}, f"ppr_{cache_tag}",
    )
    ccand, cbf, cset = cbf_scores(stats, counts, pred_users, candidates, fit_ids, meta)
    assert np.array_equal(cand, ccand)
    pop_rank = {p: i + 1 for i, p in enumerate(pop)}
    cat_aff = {}
    for u, items in counts.items():
        cats = Counter()
        total = sum(items.values())
        for p, n in items.items():
            cats[meta.get(p, {}).get("category", "")] += n
        cat_aff[u] = {k: v / total for k, v in cats.items()} if total else {}
    return {
        "cand": cand, "ppr": ppr, "ppr_settings": pset, "ppr_cache_hit": ppr_hit,
        "cbf": cbf, "cbf_settings": cset, "pop": pop, "pop_rank": pop_rank,
        "pop_counts": pop_counts, "user_index": {u: i for i, u in enumerate(pred_users)},
        "cat_aff": cat_aff, "hist_exp": hr, "fut_exp": fr, "ratio": ratio,
        "graph": {"users": graph["users"], "nodes": graph["nodes"], "edges": graph["edges"]},
    }


def shortlist_and_features(stats, counts, targets, first, pred_users, sources, meta,
                           history_end, max_candidates, schema):
    cand = sources["cand"]
    candidate_set = set(map(int, cand))
    rows, labels, keys, lists = [], [], [], {}
    for u in pred_users:
        i = sources["user_index"][u]
        prow = sources["ppr"][i]
        crow = sparse_row(sources["cbf"], i, len(cand))
        freq_full = frequency_rank(counts, u, candidate_set)
        ppr_full = top_ids(prow, cand, len(cand), True)
        cbf_full = top_ids(crow, cand, len(cand), True)
        pop_full = sources["pop"]
        freq30, ppr50, cbf30, pop30 = freq_full[:30], ppr_full[:50], cbf_full[:30], pop_full[:30]
        chosen = []
        for seq in (freq30, ppr50, cbf30, pop30):
            for p in seq:
                if p not in chosen:
                    chosen.append(p)
        for p in ppr_full[:100]:
            if len(chosen) >= 100:
                break
            if p not in chosen:
                chosen.append(p)
        for p in pop_full:
            if len(chosen) >= 50:
                break
            if p not in chosen:
                chosen.append(p)
        chosen = chosen[:100]
        if max_candidates == 200:
            seen = set(chosen)
            positions = [0, 0, 0, 0]
            tails = [freq_full, ppr_full, cbf_full, pop_full]
            while len(chosen) < 200:
                added = False
                for j, seq in enumerate(tails):
                    while positions[j] < len(seq) and seq[positions[j]] in seen:
                        positions[j] += 1
                    if positions[j] < len(seq):
                        p = seq[positions[j]]; positions[j] += 1
                        chosen.append(p); seen.add(p); added = True
                        if len(chosen) >= 200:
                            break
                if not added:
                    break
        lists[u] = chosen
        fmap = {p: r + 1 for r, p in enumerate(freq30)}
        pmap = {p: r + 1 for r, p in enumerate(ppr_full[:100])}
        cmap = {p: r + 1 for r, p in enumerate(cbf_full[:100])}
        cpos = {int(p): j for j, p in enumerate(cand)}
        user_total = sum(counts.get(u, {}).values()) or 1
        history_days = max((history_end - first.get(u, history_end - timedelta(days=1))).total_seconds() / 86400.0, 1.0)
        for p in chosen:
            a = stats.get(u, {}).get(p, [0, 0, 0, 0.0, None, 0])
            last = a[4]
            category = meta.get(p, {}).get("category", "")
            feat = [
                a[0], a[1], a[2], math.log1p(a[3]), a[1] / user_total, a[5],
                (history_end - last).total_seconds() / 86400.0 if last else 999.0,
                fmap.get(p, 999), float(prow[cpos[p]]), pmap.get(p, 999),
                float(crow[cpos[p]]), cmap.get(p, 999),
                math.log1p(sources["pop_counts"].get(p, 0)),
                sources["hist_exp"].get(p, 0), sources["fut_exp"].get(p, 0),
                sources["ratio"].get(p, 1), sources["cat_aff"].get(u, {}).get(category, 0),
                float(p in freq30), float(p in ppr50), float(p in cbf30), float(p in pop30),
            ]
            if schema == "v4_25":
                feat.extend([
                    a[1] / history_days, a[3] / history_days,
                    a[5] / a[1] if a[1] else 0.0, history_days,
                ])
            rows.append(feat); labels.append(int(p in targets.get(u, set()))); keys.append((u, p))
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.uint8), keys, lists


def feature_cache(name, aggregate, pred_users, candidates, meta, fit_ids, schedule_rows,
                  history_start, history_end, future_start, future_end, max_candidates,
                  schema, fp_data, timings):
    path = OUT / f"features_{name}.npz"
    expected = digest({**fp_data, "name": name, "history_start": history_start.isoformat(),
                       "history_end": history_end.isoformat(), "future_start": future_start.isoformat(),
                       "future_end": future_end.isoformat(), "max_candidates": max_candidates,
                       "schema": schema, "candidate_policy": CANDIDATE_POLICY})

    def work():
        if path.exists():
            z = np.load(path, allow_pickle=False)
            if str(z["fingerprint"]) == expected:
                keys = list(zip(z["users"].astype(str).tolist(), z["pids"].astype(int).tolist()))
                return (z["X"], z["y"], keys, json.loads(str(z["info"]))), True
        stats, counts, targets, first, _ = aggregate
        sources = source_bundle(stats, counts, pred_users, candidates, meta, fit_ids,
                                schedule_rows, history_start, history_end, future_start,
                                future_end, fp_data, name)
        x, y, keys, lists = shortlist_and_features(
            stats, counts, targets, first, pred_users, sources, meta, history_end,
            max_candidates, schema,
        )
        coverage = coverage_diagnostics(targets, pred_users, candidates, lists)
        info = {"rows": len(y), "positives": int(y.sum()), "coverage": coverage,
                "graph": sources["graph"], "ppr": sources["ppr_settings"],
                "cbf": sources["cbf_settings"]}
        np.savez_compressed(path, fingerprint=expected, X=x, y=y,
                            users=np.array([k[0] for k in keys]),
                            pids=np.array([k[1] for k in keys], dtype=np.int64),
                            info=canonical(info))
        save_json(path.with_suffix(".fingerprint.json"), {
            "fingerprint": expected, "inputs": fp_data, "feature_schema": schema,
            "features": FEATURES_V4 if schema == "v4_25" else FEATURES_V3,
            "candidate_policy": CANDIDATE_POLICY, "model_settings": HGB_PARAMS,
        })
        return (x, y, keys, info), False
    return timed(timings, f"features_{name}", work)


def coverage_diagnostics(targets, cohort, eligible, lists):
    total = eligible_total = covered = absent = missed = 0
    for u in cohort:
        rel = targets.get(u, set())
        shortlist = set(lists.get(u, []))
        total += len(rel)
        eligible_rel = rel & eligible
        eligible_total += len(eligible_rel)
        absent += len(rel - eligible)
        covered += len(eligible_rel & shortlist)
        missed += len(eligible_rel - shortlist)
    return {
        "target_items": total, "eligible_target_items": eligible_total,
        "absent_from_schedule_pool": absent, "eligible_missed_by_shortlist": missed,
        "covered_relevant_items": covered,
        "coverage_all_targets": covered / total if total else 0.0,
        "coverage_eligible_targets": covered / eligible_total if eligible_total else 0.0,
    }


def rank_model(model, x, keys, cohort, counts):
    probability = model.predict_proba(x)[:, 1]
    grouped = defaultdict(list)
    for (u, p), score in zip(keys, probability):
        grouped[u].append((p, float(score)))
    ranked = {u: [p for p, _ in sorted(grouped.get(u, []), key=lambda z: (-z[1], z[0]))] for u in cohort}
    for u in cohort:
        if not counts.get(u):
            ranked[u] = []
    return ranked


def per_user_ap(predictions, targets, cohort):
    result = {}
    for u in cohort:
        rel = targets.get(u, set())
        hits = score = 0.0
        for rank, p in enumerate(predictions[u], 1):
            if p in rel:
                hits += 1
                score += hits / rank
        result[u] = score / min(5, len(rel)) if rel else 0.0
    return result


def evaluate(name, ranked, cohort, targets, pop, candidates, comparison, predictions,
             runtime, settings):
    pred, fallback = fill(cohort, ranked, pop, candidates)
    ap = per_user_ap(pred, targets, cohort)
    score = float(sum(ap.values()) / len(cohort))
    comparison.append({"method": name, "map5": score, "runtime_sec": runtime,
                       "fallback_users": fallback, "settings": canonical(settings)})
    predictions[name] = pred
    return score, ap


def train_model_cached(name, x, y, schema, fp_data, timings):
    path = OUT / f"{name}.pkl"
    sidecar = path.with_suffix(".fingerprint.json")
    expected_data = {**fp_data, "schema": schema, "features": FEATURES_V4,
                     "model": "HistGradientBoostingClassifier", "settings": HGB_PARAMS,
                     "rows": len(y), "positives": int(y.sum())}
    expected = digest(expected_data)

    def work():
        if path.exists() and sidecar.exists():
            info = json.loads(sidecar.read_text(encoding="utf-8"))
            if info.get("fingerprint") == expected:
                return pickle.load(path.open("rb")), True
        model = HistGradientBoostingClassifier(**HGB_PARAMS)
        model.fit(x, y)
        pickle.dump(model, path.open("wb"), protocol=5)
        save_json(sidecar, {"fingerprint": expected, "fingerprint_data": expected_data})
        return model, False
    return timed(timings, name, work)


def write_table(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    wall_start = time.perf_counter()
    timings = {}
    identities = {
        "train_cache": file_identity(TRAIN_CACHE), "val_cache": file_identity(VAL_CACHE),
        "template": file_identity(RAW / "submission.csv"),
        "saved_validation_model": file_identity(V3 / "reranker_validation.pkl"),
        "saved_final_model": file_identity(V3 / "reranker_final.pkl"),
        "schedules": {k: file_identity(v) for k, v in SCHEDULES.items()},
    }
    base_fp = digest({"inputs": identities, "T": T.isoformat(), "U": U.isoformat(),
                      "thresholds": {"history": 0.50, "labels": 0.80},
                      "target_top_k": 5})
    cohort = users()
    schedule_rows, split_rows, meta = load_schedule()
    test_end = max(r[2] for r in split_rows["test"]) + timedelta(seconds=1)

    saved_model = pickle.load((V3 / "reranker_validation.pkl").open("rb"))
    snapshot_checks = []
    for name in ("snapshot_s1.npz", "snapshot_s2.npz"):
        z = np.load(V3 / name, allow_pickle=True)
        info = json.loads(str(z["info"]))
        snapshot_checks.append({"file": name, "features": int(z["X"].shape[1]),
                                "label_end": info["label_end"],
                                "strictly_before_T": datetime.fromisoformat(info["label_end"]) < T})
    saved_compatible = (
        getattr(saved_model, "n_features_in_", None) == len(FEATURES_V3)
        and all(x["features"] == len(FEATURES_V3) and x["strictly_before_T"] for x in snapshot_checks)
    )
    if not saved_compatible:
        raise RuntimeError("Saved v3 validation reranker is not temporally/schema compatible")

    outer = timed(timings, "aggregate_outer", lambda: aggregate_cached(
        "outer_T_U", T, T, U, base_fp))
    stats, counts, targets, first, outer_observed = outer
    val_candidates, _, _ = schedule_window(schedule_rows, T, U)
    fit_ids, _, _ = schedule_window(schedule_rows, ORIGIN, T)
    comparison, predictions, aps = [], {}, {}

    # Build clean outer sources once; all baseline and reranker comparisons share them.
    source_started = time.perf_counter()
    outer_sources = source_bundle(stats, counts, cohort, val_candidates, meta, fit_ids,
                                  schedule_rows, ORIGIN, T, T, U,
                                  {"base": base_fp, "cutoff": T.isoformat(), "schema": "v3_21"},
                                  "outer_clean")
    timings["outer_sources"] = {"seconds": time.perf_counter() - source_started,
                                "cache_hit": outer_sources["ppr_cache_hit"]}
    pop = outer_sources["pop"]

    t0 = time.perf_counter()
    score, ap = evaluate("popularity_clean", {}, cohort, targets, pop, val_candidates,
                         comparison, predictions, time.perf_counter() - t0,
                         {"cutoff": T.isoformat(), "recomputed": True})
    aps["popularity_clean"] = ap
    t0 = time.perf_counter()
    freq_rank = {u: frequency_rank(counts, u, val_candidates) for u in cohort}
    score, ap = evaluate("frequency_clean", freq_rank, cohort, targets, pop, val_candidates,
                         comparison, predictions, time.perf_counter() - t0, {"threshold": 0.50})
    aps["frequency_clean"] = ap
    t0 = time.perf_counter()
    cbf_rank = {u: top_ids(sparse_row(outer_sources["cbf"], i, len(outer_sources["cand"])),
                            outer_sources["cand"], len(outer_sources["cand"]), True)
                for i, u in enumerate(cohort)}
    score, ap = evaluate("cbf_clean", cbf_rank, cohort, targets, pop, val_candidates,
                         comparison, predictions, time.perf_counter() - t0,
                         outer_sources["cbf_settings"])
    aps["cbf_clean"] = ap
    t0 = time.perf_counter()
    ppr_rank = {u: top_ids(outer_sources["ppr"][i], outer_sources["cand"],
                            len(outer_sources["cand"]), True) for i, u in enumerate(cohort)}
    score, ap = evaluate("ppr_a0.75_clean", ppr_rank, cohort, targets, pop, val_candidates,
                         comparison, predictions, time.perf_counter() - t0,
                         outer_sources["ppr_settings"])
    aps["ppr_a0.75_clean"] = ap

    # Outer feature matrices share exactly the source scores above.
    def outer_features(max_candidates, schema, name):
        path = OUT / f"features_{name}.npz"
        expected_data = {"base": base_fp, "name": name, "cutoff": T.isoformat(),
                         "label_end": U.isoformat(), "schema": schema,
                         "features": FEATURES_V4 if schema == "v4_25" else FEATURES_V3,
                         "candidate_policy": CANDIDATE_POLICY,
                         "model_settings": HGB_PARAMS}
        expected = digest(expected_data)
        def work():
            if path.exists():
                z = np.load(path, allow_pickle=False)
                if str(z["fingerprint"]) == expected:
                    keys = list(zip(z["users"].astype(str).tolist(), z["pids"].astype(int).tolist()))
                    lists = defaultdict(list)
                    for u, p in keys: lists[u].append(p)
                    return (z["X"], z["y"], keys, dict(lists), json.loads(str(z["coverage"]))), True
            x, y, keys, lists = shortlist_and_features(
                stats, counts, targets, first, cohort, outer_sources, meta, T,
                max_candidates, schema)
            coverage = coverage_diagnostics(targets, cohort, val_candidates, lists)
            np.savez_compressed(path, fingerprint=expected, X=x, y=y,
                                users=np.array([k[0] for k in keys]),
                                pids=np.array([k[1] for k in keys], dtype=np.int64),
                                coverage=canonical(coverage))
            save_json(path.with_suffix(".fingerprint.json"), {
                "fingerprint": expected, "fingerprint_data": expected_data,
                "input_identities": identities,
            })
            return (x, y, keys, lists, coverage), False
        return timed(timings, name, work)

    x100, y100, keys100, lists100, cov100 = outer_features(100, "v3_21", "outer_saved100")
    x200, y200, keys200, lists200, cov200 = outer_features(200, "v3_21", "outer_saved200")
    if any(lists100[u] != lists200[u][:len(lists100[u])] for u in cohort):
        raise AssertionError("200-candidate shortlist is not a nested extension of 100")

    t0 = time.perf_counter()
    saved100_rank = rank_model(saved_model, x100, keys100, cohort, counts)
    saved100_score, ap = evaluate("saved_reranker_100_clean", saved100_rank, cohort, targets,
                                  pop, val_candidates, comparison, predictions,
                                  time.perf_counter() - t0,
                                  {"model": "tuning_v3/reranker_validation.pkl", "schema": "v3_21", "candidates": 100})
    aps["saved_reranker_100_clean"] = ap
    t0 = time.perf_counter()
    saved200_rank = rank_model(saved_model, x200, keys200, cohort, counts)
    saved200_score, ap = evaluate("saved_reranker_200_clean", saved200_rank, cohort, targets,
                                  pop, val_candidates, comparison, predictions,
                                  time.perf_counter() - t0,
                                  {"model": "tuning_v3/reranker_validation.pkl", "schema": "v3_21", "candidates": 200})
    aps["saved_reranker_200_clean"] = ap

    # Exactly one new validation reranker, trained on the two requested long-horizon snapshots.
    snapshot_specs = [
        ("long_apr05_T", datetime(2020, 4, 5), datetime(2020, 4, 5), T),
        ("long_apr19_T", datetime(2020, 4, 19), datetime(2020, 4, 19), T),
    ]
    parts_x, parts_y, snapshot_info = [], [], []
    for name, history_end, label_start, label_end in snapshot_specs:
        agg = timed(timings, f"aggregate_{name}", lambda n=name, h=history_end, a=label_start, b=label_end:
                    aggregate_cached(n, h, a, b, base_fp))
        sstats, scounts, stargets, sfirst, sobserved = agg
        scandidates, _, _ = schedule_window(schedule_rows, label_start, label_end)
        sfit, _, _ = schedule_window(schedule_rows, ORIGIN, history_end)
        pred_users = sorted(set(stargets) & set(scounts))
        x, y, keys, info = feature_cache(
            name, agg, pred_users, scandidates, meta, sfit, schedule_rows,
            ORIGIN, history_end, label_start, label_end, 200, "v4_25",
            {"base": base_fp, "snapshot": name, "cutoff": history_end.isoformat(),
             "label_end": label_end.isoformat()}, timings)
        parts_x.append(x); parts_y.append(y)
        snapshot_info.append({"name": name, "history_end": history_end.isoformat(sep=" "),
                              "label_start": label_start.isoformat(sep=" "),
                              "label_end": label_end.isoformat(sep=" "),
                              "users": len(pred_users), **info, "boundary_observed": sobserved})
    train_x, train_y = np.vstack(parts_x), np.concatenate(parts_y)
    new_model = train_model_cached(
        "reranker_validation_v4", train_x, train_y, "v4_25",
        {"base": base_fp,
         "snapshots": [[n, h.isoformat(), a.isoformat(), b.isoformat()]
                       for n, h, a, b in snapshot_specs],
         "candidate_policy": CANDIDATE_POLICY}, timings)
    xnew, ynew, keysnew, listsnew, covnew = outer_features(200, "v4_25", "outer_new200")
    t0 = time.perf_counter()
    new_rank = rank_model(new_model, xnew, keysnew, cohort, counts)
    new_score, ap = evaluate("new_reranker_200_clean", new_rank, cohort, targets,
                             pop, val_candidates, comparison, predictions,
                             time.perf_counter() - t0,
                             {"model": "HGB", "schema": "v4_25", "candidates": 200,
                              "training_snapshots": 2, **HGB_PARAMS})
    aps["new_reranker_200_clean"] = ap

    candidates_to_select = {
        "saved_reranker_100_clean": saved100_score,
        "saved_reranker_200_clean": saved200_score,
        "new_reranker_200_clean": new_score,
    }
    best_name = max(candidates_to_select, key=lambda n: (candidates_to_select[n], n))
    improves = candidates_to_select[best_name] > saved100_score + 1e-12
    selected = best_name if improves else "existing_scored_submission_tuning_v3_reranker"

    # Export validation predictions and per-user AP/differences.
    validation_rows = []
    for u in cohort:
        row = {"user_id": u}
        for name, pred in predictions.items(): row[name] = " ".join(map(str, pred[u]))
        validation_rows.append(row)
    write_table(OUT / "validation_predictions.csv", validation_rows)
    ap_rows = []
    for u in cohort:
        row = {"user_id": u, "target_count": len(targets.get(u, set()))}
        for name in aps: row[f"ap_{name}"] = aps[name][u]
        row["diff_saved200_minus_saved100"] = aps["saved_reranker_200_clean"][u] - aps["saved_reranker_100_clean"][u]
        row["diff_new200_minus_saved100"] = aps["new_reranker_200_clean"][u] - aps["saved_reranker_100_clean"][u]
        ap_rows.append(row)
    write_table(OUT / "per_user_ap.csv", ap_rows)
    target_rows = [{"user_id": u, "targets": " ".join(map(str, sorted(targets.get(u, set())))),
                    "target_count": len(targets.get(u, set()))} for u in cohort]
    write_table(OUT / "validation_targets.csv", target_rows)

    final_submission = None
    submission_checks = None
    final_info = {}
    if improves:
        final_agg = timed(timings, "aggregate_final_U", lambda: aggregate_cached(
            "final_U", U, None, None, base_fp))
        fstats, fcounts, _, ffirst, fobserved = final_agg
        test_candidates = {r[3] for r in set(split_rows["test"]) if r[3] > 0}
        final_fit, _, _ = schedule_window(schedule_rows, ORIGIN, U)
        final_sources = source_bundle(
            fstats, fcounts, cohort, test_candidates, meta, final_fit, schedule_rows,
            ORIGIN, U, U, test_end,
            {"base": base_fp, "cutoff": U.isoformat(), "selected": best_name}, "final_U")
        if best_name == "saved_reranker_200_clean":
            final_model = pickle.load((V3 / "reranker_final.pkl").open("rb"))
            if getattr(final_model, "n_features_in_", None) != len(FEATURES_V3):
                raise RuntimeError("Saved final model schema mismatch")
            schema = "v3_21"
            final_info["model"] = "reused tuning_v3/reranker_final.pkl"
        else:
            final_specs = [
                ("final_jun14_U", datetime(2020, 6, 14), datetime(2020, 6, 14), U),
                ("final_jun28_U", datetime(2020, 6, 28), datetime(2020, 6, 28), U),
            ]
            final_parts_x, final_parts_y, final_snapshots = [], [], []
            for name, history_end, label_start, label_end in final_specs:
                agg = timed(timings, f"aggregate_{name}", lambda n=name, h=history_end, a=label_start, b=label_end:
                            aggregate_cached(n, h, a, b, base_fp))
                qstats, qcounts, qtargets, qfirst, qobserved = agg
                qcandidates, _, _ = schedule_window(schedule_rows, label_start, label_end)
                qfit, _, _ = schedule_window(schedule_rows, ORIGIN, history_end)
                pred_users = sorted(set(qtargets) & set(qcounts))
                qx, qy, qkeys, qinfo = feature_cache(
                    name, agg, pred_users, qcandidates, meta, qfit, schedule_rows,
                    ORIGIN, history_end, label_start, label_end, 200, "v4_25",
                    {"base": base_fp, "snapshot": name, "cutoff": history_end.isoformat(),
                     "label_end": label_end.isoformat()}, timings)
                final_parts_x.append(qx); final_parts_y.append(qy)
                final_snapshots.append({"name": name, "history_end": history_end.isoformat(sep=" "),
                                        "label_end": label_end.isoformat(sep=" "), **qinfo,
                                        "boundary_observed": qobserved})
            final_model = train_model_cached(
                "reranker_final_v4", np.vstack(final_parts_x), np.concatenate(final_parts_y),
                "v4_25", {"base": base_fp,
                           "snapshots": [[n, h.isoformat(), a.isoformat(), b.isoformat()]
                                         for n, h, a, b in final_specs],
                           "candidate_policy": CANDIDATE_POLICY}, timings)
            schema = "v4_25"
            final_info.update({"model": "reranker_final_v4.pkl", "snapshots": final_snapshots})
        fx, fy, fkeys, flists = shortlist_and_features(
            fstats, fcounts, {}, ffirst, cohort, final_sources, meta, U, 200, schema)
        final_rank = rank_model(final_model, fx, fkeys, cohort, fcounts)
        final_pred, fallback = fill(cohort, final_rank, final_sources["pop"], test_candidates)
        safe = "saved200" if best_name == "saved_reranker_200_clean" else "new200"
        final_submission = f"submission_temporal_v4_{safe}.csv"
        write_pred(OUT / final_submission, final_pred, cohort)
        with (OUT / final_submission).open(encoding="utf-8", newline="") as f:
            submitted = list(csv.DictReader(f))
        submission_checks = {
            "rows": len(submitted),
            "template_order": [r["user_id"] for r in submitted] == cohort,
            "complete_coverage": len(submitted) == len(cohort),
            "five_distinct_positive_ids": all(len(v := r["tv_show_id"].split()) == 5 and len(set(v)) == 5 and all(int(x) > 0 for x in v) for r in submitted),
            "schedule_available": all(all(int(x) in test_candidates for x in r["tv_show_id"].split()) for r in submitted),
            "history_strictly_before_U": datetime.fromisoformat(fobserved["history_max_airing_end"]) < U,
            "no_test_period_viewing": True,
            "fallback_users": fallback,
        }
        final_info["history_boundary_observed"] = fobserved

    mean_differences = {
        "saved200_minus_saved100": saved200_score - saved100_score,
        "new200_minus_saved100": new_score - saved100_score,
    }
    coverage = {"saved100": cov100, "saved200": cov200, "new200": covnew}
    boundary_assertions = {
        "T": T.isoformat(sep=" "), "U": U.isoformat(sep=" "),
        "validation_history_rule": "airing_end < T",
        "validation_label_rule": "T <= airing_end < U",
        "final_history_rule": "airing_end < U",
        "outer_observed": outer_observed,
        "history_strictly_before_T": datetime.fromisoformat(outer_observed["history_max_airing_end"]) < T,
        "labels_start_at_or_after_T": datetime.fromisoformat(outer_observed["label_min_airing_end"]) >= T,
        "labels_strictly_before_U": datetime.fromisoformat(outer_observed["label_max_airing_end"]) < U,
        "all_1260_users": len(cohort) == 1260,
        "empty_targets": sum(not targets.get(u) for u in cohort),
        "saved_model_compatibility": {"compatible": saved_compatible, "snapshots": snapshot_checks,
                                      "model_features": int(saved_model.n_features_in_)},
        "nested_200_preserves_100": True,
    }
    comparison_map = {r["method"]: r["map5"] for r in comparison}
    summary = {
        "version": "temporal_v4", "selected_method": selected,
        "clean_baseline": comparison_map, "mean_differences": mean_differences,
        "coverage": coverage, "comparison": comparison,
        "boundaries": boundary_assertions, "snapshots": snapshot_info,
        "feature_schemas": {"saved": FEATURES_V3, "new": FEATURES_V4},
        "candidate_policy": CANDIDATE_POLICY, "fingerprint": base_fp,
        "input_identities": identities, "model_settings": HGB_PARAMS,
        "final_submission": final_submission, "final_info": final_info,
        "submission_checks": submission_checks,
        "runtime_total_sec": time.perf_counter() - wall_start,
        "timings": timings,
        "libraries": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
    }
    write_table(OUT / "comparison.csv", comparison)
    save_json(OUT / "coverage_diagnostics.json", coverage)
    save_json(OUT / "boundary_assertions.json", boundary_assertions)
    save_json(OUT / "timings.json", timings)
    save_json(OUT / "submission_checks.json", submission_checks)
    save_json(OUT / "summary.json", summary)
    print(json.dumps({"selected": selected, "scores": comparison_map, "coverage": coverage,
                      "submission": final_submission, "checks": submission_checks,
                      "runtime_total_sec": summary["runtime_total_sec"]}, indent=2))


if __name__ == "__main__":
    main()
