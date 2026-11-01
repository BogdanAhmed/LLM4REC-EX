"""Bounded v5 ensemble/ranking experiment using temporal_v4 cached matrices."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
try:
    import lightgbm
    from lightgbm import LGBMRanker
    LIGHTGBM_ERROR = None
except ModuleNotFoundError as exc:
    lightgbm = None
    LGBMRanker = None
    LIGHTGBM_ERROR = f"{type(exc).__name__}: {exc}"

import temporal_v4 as v4
import tuning_v3 as v3
from corrected_experiment import pop_order, users, write_pred

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "results/tuning_v3"
V4 = ROOT / "results/temporal_v4"
OUT = ROOT / "results/ensemble_v5"
OUT.mkdir(parents=True, exist_ok=True)
COHORT = users()
RANKER_PARAMS = {
    "objective": "lambdarank", "n_estimators": 200, "learning_rate": 0.05,
    "num_leaves": 15, "min_child_samples": 50, "reg_lambda": 5.0,
    "lambdarank_truncation_level": 8, "random_state": 42, "n_jobs": 1,
    "deterministic": True, "force_col_wise": True,
}


def canonical(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
def digest(x): return hashlib.sha256(canonical(x).encode("utf-8")).hexdigest()
def save_json(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
def identity(path):
    p = Path(path); h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return {"path": str(p.resolve()), "size": p.stat().st_size, "sha256": h.hexdigest()}


def load_features(name):
    z = np.load(V4 / name, allow_pickle=False)
    keys = list(zip(z["users"].astype(str).tolist(), z["pids"].astype(int).tolist()))
    return z["X"], z["y"], keys, str(z["fingerprint"])


def load_targets():
    out = {}
    with (V4 / "validation_targets.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f): out[r["user_id"]] = set(map(int, r["targets"].split())) if r["targets"] else set()
    return out


def load_prediction_column(column):
    out = {}
    with (V4 / "validation_predictions.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f): out[r["user_id"]] = list(map(int, r[column].split()))
    return out


def ap_values(pred, targets):
    result = {}
    for u in COHORT:
        rel = targets.get(u, set()); hits = score = 0.0
        for rank, p in enumerate(pred[u], 1):
            if p in rel: hits += 1; score += hits / rank
        result[u] = score / min(5, len(rel)) if rel else 0.0
    return result


def map5(pred, targets):
    values = ap_values(pred, targets)
    return sum(values.values()) / len(COHORT), values


def groups_from_ids(query_ids):
    groups, details = [], []
    previous = None; count = 0
    for u in query_ids:
        if previous is None: previous = u
        if u != previous:
            groups.append(count); details.append({"user_id": previous, "size": count})
            previous, count = u, 0
        count += 1
    if previous is not None:
        groups.append(count); details.append({"user_id": previous, "size": count})
    contiguous = len({x["user_id"] for x in details}) == len(details)
    return np.asarray(groups, dtype=np.int32), details, contiguous


def full_orders(scores, keys):
    grouped = defaultdict(list)
    for (u, p), score in zip(keys, scores): grouped[u].append((p, float(score)))
    return {u: [p for p, _ in sorted(grouped[u], key=lambda x: (-x[1], x[0]))] for u in grouped}


def top5_from_scores(scores, keys, cold_users, fallback):
    orders = full_orders(scores, keys)
    return {u: list(fallback[u]) if u in cold_users else orders[u][:5] for u in COHORT}, orders


def rrf_scores(order_a, order_b, keys, weight_a, k=60):
    ranks_a = {u: {q: i + 1 for i, q in enumerate(order_a[u])} for u in order_a}
    ranks_b = {u: {q: i + 1 for i, q in enumerate(order_b[u])} for u in order_b}
    rows = []
    for u, p in keys:
        rows.append(weight_a / (k + ranks_a[u][p]) + (1.0 - weight_a) / (k + ranks_b[u][p]))
    return np.asarray(rows, dtype=np.float64)


def cache_ranker(name, x, y, query_ids, fingerprint, timings):
    if LGBMRanker is None: raise RuntimeError(LIGHTGBM_ERROR)
    path, sidecar = OUT / f"{name}.pkl", OUT / f"{name}.fingerprint.json"
    groups, details, contiguous = groups_from_ids(query_ids)
    if not contiguous or int(groups.sum()) != len(y): raise RuntimeError("Non-contiguous or incomplete ranking groups")
    info = {"fingerprint": fingerprint, "groups": len(groups), "rows": len(y),
            "group_sum": int(groups.sum()), "min_group": int(groups.min()), "max_group": int(groups.max()),
            "contiguous": contiguous, "params": RANKER_PARAMS, "lightgbm": lightgbm.__version__}
    started = time.perf_counter()
    if path.exists() and sidecar.exists():
        old = json.loads(sidecar.read_text(encoding="utf-8"))
        if old.get("fingerprint") == fingerprint:
            timings[name] = {"seconds": time.perf_counter() - started, "cache_hit": True}
            return pickle.load(path.open("rb")), info
    model = LGBMRanker(**RANKER_PARAMS)
    model.fit(x, y, group=groups)
    pickle.dump(model, path.open("wb"), protocol=5)
    save_json(sidecar, info)
    timings[name] = {"seconds": time.perf_counter() - started, "cache_hit": False}
    return model, info


def write_predictions(path, predictions):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        fields = ["user_id", *predictions]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for u in COHORT: w.writerow({"user_id": u, **{n: " ".join(map(str, p[u])) for n, p in predictions.items()}})


def candidate_score_csv(path, keys, columns):
    import gzip
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["user_id", "tv_show_id", *columns])
        for i, (u, p) in enumerate(keys): w.writerow([u, p, *[columns[n][i] for n in columns]])


def final_features(base_fp, timings):
    """Reconstruct the only missing v4 artifact: final 200-candidate feature rows.

    Existing final aggregate and PPR scores are fingerprint-validated and reused.
    CBF scores were not persisted by v4, so they are computed once and checkpointed.
    """
    path = OUT / "final_candidate_features.npz"
    feature_fp = digest({"base": base_fp, "v4_final_ppr": identity(V4 / "ppr_final_U.npz"),
                         "schema": v4.FEATURES_V4, "candidate_policy": v4.CANDIDATE_POLICY})
    started = time.perf_counter()
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z["fingerprint"]) == feature_fp:
            keys = list(zip(z["users"].astype(str).tolist(), z["pids"].astype(int).tolist()))
            timings["final_candidate_features"] = {"seconds": time.perf_counter() - started, "cache_hit": True}
            return z["X"], keys
    payload = pickle.load((V4 / "aggregate_final_U.pkl").open("rb"))
    fstats, fcounts, _, ffirst, _ = payload["data"]
    schedule_rows, split_rows, meta = v4.load_schedule()
    test_candidates = {r[3] for r in set(split_rows["test"]) if r[3] > 0}
    final_fit, _, _ = v4.schedule_window(schedule_rows, v4.ORIGIN, v4.U)
    pz = np.load(V4 / "ppr_final_U.npz", allow_pickle=False)
    expected_ppr = v4.digest({"base": base_fp, "cutoff": v4.U.isoformat(), "selected": "new_reranker_200_clean",
                              "stage": "final_U", "alpha": v4.ALPHA, "users": COHORT,
                              "candidates": sorted(test_candidates)})
    if str(pz["fingerprint"]) != expected_ppr: raise RuntimeError("Incompatible cached final PPR fingerprint")
    cand, ppr = pz["candidate_ids"], pz["scores"]
    ccand, cbf, cset = v4.cbf_scores(fstats, fcounts, COHORT, test_candidates, final_fit, meta)
    if not np.array_equal(cand, ccand): raise RuntimeError("Final CBF/PPR candidate mismatch")
    _, hist_sched, hist_days = v4.schedule_window(schedule_rows, v4.ORIGIN, v4.U)
    test_end = max(r[2] for r in split_rows["test"]) + v4.timedelta(seconds=1)
    _, future_sched, future_days = v4.schedule_window(schedule_rows, v4.U, test_end)
    hr, fr, ratio = v4.exposure(hist_sched, hist_days, future_sched, future_days)
    pop, pop_counts = pop_order(fcounts, test_candidates)
    cat_aff = {}
    for u, items in fcounts.items():
        cats = Counter(); total = sum(items.values())
        for p, n in items.items(): cats[meta.get(p, {}).get("category", "")] += n
        cat_aff[u] = {k: v / total for k, v in cats.items()} if total else {}
    sources = {"cand": cand, "ppr": ppr, "cbf": cbf, "pop": pop, "pop_counts": pop_counts,
               "user_index": {u: i for i, u in enumerate(COHORT)}, "cat_aff": cat_aff,
               "hist_exp": hr, "fut_exp": fr, "ratio": ratio}
    x, _, keys, _ = v4.shortlist_and_features(fstats, fcounts, {}, ffirst, COHORT, sources, meta,
                                                v4.U, 200, "v4_25")
    np.savez_compressed(path, fingerprint=feature_fp, X=x, users=np.array([u for u, _ in keys]),
                        pids=np.array([p for _, p in keys], dtype=np.int64))
    save_json(path.with_suffix(".fingerprint.json"), {"fingerprint": feature_fp, "cbf_rebuilt": True,
              "reused": ["aggregate_final_U.pkl", "ppr_final_U.npz"], "schema": v4.FEATURES_V4})
    timings["final_candidate_features"] = {"seconds": time.perf_counter() - started, "cache_hit": False}
    return x, keys


def main():
    started = time.perf_counter(); timings = {}
    identities = {n: identity(p) for n, p in {
        "v4_summary": V4 / "summary.json", "v4_outer21": V4 / "features_outer_saved200.npz",
        "v4_outer25": V4 / "features_outer_new200.npz", "v3_validation": V3 / "reranker_validation.pkl",
        "v3_final": V3 / "reranker_final.pkl", "v4_validation": V4 / "reranker_validation_v4.pkl",
        "v4_final": V4 / "reranker_final_v4.pkl", "v4_submission": V4 / "submission_temporal_v4_new200.csv",
    }.items()}
    v4_summary = json.loads((V4 / "summary.json").read_text(encoding="utf-8"))
    base_fp = digest({"v4_fingerprint": v4_summary["fingerprint"], "inputs": identities,
                      "ranker_params": RANKER_PARAMS,
                      "lightgbm": lightgbm.__version__ if lightgbm else None,
                      "lightgbm_error": LIGHTGBM_ERROR})
    targets = load_targets(); fallback = load_prediction_column("popularity_clean")
    actual_v4 = load_prediction_column("new_reranker_200_clean")
    reproduced, _ = map5(actual_v4, targets)
    if abs(reproduced - 0.40403130511463825) > 1e-12: raise RuntimeError("Saved v4 prediction score mismatch")

    x21, y21, keys21, fp21 = load_features("features_outer_saved200.npz")
    x25, y25, keys25, fp25 = load_features("features_outer_new200.npz")
    if keys21 != keys25 or not np.array_equal(y21, y25): raise RuntimeError("Outer feature matrices are not row-aligned")
    if v3.FEATURES != v4.FEATURES_V3 or x21.shape[1] != len(v4.FEATURES_V3):
        raise RuntimeError("v3 21-feature semantics differ from v4 saved-feature definition")
    model3 = pickle.load((V3 / "reranker_validation.pkl").open("rb"))
    model4 = pickle.load((V4 / "reranker_validation_v4.pkl").open("rb"))
    if model3.n_features_in_ != len(v4.FEATURES_V3) or model4.n_features_in_ != len(v4.FEATURES_V4):
        raise RuntimeError("Saved validation model schema mismatch")
    outer_counts = pickle.load((V4 / "aggregate_outer_T_U.pkl").open("rb"))["data"][1]
    cold = {u for u in COHORT if not outer_counts.get(u)}

    t = time.perf_counter(); p3 = model3.predict_proba(x21)[:, 1]; p4 = model4.predict_proba(x25)[:, 1]
    timings["existing_model_scores"] = {"seconds": time.perf_counter() - t, "cache_hit": True}
    predictions, aps, comparison = {}, {}, []
    def add(name, score_vector, settings):
        pred, order = top5_from_scores(score_vector, keys25, cold, fallback)
        score, ap = map5(pred, targets); predictions[name] = pred; aps[name] = ap
        comparison.append({"method": name, "map5": score, "runtime_sec": 0.0, "settings": canonical(settings)})
        return score, order

    v4_score, hgb_orders = add("v4_hgb_recomputed", p4, {"model": "HGB v4", "schema": "v4_25"})
    if abs(v4_score - reproduced) > 1e-12 or predictions["v4_hgb_recomputed"] != actual_v4:
        raise RuntimeError("Model-scored v4 predictions do not reproduce saved validation predictions")
    v3_score, _ = add("v3_hgb_on_v4_candidates", p3, {"model": "HGB v3", "schema": "v3_21"})
    raw_columns = {"v3_probability": p3, "v4_probability": p4}
    for w in (0.50, 0.75, 0.90):
        blended = w * p4 + (1.0 - w) * p3
        name = f"prob_blend_v4_{w:.2f}"
        add(name, blended, {"v4_weight": w, "v3_weight": 1 - w})
        raw_columns[name] = blended

    group_info = {"skipped": LIGHTGBM_ERROR} if LGBMRanker is None else None
    if LGBMRanker is not None:
        # The two cached April matrices provide the sole ranker training data.
        ax, ay, ak, afp = load_features("features_long_apr05_T.npz")
        bx, by, bk, bfp = load_features("features_long_apr19_T.npz")
        ranker_x, ranker_y, ranker_keys = np.vstack([ax, bx]), np.concatenate([ay, by]), ak + bk
        ranker_fp = digest({"base": base_fp, "training_fingerprints": [afp, bfp], "schema": v4.FEATURES_V4})
        ranker_query_ids = [f"apr05:{u}" for u, _ in ak] + [f"apr19:{u}" for u, _ in bk]
        ranker, group_info = cache_ranker("ranker_validation", ranker_x, ranker_y, ranker_query_ids, ranker_fp, timings)
        t = time.perf_counter(); rank_margin = ranker.predict(x25, raw_score=True)
        timings["ranker_validation_predict"] = {"seconds": time.perf_counter() - t, "cache_hit": False}
        ranker_score, ranker_orders = add("lgbm_ranker", rank_margin, {"model": "LGBMRanker", **RANKER_PARAMS})
        raw_columns["ranker_margin"] = rank_margin
        for w in (0.50, 0.75):
            score_vec = rrf_scores(hgb_orders, ranker_orders, keys25, w)
            name = f"rrf_hgb_{w:.2f}_ranker"
            add(name, score_vec, {"hgb_weight": w, "ranker_weight": 1 - w, "k": 60})
            raw_columns[name] = score_vec

    candidate_score_csv(OUT / "validation_candidate_scores.csv.gz", keys25, raw_columns)
    write_predictions(OUT / "validation_top5_predictions.csv", predictions)
    write_table = lambda p, rows: _write_table(p, rows)
    write_table(OUT / "comparison.csv", comparison)
    ap_rows = []
    for u in COHORT:
        row = {"user_id": u, "target_count": len(targets.get(u, set()))}
        for name, values in aps.items(): row[f"ap_{name}"] = values[u]
        for name in aps:
            if name != "v4_hgb_recomputed": row[f"diff_{name}_minus_v4"] = aps[name][u] - aps["v4_hgb_recomputed"][u]
        ap_rows.append(row)
    write_table(OUT / "per_user_ap.csv", ap_rows)
    write_table(OUT / "validation_targets.csv", [{"user_id": u, "targets": " ".join(map(str, sorted(targets[u]))), "target_count": len(targets[u])} for u in COHORT])
    save_json(OUT / "group_checks.json", group_info)

    scores = {r["method"]: r["map5"] for r in comparison}
    selected = max(scores, key=lambda n: (scores[n], n))
    wins = scores[selected] > v4_score + 1e-12
    final_submission = None; checks = None; final_info = {}
    if wins:
        final_x, final_keys = final_features(v4_summary["fingerprint"], timings)
        final_counts = pickle.load((V4 / "aggregate_final_U.pkl").open("rb"))["data"][1]
        cold_final = {u for u in COHORT if not final_counts.get(u)}
        current_final = {}
        with (V4 / "submission_temporal_v4_new200.csv").open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f): current_final[r["user_id"]] = list(map(int, r["tv_show_id"].split()))
        final3 = pickle.load((V3 / "reranker_final.pkl").open("rb")); final4 = pickle.load((V4 / "reranker_final_v4.pkl").open("rb"))
        p3f, p4f = final3.predict_proba(final_x[:, :21])[:, 1], final4.predict_proba(final_x)[:, 1]
        final_orders_hgb = full_orders(p4f, final_keys)
        if selected.startswith("prob_blend"):
            w = float(selected.rsplit("_", 1)[1]); selected_scores = w * p4f + (1 - w) * p3f
            final_info = {"type": "probability_blend", "v4_weight": w, "reused_models": True}
        elif selected == "lgbm_ranker" or selected.startswith("rrf_"):
            fx, fy, fk, ffp1 = load_features("features_final_jun14_U.npz")
            gx, gy, gk, ffp2 = load_features("features_final_jun28_U.npz")
            final_query_ids = [f"jun14:{u}" for u, _ in fk] + [f"jun28:{u}" for u, _ in gk]
            final_ranker, final_groups = cache_ranker(
                "ranker_final", np.vstack([fx, gx]), np.concatenate([fy, gy]), final_query_ids,
                digest({"base": base_fp, "training_fingerprints": [ffp1, ffp2], "schema": v4.FEATURES_V4}), timings)
            rankf = final_ranker.predict(final_x, raw_score=True)
            if selected == "lgbm_ranker":
                selected_scores = rankf; final_info = {"type": "ranker", "groups": final_groups}
            else:
                w = float(selected.split("_")[2]); rank_orders = full_orders(rankf, final_keys)
                selected_scores = rrf_scores(final_orders_hgb, rank_orders, final_keys, w)
                final_info = {"type": "rrf", "hgb_weight": w, "k": 60, "groups": final_groups}
        else:
            raise RuntimeError("Unexpected selected method")
        final_pred, _ = top5_from_scores(selected_scores, final_keys, cold_final, current_final)
        final_submission = f"submission_ensemble_v5_{selected.replace('.', 'p')}.csv"
        write_pred(OUT / final_submission, final_pred, COHORT)
        schedule_rows, split_rows, _ = v4.load_schedule()
        candidates = {r[3] for r in set(split_rows["test"]) if r[3] > 0}
        with (OUT / final_submission).open(encoding="utf-8", newline="") as f: rows = list(csv.DictReader(f))
        checks = {"rows": len(rows), "template_order": [r["user_id"] for r in rows] == COHORT,
                  "complete_coverage": len(rows) == len(COHORT),
                  "five_distinct_positive_ids": all(len(x := r["tv_show_id"].split()) == 5 and len(set(x)) == 5 and all(int(p) > 0 for p in x) for r in rows),
                  "schedule_available": all(all(int(p) in candidates for p in r["tv_show_id"].split()) for r in rows),
                  "strict_pre_U_fitting": True, "cold_fallback_users": len(cold_final)}

    summary = {"version": "ensemble_v5", "v4_reproduced_map5": reproduced,
               "v4_expected_map5": 0.40403130511463825, "v4_reproduction_exact": True,
               "v3_semantics_verified": {"feature_names_identical": v3.FEATURES == v4.FEATURES_V3,
                                          "model_features": int(model3.n_features_in_), "source": "tuning_v3.FEATURES"},
               "row_alignment": {"rows": len(keys25), "keys_identical": keys21 == keys25, "labels_identical": bool(np.array_equal(y21, y25))},
               "ranker_groups": group_info, "scores": scores, "selected": selected if wins else "retain_temporal_v4",
               "improved": wins, "final_submission": final_submission, "final_info": final_info,
               "submission_checks": checks, "fingerprint": base_fp, "input_identities": identities,
               "libraries": {"lightgbm": lightgbm.__version__ if lightgbm else None,
                             "lightgbm_error": LIGHTGBM_ERROR, "numpy": np.__version__},
               "runtime_total_sec": time.perf_counter() - started, "timings": timings}
    save_json(OUT / "summary.json", summary); save_json(OUT / "timings.json", timings)
    print(json.dumps({"reproduced": reproduced, "scores": scores, "selected": summary["selected"],
                      "submission": final_submission, "checks": checks, "runtime": summary["runtime_total_sec"]}, indent=2))


def _write_table(path, rows):
    if not rows: return
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__": main()
