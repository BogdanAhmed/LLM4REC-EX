"""Streaming SWEET.TV preparation and deterministic popularity baselines."""
from __future__ import annotations
import bisect, csv, gzip, json, time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "baseline"
OUT.mkdir(parents=True, exist_ok=True)
ENC, TS, SCH_TS = "utf-8-sig", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M:%S"
LOG = RAW / "dataset11-30.csv"
SCHEDULES = {"train": RAW / "export_arh_11-20-final.csv", "val": RAW / "export_arh_21-30-final.csv", "test": RAW / "export_arh_31-42-final.csv"}

def dt(s, fmt=TS): return datetime.strptime(s, fmt)

def find_overlaps(rows, starts, prefix_max_ends, view_start, view_end):
    """Return schedule row indices satisfying strict interval overlap."""
    left = bisect.bisect_right(prefix_max_ends, view_start)
    right = bisect.bisect_left(starts, view_end)
    return [j for j in range(left, right) if rows[j][0] < view_end and rows[j][1] > view_start]

def schedule_index(path):
    by_channel, seen, items, title_example, bounds = defaultdict(list), set(), {}, None, [None, None]
    with path.open(encoding=ENC, newline="", errors="strict") as f:
        for r in csv.DictReader(f):
            pid, start, dur = int(r["tv_show_id"]), dt(r["start_time"], SCH_TS), int(r["duration"])
            bounds[0] = start if bounds[0] is None or start < bounds[0] else bounds[0]
            end_dt = start.timestamp() + dur
            bounds[1] = end_dt if bounds[1] is None or end_dt > bounds[1] else bounds[1]
            key = (int(r["channel_id"]), start, dur, pid)
            if key in seen: continue
            seen.add(key)
            if pid and title_example is None: title_example = r["tv_show_title"]
            by_channel[int(r["channel_id"])].append((start.timestamp(), start.timestamp()+dur, pid, r["tv_show_title"], r["tv_show_category"]))
            if pid: items.setdefault(pid, (r["tv_show_title"], r["tv_show_category"]))
    for rows in by_channel.values(): rows.sort(key=lambda x: x[0])
    return by_channel, items, title_example, bounds

def join_views(schedule_path, out_path, cutoff=None):
    idx, _, example, _ = schedule_index(schedule_path); starts = {}; prefix = {}
    for ch, rows in idx.items():
        starts[ch] = [x[0] for x in rows]; high = []; current = float("-inf")
        for row in rows:
            current = max(current, row[1]); high.append(current)
        prefix[ch] = high
    best = {}
    with LOG.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            vs, ve = dt(r["start_time"]).timestamp(), dt(r["stop_time"]).timestamp()
            if cutoff is not None and vs >= cutoff: continue
            ve = min(ve, cutoff) if cutoff is not None else ve; ch = int(r["vsetv_id"]); rows = idx.get(ch)
            if not rows or ve <= vs: continue
            for j in find_overlaps(rows, starts[ch], prefix[ch], vs, ve):
                a, b, pid, _, _ = rows[j]
                ov = min(ve, b)-max(vs, a)
                if pid and ov > 0:
                    key = (r["user_id"], ch, a, b, pid); frac = ov/(b-a)
                    if key not in best or ov > best[key][0]: best[key] = (ov, frac)
    with gzip.open(out_path, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["user_id","tv_show_id","airing_start","airing_end","overlap_sec","overlap_frac"])
        for (u, ch, a, b, pid), (ov, frac) in sorted(best.items()):
            w.writerow([u, pid, datetime.fromtimestamp(a).isoformat(sep=" "), datetime.fromtimestamp(b).isoformat(sep=" "), round(ov,3), round(frac,6)])
    return {"rows": len(best), "title_example": example}

def read_interactions(path, fraction):
    out = defaultdict(Counter)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if float(r["overlap_frac"]) >= fraction: out[r["user_id"]][int(r["tv_show_id"])] += 1
    return out

def users_from_template():
    with (RAW/"submission.csv").open(encoding="utf-8-sig", newline="") as f: return [r["user_id"] for r in csv.DictReader(f)]

def recommend(users, personal, popularity, candidates):
    pop = [p for p in popularity if p in candidates] + sorted(candidates-set(popularity)); result = {}
    for u in users:
        own = [p for p, _ in sorted(personal.get(u, {}).items(), key=lambda x: (-x[1], x[0])) if p in candidates]
        result[u] = (own + [p for p in pop if p not in own])[:5]
    return result

def map5(pred, targets, users):
    aps = []
    for u in users:
        rel, hits, score = targets.get(u, set()), 0, 0.0
        for rank, pid in enumerate(pred[u], 1):
            if pid in rel: hits += 1; score += hits/rank
        aps.append(score/min(5, len(rel)) if rel else 0.0)
    return sum(aps)/len(aps), sum(bool(targets.get(u)) for u in users), sum(not targets.get(u) for u in users)

def save_predictions(path, preds, users):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["user_id","tv_show_id"])
        for u in users: w.writerow([u, " ".join(map(str, preds[u]))])

def main():
    started = time.perf_counter(); users = users_from_template(); stats = {}
    _, _, _, train_bounds = schedule_index(SCHEDULES["train"]); cutoff = train_bounds[1]
    for split, stop in (("train", cutoff), ("val", None)):
        stats[split] = join_views(SCHEDULES[split], OUT/f"{split}_interactions.csv.gz", stop)
    train, val_raw = read_interactions(OUT/"train_interactions.csv.gz", .50), read_interactions(OUT/"val_interactions.csv.gz", .80)
    _, val_items, val_title, _ = schedule_index(SCHEDULES["val"]); _, test_items, test_title, _ = schedule_index(SCHEDULES["test"])
    targets = {u: set(sorted(c, key=lambda p: (-c[p], p))[:5]) for u, c in val_raw.items()}; pop_counts = Counter(p for c in train.values() for p in c)
    pop_order = [p for p, _ in sorted(pop_counts.items(), key=lambda x: (-x[1], x[0]))]
    pop_val, freq_val = recommend(users, {}, pop_order, set(val_items)), recommend(users, train, pop_order, set(val_items))
    mp, mf = map5(pop_val, targets, users), map5(freq_val, targets, users)
    save_predictions(OUT/"validation_popularity.csv", pop_val, users); save_predictions(OUT/"validation_frequency.csv", freq_val, users)
    pop_test, freq_test = recommend(users, {}, pop_order, set(test_items)), recommend(users, train, pop_order, set(test_items))
    save_predictions(OUT/"submission_baseline_popularity.csv", pop_test, users); save_predictions(OUT/"submission_baseline_frequency.csv", freq_test, users)
    summary = {"settings":{"encoding":ENC,"train_fraction":.50,"target_fraction":.80,"repeats_allowed":True,"candidate_rule":"positive IDs in target schedule","fallback":"training global popularity","cohort":len(users)},"join":stats,"metrics":{"popularity":{"map5":mp[0],"users_with_targets":mp[1],"empty_targets":mp[2]},"frequency":{"map5":mf[0],"users_with_targets":mf[1],"empty_targets":mf[2]}},"examples":{"decoded_train_title":stats["train"]["title_example"],"decoded_val_title":val_title,"decoded_test_title":test_title,"manual_ap":{"ranks":[1,4],"ap5":.75}},"runtime_sec":time.perf_counter()-started}
    (OUT/"baseline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
