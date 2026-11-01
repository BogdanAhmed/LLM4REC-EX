"""Post-run checks and human-verification examples for the baseline outputs."""
import csv, gzip, json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; RAW = ROOT/"data/raw"; OUT = ROOT/"results/baseline"
def ts(s, fmt): return datetime.strptime(s, fmt).timestamp()

def main():
    train = Counter(); first = None
    with gzip.open(OUT/"train_interactions.csv.gz", "rt", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            train[r["user_id"]] += 1
            if first is None: first = r
    with (RAW/"submission.csv").open(encoding="utf-8-sig", newline="") as f: users = [r["user_id"] for r in csv.DictReader(f)]
    candidates = set(); airing = None
    with (RAW/"export_arh_31-42-final.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if int(r["tv_show_id"]): candidates.add(int(r["tv_show_id"]))
    with (RAW/"export_arh_11-20-final.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if first and airing is None and r["tv_show_id"] == first["tv_show_id"] and ts(r["start_time"], "%d.%m.%Y %H:%M:%S") == ts(first["airing_start"], "%Y-%m-%d %H:%M:%S"):
                airing = {"channel_id": r["channel_id"], "start": r["start_time"], "duration_sec": r["duration"], "tv_show_id": r["tv_show_id"], "title": r["tv_show_title"]}
    cold = next((u for u in users if u not in train), None); heavy = max(train, key=lambda u: (train[u], u)); light = min((u for u in train if u in users), key=lambda u: (train[u], u))
    checks = {}
    for name in ("submission_baseline_popularity.csv", "submission_baseline_frequency.csv"):
        ok = True; n = 0
        with (OUT/name).open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                vals = r["tv_show_id"].split(); ok &= len(vals) == 5 and len(set(vals)) == 5 and all(int(x) > 0 and int(x) in candidates for x in vals); n += 1
        checks[name] = {"rows": n, "complete": n == len(users), "five_distinct_positive_available_ids": ok}
    example = None
    if first and airing:
        with (RAW/"dataset11-30.csv").open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if r["user_id"] != first["user_id"] or int(r["vsetv_id"]) != int(airing["channel_id"]): continue
                ov = max(0.0, min(ts(r["stop_time"], "%Y-%m-%d %H:%M:%S"), ts(airing["start"], "%d.%m.%Y %H:%M:%S") + int(airing["duration_sec"])) - max(ts(r["start_time"], "%Y-%m-%d %H:%M:%S"), ts(airing["start"], "%d.%m.%Y %H:%M:%S")))
                if ov > 0:
                    example = {"user_id":r["user_id"],"view_start":r["start_time"],"view_stop":r["stop_time"],"channel_id":airing["channel_id"],"airing_start":airing["start"],"programme_id":airing["tv_show_id"],"programme_title":airing["title"],"overlap_sec":ov,"programme_duration_sec":int(airing["duration_sec"]),"overlap_fraction":ov/int(airing["duration_sec"])}; break
    summary = json.loads((OUT/"baseline_summary.json").read_text(encoding="utf-8")); summary["case_users"] = {"heavy":{"user_id":heavy,"train_interactions":train[heavy]},"light":{"user_id":light,"train_interactions":train[light]},"cold":{"user_id":cold,"train_interactions":0 if cold else None}}; summary["automated_checks"] = checks; summary["examples"]["real_overlap"] = example; summary["examples"]["manual_check_status"] = "automated example generated; hand arithmetic not yet performed by student"
    (OUT/"baseline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"case_users":summary["case_users"],"checks":checks,"real_overlap":example}, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
