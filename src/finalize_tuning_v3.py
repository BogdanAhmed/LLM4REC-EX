"""Final integrity checks and evidence annotations for the completed v3 run."""
import csv, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"results/tuning_v3"; V2=ROOT/"results/corrected_v2"
summary=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
summary["unsuccessful_runs"]=[{"stage":"validation reranker fit","failure":"sandbox denied scikit-learn thread-pool handle creation","correction":"OMP_NUM_THREADS=1 and LOKY_MAX_CPU_COUNT=1"},{"stage":"case-user export","failure":"heavy case user is outside the 1,260-user cohort","correction":"separate diagnostic prediction users from metric users"},{"stage":"cold-start audit","failure":"zero PPR scores were ranked as candidates","correction":"filter zero scores and apply common corrected-popularity fallback"}]
summary["runtime_note"]="Final per-method PPR/model times are cache-load plus ranking times where settings say cached=true; snapshot construction times are original measured runtimes. Failed-attempt wall time was not retained."
comparison={r["method"]:r for r in summary["comparison"]}
submission=OUT/summary["final_submission"]; template=[]; rows=[]
with (ROOT/"data/raw/submission.csv").open(encoding="utf-8-sig",newline="") as f: template=[r["user_id"] for r in csv.DictReader(f)]
with submission.open(encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
cases=json.loads((OUT/"case_users.json").read_text(encoding="utf-8"))
verification={"ppr_075_reproduces_corrected_v2":abs(comparison["ppr_a0.75"]["map5"]-.37907341269841294)<1e-12,"corrected_popularity_recomputed":abs(summary["popularity_fix"]["corrected_recomputed_map5"]-.2746549823633161)<1e-12,"selected_improves_validation":summary["selected"]["map5"]>.37907341269841294,"comparison_rows":len(summary["comparison"]),"case_rows":len(cases),"two_case_users":len(set(x["user_id"] for x in cases))==2,"submission_rows":len(rows),"template_order":[r["user_id"] for r in rows]==template,"submission_sha256":hashlib.sha256(submission.read_bytes()).hexdigest(),"corrected_v2_submission_sha256":hashlib.sha256((V2/"submission_corrected_v2_ppr_a0.75.csv").read_bytes()).hexdigest()}
assert all(verification[k] for k in ("ppr_075_reproduces_corrected_v2","corrected_popularity_recomputed","selected_improves_validation","two_case_users","template_order"))
(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); (OUT/"verification.json").write_text(json.dumps(verification,indent=2),encoding="utf-8"); print(json.dumps(verification,indent=2))
