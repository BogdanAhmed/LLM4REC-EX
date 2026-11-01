"""Verify the selected method's test submission against the template and schedule."""
import csv, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/"data/raw"; OUT=ROOT/"results/methods"
def main():
    summary=json.loads((OUT/"methods_summary.json").read_text(encoding="utf-8")); name=summary["selected_method"]; path=OUT/f"submission_selected_{name}.csv"
    with (RAW/"submission.csv").open(encoding="utf-8-sig",newline="") as f: template=[r["user_id"] for r in csv.DictReader(f)]
    candidates=set()
    with (RAW/"export_arh_31-42-final.csv").open(encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            if int(r["tv_show_id"]): candidates.add(int(r["tv_show_id"]))
    rows=list(csv.DictReader(path.open(encoding="utf-8",newline=""))); ids=[r["user_id"] for r in rows]; ok=all((v:=r["tv_show_id"].split()) and len(v)==5 and len(set(v))==5 and all(int(x)>0 and int(x) in candidates for x in v) for r in rows)
    result={"file":str(path),"selected_method":name,"rows":len(rows),"template_rows":len(template),"user_order_matches":ids==template,"five_distinct_positive_available_ids":ok}
    (OUT/"submission_checks.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
