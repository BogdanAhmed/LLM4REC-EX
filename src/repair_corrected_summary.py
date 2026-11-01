"""Refresh reporting fields after the completed corrected run without recomputing data."""
import json
from pathlib import Path
from corrected_experiment import OUT, randomized_checks

summary=json.loads((OUT/"corrected_summary.json").read_text(encoding="utf-8"))
summary["randomized_checks"]=randomized_checks()
summary["before_after"]["fixes"]=["repaired interval candidate search","rebuilt both caches and validation targets","separated fixed graph construction from user inference","rebuilt final fit from corrected weeks 11–30 caches"]
(OUT/"corrected_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(summary["randomized_checks"],indent=2))
