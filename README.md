# SWEET.TV — LLM4Rec Exam
# Baseline preparation

From `C:\Users\bogdan\Desktop\Recsys\Exam` run:

```powershell
py src\baseline.py
py src\verify_baseline.py
py src\methods.py
py src\verify_methods.py
py src\corrected_experiment.py
py src\tuning_v3.py
py src\temporal_v4.py
py src\channel_v8.py
py src\time_channel_v9.py
```

`time_channel_v9.py` reuses the corrected interaction caches and frozen v4/v8
candidate matrices. It streams raw sessions once into a fingerprinted channel-
time profile cache, evaluates its fixed 33/35-feature HGB variants, and writes
only `results\time_channel_v9`. Re-running it resumes compatible caches.

`tuning_v3.py` reuses `results\corrected_v2\*_interactions.csv.gz`, writes all
new score caches, comparisons, reranker models, checks, and the candidate
submission under `results\tuning_v3`, and does not modify corrected_v2.

The script preserves raw CSVs, decodes schedules as UTF-8, streams a
same-channel interval join, and caches deduplicated interactions in
`results\baseline\*.csv.gz`. Training uses weeks 11–20, validation weeks
21–30, and test candidates weeks 31–42. Training watches require overlap /
programme duration ≥0.50; validation targets require ≥0.80 and retain each
user's five most frequent programmes. Repeats are allowed. Both baselines
share each window's candidate pool and the training-only global-popularity
fallback.

Outputs: validation predictions, two baseline submissions, and metrics/settings
in `results\baseline\baseline_summary.json`.
