# SWEET.TV — Two-user recommendation analysis (expanded evidence)

Post-submission supplementary evidence for the individual LLM4Rec exam. This updates the existing `docs/two_user_analysis.md` in the *same exam repository*. It **does not change** the original report, trained models or designated Kaggle submission. All values below come from archived CSV/JSON files; no model was retrained to create this appendix.

![Compact two-user comparison](two_user_comparison.png)

## 1. What changed and why

- The original five-page report and earlier supplement discussed **user 13866173795451203812**, with 4,091 training interactions in `corrected_v2`, and sparse **user 3799694907521863650**, with one interaction in that *older case export*.
- **Crucial distinction:** the former active user is **not among the 1,260 users** in the later clean chronological prediction files. Therefore no valid clean-window comparison or v12 recommendation list can be claimed for that user.
- For an exact, same-split comparison, this appendix uses a **new active user `13571199046623737884`** (1,031 pre-T eligible overlaps over 251 programmes) and retains **sparse user `3799694907521863650`** (3 eligible overlaps over 2 programmes in the uploaded pre-T interaction file). The earlier examples are retained separately in §6.
- All six methods listed below are compared using the **same 1,260-user clean-validation predictions, the same target file and fixed cutoff**. In particular, CBF and PPR are evaluated individually, rather than inferred from the final hybrid.

## 2. Protocol and independent checks

- History cutoff **T = 2020-05-18 00:00:00**; target end **U = 2020-07-27 00:00:00**. The history examples were reconstructed from the supplied joined interaction export using `airing_end < T`, `overlap_frac >= 0.50`, and `tv_show_id != 0`. The validation target examples use `T <= airing_end < U` and `overlap_frac >= 0.80`.
- The 1,260-user prediction, target, and per-user AP tables have identical user IDs. Each listed prediction has exactly five distinct positive IDs. All selected target IDs are observed among qualifying validation overlaps in the supplied interaction export; the official target export caps target IDs at five, so some other qualifying validation IDs are deliberately outside the scored target set.
- AP@5 = Σ over relevant recommended ranks (Precision@rank) / min(5, number of relevant target items); MAP@5 averages AP@5 over **all 1,260 users**, including zero-target users (AP = 0).
- The attached original interaction exports were used only to **describe these users**; they are *not* claimed to be the complete, newly retrained v4 history cache. Their cutoff filtering is explicit, and the reported predictions and labels are taken from their respective saved validation exports.
- Item title/category metadata comes only from the uploaded archived `case_users.json`. Two IDs in the displayed recommendations, `200344` and `6500479`, lack metadata in that file; they remain **unresolved**, not guessed. Their IDs and hit flags remain verifiable.

### Overall clean-window comparison (recomputed directly from rows)

| Method | Mean AP@5 across 1,260 users |
|---|---:|
| Popularity | 0.274613 |
| Frequency | 0.370150 |
| CBF | 0.277572 |
| PPR (α=.75) | 0.379185 |
| HGB v4 | 0.404031 |
| RRF v12 | 0.414097 |

The selected RRF formula uses `k = 60`, with weights **v9 0.20 / LambdaRank v10 0.80**. Its recalculated MAP@5 **0.4140972222** matches the recorded v12 result. The final Kaggle private score **0.44366** is separate, and is not calculated from these validation files.

## 3. User selection, actual history and targets

### Active user: `13571199046623737884`

- **Past behavior:** 1,031 eligible pre-T overlaps across 251 programme IDs (`overlap_frac >= 0.50`).
- **Held-out target IDs (5):** `200352 240081 12001734 90049236 90071741`.

| Past item ID | Title | Category | Eligible pre-T overlaps |
|---:|---|---|---:|
| 12001682 | М/с 'Губка Боб Квадратные Штаны'. | Детям | 82 |
| 240081 | Факты. День. | Инфо | 74 |
| 2400480 | 'ТСН. Неделя'. | Инфо | 73 |
| 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | 43 |
| 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | 40 |
| 12001734 | Not in provided case metadata | Not in provided case metadata | 26 |
| 12002805 | Not in provided case metadata | Not in provided case metadata | 21 |
| 200337 | Not in provided case metadata | Not in provided case metadata | 20 |
| 10002451 | Not in provided case metadata | Not in provided case metadata | 19 |
| 2400467 | 'Викна-Новости'. | Инфо | 19 |
| 6500479 | Not in provided case metadata | Not in provided case metadata | 18 |
| 12002955 | М/с 'Свинка Пеппа'. | Детям | 17 |

This viewer has substantial history, including animation, news and other shows, but most held-out target items are not recovered by the illustrated Top-5 lists. This is an **observed failure case** despite the long history—not proof that more history always improves an individual recommendation.

### Sparse user: `3799694907521863650`

- **Past behavior:** 3 eligible pre-T overlaps across 2 programme IDs (`overlap_frac >= 0.50`).
- **Held-out target IDs (4):** `2400467 2400480 12002733 90044740`.

| Past item ID | Title | Category | Eligible pre-T overlaps |
|---:|---|---|---:|
| 400426 | 'Меняю жену'. | Развлечения | 2 |
| 2400480 | 'ТСН. Неделя'. | Инфо | 1 |

In the filtered pre-T source this viewer has two distinct historical IDs: `400426` (2 overlapping airings) and `2400480` (1). This differs from the older `corrected_v2` case file describing a one-item history. That difference is explicitly version-specific; it is not silently reconciled.

## 4. Full Top-5: same two users, same targets, same chronological evaluation

### Active user

| Method | Rank | ID | Original title | Category | Target hit |
|---|---:|---:|---|---|:---:|
| Popularity | 1 | 2400480 | 'ТСН. Неделя'. | Инфо | — |
| Popularity | 2 | 240081 | Факты. День. | Инфо | ✓ |
| Popularity | 3 | 2400467 | 'Викна-Новости'. | Инфо | — |
| Popularity | 4 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | — |
| Popularity | 5 | 200344 | **Not available in archived case metadata** | *unavailable* | — |
| Frequency | 1 | 12001682 | М/с 'Губка Боб Квадратные Штаны'. | Детям | — |
| Frequency | 2 | 240081 | Факты. День. | Инфо | ✓ |
| Frequency | 3 | 2400480 | 'ТСН. Неделя'. | Инфо | — |
| Frequency | 4 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | — |
| Frequency | 5 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | — |
| CBF | 1 | 240081 | Факты. День. | Инфо | ✓ |
| CBF | 2 | 2400480 | 'ТСН. Неделя'. | Инфо | — |
| CBF | 3 | 5900319 | Абзац! | Инфо | — |
| CBF | 4 | 5900347 | 'Деньги'. | Инфо | — |
| CBF | 5 | 2400467 | 'Викна-Новости'. | Инфо | — |
| PPR (α=.75) | 1 | 2400480 | 'ТСН. Неделя'. | Инфо | — |
| PPR (α=.75) | 2 | 240081 | Факты. День. | Инфо | ✓ |
| PPR (α=.75) | 3 | 12001682 | М/с 'Губка Боб Квадратные Штаны'. | Детям | — |
| PPR (α=.75) | 4 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | — |
| PPR (α=.75) | 5 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | — |
| HGB v4 | 1 | 2400480 | 'ТСН. Неделя'. | Инфо | — |
| HGB v4 | 2 | 240081 | Факты. День. | Инфо | ✓ |
| HGB v4 | 3 | 12001682 | М/с 'Губка Боб Квадратные Штаны'. | Детям | — |
| HGB v4 | 4 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | — |
| HGB v4 | 5 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | — |
| RRF v12 | 1 | 2400480 | 'ТСН. Неделя'. | Инфо | — |
| RRF v12 | 2 | 240081 | Факты. День. | Инфо | ✓ |
| RRF v12 | 3 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | — |
| RRF v12 | 4 | 12001682 | М/с 'Губка Боб Квадратные Штаны'. | Детям | — |
| RRF v12 | 5 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | — |

| Method | Held-out Top-5 hits | AP@5 |
|---|---:|---:|
| Popularity | 1/5 | 0.100000 |
| Frequency | 1/5 | 0.100000 |
| CBF | 1/5 | 0.200000 |
| PPR (α=.75) | 1/5 | 0.100000 |
| HGB v4 | 1/5 | 0.100000 |
| RRF v12 | 1/5 | 0.100000 |

### Sparse user

| Method | Rank | ID | Original title | Category | Target hit |
|---|---:|---:|---|---|:---:|
| Popularity | 1 | 2400480 | 'ТСН. Неделя'. | Инфо | ✓ |
| Popularity | 2 | 240081 | Факты. День. | Инфо | — |
| Popularity | 3 | 2400467 | 'Викна-Новости'. | Инфо | ✓ |
| Popularity | 4 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | — |
| Popularity | 5 | 200344 | **Not available in archived case metadata** | *unavailable* | — |
| Frequency | 1 | 400426 | 'Меняю жену'. | Развлечения | — |
| Frequency | 2 | 2400480 | 'ТСН. Неделя'. | Инфо | ✓ |
| Frequency | 3 | 240081 | Факты. День. | Инфо | — |
| Frequency | 4 | 2400467 | 'Викна-Новости'. | Инфо | ✓ |
| Frequency | 5 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | — |
| CBF | 1 | 400426 | 'Меняю жену'. | Развлечения | — |
| CBF | 2 | 2400480 | 'ТСН. Неделя'. | Инфо | ✓ |
| CBF | 3 | 6000483 | 'Свадьба вслепую', 6 сезон, 7 эп. | Развлечения | — |
| CBF | 4 | 200402 | Дом на зависть всем. | Развлечения | — |
| CBF | 5 | 400335 | 'Хата на тата', 7 сезон, 1 эп. | Развлечения | — |
| PPR (α=.75) | 1 | 400426 | 'Меняю жену'. | Развлечения | — |
| PPR (α=.75) | 2 | 2400480 | 'ТСН. Неделя'. | Инфо | ✓ |
| PPR (α=.75) | 3 | 240081 | Факты. День. | Инфо | — |
| PPR (α=.75) | 4 | 2400467 | 'Викна-Новости'. | Инфо | ✓ |
| PPR (α=.75) | 5 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | — |
| HGB v4 | 1 | 2400480 | 'ТСН. Неделя'. | Инфо | ✓ |
| HGB v4 | 2 | 240081 | Факты. День. | Инфо | — |
| HGB v4 | 3 | 2400467 | 'Викна-Новости'. | Инфо | ✓ |
| HGB v4 | 4 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | — |
| HGB v4 | 5 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | — |
| RRF v12 | 1 | 2400480 | 'ТСН. Неделя'. | Инфо | ✓ |
| RRF v12 | 2 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | — |
| RRF v12 | 3 | 6500479 | **Not available in archived case metadata** | *unavailable* | — |
| RRF v12 | 4 | 6000483 | 'Свадьба вслепую', 6 сезон, 7 эп. | Развлечения | — |
| RRF v12 | 5 | 240081 | Факты. День. | Инфо | — |

| Method | Held-out Top-5 hits | AP@5 |
|---|---:|---:|
| Popularity | 2/5 | 0.416667 |
| Frequency | 2/5 | 0.250000 |
| CBF | 1/5 | 0.125000 |
| PPR (α=.75) | 2/5 | 0.250000 |
| HGB v4 | 2/5 | 0.416667 |
| RRF v12 | 1/5 | 0.250000 |

### Interpreting the differences

- **Active viewer:** Frequent/graph-based lists retrieve mostly commonly watched items, but in this case only `240081` is in the *five selected validation targets*. Content similarity CBF ranks `240081` first (AP@5 = 0.20), whereas the selected RRF puts it second (AP@5 = 0.10). The case shows a real limitation of the aggregate-winning strategy for a particular viewer.
- **Sparse viewer:** Popularity retrieves `2400480` at rank 1 and `2400467` at rank 3 (AP@5 = 0.416667), whereas CBF prioritizes entertainment titles like «Свадьба вслепую» and «Хата на тата», with just one target hit at rank 2 (AP@5 = 0.125). PPR/Frequency have two hits but at lower positions (AP@5 = 0.25). RRF has one hit at rank 1 (AP@5 = 0.25).
- Hit counts are not AP@5: rank position and the number of held-out relevant items matter. These are **offline viewing proxies**, not satisfaction or business uplift.

## 5. Why select the hybrid despite these cases?

- **Global validation rather than cherry-picked cases:** v12 is selected using the all-user MAP@5 = 0.414097, above the independent CBF, PPR, popularity and HGB-v4 results listed above. The final Kaggle private score is **0.44366**. Users in this case study are selected for contrasting history lengths, not to maximize the hybrid’s apparent benefit.
- **Ranking cost:** a past-only popularity fallback is cheap; CBF and PPR require item metadata/graph work; v9 and LambdaRank require feature computation and trained scoring. Previously reported v12 ~108.49 s was **a cached experiment wall time**, not a measured live-service latency.
- **Limits:** missing/zero programme IDs in schedules, incomplete metadata in this small archive, recency and repeat-view biases, nested-interval matching issues fixed during development, and no online A/B, retention or revenue evidence. The v12 gain over v10 is only ~0.000463 MAP@5, so it should not be oversold.
- **Transparency about component examples:** Individual user-level v9/LambdaRank-v10 Top-5 outputs are **not contained in the supplied prediction CSVs**, and are not reconstructed from RRF. The full common-split per-user table covers popularity, frequency, CBF, PPR, HGB v4 and final RRF v12. The original report already reports global v9 and v10 evaluations.

## 6. Earlier corrected-v2 two-user examples (kept for continuity with the submitted PDF)

The original report and first supplement used `corrected_v2` examples. In that source: active user `13866173795451203812` had 4,091 training interactions; sparse user `3799694907521863650` was represented with one interaction. The two original case examples were scored using the **older case-study protocol**, NOT the strict clean T/U results above. They should not be used to calculate MAP@5 in §2.

| Archived user | Method | Top-5 IDs (ranked) | Target hits |
|---|---|---|---:|
| 13866173795451203812 | frequency | 2400480 → 240081 → 12001682 → 12002955 → 2400508 | 4/5 |
| 13866173795451203812 | cbf | 2400480 → 240081 → 5900319 → 5900347 → 2400467 | 3/5 |
| 13866173795451203812 | ppr_a0.75 | 2400480 → 240081 → 2400508 → 12001682 → 20088 | 4/5 |
| 3799694907521863650 | frequency | 400426 → 2400480 → 240081 → 2400467 → 2400508 | 2/5 |
| 3799694907521863650 | cbf | 400426 → 2400480 → 6000483 → 200402 → 400335 | 1/5 |
| 3799694907521863650 | ppr_a0.75 | 400426 → 2400480 → 240081 → 2400467 → 2400508 | 2/5 |

Their unabridged IDs, titles, categories and hit flags are archived in [`evidence/archived_corrected_v2_case_lists.csv`](evidence/archived_corrected_v2_case_lists.csv) and original [`results/corrected_v2/case_users.json`](../results/corrected_v2/case_users.json) in the same repository.

## 7. Evidence files and how to reproduce the checks

| Evidence file | Contents |
|---|---|
| [`docs/evidence/two_user_clean_validation_top5.csv`](evidence/two_user_clean_validation_top5.csv) | 60 predictions: 2 users × 6 methods × 5 ranked programme IDs, titles, categories, hit flags and AP@5 |
| [`docs/evidence/two_user_pre_T_view_history.csv`](evidence/two_user_pre_T_view_history.csv) | all distinct eligible pre-T programme IDs and viewing-overlap frequencies for the two selected users |
| [`docs/evidence/two_user_validation_targets.csv`](evidence/two_user_validation_targets.csv) | held-out target labels for both users |
| [`docs/evidence/two_user_case_summary.csv`](evidence/two_user_case_summary.csv) | user-selection counts and target cardinalities |
| [`docs/evidence/validation_metrics_recomputed.csv`](evidence/validation_metrics_recomputed.csv) | recomputed average AP@5 for six compared methods over all 1,260 user rows |
| [`docs/evidence/validation_per_user_ap6.csv`](evidence/validation_per_user_ap6.csv) | per-user AP@5 for all 1,260 users (six methods), enough to independently check all reported MAP@5 means |
| [`docs/evidence/verify_evidence.py`](evidence/verify_evidence.py) | standard-library verification of per-user examples, AP@5, and all-user mean metrics |
| [`docs/evidence/archived_corrected_v2_case_lists.csv`](evidence/archived_corrected_v2_case_lists.csv) | original corrected-v2 user cases retained for audit |

Run `python docs/evidence/verify_evidence.py` from the repository root to verify per-user AP@5 examples and global MAP@5 means using only these small derived evidence files.

Inputs (stored locally, not uploaded to this repository as raw viewing logs): `validation_predictions(5).csv`, `validation_predictions(4).csv` (v12 fusion results preserved from the exam workflow), `validation_targets.csv`, `per_user_ap(1).csv`, `train_interactions.csv.gz`, `val_interactions.csv.gz`, `case_users.json`. Original large joined view logs are intentionally excluded from GitHub. The derived outputs here are intentionally limited to evidence needed for the case study.

Reconstruction steps: (1) read both prediction exports with `user_id` as string; (2) match 1,260 user keys to `validation_targets.csv`; (3) calculate every user’s AP@5 using the target set; (4) confirm all-user mean matches `validation_metrics_recomputed.csv`; (5) select the two IDs in §3; (6) filter archived train interactions to `airing_end < T`, `overlap_frac >= .5`, nonzero IDs and count per title; (7) use archived case metadata for any known title/category. Unknown metadata is never guessed.
