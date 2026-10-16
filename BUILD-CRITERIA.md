# What this checkout contains, and why

Every file in this repository was admitted by a rule, not picked by hand. This file
records the rules and the resulting file set for the build that produced this tree.

- **Entry points**: 13 scripts reachable as reproducibility entry points.
- **Data faces**: question pool / provenance registry / release evidence / humaneval
  harness. The manuscript sources are **not** distributed here — the two papers are
  published separately (see `README.md`).
- **Code closure**: 10 packages (`mineru_agent`, `mineru_api`, `mineru_benchmark`, `mineru_cli`, `mineru_core`, `mineru_eval`, `mineru_generator`, `mineru_medical`, `mineru_pipeline`, `mineru_tools`) and 26 scripts.

## Excluded outright (the file is not in this tree at all)

Unlike the entries below, these are not merely left unrun — they are not copied in.
Each one validates artefacts this repository does not distribute, so it could not do
anything here:

  - `scripts/build_submission_surface.py` — Fingerprint registry for those same manuscript-side artefacts, likewise absent. It also imports the check above, so shipping it alone would break on import
  - `scripts/build_supplementary_sample.py` — Its generated products (`supplementary_items.tex`, `rubrics.tex`) are written into the manuscript tree, which this repository does not distribute. With that tree absent it reports [STALE] twice and exits 1
  - `scripts/check_blind_submission.py` — Validates manuscript-side artefacts — the LaTeX sources plus the PDFs built from them. Neither is distributed in this repository, so the check would have no object here and would fail closed on every run

## Test files

Two passes decide which tests ship: (1) every first-order import resolves inside the
code closure and the file reads no data face that this repository does not ship;
(2) a collection run — anything pytest cannot import is dropped.

- Dropped during collection: `tests/test_a800_scheduler.py`
- Excluded data faces (any first-order reference disqualifies a test): upstream working docs, benchmark experiment outputs, scratch trees, model weights, logs, archives, and SQLite databases.

- Kept 17 test files; dropped 58:

  - `tests/test_a800_common.py`
  - `tests/test_a800_merge.py`
  - `tests/test_a800_snapshot.py`
  - `tests/test_analyze_dataset_embeddings.py`
  - `tests/test_arbitration_patch.py`
  - `tests/test_artifact_packs.py`
  - `tests/test_audit_branch_readings.py`
  - `tests/test_batch_sensitivity.py`
  - `tests/test_blind_submission.py`
  - `tests/test_body_number_binding.py`
  - `tests/test_check_doc_dates.py`
  - `tests/test_check_figure_ledger.py`
  - `tests/test_claim_anchors.py`
  - `tests/test_claim_specs.py`
  - `tests/test_display_exposure.py`
  - `tests/test_epoch_and_denominator.py`
  - `tests/test_evidence_projection.py`
  - `tests/test_fix_failed_task_marks.py`
  - `tests/test_inventory_experiment_assets.py`
  - `tests/test_inventory_matrix_disposition.py`
  - `tests/test_judge_qa_acceptance.py`
  - `tests/test_judge_qa_common.py`
  - `tests/test_judge_qa_epoch.py`
  - `tests/test_judge_qa_perturb.py`
  - `tests/test_judge_qa_ranking.py`
  - `tests/test_judge_qa_select.py`
  - `tests/test_judge_qa_spec.py`
  - `tests/test_judge_qa_stats.py`
  - `tests/test_judge_qa_summarize.py`
  - `tests/test_map_paper_display.py`
  - `tests/test_model_identities.py`
  - `tests/test_n3_gen1_source.py`
  - `tests/test_pack_verify.py`
  - `tests/test_paper_bibliography.py`
  - `tests/test_paper_claims.py`
  - `tests/test_paper_gap_supplement.py`
  - `tests/test_pipeline_watchdog.py`
  - `tests/test_probe_readers.py`
  - `tests/test_product_self_consistency.py`
  - `tests/test_provenance_index.py`
  - `tests/test_public_release_face.py`
  - `tests/test_register_consistency.py`
  - `tests/test_release_anonymity.py`
  - `tests/test_score_claim_matrix.py`
  - `tests/test_sigma_definition.py`
  - `tests/test_submission_surface.py`
  - `tests/test_superseded_venue_calibers.py`
  - `tests/test_supplementary_sample.py`
  - `tests/test_surface_aggregation_columns.py`
  - `tests/test_surface_direction.py`
  - `tests/test_surface_ledger_consistency.py`
  - `tests/test_trunc_manifest.py`
  - `tests/test_trunc_overlap.py`
  - `tests/test_trunc_sensitivity.py`
  - `tests/test_trunc_supplement_health.py`
  - `tests/test_trunc_supplement_orchestrator.py`
  - `tests/test_unattended_walltime.py`
  - `tests/test_verify_judge_drift_records.py`

## Scripts excluded from the closure

Scripts under `scripts/` that no entry point can reach, and that do not belong to a
shipped package:

  - `scripts/audit_branch_readings.py`
  - `scripts/audit_corpus_license.py`
  - `scripts/audit_data_trace.py`
  - `scripts/audit_pool_license.py`
  - `scripts/audit_pool_public_face.py`
  - `scripts/audit_question_pool.py`
  - `scripts/check_b1_gen4_health.py`
  - `scripts/check_blind_submission.py`
  - `scripts/check_body_number_binding.py`
  - `scripts/check_caliber_numbers.py`
  - `scripts/check_doc_dates.py`
  - `scripts/check_evidence_projection.py`
  - `scripts/check_judge_audit_health.py`
  - `scripts/check_local9b_health.py`
  - `scripts/check_n3_regen_health.py`
  - `scripts/check_openbook_control_health.py`
  - `scripts/check_product_self_consistency.py`
  - `scripts/check_register_consistency.py`
  - `scripts/check_release_anonymity.py`
  - `scripts/check_superseded_values.py`
  - `scripts/check_superseded_venue_calibers.py`
  - `scripts/check_trunc_supplement_health.py`
  - `scripts/verify_judge_drift_aggregate.py`
