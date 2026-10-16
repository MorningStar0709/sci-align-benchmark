# ARTIFACT — reproducibility entry points

This file states, item by item, **what can be recomputed from this repository alone**.
It deliberately does not claim that everything in the two papers is reproducible here: some
raw inputs (operator logs, training corpora, judge API transcripts) are not distributed,
and for those the aggregate ships together with its fingerprint instead. Per-item
evaluation products for the headline experiments **do** ship here (see "Which per-item
products ship" below); the eval pipeline that produced them does not.

Every reading below was measured in a clean checkout of this repository — tracked files
only, no working-tree-only artefacts. Rows that are not measurable here say so rather than
implying a pass.

## Three-state legend

| State | Meaning |
|:--|:--|
| **Recomputable** | Runs to completion in a fresh clone and derives the published number from files that ship here |
| **Registered only** | The record ships with its source's `sha256`, but the raw inputs do not — you can verify provenance metadata and see which fields were withheld; you cannot re-derive the number |
| **Fails closed** | Detects missing inputs and exits non-zero. Intended behaviour for a public clone, not a bug you found |

## Item by item

| # | Published reading | Where | Measured in a fresh checkout | State |
|:--:|:--|:--|:--|:--|
| 1 | Claim anchors: 26 claims, each with a verbatim quote and the file it was read from | `provenance/claim-anchors.json` | `claim_anchors.py --check` → **exit 1**; all **26/26** anchors report `文件不存在` — 19 point at the findings manuscript, 7 at the methods manuscript | **Fails closed here** — an anchor is a (quote, manuscript line) pair, and the manuscript sources are not distributed in this repository (they are published separately, see `README.md`). The registry ships as the record; re-deriving it needs the working tree |
| 2 | Per-experiment aggregates behind the tables (judge drift, B1 verdict, counterfactual analysis, medical rejudge …) | `release/evidence/*.json` — **55 files**: 54 per-experiment projections + `manifest.json` (the index itself) | census: **all 54 projections** carry `source`, `source_sha256`, `source_bytes`, `dropped_fields`, `required_by`; **0 of 55 carry per-item `records`** (per-item records ship separately — row 10) | **Registered only** — `dropped_fields` names exactly what was withheld (per-item answer and reasoning text) |
| 3 | Model-identity table: 81 distinct model identifiers | `provenance/model_identities.json` (`scanned_files`, `tracked_only_count` are self-reported) | `collect_model_identities.py --check` → **exit 0**, `[SKIP] 本机树不完整（23 < 81），无法判定同代` — this tree ships a subset of the eval products (row 10), so only 23 of the 81 identifiers are reachable here and the check can no longer judge generations. A skip that returns success means nothing was checked | **Registered only** — the shipped tree cannot re-derive this table (23 of the 81 identifiers are reachable here); the JSON ships as the record |
| 4 | Unattended wall-clock: 12 segments / 11,637.1 min = 194.0 h | `provenance/unattended_walltime.json` + `provenance/README-unattended-walltime.md` | `collect_unattended_walltime.py --check` → **exit 1**, `[FAIL] 日志目录不存在：…/logs（干净检出没有 logs/，无法重扫）` — the operator log directory is not published. Running the gate discipline recreates `logs/`, after which the same absence is reported as `入仓件与本机 logs 现扫不同代`; either way there is nothing here to re-scan | **Fails closed** — the operator logs themselves are not distributed. Narrative and the double-counting pitfall are in the README |
| 5 | Artifact index: 67 products with `sha256`, size, tracked status and which claim requires each | `provenance/artifact-index.json` | `emit_provenance_index.py --check` → **exit 1**, `[WORKBENCH-ONLY] 主张矩阵所需产物不在本树：benchmark/eval/n3_regen/b1_verdict.json` + `[UNCHECKED] 这一项在本树不可复算 ⇒ 记为未验，不得当通过`, **no stack trace** | **Fails closed with a voice** — the missing input is named as a stated reason rather than crashing the interpreter. The shipped JSON remains a fingerprint registry (7 of 67 marked `public_safe`) |
| 6 | Question pool | `benchmark/dataset/core_v3.json` (1,979 items; per-item `audit_status` + `source_license_bucket`) + `schema.json`. **The pool ships as a projection.** In items whose source is *not* redistributable (`arxiv-only` 927 / `restricted` 241 / `textbook-unchecked` 126), any **verbatim source span of ≥ 300 characters** inside an evidence note is replaced by `[withheld: verbatim source excerpt]`; our own audit wording and shorter quotations stay, and the 685 `redistributable` items ship whole. The counts are recorded under `meta.license_tagging` | **not runnable as published** — no JSON-Schema validator is declared in the lockfile, so "validate against `schema.json`" is currently an empty instruction | **Pending** a stdlib checker or a declared dependency |
| 7 | Figures | one rendered figure ships (`figures/protocol.png`); the plotting scripts (`papers/*/figures/scripts/fig*.py`, 10 scripts) do not — but the **statistics** module they share does (row 12) | nothing to re-render in this tree | **Registered only** — the shipped image is a rendered artefact, and re-rendering it needs plotting code that is not distributed. The numbers the figures carry are a separate matter, and those *are* re-derivable (row 12) |
| 8 | Medical axis sensitivity (referenced in paper B) | `provenance/e12_e4_medical.json` (4,377 B; keys `analysis`, `e12`, `e4`), `provenance/e5_sigma_sensitivity.json` (1,686 B; keys `frozen_check`, `scan`, `survivors_by_k`) | **both files are present and tracked**; what is absent is their **generators** (working-tree-only scripts), so the aggregates cannot be re-derived here | **Registered only** — read the shipped JSON; re-derivation needs the working tree |
| 9 | Every shipped `--check` entry point actually runs without crashing | `tests/test_entry_smoke.py` | the test invokes four entries the way a reader would (`claim_anchors`, `collect_model_identities`, `collect_unattended_walltime`, `emit_provenance_index`) → **all exit 1 with a stated reason, 0 tracebacks** | **Verified here** — and the reason this row exists: a green suite over entry points that no test ever invoked is not evidence |
| 10 | Per-item evaluation products for the headline experiments — 237 distinct files across six dirs (`v3_phase4_300`, `gemma4_300q`, `v3_9b_300q`, `n3_regen`, `run1_rejudge`, plus a run1/run1p subset of `medical_dual_rejudge`): every model arm (raw / sft / rlvr_c_v2 / mgpo_eg / mgpo_fixed / medical_*) × every run/generation, each with 300 per-item records (dual-judge scores, disputes, issues) | `benchmark/eval/<dir>/*.json` + `.status.json` sidecars (judge model, completion counts) | `tools/verify_pack.py` CHECKs — (a) mean of `records[].score` re-derives each sampled run's self-reported `overall_score` to 4 dp; (b) for the 13 runs that also have projections in `release/evidence/`, the file's sha256 equals the projection's `source_sha256` and `overall_score` matches. CRLF originals are preserved byte-for-byte (`benchmark/eval/** -text -eol` in `.gitattributes`; the `-eol` half is load-bearing, since `*`'s `eol=lf` still normalises stored bytes when only `-text` is set) so blob, working tree and fingerprint agree | **Recomputable (in-file aggregates)** — any aggregate printed inside these files can be re-derived from their per-item records; both members of every figure-3 drift pair ship here (control run1 in `v3_*/` / `gemma4_300q/` with its run1p in `run1_rejudge/`; the medical pair together in `medical_dual_rejudge/`). What is **not** re-derivable here: the judge calls themselves (cloud API transcripts are not distributed) and any number that lives only in the manuscripts |
| 11 | In-tree integrity + internal consistency verification | `tools/verify_pack.py` (stdlib-only; `--dir` / `--zip`) + per-pack `manifest.json` | in **this** tree it exits non-zero by design: `verify_pack.py --dir .` → `[FAIL] missing: manifest.json at pack root`, **exit 1**. `manifest.json` is generated at pack time and does not exist at the repository root. Pointed at any directory that does carry a manifest, it verifies the sha ledger, runs a JSON parse census and the CHECK table | **Fails closed in-tree by design** |
| 12 | The scale × method gain surface behind the findings paper: SFT / RLVR-C v2 / MGPO-EG across six scales (18 cells) | `papers/paper-a/figures/scripts/surface_stats.py` + `benchmark/eval/n3_regen/n3_aggregation_summary.json` (the registered means) + `release/adapter_manifest.json` | `surface_stats.gains()` returns every cell: 0.8B **+5.8 / −2.4 / +3.3**, 2B **−2.3 / +4.7 / +1.3**, 4B **+14.9 / −0.9 / −5.1**, 9B **+13.0 / +2.0 / +7.6**, E2B **−5.0 / +2.6 / −15.7**, E4B **−12.4 / −0.5 / −9.8** | **Recomputable** — the statistics module and its registered input ship, so the percentages quoted on the front page are re-derivable in a fresh clone |
| 13 | The post-training matrix: 25 frozen arms across two domains × three model sizes (6 base/raw, 19 trained LoRA adapters), each with its base model, adapter `sha256` and eval run paths; plus 11 surface adapters, the judge prompts and the evaluation freeze | `release/adapter_manifest.json` | field census — `models` 25 entries, 19 with a non-null `adapter`, `surface_adapters` 11, and `datasets` / `judge_prompts` / `eval_freeze` present | **Registered only** — the registry is the record. The adapters are not published and the training scripts are not distributed, so the matrix can be read here but not re-derived |

Re-derive the gain surface:

```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "papers/paper-a/figures/scripts")
import surface_stats
for col, cells in surface_stats.gains().items():
    print(col, {k: round(v, 2) for k, v in cells.items() if k != "gens"})
PY
```

The denominators are part of the definition, not a detail: SFT is measured against the raw model, and the two reinforcement arms against the SFT baseline.

## Which per-item products ship

Everything lives in the git tree — this artifact has no separate per-paper ZIP. The
per-item products under `benchmark/eval/` are the **union** of what the two papers' figures
read: 237 files across six dirs (`v3_phase4_300`, `gemma4_300q`, `v3_9b_300q`, `n3_regen`,
`run1_rejudge`, plus a run1/run1p subset of `medical_dual_rejudge`). Beyond those per-item files,
`benchmark/eval/` holds one more file: `n3_regen/n3_aggregation_summary.json`, the registered means
that row 12's gain surface is computed from. It is materialised here from its public-safe structure
projection (`release/evidence/n3_aggregation_summary__n3_regen.json`), which is the route that
source normally takes to the public face. Nothing else from `benchmark/eval/` ships.

`medical_dual_rejudge` ships as a **run1 + run1p subset only** (24 files) — what the
figure-3 drift comparison reads. The rest of that directory is not shipped; its
aggregates remain available as projections in `release/evidence/`, and the medical axis
of the n=3 decomposition is covered by the `medical_*` generations inside `n3_regen`.
`*.phase1_recovery.*` / `*_audit.json` intermediates and `*.log` files are regenerable
working-tree products and are excluded everywhere.

## Claim ↔ recompute map

| Claim family | Paper | Recompute from |
|:--|:--:|:--|
| Training gain = f(method × scale × base alignment) | A | mean of `records[].score` per run file, cross-compared across arms and scales |
| Judge-drift discrimination between run1 and run1p | B | per-item `score`/`disputed` fields; **both members of each pair ship here** — control run1 in `v3_*/` / `gemma4_300q/` with its run1p in `run1_rejudge/`; medical run1 and run1p together in `medical_dual_rejudge/` (23 model pairs) |
| n=3 variance decomposition (incl. medical axis) | B | per-generation `overall_score` of every arm in `n3_regen/*_gen*.json` |
| Dual-judge dispute rates | both | per-item `judge1_score`, `judge2_score`, `disputed`, `arbitrated` fields in every shipped run file |

## What we are not claiming

The papers state three tiers: **numbers are recomputable**, the **σg noise scale is
reproducible within the retained generations**, and we do **not** claim end-to-end
reproduction. The third tier is not modesty — reproducing a run requires the source
articles our pool items were derived from, and redistribution rights for those articles
are not ours to grant. This file is the operational form of that sentence.

The first tier rests on more than the aggregate projections: per-item evaluation outputs
for the headline experiments ship here (row 10), so the *aggregates printed inside those
files* are re-derivable from per-item evidence. The aggregate-only projections in
`release/evidence/` remain **Registered only**; `tools/verify_pack.py` cross-checks them
against the per-item face wherever both ship.

## Known gaps

- Row 3 hands the reader an exit code that can mean "nothing was checked": with only 23 of
  the 81 identifiers reachable here, `--check` prints a skip and exits 0. Until there is an
  explicit `--verify-registered` path — re-read the shipped JSON, check its internal
  consistency, exit non-zero on inconsistency — the honest reading of that row is *skip*,
  not *pass*.
- Row 6 needs either a dependency declaration or a stdlib structural check; otherwise the
  instruction is void.
