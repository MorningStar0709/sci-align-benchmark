# The audit-the-instrument evaluation protocol

A one-page statement of the method. The canonical version, with the full argument, related work and
ablations, is the preprint: **[Benchmarking Post-Training Under Constrained Compute
(10.5281/zenodo.23409480)](https://doi.org/10.5281/zenodo.23409480)**.

([中文说明 · Chinese](PROTOCOL.zh-CN.md))

## The problem it addresses

When a task has no reference answer, post-training gains are usually scored by another model. That
turns the judge into the measuring instrument, and the instrument is rarely audited in the studies it
serves:

- **Scores drift.** The scoring service changes between evaluation epochs, so the same answer can be
  scored differently without any model having changed.
- **One generation is one draw.** Generation is stochastic, so a single reported score is a sample
  from a distribution nobody measured. Differences within that spread look like findings.

## The protocol

![Corpus, dual-blind questions, n=3 answers, dual judge, verdicts, with item-level audit and drift re-judging](figures/protocol.png)

1. **Curate the corpus, then the pool.** Questions are drawn from a curated source set, and every item
   is registered and labelled with its licence bucket before it is used for anything. (This repository
   ships the pool and its labels: 1,979 items, `SOURCES.md`, `LICENSE-LEDGER.md`.)
2. **Generate questions dual-blind, arbitrate with a third model.** Question generation is separated
   from scoring, and disagreements between the two generators go to a third model rather than to the
   author.
3. **Regenerate each answer n=3 per scale, and fix the answers.** Repeating generation per model scale
   gives an empirical between-generation spread (σg), so a within-noise difference is not read as a
   gain. Once generated, the answers are frozen, so that later scoring differences can only come from
   the judge.
4. **Score with a dual judge, log the epoch.** Two judges score independently, with the epoch recorded,
   so any later re-scoring of the same answers is attributable to the instrument.
5. **Decide under criteria frozen in advance.** Promotion criteria are written down *before* any scored
   comparison. Drift is countered by **regeneration, not post-hoc correction** — there is no rescue
   step in which a drifting judge's output is adjusted after the fact.

Two loops run alongside: an **item-level audit** checks the pool for monotonicity and repeatability
(the preprint reports 62.5% monotone and 83.3% repeatable; the audit's aggregation is not
distributed here), and a **drift re-judge** re-scores the frozen answers on a later judge epoch and
reports the shift.

## What the protocol decided, applied

The same instrument was applied twice, which is the reason to trust its verdicts:

- **Control domain.** Re-judging fixed answers on a later epoch shows a systematic upward shift
  (published aggregate +0.0606, all 11 model arms moving up; pairing the shipped per-item scores per
  arm re-derives ≈+0.06). Under criteria frozen beforehand, **one of four preregistered comparisons**
  was promoted.
- **Medical domain.** No comparable drift (the preprint reports +0.0002, 5 of 12 arms up). The frozen
  criterion decided **all nine** trained-versus-base comparisons, and **four of them were losses**.

The findings paper ([10.5281/zenodo.23409394](https://doi.org/10.5281/zenodo.23409394)) reports what
that instrument then measured about post-training gains, including the scale-dependent reversal and
the base-model-dependent sign flip.

## Scope, stated

- The headline experiments were scored on **300-item evaluation pools per experiment**; the registered
  pool this repository ships is larger (1,979 items) and is the source of the questions and their
  licence labels.
- The judge is a **cloud service**; its API transcripts are not distributed. What ships is the
  per-item scoring output (`benchmark/eval/`), from which the aggregates are re-derivable.
- The protocol is stated as a method, not as a package. Where a claim cannot be recomputed from the
  files shipped here, [`ARTIFACT.md`](ARTIFACT.md) says so item by item.
