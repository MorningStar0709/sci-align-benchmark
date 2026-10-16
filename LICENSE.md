# License

Two licenses, split by what the file is:

| Material | License |
|:--|:--|
| Code (`packages/`, `scripts/`, `configs/`, figure-generation scripts) | **Apache-2.0** |
| Data and paper sources (`benchmark/dataset/`, `provenance/`, `release/evidence/`, `papers/`) | **CC-BY-4.0** |

Canonical texts: https://www.apache.org/licenses/LICENSE-2.0 and
https://creativecommons.org/licenses/by/4.0/ . The verbatim Apache-2.0 text ships in
`LICENSE`; this file states what covers what.

## What our license does not cover

CC-BY-4.0 on our data **cannot** license the third-party articles it was derived from.
Those rights remain with their authors and publishers.

- **Pool items** cite their sources by `source_ref` and carry short quotations rather than whole
  articles. Where an audit note quotes a source, any **verbatim span of 300 characters or more is
  replaced by a placeholder**; our own wording and shorter quotations stay.
- **Per-source status** is recorded in [`LICENSE-LEDGER.md`](LICENSE-LEDGER.md) and
  [`SOURCES.md`](SOURCES.md), both of which ship in this repository.
- **Sources marked non-commercial or no-derivatives** keep that status per item in the pool, so a
  reuser can filter them out without guessing.
- **Textbook-derived items (22 sources) have no license determination at all.** Stated as unknown,
  not asserted as cleared.

## Deliberately not published

- **Training corpora** — they embed whole articles we are not licensed to redistribute.
- **Base model weights** — use the upstream public checkpoints.
- **Some per-item evaluation outputs** — see `ARTIFACT.md`, which says per item whether a
  number can be recomputed here or is registered with a fingerprint only.

## Takedown

If you hold rights to material here and want it removed, open an issue naming the file and
the source. Removing it here is enough on our side; mirrors are handled separately.
