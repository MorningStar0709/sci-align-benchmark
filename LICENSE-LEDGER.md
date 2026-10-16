# License ledger

Per-item status lives in `benchmark/dataset/core_v3.json` → `source_license_bucket`.
Buckets were derived mechanically from a per-source license lookup; the source ledger itself is not
published, but its SHA-256 is recorded under `meta.license_tagging` so this file can be re-derived.

| bucket | meaning | items | sources |
|:--|:--|--:|--:|
| `redistributable` | redistributable (CC BY etc.) | 685 | 126 |
| `arxiv-only` | distributed via arXiv only | 927 | 170 |
| `restricted` | restricted (NC / ND etc.) | 241 | 43 |
| `textbook-unchecked` | textbook source; no per-item determination | 126 | 22 |

**What this pool is, and what it is not**

- Every item's question and reference answer are **written by us**. Sources are cited by ID
  (`source_ref`), never reproduced wholesale.
- Where a note quotes a source to justify an audit judgement, **verbatim spans of 300
  characters or more are replaced by a placeholder** (255 evidence notes in the buckets
  below are affected). Our own wording and shorter quotations stay, the way a review keeps them.
- 126 items come from 22 textbooks with no per-source licence
  determination. Stated, not hidden.
- `arxiv-only` means the paper may be distributed *through arXiv*; it is not a grant to redistribute
  derivatives. Our pool items cite those sources at short length and are labelled as such.
- If you hold rights to any item's source and want it removed, open an issue naming the item IDs and
  we will remove it (see `LICENSE.md`).
