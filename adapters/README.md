# LoRA Adapters

Reference identities for the fine-tuned adapters used in the experiments. The weight
files themselves are hosted elsewhere and are **not** part of this artifact.

## Adapter list

| Key | Label | Domain | Batch | SHA256 (first 16 hex) |
|:--|:--|:--|:--|:--|
| sft_08b | SFT Baseline 0.8B | control | 0.8b | 4c35a3f0400fcb66 |
| rlvr_08b | RLVR-C v2 0.8B | control | 0.8b | 0c74318b2f669001 |
| mgpo_eg_08b | MGPO-EG 0.8B | control | 0.8b | 62ff06eeac0dba35 |
| sft_2b | SFT Baseline 2B | control | 2b | 7192a1ed7b85ec10 |
| rlvr_2b | RLVR-C v2 2B | control | 2b | d48fd45321d6a638 |
| mgpo_eg_2b | MGPO-EG 2B | control | 2b | ff9f070ac10e6e56 |
| sft_e2b | SFT Baseline E2B | control | e2b | 45f538ed0c59948d |
| rlvr_e2b | RLVR-C v2 E2B | control | e2b | ae89c4fb37d87b82 |
| mgpo_eg_e2b | MGPO-EG E2B | control | e2b | 7a05d4ffca9b27ed |
| mgpo_fixed_08b | MGPO-Fixed 0.8B | control | 0.8b | 56812ba30db65892 |
| mgpo_fixed_2b | MGPO-Fixed 2B | control | 2b | 8fff52885c5b6513 |

## Hosting links

*Not published yet.* The weights are not distributed with this artifact; a separate
Hugging Face release is planned but has not run. The 16-hex-digit SHA256 prefixes above
are URL-independent **identity credentials**: anyone holding the weights can match them
against this list regardless of where the weights eventually live.

## Scope

The full arm-level registry (with per-item `records` and the `models` mapping) is not
distributed here; this table is the complete adapter identity information that ships,
as described in `../ARTIFACT.md`. Generated 2026-09-29 from working-tree HEAD
`3c4e594d49008e600425bcab78243741d4e3726f`.
