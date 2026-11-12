# Chronos-2 TukuruSF execution analysis

Parent [#548](https://github.com/oka-yuji/coreai-samples/issues/548) separates conversion, Swift runtime, Models registration, fixed NYC interaction, verified API dates and immutable prediction records into six implementation Issues.

This report captures the implementation workflow before the final journal review and merge. SF execution, protected verification, parent checks, independent review, integration and user acceptance are separate states. The live local HTML also displays these distinctions.

## Reproducible task boundary

Each SF submission records a fixed base, specification hash, allowed/protected paths, an isolated worker and protected verifier. Filtered source snapshots retain the original target commit and per-file hashes. Hash-pinned external Swift dependencies remain separate from the 32 MB source snapshot. The parent checks the candidate fingerprint and unchanged original source before applying it to the isolated real checkout.

The parent then runs the repository checks, opt-in model/normal-app verification under the shared model lock, self-review and actual Claude Code branch review. Review receipts bind exact base/head trees and file coverage. Passing CPU fixtures does not authorize a native or UI claim. Passing a process alone does not authorize merge.

## Runs

| Run | SF state | Elapsed seconds | Changed paths | Parent outcome at capture |
| --- | --- | ---: | ---: | --- |
| `cf98e9fa3a354a8ab69816050cfd2100` | failed | 900.16 | 0 | No acceptance recorded in this field; inspect the retained execution result. |
| `632c73462f554d36869a4ffcc9c119b6` | failed | 224.48 | 3 | No acceptance recorded in this field; inspect the retained execution result. |
| `9c048ee5b2724790a8cf58a6bf236b25` | verified_pending_review | 207.05 | 2 | No acceptance recorded in this field; inspect the retained execution result. |
| `0cdb8c7ab25841e980a09339f29cb0d2` | verified_pending_review | 1699.07 | 17 | 親側の検証・レビュー完了 / PR #542をmainへ統合 |
| `80e8da5aec6f4c9ea64a09291420f3e2` | verified_pending_review | 1505.19 | 20 | 親側の検証・レビュー完了 / PR #551をmainへ統合 |
| `3759279a1c9644608612e94f660f7550` | verified_pending_review | 859.46 | 18 | 親側の実bundle検査後、UI配線を追加修正 / PR #553統合 |
| `9edc4896874b417d93741a234d98dde4` | verified_pending_review | 818.48 | 10 | Mac実入口・分類翻訳の修正を通常UIと両レビューで確認 / PR #553統合 |
| `4f19867b797140b2b3545c4425d34314` | verified_pending_review | 1350.09 | 56 | 親の全自動検証と実Claude再レビュー完了／PR555をmainへ統合／D7本人待ち |
| `86e55f1814764920b00b786909d99dc6` | verified_pending_review | 1993.89 | 34 | 親の参照範囲修正・全検証・両レビュー後PR556をmainへ統合 |
| `1cc579f82ed84eee8fda719e2294d24b` | verified_pending_review | 1233.11 | 29 | 範囲修正と親の全実測・GUI修正・両レビュー完了／PR556統合 |
| `32b22fcec8bd4cf39355d7420938be5d` | verified_pending_review | 2015.65 | 26 | SF CPU合格後、親がDate・日時domain・公開許可の不足を検出。修正と最終native/GUI成功、最終レビュー待ち。 |
| `460da822a9784b38863c39b5af4f7298` | failed | 3108.79 | 19 | SF容量条件のfailedを保持。同じ候補/保護370件はhost合格。別のlock競合とlocale修正後、親native/GUI成功。最終レビュー待ち。 |

The [machine-readable run ledger](runs.json) preserves the full identities, verification results and reported CLI token usage. Usage describes only the events reported by that worker CLI, not the parent conversation or all review work. No cost or resolved-model identifier is inferred. A nonzero worker command can be an intentional red test or a no-match search; it is not automatically a failed SF run.

## Failures and interventions

The first exploration reached its 900-second timeout and remained incomplete. Another worker completed, but the parent-supplied isolated Python verifier could not import Tools; a new run used the corrected invocation. Both original failures remain in the ledger. The full-source size limit was detected before launching that unsupported snapshot and is not counted as an executed failure.

Parent integration work found gaps that CPU verification alone did not cover: the actual Mac sidebar route, preview interaction, localized interpolation keys, scroll ownership and overlapping-chart accessibility. The API review also showed that annual metadata alone could not prove an absent group was zero. A new SF task restricted the app to independently verified counts, revisions and periods. January/February source reconciliation preserved the first nine-cell mismatch instead of treating it as acceptable error.

Parent command/path mistakes, host permission limits, service responses and daemon interruption are retained separately from SF worker defects. Recovery rechecked running processes, shared-lock ownership and source/artifact identities before continuing. Existing user changes were backed up and preserved with per-entry catalog comparison; no reset or stash discarded them.

The journal publication repair retained an SF failure when its sandbox reported zero important-usage capacity. The parent reran the identical immutable protected verifier on the host, with all checks enabled, and recorded a separate receipt binding the same specification and candidate fingerprint. SF state and result files were not rewritten. Applying that candidate still required parent validation and both reviews. An additional lock-initialization race reproduced outside Swift; it was tracked and corrected separately, not dismissed as part of the capacity limitation.

## Improvement candidates

The [evidence-linked improvement list](improvements.json) identifies the affected component, observation, proposed change and the workaround actually used in this task. These are candidates for TukuruSF development, not claims that TukuruSF itself was modified.

The most useful next changes are verifier preflight, explicit external-dependency inputs, artifact-level UI acceptance, candidate/coverage-aware review gates and durable parent acceptance receipts. Process recovery should report the existing job before offering another execution. Public-text history audits would benefit from progress counters.

## Remaining acceptance boundary

Physical iPhone inference and the user’s Mac/iPhone acceptance stay unchecked until performed. A signed host build does not replace device interaction, model memory measurement or forecasting-quality evaluation. The final parent Issue and PR retain their current acceptance status after this report snapshot.
