# JEV-27B Hugging Face release

The distribution is [okayuji/jev-27b-coreai](https://huggingface.co/okayuji/jev-27b-coreai).
The model is in `bundle/`; the model card, root license/notices and authored examples sit outside that directory.
This keeps the Hugging Face local download cache outside the exact inventory checked by `TypedDecisionBundle`.

## Bundle identity

The evaluated bundle is preserved in the [original bundle record](README.md#bundle):
13 files, 16,064,815,496 bytes, decision manifest SHA-256
`a2b186a87a64d9e93a56f185a5ebc2a0613d00da7440eeb8c8c2229b64f6adf3`.

The release model directory has 13 files, 16,064,821,387 bytes, decision manifest SHA-256
`2f34243a3d28500075e09d02d9f9f565a7c4b02f94a7d47b2a3caba7262d357d`.
Only `NOTICE.md`, `metadata.json`, `calibration-refit.json`, `decision.json` and `SHA256SUMS` changed for release packaging.
The backbone asset, head, tokenizer files and license retain the hashes in the evaluated manifest. The backbone `main.mlirb` remains
`965396ce46af30aca1e65df8fe2e0fa4639be120b8bb18a7edeefc5dbbc86de0`.

The public calibration record has schema `typed-decision-calibration-refit-public/1` and SHA-256
`93bcba90332627cef0f5b67cce78eb29b4f8cc55ce4701000d4cef6829be0f79`. It replaces all 38 excluded-row identifiers
with their UTF-8 SHA-256 digests in the same order. `source_schema` and `source_record_sha256` retain the original
representation and digest (`449492cee3d8b363589e6037c0390758455234510015f59a5f18621e17c31eb8`).
The temperatures, complete numeric statistics, fit settings, fixture/input hashes, targets, asset identity and
holdout counts/rule are unchanged. The published calibration ID is `jev27b-refit/93bcba90332627ce`, while the
original evaluation used `jev27b-refit/449492cee3d8b363`. This identifies a publication representation, not a new fit.

The notices name the source model, base model and corpus with pinned revisions; identify the merge, quantization,
float32 gated delta update and temperature refit; and provide a provenance contact and investigation procedure.
No evaluation corpus text is distributed. The request in `examples/request.json` was written for this release.

These packaging changes do not rerun or replace the existing fidelity, quality, Japanese or performance measurements.
Their original artifact revisions, counts, dates and conditions remain unchanged. Native-app integration, iOS,
minimum-memory validation and sustained-use stability are not inferred from the release.

## Publication verification

Publication verification is pending. The model files, complete remote history and final local/remote identity are
checked before visibility is changed; anonymous acquisition is checked after publication. The final results are
recorded here when those operations complete.

## Known limits

The [model guide](../../jev27b.md#verified-scope-and-limits) retains the failed adapter-merge threshold, the scope of
the subsequent fidelity pass, the confidence interval crossing the choice-quality threshold, the distinction between
teacher agreement and real-world correctness, and the repeated-use memory growth. Publication does not change those
results. The distribution is introduced as an Apple silicon Mac Core AI model; no speed advantage over the original
model or Core ML is claimed.
