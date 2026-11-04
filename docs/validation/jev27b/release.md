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

The repository became public on 2026-11-04 at 12:54:47 UTC, with `main` at
`acc4cc835f64a75ab9ecee9d5d2e119a3bd29212`. The complete distribution contains 19 files and 16,064,854,276 bytes;
root documentation and examples account for the difference from the model directory. [release.json](release.json)
records each file's SHA-256, the model identities, the checks and their scope.

Before publication, `model-release-audit` passed the complete local payload, the private repository's three
reachable commits and the final local/remote gate. There were no unresolved path/credential findings, exceptions or
read failures. The remote head manifest matched the local one. LFS attributes were committed and read back before
uploading the full payload so the server did not add an unaccounted file change.

The authenticated `hf download --local-dir` path used the official cache populated from the fixed repository
revision. All 19 files matched their remote sizes and SHA-256 or Git blob identities; the strict `decision.json`
check and `SHA256SUMS` passed. The downloaded bundle ran the authored 61-token example and returned the same full
logits and probabilities as the packaged response. The download cache stayed outside `bundle/`.

After publication, fresh anonymous HTTP requests fetched README and decision.json with HTTP 200 and identical
bytes. A model-body Range request returned HTTP 206 with `bytes 0-0/16044285337`; only one byte was read. The client
used `trust_env=False`, and all seven requests carried no Authorization header. This establishes anonymous resolve
access; it is not a fresh anonymous full-model download.

The anonymous full-history audit passed on 2026-11-04 at 13:33:26 UTC. It rehashed and scanned every fixed cached
file from the three reachable commits (18 unique objects; 24 records including ref/commit metadata), with no
unresolved findings, exceptions or read failures. Its head manifest, refs and scanner identity matched the private
checks. All 15 HTTP sends, including the initial access checks, carried no Authorization header; the helper files
were unchanged. A final anonymous read at 13:33:28 UTC confirmed the same public, ungated repository, HEAD and refs
with two more requests and no Authorization header. Raw operational reports are retained privately; their digests
and scoped results are recorded in release.json.

## Known limits

The [model guide](../../jev27b.md#verified-scope-and-limits) retains the failed adapter-merge threshold, the scope of
the subsequent fidelity pass, the confidence interval crossing the choice-quality threshold, the distinction between
teacher agreement and real-world correctness, and the repeated-use memory growth. Publication does not change those
results. The distribution is introduced as an Apple silicon Mac Core AI model; no speed advantage over the original
model or Core ML is claimed.
