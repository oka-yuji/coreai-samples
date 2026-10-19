# EmbeddingGemma 2 INT4 / INT8 comparison

2026-10-18. Issue #182 / PR #183. Implementation `71fb61afd24e1fc5324f57a1abc02744d07ae09c`, integrated with main `e300a37e38dab3030a56c3baa5380041734d4901`. Reviewed numeric evidence: [results.json](embeddinggemma2-int4-int8-20261018/results.json). Source model revision remains `914f7f89142e33e77833254d9c9b90c3cef7303b`, text128 / vision70 / 768 output dimensions.

## Delivered behavior

An experimental weight-only PTQ profile can be selected alongside the existing FP16 model in image search and camera search. Text projections and the external vocabulary are INT4; vision projections and the learned position table are INT8. Normalization, bias and scalar parameters retain floating precision and are listed in the results. The existing safe FP16 activation path remains. This is not Google's QAT checkpoint.

The two iPhone folders are `Documents/embeddinggemma2` and `Documents/embeddinggemma2-int4-int8`. FP16 remains the default. Stop camera search before selecting the other folder in Search settings. The image-search Open model menu offers both installed folders and switching clears the old image index. Re-index the same images and keep text, threshold and camera settings fixed for comparison.

## Conversion and numerical fidelity

| Check | Result |
|---|---:|
| FP16 bundle, including tokenizer and notices | 913,677,333 bytes |
| PTQ bundle, including tokenizer, scales and comparison references | 341,880,041 bytes |
| Serialized bundle reduction | 62.58% |
| Fresh seven-stage pipeline after environment setup | 81.566 seconds |
| Native verification stage execute time | 6.581 seconds |
| Matched PTQ PyTorch → native Core AI | 20/20 |
| Minimum native cosine / maximum absolute difference | 0.999257008 / 0.004574524 |
| Original precision → PTQ native minimum cosine | 0.947864461 |

These are M4 Max/macOS27 observations. The pipeline's stage time includes CPU work, loading and I/O; it is not GPU busy time. No new GPU-time counter or iPhone RAM measurement was made. Model file sizes are not RAM footprints. Two earlier exploratory runs took 93.426 and 107.046 seconds; the latter overlapped builds and is not the final timing baseline. Input download and initial environment preparation are excluded.

INT4 vocabulary values occupy 67,108,864 bytes and their FP32 scales 1,048,576 bytes, replacing the 268,435,456-byte BF16 table. The final encoder inventory contains 136,708,096 quantized text elements and 167,706,624 quantized vision elements, including 15,728,640 position-table elements. Text/vision retained parameters are 76,800/51,200 elements; constants and rotary buffers are separate.

PTQ references are generated using the same quantization function as export. The native comparison retains cosine ≥0.999, maximum absolute difference ≤0.01, finite output and unit norm within 0.001. It validates all twenty ordered unique IDs and token sequences. Original-precision differences are reported separately; passing the matched reference does not establish unchanged quality.

## Observed search-quality changes

| Fixed check on Mac | FP16 | PTQ |
|---|---:|---:|
| Original three demo examples | 3/3 | 1/3 |
| Six held-out queries | 5/6 | 5/6 |
| All nine demo queries | 8/9 | 6/9 |
| Original-to-PTQ top-1 agreement, twelve text fixtures against eight images | — | 7/12 |

The Japanese food query still ranks the caprese image first, but PTQ scores about 0.6688, below the unchanged 0.67 display threshold. The egg query changes its first result from omelette to oyakodon. The pre-existing held-out caprese paraphrase failure remains. These small fixtures do not estimate general detection accuracy.

The FP16 application self-test retains its original three-example quality gate. PTQ explicitly records quality separately with `originalExamplesPassed: false`; its `passed` covers numerical and interaction checks. Both profiles pass five-image indexing, original-image access, stop/queued retry and twenty vector checks. Single-run indexing/query times are preserved in JSON, not presented as a controlled speed comparison. The user's live camera evaluation remains necessary.

## Device preparation and limits

The signed iOS Release app is installed on the connected iPhone 17 Pro. Both model folders are present. Each device manifest was copied back and its SHA256 matched the corresponding local bundle; FP16 was preserved. The final PTQ manifest is `7c37ea823af7a4028ec3757e124b53fe8e16a75c99c46d24ca0a7007cca55b85`.

The device rejected automated launch with `Locked` / `FBSOpenApplicationErrorDomain 7`. Therefore iPhone inference, camera detection quality, sustained speed, thermal behavior and peak RAM remain unverified. A user unlock request is pending. This is a prepared comparison demo, not production qualification.

The first installation attempt was unsigned because the isolated worktree lacked ignored Local.xcconfig. The existing local signing configuration was then applied, the signed build and codesign verification passed, and installation succeeded. The source build and signing configuration are distinguished from actual device execution.

## Acceptance and review

| ID | Evidence / state |
|---|---|
| Q1 | CPU tests for signed nibble order, row scale, padding, BF16 compatibility, invalid values/scales/sizes, profile requirements and unknown profiles |
| Q2 | Completed fresh pipeline, encoder inventory, byte counts, separate FP16/PTQ bundles and timing JSON |
| Q3 | Native matched-PTQ20/20; original-precision degradation and rankings preserved separately; saved Mac vectors independently recomputed by a reviewer |
| Q4 | Profile-switch tests clear old image state, reject stale completion, and require camera stop acknowledgement; existing runtime ownership retained |
| Q5 | 79 Python tests; Swift suite success (runner totals551, 35 skip lines for gated cases); strict lint331 files/0 violations; format14 files; macOS and signed iOS Release builds pass |
| Q6 | Signed app and both bundles installed; manifest identity verified; device launch restriction recorded |
| Q7 | Pending user camera/image-search comparison after main merge |

Two independent review passes covered Python conversion/packaging/cache/device verification and Swift runtime/UI/state. The primary reviewed integration and evidence. One confirmed P2 was fixed: an old app could ignore the PTQ launch flag, validate FP16 and be reported as a PTQ success. The helper now requires the returned precision and manifest SHA256 to match the requested bundle, with CPU regression tests for missing/wrong identity. No confirmed code findings remain; the device execution gap is retained.

The planned split alternatives were conversion/runtime versus UI. One PR keeps the packed-table contract, producer, consumer, matched reference and device selection verifiable together. The original provisional8 SP is unchanged. Tests, evidence and generated validation data are counted separately from implementation.

The Core AI guide's production checklist is only partially covered: fixed-shape numerical fidelity, invalid-data rejection, state/stop regression and provenance are checked. General quality, sustained performance/memory, broader shapes and actual iPhone inference are not certified. QAT or alternative PTQ schemes may improve quality but are outside this implementation. No model weights were published.
