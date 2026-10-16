# EmbeddingGemma 2 pipeline verification

2026-10-16 JST. Issue #126 connects the existing safe FP16 text/vision conversion to the sample repository's common runner. CoreLLM remains retired. Model math and numerical thresholds are unchanged; source identities, environment setup, packaging and native verification now share explicit stages.

## Measured result

M4 Max 128 GB, macOS 27.0 (26A428), Xcode 27.0 (27A266a), Python 3.11.9. Source `google/embeddinggemma-2` revision `914f7f89142e33e77833254d9c9b90c3cef7303b`, text 128, vision 70, 768-dimensional output. Model files, Python environment and Swift build were already prepared. Timings are single observations, not a controlled speed benchmark or an estimate for unrelated architectures.

| Check | Result |
|---|---|
| Fresh reference, two exports, packaging, native verification | All six stages executed; 71.765 s wall time |
| Identical repeat | Prepare executed; remaining five stages revalidated/reused; 7.379 s |
| Native raw text/image vectors | 20/20 pass; minimum cosine 0.999483553; maximum absolute difference 0.004896015 |
| Normalization | Maximum norm deviation 0.000413965 |
| Model-free tests | 74 tests pass, including existing Qwen/runner regressions |
| Old-workspace independence | Both runs succeeded with `sandbox-exec` denying file reads under CoreLLM and the old conversion experiment |

The fresh run and reuse run are distinct. The reused numerical result is not a fresh measurement. The default prepared demo bundle is a hash-checked copy at `.local/conversion/bundles/embeddinggemma2`; later pipeline runs do not silently replace it.

Evidence: [fresh timing/status](embeddinggemma2-pipeline-20261016/fresh.json), [reuse timing/status](embeddinggemma2-pipeline-20261016/reuse.json), [twenty numerical comparisons](embeddinggemma2-pipeline-20261016/comparison.json). Source paths, environment values, signed-device identity and raw private logs are excluded. These JSON files are reviewed numeric projections; the complete provenance remains in the ignored local run reports.

## Failure boundaries and review

Prepare checks fourteen source file identities, shape/precision, the frozen environment inputs, installed direct versions, native sources, binary and toolchain. Packaging requires explicit matching source revisions and fixture hashes and checks the copied tokenizer/card bytes. A missing revision is rejected; it is not filled with an assumed value. Native verification requires twenty unique ordered cases, correct token IDs and independently recomputed vector agreement.

The first independent environment resolution selected torchao 0.18.0 and failed importing `ScalingType` with Torch 2.9.0. Pinning torchao 0.17.0 and adding the setup import check resolved it. That failed attempt has no completed reference manifest. A macOS `/var` versus `/private/var` test assertion was also corrected to compare resolved paths.

Independent review exercised tiny CPU positive/negative controls for revision and tokenizer mismatch, environment changes, malformed/non-finite/wrong native vectors, ID count/order, and stage reuse/invalidation. It did not run models. A fixture edit invalidates reference and both exports because export uses the FP32 reference for its wrapper check. The final source includes that dependency explicitly.

## Acceptance and limits

| ID | Evidence |
|---|---|
| E1 | Independent dedicated environment; fixed source hashes; old-workspace read denial; all stages executed |
| E2 | Twenty native raw-input comparisons under the original cosine, difference and norm criteria |
| E3 | 74 CPU tests; independent tiny boundary checks; failed run not published as reusable |
| E4 | Five stages reused with recorded origin in the complete private report; prepare checked again |
| E5 | [Recipe instructions](../../Tools/Conversion/recipes/embeddinggemma2/README.md), prepared bundle path, ignored private data, Qwen regressions |

Text/image fidelity is separate from retrieval usefulness. The [demo report](embeddinggemma2-ready-20261016.md) records the 5/6 held-out search result and pending user acceptance. Audio/video, long inputs, arbitrary-image retrieval quality, sustained memory/performance and production qualification are outside this measured contract. No model weights were published.
