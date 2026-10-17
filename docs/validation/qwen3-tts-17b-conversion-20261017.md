# Qwen3-TTS 1.7B CustomVoice conversion validation

2026-10-17. Issue #169, criteria Q17-1–Q17-7. Four FP32 Core AI components passed
52 original-output comparisons and generated all three Japanese fixtures to EOS.
The shared adapter also passed a fresh 0.6B regression. Listening remains pending.

The [numeric summary](qwen3-tts-17b-conversion-20261017.json) contains source
identities, asset hashes, numerical metrics, audio results, memory observations
and the 0.6B regression. The model revision is `0c0e3051f131929182e2c023b9537f8b1c68adfe`;
Qwen source remains `022e286b98fbec7e1e916cb940cdf532cd9f488e`.

Execution used M4 Max 128 GiB, macOS 27.0 (26A428), Xcode 27.0 (27A266a),
PyTorch 2.9.0, Transformers 4.57.3 and coreai-torch 0.4.3. Original FP32 outputs
used CPU/eager; the converted components used OS Core AI with GPU preferred.
All model jobs ran sequentially under the shared heavy-job lock.

## Acceptance evidence

| Criterion | Evidence | Result |
|---|---|---|
| Q17-1 | Thirteen files verified against HF Git blob/LFS identities and SHA-256; source/width/profile tests and mixed-export rejection | Pass |
| Q17-2 | Projection 12, talker 16, predictor 21, codec 3 records; includes positions 0/255/256/511, finite equal-shape outputs, maximum normalized error 8.7973e-6 | Pass |
| Q17-3 | Same-code audio minimum cosine 0.99999999999772, maximum RMS error 1.0863e-7; gates remain 0.999 and 0.001 | Pass |
| Q17-4 | Three EOS WAVs, lengths, times, peak RSS and actual call shapes saved; category-level footprint snapshots retained separately | Pass |
| Q17-5 | 91 conversion CPU tests, no skips; independent width/export/generation probes; 0.6B real-model 52-record and three-utterance regression | Pass |
| Q17-6 | Repeat run integrity-checked and reused all seven stages | Pass |
| Q17-7 | Three generated Japanese WAVs available in the local bundle | User listening pending |

## Model boundaries and results

The talker width changes from 1024 to 2048. Predictor embeddings also use the
talker width, while its original biased projection maps those inputs to the
internal width 1024. The wrapper already preserves that projection. Profile-aware
source and runtime checks now validate the raw width and reject wrong-size inputs.
The source's two generation loops, sampling order and fixed causal padding are
unchanged. Existing 0.6B contracts are preserved without new required fields.

| Component | Asset bytes | Largest normalized error |
|---|---:|---:|
| Text projection | 33,573,846 | 2.1516e-6 |
| Talker and codec head | 5,663,351,546 | 2.3178e-6 |
| Code predictor and fifteen heads | 448,914,150 | 3.4264e-6 |
| 12 Hz codec decoder | 456,253,659 | 8.7973e-6 |

The four assets total 6,602,093,201 bytes. They require the pinned source Python
host for tokenization, prompt preparation and embedding lookups. They are not a
standalone Swift or iPhone bundle. Core AI calls remain stateless with no KV cache.

Fixtures use Ono_Anna/Japanese, non-streaming input, seed 42, sampling in both
decoders, temperature 0.9, top-k 50, top-p 1.0, repetition penalty 1.05 and a
128-token budget. Numerical gates remain normalized maximum error ≤0.001 and
same-code audio cosine ≥0.999 / RMS error ≤0.001.

| Fixture | Audio seconds | Generation seconds | Talker tokens | Stop |
|---|---:|---:|---:|---|
| greeting | 4.56 | 11.141 | 58 | EOS |
| instruction | 3.84 | 10.152 | 49 | EOS |
| reading | 3.68 | 10.089 | 47 | EOS |

All main token sequences and audio lengths matched the original model. Complete
WAV RMS differences were 7.32e-8–1.09e-7, with minimum cosine 0.99999999999772.
These observations cover the three fixtures and do not establish Japanese reading
accuracy or naturalness. Generation timing excludes source hashing, model loading
and WAV saving. Footprint diagnostics were active, so these are observational
times, not an isolated latency benchmark or a 0.6B-versus-1.7B speed comparison.

## Memory and regression

Peak process RSS during the three-utterance generation was 20,477,050,880 bytes
(about 19.07 GiB), including the coexisting source PyTorch model. Talker and
predictor each used one execution shape: 512 positions at raw width 2048, and
16 positions at raw width 2048, respectively.

Three `footprint -w -f bytes --json` snapshots measured 9,313,252,000,
9,522,770,832 and 9,689,232,464 bytes. At the last snapshot, 18:03:55 JST, the
kernel-reported peak through that time was 9,750,459,472 bytes. This is not the
process-exit peak. Category data, RSS samples and system memory snapshots are
retained locally; RSS and physical footprint are different measurements. The
three sentences do not qualify endurance or iPhone memory requirements.

The fresh 0.6B regression passed all 52 numerical records with maximum normalized
error 7.0658e-6, then generated all three fixtures to EOS. Its main sequences and
decoded Float32 samples were identical to the previously validated 0.6B run.
The WAV containers differ in their PEAK timestamp bytes; file-byte equality is
not claimed.
Independent CPU checks also accepted all four old 0.6B export contracts without
loading a Core AI model and rejected their use with the 1.7B contract.

The package CPU tests, strict SwiftLint, formatting, Rust input build and macOS
Release build already passed at `dfc037a`. Their ten source/configuration/build
input Git objects are identical in this branch; that evidence is reused and its
object identities are recorded in the numeric JSON. No new Swift build is claimed.

## Provenance and review

- `20261017T090123-f7c8476a`: fresh source reference, four exports, 52 comparisons and three generated WAVs; 168.894 seconds for the seven-stage pipeline.
- `20261017T090504-4bd7e786`: fresh 0.6B regression under
  `.local/conversion/work/qwen3_tts_06b_regression_169/runs`.
- `20261017T090757-2efbe59e`: all seven 1.7B stages integrity-checked and reused; no fresh model measurement.

Raw data remains under `.local/conversion/work/qwen3_tts_17b` and
`.local/conversion/qwen3_tts_17b`. The convenient copy at
`.local/conversion/bundles/qwen3-tts-17b-customvoice-fp32-20261017` contains real
component files, metadata, three WAVs and a SHA-256 manifest checked against the
verified originals.

Independent numeric and pipeline reviewers inspected model identities, raw versus
projected widths, source capture, fixed exports, generation, old contracts and
artifact mixing. Tiny identity-width and 16→8 projection models matched native
capture, all fifteen exported heads, and sampled/greedy/token-limit generation.
No confirmed implementation findings remained. Actual model execution and memory
observations were performed by the parent under the shared lock.

## Reconciliation and remaining work

This change depends on PR #168 and is reviewed against its branch at `9991056`.
Merge order is #168 then this PR; rollback order is reversed. The 0.6B regression
covers the combined shared adapter. No PR was merged or model released.

The scope and numerical criteria remain as approved. The provisional 5–8 SP
estimate is retained independently of elapsed time. The recipe, validation record,
TTS plan and Core AI guide describe the new variant and its limits. Listening is
the remaining user check. Swift, iPhone, quantization, KV optimization and broader
quality/endurance evaluation remain separate work.
