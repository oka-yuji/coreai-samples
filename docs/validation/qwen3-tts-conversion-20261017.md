# Qwen3-TTS 0.6B CustomVoice conversion validation

2026-10-17. Issue #167, criteria QT1–QT8. Four FP32 Core AI components passed
52 original-output comparisons and generated all three Japanese fixtures to EOS.
Listening (QT8), merge and distribution remain pending.

The [numeric summary](qwen3-tts-conversion-20261017.json) records pinned source
identities, criteria, asset hashes, runtime shapes and generated-audio metrics.
Execution used M4 Max 128 GiB, macOS 27.0 (26A428), Xcode 27.0 (27A266a),
PyTorch 2.9.0, Transformers 4.57.3 and coreai-torch 0.4.3. The reference used
original Qwen FP32 CPU/eager; exported components used OS Core AI with GPU preferred.

## Acceptance evidence

| Criterion | Evidence | Result |
|---|---|---|
| QT1 | All thirteen source files match the fixed hashes; main and tokenizer weights load strictly; unmanaged tokenizer files and existing outputs are rejected | Pass |
| QT2 | Original source generated three Japanese WAVs, token sequences and 52 tensor records, including positions 0/255/256/511 | Pass |
| QT3 | Projection 12, talker 16, predictor 21 and codec 3 records; finite outputs, matching shapes, normalized maximum error ≤0.001 | Pass |
| QT4 | Same-code codec output has minimum cosine 0.999999999996 and maximum RMS error 8.11e-8; gates are 0.999 and 0.001 | Pass |
| QT5 | Three Core AI WAVs with time, audio length, output path and EOS recorded; original talker sequences and audio lengths match | Pass |
| QT6 | Fresh seven-stage run completed; repeat integrity-checked and reused all seven stages | Pass |
| QT7 | 86 conversion tests passed, no skips; independent numeric and pipeline reviews and rebuttal completed | Pass |
| QT8 | Three generated WAVs are available under the local bundle's `samples` directory | User check pending |

| Component | Asset bytes | Largest normalized error |
|---|---:|---:|
| Text projection | 25,181,219 | 1.5634e-6 |
| Talker and codec head | 1,774,998,280 | 5.1460e-6 |
| Code predictor and fifteen heads | 440,520,792 | 3.5107e-6 |
| 12 Hz codec decoder | 456,253,648 | 7.0658e-6 |

The assets total 2,696,953,939 bytes. The host retains the pinned source tokenizer,
prompt preparation and embedding-table lookups. These assets are not a standalone
Swift or iPhone deployment bundle. Core AI components are stateless and do not
retain a KV cache.

The generation profile uses Ono_Anna/Japanese, non-streaming input, seed 42,
sampling in both decoders, temperature 0.9, top-k 50, top-p 1.0, repetition penalty
1.05 and a 128-token budget. Each talker token is completed by fifteen predictor
tokens before the next talker call. The source's EOS and budget-boundary behavior
is retained, including an incomplete final frame at token-limit termination.

| Fixture | Audio seconds | Generation seconds | Talker tokens | Stop |
|---|---:|---:|---:|---|
| greeting | 3.68 | 4.918 | 47 | EOS |
| instruction | 8.88 | 11.787 | 112 | EOS |
| reading | 4.88 | 6.827 | 62 | EOS |

These times cover the source host's generation call through PCM output, excluding
model loading, source hashing and file saving. They include first-use shapes in
this process but are not guaranteed cold-specialization latency. Complete WAVs
matched source lengths, with RMS differences 2.97e-8–8.11e-8 and minimum cosine
0.999999999996. This covers three fixtures and does not score Japanese naturalness.

## Shape and memory correction

The first variable-shape Core AI profile passed all 52 tensor records. During
subsequent free generation, however, the first utterance took 53.75 seconds and
the second grew to about 60 GiB RSS. System swap usage reached 24,414 MiB; that
is a machine-wide value, not an allocation attributed solely to this process.
The owned runner was cancelled after about eight minutes of the synthesis child
process, and its incomplete run was retained. The separate 14:24 sample displayed
physical footprint 6.0G (peak 6.2G); RSS is not exclusive physical footprint.
Interactive ps/sysctl observations were transcribed into a local JSON record,
not retained as a raw terminal log.

The final profile right-pads talker inputs to 512 positions and predictor inputs
to 16. Causal attention hides future padding, and a Tensor index selects the last
real position. The contract includes these execution shapes and rejects old
assets. Original inputs remain unpadded in reference recordings.

After re-export, all 52 original comparisons passed again. Three sequential
utterances completed with peak process RSS 8,649,900,032 bytes, including the
retained source PyTorch model. The runtime recorded one talker shape over 221
calls and one predictor shape over 3,270 calls. This bounded the observed memory
growth in this run; it does not establish long-duration footprint behavior or
identify the runtime's internal allocation cause.

## Review and validation

Independent reviews found and verified fixes for dynamic codec padding/mask
export failures and an extra `tokenizer.json` overriding pinned vocab/merges.
Native float32 position IDs are now checked for exact integer values and range
before int32 normalization. Numeric acceptance thresholds were unchanged.

Small CPU tests compared fixed padded graphs with original unpadded models at
talker lengths up to 512, position values up to 32767 and all fifteen predictor
heads. A biased width projection and random future-padding values did not change
selected prefix outputs. The complete small codec also matched original eager
output exactly at 1/3/73 frames, across its 72-position sliding-attention boundary.
Native and adapted sampling, forced EOS and token-limit termination were compared.

The package CPU tests, strict SwiftLint, format check, Rust input build and macOS
Release build already passed for the Magpie branch at `dfc037a`. Their ten source,
configuration and build-input Git objects are identical here; that evidence is
reused, not reported as a fresh build. Only conversion Python and documentation
differ. The numeric JSON records those object identities.

Private runs under `.local/conversion/work/qwen3_tts/runs`:

- `20261017T045528-349a607c`: original greedy/repetition-1.0 experiment; all three hit the token limit.
- `20261017T051839-f1ab1a75`: variable-shape numerical pass, cancelled during second synthesis.
- `20261017T053101-a3ace818`: final fixed-shape source, exports, 52 comparisons and three EOS utterances.
- `20261017T053431-dc5b07aa`: all seven stages verified/reused, without fresh model measurements.

The convenient copy at `.local/conversion/bundles/qwen3-tts-06b-customvoice-fp32-20261017`
contains real files, component metadata, three WAVs and a SHA-256 manifest. Copied
assets were checked against the verified originals.

## Reconciliation and remaining work

The approved Magpie → Qwen order and numerical gates were preserved. Sampling
was added after an original-model greedy limit result; fixed shapes address the
observed memory failure. Both earlier outcomes remain recorded. The original
13–21 SP provisional estimate is retained without converting elapsed time into SP.

The recipe, issue and validation summary describe the final profile. New findings
were recorded immediately in the Core AI guide inbox and consolidated with the
TTS plan. Listening remains with the user. KV optimization, Swift integration,
quantization, longer quality tests and iPhone validation are separate work; none
is implied by the FP32 conversion result.
