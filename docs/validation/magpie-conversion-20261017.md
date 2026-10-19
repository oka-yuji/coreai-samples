# Magpie TTS v2607 conversion validation

2026-10-17. Issue #165, criteria MG1–MG8. The macOS FP32 research conversion,
reference comparison, Core AI speech generation and repeat-run checks are complete.
Subjective listening (MG8) and publication remain pending.

The source identities, exact numerical criteria and reviewed results are in
[the numeric summary](magpie-conversion-20261017.json). Execution used M4 Max 128 GiB,
macOS 27.0 (26A428), Xcode 27.0 (27A266a), PyTorch 2.9.0 and coreai-torch 0.4.3.
Core AI used the OS runtime with GPU preferred. The three sources of correctness
evidence are kept separate: original NeMo output, wrapper equivalence before
export, and exported Core AI output.

## Acceptance evidence

| Criterion | Evidence | Result |
|---|---|---|
| MG1 | `MagpieArchiveTests`: final duplicate tar member, unsafe path/link rejection, existing-output protection, changed-source rejection; source archive SHA-256 and sizes checked | Pass |
| MG2 | Original NeMo FP32 CPU: three Japanese utterances, token IDs, 39 intermediate tensor records; all three ended at EOS | Pass |
| MG3 | Encoder 3, decoder 12, local transformer 21, codec 3 records; every output finite and below the fixed normalized maximum error 0.001 | Pass |
| MG4 | Three codec outputs met cosine ≥0.999 and RMS error ≤0.001 against the same reference latents | Pass |
| MG5 | Core AI generated three WAV files; 71/83/73 frames, 3.297/3.855/3.390 seconds of audio, all EOS | Pass |
| MG6 | Fresh source/export run followed by full verification/synthesis; repeat of the full recipe reused all eight stages after integrity checks | Pass |
| MG7 | 94 conversion tests passed with NumPy and NeMo available; no skips. Package CPU tests, strict SwiftLint, format check and macOS Release build passed | Pass |
| MG8 | `greeting.wav`, `instruction.wav`, `reading.wav` are available for listening | User check pending |

| Component | Asset bytes | Largest normalized error |
|---|---:|---:|
| Text encoder | 429,825,889 | 4.6604e-7 |
| Main decoder and head | 481,566,058 | 3.0399e-6 |
| Local codebook transformer and head | 156,324,098 | 1.5520e-6 |
| NanoCodec decoder | 126,470,225 | 4.0784e-5 |

The four assets total 1,194,186,270 bytes. Source embeddings, tokenizers and the
Python host remain separate required inputs; this is not a standalone Swift
deployment bundle. The strict codec audit confirmed 291 encoder, 387 decoder and
16 quantizer state entries. These are state entries, not parameter counts.

With the fixed seed, complete generated WAVs also matched source lengths and had
RMS differences from 2.69e-7 to 5.10e-7 and minimum cosine 0.999999999972.
This observation covers the three fixtures only and does not establish naturalness
or general Japanese quality. It is additional evidence to the forced-input checks.

Private provenance under `.local/conversion/work/magpie/runs`:

- `20261017T044502-07b610a8`: fresh preparation, CPU reference and four exports.
- `20261017T044732-f84ac127`: prior stages verified/reused; Core AI comparison and speech executed.
- `20261017T045149-9cfbdef7`: all eight stages verified/reused, without fresh model measurements.

Raw reports and logs stay local. They are not release artifacts. The convenient
local copy under `.local/conversion/bundles/magpie-v2607-fp32-20261017` was checked
byte-for-byte against the verified export directories and includes the three WAVs.

## Review and corrections

Independent numeric and pipeline reviewers inspected the branch, and a separate
rebuttal pass checked the two numerical blockers. All confirmed findings were
corrected and rechecked:

- N001: the pinned NeMo helper uses the old weight-norm removal API on a new
  parametrization. The export now copies effective weights into fresh plain
  convolutions.
- N002: Tensor-derived padding caused a data-dependent export shape. The fixed
  stride-one causal padding and transposed-convolution trimming are represented
  with integer shape arithmetic in the wrapper.
- N010: an intermediate fix removed parametrizations on a deepcopy and also
  invalidated the original module's shared dynamic class descriptor. The final
  implementation never removes that descriptor. Original/wrapper/export audio
  matched exactly for small CPU models at 2, 7 and 32 frames, including partial masks.
- PIPE-01: the documented generic run command omitted the mandatory shared lock;
  the example now supplies it.

The source codec loader ignores its strict argument. The adapter therefore strictly
loads each inference child and rejects missing weights. This audit passed on the
actual fixed checkpoint. The independent CPU audit also tested legacy weight-norm
keys and missing decoder weights. The host sampler matched NeMo in 32 additional
CPU combinations of sampling, CFG and EOS handling.

## Reconciliation and remaining work

The implementation keeps the approved conversion order and comparison criteria.
It uses full-prefix stateless graphs, as planned for the first research profile.
Shape limits are enforced by the host; production latency, sustained memory,
Swift integration, quantization and iPhone operation have not been qualified.
The original 13–21 SP provisional estimate is retained; elapsed time is not a
conversion of that estimate.

The recipe README and numeric summary describe the verified scope. New conversion
contracts and failures were added to the Core AI guide inbox as they were found.
The next requested model is Qwen3-TTS under #167. Magpie listening remains with the
user; no merge or model publication was performed.

## Integration decision

On 2026-10-19 the user explicitly requested merging PR #166 before the remaining listening check. MG8 stays unchecked; merging the implementation does not complete subjective audio acceptance or publish model weights. The original numerical results and criteria above are retained. Integration checks for the current main are recorded in the PR.
