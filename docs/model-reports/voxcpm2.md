# VoxCPM2

[Home](../../README.md) · [日本語](voxcpm2.ja.md)

Speech generation

The six FP32 Core AI assets of OpenBMB VoxCPM2 match the original components within
the recorded criteria, and FP32 generation from text was checked on a Python host. Three
app bundles (FP32, FP16, and FP16 with `proj` in FP32) are made, but no app reads them
yet. The default configuration will be chosen after a listening comparison.

The audio comes from the same weights run with Core AI. Numerical differences near
rounding boundaries inside the model can make the generated result differ from the
reference implementation, even with the same seed.

Configurations with FP16 assets produce audio on a different trajectory from FP32. Their
stop patch equaled FP32 in 10 of 15 generations. Their audio quality has not been
evaluated; no listening comparison has been done.

Not verified: iPhone, the Mac app, reference audio (voice cloning), streaming playback,
the Neural Engine, 8-bit and 4-bit quantization, behavior for general sentences and
seeds, and quantitative audio quality.

<!-- model-report:start -->
<a id="model-report"></a>

## Evaluation summary

[Measurement protocol](../benchmarking.md) · [Source record](records/voxcpm2.json)

Results below retain their original dates and workloads. Different rows may use different runs; they are not a controlled comparison. Legacy records explicitly retain unrecorded conditions. Missing values are not zero or a pass.

### Demo support

| Platform | State | Verified scope / remaining work | Evidence |
|---|---|---|---|
| macos | — | No native demo yet. Results of the Python host are not app support. | — |
| ios | — | No native demo yet. Results of the Python host are not app support. | — |

### Performance and quality

| Metric | Result | Statistic / samples | Scope and evidence |
|---|---|---|---|
| Bundle / payload size | 9,349,325,520 B | single; n=1 | [E1](../validation/voxcpm2-conversion-20261117.md) · fp32 bundle: 22 files, including the manifest, tokenizer, license and notice. |
| Bundle / payload size | 4,677,413,664 B | single; n=1 | [E1](../validation/voxcpm2-conversion-20261117.md) · fp16 bundle: 22 files. FP16 results are recorded only. |
| Bundle / payload size | 4,715,188,166 B | single; n=1 | [E1](../validation/voxcpm2-conversion-20261117.md) · fp16_proj32 bundle (FP16 with proj in FP32): 22 files. FP16 results are recorded only. |
| Prepared cache size | Not measured | — | Not measured; the size of the specialization cache was not recorded. |
| First load, including specialization | Not measured | — | Not measured as defined: the load times of the memory runs (fp32 7.07 s, fp16 3.40 s) did not control the specialization cache. |
| Load from prepared cache | Not measured | — | Not measured as defined: the load times of the memory runs did not control the specialization cache. |
| Request / pipeline completion time | 7.81934967 s | single; n=1 | [E3](../validation/voxcpm2-conversion-20261117.md) · fp32: one generation of long (16.8 s of audio) as the first of a new process; excludes the 7.07 s load. |
| Request / pipeline completion time | 5.10555254 s | single; n=1 | [E3](../validation/voxcpm2-conversion-20261117.md) · fp16: one generation of long (16.8 s of audio) as the first of a new process; excludes the 3.40 s load. |
| Peak RSS | 22,462,464,000 B | max; n=1 | [E3](../validation/voxcpm2-conversion-20261117.md) · fp32: process lifetime maximum resident size (getrusage), including file-backed weight pages. Python host, not app memory. |
| Peak RSS | 11,290,116,096 B | max; n=1 | [E3](../validation/voxcpm2-conversion-20261117.md) · fp16: process lifetime maximum resident size (getrusage). Python host, not app memory. |
| Peak physical footprint | 842,347,816 B | max; n=1 | [E3](../validation/voxcpm2-conversion-20261117.md) · fp32: kernel lifetime maximum physical footprint of the process. Python host, not app memory. |
| Peak physical footprint | 662,626,224 B | max; n=1 | [E3](../validation/voxcpm2-conversion-20261117.md) · fp16: kernel lifetime maximum physical footprint of the process. Python host, not app memory. |
| Resident memory during inference | 16,731,029,504 B | single; n=1 | [E3](../validation/voxcpm2-conversion-20261117.md) · fp32: resident size sampled after the generation with the assets still loaded. |
| Resident memory during inference | 8,580,218,880 B | single; n=1 | [E3](../validation/voxcpm2-conversion-20261117.md) · fp16: resident size sampled after the generation with the assets still loaded. |
| Memory after release minus baseline | 670,462,648 B | single; n=1 | [E3](../validation/voxcpm2-conversion-20261117.md) · fp32: physical footprint after closing the components minus before loading. System wired memory had not fallen at that sample (see evidence). |
| Memory after release minus baseline | 533,831,024 B | single; n=1 | [E3](../validation/voxcpm2-conversion-20261117.md) · fp16: physical footprint after closing the components minus before loading. System wired memory had not fallen at that sample (see evidence). |
| Task success fraction | Not measured | — | Not measured; no task-quality evaluation set. |
| Reference error (definition in evidence) | 4.62576051e-06 error | max; n=2151 | [E1](../validation/voxcpm2-conversion-20261117.md) · Largest FP32 component error (max abs error / max abs reference), dit.main; proj.fuse is judged by classes instead (233 match, 1 boundary-consistent, 0 fail). n counts dit.main calls. |
| Stop acknowledgement time | Not measured | — | Not measured; no app or stop path yet. |
| Time to first playable audio | Not measured | — | Not measured. The Python host does not stream audio, and no app reads the bundles yet. |
| Real-time factor (generation / audio duration) | 0.455897562 ratio | median; n=15 | [E2](../validation/voxcpm2-conversion-20261117.md) · fp32: generation seconds per second of audio, after one warm-up; excludes loading and saving. |
| Real-time factor (generation / audio duration) | 0.294079473 ratio | median; n=15 | [E2](../validation/voxcpm2-conversion-20261117.md) · fp16: as above. Audio follows a different trajectory from FP32; listening pending. |
| Real-time factor (generation / audio duration) | 0.294309107 ratio | median; n=15 | [E2](../validation/voxcpm2-conversion-20261117.md) · fp16_proj32: as above. Audio follows a different trajectory from FP32; listening pending. |
| Character error rate | Not measured | — | Not measured; no speech recognition evaluation. |
| Human quality score (rubric required) | Not measured | — | Not measured; no listening comparison has been done. |

### Verification coverage

| Check | State | Scope / remaining work | Evidence |
|---|---|---|---|
| Conversion / reference fidelity | Passed (stated scope) | FP32 components, FP32 generation from text of five sentences with three replaced patches, and the FP32 bundle re-check pass. FP16 is recorded only; listening is pending. | [E1](../validation/voxcpm2-conversion-20261117.md) |
| Task quality | Not measured | No quality evaluation; the default configuration will be chosen after a listening comparison. | — |
| Repeated use / resource release | Not measured | No repeated use or release check in an app. | — |
| Stop / retry / recovery | Not measured | No stop or retry check. | — |

### Evidence and conditions

| Evidence | Date / record / route level | Device / OS / tools | Workload and recorded conditions |
|---|---|---|---|
| [E1](../validation/voxcpm2-conversion-20261117.md) | 2026-11-17 / legacy / conversion | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Python 3.11.9; torch 2.9.0; transformers 4.57.3; coreai-core 1.0.0b3; coreai-torch 0.4.3 | Five fixed sentences with seed 1708: 3,934 component comparisons that decide the result, FP32 generation from text with at most one replaced patch per sentence (3 of 239), and the same comparisons with the FP32 bundle after debug-information removal. No app; the Python host only. · variant=fp32 bundle; backend=Python Core AI host, GPU-preferred specialization; compute units not profiled; route=VoxCPM2 recipe: original FP32 CPU reference -> component comparisons -> FP32 generation from text -> debug-information removal -> FP32 bundle re-check; source_revision=7936080b41163874f38610f46f438e2c0757f9fd; artifact_revision=f771b72d2b18eb01b0a832cadee4bfff37ca7c76a3627defbf1f868d158d70e8; batch_size=1; concurrency=1; unrecorded: code_revision, warmups, power, thermal, plan_sha256 |
| [E2](../validation/voxcpm2-conversion-20261117.md) | 2026-11-17 / legacy / diagnostic | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Python 3.11.9; torch 2.9.0; transformers 4.57.3; coreai-core 1.0.0b3; coreai-torch 0.4.3 | Five sentences x seeds 1708-1710 per configuration, each timed after one warm-up generation in the same worker; load average 4.1-6.1. Recorded without a pass or fail. The bundles hold the same programs without debug information. · variant=fp32, fp16 and fp16_proj32 configurations; backend=Python Core AI host, GPU-preferred specialization; compute units not profiled; route=VoxCPM2 recipe synthesis_compare: exported assets, one worker process per generation; source_revision=7936080b41163874f38610f46f438e2c0757f9fd; batch_size=1; concurrency=1; warmups=1; unrecorded: artifact_revision, code_revision, power, thermal, plan_sha256 |
| [E3](../validation/voxcpm2-conversion-20261117.md) | 2026-11-17 / legacy / diagnostic | Apple M5 Max / 137438953472 B RAM / macOS 27.2.1 (26B131) / Xcode 27.2 (27B52); Python 3.11.9; torch 2.9.0; transformers 4.57.3; coreai-core 1.0.0b3; coreai-torch 0.4.3 | Sentence long (97 text tokens, 105 patches, 16.8 s of audio), seed 1708, first generation of the process. Process memory sampled before and after load, after the generation and after release by a one-off script outside the repository. Recorded without a pass or fail; not app memory. · variant=fp32 and fp16 bundles; backend=Python Core AI host, GPU-preferred specialization; compute units not profiled; route=One new Python process per bundle: load -> one generation of long -> release, under the shared heavy-job lock; source_revision=7936080b41163874f38610f46f438e2c0757f9fd; artifact_revision=bundles of 20261116T204724-264d1994 (same main.mlirb and main.hash as the current fp32 and fp16 bundles); batch_size=1; concurrency=1; warmups=0; unrecorded: code_revision, power, thermal, plan_sha256 |

### Setup, limitations and progress

- [#573](https://github.com/oka-yuji/coreai-samples/issues/573)

- [Conversion recipe](../../Tools/Conversion/recipes/voxcpm2/README.md)

- [Conversion and bundle validation](../validation/voxcpm2-conversion-20261117.md)
<!-- model-report:end -->
