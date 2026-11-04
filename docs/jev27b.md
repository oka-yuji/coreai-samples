# JEV-27B typed decisions for Core AI

[Home](../README.md) · [日本語](jev27b.ja.md) · [Download on Hugging Face](https://huggingface.co/okayuji/jev-27b-coreai)

JEV-27B answers typed questions about a situation: `noul` (false or true), `choice` (2 to 16 options) and `score` (0 to
5). This page describes the Core AI bundle that the [`jev27b` recipe](../Tools/Conversion/recipes/jev27b/README.md)
builds from [autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B) and the checks recorded for it. One prefill of
a bare template runs on the Core AI asset, and a 24-row decision head turns the hidden state of the last token into the
probabilities of the options. Download the bundle from Hugging Face and run it on the Mac with
`coreai-verify typed-decide`. The sample app does not use this model; app integration remains separate.

**For:** Swift developers running and measuring typed decisions on the Mac. **Not for:**
free-form chat, production decisions, or probabilities used as real-world correctness without further checks.

Requirements: Apple silicon, macOS 27, Xcode 27 with Metal Toolchain, GPU execution. Measured on a Mac16,6 (Apple M4
Max) with 128 GiB of memory, macOS 27.0 (26A428). A minimum-memory configuration has not been established. Downloading the bundle does not require conversion. Building it from source needs the reference and converter environments of the recipe, the pinned inputs and free disk space (the
export steps check for 160 GB).

<!-- model-report:start -->
<a id="model-report"></a>

## Evaluation summary

[Measurement protocol](benchmarking.md) · [Source record](model-reports/records/jev27b.json)

Results below retain their original dates and workloads. Different rows may use different runs; they are not a controlled comparison. Legacy records explicitly retain unrecorded conditions. Missing values are not zero or a pass.

### Demo support

| Platform | State | Verified scope / remaining work | Evidence |
|---|---|---|---|
| macos | — | Only the coreai-verify command line tool runs the bundle; app integration is #449. | — |
| ios | — | Native demo is not provided for this platform. | — |

### Performance and quality

| Metric | Result | Statistic / samples | Scope and evidence |
|---|---|---|---|
| Bundle / payload size | 16,064,815,496 B | single; n=1 | [E5](validation/jev27b/README.md#bundle) · The 13 files of the calibrated bundle, including the 16 GB asset; counted by performance.py records after plan revision 3. |
| Prepared cache size | Not measured | — | The size of the specialization cache entry is not exposed by the command line tool. |
| First load, including specialization | 28.2243533 s | median; n=3 | [E4](validation/jev27b/README.md#ac-94-w) · New process with the specialization cache entry deleted (--clear-cache); bundle check, tokenizer, head and backbone. File caches not flushed. On AC power (140 W adapter, 94 W), energy mode High Power (plan-d2b-ac94). |
| Load from prepared cache | 9.50495433 s | median; n=3 | [E4](validation/jev27b/README.md#ac-94-w) · New process with the specialization cache entry present; same parts as the first load. On AC power (140 W adapter, 94 W), energy mode High Power (plan-d2b-ac94). |
| Request / pipeline completion time | 0.614704083 s | median; n=3 | [E4](validation/jev27b/README.md#ac-94-w) · P row of exactly 100 tokens; TypedDecisionModel.decide from call to return; lengths interleaved in one process after one warmup each; memory sampled every 100 ms during each row (--sample-memory); every row started at a nominal thermal state (--wait-nominal-thermal 600). The third round's row started after waiting 54.51 s for a nominal thermal state. On AC power (140 W adapter, 94 W), energy mode High Power (plan-d2b-ac94). |
| Request / pipeline completion time | 2.20761833 s | median; n=3 | [E4](validation/jev27b/README.md#ac-94-w) · P row of exactly 500 tokens; TypedDecisionModel.decide from call to return; lengths interleaved in one process after one warmup each; memory sampled every 100 ms during each row (--sample-memory); every row started at a nominal thermal state (--wait-nominal-thermal 600). On AC power (140 W adapter, 94 W), energy mode High Power (plan-d2b-ac94). |
| Request / pipeline completion time | 8.97832325 s | median; n=3 | [E4](validation/jev27b/README.md#ac-94-w) · P row of exactly 2,000 tokens; TypedDecisionModel.decide from call to return; lengths interleaved in one process after one warmup each; memory sampled every 100 ms during each row (--sample-memory); every row started at a nominal thermal state (--wait-nominal-thermal 600). On AC power (140 W adapter, 94 W), energy mode High Power (plan-d2b-ac94). |
| Request / pipeline completion time | 37.1391402 s | median; n=3 | [E4](validation/jev27b/README.md#ac-94-w) · P row of exactly 8,192 tokens; TypedDecisionModel.decide from call to return; lengths interleaved in one process after one warmup each; memory sampled every 100 ms during each row (--sample-memory); every row started at a nominal thermal state (--wait-nominal-thermal 600). The second round's row turned heavy about 20 s into it; the median row (first round) ran at nominal pressure. On AC power (140 W adapter, 94 W), energy mode High Power (plan-d2b-ac94). |
| Peak RSS | 32,333,840,384 B | single; n=1 | [E4](validation/jev27b/README.md#ac-94-w) · /usr/bin/time -l of the latency process, lifetime maximum including the load; not added to the footprint. On AC power (140 W adapter, 94 W), energy mode High Power (plan-d2b-ac94). |
| Peak physical footprint | 21,336,815,344 B | single; n=1 | [E4](validation/jev27b/README.md#ac-94-w) · /usr/bin/time -l of the latency process (peak memory footprint). Not a bound: the footprint grew by 4.1 to 4.5 GB in each round of alternating lengths and had not levelled off. On AC power (140 W adapter, 94 W), energy mode High Power (plan-d2b-ac94). |
| Resident memory during inference | 17,298,096,128 B | max; n=1 | [E4](validation/jev27b/README.md#ac-94-w) · Largest RSS sampled every 100 ms during the measured 8,192-token rows (--sample-memory). On AC power (140 W adapter, 94 W), energy mode High Power (plan-d2b-ac94). |
| Memory after release minus baseline | Not measured | — | The command line process never releases the model before it exits, so no after-release baseline exists. |
| Task success fraction | 0.866666667 ratio | mean; n=300 | [E2](validation/jev27b/README.md#quality) · Stage 3 choice rows of Q whose top option is the teacher top (agreement with the teacher distribution, not real-world accuracy); stage 1' 0.8733. |
| Reference error (definition in evidence) | 0.00654150054 error | max; n=301 | [E1](validation/jev27b/README.md#fidelity) · Largest probability difference of F at temperature 1 between Core AI and the float32 PyTorch run of the same quantized weights. |
| Stop acknowledgement time | 8.84007833 s | median; n=3 | [E4](validation/jev27b/README.md#ac-94-w) · 2,000-token request (the state of the refund card example repeated 164 times, not a P row) cancelled after the test backbone reported its forward call; the backbone does not pass the cancellation on, so the request ends only when the run returns; the times (8.72 to 9.00 s) are longer than the same request without cancellation (8.20 s), for a reason not investigated. Debug build of the test target (swift test). Nominal thermal pressure (the test started after 30 s of waiting for 120 s without user input). On AC power (140 W adapter, 94 W), energy mode High Power (plan-d2b-ac94). |
| Reused-document request time | Not measured | — | Reusing the state of a document between questions is not implemented for typed decisions. |

### Verification coverage

| Check | State | Scope / remaining work | Evidence |
|---|---|---|---|
| Conversion / reference fidelity | Passed (stated scope) | Every band of F within a largest difference of 0.03 and a mean of 0.01 at temperature 1 (largest 0.0065); no top-1 change. | [E1](validation/jev27b/README.md#fidelity) |
| Task quality | Passed (stated scope) | Choice accuracy passes on its point estimate (-0.67 points, limit -2 points), but its 95% interval [-3.00, +1.67] contains the limit (n = 300). noul Brier +0.00024, score normalized MAE +0.00051. Agreement with the teacher distribution, not real-world accuracy. | [E2](validation/jev27b/README.md#quality) |
| Repeated use / resource release | Not measured | Two runs of F with the variant bundle (fresh and cached specialization) gave identical logits for all 301 rows, the GPU test returns bit-identical logits for each of its ten rows (at most 512 tokens) run twice in one process, and up to 3,000 consecutive rows ran in one process (Cal). Releasing and reloading the model in one process was not verified. | — |
| Stop / retry / recovery | Passed (stated scope) | Three requests cancelled during the forward run ended with CancellationError when the run returned, and a request cancelled at the forward checkpoint ended before the run. After both kinds of cancellation the next row returned logits bit-identical to the same row before the cancellation. The test backbone does not pass the cancellation on; whether a Core AI run itself stops was not measured. | [E4](validation/jev27b/README.md#ac-94-w) |

### Evidence and conditions

| Evidence | Date / record / route level | Device / OS / tools | Workload and recorded conditions |
|---|---|---|---|
| [E1](validation/jev27b/README.md#fidelity) | 2026-11-03 / legacy / diagnostic | Mac16,6 (Apple M4 Max), 128 GiB RAM / macOS 27.0 (26A428) / Xcode 27.0 (27A266a), Apple Swift 6.4 (swiftlang-6.4.0.34.1) | F: 302 rows (301 forwarded, one above 8,192 tokens rejected) at temperature 1 against the float32 PyTorch run of the same quantized weights, per input band. · variant=uint4 backbone (gdn-fp32) / float32 head; backend=Core AI; GPU preferred; expectFrequentReshapes true; route=coreai-verify typed-decide (recipe step typed_decide_f) and the PyTorch reference stage 2; source_revision=14b425269ed24452e0a5ad5a8f3e51d139647a88; artifact_revision=26a11dca07a265feca4970dccdf2ebabaa005730132035e9df37b207cd26051e; code_revision=c626c9a27f448961cc9d84addba31ce74573e183; batch_size=1; concurrency=1; unrecorded: warmups, power, thermal, plan_sha256 |
| [E2](validation/jev27b/README.md#quality) | 2026-11-04 / legacy / diagnostic | Mac16,6 (Apple M4 Max), 128 GiB RAM / macOS 27.0 (26A428) / Xcode 27.0 (27A266a), Apple Swift 6.4 (swiftlang-6.4.0.34.1) | Q (900 rows), Cal (3,000) and H (1,500) against the teacher distribution; quantization and adaptation conditions fixed before measuring; temperatures refitted on Cal and selected on H'. Agreement with the teacher, not real-world correctness. Results of 2026-11-03 (stages 0q and 2) to 2026-11-04 (the other stages, the scoring and the calibration). · variant=uint4 backbone (gdn-fp32) / float32 head; backend=Core AI; GPU preferred; expectFrequentReshapes true; route=coreai-verify typed-decide for stage 3 (recipe steps typed_decide_q / cal / h) and the PyTorch reference stages 0, 0q, 1' and 2; source_revision=14b425269ed24452e0a5ad5a8f3e51d139647a88; artifact_revision=26a11dca07a265feca4970dccdf2ebabaa005730132035e9df37b207cd26051e; code_revision=31071af7acef18f77362609bb548233b61ad624a; batch_size=1; concurrency=1; unrecorded: warmups, power, thermal, plan_sha256 |
| [E3](validation/jev27b/README.md#japanese) | 2026-11-04 / legacy / diagnostic | Mac16,6 (Apple M4 Max), 128 GiB RAM / macOS 27.0 (26A428) / Xcode 27.0 (27A266a), Apple Swift 6.4 (swiftlang-6.4.0.34.1) | JGLUE J (200 rows per task, two prompt variants, stages 1' and 3). Measurement completed; no pass line, so passed means completed, not a criterion met. · variant=uint4 backbone (gdn-fp32) / float32 head; backend=Core AI; GPU preferred; expectFrequentReshapes true; route=coreai-verify typed-decide and PyTorch MPS bf16 (recipe steps typed_decide_j / reference_j); source_revision=14b425269ed24452e0a5ad5a8f3e51d139647a88; artifact_revision=a2b186a87a64d9e93a56f185a5ebc2a0613d00da7440eeb8c8c2229b64f6adf3; code_revision=b64eae58e9b4a8954b6b9270a109e6edd6bd22c8; batch_size=1; concurrency=1; unrecorded: warmups, power, thermal, plan_sha256 |
| [E4](validation/jev27b/README.md#ac-94-w) | 2026-11-04 / standard / diagnostic | Mac16,6 (Apple M4 Max), 128 GiB RAM / macOS 27.0 (26A428) / Xcode 27.0 (27A266a), Apple Swift 6.4 (swiftlang-6.4.0.34.1) | P: four rows of exactly 100, 500, 2,000 and 8,192 tokens; latency one warmup and three measured rounds per length in one process, every row started at a nominal thermal state; three fresh and three cached loads in new processes; /usr/bin/time -l and 100 ms sampling for memory; three cancellations of a 2,000-token request. Measured on AC power (140W USB-C Power Adapter, 94 W; battery 100%, charged) in the energy mode High Power, the owner's setting (plan-d2b-ac94). Every stage started at 94 W, thermal pressure level 0, at least 120 s without user input and no audio, video or automation assertions; user input happened during the latency process, the last at about 11:26:13 while it waited for a nominal state (no row ran then); earlier inputs in that process cannot be excluded (HIDIdleTime keeps only the last one). The 8,192-token row of the second round turned heavy about 20 s into it. The warmup is for the latency rounds only; loads and cancellations have none. Other load: no other model process (pgrep before the measurement); the CPU and GPU use of other processes was not recorded. · variant=uint4 backbone (gdn-fp32) / float32 head; backend=Core AI; GPU preferred; expectFrequentReshapes true; route=coreai-verify typed-decide --wait-nominal-thermal 600 (release build); TypedDecisionInferenceTests in the debug build of the test target for cancellation; source_revision=14b425269ed24452e0a5ad5a8f3e51d139647a88; artifact_revision=a2b186a87a64d9e93a56f185a5ebc2a0613d00da7440eeb8c8c2229b64f6adf3; code_revision=18890f539107dd14a65458981db7853d2d959dcb; batch_size=1; concurrency=1; warmups=1; power=AC: 140W USB-C Power Adapter, 94 W (pmset -g ac); battery 100% charged; energy mode High Power (powermode 2 in pmset -g); thermal=pmset -g therm: no warning; ProcessInfo thermalState nominal at the start of every row (the longest wait 54.51 s) and thermalmonitord pressure (mTPL) 0 at the start of every stage; during the latency process moderate from 11:25:51 and heavy 11:25:53-11:26:52 (the 8,192-token row of the second round) |
| [E5](validation/jev27b/README.md#bundle) | 2026-11-04 / standard / conversion | Mac16,6 (Apple M4 Max), 128 GiB RAM / macOS 27.0 (26A428) / Xcode 27.0 (27A266a), Apple Swift 6.4 (swiftlang-6.4.0.34.1) | File sizes of the calibrated bundle, counted after plan revision 3 was frozen; the SHA-256 of decision.json equals the plan's artifact revision. No model load. · variant=uint4 backbone (gdn-fp32) / float32 head; backend=Core AI; GPU preferred; expectFrequentReshapes true; route=performance.py records over the calibrated bundle written by the recipe step calibrate; source_revision=14b425269ed24452e0a5ad5a8f3e51d139647a88; artifact_revision=a2b186a87a64d9e93a56f185a5ebc2a0613d00da7440eeb8c8c2229b64f6adf3; code_revision=e879c67ff30344fc69f5cb6dcde93ed432d9339c; batch_size=1; concurrency=1; warmups=0; power=not applicable (file sizes); thermal=not applicable (file sizes) |
| [E6](validation/jev27b/README.md#battery-power) | 2026-11-04 / standard / diagnostic | Mac16,6 (Apple M4 Max), 128 GiB RAM / macOS 27.0 (26A428) / Xcode 27.0 (27A266a), Apple Swift 6.4 (swiftlang-6.4.0.34.1) | P: four rows of exactly 100, 500, 2,000 and 8,192 tokens; latency one warmup and three measured rounds per length in one process; three fresh and three cached loads in new processes; /usr/bin/time -l and 100 ms sampling for memory; three cancellations of a 2,000-token request. Measured on battery power by the owner's decision (90% to 76%, no adapter, Low Power Mode off; plan revision 3 records this). Thermal pressure moderate to heavy in the third round and the cancellation test. The warmup is for the latency rounds only; loads and cancellations have none. Other load: no other model process (pgrep before the measurement); GPU use of other processes was not recorded. · variant=uint4 backbone (gdn-fp32) / float32 head; backend=Core AI; GPU preferred; expectFrequentReshapes true; route=coreai-verify typed-decide (release build); TypedDecisionInferenceTests in the debug build of the test target for cancellation; source_revision=14b425269ed24452e0a5ad5a8f3e51d139647a88; artifact_revision=a2b186a87a64d9e93a56f185a5ebc2a0613d00da7440eeb8c8c2229b64f6adf3; code_revision=9d8fdf821470168c3d92276808cb9ff1f815d7de; batch_size=1; concurrency=1; warmups=1; power=Battery power, no AC adapter attached (pmset -g ac); 90% at start, Low Power Mode off (powermode 0 in pmset -g). Measured on battery by the owner's instruction on 2026-11-04 06:20 JST; thermal=pmset -g therm: no warning in 10 snapshots; thermalmonitord pressure moderate from 06:28:52 (end of the second round) and heavy 06:29:08-06:32:08 (from the end of the third 2,000-token row through the cancellation test) |
| [E7](validation/jev27b/README.md#ac-70-w) | 2026-11-04 / standard / diagnostic | Mac16,6 (Apple M4 Max), 128 GiB RAM / macOS 27.0 (26A428) / Xcode 27.0 (27A266a), Apple Swift 6.4 (swiftlang-6.4.0.34.1) | P: four rows of exactly 100, 500, 2,000 and 8,192 tokens; latency one warmup and three measured rounds per length in one process; three fresh and three cached loads in new processes; /usr/bin/time -l and 100 ms sampling for memory; three cancellations of a 2,000-token request. Measured on AC power (70W USB-C Power Adapter, 68 W; battery 64% to 67%, charging) in the energy mode High Power, the owner's setting (plan-d2b-ac). Every stage started at thermal pressure level 0; the pressure turned heavy during the latency process (the 8,192-token row of the second round and the third round). The warmup is for the latency rounds only; loads and cancellations have none. Other load: no other model process (pgrep before the measurement); pmset -g log shows other activity during the run (audio sessions, a browser video wake lock, a UI automation session) that the battery run did not have; the CPU and GPU use of other processes was not recorded. · variant=uint4 backbone (gdn-fp32) / float32 head; backend=Core AI; GPU preferred; expectFrequentReshapes true; route=coreai-verify typed-decide (release build); TypedDecisionInferenceTests in the debug build of the test target for cancellation; source_revision=14b425269ed24452e0a5ad5a8f3e51d139647a88; artifact_revision=a2b186a87a64d9e93a56f185a5ebc2a0613d00da7440eeb8c8c2229b64f6adf3; code_revision=9b610f88ba0b7810555384200686472e02cd11f2; batch_size=1; concurrency=1; warmups=1; power=AC: 70W USB-C Power Adapter, 68 W (pmset -g ac); battery 64% to 67%, charging; energy mode High Power (powermode 2 in pmset -g); thermal=pmset -g therm: no warning; thermalmonitord pressure (mTPL) 0 at the start of every stage (the cancellation test waited 50 s); during the latency process moderate from 09:26:46 and heavy 09:26:48-09:29:05 (the 8,192-token row of the second round and the third round) |

### Setup, limitations and progress

- [Conversion recipe](../Tools/Conversion/recipes/jev27b/README.md)

- [Validation records](validation/jev27b/README.md)

- [Japanese (JGLUE), no pass line](jev27b.md#japanese-jglue)
<!-- model-report:end -->

## Quick start

Install the [common prerequisites](../README.md#get-started), including Rust/Cargo, and the Hugging Face CLI.
From the repository root, download the model and run the authored example:

~~~sh
hf download okayuji/jev-27b-coreai --local-dir models/jev27b
./Tools/Irodori/build-inputs.sh
swift build -c release --package-path Tools/Verify --product coreai-verify --force-resolved-versions
Tools/Verify/.build/release/coreai-verify typed-decide --bundle models/jev27b/bundle --input models/jev27b/examples/request.json --output jev-decision-output.json
~~~

The output file must not exist yet. Run model operations under the shared machine lock in `AGENTS.md`.
Keep the `bundle/` subfolder: the Hugging Face download cache, model card and example files belong outside the
strictly checked model inventory. The [release record](validation/jev27b/release.md) connects the published
files with the previously evaluated bundle. The model card includes the authored request and its recorded answer.

To build the bundle from source, use the [conversion recipe](../Tools/Conversion/recipes/jev27b/README.md).
Place the pinned inputs under `.local/conversion`, run setup, build and copy the `coreai-verify` executable,
and run the recipe. The `calibrate` step writes the calibrated bundle
`jev27b_s1_gdnfp32_qwen3_5_cfg_dynamic`. Check a copy from inside its folder with
`shasum -a 256 -c SHA256SUMS`, or from the repository root with
`python3 -m Tools.Conversion.adapters.jev27b.decision <bundle>/decision.json --check-files`.

Custom input JSON holds rows with `id`, `kind`, `state`, `question` and `options`. The runner's output records the
token count, the 24 head logits, probabilities at the bundle temperatures and at temperature 1, the top option and
the time of each decision ([Running on Mac](#running-on-mac)).

## Recorded result

The numbers come from the recipe runs and the performance measurement recorded in
[docs/validation/jev27b](validation/jev27b/README.md). Quality and calibration measure agreement with the teacher
distribution of the corpus (output distributions of Jev 1.13), not real-world correctness.

- **Fidelity** (fixed set F, 302 rows; 301 forwarded and one above 8,192 tokens rejected): at temperature 1, the
  Core AI output differs from the float32 PyTorch run of the same quantized weights by at most 0.0065 and on average
  0.0011, within 0.03 and 0.01 in every input band up to 8,192 tokens, with no change of the top option. Details:
  [`fidelity.json`](validation/jev27b/fidelity.json) (per band) and [Running on Mac](#running-on-mac).
- **Quantization and adaptation** (Q, 900 rows): stage 3 against stage 1' ([Reference stages](#reference-stages)) at
  temperature 1 passes the three conditions fixed before measuring: choice accuracy -0.67 points (limit -2), noul Brier
  +0.00024 (limit +0.01) and score MAE / 5 +0.00051 (limit +0.02). The choice accuracy passes on its point estimate,
  while its 95% interval [-3.00, +1.67] contains the limit (n = 300, two rows of difference). The adaptation keeps at
  least 80% of its bf16 gain at 4 bits in all four metrics. Stage 3: KL 0.0202, top-1 agreement 0.903, choice accuracy
  86.67% (stage 1' 87.33%), noul AUROC 0.993. Details: [Quality and calibration](#quality-and-calibration).
- **Calibration**: the temperatures refitted on Cal from the stage 3 logits are adopted for all three kinds (noul
  1.0056266, choice 1.0053791, score 1.010558), because neither the NLL nor the Brier score on the held-out H' is worse
  than at the card temperatures. The fidelity of F at these temperatures still passes.
- **Japanese** (JGLUE J, 200 rows per task, no pass line): at stage 3 with the adopted prompt variant, JNLI choice
  accuracy 0.855 (`en`), JNLI noul accuracy 0.920 and AUROC 0.978 (`en`), JCommonsenseQA accuracy 0.960 (`ja`), and
  JSTS MAE 0.529, Pearson 0.859 and Spearman 0.792 (`en`). Details: [Japanese (JGLUE)](#japanese-jglue).
- **Performance** (medians of three measured runs; one warmup of each length before the latency rounds; on AC power with
  a 140 W adapter at 94 W, energy mode High Power; every latency row started at a nominal thermal state): one decision
  takes 0.61 s at 100 tokens, 2.21 s at 500, 8.98 s at 2,000 and 37.14 s at 8,192; loading takes 28.22 s with a fresh
  specialization and 9.50 s from the cache; the latency process peaks at 32.3 GB RSS and 21.3 GB physical footprint (not
  levelled off) and keeps about 17.3 GB resident during 8,192-token rows; a request cancelled during its 2,000-token run
  through the test backbone ends 8.84 s later, when the run returns. The median rows ran at nominal thermal pressure.
  Earlier measurements on battery power and with a 70 W adapter are kept with their conditions. Details:
  [Performance](#performance).

## Verified scope and limits

- The merged checkpoint did not pass its output check: stage 1 (the unmerged adapter) and stage 1' (the merged
  checkpoint) differ by up to 0.038 against the limit 0.01. The owner decided to record this and continue with the
  merged checkpoint as a limited exploration; the limit was not changed.
- The bundle is the variant with a float32 gated delta update (`export_gdnfp32`); its quantized weights are identical to
  those of the first bundle. The first bundle exceeded the fidelity limits above 512 tokens (largest difference 0.138 in
  the 4,097 to 8,192 band, #475); the variant stays within them in every band.
- The quality conditions compare stage 3 with stage 1' and the adapted stages with the base stages on fixed sets drawn
  from the distillation corpus. They are agreement with the teacher, not accuracy against human labels. The choice
  accuracy condition passes on its point estimate (-0.67 points against the limit -2), but its 95% interval
  [-3.00, +1.67] contains the limit (n = 300, two rows of difference). The gain condition (the 4-bit gain at least
  0.8 of the bf16 gain) holds for all four metrics (shares 0.96 to 1.12).
- The adopted temperatures were refitted on Cal and selected on H' against the teacher distribution, so the shipping
  probabilities are calibrated to the teacher, not to real-world correctness.
- The Japanese results have no pass line and are not a claim of Japanese quality or of calibration. The prompt variant
  was adopted on a separate set before J was run.
- Inputs up to 8,192 tokens (the whole rendered template) are accepted; longer rows are rejected without a forward
  pass. Only the fidelity set F covers every band: 269 rows up to 512 tokens, 11 from 513 to 2,048, 10 from 2,049 to
  4,096 and 11 from 4,097 to 8,192. The quality, calibration and Japanese sets are short: at most 593 tokens in Q (897
  of its 900 rows up to 512), 672 in Cal, 689 in H and 129 in J with the prompt wrapper. Quality and calibration above
  about 700 tokens were not measured. The performance rows have exactly 100, 500, 2,000 and 8,192 tokens.
- The peak physical footprint of 21.3 GB is the value after 16 requests of alternating lengths in one process, not a
  bound. At the 8,192-token rows it grew from 8.2 to 12.5, 16.6 and 21.2 GB, by 4.1 to 4.5 GB in each round, and had not
  levelled off; the RSS of those rows stayed at about 17.3 GB. The earlier measurements showed the same growth.
- A request cancelled during its forward run keeps the model busy until the run returns and then ends with
  `CancellationError`; a request cancelled before the run ends at once. The cancellation test runs the forward through
  a test backbone that does not pass the cancellation on, so whether a Core AI run itself stops on a cancellation was
  not measured.
- Not verified: iOS, integration into the app (#449), Macs with less memory, releasing and reloading the model in one
  process, and the reuse of a document between questions.

## Input and files

The calibrated bundle used for the recorded evaluations, before release-notice changes, had 13 files and
16,064,815,496 bytes; its `decision.json` SHA-256 was
`a2b186a87a64d9e93a56f185a5ebc2a0613d00da7440eeb8c8c2229b64f6adf3`. The published `bundle/` has 13 files and
16,064,821,387 bytes, with manifest SHA-256
`2f34243a3d28500075e09d02d9f9f565a7c4b02f94a7d47b2a3caba7262d357d`. The [release record](validation/jev27b/release.md)
identifies the publication-metadata changes, the calibration record with hashed identifiers, and the unchanged model files. The layout, file contract, Swift types and
command line tool are described below.

### Bundle layout

```text
jev-27b-coreai/
  README.md
  LICENSE
  NOTICE.md
  examples/
  bundle/
    jev27b_s1_gdnfp32_qwen3_5_cfg_dynamic.aimodel/
    metadata.json
    tokenizer/
    head.safetensors
    calibration-refit.json
    decision.json
    LICENSE
    NOTICE.md
    SHA256SUMS
```

- The asset has one function, `main`. Inputs: `input_ids` and `position_ids` (Int32, 1 x ?). Output: `hidden_states`
  (Float16, 1 x ? x 5120). States: `keyCache` and `valueCache` (BFloat16, 16 x 1 x 4 x ? x 256), `convStates`
  (BFloat16, 48 x 1 x 4 x 10240) and `recurrentStates` (BFloat16, 48 x 1 x 48 x 128 x 128). There is no `prefill`
  function and no `logits` output; the language-model head is not part of the asset.
- `metadata.json` still says `"kind": "llm"` because the exporter writes it for every language-model export. Tools that
  open the bundle identify it by `decision.json` and by the output name `hidden_states`.
- `head.safetensors` and the tokenizer files are byte-for-byte copies of the files distributed with JEV-27B. The
  tokenizer is never saved again.

#### Variant with a float32 gated delta update

The recipe also exports `jev27b_s1_gdnfp32_qwen3_5_cfg_dynamic` (step `export_gdnfp32`), and the Core AI steps run on
this bundle. It has the same layout, states and outputs. In each of the 48 linear-attention layers, the six inputs of
the composite `gated_delta_update` operation (query, key, value, g, beta and the recurrent state) are cast to float32
before the operation, and its two outputs are cast back to bfloat16. In the first bundle, the operation receives
bfloat16 values. The derived class `TypedDecisionBackboneGdnFp32` keeps the original module as a child of the wrapper,
so the exporter still externalizes it as the same composite operation. Quantization and the quantized weights do not
change: the step requires the per-tensor SHA-256 of the quantized model to equal the record of `export`.

- `variant.json` records the hash comparison, the number of wrapped layers, the input types of every
  `gated_delta_update` call in the IR of both assets (read without the weights), the number of `cast` operations and
  the inspection of the asset. Its `passed` is the check of the step.
- `decision.json` differs from the first bundle only in `backbone.asset` and `files`. `NOTICE.md` adds a line for the
  float32 update, and `provenance.json` records `variant: gdn-fp32` and the key of the `export` object it was checked
  against.

Check a copy of the bundle from inside its folder:

~~~sh
shasum -a 256 -c SHA256SUMS
python3 -m Tools.Conversion.adapters.jev27b.decision <bundle>/decision.json --check-files
~~~

The second command runs from the repository root, needs no third-party package, and checks the schema below and the
size and SHA-256 of every listed file.

### decision.json (typed-decision-v1)

| Field | Content |
|---|---|
| `format` | `typed-decision-v1` |
| `backbone` | `asset` (folder name of the `.aimodel`), `function` (`main`), `inputs` (`input_ids`, `position_ids`), `output` (`hidden_states`), `hidden` (`dtype` `float16`, `size` 5120), `states` (the four state names in order), `context_length` (262144, the exported limit) |
| `head` | `file` (`head.safetensors`), `weight` (`proj.weight`), `bias` (`proj.bias`), `weight_shape` [24, 5120], `bias_shape` [24], `dtype` `float32`, `sha256` |
| `slots` | Head rows per kind as `[start, end)`: `noul` [0, 2), `score` [2, 8), `choice` [8, 24). The ranges are disjoint and cover all 24 rows. A choice with n options uses the first n rows of its range. |
| `verbalizers`, `verbalizer_ids` | Copied unchanged from the distributed `judge_config.json`: the 24 verbalizer strings and their token IDs. |
| `template` | `version` `bare-v1`; `text` with the placeholders `{kind}`, `{state}`, `{question}` and `{options}`; `option_separator` (a line feed); `choice_line` `{label}) {option}` with `labels` A to P; `fixed_options` for `noul` (`false`, `true`) and `score` (`0` to `5`); `add_special_tokens` false; `last_token` `]:` |
| `temperatures` | `per_kind` (the `per_kind` values of the distributed `calibration.json`, unchanged, all above 0), `calibration_id` (`<model>@<revision>/calibration.json`), `calibration_sha256`, `source` (`card`). The calibrated copy written by `calibrate` changes these four ([Calibrated bundle](#calibrated-bundle)). |
| `limits` | `options` 16 and `input_tokens` 8192 (the whole rendered input, template included) |
| `ties` | `lowest-index`: equal probabilities select the earliest option |
| `source` | `model` and `base_model` (ID and revision), `adapter_sha256`, `adapter_config_sha256`, `head_sha256`, `merged_model_id`, `merged_revision`, `upstream_revision` (apple/coreai-models), `compression`, `compute_precision` |
| `files` | `path`, `bytes` and `sha256` of every bundle file except `decision.json` and `SHA256SUMS`, sorted by path |

Inputs follow the reference renderer exactly. A `choice` has 2 to 16 distinct options, and an option must not contain a
line break (U+000A, U+000B, U+000C, U+000D, U+001C, U+001D, U+001E, U+0085, U+2028 or U+2029); its lines use
`choice_line` with the labels in order. `noul` and `score` take only their `fixed_options`, written one per line without
labels. Each placeholder of `text` is replaced once, with the value inserted as it is; text that looks like a placeholder
inside a value is not replaced again. Inputs that break these rules are rejected rather than rendered differently.

A decision is computed as follows: render the template, tokenize it with the bundled tokenizer without special tokens,
run `main` once from zero states, read the hidden state of the last position (the token `]:`), compute
`z = proj.weight @ h + proj.bias` in float32, take the slot range of the kind, and apply a softmax to `z / T` over the
options of the question. Temperature 1 gives the uncalibrated probabilities; `per_kind` gives the shipping ones.

### Swift contract

`CoreAIBackend` opens the bundle, renders the template and applies the head without loading the Core AI asset. These
types are separate from the Clef-Flash decision types.

| API | Behavior |
|---|---|
| `TypedDecisionBundle(url:)` | Reads `decision.json` and checks it against the schema above, with the slots exactly as shown and a float16 hidden state; a template that differs from bare-v1 in any key or value is refused. `SHA256SUMS` must list every entry of `files` and `decision.json` with the same digests, the folder must hold exactly the listed files and no links, and the size and SHA-256 of every listed file are computed. Exposes `url`, `name`, `version` (SHA-256 of `decision.json`), `hiddenSize`, `limits` (`options`, `inputTokens`), `temperatures`, `verbalizerIDs`, `headURL`, `tokenizerURL`, `assetURL`, `backboneFunction`, `backboneInputs`, `backboneOutput`, `stateNames` and `contextLength`. Failures throw `TypedDecisionError.invalidBundle`; cancellation passes through. |
| `TypedDecisionRequest` | `state`, `question`, `kind` (`TypedDecisionKind`: `noul`, `choice`, `score`) and `options`. |
| `TypedDecisionRenderer.render(_:)` | The text of the reference renderer. Options that break the input rules throw `invalidOptions`. Options are compared as UTF-8 bytes, so canonically equivalent options with different scalars stay distinct in the text; the bundled tokenizer normalizes to NFC, so their tokens are the same. |
| `TypedDecisionRenderer.tokenIDs(for:tokenizer:limit:)` | Encodes the rendered text without special tokens. More than `limit` tokens throw `tooManyTokens`. |
| `TypedDecisionHead(url:width:)` | Maps `head.safetensors` and checks that it holds `proj.weight` [24, width] and `proj.bias` [24] in float32 inside the file. |
| `logits(hidden:)` | `proj.weight @ h + proj.bias` in float32. A hidden state of another length throws `invalidHidden`. |
| `probabilities(logits:kind:count:temperature:)` | The softmax over the options of the kind, computed in double precision and returned as float32, at the given temperature (`shipping`) and at 1 (`plain`). Non-finite logits and a temperature that is not positive and finite throw `nonFinite`; an option count outside the slots throws `invalidOptions`. |
| `TypedDecisionHead.top(of:)` | The index of the largest value; ties select the earliest. |
| `TypedDecisionAnswer` | `kind`, `options`, `logits`, `probabilities`, `probabilitiesT1`, `top` and `tokenCount` of one decision. |

`swift test --package-path CoreAISamplesKit --filter TypedDecision` runs the rendering, rejection, synthetic head,
bundle and model tests; the model tests use a fake backbone. With `JEV_BUNDLE_DIR` set to the bundle folder, the tests
also compare the token IDs of every fixture row with the bundled tokenizer, compare the bundled head with stage 1'
([Reference stages](#reference-stages); logits within 1e-4, probabilities within 1e-5) and open the bundle, which reads
every listed file, including the 16 GB asset, for its SHA-256. None of these tests loads or runs the Core AI asset;
`TypedDecisionInferenceTests` (below) does, only with `COREAI_TYPED_DECISION_BUNDLE`.

~~~sh
JEV_BUNDLE_DIR=/path/to/jev27b_s1_qwen3_5_cfg_dynamic swift test --package-path CoreAISamplesKit --filter TypedDecision
~~~

The fixtures are in `CoreAISamplesKit/Tests/CoreAIBackendTests/Fixtures/TypedDecision/`: `render.json` (15 synthetic
English rows with the input boundaries, the two card examples of the model card and 8 JGLUE rows of F with their text,
rendered and tokenized in Python), `reject.json` (19 option lists that the reference renderer refuses), `head.json` (a
head of width 4 with six cases computed with `decide`), `real-head.json` (four rows of F with their stage 1' hidden
state, logits and probabilities, and token IDs but no text) and `inference.json` (the card examples and JGLUE rows of
`render.json` with their token count, token SHA-256, band and the stage 2 and 1' values of `reference_q.json`).
Regenerate them in the reference environment:

~~~sh
python -m Tools.Conversion.adapters.jev27b.swift_fixture --snapshot <JEV-27B folder> --corpus <corpus folder> \
  --jglue <JGLUE folder> --fixtures Tools/Conversion/recipes/jev27b/fixtures --reference <reference.json> \
  --merged-reference <reference-merged.json> --hidden <hidden.safetensors> \
  --quantized-reference <reference_q.json> --output CoreAISamplesKit/Tests/CoreAIBackendTests/Fixtures/TypedDecision
~~~

The Swift tokenizer (swift-transformers 1.3.4) matches the Split pre-tokenizer pattern on whole grapheme clusters, so
its token IDs differ from those of the Python tokenizer where a pattern boundary falls inside a cluster. Measured
examples are consecutive CRLF line breaks and Hindi and Thai words with combining marks. The fixtures contain no such
text.

### Running on Mac

`TypedDecisionModel` (an actor in `CoreAIBackend`) runs the bundle on Core AI. These types are also separate from the
Clef-Flash decision types; `DecisionRuntime` and `DecisionModel` are unchanged.

| API | Behavior |
|---|---|
| `TypedDecisionModel.load(bundle:)` | Opens the bundle with `TypedDecisionBundle(url:)`, so a bundle that breaks the contract fails before any asset is read. Then it loads the bundled tokenizer, maps the head and specializes `main` with the GPU preferred and `expectFrequentReshapes = true`, reusing the specialization cache entry when there is one. Before specializing it checks that the asset summary lists the `hidden_states` output; afterwards it checks the descriptors: `input_ids` and `position_ids` Int32 1 x ?, `hidden_states` Float16 1 x ? x 5120, the four states of `decision.json`, no interleaved layout and at most one dynamic dimension per state. `loadStats` holds the seconds of each part and whether the specialization came from the cache. |
| `decide(_:)` | Renders and tokenizes the request (rejections throw before the model runs), runs `main` once from zero states and applies the head. Each request allocates its own four states, with the dynamic dimension set to `capacity(forTokens:)`, the token count rounded up to a multiple of 256 and at most `input_tokens`. `position_ids` are 0 to T - 1 without padding, and only the hidden state of the last position is read. Returns a `TypedDecisionAnswer`. |
| One request at a time | A request that arrives while another is running throws `TypedDecisionError.busy`; requests are not queued. The running request keeps the model busy until its forward pass returns, also after its task is cancelled. Cancellation is checked before and after the forward pass and ends the request with `CancellationError`. |
| `tokenIDs(for:)`, `tokenCount(for:)` | The token IDs and their count for a request, with the same rejections as `decide(_:)`. |

`coreai-verify typed-decide` (in `Tools/Verify`) loads the model once and answers the rows of a JSON file one at a time:

~~~sh
swift build -c release --package-path Tools/Verify --product coreai-verify
Tools/Verify/.build/release/coreai-verify typed-decide --bundle <bundle> --input <in.json> --output <out.json> [--limit N] \
  [--clear-cache] [--sample-memory] [--wait-nominal-thermal S]
~~~

- Input: `{"rows": [{"id", "kind", "state", "question", "options"}]}`; other keys are ignored. `--limit N` runs the
  first N rows. `--clear-cache` deletes the specialization cache entry of the bundle's asset before loading,
  `--sample-memory` adds `resident_bytes_max` to each row, and `--wait-nominal-thermal S` waits before every row, in
  steps of 1 s and for at most S seconds, until the thermal state of `ProcessInfo` is nominal, and adds `started`
  (UTC, whole seconds), `thermal_wait_seconds` and `thermal_state_at_start` to the row ([Performance](#performance)).
- Output (`typed-decide/1`), written once at the end to a file that must not exist yet:
  `bundle` (`name`, `version` (the SHA-256 of `decision.json`), `asset_sha256` (the `main.mlirb` digest in
  `SHA256SUMS`), `specialization` (`cached` or `fresh`), `load_seconds` (`bundle`, `tokenizer`, `head`, `backbone`)),
  `input_sha256`, `started` (ISO 8601, UTC), `host` (`os_build`, `machine`) and `rows`. A row holds `id`, `kind`,
  `token_count`, `tokens_sha256` (the SHA-256 of the token IDs written in decimal and joined by commas), `capacity`,
  `logits` (24 values before the temperature), `probabilities` (shipping temperature), `probabilities_t1`, `top`
  and `elapsed_seconds`. A rejected row holds `id`, `kind`, `error` (`tooManyTokens` or `invalidOptions`) and, for
  `tooManyTokens`, `token_count`. Progress goes to standard error. Any other failure stops the run without an output.
- Running the model is a heavy job: hold the shared machine lock.

The recipe adds three steps after `verify_quantization`, still before `verify_merge`:

- `typed_input` restores the text of F from the corpus and JGLUE, checks that the token counts and bands equal the
  fixture, and writes `input-F.json` and `check.json`. `--sets` takes several sets (`F`, `Q`, `Cal`, `H`, `J`) and
  writes one `input-<set>.json` for each.
- `typed_decide_f` runs `coreai-verify typed-decide` on `input-F.json` with the variant bundle of `export_gdnfp32`
  and writes `swift-F.json`. The executable is the variable `verify`, which must be passed with `--set`
  ([recipe README](../Tools/Conversion/recipes/jev27b/README.md)).
- `compare_f` scores `swift-F.json` against stage 2 ([Reference stages](#reference-stages)) in `reference_q.json` and
  writes `comparison.json` (`typed-decision-fidelity/1`) and `report.md`. `passed` requires the rows of F in order with
  the reference token IDs (by `tokens_sha256`) and a `tooManyTokens` rejection exactly for the rejected row, finite
  probabilities between 0 and 1 that sum to 1 within 1e-3, and at temperature 1, for every input band with rows, a
  largest `e_j = max_i |p3_i - p2_i|` of at most 0.03 and a mean of at most 0.01. Bands without rows are listed as
  unevaluated. Top-1 changes are reported with whether the new option is within 0.06 of the reference top; they are not
  a criterion. The shipping-temperature statistics are recomputed from the 24 logits of both sides and reported
  separately; `--temperatures <json>` (a `per_kind` object) scores them at other temperatures. The difference from stage
  1' is shown for reference only.

With `COREAI_TYPED_DECISION_BUNDLE` set to the absolute path of a bundle folder (the variant, as in the recipe, or
the first bundle), `TypedDecisionInferenceTests` runs ten rows of F (the two card examples and the eight JGLUE rows of
`render.json`, all at most 512 tokens) on Core AI: the band limits against stage 2 at temperature 1 with the token IDs
of the reference, bit-identical results for the same row run twice, a request cancelled at the forward checkpoint and
a request cancelled while its forward pass runs (through a backbone that does not pass the cancellation on), each
followed by a row that still matches, and the difference between the rounded capacity and 8,192 positions. The second
cancellation test then cancels a 2,000-token request three times after the test backbone has reported the start of
its forward call, and writes the time from each cancellation request to the `CancellationError` to
`typed-cancel-run.json` ([Performance](#performance)). The fixture is `Fixtures/TypedDecision/inference.json`,
written by `swift_fixture` with `--quantized-reference <reference_q.json>`. The tests load the model, so hold the
shared machine lock; without the variable they are skipped. `DECIDE_METRICS_DIR` saves their measurements as
`typed-*.json`.

~~~sh
COREAI_TYPED_DECISION_BUNDLE=/path/to/jev27b_s1_gdnfp32_qwen3_5_cfg_dynamic swift test --package-path CoreAISamplesKit \
  --filter TypedDecisionInferenceTests
~~~

## Evaluation details

The stages, the fixed definitions and the full tables of every check.

### Reference stages

| Stage | Model | Precision | Where |
|---|---|---|---|
| 0 | Distributed backbone with the decision head, adapter disabled | bfloat16 | `reference` (F), `reference_q` (Q) |
| 0q | Stage 0 quantized with the export recipe, computed in float32 | float32 | `probe_base` |
| 1 | Backbone with the unmerged LoRA adapter (PEFT) | bfloat16 | `reference` (F), `reference_q` (Q) |
| 1' | Merged checkpoint | bfloat16 | `reference_merged` (F), `reference_merged_q` (Q) |
| 2 | Merged checkpoint quantized with the export recipe, computed in float32 | float32 | `probe_merged` |
| 3 | Core AI asset of the float32 gated delta update variant, run from Swift (`TypedDecisionModel`) | asset | `typed_decide_f` (F), `typed_decide_q`, `typed_decide_cal`, `typed_decide_h` |

Stages 2 and 0q are built in their own processes through the exporter's own loading and quantization path. Every tensor
of the quantized model (quantized data, scales, offsets and unquantized weights) is hashed before any float32 cast.
Stage 2 must reproduce the hashes recorded during the export; stage 0q has no exported counterpart and keeps its own
hash record. The quantized models then run in float32 with float32 zero states; the key and value caches hold
max(T, 2048) positions for an input of T tokens.

`verify_quantization` writes `reference_q.json`: one row per input of F (stages 0, 1, 1', 2 and 0q) and Q (stages 2 and
0q) with `set` (`F` or `Q`), `id`, `kind`, `options`, `band`, `row_sha256`, `text_sha256`, `token_count`, `tokens`,
`rejected`, and for each stage `logits<stage>` (24 values before the temperature), `p<stage>` (temperature 1),
`p<stage>_T` (shipping temperature) and `elapsed<stage>_s`, where `<stage>` is `0`, `1`, `1m` (stage 1'), `2` or `0q`; its header gives the device and dtype of each stage (`stages`). Rows longer than
8,192 tokens are rejected without a forward pass. Hidden states
of F are kept in `hidden2.safetensors` and `hidden0q.safetensors` (`hidden` [302, 5120] float32 in the order of F,
`mask` [302] uint8 for rows with a forward pass). `quantization.json` (`typed-decision-quantization/1`) records the
differences of stage 2 from stage 1' and of stage 0q from stage 0. Its `passed` covers hash agreement, complete rows
and finite probabilities only; no numeric threshold is applied at this step.

### Quality and calibration

The recipe measures stage 3 (the Core AI asset of the variant bundle) against the other stages on Q and refits the
temperatures from stage 3 on Cal, with the conditions fixed before any measurement. Every number is agreement with the
teacher distribution of the corpus (`target`, output distributions of Jev 1.13), not calibration against real-world
correctness.

These steps also run before `verify_merge`:

- `reference_q` and `reference_merged_q` run Q through stages 1 and 0 (PEFT, `disable_adapter()`) and through stage 1'
  with `reference_sets.py`, which repeats the model loading and checks of `reference.py` and `reference_merged.py`,
  imports their head, forward and header functions, and leaves both modules unchanged. They write `reference-Q.json` and
  `reference-merged-Q.json` (rows in the form of `reference_q.json` with `set`, without hidden states). Stage 1 is
  computed but not used.
- `typed_input_eval` writes `input-Q.json`, `input-Cal.json` and `input-H.json`, and `typed_decide_q`,
  `typed_decide_cal` and `typed_decide_h` run them through `coreai-verify typed-decide` with the variant bundle.
- `score_q` (`score.py`) writes `quality.json` (`typed-decision-quality/1`) and `report.md`. `recorded` is its check:
  missing rows, rows that differ in ID, order, token IDs or rejection, non-finite values and invalid targets fail the
  step, while a quality condition that is not met is recorded and does not.
- `calibrate` (`calibrate.py`) writes `calibration.json` (`typed-decision-calibration/1`), `report.md`, a calibrated
  copy of the bundle and `comparison-refit.json`. Its checks are `recorded` and the `passed` of `comparison-refit.json`.

Definitions:

- Probabilities of every stage are recomputed from the 24 head logits before the temperature, with the slot range, the
  division by T and the softmax of `typed_head.decide`, at temperature 1 and at the card temperatures. Log-probabilities
  use log-sum-exp in the same order and are not clipped.
- Targets are restored with `load_rows` after the corpus integrity check. A target needs one non-negative value per
  option with a sum within 1e-3 of 1 and is divided by its sum. The teacher top is the first largest target value.
- choice accuracy: the argmax of p equals the teacher top, in points. noul Brier (soft): mean of (P(true) - t(true))^2;
  the hard form uses 1 when the teacher top is `true`. Multiclass Brier: the sum of (p_i - t_i)^2 over the options.
  score MAE / 5: mean of |sum k p(k) - sum k t(k)| / 5. KL(t || p) sums the terms with t_i > 0. noul AUROC ranks
  P(true) with the teacher top as the label, leaves out rows with a 0.5 target, counts tied predictions one half and is
  undefined with one class. ECE uses the confidence max p in 15 equal bins (1.0 falls in the last), against the mean
  target of the predicted option (soft) or the accuracy (hard); empty bins weigh 0.
- Intervals: 2,000 bootstrap replicates with seed 20261102 and the same draws for every stage; within a kind, rows that
  share a state are drawn together. The 95% interval takes the 2.5 and 97.5 percentiles with linear interpolation.

Conditions at temperature 1 on Q:

| Condition | Compared | Passes when |
|---|---|---|
| Quantization, choice | accuracy of stage 3 minus stage 1' | at least -2 points |
| Quantization, noul | Brier (soft) of stage 3 minus stage 1' | at most +0.01 |
| Quantization, score | MAE / 5 of stage 3 minus stage 1' | at most +0.02 |
| Adaptation | gain(bf16) = stage 1' against stage 0, gain(4bit) = stage 3 against stage 0q, for choice accuracy, noul Brier, score MAE / 5 and KL | gain(4bit) - 0.8 gain(bf16) >= 0 for each metric |

The point estimates decide the quantization conditions; the 95% interval of each paired difference is reported with
whether it contains the limit. Accuracy gains are adapted minus base, the other gains base minus adapted. A metric with
gain(bf16) at most 0 is undefined, which is not a pass. Stage 2 against stage 0q is reported beside the gains to
separate the contribution of the adapter at 4 bits from the difference between stage 3 and stage 2.

Temperatures: for each kind, beta = 1 / T in [0.25, 4] minimizes the mean cross-entropy to the target of the stage 3
logits on Cal with a golden-section search until the bracket is narrower than 1e-7. Reaching an end of the range is
recorded; the range is not widened. The candidate is rounded to float32, the value that the Swift runtime uses, and
written as the shortest decimal that rounds back to it. A kind adopts its candidate only when neither the NLL nor the
multiclass Brier on H' is larger than at the card temperature, both at float32; H' is H without the 38 rows whose
state also occurs in Cal. All of H is recorded too, but H selects the temperature and is not a final evaluation. ECE
intervals are conditional on the chosen temperature and leave out the uncertainty of the fit and of the selection.

Results (run `20261103T154310-65e7b5bb` for Q, Cal and H; `calibrate` in run `20261103T175311-463e881b`). The
quantization conditions and every adaptation gain pass:

| Condition (temperature 1, Q) | Stage 1' | Stage 3 | Difference [95% interval] | Limit | Result |
|---|---:|---:|---|---|---|
| choice accuracy (points) | 87.33 | 86.67 | -0.67 [-3.00, +1.67] | >= -2 | pass; the interval contains the limit |
| noul Brier (soft) | 0.00141 | 0.00165 | +0.00024 [+0.00009, +0.00042] | <= +0.01 | pass |
| score MAE / 5 | 0.02120 | 0.02171 | +0.00051 [-0.00029, +0.00129] | <= +0.02 | pass |

| Metric | gain(bf16) | gain(4bit) | gain(4bit) - 0.8 gain(bf16) | Stage 2 over 0q | Result |
|---|---:|---:|---:|---:|---|
| choice accuracy (points) | +25.33 | +26.33 | +6.07 | +26.67 | pass |
| noul Brier (soft) | +0.0648 | +0.0723 | +0.0205 | +0.0723 | pass |
| score MAE / 5 | +0.0937 | +0.0901 | +0.0151 | +0.0901 | pass |
| KL | +0.3378 | +0.3555 | +0.0852 | +0.3555 | pass |

| Stage (temperature 1, Q) | KL | Top-1 | Choice accuracy | noul AUROC | noul Brier (soft / hard) | score MAE |
|---|---:|---:|---:|---:|---|---:|
| 0 | 0.3577 | 0.5844 | 62.00 | 0.8999 | 0.0662 / 0.1934 | 0.5746 |
| 0q | 0.3757 | 0.5900 | 60.33 | 0.8925 | 0.0740 / 0.2008 | 0.5591 |
| 1' | 0.0198 | 0.9089 | 87.33 | 0.9941 | 0.0014 / 0.0843 | 0.1060 |
| 2 | 0.0203 | 0.9044 | 87.00 | 0.9933 | 0.0017 / 0.0859 | 0.1087 |
| 3 | 0.0202 | 0.9033 | 86.67 | 0.9934 | 0.0017 / 0.0857 | 0.1085 |

`quality.json` and `report.md` also hold the same table at the card temperatures, the values per kind and the card's
own figures, which cover all 29,955 rows of `test_set_30k` (placeholder rows included) at the card temperatures and are
therefore shown for reference only. One noul row of Q has a 0.5 target and is left out of the AUROC.

| Kind | Card T | Refitted T (no end reached) | H' NLL, card then refitted | H' multiclass Brier, card then refitted | Adopted |
|---|---:|---:|---|---|---|
| noul | 1.0143135 | 1.0056266 | 0.384497, 0.384488 | 0.003832, 0.003823 | yes |
| choice | 1.0161281 | 1.0053791 | 0.732314, 0.732290 | 0.020188, 0.020145 | yes |
| score | 1.0036213 | 1.010558 | 1.062951, 1.062910 | 0.009226, 0.009215 | yes |

ECE on H' at the card and the adopted temperatures, with 95% intervals:

| Kind | Soft, card T | Soft, adopted T | Hard, card T | Hard, adopted T |
|---|---|---|---|---|
| noul | 0.0041 [0.0030, 0.0090] | 0.0035 [0.0027, 0.0086] | 0.1619 [0.1446, 0.1795] | 0.1609 [0.1437, 0.1785] |
| choice | 0.0122 [0.0093, 0.0224] | 0.0092 [0.0079, 0.0199] | 0.2184 [0.1949, 0.2430] | 0.2163 [0.1932, 0.2420] |
| score | 0.0043 [0.0048, 0.0122] | 0.0042 [0.0045, 0.0120] | 0.3349 [0.3077, 0.3609] | 0.3365 [0.3093, 0.3626] |

ECE sums absolute bin gaps, so resampling biases it upward; for score the soft value lies below its percentile
interval. The hard ECE compares the confidence with the agreement of the top option with the teacher top; for score
the mean confidence on H' (about 0.56) lies far below that agreement (about 0.89) because the teacher distributions
are spread out. The fidelity criteria of F, re-evaluated from the saved logits at the adopted temperatures, still pass
(`comparison-refit.json`; at the adopted temperatures, maximum 0.0065, mean 0.0011, no top-1 change).

At stage 3 the `elapsed_seconds` of the rows (from the call of `decide` to its return) add up to 458.7 s for Q,
1,866.3 s for Cal and 919.6 s for H (about 180 tokens per second over all rows, median 0.48 s per row). Stages 1 and
0 of Q took 679.4 s and 644.7 s and stage 1' took 628.9 s on MPS.

#### Calibrated bundle

`calibrate` writes `jev27b_s1_gdnfp32_qwen3_5_cfg_dynamic/` into its own output as an APFS clone (`cp -c`) of the
variant bundle and then checks that every file of the variant still matches its `SHA256SUMS` and that the copy holds
the same files. The variant itself is not modified.

- `calibration-refit.json` (`typed-decision-calibration-refit/1`) records, for each kind, the card temperature, the fit
  (beta, bracket, iterations, end reached, cross-entropy), the candidate, the adopted value and the reason, the
  statistics of Cal, H' and H at both temperatures, the excluded rows of H, the settings, the SHA-256 of `main.mlirb`
  and of the Cal and H fixtures, and the input digests of the stage 3 runs.
- `decision.json` differs from the variant in `files`, which lists `calibration-refit.json`, and in `temperatures`:
  `per_kind` holds the adopted values (the refitted float32 decimals, or the card value for a kind that keeps it),
  `source` is `refit` when any kind adopts its candidate, `calibration_sha256` is the SHA-256 of
  `calibration-refit.json` and `calibration_id` is `jev27b-refit/<its first 16 hexadecimal digits>`. `SHA256SUMS` is
  written after `decision.json` and lists both files.
- The asset, head and tokenizer are byte-identical to the variant, so the specialization cache entry of the variant
  applies. `TypedDecisionBundle` reads the temperatures as `Float`; with `JEV_BUNDLE_DIR` set to the calibrated bundle,
  the bundle test compares them with the `temperatures` of `calibration-refit.json`.

~~~sh
python3 -m Tools.Conversion.adapters.jev27b.decision <calibrated bundle>/decision.json --check-files
JEV_BUNDLE_DIR=/path/to/calibrated/jev27b_s1_gdnfp32_qwen3_5_cfg_dynamic swift test --package-path CoreAISamplesKit \
  --filter TypedDecision
~~~

### Japanese (JGLUE)

The recipe also runs the Japanese set J (200 rows each of the JNLI, JCommonsenseQA and JSTS validation files of
JGLUE v1.2.0) through stage 1' and stage 3 with two prompt variants, after fixing which variant to adopt on a separate
selection set. No pass line is set. The numbers below are descriptive results on 200 rows per task without intervals;
they are not a claim of Japanese quality or of calibration.

These steps run after `calibrate`, still before `verify_merge`:

- `jdev_fixture` (`japanese.py jdev`) writes `jdev.json`: 100 rows per task of the pinned JGLUE validation files. A
  row is eligible when it is not in J or F, when neither its `sentence1` nor its `sentence2` (JNLI, JSTS), or its
  question (JCommonsenseQA), equals any `sentence1`, `sentence2` or question of a JGLUE row of J or F (compared across
  the three tasks), and when its options render (no row was left out for this). The native IDs (`sentence_pair_id`,
  and `q_id` turned into a string) of the eligible rows of each task are sorted as strings, and one
  `random.Random("20261102:Jdev")` draws `rng.sample(ids, 100)` for JNLI, JCommonsenseQA and JSTS in this order.
  `jdev.json` records the ID and the SHA-256 of each row and the size of every exclusion; it holds no text.
- `typed_input_jdev` and `typed_input_j` (`japanese.py input`) wrap the rows with both variants and write the CLI input
  (`input-<set>.json`, with text, kept in the work directory) and `rows-<set>.json`: for each row the key, the SHA-256
  of the JGLUE row (the fixture value), the SHA-256 of the rendered text after wrapping, the token count and the label,
  without text.
- `typed_decide_jdev` and `typed_decide_j` run `coreai-verify typed-decide` with the calibrated bundle of `calibrate`.
- `prompt_adoption` (`japanese.py adopt`) scores Jdev and writes `prompt-adoption.json` with `frozen: true`. Its check is
  `frozen`. `typed_input_j` refuses an adoption that is not frozen for the same `wrappers.json`, so J runs only after
  the adoption is fixed.
- `reference_j` (`reference_j.py`) builds the same wrapped rows of J (without the permuted rows) and runs them through
  stage 1' with the `merged_stage` of `reference_sets.py`; it passes with the same checks as `reference_merged_q`.
- `score_j` (`japanese.py score`) writes `japanese.json` (`typed-decision-japanese/1`) and `report.md`. Its check is
  `recorded`: missing rows, rows that differ in ID, order, token IDs, hashes or labels, non-finite values and stage 3
  probabilities that differ from the bundle temperatures fail the step.

Both variants put an instruction of the same content around the unchanged JGLUE text and differ only in its language;
the bare form of the A1 rows (the question without an instruction) is not evaluated. The structure tags stay `bare-v1`.
The prefix, suffix and JNLI options of each variant are in
[`wrappers.json`](../Tools/Conversion/recipes/jev27b/wrappers.json):

| Task | Kind | State | Question | Options |
|---|---|---|---|---|
| JNLI | `choice` | `sentence1` | prefix + `sentence2` + suffix | entailment, contradiction, neutral in the language of the variant |
| JNLI noul (derived from the 200 JNLI rows, ID `<id>:noul`) | `noul` | `sentence1` | prefix + `sentence2` | `false`, `true`; true only for entailment |
| JCommonsenseQA | `choice` | empty | prefix + question + suffix | the five choices, not translated |
| JSTS | `score` | `sentence1` | prefix + `sentence2` + suffix | `0` to `5` |

A row key is the JGLUE ID (with `:noul` for the derived rows), the variant and the option order.

Adoption, fixed before any run: on Jdev at temperature 1, each task adopts the variant with more correct JNLI choice
rows, more correct JCommonsenseQA rows, or the smaller MAE of the expected JSTS score; a tie adopts `ja`. JNLI noul
follows the JNLI choice adoption as a common policy, which is not a claim that it is best for noul. The adoption chooses
the reported variant and the variant of the order test; J itself runs with both variants.

Definitions (fixed before any run):

- Probabilities are recomputed from the 24 logits of each stage at temperature 1 and at the adopted temperatures of the
  calibrated bundle. KL and ECE are not computed.
- JNLI choice and JCommonsenseQA accuracy: the first largest probability is the label. JNLI noul accuracy: P(true) > 0.5
  is true and P(true) = 0.5 is false. AUROC ranks P(true) with entailment as the positive class, ties count one half,
  and it is undefined with one class. The hard Brier score is the mean of (P(true) - y)^2 with y = 1 for entailment.
- JSTS: E[p] = sum k p(k) against the continuous label, as MAE, Pearson and Spearman (average ranks for ties); either is
  undefined when one side is constant.
- Option order (stage 3, adopted JCommonsenseQA variant): the options in four fixed orders (as given, reversed, rotated
  left by one and by two); the label moves with its option, and each answer is mapped back to the original option. The
  accuracy of each order and the number of rows whose four answers agree are recorded.
- Stage 3 minus stage 1' on the same rows, wrapper and temperature is the difference of quantization and execution path
  together.

Results (run `20261103T184500-52091525`: `prompt_adoption` for Jdev, `score_j` for J). The adoption, frozen before J
ran:

| Task (Jdev, temperature 1) | Metric | `ja` | `en` | Adopted |
|---|---|---:|---:|---|
| JNLI choice | accuracy | 0.8300 (83/100) | 0.8900 (89/100) | `en` |
| JCommonsenseQA | accuracy | 0.9600 (96/100) | 0.9500 (95/100) | `ja` |
| JSTS | MAE of E[p] | 0.5806 | 0.5577 | `en` |
| JNLI noul | follows JNLI choice | - | - | `en` |

J at temperature 1 (200 rows per task; the last column is the adopted variant):

| Task | Metric | Stage 1' `ja` | Stage 1' `en` | Stage 3 `ja` | Stage 3 `en` | 3 - 1' `ja` | 3 - 1' `en` | Adopted |
|---|---|---:|---:|---:|---:|---:|---:|---|
| JNLI choice | accuracy | 0.8200 | 0.8700 | 0.8050 | 0.8550 | -0.0150 | -0.0150 | `en` |
| JNLI noul | accuracy | 0.9350 | 0.9200 | 0.9250 | 0.9200 | -0.0100 | +0.0000 | `en` |
| JNLI noul | AUROC | 0.9856 | 0.9801 | 0.9842 | 0.9777 | -0.0014 | -0.0024 | `en` |
| JNLI noul | hard Brier | 0.0499 | 0.0531 | 0.0541 | 0.0588 | +0.0043 | +0.0057 | `en` |
| JCommonsenseQA | accuracy | 0.9600 | 0.9700 | 0.9600 | 0.9600 | +0.0000 | -0.0100 | `ja` |
| JSTS | MAE of E[p] | 0.5819 | 0.5412 | 0.5778 | 0.5287 | -0.0041 | -0.0124 | `en` |
| JSTS | Pearson | 0.8501 | 0.8555 | 0.8474 | 0.8591 | -0.0027 | +0.0035 | `en` |
| JSTS | Spearman | 0.7818 | 0.7858 | 0.7796 | 0.7924 | -0.0022 | +0.0066 | `en` |

- At the adopted temperatures the accuracies and AUROC values are the same, and the other values move by at most
  0.0007 (`japanese.json` and `report.md` hold both tables). 35 of the 200 JNLI rows are entailment.
- JCommonsenseQA option order (stage 3, `ja`): accuracy 0.9600 as given, 0.9700 reversed, 0.9600 rotated by one and
  by two; the four answers agree on 191 of 200 rows.
- Stage 3 against stage 1' row by row at temperature 1: the top option is the same on 179 to 200 of the 200 rows of
  each task and variant (JSTS is lowest), the largest probability difference is 0.17 and the mean 0.011 to 0.033.
- At stage 3 the `elapsed_seconds` of the rows (from the call of `decide` to its return) add up to 278.2 s for Jdev
  (600 rows) and 953.0 s for J (2,200 rows), with medians of 0.464 and 0.436 s per row; the forward runs of stage 1'
  took 975.8 s for the 1,600 rows of J on MPS.

### Performance

The performance of the calibrated bundle is measured outside the recipe with the steps of
[docs/benchmarking.ja.md](benchmarking.ja.md): a frozen plan, run records in JSONL and `python3 -m Tools.ModelReports
summarize`. No pass line is set. It was measured three times with the same rows, boundary and counts: on AC power
with a 140 W adapter at 94 W under `plan-d2b-ac94` (#490), whose values the model record uses, earlier on AC power with
a 70 W adapter under `plan-d2b-ac` (#488), and first on battery power under plan revision 3 (#480).

- The plan comes from `python3 -m Tools.ModelReports template --kind decision --model jev27b`, with every field filled
  and its SHA-256 recorded before the first measurement (`plan_sha256` of every run record). It fixes three measured
  runs per condition and, for `latency_s`, one warmup of each length; with fewer than 20 runs `summarize` reports no
  p95.
- The rows P (`performance.py rows`) are four rows of exactly 100, 500, 2,000 and 8,192 tokens (`noul`, `choice`,
  `score`, `noul`), each built like the boundary rows of F (`Builder.exact`): a base row of the kind, states of further
  rows of `test_set_30k` joined after it and the last one cut so that the rendered input has exactly that many tokens.
  The rows are drawn with seed `20261102:P` and share no corpus row with F or Q or with each other. `p.json` records
  the IDs of the rows and their parts, the cut, the SHA-256 and the token count; the text stays in the local work
  directory.
- `coreai-verify typed-decide` has three options for these measurements. `--clear-cache` deletes the specialization
  cache entry of the bundle's asset before loading, so the load specializes again (`specialization: fresh`).
  `--sample-memory` samples the RSS and the physical footprint of the process every 100 ms while each row runs, and once
  more after the row, and writes `resident_bytes_max` (`rss_bytes` and `footprint_bytes`, the largest values, with
  `samples`, `failed_samples` and `interval_seconds`) to the row. `--wait-nominal-thermal S` (from 18890f5, used in the
  94 W measurement with 600) waits before every row, warmups included, in steps of 1 s and for at most S seconds, until
  `ProcessInfo.processInfo.thermalState` is nominal, and writes the row's `started` (UTC, whole seconds),
  `thermal_wait_seconds` and `thermal_state_at_start`. The recipe keeps running the earlier build of the executable, so
  its cached steps stay valid.
- `latency_s` is the `elapsed_seconds` of a row: from the call of `TypedDecisionModel.decide` to its return
  (rendering, tokenization, the forward run and the head, without the JSON input and output). One process runs one
  warmup of each length and then three rounds with the lengths interleaved.
- `first_load_s` and `warm_load_s` are the sum of `bundle.load_seconds` (bundle check, tokenizer, head and backbone)
  of three new processes with `--clear-cache --limit 1` and three without it. The API does not separate loading from
  specialization. File caches of the system are not flushed.
- `peak_rss_bytes` and `peak_footprint_bytes` come from `/usr/bin/time -l` around the latency process (its whole
  lifetime, including the load), and `resident_bytes` is the largest RSS sampled during the measured 8,192-token rows.
  The values are not added together.
- `cancel_s` is measured by `TypedDecisionInferenceTests` with `COREAI_TYPED_DECISION_BUNDLE` and `DECIDE_METRICS_DIR`.
  The request is the first fixture row (the refund card example) with its state repeated 164 times, 2,000 tokens; it
  is not a row of P. It runs through a test backbone that starts the Core AI forward in a task of its own, which the
  cancellation does not reach. After the backbone reports the start of its forward call, the request is cancelled;
  this is done three times, and the time from the cancellation request to the `CancellationError` is written to
  `typed-cancel-run.json`. With this backbone the time is, by construction, the rest of the run; whether
  `InferenceFunction.run` itself stops on a cancellation is not measured here.
- `performance.py records` converts the outputs into run records per condition, which `summarize` aggregates. It
  marks a run as failed when its output comes from another bundle or input, its token IDs differ from P, a load did
  not specialize as expected, a memory sample failed, or a cancellation took less than half of the uncancelled run,
  and it refuses outputs without the planned number of runs.

Results on AC power with a 140 W adapter at 94 W (2026-11-04, `plan-d2b-ac94`, SHA-256
`47a12e99d22dbd40e2a41cedc2b91665bc587e44bbdeb6157d97c24d299c4cb2`; the calibrated bundle, `decision.json` SHA-256
`a2b186a87a64d9e93a56f185a5ebc2a0613d00da7440eeb8c8c2229b64f6adf3`). These are the values of the model record. The last
two columns compare the medians with the earlier measurements below:

| Metric | Value | Runs | Against 70 W | Against battery |
|---|---:|---|---:|---:|
| `latency_s`, 100 tokens (`noul`) | 0.6147 s | median of 3 (0.5979 to 0.6752) | -5.0% | +2.7% |
| `latency_s`, 500 tokens (`choice`) | 2.2076 s | median of 3 (1.9583 to 2.2709) | -7.6% | -2.7% |
| `latency_s`, 2,000 tokens (`score`) | 8.9783 s | median of 3 (7.9626 to 9.0310) | -4.2% | +0.7% |
| `latency_s`, 8,192 tokens (`noul`) | 37.1391 s | median of 3 (35.9482 to 37.2891) | -12.3% | -0.9% |
| `first_load_s` (`specialization: fresh` in all three) | 28.2244 s | median of 3 (28.0284 to 28.5118) | -1.3% | +1.9% |
| `warm_load_s` (`specialization: cached` in all three) | 9.5050 s | median of 3 (9.4792 to 9.5061) | -8.5% | +0.9% |
| `peak_rss_bytes` (latency process) | 32,333,840,384 B | one process | 0.0% | 0.0% |
| `peak_footprint_bytes` (latency process) | 21,336,815,344 B | one process | +2.4% | -1.1% |
| `resident_bytes` (largest RSS during the 8,192-token rows) | 17,298,096,128 B | one process, 3 rows | -0.2% | 0.0% |
| `cancel_s` (2,000 tokens) | 8.8401 s | median of 3 (8.7246 to 9.0025) | -0.5% | -3.4% |

- **Power:** AC through the 140W USB-C Power Adapter at 94 W (`pmset -g ac`) at every stage, the battery at 100%,
  charged, between 11:21:07 and 11:30:03 JST. The energy mode on AC was High Power (`powermode 2` in `pmset -g`), the
  owner's setting, which was not changed. The plan `plan-d2b-ac94` recorded this before the measurement; it differs from
  `plan-d2b-ac` only in the power, the thermal condition, the other load, the code revision (18890f5), the binary and
  the per-row thermal wait.
- **Gate:** every stage started only at 94 W, with at least 120 s without user input, with no audio, video wake lock or
  automation assertion and at thermal pressure level 0, waiting up to 10 minutes for each; only the cancellation test
  had to wait (30 s, for 120 s without input).
- **Thermal pressure:** `coreai-verify typed-decide --wait-nominal-thermal 600` started every latency row, the warmups
  included, at a nominal `ProcessInfo` thermal state; only the 100-token row of the third round waited (54.51 s). During
  the latency process the level of `thermalmonitord` went to moderate at 11:25:51 and heavy from 11:25:53 to 11:26:52:
  the 8,192-token row of the second round turned moderate and heavy about 20 s into it (37.2891 s, the slowest of its
  length), and the next row waited until the state was nominal again. The median rows of all four lengths, the loads and
  the cancellation test ran at nominal pressure. `pmset -g therm` recorded no warning.
- **Other load:** before the measurement no other model process was running (`pgrep`). The assertions of `pmset -g log`
  show those of the measurement, a sleep-prevention `caffeinate` that had started before the measurement and ended at
  11:22:19, and network tasks of `cloudd` between the latency process and the cancellation test, without audio, video or
  automation activity. There was user input during the latency process, the last at about 11:26:13 while it waited for a
  nominal state, so no row ran then; earlier inputs in that process cannot be excluded. The CPU and GPU use of other
  processes was not recorded.
- **Against the earlier measurements:** the latency and the loads were within 3% of the battery values (within 3.3% in
  the first round) and 1 to 12% faster than with the 70 W adapter. The rows of the third round, which followed the 54.51
  s wait, were the fastest of their lengths except at 100 tokens (1.9583 s at 500, 7.9626 s at 2,000 and 35.9482 s at
  8,192), close to the warmups (1.8967, 7.9032 and 36.5796 s). The cause was not investigated.
- The physical footprint sampled during the rows again grew in each round of four rows, by 4.1 to 4.5 GB: about 8.2 GB
  during the 8,192-token warmup, then 12.5, 16.6 and 21.2 GB during the three measured 8,192-token rows, without
  levelling off; `peak_footprint_bytes` is not an upper bound. The RSS sampled during the rows stayed between 16.52 and
  17.30 GB; the lifetime peak of 32.3 GB lies outside the sampled rows.
- In the cancellation test the backbone reported the start of its forward call before every cancellation; each request
  ended with `CancellationError` 8.72 to 9.00 s after the cancellation request, longer than the 8.20 s of the same
  request without cancellation, for a reason not investigated. The cancellation at the forward checkpoint ended a
  request before the run (0.17 s). All five GPU tests of `TypedDecisionInferenceTests` passed.

#### Earlier measurement on AC power with a 70 W adapter

Results on AC power with a 70 W adapter (2026-11-04, `plan-d2b-ac`, SHA-256
`8763b18a0830395744f3f65e2bd27d5a151d85dc201b1296ea15e175af9128cb`; the same bundle). The last column compares the
medians with the measurement on battery power below:

| Metric | Value | Runs | Against battery |
|---|---:|---|---:|
| `latency_s`, 100 tokens (`noul`) | 0.6471 s | median of 3 (0.6209 to 0.6602) | +8.1% |
| `latency_s`, 500 tokens (`choice`) | 2.3891 s | median of 3 (2.3703 to 2.4801) | +5.2% |
| `latency_s`, 2,000 tokens (`score`) | 9.3733 s | median of 3 (9.3328 to 10.3874) | +5.1% |
| `latency_s`, 8,192 tokens (`noul`) | 42.3680 s | median of 3 (41.6706 to 53.1283) | +13.0% |
| `first_load_s` (`specialization: fresh` in all three) | 28.6010 s | median of 3 (28.1471 to 30.7716) | +3.3% |
| `warm_load_s` (`specialization: cached` in all three) | 10.3842 s | median of 3 (10.1384 to 10.4565) | +10.2% |
| `peak_rss_bytes` (latency process) | 32,343,474,176 B | one process | 0.0% |
| `peak_footprint_bytes` (latency process) | 20,837,938,928 B | one process | -3.4% |
| `resident_bytes` (largest RSS during the 8,192-token rows) | 17,333,698,560 B | one process, 3 rows | +0.2% |
| `cancel_s` (2,000 tokens) | 8.8854 s | median of 3 (8.5951 to 9.0007) | -2.9% |

- **Power:** AC through the 70W USB-C Power Adapter at 68 W (`pmset -g ac`) at every stage, the battery charging from
  64% to 67% between 09:22:08 and 09:31:10 JST. The energy mode on AC was High Power (the `powermode` line of `pmset
  -g` was 2), the owner's setting, which was not changed. The plan `plan-d2b-ac` recorded this before the measurement;
  it differs from plan revision 3 only in the power, the thermal condition, the code revision (9b610f8) and the
  binary, a release build of that commit that is byte-identical to the build used on battery.
- **Thermal pressure:** each stage started only when the pressure level of `thermalmonitord` was 0 (nominal), waiting up
  to 10 minutes; only the cancellation test had to wait (50 s). During the latency process the level went to moderate at
  09:26:46 and heavy from 09:26:48 to 09:29:05 (moderate until 09:29:19): the 8,192-token row of the second round and
  the whole third round ran at heavy pressure; the loads, the first round and the cancellation test at nominal. The
  2,000-token row of the second round, a median, ended within about a second after the change to moderate; the rows
  carry no clock times, so their times are rebuilt from the logs within about a second. Of the medians, only the
  8,192-token one ran at heavy pressure; the first round's 8,192-token row took 41.6706 s. The third round's rows were
  the slowest of each length (53.1283 s at 8,192 tokens). `pmset -g therm` recorded no warning.
- **Against battery:** the latency and the loads on AC were 3 to 13% slower than on battery, also where both ran at
  nominal pressure: in the first round, by 4.3% at 100 tokens, 5.2% at 500, 4.6% at 2,000 and 11.9% at 8,192. The bundle
  check (5.29 to 5.93 s, against about 5.25 s) and the backbone of the loads (22.2 to 24.3 s fresh and 3.8 s cached,
  against 21.8 to 22.1 s and 3.5 s) were slower too. The cancellation test was 2.9% faster; on battery it ran at heavy
  pressure. The cause was not investigated. Besides the power source, this run differed in the energy mode, the battery
  charging from the 68 W of the adapter, and the other activity on the machine below (audio sessions, a browser video
  and a UI automation session), which the battery run did not have.
- Other load: before the measurement no other model process was running (checked with `pgrep`). The assertions of `pmset
  -g log` show other activity during this run: audio sessions from about 09:22 to 09:28:47 and from about 09:29:36 to
  09:30:52, a video wake lock of a browser from 09:25:56 to 09:25:58, a UI automation session at 09:23:31 (during the
  third fresh load, the slowest of the three), Handoff and network tasks. The battery run had none of these except one
  short network task. The CPU and GPU use of other processes was not recorded.
- The physical footprint sampled during the rows again grew in each round of four rows, by 4.1 to 4.3 GB: about 8.2 GB
  during the 8,192-token warmup, then 12.6, 16.6 and 20.8 GB during the three measured 8,192-token rows, without
  levelling off; `peak_footprint_bytes` is not an upper bound. The RSS sampled during the rows stayed between 16.53 and
  17.33 GB. The lifetime peak of 32.3 GB lies outside the sampled rows; the load, where it presumably occurred, was not
  sampled.
- In the cancellation test the backbone reported the start of its forward call before every cancellation; each request
  ended with `CancellationError` 8.60 to 9.00 s after the cancellation request, longer than the 7.98 s of the same
  request without cancellation, for a reason not investigated. The cancellation at the forward checkpoint ended a
  request before the run (0.17 s). All five GPU tests of `TypedDecisionInferenceTests` passed.

#### Earlier measurement on battery power

Results on battery power (2026-11-04, plan revision 3, SHA-256
`d361e027c9f6c4929bc8cafe7ea4f378fcdbfe7e01e795a1229e6cbc5cd5d4b7`; the calibrated bundle, `decision.json` SHA-256
`a2b186a87a64d9e93a56f185a5ebc2a0613d00da7440eeb8c8c2229b64f6adf3`):

| Metric | Value | Runs |
|---|---:|---|
| `latency_s`, 100 tokens (`noul`) | 0.5984 s | median of 3 (0.5951 to 0.6055) |
| `latency_s`, 500 tokens (`choice`) | 2.2700 s | median of 3 (2.2661 to 2.2769) |
| `latency_s`, 2,000 tokens (`score`) | 8.9191 s | median of 3 (8.9136 to 9.0762) |
| `latency_s`, 8,192 tokens (`noul`) | 37.4950 s | median of 3 (37.2306 to 37.6194) |
| `first_load_s` (`specialization: fresh` in all three) | 27.6992 s | median of 3 (27.6406 to 27.9684) |
| `warm_load_s` (`specialization: cached` in all three) | 9.4218 s | median of 3 (9.3989 to 9.4533) |
| `peak_rss_bytes` (latency process) | 32,343,687,168 B | one process |
| `peak_footprint_bytes` (latency process) | 21,565,667,056 B | one process |
| `resident_bytes` (largest RSS during the 8,192-token rows) | 17,290,887,168 B | one process, 3 rows |
| `cancel_s` (2,000 tokens) | 9.1467 s | median of 3 (9.0557 to 9.2045) |
| `bundle_bytes` (13 files) | 16,064,815,496 B | one count |

- **Power:** measured on battery by the owner's decision (2026-11-04), without an AC adapter, because the adapter was
  disconnected before the measurement. The plan was revised to record this (revision 3) before any measurement; it
  replaced revision 2, which named AC power. The battery went from 90% to 76% between 06:23:59 and 06:31:34 JST, no
  adapter was attached during the measurement, and Low Power Mode was off (the `powermode` line of `pmset -g` was 0).
  The two measurements on AC power above differ from this one in more than the power source.
- **Thermal pressure:** `pmset -g therm` recorded no thermal or performance warning in any of the ten snapshots, but
  it does not show the thermal pressure. The pressure level of `thermalmonitord` in the system log went from nominal
  to moderate at 06:28:52 and to heavy from 06:29:08 to 06:32:08 (moderate until 06:32:23). The loads and the first two
  rounds ran at nominal pressure, except the last seconds of the second 8,192-token row (moderate). The third round
  ran at moderate pressure until the last seconds of its 2,000-token row and at heavy pressure for its 8,192-token row,
  and the whole cancellation test ran at heavy pressure. The third 2,000-token row
  (9.0762 s) is about 1.8% slower than the first two. The medians include rows of the third round (100 tokens) and of
  the end of the second round (8,192 tokens).
- Other load: before the measurement no other model process was running (checked with `pgrep`); the GPU use of other
  processes was not recorded.
- In the loads, the bundle check took about 5.25 s and the tokenizer about 0.6 s every time; the backbone took 21.8 to
  22.1 s with a fresh specialization and 3.5 s from the cache. File caches of the system were not flushed between the
  processes.
- The first row of each process took about 2.1 s for 100 tokens; the warmup rows were not counted. In the latency
  process the warmups of 500 and 2,000 tokens (1.9 s and 7.7 s) were faster than their measured rounds, which followed
  an 8,192-token row.
- The physical footprint sampled during the rows grew in each round of the interleaved lengths, by 4.2 to 4.8 GB per
  round of four rows: about 8.0 GB during the 8,192-token warmup, then 12.6, 16.8 and 21.6 GB during the three
  measured 8,192-token rows, without levelling off. `peak_footprint_bytes` is therefore the value after these sixteen
  interleaved requests, not an upper bound: it grows with the number of requests whose length changes. The cause was
  not investigated; it has the same form as the work memory that grows in one model when the length of consecutive
  calls changes several times over (an assumption, not measured here).
- The RSS sampled during the rows stayed between 16.50 and 17.29 GB (it includes the mapped weights). The lifetime peak
  of 32.3 GB, about twice the 16.0 GB asset, occurred outside the rows, during the load, which was not sampled. The
  process lifetime peaks of `/usr/bin/time -l` are reported as they are; RSS and footprint are not added together.
- In the cancellation test the backbone reported the start of its forward call before every cancellation; each request
  ended with `CancellationError` 9.06 to 9.20 s after the cancellation request, longer than the 8.72 s of the same
  request without cancellation, for a reason not investigated; by the construction of the test backbone a request cannot
  end before the run returns. In the same test, the cancellation of a request at the forward checkpoint of
  `TypedDecisionModel` ended it before the run (0.16 s). The cancellation test ran in the debug build of the test
  target; `swift test -c release` fails to build another test target of the package.
- `bundle_bytes` is the size of the 13 regular files of the calibrated bundle, counted by `performance.py records`
  after plan revision 3.

## Terms

Qwen3.8-27B (Qwen) and JEV-27B (AutoTrust AI) are Apache-2.0. JEV-27B was trained on
`SargeDev/jev-distill-corpus-v3` (Apache-2.0), most of whose labels are output distributions of TypeSafe Jev 1.13. The
fixed sets are built from that corpus and from JGLUE (CC BY-SA 4.0) and are not redistributed. The Swift test
fixtures contain the text of eight JGLUE rows under CC BY-SA 4.0; their notice is in `render.json`.
