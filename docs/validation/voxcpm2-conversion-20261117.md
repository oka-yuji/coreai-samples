# VoxCPM2 conversion validation

2026-11-17. Issues [#575](https://github.com/oka-yuji/coreai-samples/issues/575)
(component conversion), [#576](https://github.com/oka-yuji/coreai-samples/issues/576)
(generation from text) and [#577](https://github.com/oka-yuji/coreai-samples/issues/577)
(app bundles). The [numeric record](voxcpm2-conversion-20261117.json) keeps the pinned
source, the criteria, the asset and bundle identities, every component error, the
per-sentence generation results, the precision comparison and the memory samples. The
procedure and the rules are in the [recipe](../../Tools/Conversion/recipes/voxcpm2/README.md).

The six FP32 Core AI assets met the component criteria against the original PyTorch
implementation. The FP32 generation from text of five sentences met the end-to-end rule,
with three of 239 patches replaced. Debug information was removed from all twelve FP32
and FP16 assets, and the programs without location information did not change. The
three app bundles have fixed
inventories and contain no absolute path of the working environment, and the FP32
bundle repeated the component comparisons with the same results. No app reads the
bundles yet. Listening, the Swift runtime, the Mac app and iPhone are not verified.

The audio comes from the same weights run with Core AI. Numerical differences near
rounding boundaries inside the model can make the generated result differ from the
reference implementation, even with the same seed.

## Source and environment

- Model `openbmb/VoxCPM2` at revision `7936080b41163874f38610f46f438e2c0757f9fd`. All
  nine snapshot files are pinned by size and SHA-256. Official code `OpenBMB/VoxCPM` at
  `300e4260b509066151d623dc9c0aa922cf82232b`. Weights and code are Apache-2.0.
- Apple M5 Max (Mac17,6), 137,438,953,472 B RAM, macOS 27.2.1 (26B131), Xcode 27.2
  (27B52).
- Python 3.11.9 with the recipe's lock: torch 2.9.0, transformers 4.57.3, tokenizers
  0.22.2, numpy 2.3.5, coreai-core 1.0.0b3, coreai-torch 0.4.3, coreai-opt 0.3.0.
- The original ran in PyTorch on the CPU in FP32 with four threads. Core AI ran from the
  Python host with GPU-preferred specialization. Which compute unit ran each operation
  was not profiled.
- Private runs under `.local/conversion/work/voxcpm2/runs`:
  - `20261116T161959-fc494398`: the eight export stages;
  - `20261116T181059-4ca849d6`: the original generations and the component comparisons;
  - `20261116T195122-3ad68b2e`: the generation check and the precision comparison;
  - `20261116T201921-cc114781`: the first `strip` implementation, which saved the
    programs without removing debug information (not used);
  - `20261116T202807-b4d07f2b` and `20261116T204724-264d1994`: debug-information
    removal, bundles and re-checks that were replaced after source changes (not used;
    the memory measurements below used their bundles);
  - `20261116T212827-22314097`: the current debug-information removal, bundles and re-check;
  - `20261116T214150-e16889ef`: all 18 stages verified and reused, without new measurements.

## Component comparisons

The criterion: every compared output is finite, and its largest absolute error divided
by the largest absolute reference value is at most 0.001. The stop decision must match
at every patch. The inputs and outputs come from five generations of the original with
seed 1708: `plain`, `design` and `long`, and `heldout_ja` and `heldout_en`, which were
added after the `proj.fuse` rule was fixed. They cover 14 to 104 language-model steps
per sentence. FP16 errors are recorded without a pass or fail.

| Function | FP32 comparisons within | FP32 largest error | FP16 largest error |
|---|---:|---:|---:|
| `tslm.embed` | 5/5 | 0 | 8.6e-08 |
| `tslm.prefill` (hidden, caches) | 15/15 | 2.2e-06 | 3.8e-03 |
| `tslm.step` | 234/234 | 1.1e-06 | 3.2e-03 |
| `ralm.prefill` (hidden, caches) | 15/15 | 1.1e-06 | 2.6e-03 |
| `ralm.step` | 234/234 | 1.6e-06 | 6.0e-03 |
| `proj.heads` (`dit_hidden`, `stop_logits`) | 478/478 | 3.4e-07 | 6.1e-04 |
| `proj.fuse` | 234/234 calls (below) | — | 3.8e-03 |
| `proj.prefill_fuse` | 5/5 | 1.1e-07 | 4.5e-04 |
| `locenc.main` | 239/239 | 3.1e-06 | 1.7e-03 |
| `dit.main` | 2,151/2,151 | 4.6e-06 | 7.9e-03 |
| `vae.main`, joined waveform | 5/5 | 4.2e-06 | 3.6e-03 |

All 239 stop decisions match. Noise in the padded prefill positions leaves the outputs
and caches bitwise equal, and cache rows from `length` on are zero. 229 of 239 VAE
patches meet the criterion on their own; the largest per-patch error, 1.29e-3, is in
near-silent audio and is recorded only. With the padding and count checks, the FP32
comparisons that decide the result number 3,934, and all of them pass.

`proj.fuse` contains the scalar quantizer `round(tanh(x) * 9)`. A call is a match when
both outputs meet the criterion. It is boundary-consistent when the reference has a
dimension within 1e-5 of a rounding boundary and both Core AI outputs meet the
criterion against the original outputs for exactly one neighboring code of one such
dimension. At most 2% of the calls may be boundary-consistent and none may fail.

| Precision | Calls | Match | Boundary-consistent | Fail |
|---|---:|---:|---:|---:|
| FP32 | 234 | 233 | 1 | 0 |
| FP16 (recorded only) | 234 | 188 | 2 | 44 |

The FP32 boundary-consistent call is step 8 of `long`, dimension 283: the original value
is 4.77e-7 from a boundary, the original code is 3 and the candidate code is 2, and the
Core AI outputs differ from the candidate's original outputs by 5.8e-8 and 1.0e-7. The
distance 1e-5 was chosen after this disagreement had been seen. It is a provisional
acceptance policy, not a derived bound, and it does not show that the Core AI value
before rounding was within that distance.

Function times in milliseconds, two rounds, each the median of 20 calls after 3 warm-up
calls on the Python host. The one-minute load average was 3.9 to 5.1.

| Function | FP32 | FP16 |
|---|---:|---:|
| `tslm.embed` | 0.60 / 0.60 | 0.54 / 0.53 |
| `tslm.prefill` | 39.85 / 39.72 | 14.31 / 13.35 |
| `tslm.step` | 13.37 / 13.35 | 8.57 / 8.22 |
| `ralm.prefill` | 11.66 / 11.51 | 4.25 / 4.25 |
| `ralm.step` | 4.23 / 4.23 | 2.85 / 2.81 |
| `proj.heads` | 0.66 / 0.66 | 0.66 / 0.68 |
| `proj.fuse` | 0.54 / 0.55 | 0.51 / 0.51 |
| `proj.prefill_fuse` | 0.84 / 0.86 | 0.70 / 0.71 |
| `locenc.main` | 3.13 / 3.16 | 2.13 / 2.25 |
| `dit.main` | 4.88 / 4.92 | 3.08 / 3.12 |
| `vae.main` | 4.00 / 4.01 | 3.13 / 3.13 |

The contract uses a cache capacity of 1024. Language models exported with capacity 2048
pass the same comparisons and are slower (`tslm.step` 14.13 / 14.48 ms against 13.30 /
13.64 ms in the same stage).

## Generation from text

The Python host follows the zero-shot steps of the asset contract. For all five
sentences, the token IDs and the noise of every patch equal the original bit for bit.

The judged generation replaces values at two kinds of points, and every point is judged
on its value before any replacement:

1. **`proj.fuse` outputs**, at every step. Outputs within the criterion continue
   unchanged. A boundary-consistent call continues with the original outputs of that
   call. Anything else fails the sentence.
2. **Patch latents**, before LocEnc, the VAE and the next `cond` use them. A latent
   within the criterion continues unchanged. A latent outside it continues as the
   original latent only when its solver inputs match the original (the noise bit for
   bit, `dit_hidden` and `cond` within the criterion) and the original solver, given
   the inputs that Core AI actually used, gives a result within the criterion of the
   Core AI latent. Anything else fails the sentence.

At most one patch per sentence and 2% of the original patches (4 of 239) may be
replaced. These limits were set after the observations below and are operating limits,
not a proof of correctness. The stage also generates every sentence without any
replacement and records the result without a pass or fail.

| Sentence | Text tokens | Patches, Core AI / original | Stop patch | Result | Joined audio error | Without replacement: first patch outside, largest error, stop patch |
|---|---:|---:|---:|---|---:|---|
| `plain` | 19 | 15 / 15 | 14 / 14 | Latent replaced with the original value (patch 9) | 9.1e-06 | 9, 6.4e-3, 14 |
| `design` | 28 | 34 / 34 | 33 / 33 | Within the criterion without replacement | 7.6e-05 | none, 4.0e-5, 33 |
| `long` | 97 | 105 / 105 | 104 / 104 | `proj.fuse` outputs replaced with the original values (step 8, patch 9) | 4.1e-04 | 9, 2.19, 104 |
| `heldout_ja` | 51 | 52 / 52 | 51 / 51 | `proj.fuse` outputs replaced with the original values (step 29, patch 30) | 8.1e-04 | 30, 0.042, 51 |
| `heldout_en` | 25 | 33 / 33 | 32 / 32 | Within the criterion without replacement | 5.6e-05 | none, 9.0e-6, 32 |

- `long`, step 8: dimension 283, 4.77e-7 from a boundary, code 3 to 2; the remaining
  errors were 5.8e-8 and 1.1e-6. `heldout_ja`, step 29: dimension 328, 1.61e-6 from a
  boundary, code 1 to 0; the remaining errors were 9.8e-8 and 1.2e-6.
- `plain`, patch 9: before replacement the latent was 3.87e-3 from the original. Its
  inputs matched the original (noise bit for bit, `dit_hidden` 1.35e-6, `cond` 7.3e-6).
  With those inputs the original solver gave a latent 2.6e-5 from the Core AI latent and
  3.84e-3 from the original latent. From the original inputs, the same solver process
  reproduced the original latent bit for bit.
- In the judged generations, the latents that were not replaced stayed within the
  criterion (largest 4.3e-4, in `long`).
- Without replacement, `long` and `heldout_ja` leave the original trajectory from the
  patch that uses the `proj.fuse` call above, and `plain` from patch 9. All five
  sentences still stop at the original patch.
- `plain` limited to 8 patches ended with `limit`, was not a success and wrote no audio.

Two experiments with the original solver examined `plain` patch 9. Their decision rules
were fixed before they ran, and their records stay outside Git.

1. Independent random perturbations of 4e-6 of every estimator output: 32 trials
   changed the latent by at most 3.2e-4 (median 1.3e-4), about 1/12 of the observed
   3.87e-3. The fixed verdict was "undetermined".
2. Given the inputs that the Core AI generation without replacement actually used, the
   original solver agreed with the Core AI latent within 2.6e-5 at every one of the 239
   patches, and at `plain` patch 9 differed from the original latent by 3.84e-3.

## Precision comparison

`plain`, `long` and `design`, a Japanese sentence with a date, times, a number and a
price, and an English sentence were generated with seeds 1708, 1709 and 1710 in three
configurations. The same sentence and seed use the same noise in every configuration.
Each worker generated `plain` once before the timed generation. The load average was
4.1 to 6.1.

| Configuration | Stopped | Stop patch equal to FP32 | Latent error against FP32, median of medians / largest | ms per patch, median | Seconds per second of audio, median |
|---|---:|---:|---:|---:|---:|
| FP32 | 15/15 | — | — | 72.9 | 0.456 |
| FP16 | 15/15 | 10/15 | 0.118 / 2.14 | 47.1 | 0.294 |
| FP16 with `proj` in FP32 | 15/15 | 10/15 | 0.055 / 2.67 | 47.1 | 0.294 |

No sample was clipped when rounding to 16 bits. Both FP16 configurations leave the FP32
latent criterion from the first patches (patch 1 in the median generation), so they
produce audio on a different trajectory from FP32. Their stop patch equaled FP32 in 10
of 15 generations. Their audio quality has not been evaluated; no listening comparison
has been done.

## Debug information and bundles

`strip` (its own module, `strip.py`) loads each exported asset with `AIModelAsset.load`,
applies `strip_debug_info` to the program and saves that same program with `save_asset`
and the original metadata. The program is printed as generic MLIR without locations and
with every constant and resource, and the SHA-256 of that text is compared after
loading, after the removal and after loading the saved asset in a new process. This
compares the programs without location information; `main.hash` and the creation date
in `metadata.json` change.

- All twelve digests are equal. `main.mlirb` became 177 to 18,277 bytes smaller (63,625
  bytes in total), `main.hash` changed, and only `creationDate` changed in
  `metadata.json`.
- Before the stage, a probe loaded `proj` FP32 (three functions), `vae` FP32 (26 states)
  and `ralm` FP16 (two functions that share two states) before and after the removal.
  Function descriptors were equal, and outputs and final states were bitwise equal over
  a sequence of calls. The stage wrote the same `main.mlirb` as the probe for these
  three assets. The probe was a one-off script outside the repository: it loaded both
  versions with GPU-preferred specialization, called every function in the order of the
  contract with fixed random inputs and zeroed states, and compared descriptors, outputs
  and final states bit for bit. The repository has no entry point for it; repeating it
  means writing that procedure again.
- The first implementation (`20261116T201921-cc114781`) stripped a temporary program:
  `AIModelAsset.program` reads the bytecode again on every access, so all twelve saved
  programs were unchanged while the digests were equal. Its outputs are not used, and
  the stage now also requires `main.mlirb` to change.
- After that fix, `strip` ran three times on the same exported assets, each time after
  a source change: `20261116T202807-b4d07f2b`, `20261116T204724-264d1994` and
  `20261116T212827-22314097`. `main.mlirb` and `main.hash` were byte-identical for all
  twelve assets in the three runs; only `creationDate` in `metadata.json` differed. The
  bundles and inventories come from `20261116T212827-22314097`.

| Asset | FP32 bytes | FP16 bytes |
|---|---:|---:|
| `tslm` | 5,888,743,347 | 2,944,859,765 |
| `ralm` | 1,510,408,578 | 755,394,848 |
| `proj` | 75,554,982 | 37,780,488 |
| `locenc` | 839,324,907 | 419,754,534 |
| `dit` | 848,519,242 | 424,354,676 |
| `vae` | 183,070,987 | 91,565,735 |
| Total | 9,345,622,043 | 4,673,710,046 |

Each bundle holds `manifest.json`, the unmodified `tokenizer.json` of the pinned
revision, the Apache-2.0 `LICENSE` of the official checkout, a `NOTICE` and the six
assets. No configuration is the default yet; it will be chosen after a listening
comparison. The stage checks every file of a bundle against its source: the asset files
against the sizes and SHA-256 values that `strip` recorded after the removal,
`tokenizer.json` against `model.json`, `LICENSE` against the file at the pinned commit
(11,298 bytes, SHA-256 `4f10acc209addacfad28293315c74c4cd648f771ee1263a748f1781d1e0265e4`),
and `manifest.json` and `NOTICE` against the bytes it wrote. A missing, extra or
changed file fails the stage. In `20261116T212827-22314097`, `clonefile` cloned all 18
asset files of each bundle and none was copied (`check.json` records both counts), so
the bundles share their blocks with the stripped assets. Copies of the three bundles
would need about 19 GB; the stage requires 25 GB of free space.

| Bundle | Precisions | Files | Bytes | Inventory |
|---|---|---:|---:|---|
| `fp32` | all FP32 | 22 | 9,349,325,520 | [fp32](../../Tools/Conversion/recipes/voxcpm2/bundles/fp32.inventory.json) |
| `fp16` | all FP16 | 22 | 4,677,413,664 | [fp16](../../Tools/Conversion/recipes/voxcpm2/bundles/fp16.inventory.json) |
| `fp16_proj32` | FP16, `proj` FP32 | 22 | 4,715,188,166 | [fp16_proj32](../../Tools/Conversion/recipes/voxcpm2/bundles/fp16_proj32.inventory.json) |

The path scan reads every byte and counts 132 byte patterns: the home,
external-volume and temporary-folder prefixes written with `/`, `\/`, `\u002f`,
`\u002F`, `%2f` or `%2F` as the separator, `-Users-` and the user name, each in UTF-8,
UTF-16LE and UTF-16BE. It records only counts and file names.

- The 66 files (18,741,927,350 bytes) of the three bundles: 0 matches.
- The six export folders before the removal (42 files, 14,019,535,475 bytes): 0 matches.
  The removal therefore did not have to remove a path. Planted strings confirmed that
  the scan detects a home path in UTF-8, UTF-16LE and UTF-16BE, with the JSON escapes
  `\/` and `\u002F` and the percent escape `%2f`, an external volume, a temporary
  folder, the user name and a match across a read boundary, and that it ignores a clean
  relative path.

`bundle_verify` loaded every asset of the three bundles; all function descriptors match
the contract. With the FP32 bundle, all 3,934 comparisons that decide the result were
within the criterion, `proj.fuse` gave 233 matches, 1 boundary-consistent call and no
failure, and every count, result and error equals the comparison before the removal.

## Time

The component times and the precision comparison above are the timing records. Each
memory run below also timed one generation of `long` (16.8 seconds of audio) as the
first generation of a new process: FP32 took 7.82 seconds (0.465 seconds per second of
audio) after a 7.07-second load, FP16 5.11 seconds (0.304) after a 3.40-second load.
The load includes specialization; whether a specialization cache was used was not
controlled. Time to the first playable audio, cold specialization and load times in an
app were not measured. In `20261116T212827-22314097`, `strip` executed for 71 seconds,
`bundle` for 607 seconds (most of it the path scan with 132 patterns) and
`bundle_verify` for 39 seconds; the record keeps the public run summary.

## Memory

The FP32 and FP16 bundles each generated `long` (seed 1708, 97 text tokens, 105
patches) once from a copy of the bundle, in a new process of the Python host, under the
shared heavy-job lock. The values below were measured with the bundles of
`20261116T204724-264d1994`. Their `main.mlirb` and `main.hash` are byte-identical to
those of the current bundles (`20261116T212827-22314097`), and only the creation dates in
`metadata.json` differ; the measurement was not repeated for the current bundles.

A one-off script outside the repository made the measurement. It loaded the assets of
a bundle through the recipe's Python host, generated once, and sampled
`proc_pid_rusage` (current and lifetime maximum physical footprint, resident size),
`getrusage` (maximum resident size), `vm_stat` (wired memory) and `sysctl vm.swapusage`
before loading, after loading, after the generation and after release. The repository
has no entry point for it; repeating it means writing that procedure again. The first
sample follows the imports and the tokenizer. Wired memory and swap are whole-system
values.

| Value (bytes) | FP32 | FP16 |
|---|---:|---:|
| Physical footprint before loading | 84,247,128 | 84,050,496 |
| Physical footprint after loading | 215,053,080 | 177,102,288 |
| Physical footprint after the generation | 838,137,128 | 661,102,512 |
| Physical footprint after release | 754,709,776 | 617,881,520 |
| Lifetime maximum physical footprint of the process | 842,347,816 | 662,626,224 |
| Resident size after the generation, assets held | 16,731,029,504 | 8,580,218,880 |
| Lifetime maximum resident size of the process (`getrusage`) | 22,462,464,000 | 11,290,116,096 |
| Physical footprint after release minus before loading | 670,462,648 | 533,831,024 |
| System wired memory after loading | 7,416,659,968 | 7,378,468,864 |
| System wired memory after the generation | 16,450,404,352 | 12,033,949,696 |
| System wired memory after release | 16,374,087,680 | 11,968,528,384 |

Release means closing the components and collecting garbage. Swap use stayed at
12,210.25 MB for the whole system at every sample. The resident size includes the
file-backed pages of the mapped weights, which the physical footprint does not count;
the two are not added. System wired memory rose during the first generation by about the
size of the assets and had not fallen at the sample right after release; it is not
attributed to this process alone. The same measurement on the bundles of
`20261116T202807-b4d07f2b`, whose programs are also byte-identical, gave process memory
values within 2.1%
and generation times within 1.1%; the load times differed by up to 25% (the JSON keeps
both). These are diagnostic values of the Python host, not the memory of an app, and
they do not set a minimum RAM.

## Not verified

- iPhone, and the behavior in the Mac app. No app reads the bundles yet.
- Reference audio (voice cloning), the VAE encoder, streaming playback, the Neural
  Engine and 8-bit or 4-bit quantization.
- Behavior for other sentences and seeds. The generation check covers five sentences
  with one seed; the precision comparison covers five sentences with three seeds.
- Same-input comparisons of the language models, their caches and `proj.fuse` along the
  generations without replacement.
- Quantitative audio quality (such as CER or human ratings) and listening. FP16 assets
  and configurations are recorded, not accepted.
- The memory of the `fp16_proj32` configuration; only the FP32 and FP16 bundles were
  measured.
- Publication. Publishing the bundles needs a separate information check.
