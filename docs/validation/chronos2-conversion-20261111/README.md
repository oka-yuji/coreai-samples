# Chronos-2 conversion verification — 2026-11-11

Issue [#541](https://github.com/oka-yuji/coreai-samples/issues/541). The production recipe at `e749b5653298b826770f57ac6a814967fad7162a` completed prepare, independent official reference, export, native verification and bundle creation. [Results](results.json), [reference vectors](reference.json), [native vectors](actuals.json), [bundle inventory](inventory.json) and [sanitized plan](plan.json) retain the exact input/output contract and identities.

## Conditions and gate

- Official model: `amazon/chronos-2@29ec3766d36d6f73f0696f85560a422f50e8498c`. FP32, one series, one 16-step output patch, all 21 quantiles, no state.
- Apple M5 Max, 137,438,953,472 B RAM, macOS 27.2.1 (26B131), Xcode 27.2 (27B52), macOS SDK 27.2. Frozen Python 3.11.9, chronos-forecasting 2.3.2, Torch 2.9.0, coreai-torch 0.4.3 and coreai-core 1.0.0b3.
- The shared heavy-job lock covered model reads. GPU preference was requested; the actual delegate was not profiled. AC power and the absence of reported thermal/performance warnings were sampled before and after. Temperature, utilization and memory were not measured.
- Eighteen fixed synthetic cases cover raw lengths 1/15/16/17/32/64/1008/8192, missing values, a single observation, zero/positive constants, periodicity, negative scale and larger counts. One inference per case; no warmup. This is a fidelity check, not a latency benchmark.
- All 6,048 scalar outputs satisfy `abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`. The largest absolute difference is `0.00152587890625`; the threshold is evaluated per value and includes the relative term.
- Frozen controller plan SHA-256: `803abab0c116013d8c4c0f2c9d3e567043f1f4ccec531ef5dfa5ef2a68a9f88d`. The public plan view removes the host developer directory and is not byte-identical to that private plan. The input fixture, recipe and all source hashes remain available in the public view.

The official CPU reference loads an unmodified model in its own process. The export wrapper is checked against those vectors before conversion. Native vectors are independently rechecked during packaging; a pass flag alone cannot publish a bundle. No live data, customer data, weights or raw logs are committed here.

## Official raw-input API comparison

The pinned `chronos-forecasting` 2.3.2 `Chronos2Pipeline.predict` was also called directly with each raw, unpadded fixture as a single series, `prediction_length=16`, `context_length=8192`, batch size 1 and cross-learning disabled. All 18 cases / 6,048 outputs were bit-identical to the saved official-forward references. [The API parity record](official-pipeline-parity.json) includes per-case Float32 tensor hashes and the installed upstream source-file hashes. This confirms the supported profile's raw-input preprocessing; inputs beyond the declared limit, multiple series and covariates remain outside the profile.

## Retained failures and fixes

The initial plain export stopped on unsupported `aten.isnan.default` and `aten.unfold.default`. The limited decomposition uses self-inequality for NaN and reshape only for complete nonoverlapping last-axis patches of size/stride 16.

The initial scalar-epsilon export passed 16 of 18 cases. The [failed probe](initial-probe.json) retains both failures: one-observation inputs amplified a small normalization constant error at the tail quantiles. An isolated normalization probe returned `1.0013580322265625e-5` for the original FP32 epsilon `9.999999747378752e-6`. `torch.full_like(scale, eps)` preserved the FP32 constant. The fixture set and tolerance were unchanged for the passing result.

The first repeated production run recomputed every stage because the cache fingerprint included a fresh preparation directory's physical name. A model-free regression failed before the fix and passed afterward. The fingerprint now uses the declared dependency identity and verified contents while actual execution keeps real paths. In `20261111T124219-59424758`, preparation ran and the other four stages reused their verified artifacts. Reuse is not a new numerical measurement.

Changing the runner invalidated the setup stamp: `20261111T123427-90e7e6a1` failed in preparation before any later stage. Setup was rerun against the changed source. The final fresh fidelity run is `20261111T124413-e2465045` and used `--refresh`.

## Verification scope

- Chronos CPU checks: 68 tests; SF's independent contract, recipe and test checks passed, with scope/base/protected-input and source-immutability checks.
- Full Conversion regression after the cache fix: 518 tests, no errors; 46 optional checks require other model-specific environments and remain skipped. NumPy was installed and all Chronos numerical checks ran. Earlier missing-coreai-opt and sandbox process-observation failures were retained and rerun after fixing the validation environment.
- Model-report metric tests: 29 tests. WAPE is an uncapped ratio; zero-denominator results remain unmeasured rather than becoming zero.
- Required Swift lint/format, Rust input build, package tests, Mac Release build and package pin checks passed on the unchanged Swift source at `fcdbc74fea743684fcccf60c79b84b7a79387209`. Later changes in this unit are Python, recipe inputs and documentation; final privacy/report checks are separate.

Swift native equality, model-management integration, actual UI, iPhone acceptance, held-out forecasting quality, latency, memory, cancellation and release are not established by this conversion record. The remaining implementation is assigned to [#543](https://github.com/oka-yuji/coreai-samples/issues/543) through [#547](https://github.com/oka-yuji/coreai-samples/issues/547).
