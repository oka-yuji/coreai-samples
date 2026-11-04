# JEV-27B validation records

These JSON files copy the numbers of the JEV-27B checks from the local run outputs of the
[`jev27b` recipe](../../../Tools/Conversion/recipes/jev27b/README.md) and of the performance measurements, so that the
[model record](../../model-reports/records/jev27b.json) can check its values against them. They hold IDs, counts,
SHA-256 values and metrics only: no corpus or JGLUE text, no local paths. Each file names its sources (the run and
step, or the file of the measurement) with their SHA-256; the run outputs themselves stay in the local work
directory. The definitions and the discussion are in [docs/jev27b.md](../../jev27b.md).

All runs used a Mac16,6 (Apple M4 Max) with 128 GiB of memory, macOS 27.0 (26A428), Xcode 27.0 (27A266a) and Apple
Swift 6.4, with one model job at a time under the shared machine lock. Run IDs carry the UTC start time; the dates in
this file and in the record are JST.

The recipe runs record neither the power source nor a git revision.

- **Power:** the record leaves `power` empty for the fidelity, quality and Japanese evidence. The system power log
  (`pmset -g log`) shows an adapter attached at each of its entries during these runs, except 39 seconds on battery at
  12:15 on 2026-11-03 during the `probe_base` step (stage 0q). The battery charge also fell during the long PyTorch
  steps of that day while the adapter was attached. The log is written at events, not continuously.
- **Code revision:** `code_revision` is the commit a run started from when that commit is still published. Otherwise
  it is a published commit in which the `Tools/Conversion` files used by the run are unchanged (`git diff`). The
  commit of each run is named below; the command line tool is the next item.
- **Evaluation CLI:** the recipe steps ran the evaluation build of `coreai-verify`, made at 22:01 on 2026-11-03, a
  minute after commit 5b4e1c6 (rebased as b6f1e0d in #474 with the same `Tools/Verify` sources). Its outputs write
  `arm64` as the machine, so it predates e247258, which changed only that name and two checks made before the rows
  run.

## Fidelity

[`fidelity.json`](fidelity.json): the `compare_f` step of run `20261103T142317-7668a060` (2026-11-03, #475). The
variant bundle (`decision.json` SHA-256 `26a11dca07a265feca4970dccdf2ebabaa005730132035e9df37b207cd26051e`, asset
`965396ce46af30aca1e65df8fe2e0fa4639be120b8bb18a7edeefc5dbbc86de0`) ran the 302 rows of F through
`coreai-verify typed-decide` (stage 3), compared at temperature 1 with stage 2 (the same quantized weights in PyTorch
float32). It records the criteria (largest difference at most 0.03 and mean at most 0.01 in every band with rows), the
row counts, and the largest and mean difference overall and per band. At the card temperatures it records the overall
largest and mean difference and the top-1 changes.

No measurement plan was fixed before this run in the form of the common protocol, so the record marks it `legacy`.
The run started from commit eb8e744, which the squash merge of #476 replaced. The record's `code_revision` is that
merge, c626c9a, in which the `Tools/Conversion` files used by the run are unchanged.

## Quality

[`quality.json`](quality.json): the `score_q` step of run `20261103T154310-65e7b5bb` and the `calibrate` step of run
`20261103T175311-463e881b` (both 2026-11-04, #447). Stages 0q and 2 of Q come from the `probe_base`,
`probe_merged` and `verify_quantization` steps of 2026-11-03, so the results span 2026-11-03 to 2026-11-04.

The file holds the following, as agreement with the teacher distribution, not real-world correctness:

- stages 0, 0q, 1', 2 and 3 of Q (900 rows) against the teacher distribution;
- the conditions fixed before measuring (D1-1 and D1-2) with their values, intervals and results;
- the temperature refit: the card, candidate and adopted temperature of each kind, the NLL, Brier score and ECE on H',
  and the fidelity of F at the adopted temperatures.

The record marks it `legacy` for the same reason as the fidelity run. `score_q` ran at commit d98083b, which a rebase
replaced, and `calibrate` at 31071af. The record uses 31071af, which keeps the `score_q` code unchanged and adds only
the `calibrate` step.

## Japanese

[`japanese.json`](japanese.json): the `jdev_fixture`, `prompt_adoption` and `score_j` steps of run
`20261103T184500-52091525` (2026-11-04, #448). It holds:

- the IDs of the 300 Jdev rows and the adoption of the prompt variant;
- the J results for both variants, stages 1' and 3, and temperatures 1 and the adopted ones;
- the option-order test, the stage differences and the per-row agreement.

There is no pass line: the evidence records that the measurement was completed, not that a criterion was met. The
prompt text is in [`wrappers.json`](../../../Tools/Conversion/recipes/jev27b/wrappers.json), whose SHA-256 the file
records. The adoption rule and the sets were fixed before J ran, but no measurement plan in the form of the common
protocol was fixed, so the record marks it `legacy`. The run started from commit b64eae5.

## Performance

The performance was measured three times on 2026-11-04 with the same rows P, timing boundary and counts: on AC power
with a 140 W adapter at 94 W under `plan-d2b-ac94` (#490), earlier on AC power with a 70 W adapter under `plan-d2b-ac`
(#488), and first on battery power under `plan-d2b` revision 3 (#480). The record's performance values come from the
94 W measurement (evidence `performance`); the other two stay as evidence `performance-ac-70w` and
`performance-battery` with their own files. Only the 94 W measurement started every row at a nominal thermal state
and every stage without user input and without audio, video or automation activity. The three are not a controlled
comparison of power sources: the thermal pressure, the battery state, the energy mode and the other activity on the
machine differ (see [docs/jev27b.md](../../jev27b.md#performance)).

### AC 94 W

[`performance-ac94.json`](performance-ac94.json): the measurement of 2026-11-04 from 11:21:07 to 11:30:03 (#490) under
the fixed plan `plan-d2b-ac94` with SHA-256 `47a12e99d22dbd40e2a41cedc2b91665bc587e44bbdeb6157d97c24d299c4cb2`. The plan
is `plan-d2b-ac` with only the power, the thermal condition, the other load, the code revision, the binary and the
per-row thermal wait changed, and it was frozen before the measurement. Besides the fields of the earlier files, every
latency row records its start (`started`, UTC in whole seconds), the seconds it waited for a nominal thermal state
(`thermal_wait_seconds`) and the state it started with (`thermal_state_at_start`), and `conditions` hold the gate of
every stage (watts, battery, energy mode, thermal pressure level, seconds without user input and matching assertions).
Its `sources` include the measurement log, the thermal and power logs and the assertions of `pmset -g log`.

The plan's code revision is 18890f5, which added `--wait-nominal-thermal` to `coreai-verify typed-decide`; the command
line measurements used a release build of that commit. Before every latency row (warmups included) the tool waited, up
to 600 s, until `ProcessInfo.processInfo.thermalState` was nominal. The cancellation test ran in the debug build of the
test target, as the plan states. Before the measurement `pgrep` found no other model process.

- **Power:** the 140W USB-C Power Adapter at 94 W (`pmset -g ac`) at every stage, the battery at 100%, charged. The
  energy mode on AC was High Power (`powermode 2` in `pmset -g`), the owner's setting, which was not changed.
- **Gate:** every stage started only at 94 W or more, with at least 120 s without user input (`HIDIdleTime`), with no
  `coreaudiod`, video wake lock or Computer Use assertion and at thermal pressure level 0, waiting for each condition
  up to 10 minutes. Only the cancellation test had to wait, 30 s, for 120 s without input.
- **Thermal pressure:** all 16 latency rows started at a nominal thermal state. Only the 100-token row of the third
  round had to wait, 54.51 s. During the latency process the level of `thermalmonitord` went to moderate at 11:25:51
  and heavy from 11:25:53 to 11:26:52 (moderate until 11:27:02): the 8,192-token row of the second round turned
  moderate and heavy about 20 s into it, and the next row waited until the level was back to nominal. The median rows
  of all four lengths, the loads and the cancellation test ran at nominal pressure. `pmset -g therm` showed no warning.
- **Other load:** the assertions of `pmset -g log` show the assertions of the measurement, a sleep-prevention
  `caffeinate` that had started before the measurement and ended at 11:22:19, and network tasks of `cloudd` from
  11:27:59 to 11:28:17, between the latency process and the cancellation test; no audio, video or automation activity.
  There was user input during the latency process, the last at about 11:26:13, while it waited for a nominal thermal
  state before the third round, so no row ran then; earlier inputs in that process cannot be excluded, since
  `HIDIdleTime` keeps only the last one. The CPU and GPU use of other processes was not recorded.

The record keeps this evidence as `standard` for the same reasons as the earlier ones: the plan named the power, the
energy mode, the thermal and other-load conditions and the debug build of the cancellation test before the
measurement. The same request without cancellation took 8.20 s in the debug build, within the 7.96 to 9.03 s of the
three measured 2,000-token rows of the release command line tool, so the debug build does not make `cancel_s` look
longer. The cancelled requests took 8.72 to 9.00 s, longer than that uncancelled run, for a reason not investigated.

### AC 70 W

[`performance-ac.json`](performance-ac.json): the measurement of 2026-11-04 from 09:22:08 to 09:31:10 (#488) under the
fixed plan `plan-d2b-ac` with SHA-256 `8763b18a0830395744f3f65e2bd27d5a151d85dc201b1296ea15e175af9128cb`. The plan is
plan revision 3 with only the power, the thermal condition, the code revision and the binary changed, and it was frozen
before the measurement. The file has the fields of the battery file, except that its `conditions` hold the adapter,
watts, battery, energy mode and thermal pressure level at the start of every stage, the waits before the stages, the
thermal pressure history of the run and the other activity on the machine. Its `sources` also include the measurement
log, the thermal and power logs and the assertions of `pmset -g log`.

The plan's code revision is 9b610f8 (main). The command line measurements used a release build of `coreai-verify` at
that commit, byte-identical to the build of 9d8fdf8 (the sources had not changed). The cancellation test ran in the
debug build of the test target, as the plan states.

- **Power:** the 70W USB-C Power Adapter at 68 W (`pmset -g ac`) at every stage; the battery charged from 64% to 67%.
  The plan says 62%, the value when it was frozen at 09:20. The energy mode on AC was High Power (`powermode 2` in
  `pmset -g`), the owner's setting, which was not changed.
- **Thermal pressure:** every stage started at level 0; the plan made each stage wait for it for up to 10 minutes. Only
  the cancellation test had to wait, 50 seconds. During the latency process the level of `thermalmonitord` went to
  moderate at 09:26:46 and heavy from 09:26:48 to 09:29:05 (moderate until 09:29:19). So the 8,192-token row of the
  second round and the whole third round ran at heavy pressure. The loads, the first round and the cancellation test
  ran at nominal pressure. The 2,000-token row of the second round, a median, ended within about a second after the
  change to moderate; the rows carry no clock times, so their times are rebuilt from the logs within about a second.
  Of the medians, only the 8,192-token one (second round) ran at heavy pressure. `pmset -g therm` showed no warning.
- **Other load:** before the measurement `pgrep` found no other model process. The assertions of `pmset -g log`
  show other activity during this run: audio sessions from about 09:22 to 09:28:47 and from about 09:29:36 to
  09:30:52, a video wake lock of a browser from 09:25:56 to 09:25:58, a UI automation session at 09:23:31 (during the
  third fresh load, the slowest of the three), Handoff and network tasks. The battery run had none of these except one
  short network task. The CPU and GPU use of other processes was not recorded.

The common protocol requires the conditions to be recorded and the plan and revisions to be fixed; the power source,
its watts and the energy mode are recorded conditions, not requirements. This plan named the AC power, the energy
mode, the wait for level 0 and the debug build of the cancellation test before the measurement, so the record keeps
this evidence as `standard`. The debug build does not make `cancel_s` look longer: the head is not computed after a
cancellation, and the same request without cancellation took 7.98 s in that build, less than the 9.33 to 10.39 s of
the three measured 2,000-token rows of the release command line tool. The cancelled requests took 8.60 to 9.00 s,
longer than that uncancelled run, for a reason not investigated (on battery, 9.06 to 9.20 s against 8.72 s). The
condition on other load rests on `pgrep` finding no other model process; the other activity above is recorded, not
controlled.

### Battery power

[`performance.json`](performance.json): the measurement of 2026-11-04 (#480) under the fixed plan with SHA-256
`d361e027c9f6c4929bc8cafe7ea4f378fcdbfe7e01e795a1229e6cbc5cd5d4b7` (plan-d2b revision 3; evidence
`performance-battery`). It records:

- the plan fields;
- the four rows P (exactly 100, 500, 2,000 and 8,192 tokens), with the IDs of their corpus parts and their SHA-256;
- the `summarize` output of every condition (latency per length, fresh and cached loads, memory, cancellation, bundle
  size);
- the per-row times and in-run memory maxima, and the load parts and specialization of each load;
- the cancellation attempts, the cancellation at the forward checkpoint, and the SHA-256 of every raw output
  (`sources`).

The plan's code revision is 9d8fdf8. The command line measurements used a release build of `coreai-verify` at that
commit. The cancellation test ran in the debug build of the test target, as the plan states. Before the measurement
`pgrep` found no other model process; the GPU use of other processes was not recorded.

The measurement ran on battery power by the owner's decision: from 90% to 76% between 06:23:59 and 06:31:34, no
adapter attached and Low Power Mode off. Plan revision 3 recorded this before any measurement.

`pmset -g therm` showed no warning, but the thermal pressure level of `thermalmonitord` went up. It was moderate from
06:28:52, in the last seconds of the second 8,192-token row, and heavy from 06:29:08 to 06:32:08, from the last
seconds of the third 2,000-token row. So the third round and the cancellation test ran under higher thermal pressure
(`conditions` in the file). Of the medians, the 100-token one (third round) and the end of the 8,192-token one (second
round) ran at moderate pressure.

The cancellation test runs the forward through a test backbone that does not pass the cancellation on, so a request
cannot end before the run returns and `cancel_s` includes the rest of the run. That the next row is bit-identical
after each kind of cancellation is an assertion of the five passing GPU tests listed in `tests`, not a recorded
number.

The common protocol requires the conditions to be recorded and the plan and revisions to be fixed. It does not
require a particular power source. This plan named the battery power and the debug build of the cancellation test
before the measurement, so the record keeps this evidence as `standard`. The debug build does not make `cancel_s`
look longer: the head is not computed after a cancellation, and the same request without cancellation took 8.72 s in
that build, less than the 8.91 to 9.08 s of the three measured 2,000-token rows of the release command line tool.
The cancelled requests took 9.06 to 9.20 s, longer than that uncancelled run, for a reason not investigated. The
condition on other load rests only on `pgrep` finding no other model process.

## Bundle

The calibrated bundle written by the `calibrate` step of run `20261103T175311-463e881b`: 13 files, 16,064,815,496
bytes in total, `decision.json` SHA-256 `a2b186a87a64d9e93a56f185a5ebc2a0613d00da7440eeb8c8c2229b64f6adf3`, asset
`965396ce46af30aca1e65df8fe2e0fa4639be120b8bb18a7edeefc5dbbc86de0`. `performance.py records` counted its size at 07:12,
after plan revision 3 was frozen, with the working tree that was committed as e879c67 at 07:13 (`bundle` in
`performance.json`). `shasum -a 256 -c SHA256SUMS` and `decision.py --check-files` pass on it (run again at 08:29 on
2026-11-04; the log stays with the local outputs).

## Memory stability (2026-11-05)

The published artifact is unchanged. [The memory record](memory-stability-20261105.json) compares the normal Release
CLI before and after the runtime change, then checks three longer workloads on a MacBook Pro (14-inch, 2024),
Apple M4 Max, 128 GiB (Mac16,6), macOS 27.0. The adapter reports 68 W on AC. Every row starts at a nominal
ProcessInfo thermal state; CPU builds and tests were allowed during these stability runs. These are separate
from the controlled latency comparison. Input text and raw corpus identifiers are not included.

| Workload | Warmup / measured calls | Kernel footprint peak | Peak RSS, including load | Added system swap | Output comparison |
|---|---:|---:|---:|---:|---|
| 100/500/2000/8192 tokens alternating | 4 / 100 | 8,393,766,472 B | 32,646,430,720 B | 0 B | Every logit and both probability arrays bit-identical |
| Same 8192-token input | 1 / 100 | 6,827,587,384 B | 32,336,232,448 B | 0 B | Bit-identical |
| 6144/8192 tokens alternating | 2 / 20 | 9,169,450,928 B | 32,669,925,376 B | 0 B | Bit-identical |

The first two cases require a kernel peak below 12 GiB and a first-five versus last-five median increase
of at most 512 MiB for each same-length series. The largest increase was 127,090,760 B in the mixed case
and 34,078,792 B in the same-length case. The nearby-length case has a separate 24 GiB peak limit; its
median changes are recorded without applying the 512 MiB gate. All cases require swap growth below 128 MiB.
RSS and footprint are different counters and are not added. A sampled peak is not substituted for the kernel counter.

The normal CLI's original 16 alternating calls reached a kernel footprint peak of 21,396,977,368 B.
The corresponding 16-call smoke with runtime recycling reached 8,137,602,608 B with identical outputs.
Holding AIModel and recreating only InferenceFunction was rejected: it completed three rows before
the stop guard triggered, and its kernel peak reached 50,319,340,624 B. A polling stop trigger is not a strict
maximum-memory bound. The specific allocation site inside Core AI remains unidentified.

The runtime permits three shape changes and releases both model and function before the fourth; an unchanged
shape stays loaded. A bookmark keeps the identity of the specialization without retaining the model.
If the cache entry is absent, the original bundle version and all file hashes are revalidated before reloading.
A successful reload updates the bookmark. A failed or cancelled reload or backbone run discards the runtime, and busy remains set
until the running operation finishes. The private direct-runtime diagnostic found about 3.7 seconds per reload;
its timings are not the final before/after benchmark. The earlier 94 W latency measurements remain historical evidence.

Six serialized GPU tests passed in Release with testability enabled. The added session test waits for an
uncooperative 1472-token forward to finish, reports cancellation, reloads once and returns a bit-identical next
answer. CPU tests separately cover busy, failed reloads, stale bookmarks and modified bundles. The GPU suite
is diagnostic evidence without a separate benchmark plan, so its record remains legacy rather than a latency claim.

## Final-token output (2026-11-05)

[`last-token-20261105.json`](last-token-20261105.json) records the private candidate from #500. Conversion
`20261104T194219-e2447dc1` at `8fcb8843bcb90171b7f5f22473df052cead75758` produced decision SHA-256
`0db5ca716c7fea75605bffddf2d1d6de03fd80c471b3528277beb98a663d9880` and asset SHA-256
`be2b71cf1cf8385b0b096ebd1f103f705a3e4ff4e86819a1a999845f47f7a3b7`. All 1,622 quantized tensor hashes and
their layout match the published asset; all 48 gated delta updates retain float32 inputs. Head, tokenizer,
template, slot layout and published temperatures are unchanged. The bundle is marked `unpublished-candidate`.

The output descriptor is float16 `[1,1,5120]`. It contains 10,240 bytes of elements, compared with 83,886,080
for the previous 8,192-token output. The backbone still computes every layer; this number is not total memory
or an acceleration ratio. The first conversion (`20261104T193249-84183233`) failed the descriptor check:
`select` followed by `unsqueeze` produced `[?,1,?]`. An explicit reshape fixes the descriptor without relaxing it.

| Check | Result |
|---|---|
| F, 302 rows | 301 forward passes; the 8,193-token row rejected; every band passes max ≤0.03 and mean ≤0.01 at T=1 |
| F versus same-weight quantized reference | Maximum probability error 0.00644122, mean 0.00114157; no top change |
| Q, 900 rows | Existing D1-1 and D1-2 pass; no top change versus previous native results |
| Old versus candidate native values | Not bit-identical; maximum T=1 probability difference 0.00043249 on F and 0.00061208 on Q |
| Q choice agreement with teacher, T=1 | 86.6667% before and after; merged reference 87.3333% |
| Q noul soft Brier, T=1 | 0.001653858 before; 0.001654508 after |
| Q score normalized MAE, T=1 | 0.021708660 before; 0.021713385 after |
| Old bundle through new runtime | All outputs bit-identical at 100/500/2,000/8,192 tokens |
| Candidate runtime | Six serialized GPU tests passed in Release with testability enabled, including cancellation and recovery |

These quality metrics measure agreement with the teacher distribution. The Q choice difference against the merged
reference is −0.667 percentage points; its 95% bootstrap interval [−3.000, +1.667] crosses the −2-point limit.
The existing decision rule uses the point estimate, so it passes without establishing the whole interval above
that limit. The old native and candidate outputs were scored with the same frozen references and criteria.
For the F/Q native-to-native probability comparison, saved old logits were normalized with the production Swift
head at the published temperatures after verifying their original T=1 probabilities bit for bit; the old backbone
was not rerun. Q acceptance metrics above use T=1. The original source-card temperatures in the scorer remain
separate from the published refitted temperatures in the bundles.

Calibration files and values were reused. Cal/H and the Japanese set were not rerun for this candidate.
These are diagnostic checks under the shared model lock on the same M4 Max 128 GiB machine, on AC reporting
68 W. Per-row thermal state was not recorded for these CLI runs or GPU tests. CPU builds/tests were permitted during validation;
these rows are not the controlled speed comparison. There is no separate common-protocol benchmark plan for
these checks, so the model record marks them legacy. All source-file hashes are in the JSON; private inputs
and raw outputs are excluded. The public HF artifact and its prior evidence remain unchanged.

## Controlled comparison (2026-11-05)

[`comparison-20261105.json`](comparison-20261105.json) preserves all 128 timing observations, including 32 warmups,
source-file hashes, per-process memory/load/power observations, and the fixed plan
`280dc8957a73de74f944a4937afd585516672f8f7bbd9fd41c0329ae196c1498`. Runs took place from 2026-11-04 21:15:15 to
21:54:25 UTC (2026-11-05 JST). Eight fresh processes completed in order A→B→C→D→D→C→B→A, without a guard stop.
A is the published runtime/artifact, B adds bounded shape reuse, C adds the unpublished final-token artifact, and
D runs the pinned original BF16 backbone with active, unmerged LoRA in PyTorch/MPS and its FP32 CPU decision head.
The JSON pins code, binary, artifacts, source revision, dependencies and measurement helpers for each condition.

All runs used the same MacBook Pro (14-inch, 2024), Apple M4 Max, 128 GiB, macOS 27.0 (26A428), AC reporting 68 W,
and High Power mode. Battery charge ranged from 100% to 94% over the sequence. Every row, including warmups,
started at ProcessInfo thermalState nominal; this is not a constant-temperature guarantee. No build, test or
other model job ran during the measured processes. Background CPU/GPU utilization was not fully instrumented.
All processes added 0 B of system swap; approximately 2.0 GB of pre-existing system swap was present.

Each process runs one warmup at each of 100/500/2000/8192 tokens, then three measured rounds. Every reported time
below has n=6 and is **median [minimum, maximum] in seconds**. The four-request sum adds API times within each
measured round. The boundary is one render/tokenization, backbone, head/softmax and a synchronized CPU result.
Input prevalidation, initial load, JSON serialization and thermal waits are excluded. Native JSON `elapsed_seconds`
is the API interval; the diagnostic stderr line also includes prevalidation and is not used for this table.

| Tokens | A: published | B: memory fix | C: final token | D: source |
|---:|---:|---:|---:|---:|
| 100 | 0.649915 [0.583927, 0.712413] | 6.021963 [5.950145, 6.077657] | 6.103281 [5.955404, 6.418380] | 0.843730 [0.813807, 0.900861] |
| 500 | 2.219713 [1.956196, 2.359130] | 1.953377 [1.947658, 1.956321] | 1.956813 [1.951799, 1.958022] | 2.623212 [2.503627, 2.710381] |
| 2000 | 9.226327 [8.043565, 9.379876] | 8.294314 [8.138351, 8.306313] | 8.207459 [7.648074, 8.278129] | 10.847312 [10.050650, 11.272265] |
| 8192 | 38.925713 [38.165324, 40.300910] | 37.737326 [36.815382, 38.595914] | 38.534624 [38.115242, 38.729626] | 46.967334 [45.767156, 47.754686] |
| Four-request sum | 50.960642 [48.877498, 52.646836] | 53.840935 [53.153956, 54.930249] | 54.837211 [53.675951, 55.328512] | 60.705899 [60.365768, 62.602024] |

B/C recycle before each measured 100-token row, so those rows include reload and first-run costs. They are not
same-shape short-input measurements. The medians of the four lengths need not add up to the median round sum.
Six observations do not support a p95, and the ranges are observed minima/maxima rather than confidence intervals.

Memory below is the maximum across the two complete processes, including warmup and load:

| Condition | Observed kernel footprint peak (GiB) | Peak RSS (GiB) |
|---|---:|---:|
| A: Published runtime | 19.862592 | 30.123672 |
| B: Memory stabilization | 7.671247 | 30.368088 |
| C: Final-token candidate | 8.029096 | 30.380081 |
| D: Source PyTorch/MPS | 65.535377 | 51.720261 |

RSS comes from wait4; footprint uses the largest observed kernel lifetime peak counter, sampled externally every
100 ms for every condition. The largest observed sampling gap was 0.122 s. A sampled current-footprint maximum
is kept separately, and the final sampling gap remains unobserved. RSS and footprint are different counters;
they are not added or treated as the machine's minimum RAM requirement. Source row-end MPS allocator counters
remain in the private raw results. D loads an unused 2,542,796,800-byte generation head, but does not forward it.

Load phase observations are stored per process. No filesystem or model-cache purge was performed, and every native
process used a cached specialization. The source adapter-load interval includes its final MPS synchronization;
these are implementation intervals, not cold-start or individual GPU-stage rankings. Native precision/LoRA merging
and source BF16/unmerged LoRA differ, so this is a complete-representation comparison rather than framework overhead.

Across the 32 P rows per condition, A/B logits and both probability arrays are bit-identical. C differs from A by
at most 0.000081540 at T=1, and D by at most 0.021733382; neither changes the top choice on these four inputs.
Shipping temperatures differ between native and source, so cross-representation probability comparisons use T=1.

The following descriptive Q comparison uses the same 900 frozen teacher distributions and existing scoring functions:

| Q at T=1 | Source BF16, unmerged LoRA | Published native | Final-token native |
|---|---:|---:|---:|
| Choice teacher agreement (%) | 87.333333333 | 86.666666667 | 86.666666667 |
| Noul soft Brier | 0.001398183 | 0.001653858 | 0.001654508 |
| Score normalized MAE | 0.021251174 | 0.021708660 | 0.021713385 |

Source Q uses the saved original MPS/BF16 outputs from `20261103T154310-65e7b5bb`; the source backbone was not
rerun for Q. Published-native Q also reuses saved logits; the candidate Q was rerun for #500. Full Q was not rerun
for the memory-only runtime. All metrics use T=1 and measure agreement with the teacher, not real-world accuracy.
The original scoring attempt supplied the reduced normalization document to the full input checker and stopped;
it was corrected to the complete original native result without changing input data or acceptance criteria.
No new pass threshold is imposed on this descriptive source comparison. The existing F/Q criteria, point-estimate
rule, choice interval crossing the threshold, and Cal/H/Japanese not rerun are retained in the candidate record.

The adopted change is bounded shape reuse: it reduces the observed kernel peak while adding reload cost. The
final-token variant has no demonstrated speed or overall-memory advantage over B, so it remains a candidate and
the public HF artifact is retained. Further work should first address reload cost and shape-dependent retention,
then inspect state/output lifetimes in a dedicated decision graph. Layer/width reduction or distillation requires
fresh quality evaluation. No Core ML result is inferred from these measurements.

Recompute the statistics from the public observations:

~~~sh
python3 -m Tools.Conversion.adapters.jev27b.summarize_comparison   docs/validation/jev27b/comparison-20261105.json --check
~~~

The summarizer rejects missing, reordered, duplicate, failed or non-nominal processes/rows and non-finite or
invalid times, including invalid warmups. It requires eight unique run IDs, excludes exactly the first four
rows of each process, and keeps all valid observations. Private corpus text, raw corpus IDs and local paths are excluded.
