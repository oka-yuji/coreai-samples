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
