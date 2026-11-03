# JEV-27B validation records

These JSON files copy the numbers of the JEV-27B checks from the local run outputs of the
[`jev27b` recipe](../../../Tools/Conversion/recipes/jev27b/README.md) and of the performance measurement, so that the
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
  it is the first published commit whose `Tools/Conversion` and `Tools/Verify` files used by the run are unchanged
  (`git diff`). The commit of each run is named below.
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
merge, c626c9a, whose files used by the run are unchanged.

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

[`performance.json`](performance.json): the measurement of 2026-11-04 (#480) under the fixed plan with SHA-256
`d361e027c9f6c4929bc8cafe7ea4f378fcdbfe7e01e795a1229e6cbc5cd5d4b7` (revision 3). It records:

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

The cancellation test runs the forward through a test backbone that does not pass the cancellation on, so `cancel_s`
is the rest of the run by construction.

The common protocol requires the conditions to be recorded and the plan and revisions to be fixed. It does not
require a particular power source. This plan named the battery power and the builds before the measurement, so the
record keeps this evidence as `standard`.

## Bundle

The calibrated bundle written by the `calibrate` step of run `20261103T175311-463e881b`: 13 files, 16,064,815,496
bytes in total, `decision.json` SHA-256 `a2b186a87a64d9e93a56f185a5ebc2a0613d00da7440eeb8c8c2229b64f6adf3`, asset
`965396ce46af30aca1e65df8fe2e0fa4639be120b8bb18a7edeefc5dbbc86de0`. `performance.py records` counted its size with the
code of commit e879c67 after plan revision 3 was frozen (`bundle` in `performance.json`). `shasum -a 256 -c SHA256SUMS`
and `decision.py --check-files` pass on it (run again on 2026-11-04).
