# JEV-27B validation records

These JSON files copy the numbers of the JEV-27B checks from the local run outputs of the
[`jev27b` recipe](../../../Tools/Conversion/recipes/jev27b/README.md) and of the performance measurement, so that the
[model record](../../model-reports/records/jev27b.json) can check its values against them. They hold IDs, counts,
SHA-256 values and metrics only: no corpus or JGLUE text, no local paths. Each file names its source run, step and the
SHA-256 of the source file; the run outputs themselves stay in the local work directory. The definitions and the
discussion are in [docs/jev27b.md](../../jev27b.md).

All runs used a Mac16,6 (Apple M4 Max) with 128 GiB of memory, macOS 27.0 (26A428), Xcode 27.0 (27A266a) and Apple
Swift 6.4, with one model job at a time under the shared machine lock. The recipe runs ran on AC power; the
performance measurement ran on battery power (below).

## Fidelity

[`fidelity.json`](fidelity.json): the `compare_f` step of run `20261103T142317-7668a060` (2026-11-03, #475). The
variant bundle (`decision.json` SHA-256 `26a11dca07a265feca4970dccdf2ebabaa005730132035e9df37b207cd26051e`, asset
`965396ce46af30aca1e65df8fe2e0fa4639be120b8bb18a7edeefc5dbbc86de0`) ran the 302 rows of F through
`coreai-verify typed-decide` (stage 3), compared at temperature 1 with stage 2 (the same quantized weights in PyTorch
float32). It records the criteria (largest difference at most 0.03 and mean at most 0.01 in every band with rows), the
row counts, the largest and mean difference overall and per band, the top-1 changes and the same statistics at the
card temperatures. No measurement plan was fixed before this run in the form of the common protocol, so the record
marks it `legacy`.

## Quality

[`quality.json`](quality.json): the `score_q` step of run `20261103T154310-65e7b5bb` and the `calibrate` step of run
`20261103T175311-463e881b` (2026-11-03 to 2026-11-04, #447). Stages 0, 0q, 1', 2 and 3 of Q (900 rows) against the
teacher distribution, the conditions fixed before measuring (D1-1 and D1-2) with their values, intervals and results,
and the temperature refit: the card, candidate and adopted temperature of each kind, the NLL, Brier score and ECE on
H', and the fidelity of F at the adopted temperatures. The values are agreement with the teacher distribution, not
real-world correctness. Recorded as `legacy` for the same reason as the fidelity run.

## Japanese

[`japanese.json`](japanese.json): the `jdev_fixture`, `prompt_adoption` and `score_j` steps of run
`20261103T184500-52091525` (2026-11-04, #448). The IDs of the 300 Jdev rows, the adoption of the prompt variant, and
the J results for both variants, stages 1' and 3 and temperatures 1 and the adopted ones, with the option-order test,
the stage differences and the per-row agreement. There is no pass line: the evidence records that the measurement was
completed, not that a criterion was met. The prompt text is in
[`wrappers.json`](../../../Tools/Conversion/recipes/jev27b/wrappers.json), whose SHA-256 the file records.

## Performance

[`performance.json`](performance.json): the measurement of 2026-11-04 (#480) under the fixed plan with SHA-256
`d361e027c9f6c4929bc8cafe7ea4f378fcdbfe7e01e795a1229e6cbc5cd5d4b7` (revision 3). It records the plan fields, the four rows P (exactly 100, 500, 2,000 and 8,192 tokens; IDs of their corpus parts and SHA-256 values), the `summarize` output of every condition (latency per length, fresh and cached loads, memory, cancellation, bundle size), the per-row times and in-run memory maxima, the load parts and specialization of each load, and the cancellation attempts. The measurement ran on battery power by the owner's decision: from 90% to 76% between 06:23:59 and 06:31:34 JST, no
adapter attached and Low Power Mode off; plan revision 3 recorded this before any measurement. `pmset -g therm` showed
no warning, but the thermal pressure level of `thermalmonitord` was moderate from 06:28:52 and heavy from 06:29:08 to
06:32:08, so the third round and the cancellation test ran under higher thermal pressure (`conditions` in the file).
The cancellation test runs the forward through a test backbone that does not pass the cancellation on, so `cancel_s`
is the rest of the run by construction. The common protocol requires the power state to be recorded, not a particular power source, so the record keeps this evidence as `standard`.

## Bundle

The calibrated bundle written by the `calibrate` step of run `20261103T175311-463e881b`: 13 files, 16,064,815,496 bytes in
total, `decision.json` SHA-256 `a2b186a87a64d9e93a56f185a5ebc2a0613d00da7440eeb8c8c2229b64f6adf3`, asset
`965396ce46af30aca1e65df8fe2e0fa4639be120b8bb18a7edeefc5dbbc86de0`. Its size is recorded in `performance.json`
(`bundle`). `shasum -a 256 -c SHA256SUMS` and `decision.py --check-files` pass on it.
