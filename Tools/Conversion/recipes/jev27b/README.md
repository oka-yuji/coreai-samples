# JEV-27B typed decisions

This recipe pins the inputs, environments and evaluation sets for converting the System 1
path of [autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B) (revision
`14b425269ed24452e0a5ad5a8f3e51d139647a88`): the Qwen3.8-27B text backbone with its
unmerged LoRA and the 24-slot fp32 decision head. The recipe merges the LoRA, records PyTorch
references, converts the merged backbone to a 4-bit Core AI asset that returns hidden states,
records the quantized references, assembles a typed-decision bundle
([docs/jev27b.md](../../../../docs/jev27b.md)) and compares the Core AI output of the bundle with the
quantized reference. The later Q steps measure agreement with the frozen teacher distribution; this is not real-world accuracy. These files record what the
steps rely on:

- `model.json`: the contract checked against the downloaded adapter, head, judge and
  calibration files. A mismatch fails the check; the contract is not adjusted to the files.
- `ready.json` from the `prepare` step: integrity, tokenizer comparison, adapter and head
  layout, and the versions of both environments and the Apple toolchain.
- `fixtures/`: identifiers and hashes of the fixed sets F, Q, Cal, H and J
  ([fixtures/README.md](fixtures/README.md)). No corpus or JGLUE text is stored.
- `Tools/Conversion/adapters/jev27b/typed_head.py`: rendering of the `bare-v1` template,
  slot ranges, temperature and softmax, token bands, tokenization and the final hidden state.

## Local inputs

Setup never downloads models or datasets. Place the pinned inputs under the workspace
(default `.local/conversion`), each with a sibling `<folder>.integrity.json` that records the
size and the LFS SHA-256 or git blob SHA-1 of every file:

| Folder | Source |
|---|---|
| `models/autotrust--JEV-27B` | all 65 files at the revision above |
| `models/Qwen--Qwen3.8-27B-meta` | `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`: config, tokenizer files, README, LICENSE |
| `datasets/SargeDev--jev-distill-corpus-v3` | `README.md`, `test.jsonl`, `test_set_30k.jsonl` and `calibration.jsonl` at `75111a13a8455032f0435050bfc7e0c9026ecfe8` |
| `datasets/yahoojapan--JGLUE` | the JNLI, JCommonsenseQA and JSTS validation files of JGLUE v1.2.0 (`c96c6af54200a404f31a3a4cd0b289786aa78809`) |

Use `hf download <id> --revision <revision> --local-dir <folder>` with `HF_HOME` and
`HF_HUB_CACHE` pointing at `.local/conversion/jev27b/hf-cache`, so that no shared Hugging Face
cache is written. The JGLUE files are fetched from
`https://raw.githubusercontent.com/yahoojapan/JGLUE/<revision>/<path>` with the paths in
`fixtures/jglue.json`. These are the files that the `shunk031/JGLUE` loading script
(`41cc99f1c01d41b1ab13435ae97b7076c2199f4c`) downloads; the script itself is not executed.

## Setup, plan and run

~~~sh
python3 -m Tools.Conversion.local setup --recipe jev27b
swift build -c release --package-path Tools/Verify --product coreai-verify
mkdir -p .local/conversion/jev27b/bin
cp Tools/Verify/.build/release/coreai-verify .local/conversion/jev27b/bin/coreai-verify
python3 -m Tools.Conversion plan Tools/Conversion/recipes/jev27b/recipe.json \
  --bindings .local/conversion/jev27b.paths.json --set verify=$PWD/.local/conversion/jev27b/bin/coreai-verify
python3 -m Tools.Conversion run Tools/Conversion/recipes/jev27b/recipe.json \
  --bindings .local/conversion/jev27b.paths.json --work-dir .local/conversion/work/jev27b \
  --lock-dir "${COREAI_HEAVY_JOB_LOCK:-$HOME/Library/Caches/coreai-samples/heavy-job}" \
  --set verify=$PWD/.local/conversion/jev27b/bin/coreai-verify
~~~

The `typed_decide_f` step runs the executable named by the variable `verify`, which has no default: pass its absolute
path with `--set verify=...`, as above. `python3 -m Tools.Conversion.local plan|run --recipe jev27b` does not pass it
and stops with an unresolved variable. The step runs a copy of the executable rather than the build product, because
the runner hashes the executable into the step's cache key: copying a new build makes every step that runs it
(`typed_decide_f`, `typed_decide_q`, `typed_decide_cal`, `typed_decide_h`, `typed_decide_jdev` and `typed_decide_j`)
and every step that depends on them run again, including `compare_f`, `score_q`, `calibrate` and the Japanese steps.
The steps before them stay reused.

Setup checks the integrity records and then, under the shared machine lock, creates:

- the reference environment `.local/conversion/jev27b/.venv` from this folder's
  `pyproject.toml` and `uv.lock` (Python 3.11, torch 2.15.0, transformers 5.16.1, peft 0.21.2);
- the converter environment `.local/conversion/upstream-jev27b/coreai-models`: a clean checkout
  of `apple/coreai-models` at `d3382a59ecce6ba13536130aa87367611b6c4676`, its frozen
  `coreai-models` package environment and the release `llm-runner`;
- `.local/conversion/jev27b.paths.json` and `.local/conversion/jev27b/setup.json`.

`--snapshot` selects another JEV-27B folder. The first two steps check the inputs and sets:

- `prepare` writes `ready.json` without re-hashing the weight shards. It requires each folder to
  hold exactly the files of its integrity record (apart from `.cache`), compares sizes and
  modification times with the record, and re-hashes files up to 32 MiB. It stops when
  `pyproject.toml`, `uv.lock`, the converter revision or `llm-runner` differ from those recorded
  by setup; run setup again after changing them.
- `fixture` regenerates every set, requires byte-identical files and restores each row from the
  corpus and JGLUE files through `load_rows` with matching hashes.

The steps that read the weights are heavy, so `run` holds the shared machine lock for the whole
pipeline. Each executed `merge` adds a 54 GB checkpoint under the work directory's `objects/`, and
a failed or cancelled `merge` leaves its partial checkpoint under `runs/<run>/merge/`. A step is
reused only when its definition, inputs and dependencies, every `Tools/Conversion/*.py` file, the
OS build and platform, the reference environment, the executable and the environment variables
passed to it (`PATH`, `HOME`, `TMPDIR`, `LANG`, `LC_ALL`, `DEVELOPER_DIR`) are unchanged, so run
it again from the same shell settings; `--refresh` always executes it again.

## Final-token candidate

`last-token.json` is a separate conversion recipe for #500. It reuses the verified merged checkpoint and
the published calibrated bundle; it does not rerun merging or change the original recipe. Add three paths
to a private bindings file: `verified_merge` (the merge artifact folder containing `checkpoint` and
`manifest.json`), `baseline_gdnfp32` (the original variant artifact folder) and `published_bundle`
(the calibrated public bundle). Keep the existing reference and converter bindings.

~~~sh
python3 -m Tools.Conversion plan Tools/Conversion/recipes/jev27b/last-token.json \
  --bindings .local/conversion/jev27b-last-token.paths.json
python3 -m Tools.Conversion run Tools/Conversion/recipes/jev27b/last-token.json \
  --bindings .local/conversion/jev27b-last-token.paths.json \
  --work-dir .local/conversion/work/jev27b-last-token \
  --lock-dir "${COREAI_HEAVY_JOB_LOCK:-$HOME/Library/Caches/coreai-samples/heavy-job}"
~~~

The output `last_token_hidden_state` is float16 `[1,1,5120]`. The exporter requires identical quantized
tensor hashes/layout and the existing float32 GDN operations. It copies the published head, tokenizer,
template, slot definitions and calibration unchanged, validates the original asset identity and writes a
new `decision.json` with `distribution.status = unpublished-candidate`. The runtime accepts both the old
`hidden_states` descriptor and this fixed final-position descriptor, and reads the tensor's actual strides.

Conversion success is separate from fidelity, quality and performance. The completed F/Q and runtime checks,
including the failed first descriptor attempt, are in the
[candidate validation record](../../../../docs/validation/jev27b/README.md#final-token-output-2026-11-05).
This recipe does not upload or replace the published model.

## Controlled comparison tools

`Tools.Conversion.adapters.jev27b.benchmark_reference` measures the pinned original System 1 path on MPS.
It uses the existing reference environment and accepts `--model`, `--snapshot`, a prepared 16-row `--input`,
the matching four-input `--manifest`, `--thermal-helper` and a new `--output` directory. The helper is built
from `thermal.swift` with the pinned Apple toolchain. The input must be the fixed P sequence repeated four
times; row/token hashes, active unmerged LoRA, evaluation mode and MPS residency are checked before timing.
MPS CPU fallback is rejected. Partial failures remain failed in `status.json`; only a complete run writes
`result.json`. Run it under the shared model lock and the same external memory/power guard as native trials.

The completed A/B/C/D comparison and its environmental limits are in the
[validation record](../../../../docs/validation/jev27b/README.md#controlled-comparison-2026-11-05).
The public numbers can be recomputed without weights or corpus text:

~~~sh
python3 -m Tools.Conversion.adapters.jev27b.summarize_comparison \
  docs/validation/jev27b/comparison-20261105.json --check
~~~

The source Q comparison reuses saved original MPS/BF16 logits with the existing `score.judge` and `score.summary`
functions at T=1. Input/teacher identity and completeness are checked against the same frozen Q; there is no
new model run or changed acceptance threshold. The record preserves the source hashes and normalization.

## Merge and references

- `merge` adds each LoRA pair to its projection as `bf16(fp32(W) + 2.0 * (fp32(B) @ fp32(A)))`,
  one tensor at a time on the CPU, and copies every other tensor bit for bit. It stops unless all
  400 pairs are used exactly once and the shards add up to the source `total_size`.
  `checkpoint/` holds the 13 shards with the source file names and membership, the keys renamed
  from `model.` to `model.language_model.` (`lm_head.weight` unchanged), a recomputed index, the
  `Qwen/Qwen3.8-27B` `config.json` copied byte for byte (`qwen3_5` with `text_config`; its
  `vision_config` is kept, but no vision or MTP tensors exist) and the three distributed tokenizer
  files copied byte for byte. `manifest.json` lists the path, size and SHA-256 of every checkpoint
  file. `merged_revision` is the first 40 hexadecimal digits of the SHA-256 of that list serialized
  with sorted keys and no spaces, so it depends only on the files. `provenance.json` records the
  input hashes, the formula, the scale, the dtypes, the pairs per target, the mapping of layers 0
  and 3, the elapsed time, the peak resident memory and the library versions.
- `reference` first checks that PEFT holds every adapter pair of the file with the scale 2.0 in eval
  mode. It renders and tokenizes every row of F, rejects rows longer than 8,192 tokens without a
  forward pass, and computes the head logits and the probabilities at temperature 1 and at the
  calibrated temperature with the adapter (stage 1) and inside `disable_adapter()` (stage 0), where
  every LoRA layer must report itself disabled.
- `reference_merged` loads the checkpoint as `Qwen3_5ForCausalLM` with
  `config=Qwen3_5Config.from_pretrained(checkpoint).text_config`. Transformers 5.16.1 removes the
  `language_model` prefix for `qwen3_5_text`, so no `key_mapping` is given; the applied conversion
  rules are recorded, and missing, unexpected and mismatched keys must be empty before any row runs.
  It runs the same rows (stage 1') and also writes `hidden.safetensors`: `hidden` [302, 5120]
  float32 in the order of F, zero for rejected rows, and `mask` [302] uint8 that is 1 for rows with
  a forward pass. Its metadata holds `merged_revision` and the row IDs (`ids`).
- `verify_merge` recomputes every merged projection from the source tensors and the adapter, and
  compares all 851 tensors bit for bit, their dtypes, the index, the shard headers, `config.json`,
  the tokenizer files and the manifest. It requires both references to have passed and their rows
  to come in the same order with the same ID, kind, options, band, hashes, token count, tokens and
  rejection, and passes only when the largest probability difference at temperature 1 is at most
  0.01 on every compared row. `comparison.json` (`typed-decision-merge/1`)
  holds the checks and the statistics by kind and band; `reference.json` joins stages 0, 1 and 1';
  `report.md` compares the card examples with the card values, which is not a pass criterion.
  `python3 -m Tools.Conversion.summary <report.json> --comparison <comparison.json>` prints the
  shareable numbers.

## Conversion, quantized references and bundle

These steps run before `verify_merge` for now. `verify_merge` fails its output check (stage 1
against stage 1' exceeds 0.01) while that result awaits a decision, and the runner stops at the
first failed step, so it stays last until the decision is made. A run therefore ends with status
`failed` after every conversion step has completed; check the steps in `report.json`. The merged
checkpoint is used as it is, as a limited exploration.

- `export_dry_run` runs in the converter environment. Under a local Hugging Face cache that maps
  `merged_model_id` at `merged_revision` to `$merge/checkpoint`, it runs the official
  `coreai_models.llm.export --dry-run` (the merged ID has no registry preset, so it passes
  `--experimental` and the recipe file through `--compression-config`) and loads the backbone
  with `from_hf_memory_efficient`. The `qwen3_5` registry entry is swapped in the same process
  for `TypedDecisionBackbone`, a subclass of the official `Qwen3_5ForCausalLM` without a prefill
  graph or `lm_head` whose `main` returns the final hidden states. `dry-run.log` and
  `dry-run.json` record the resolution, the fused projections (`in_proj_all`, `qkv_gate_proj`),
  that no parameter stayed unloaded, and the export contract.
- `export` calls the official `export_model` with `qwen3_5_cfg` (uint4, per-block 32, minval),
  bfloat16 compute and a 262,144-token context. Right after quantization it hashes every tensor
  of the quantized model (`quantized-hashes.json`). A thread samples the process RSS, swap, the
  exporter's temporary folder, free disk and the memory free percentage every 5 seconds
  (`monitor.tsv`). When swap above 16 GB rises on three samples in a row or the free percentage
  stays below 8 for three samples, it records the reason, interrupts the export and kills the
  process if it is still running 120 seconds later; the exporter runs inside an event loop that
  defers the interrupt to its next await. After the export it replaces the resaved tokenizer
  with byte copies, sets `source.revision` in `metadata.json` to the merged revision, adds the
  head, `LICENSE`, `NOTICE.md`, `decision.json` and `SHA256SUMS`, validates `decision.json`
  against the files, and writes `inspect.json` (function, input, output and state descriptors
  read from the asset summary without specializing it) and `provenance.json` (durations, memory
  and disk extremes, sizes, revisions, the Apple toolchain and library versions).
- `probe_merged` and `probe_base` rebuild the quantized model in their own processes through the
  exporter's `resolve_model` and `build_model`, hash it before any cast, and run F and Q in
  float32 with float32 zero states. `probe_merged` stops unless every hash equals the export's;
  `probe_base` loads `autotrust/JEV-27B` itself through a `qwen3_5_text` entry without the
  `text_config` unwrapping and key prefix, and keeps its own hashes. Each first estimates the
  CPU time with a quadratic fit to rows of about 0, 512, 2,048 and 4,096 tokens; the eager
  linear-attention update grows faster than linearly with the input length, so a fit to
  shorter rows underestimates the longest band. When the estimate exceeds six hours, it switches
  to MPS only if three short rows agree within 1e-3 between CPU and MPS and the MPS estimate
  fits; otherwise it stops before running the sets.
- `export_gdnfp32` exports the same merged backbone as `jev27b_s1_gdnfp32_qwen3_5_cfg_dynamic` with
  `TypedDecisionBackboneGdnFp32` (`backbone_gdnfp32.py`, through `export_variant.py`). In each linear-attention layer the
  six inputs of the composite `gated_delta_update` operation are cast to float32 and its two outputs are cast back; the
  original module stays a child of the wrapper, so the exporter externalizes it as before. The step passes only when
  every quantized tensor hashes as in `export` (`decision.compare_hashes`), all 48 layers are wrapped, every
  `gated_delta_update` call in the IR of the asset takes six float32 inputs and the inspection matches. `variant.json`
  records these checks with the IR counts of both assets, and the Core AI steps below use this bundle. The steps
  before it and their modules are unchanged and stay reused; the Core AI steps depend on this step, so a new variant
  makes them and every step after them run again, including `calibrate` and the Japanese steps.
- `verify_quantization` requires matching hashes, complete rows equal to the fixtures and to the
  earlier references, and finite probabilities. It writes `reference_q.json`, `quantization.json`
  (`typed-decision-quantization/1`) and `report.md`; it applies no numeric threshold.

`verbalizers` and `verbalizer_ids` in `decision.json` are copied from `judge_config.json`, which
defines both. The `bundle` variable and the literal output of the `export` step name the same
folder, as do `bundle_gdnfp32`, the first output of `export_gdnfp32` and the third output of `calibrate`; change
each group together.

`python3 -m Tools.Conversion.adapters.jev27b.summary <report.json> --comparison <quantization.json>`
prints the shareable numbers of a run and its quantization check. It lives beside the steps rather
than in `Tools/Conversion/summary.py` because every `Tools/Conversion/*.py` file is part of each
step's cache key: editing that file would make the next run repeat every step, including the
merge, both references, the export and both probes.

## Core AI run and fidelity

These steps also run before `verify_merge`:

- `typed_input` checks the corpus and JGLUE integrity records, restores the text of F through `load_rows`, renders
  and tokenizes every row with the distributed tokenizer, and stops unless the token counts and bands equal the
  fixture. It writes `input-F.json` (`{"rows": [{"id", "kind", "state", "question", "options"}]}`) and `check.json`
  (rows, rejected rows and the SHA-256 of each input file). The text stays in the local work directory.
- `typed_decide_f` runs `coreai-verify typed-decide` with the variant bundle of `export_gdnfp32` on `input-F.json` and
  writes `swift-F.json` ([docs/jev27b.md](../../../../docs/jev27b.md#running-on-mac)). It loads the model, so it is
  heavy.
- `compare_f` scores `swift-F.json` against stage 2 of `reference_q.json`: rows, token IDs and the rejected row
  first, then finite distributions, then at temperature 1 a largest probability difference of at most 0.03 and a mean
  of at most 0.01 in every input band with rows. Bands without rows are listed as unevaluated, top-1 changes are
  reported with the 0.06 rule, and the shipping-temperature statistics are recomputed from the logits and reported
  separately. `comparison.json` (`typed-decision-fidelity/1`) and `report.md` hold the result.

## Quality and calibration

These steps also run before `verify_merge`. The conditions and definitions are fixed in
[docs/jev27b.md](../../../../docs/jev27b.md#quality-and-calibration); no threshold is set from the results.

- `reference_q` and `reference_merged_q` run `reference_sets.py --sets Q` with `--stage peft` (stages 1 and 0, the
  adapter disabled for stage 0) and `--stage merged` (stage 1'). The module repeats the model loading and checks of
  `reference.py` and `reference_merged.py` and imports their head, forward and header functions; both modules stay
  unchanged, so `reference` and `reference_merged` keep their cache keys. Each writes `reference-Q.json` or
  `reference-merged-Q.json` (no hidden states) and passes only with the same checks as the step it mirrors.
- `typed_input_eval` writes `input-Q.json`, `input-Cal.json` and `input-H.json`; `typed_decide_q`, `typed_decide_cal`
  and `typed_decide_h` run `coreai-verify typed-decide` on them with the variant bundle of `export_gdnfp32`.
- `score_q` scores stages 0, 0q, 1', 2 and 3 of Q against the teacher distribution and writes `quality.json`
  (`typed-decision-quality/1`) and `report.md`. Its check is `recorded`: inconsistent or non-finite inputs fail the
  step, and a quality condition that is not met is recorded without failing it.
- `calibrate` refits the temperature of each kind on the stage 3 logits of Cal, selects it on H without the rows whose
  state occurs in Cal, writes a calibrated copy of the variant bundle (an APFS clone with `calibration-refit.json`, an
  updated `decision.json` and `SHA256SUMS`), and re-evaluates the fidelity of F from the saved logits at the adopted
  temperatures (`comparison-refit.json`). Its checks are `recorded` in `calibration.json` and the `passed` of
  `comparison-refit.json`.

## Japanese

These steps also run before `verify_merge`. The definitions are fixed in
[docs/jev27b.md](../../../../docs/jev27b.md#japanese-jglue); no pass line is set.

- `jdev_fixture` writes `jdev.json`: 100 rows per JGLUE task outside J and F that share no sentence or question with
  them, with their IDs and SHA-256 only.
- `typed_input_jdev` and `typed_input_j` wrap the rows with the two variants of `wrappers.json` (`ja` and `en`) and
  write `input-Jdev.json` / `input-J.json` and the text-free manifests `rows-Jdev.json` / `rows-J.json`.
  `typed_input_j` also writes the three permuted orders of the adopted JCommonsenseQA variant and refuses an adoption
  that is not frozen.
- `typed_decide_jdev` and `typed_decide_j` run `coreai-verify typed-decide` with the calibrated bundle of `calibrate`
  (`$calibrate/$bundle_gdnfp32`).
- `prompt_adoption` writes `prompt-adoption.json`; its check is `frozen`.
- `reference_j` runs the wrapped rows of J through stage 1' with the loading, checks and forward of
  `reference_merged_q` and writes `reference-merged-J.json`.
- `score_j` writes `japanese.json` (`typed-decision-japanese/1`) and `report.md`; its check is `recorded`.

`wrappers.json` holds the instruction text of both variants. Changing it makes every step from `typed_input_jdev` on
run again; changing `japanese.py` or a module it imports also `jdev_fixture`. The three heavy steps among them took
about 38 minutes.

## Tokenizer

The distributed `tokenizer.json` is used as is and is never saved again. It is the resaved
form: the pre-tokenizer pattern has no `\p{M}`, the ByteLevel pre-tokenizer trims offsets and
the ByteLevel decoder sets all three options. Its vocabulary, merges, added tokens and
normalizer equal those of `Qwen/Qwen3.8-27B`, and `chat_template.jinja` is identical. The
`pretokenize_regex` in its `tokenizer_config.json` still contains `\p{M}` and therefore differs
from its own `tokenizer.json`. `AutoTokenizer` of transformers 5.16.1 loads the file
unchanged; `prepare`, the fixtures and the load check stop if the tokenizer it builds differs
from the file. Token identifiers in Swift are compared against the same file.

## Load check

Loading the model is a heavy job. Hold the shared machine lock and run with the reference
environment and the offline Hugging Face settings of `recipe.json`:

~~~sh
python -m Tools.Conversion.adapters.jev27b.load_check --model Tools/Conversion/recipes/jev27b/model.json \
  --snapshot <JEV-27B folder> --corpus <corpus folder> --jglue <JGLUE folder> \
  --fixtures Tools/Conversion/recipes/jev27b/fixtures --output <new folder>
~~~

It loads the backbone in bfloat16 on MPS, applies the adapter with PEFT in eval mode, compares
every adapter tensor in the model with the file, and runs the two card examples and one row of
each kind through `last_hidden` and the fp32 head. `load-check.json` records the consumed LoRA
pairs, the probabilities with and without the calibrated temperature, the difference from the
card's example values, timings and the peak resident memory.

## Terms

JEV-27B and `Qwen/Qwen3.8-27B` are Apache-2.0. The JEV corpus is Apache-2.0 and asks for
credit in derived work. JGLUE is CC BY-SA 4.0. The fixtures keep identifiers and hashes only.
