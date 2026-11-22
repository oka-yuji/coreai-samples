# Gemma 4 31B BF16 reference record

This record copies the numbers of two runs of the [`gemma4_31b` recipe](../../../Tools/Conversion/recipes/gemma4_31b/README.md)
(#660): the outputs of the original Transformers implementation of
[google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) with its distributed BF16 weights, BF16 compute
and MPS, for the twelve self-written diagnostic inputs of the recipe. Later steps (the FP32 reference and the Core AI
comparison) read the local outputs of these runs. This is not a quality evaluation, and it does not measure Core AI.

[`reference-20261122.json`](reference-20261122.json) holds the states, numbers, hashes, versions and case IDs of both
runs and of their comparison, with the SHA-256 of the two `reference.json` files that the comparison was made from;
`summary` checked that they are these runs' files in this order and that each report is the run that wrote its
`reference.json`. It has no fixture text, local paths, commands or environment variables. The run outputs (the
`ready.json` and `reference.json` files, the token files and the saved logits) stay in the local work directory.

## Conditions

- Mac16,6 (Apple M4 Max, 16 cores) with 128 GiB of memory, macOS 27.0 (26A428), Xcode 27.0 (27A266a), on AC power
  at the start and end of both runs (`pmset -g batt`; the power log has no change of power source between 14:19 and
  the end of the runs). Each run held the shared machine lock from its input fingerprints to its last check, and no
  model process ran when either run started (`pgrep -x`).
- Model: revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, all 12 files (62,578,686,256 B). Weight shards
  `model-00001-of-00002.safetensors` (49,784,788,364 B, SHA-256
  `eeef8791537bc04f110967c513149e037d2a9ae97d49add7291ebfa62806bbfa`) and `model-00002-of-00002.safetensors`
  (12,761,549,884 B, `018912220f559f7025d60333e0996183cd538aa77ad6f4988a89ce47be681f10`).
- Contract `model.json` SHA-256 `8729e7d4e1d4c9dcc8616bc968e9292f4fa35b952567c2a220aa3c8321ab9531`; fixtures
  `a86c373c60bc9f86df95b2d94c9f068b34d536577109993194599c282d67e75e`.
- Environment: Python 3.11.9, torch 2.15.0, transformers 5.16.1, accelerate 1.15.0, tokenizers 0.23.3, numpy 2.4.6,
  safetensors 0.8.0, huggingface-hub 1.33.0, jinja2 3.1.6.
- Runs (2026-11-22 JST, run IDs in UTC): `20261122T055330-7fdbbddb` from 14:53:26 and `20261122T060138-0c84f8e9`
  (`--refresh`, a new process for each step) from 15:01:36. Both ran commit db735c5 with no local changes.
- The twelve runs before the code changes stay in the local work directory: `20261121T181402-3993e870` and
  `20261121T182226-ec24068e` saved only the scoring rows; `20261121T203620-82b17299` and `20261121T204316-1480d82e`
  ran before the checks of the loaded files, the first monitor sample and the second greedy run were added;
  `20261121T214327-72fc48b5` and `20261121T215114-6e010de9` ran before the check of the snapshot folder right before
  loading; `20261121T224129-9bb9ad31` and `20261121T224932-a8ee45b6` ran before a monitor sample without the resident
  memory counted as unreadable; `20261122T031132-0fabe9ce` and `20261122T032110-11b33672` ran before a monitor stop
  after the model work also wrote `failure.json`; and `20261122T044544-12544e0c` and `20261122T050447-2052ef9f` ran
  before a monitor stop while the monitor starts also wrote it. Their tokens and their logits (the scoring rows of
  the first two, every saved row of the other ten) are identical to these runs; the comparison of the results below
  also passes between `20261122T044544-12544e0c` and the first of these runs and between `20261122T050447-2052ef9f`
  and the second, with a largest difference of 0.0.

## What the runs checked

- `prepare` hashed all 12 files again and found the sizes and SHA-256 of the contract and of the integrity record of
  the download. `config.json`, `generation_config.json` and the special tokens matched the contract (stop tokens
  1 `<eos>`, 106 `<turn|>` and 50 `<|tool_response>`; BOS 2), and the tokenizer that `AutoTokenizer` builds
  (`GemmaTokenizer`) equalled the distributed `tokenizer.json`.
- `reference` started only after the monitor's first sample read the resident memory, swap and the free percentage,
  and hashed the files it loads (both weight shards, `config.json`, `generation_config.json`,
  `model.safetensors.index.json` and the three tokenizer files) right before loading them: every size and SHA-256
  equalled the contract, the snapshot folder still held exactly the 12 pinned files besides the download state under
  `.cache/`, and the weight hashes in `reference.json` come from this check. It then loaded
  `Gemma4ForConditionalGeneration` with `dtype=torch.bfloat16`, `device_map="mps"` and
  `output_loading_info=True`. The missing, unexpected and mismatched key lists and the error list were empty; all
  31,273,086,512 parameters (the tied embedding counted once) were BF16 on `mps:0`, attention was `sdpa`, and the
  output embedding was the input embedding. Inputs were text only; the vision tower was loaded but not called.
- Conversations were rendered by the distributed template with `add_generation_prompt=True` and
  `enable_thinking=False`, which ends the prompt with an empty thought channel, and tokenized with
  `add_special_tokens=False`; the template writes the single BOS. The raw case got an explicit BOS. Greedy decoding
  used `generate` with `do_sample=False`, one beam and the contract stop tokens.
- The saved logits are float32 `[input length + output length, 262144]`: every position of one forward pass of the
  input followed by the greedy output, without a cache, after the soft cap. Row `j` is the distribution of token
  `j + 1`; the scoring rows start at the last input position (`first_scoring_row`).
- The monitor stopped neither run: no swap increase above 2,000,000,000 B, no free percentage below 8 for three samples,
  no swap above 16,000,000,000 B rising for three samples, and every sample read the resident memory, swap and the
  free percentage.

## Results

Both runs passed every check of `reference.json`: clean loading, no non-finite value in any saved row or in the
step logits of either greedy run, each greedy token at the maximum of its step logits, the same tokens in a second
greedy run within the process, a clean monitor and recorded resources. The runs in new processes produced the same input and output tokens for every
case, and the saved logits of all 3,241 positions were byte-identical: the largest absolute difference is 0.0. The
comparison also found both `reference.json` files passed and from different runs; every token file and saved array
with the size and SHA-256 that its own `reference.json` records, and the recorded token counts, saved positions and
first scoring rows; every saved array float32 with the recorded shape; and no NaN or infinite value among the
compared logits.

| Case | Input tokens | Output tokens | Stop | Saved positions | Scoring positions | Non-finite (saved, steps) | Forced argmax = greedy |
|---|---:|---:|---|---:|---:|---|---:|
| `ja_question` | 26 | 64 | limit | 90 | 64 | 0, 0 | 64 |
| `ja_system` | 53 | 64 | limit | 117 | 64 | 0, 0 | 64 |
| `ja_two_turns` | 64 | 4 | `<turn\|>` | 68 | 4 | 0, 0 | 4 |
| `en_question` | 26 | 51 | `<turn\|>` | 77 | 51 | 0, 0 | 50 |
| `en_system` | 44 | 58 | `<turn\|>` | 102 | 58 | 0, 0 | 58 |
| `code_python` | 51 | 128 | limit | 179 | 128 | 0, 0 | 128 |
| `code_swift` | 99 | 128 | limit | 227 | 128 | 0, 0 | 128 |
| `json_ja` | 67 | 37 | `<turn\|>` | 104 | 37 | 0, 0 | 37 |
| `json_en` | 67 | 39 | `<turn\|>` | 106 | 39 | 0, 0 | 39 |
| `ja_long` | 2,042 | 24 | `<turn\|>` | 2,066 | 24 | 0, 0 | 24 |
| `one_word` | 14 | 12 | `<turn\|>` | 26 | 12 | 0, 0 | 12 |
| `raw_text` | 15 | 64 | limit | 79 | 64 | 0, 0 | 64 |

The table holds the same values for both runs. The last column compares the argmax of the saved logits on the
scoring rows (one forward pass of input and output without a cache) with the greedy tokens (`generate` with its
cache). They differ at one of 673 positions: position 37 of `en_question`, where the two top candidates of the saved
logits differ by 0.124 in probability. The record keeps this as a diagnostic; it is not a criterion.

On one scoring row, position 100 of `code_swift`, the two top logits of the saved logits are exactly equal. The
argmax there matches the greedy token only because both pick the smaller ID; another implementation that breaks
the tie differently would differ at that position.

The two paths of the same implementation also differ in logit values. For the top candidate of each step, the median
absolute difference is 0 in ten of the eleven conversation cases and 0.125 in `one_word`; the maximum is 0.375 or less
except in `json_ja` (0.875) and `code_swift` (3.25). Without the template, `raw_text` has a median of 0.75 and a
maximum of 7.25; over the whole vocabulary its largest difference is 23.25. Of the per-step logits of `generate`,
only the top two candidates of each step are kept in the token files.

## Resources and time

| Run | `20261122T055330-7fdbbddb` | `20261122T060138-0c84f8e9` |
|---|---:|---:|
| Pipeline | 484.2 s | 447.8 s |
| `prepare` step (hashing all files) | 83.9 s | 68.9 s |
| `reference` step | 396.6 s | 377.6 s |
| Hashing the loaded files before loading | 23.46 s | 21.41 s |
| Model load | 6.49 s | 5.78 s |
| Greedy, 673 tokens of 12 cases (with prefill) | 141.7 s, 4.75 tokens/s | 129.4 s, 5.20 tokens/s |
| Decode after the first token, short inputs | 5.59 to 6.40 tokens/s | 6.02 to 6.54 tokens/s |
| `ja_long`: first token, decode | 15.96 s, 4.29 tokens/s | 12.83 s, 4.67 tokens/s |
| Peak resident memory | 54,880,616,448 B | 56,097,046,528 B |
| Kernel lifetime footprint peak | 79,953,904,672 B | 79,961,146,424 B |
| MPS driver allocation, after load / peak | 62,556,012,544 B / 74,648,731,648 B | 62,556,012,544 B / 75,100,061,696 B |
| Swap increase (limit 2,000,000,000 B) | 0 B | 0 B |
| Lowest memory free percentage | 39 | 39 |
| Saved logits (12 files) | 3,398,436,352 B | 3,398,436,352 B |

- The model loaded with the weights in the file cache: `prepare` had just read every file. These are not cold loads.
- Swift builds and tests outside these runs ran on the same Mac during both runs. Each run started only when none of
  `xcodebuild`, `swift-build`, `swift-frontend` and `swift-test` was running, after waiting for them to end. These
  processes were sampled every 5 seconds during the runs: at least one of them was running in 67 of 95 samples of
  the first run and in 15 of 88 samples of the second. Right after the first run ended none of them was running;
  right after the second, `swift-test` was. The times and rates of both runs are values under that condition.
- The decode rates exclude the first token and the cases with four or fewer output tokens. Each step synchronizes MPS
  once to take its time. The first case of each process also includes the first use of the MPS kernels.
- `torch.mps.current_allocated_memory()` reported 3,840 B after loading, so it does not count the loaded weights;
  the table uses the driver allocation. Keeping every position raises the MPS peak above the first two earlier runs,
  which saved only the scoring rows (72.4 GB): the float32 copy of the longest case alone is 2,166,358,016 B.
- Peak resident memory and the kernel lifetime footprint peak count different things and are not compared with each
  other; the resident peak also depends on how much of the weight files is mapped at the time.

## Limits

- Twelve self-written diagnostic inputs. Five cases stopped at their output limit (64 or 128 tokens). They are not a
  quality set, and they do not cover images, audio, tools or thinking.
- The greedy output of `raw_text` (a continuation without the template) is one token, ID 86159, repeated 64 times.
  Its numeric differences between the cached and the single-pass paths are the largest of the twelve cases, so later
  comparisons treat it separately from the conversation cases.
- One machine, one OS build and one software environment. The same outputs in two processes do not show that another
  machine, OS or library version gives the same outputs.
- BF16 compute on MPS only. The FP32 reference, the Core AI conversion and the comparison with them are separate
  steps.

## Sources

| File | SHA-256 |
|---|---|
| Report of the first run | `e3935fcc0f0791bf93bcd6011283198246f45ec8a3bf51f387bae10d7ed0d540` |
| Report of the second run | `d254c3a65a7c1c6b63cc4dabfccbf368fdb3b0630ce11be3333062a8094d3ead` |
| `reference.json` of the first run | `e391dc30a64a360674b25f0b2c5c9046f41130255485e0113476bd27bfea167b` |
| `reference.json` of the second run | `6c015255df4ee88788b7d4f50246841de139a0d1f93b4719e8435e0ca4264873` |
| Comparison of the two runs | `b63fbd4f9cda719c2b5e3d984820a6d6b061e814ae2b251ec561eba808ba2c33` |

~~~sh
python3 -m Tools.Conversion.adapters.gemma4_31b.summary \
  --report <first report.json> --reference <first reference.json> \
  --report <second report.json> --reference <second reference.json> \
  --comparison <comparison.json> > docs/validation/gemma4-31b/reference-20261122.json
~~~

## FP16 export of the first six layers

This section copies the numbers of the FP16 export of the first six layers (five sliding-window layers and one
full-attention layer) of the
[`recipe-fp16.json`](../../../Tools/Conversion/recipes/gemma4_31b/README.md#fp16-export-of-the-first-layers) recipe to
two dynamic-shape Core AI bundles for macOS, one with the functions `main` and `prefill` and one with `main` only, and
of the comparison of the logits that the own engine reads from them (`coreai-verify logits`) with an FP32
reference of the same six layers (#680). Six layers do not make a usable model: the record shows that the export, the
asset and the reads of the engine are right for these layers, not the quality of the model.
[`fp16-first-layers-20261122.json`](fp16-first-layers-20261122.json) holds the states, numbers, hashes, versions and
fixed names of the records, from `fp16_summary`, which checked that each pipeline report is the run that wrote its
`export.json` and that each comparison was made from that `export.json` and the reference record. It has no local
paths, commands or environment variables; the records, bundles and logits stay in the local work directory.

### Conditions

- The Mac of the runs above (Apple M4 Max, 128 GiB, macOS 27.0 (26A428), Xcode 27.0 (27A266a), on AC power). Every
  model job held the shared machine lock, and none started while another model process ran. Other sessions ran their
  own model jobs between these runs under the same lock; the times below are records, not measurements.
- Commit aa2d35d with no local changes for every model job. Converter `apple/coreai-models`
  `c8e00c01800ba6ad324707007f7af358efc980a4` with coreai-core 1.0.0b3, coreai-models 0.1.0, coreai-opt 0.4.0,
  coreai-torch 0.4.3, torch 2.9.0, transformers 5.12.1 and safetensors 0.7.0. Reference: transformers 5.16.1,
  torch 2.15.0, numpy 2.4.6, safetensors 0.8.0. `coreai-verify` was built in release from the same commit.
- Runs (2026-11-22 JST, pipeline run IDs in UTC): the FP32 reference (`layers.py context`) from 21:39:32; the exports
  `20261122T124049-9690b4a1` (`main` and `prefill`) from 21:40:49 and `20261122T124401-25ab007b` (`main` only) from
  21:44:01, each with the `prepare` step of the recipe; the comparisons from 22:18:24 and from 22:52:56.
- Inputs: `short` (26 tokens) and `context` (16,896 tokens), fixed-seed token IDs of the recipe's FP32 tools, not
  text.

### Exports

| | `main` and `prefill` | `main` only |
|---|---:|---:|
| Pipeline | 194.1 s | 170.7 s |
| `prepare` step | 25.6 s | 25.1 s |
| `export` step | 73.8 s | 52.2 s |
| Export tool | 69.92 s | 49.80 s |
| Hashing the files to load | 21.36 s | 21.41 s |
| Loading the model class in FP16 | 3.99 s | 2.58 s |
| `export_macos_model` | 33.05 s | 16.92 s |
| Saving the asset | 7.66 s | 5.27 s |
| Hashing the bundle | 3.01 s | 3.02 s |
| Peak resident memory | 23,346,135,040 B | 23,208,198,144 B |
| Kernel lifetime footprint peak | 9,482,581,864 B | 14,444,692,880 B |
| Swap increase (limit 2,000,000,000 B) | 0 B | 0 B |
| Free disk required at the start | 81,776,897,048 B | 81,776,897,048 B |
| Lowest free disk during the run (limit 64,424,509,440 B) | 475,974,053,888 B | 475,676,606,464 B |
| `main.mlirb` | 8,676,277,835 B | 8,676,248,377 B |
| `main.hash`, the SHA-256 of `main.mlirb` | `e0687fe49f73970361cdac5c283fa75d0cf4d08c9affb0f030e6f3495c4b1993` | `f3d5f8ce716a841ebf480ce457ee0034763e29fb3de8d7771458136b0c5761b0` |
| Bundle | 8,708,469,929 B | 8,708,440,455 B |

Both assets passed the inspection. No compression setting was given and the PyTorch model had no quantized weight; its
85 parameters and buffers held 4,338,096,902 float16 elements. Both assets store Float16 4,338,096,906 elements and
the constants Float32 386, Int32 148, UInt32 29 and UInt64 1 (the same as the six-layer `slicepos` assets of #665),
and no other type. The functions are those asked for, with `input_ids` and `position_ids` int32 [1, ?], `logits` float16
[1, ?, 262144] from `main`, `keyCache` and `valueCache` float16 [1, 1, 4, ?, 512] and `slidingKeyCache` and
`slidingValueCache` float16 [5, 1, 16, ?, 256]. The three tokenizer files are the distributed files, and
`metadata.json` has vocabulary 262,144, context limit 16,896, `source.hf_model_id` `google/gemma-4-31B-it` and
`source.revision` `842da3794eaa0b77d5f08bae87a17459d91ff475`. `export.json` records the size and SHA-256 of every
bundle file. The free disk check counts twice the 8,676,193,804 B of FP16 weights plus 60 GiB.

### FP32 reference

`layers.py context` took 70.2 s: 23.07 s to hash the files to load, 2.23 s to load the first six layers of the
original implementation, 0.2 s for `short` in one call and 44.1 s for `context` in 17 calls of up to 1,024 tokens with
the cache of the implementation. It kept 26 rows of `short` (27,263,104 B) and 1,428 rows of `context`
(1,497,366,656 B), all finite. Peak resident memory 55,567,876,096 B, kernel lifetime footprint peak
55,537,901,360 B.

### Comparison

Both assets passed all twelve conditions: no non-finite value in any written or reference row, no constant written
row, every largest absolute logit difference below 2.0, and no top-1 change outside a near tie. Every one of
the twelve logits files of the asset with `main` only is byte-identical to the one of the asset with `main` and
`prefill`, also where the tokens before `--start` went through `main` instead of the `prefill` function, so the table
holds for both.

| Condition | Rows | Largest absolute logit difference | Top-1 changes in a near tie / outside | Rows of the known call shape: count, largest difference | Seconds (`main` and `prefill` / `main` only) |
|---|---:|---:|---:|---:|---:|
| `short-c1` | 26 | 0.458 | 25 / 0 | – | 15.9 / 11.1 |
| `short-c26` | 26 | 0.480 | 25 / 0 | – | 3.4 / 2.8 |
| `window-c1` | 1,300 | 0.989 | 1,266 / 0 | – | 35.1 / 33.8 |
| `window-c2` | 1,300 | 0.929 | 1,266 / 0 | – | 19.6 / 18.8 |
| `window-c4` | 1,300 | 0.855 | 1,266 / 0 | – | 11.4 / 10.4 |
| `window-c8` | 1,300 | 0.861 | 1,250 / 0 | 16, 4.266 | 8.3 / 7.6 |
| `prefix-c1` | 300 | 0.836 | 294 / 0 | – | 11.6 / 10.8 |
| `capacity-2048` | 16 | 0.437 | 15 / 0 | – | 4.5 / 4.7 |
| `capacity-4096` | 16 | 0.535 | 14 / 0 | – | 5.7 / 5.7 |
| `capacity-8192` | 16 | 0.459 | 16 / 0 | – | 7.6 / 9.7 |
| `capacity-16384` | 16 | 0.567 | 15 / 0 | – | 16.7 / 24.7 |
| `tail` | 64 | 0.466 | 61 / 0 | – | 19.4 / 23.9 |

- The rows of the known call shape are rows 96 to 111 of `window-c8`, from its calls of 8 tokens that end at 104 and
  112 positions; they are left out of the judgment. The other rows of that condition stay at 0.861.
- On the asset with `prefill`, the tokens before `--start` went through the `prefill` function (1, 1, 1, 2, 4 and 5
  calls for `prefix-c1`, the four `capacity` conditions and `tail`), and no `coreai-verify` process stopped. On the
  asset with `main` only they went through `main`.
- The first six layers saturate the logits: in the reference rows of `window-c8` and `capacity-2048`, a median of
  19,423 and 25,971 of the 262,144 logits per row are at 29.98 or above in absolute value. The top two reference
  logits of nearly every row are then within 0.01, so the top-1 criterion rarely applies; the logit difference is the
  measure here, as in #665.
- The first condition of each asset specialized the asset (11.4 s and 7.6 s); the others used the specialization
  cache. The caches took 21,660,880 KiB (`resources.bin` 22,163,497,512 B) and 16,962,060 KiB (`resources.bin`
  17,352,106,248 B), the sizes that #665 found for the same forms, and were removed after the comparisons.
- Compared with §3.2 of the #665 spike, which read a text input and a reference computed with the spike's own model
  class, the differences are of the same size: `short` 0.458 and 0.480 (0.378 and 0.559), the window with calls of 1,
  2 and 4 tokens 0.855 to 0.989 (0.64 to 0.75), the known rows 4.266 (4.34), the reads after the prefix of 1,000 tokens
  0.836 (0.561), the capacity boundaries 0.437 to 0.567 (0.29 to 0.45) and the tail 0.466 (0.531).

### Limits

- Six of the 60 layers, FP16 only, one machine, one OS build and one software environment; the inputs are token IDs
  of fixed seeds, not a quality set. The export and comparison of all 60 layers and the 8-bit assets are later steps.
- Saturated logits make the top-1 criterion weak for six layers; the probability thresholds of the quality contract
  are not used.
- The rows of the call shape that `Engine` avoids for this model (calls of 8 or more tokens after which the positions
  total 97 to 112) are recorded but not judged. `coreai-verify logits` reads the `--chunk` calls as given, so
  `window-c8` reproduces that shape.

### Sources

| File | SHA-256 |
|---|---|
| FP32 reference `record.json` | `5cfe59f947933ae79eb30ea1b0be0d354dfdd24f99b8ed065ac3758821b72141` |
| Report of the export with `main` and `prefill` | `e4d86209735c008e0e241747df2db51a0cb63f76fa5f32cd59335d707284d44f` |
| Its `export.json` | `d40d72040ccf0a2a05be0cb1583065f53e7735e334145aa6ef581388a3b6356e` |
| Its `comparison.json` | `cc532deab90dfcbf97f989e157e7cd18680e68e73f5cf700f5a13507549758a6` |
| Report of the export with `main` only | `b231e0f1a66e1ed32afc6cc6685018bbce29fda54ae505a91697572a0ea1dda0` |
| Its `export.json` | `a7f9b48d7b9c25244984170a077614ff23a4956c655126961d375a7617aac4d5` |
| Its `comparison.json` | `0cffe4a22ad63a9b12501e7c9242adc36627d6b7483cee10f951e2207a46e0ec` |

~~~sh
python3 -m Tools.Conversion.adapters.gemma4_31b.fp16_summary --reference <FP32 reference record.json> \
  --report <report.json> --export <export.json> --comparison <comparison.json> \
  --report <report.json> --export <export.json> --comparison <comparison.json> \
  > docs/validation/gemma4-31b/fp16-first-layers-20261122.json
~~~
