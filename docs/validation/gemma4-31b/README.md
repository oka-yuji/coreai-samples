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


## FP32 anchor and FP16 numerics (#662)

This section records one run of the six steps that #662 added to the recipe
([FP32 anchor and FP16 diagnostics](../../../Tools/Conversion/recipes/gemma4_31b/README.md#fp32-anchor-and-fp16-diagnostics))
and five comparisons made from its records with `anchor_compare`. [`anchor-20261122.json`](anchor-20261122.json) holds
the states, numbers, hashes and case IDs that `anchor_summary` printed from that run's report, its seven records and the
five comparisons. It has no fixture text, local paths, commands or environment variables. It uses the twelve diagnostic
inputs above; it is not a quality evaluation and does not measure Core AI.

### Conditions

- The machine, OS, Xcode and environment of the sections above, on AC power at the start and end of the run. The run
  held the shared machine lock from its input fingerprints to its last check, and no model process ran when it started.
- Run `20261122T104334-a76752c5`, 2026-11-22 19:43:34 to 20:03:47 JST (1,215.7 s), commit 4415504 with no local
  changes. The input files of the steps have not changed since commit 3e82bd4.
- The `reference` step ran again: the cache key of a step includes the path of the checkout, the environment and the
  step's input files, so the objects of the runs above were not reused. Its tokens and all 3,241 saved rows equal the
  second run above byte for byte (`compare`: passed, largest difference 0.0). The comparisons use this run's record
  (SHA-256 `4cd3b9783f7bab6cf088495ad4cdae99829684021ae0a7f2775f5e629f0b6154`) as A.
- The four FP32 steps also ran in an earlier run (`20261122T091730-364d7a60`, commit d32bdbb): all 48 saved logit
  arrays of these steps have the same SHA-256 in both runs. The `half` step of that run was stopped (see Limits).
- Build and test processes of other work were sampled every 5 seconds: `xcodebuild`, `swift-build`, `swift-frontend` or
  `swift-test` ran in 66 of 237 samples. The times below are values under that condition.

| Record | Weights | Compute | Embedding scale |
|---|---|---|---|
| `reference` (A) | BF16 | BF16, the original forward | 73.5 |
| `anchor` (A32) | BF16, replaced by FP32 values for each unit while it runs | FP32, attention in blocks of 512 query rows | 73.5 |
| `anchor_w16` (A32(W16)) | BF16 cast to float16, then FP32 for each unit | FP32, as `anchor` | 73.5 |
| `anchor_scale_fp32` | BF16 | FP32, as `anchor` | 73.32121 |
| `anchor_scale_fp16` | BF16 | FP32, as `anchor` | 73.3125 |
| `half` (FP16) | BF16 cast to float16 | float16, the original forward | 73.5 |

The comparisons are written baseline → candidate: A32 → A, A32 → A32(W16), A32(W16) → FP16 and A32 → each scale. Every
comparison passed its checks (records, files, tokens, shapes) and found no NaN or infinite logit. A row's error is the
largest absolute difference of the two probability distributions; where the top-1 changes, the gap is the probability
that the baseline gives its own top-1 minus the probability it gives the candidate's top-1.

### Results

**Layer-wise FP32 against the plain FP32 forward (first six layers).** With layers 0 to 5 of the real weights (five
sliding-window layers and one full-attention layer; the 747 keys of layers 6 to 59 were the only unexpected keys), the
largest absolute logit difference between the layer-wise computation and `model.float()` was 0.0 for the eleven inputs
of up to 227 positions and 7.53e-5 for `ja_long` (2,066 positions, beyond the window of 1,024); limit 1e-4. Neither
path produced a non-finite value.

**A32 → A, the deviation of the BF16 reference from its FP32 anchor.** Rows are the saved positions; the scoring rows
are those of the reference record; rows 992 to 1,056 of `ja_long` span the position where the BOS leaves the sliding
window.

| Rows | Scope | Count | Max of max \|p − q\| | Mean of max \|p − q\| | KL mean | KL p99 | KL max | Max \|logit diff\| | Top-1 equal | Near-tie mismatches | Other mismatches |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conversation, 1–512 input tokens (10 cases) | every row | 1,096 | 0.999 | 0.011 | 0.0569 | 0.0393 | 38.2 | 30.3 | 1,079 | 2 | 15 |
| conversation, 1–512 input tokens (10 cases) | scoring rows | 585 | 0.0886 | 0.00361 | 4.48e-04 | 0.0111 | 0.0212 | 9.7 | 583 | 0 | 2 |
| `ja_long` (2,042 input tokens) | every row | 2,066 | 1 | 0.108 | 0.521 | 10.4 | 23.9 | 31.7 | 1,787 | 2 | 277 |
| `ja_long` (2,042 input tokens) | scoring rows | 24 | 0.0127 | 6.07e-04 | 5.83e-05 | 0.00105 | 0.00136 | 0.703 | 24 | 0 | 0 |
| `raw_text` | every row | 79 | 0.0874 | 0.00487 | 0.00332 | 0.0424 | 0.0901 | 18.3 | 79 | 0 | 0 |
| `raw_text` | scoring rows | 64 | 0.0326 | 0.00146 | 0.00232 | 0.041 | 0.0901 | 18.3 | 64 | 0 | 0 |
| `ja_long` (2,042 input tokens) | rows 992–1,056 | 65 | 0.966 | 0.0695 | 0.181 | 2.39 | 3.56 | 14.8 | 60 | 0 | 5 |

- On the scoring rows the largest error is 0.0886 (conversation cases), 0.0127 (`ja_long`) and 0.0326 (`raw_text`),
  with band means of 0.0036, 0.0006 and 0.0015. Two scoring rows have a top-1 change that is not a near tie
  (`en_system` row 75 and `code_python` row 127, gaps 0.039 and 0.075). For comparison, the two paths of A itself
  (cached greedy decoding and the forced pass) differ at one of the 673 scoring positions (see Results above).
- Outside `ja_long`, 13 rows that are not scoring rows have a top-1 change that is not a near tie: 12 input rows and
  the row after the last token of `ja_two_turns`. Five of them are row 3: nine conversation cases start with the same
  four tokens (`<bos>`, `<|turn>`, `user` and a newline), and row 3 changes in six of them, `ja_long` included, each
  with a gap of 0.024. The two largest gaps are rows 6 and 9 of `ja_two_turns` (0.894 and 0.634).
- `ja_long` (30 paragraphs from one template that differ only in a number, a place, a colour and an item, then a
  question), facts: 482 of its 2,066 rows have an error above 0.1 and 166 above 0.5; every top-1 change is on an input
  row. Counted by the paragraph whose token the row predicts, the rows above 0.1 number 0 in the first paragraph, 1 in
  the second and 4 to 30 in each of the third to the thirtieth (the thirtieth includes the question); the last of the
  482 is the row after the last output token. By the decoded text of the next token, 151 of the 166 rows above 0.5
  predict a token that is neither a digit nor part of a place, colour or item value, 13 a digit and 2 part of a place
  or item value; over all 2,041 input rows these shares are 85.7 %, 7.5 % and 6.8 %.
- `ja_long`, facts: at the 166 rows the top-1 of A32 is the next forced token in 59, the top-1 of A in 57 and neither
  in 56; over all 2,041 input rows, the top-1 is the next forced token in 1,468 rows for A32 and 1,465 for A. The same
  rows also move under other changes: 82 of the 87 rows with an error above 0.1 when only the embedding scale of the
  FP32 computation changes by 0.24 % (73.5 → 73.32121), and 142 of the 164 rows above 0.1 between A32(W16) and FP16,
  are among the 482. Neither of these two changes moves a row of the first two paragraphs above 0.1. The scoring rows
  of `ja_long` stay within 0.0127.
- `ja_long`, not verified: why the repeated paragraphs make these rows sensitive. One reading is that at these rows,
  once the text has repeated, the model weighs continuing the repetition against other tokens so evenly that any small
  change of the computation flips its confident choice. Attention weights and hidden states were not recorded.

**BF16 → float16 cast of the weights.** Over all 1,188 tensors that the model loads (31,273,088,876 elements, a tied
tensor once): 20,793,006 elements change, 11,057,551 of them from nonzero to zero, the largest absolute error among
finite results is 2.98e-8, no result is NaN or infinite and no weight exceeds 65,504 in magnitude. The 832 tensors of
the text model (30,697,345,340 elements) account for 8,823,492 changes and 51,624 of the zeros; 11,000,385 zeros are in
the position embedding table of the vision tower, which text inputs do not use.

| Rows | Scope | Count | Max of max \|p − q\| | Mean of max \|p − q\| | KL mean | KL p99 | KL max | Max \|logit diff\| | Top-1 equal | Near-tie mismatches | Other mismatches |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conversation, 1–512 input tokens (10 cases) | every row | 1,096 | 4.00e-05 | 1.97e-06 | 9.21e-11 | 1.53e-09 | 7.63e-09 | 0.00134 | 1,096 | 0 | 0 |
| conversation, 1–512 input tokens (10 cases) | scoring rows | 585 | 1.63e-05 | 7.73e-07 | 1.91e-11 | 3.50e-10 | 7.13e-10 | 6.07e-04 | 585 | 0 | 0 |
| `ja_long` (2,042 input tokens) | every row | 2,066 | 0.00301 | 4.53e-05 | 1.39e-07 | 2.21e-06 | 2.05e-05 | 0.112 | 2,066 | 0 | 0 |
| `ja_long` (2,042 input tokens) | scoring rows | 24 | 1.88e-05 | 9.08e-07 | 1.11e-10 | 2.03e-09 | 2.62e-09 | 4.28e-04 | 24 | 0 | 0 |
| `raw_text` | every row | 79 | 1.59e-04 | 4.33e-06 | 1.32e-08 | 3.25e-07 | 6.03e-07 | 0.0759 | 79 | 0 | 0 |
| `raw_text` | scoring rows | 64 | 1.59e-04 | 3.76e-06 | 1.58e-08 | 3.78e-07 | 6.03e-07 | 0.0759 | 64 | 0 | 0 |
| `ja_long` (2,042 input tokens) | rows 992–1,056 | 65 | 0.00112 | 2.71e-05 | 4.62e-08 | 1.06e-06 | 2.78e-06 | 0.00913 | 65 | 0 | 0 |

**Value ranges of the FP32 computation (A32).** The largest absolute values over the twelve inputs:

| Point | Largest absolute value | Layer |
|---|---:|---|
| Residual stream leaving or entering a layer | 426.4 | leaving 21, entering 22 |
| Attention output | 435.7 | 5 |
| Input of `down_proj` (gate × up) | 2,299.1 | 15 |
| Input of the final norm | 20.2 | |
| Output of the final norm | 369.5 | |
| Logits before the soft cap | 125.8 | |

No recorded element exceeds 65,504 or 32,752 (0 of each at all 243 points), and the record of A32(W16) gives the same
largest values to within 0.01. The ten largest values are inputs of `down_proj` in layers 15, 13, 18, 19, 14, 16, 5,
11, 12 and 1 (2,299 to 1,335). The residual stream entering the layer exceeds 255.94, the square root of 65,504, in 35
of the 60 layers, and the attention output does in 8.

**FP16 compute (PyTorch, float16 weights and compute, MPS).** No saved logit of the twelve inputs is NaN or infinite.

| Rows | Scope | Count | Max of max \|p − q\| | Mean of max \|p − q\| | KL mean | KL p99 | KL max | Max \|logit diff\| | Top-1 equal | Near-tie mismatches | Other mismatches |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conversation, 1–512 input tokens (10 cases) | every row | 1,096 | 0.018 | 0.00111 | 2.62e-05 | 3.83e-04 | 0.00106 | 0.488 | 1,095 | 1 | 0 |
| conversation, 1–512 input tokens (10 cases) | scoring rows | 585 | 0.00949 | 4.97e-04 | 7.39e-06 | 1.53e-04 | 3.89e-04 | 0.318 | 585 | 0 | 0 |
| `ja_long` (2,042 input tokens) | every row | 2,066 | 1 | 0.0291 | 0.0742 | 1.11 | 18.3 | 22.7 | 1,985 | 2 | 79 |
| `ja_long` (2,042 input tokens) | scoring rows | 24 | 0.00719 | 4.17e-04 | 1.87e-05 | 3.23e-04 | 4.08e-04 | 0.242 | 24 | 0 | 0 |
| `raw_text` | every row | 79 | 0.0233 | 0.00208 | 4.21e-04 | 0.00597 | 0.0105 | 15.1 | 79 | 0 | 0 |
| `raw_text` | scoring rows | 64 | 0.0195 | 7.30e-04 | 3.16e-04 | 0.00593 | 0.0105 | 15.1 | 64 | 0 | 0 |
| `ja_long` (2,042 input tokens) | rows 992–1,056 | 65 | 0.98 | 0.035 | 0.117 | 2.99 | 5.88 | 17.5 | 63 | 0 | 2 |

On the scoring rows the largest error is 0.0095 (conversation cases), 0.0072 (`ja_long`) and 0.0195 (`raw_text`), every
top-1 is equal, and the band means are 0.0005, 0.0004 and 0.0007. Outside `ja_long` one input row (`json_en`) changes
its top-1, as a near tie; the other changes are 81 input rows of `ja_long` (79 not near ties).

**Embedding scale.** With the two other scales, the largest error of the FP32 result on the scoring rows is 0.0065
and 0.0068 (conversation cases), 0.0009 (`ja_long`) and 0.0302 and 0.0303 (`raw_text`); over all rows of `ja_long` it
reaches 0.909 and 0.920.

| Rows | Scope | Count | Max of max \|p − q\| | Mean of max \|p − q\| | KL mean | KL p99 | KL max | Max \|logit diff\| | Top-1 equal | Near-tie mismatches | Other mismatches |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conversation, 1–512 input tokens (10 cases) | every row | 1,096 | 0.00649 | 1.92e-04 | 1.29e-06 | 2.19e-05 | 1.87e-04 | 0.324 | 1,095 | 1 | 0 |
| conversation, 1–512 input tokens (10 cases) | scoring rows | 585 | 0.00649 | 1.09e-04 | 7.11e-07 | 9.55e-06 | 1.29e-04 | 0.324 | 584 | 1 | 0 |
| `ja_long` (2,042 input tokens) | every row | 2,066 | 0.909 | 0.018 | 0.0254 | 0.459 | 8.92 | 19.7 | 2,008 | 4 | 54 |
| `ja_long` (2,042 input tokens) | scoring rows | 24 | 8.61e-04 | 5.46e-05 | 2.83e-07 | 4.54e-06 | 5.55e-06 | 0.0791 | 24 | 0 | 0 |
| `raw_text` | every row | 79 | 0.0654 | 0.00446 | 0.00169 | 0.0307 | 0.0429 | 14.2 | 79 | 0 | 0 |
| `raw_text` | scoring rows | 64 | 0.0302 | 0.00103 | 8.58e-04 | 0.0178 | 0.0429 | 14.2 | 64 | 0 | 0 |
| `ja_long` (2,042 input tokens) | rows 992–1,056 | 65 | 0.594 | 0.0343 | 0.0437 | 1.04 | 1.3 | 19.7 | 63 | 0 | 2 |

| Rows | Scope | Count | Max of max \|p − q\| | Mean of max \|p − q\| | KL mean | KL p99 | KL max | Max \|logit diff\| | Top-1 equal | Near-tie mismatches | Other mismatches |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conversation, 1–512 input tokens (10 cases) | every row | 1,096 | 0.00679 | 2.02e-04 | 1.42e-06 | 2.44e-05 | 2.05e-04 | 0.342 | 1,095 | 1 | 0 |
| conversation, 1–512 input tokens (10 cases) | scoring rows | 585 | 0.00679 | 1.14e-04 | 7.81e-07 | 1.07e-05 | 1.41e-04 | 0.342 | 584 | 1 | 0 |
| `ja_long` (2,042 input tokens) | every row | 2,066 | 0.92 | 0.0186 | 0.027 | 0.497 | 10 | 19.5 | 2,010 | 4 | 52 |
| `ja_long` (2,042 input tokens) | scoring rows | 24 | 8.87e-04 | 5.50e-05 | 2.95e-07 | 4.79e-06 | 5.89e-06 | 0.0827 | 24 | 0 | 0 |
| `raw_text` | every row | 79 | 0.071 | 0.00471 | 0.0018 | 0.0331 | 0.0434 | 14 | 79 | 0 | 0 |
| `raw_text` | scoring rows | 64 | 0.0303 | 0.00106 | 8.74e-04 | 0.018 | 0.0434 | 14 | 64 | 0 | 0 |
| `ja_long` (2,042 input tokens) | rows 992–1,056 | 65 | 0.6 | 0.0354 | 0.0458 | 1.04 | 1.36 | 19.5 | 63 | 0 | 2 |

### Resources and time

| Step | Execution | Total | Peak resident memory | Kernel lifetime footprint peak | MPS driver allocation, peak | Swap increase |
|---|---:|---:|---:|---:|---:|---:|
| `prepare` | 27.3 s | 74.6 s | | | | |
| `reference` | 344.3 s | 394.8 s | 57,570,328,576 B | 79,960,605,704 B | 75,100,061,696 B | 0 B |
| `anchor` | 77.5 s | 128.2 s | 56,284,889,088 B | 82,038,636,376 B | 77,835,059,200 B | 0 B |
| `anchor_w16` | 97.0 s | 148.7 s | 55,122,460,672 B | 83,682,180,952 B | 79,483,617,280 B | 0 B |
| `anchor_scale_fp32` | 79.7 s | 132.2 s | 55,987,617,792 B | 81,932,173,168 B | 77,732,495,360 B | 0 B |
| `anchor_scale_fp16` | 79.5 s | 131.7 s | 56,049,500,160 B | 78,616,690,544 B | 74,416,898,048 B | 0 B |
| `half` | 63.1 s | 115.0 s | 54,971,056,128 B | 76,284,154,176 B | 72,597,504,000 B | 0 B |
| `layers` | 36.4 s | 87.1 s | 13,078,953,984 B | 43,426,097,784 B | 30,407,000,064 B | 0 B |

- Execution is the time of the step's command; Total, from the report, also includes the runner's work before and
  after it. Each step hashed the files it loads right before loading them (21 to 24 s of its execution), and
  `anchor_w16` spent 10.7 s on the cast statistics. The weights were in the file cache.
- `half`: loading took 10.5 s; right after loading, the MPS driver allocation was 64,131,383,296 B and the lifetime
  footprint peak 66,139,449,488 B.
- Peak resident memory and the kernel lifetime footprint peak count different things and are not compared with each
  other.

### Conclusion on FP16 numerics

FP16 numerics hold for these inputs, on the condition that the norms form their mean of squares in FP32, as the
original implementation does.

- Weights: the float16 cast causes no overflow and no non-finite value; it changes only values below the normal range
  of float16. With the cast weights, the FP32 result stays within 1.6e-4 of A32 on the scoring rows (1.9e-5 outside
  `raw_text`), with every top-1 equal.
- Activations: no recorded value reaches half of the float16 maximum; the largest, 2,299, is 3.5 % of it.
- Compute: float16 in PyTorch on MPS produced no non-finite value and stays within 0.0195 of the FP32 result with the
  same weights on every scoring row, with every top-1 equal. In each band this is closer than the BF16 reference is to
  A32 (0.0095, 0.0072 and 0.0195 against 0.0886, 0.0127 and 0.0326).
- Condition: the residual stream entering 35 of the 60 layers (up to 426) and the attention output of 8 layers (up to
  436) exceed 255.94, so the squares that the norms of these values form would exceed the float16 maximum if they were
  computed in float16. The original RMSNorm, and so this float16 run, computes them in FP32. A Core AI float16 graph
  needs the same or an equivalent computation that does not form these squares in float16; that is not checked here.
- The deviation of A from A32 is the reference's own deviation to allow for when later results are compared with A: on
  the scoring rows up to 0.0886 per position and 0.0036 on average for the conversation cases, with two top-1 changes
  that are not near ties. On the input rows of the repeated paragraphs of `ja_long`, BF16 compute, float16 compute and
  a 0.24 % change of the embedding scale move largely the same rows, so those rows show this sensitivity rather than
  the precision of an implementation.

### Limits

- Twelve self-written diagnostic inputs, one machine, one software environment, MPS only.
- The value ranges cover the listed points only: not the residual stream between the attention and the MLP, the output
  of the MLP before its norm, the inputs of the query and key norms, the attention scores (the attention scale is 1.0)
  or the squares inside the norms.
- `half` is PyTorch with the embedding scale 73.5, the weights cast from BF16 to float16 and float16 compute; these
  conditions were set before any `half` result existed. It is not a Core AI float16 asset; the Core AI asset and its
  comparison are separate steps.
- Loading with `from_pretrained(dtype=torch.float16, device_map="mps")` kept nearly two copies of the weights during the
  load: the `half` step of the earlier run reached a lifetime footprint peak of 127,517,538,720 B, swap grew by
  16,728,908,104 B and the monitor stopped it. On the first 6 and 12 layers, that load's extra (lifetime peak minus the
  allocation after loading) was 10.3 and 16.2 GB; loading on the CPU and moving to MPS reached a lifetime peak of 1.80
  times the allocation after loading on 6 layers. `half` therefore loads BF16 on MPS and casts one tensor at a time to float16:
  the extra was 3.8 and 3.2 GB on 6 and 12 layers, and the lifetime footprint peak right after the 60-layer load was
  66,139,449,488 B.

### Sources

| File | SHA-256 |
|---|---|
| Report of the run | `e09d7665c837de0749d37bbaf11e4b33bf2e8af9059f3eced1805dde82808ea4` |
| `reference.json` of the run (A) | `4cd3b9783f7bab6cf088495ad4cdae99829684021ae0a7f2775f5e629f0b6154` |
| `layers.json` | `9078f063d7c74161d8767fc16596873549d76e45cce196e568404b338ebcfdd3` |
| `reference.json` of `anchor` | `9e7974ef9071ffd97c96c95960c4e17e57196fe6226fb1a72a06183401f488f7` |
| `reference.json` of `anchor_w16` | `4a7830c3e9a5fbc15ba43762f472ecc3adafa06544d02a1376f9162e6af6bbb9` |
| `reference.json` of `anchor_scale_fp32` | `fcda51ac6423da9e21107a4657e843db64c804095ce84fa29d3949067212b967` |
| `reference.json` of `anchor_scale_fp16` | `65240039b5a636011ce26d3c3d0abe47d8c87d18f25023be297f22bf44f0f76a` |
| `reference.json` of `half` | `22b5032e6226e49267fff8ea20375984b1773271fb61536dfc78b36f4d69c1bc` |
| Comparison A32 → A | `cce5b5f1175e3deb7e1415e8ce88bf1fcd140386586e03c90cdb0c66c08aa23b` |
| Comparison A32 → A32(W16) | `f0355b1be9b2b87e8b43bcd418cda91ad977f5af27871cc7047881dc7731cadd` |
| Comparison A32(W16) → FP16 | `35633a404695829e9a1406abdf84a726ad33c46bcbf2f3f0b2cb159df2c71587` |
| Comparison A32 → scale 73.32121 | `704bb513ff58e22501816310c2bc87b74dc69f43eaf7a836dc26a9b8987b4c3b` |
| Comparison A32 → scale 73.3125 | `8dffc50e91b9519a05431adbe212381ab4a370a73cccaa39e00cb69072b7701f` |
| `compare` of the second run above and A | `93ec64982d076f433548e061308dad79286e8373429f9900a5f5be2b45d56744` |

~~~sh
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.anchor_compare \
  --baseline <artifacts of the baseline step> --candidate <artifacts of the candidate step> > <comparison.json>
python3 -m Tools.Conversion.adapters.gemma4_31b.anchor_summary --report <report.json> \
  --comparison <A32 → A> --comparison <A32 → A32(W16)> --comparison <A32(W16) → FP16> \
  --comparison <A32 → scale 73.32121> --comparison <A32 → scale 73.3125> > docs/validation/gemma4-31b/anchor-20261122.json
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
fixed names of the records, from `fp16_summary`, which checked that each pipeline report is the run whose stage
manifest recorded its `export.json` and that each comparison was made from that `export.json` and the reference
record. It has no local paths, commands or environment variables; the records, bundles and logits stay in the local
work directory.

### Conditions

- The Mac of the runs above (Apple M4 Max, 128 GiB, macOS 27.0 (26A428), Xcode 27.0 (27A266a), on AC power). Every
  model job held the shared machine lock, and none started while another model process ran. Another session held the
  lock between the two comparisons; the times below are records, not measurements.
- Commit 1494a4c4 with no local changes for the FP32 reference and the exports, and commit 474bc2e0, which changed
  only the comparison tool and its tests after 1494a4c4, for the comparisons. Converter `apple/coreai-models`
  `c8e00c01800ba6ad324707007f7af358efc980a4` with coreai-core 1.0.0b3, coreai-models 0.1.0, coreai-opt 0.4.0,
  coreai-torch 0.4.3, torch 2.9.0, transformers 5.12.1 and safetensors 0.7.0. Reference: transformers 5.16.1,
  torch 2.15.0, numpy 2.4.6, safetensors 0.8.0. `coreai-verify` was built in release from the same commit.
- Runs (2026-11-23 JST, pipeline run IDs in UTC): the FP32 reference (`layers.py context`) from 02:05:41; the exports
  `20261122T170658-b5e24d4e` (`main` and `prefill`) from 02:06:58 and `20261122T171023-84a7c490` (`main` only) from
  02:10:23, each with the `prepare` step of the recipe (the `export` step ran in both, not reused); the comparisons
  from 02:56:49 and from 03:10:16.
- Inputs: `short` (26 tokens) and `context` (16,896 tokens), fixed-seed token IDs of the recipe's FP32 tools, not
  text.

### Exports

| | `main` and `prefill` | `main` only |
|---|---:|---:|
| Pipeline | 207.7 s | 187.8 s |
| `prepare` step | 27.1 s | 27.4 s |
| `export` step | 75.8 s | 57.7 s |
| Export tool | 72.71 s | 55.07 s |
| Hashing the files to load | 23.24 s | 23.62 s |
| Loading the model class in FP16 | 4.08 s | 2.62 s |
| `export_macos_model` | 34.46 s | 17.87 s |
| Saving the asset | 6.91 s | 7.05 s |
| Hashing the bundle | 3.30 s | 3.27 s |
| Peak resident memory | 20,772,356,096 B | 23,251,156,992 B |
| Kernel lifetime footprint peak | 9,483,761,440 B | 14,485,538,192 B |
| Swap increase (limit 2,000,000,000 B) | 0 B | 0 B |
| Free disk required at the start | 81,776,897,048 B | 81,776,897,048 B |
| Lowest free disk during the run (limit 64,424,509,440 B) | 402,811,510,784 B | 392,170,766,336 B |
| `main.mlirb` | 8,676,277,835 B | 8,676,248,369 B |
| `main.hash`, the SHA-256 of `main.mlirb` | `6cb1ece64c6f1a928d1c2e8e41c60026e346d9f89fb4421460bf426a98e0f08f` | `5441cb49d1846fae7292fe4eab0b09a4ac1d0b263b531043fb448b26028afa42` |
| Bundle | 8,708,469,929 B | 8,708,440,447 B |

Both assets passed the inspection. No compression setting was given and the PyTorch model had no quantized weight; its
85 parameters and buffers held 4,338,096,902 float16 elements, and no other type. Both assets store Float16
4,338,096,906 elements (4 more than the weights, a difference that is recorded and not judged) and the constants
Float32 386 (the expected 512 / 2 + 256 / 2 + 2), Int32 148, UInt32 29 and UInt64 1 (the same as the six-layer
`slicepos` assets of #665), and no other type. The functions are those asked for, with `input_ids` and `position_ids`
int32 [1, ?], `logits` float16 [1, ?, 262144] from `main`, `keyCache` and `valueCache` float16 [1, 1, 4, ?, 512] and
`slidingKeyCache` and `slidingValueCache` float16 [5, 1, 16, ?, 256]. The three tokenizer files are the distributed
files, and
`metadata.json` has vocabulary 262,144, context limit 16,896, `source.hf_model_id` `google/gemma-4-31B-it` and
`source.revision` `842da3794eaa0b77d5f08bae87a17459d91ff475`. `export.json` records the size and SHA-256 of every
bundle file. The free disk check counts twice the 8,676,193,804 B of FP16 weights plus 60 GiB.

### FP32 reference

`layers.py context` took 71.9 s: 23.76 s to hash the files to load, 2.22 s to load the first six layers of the
original implementation, 0.2 s for `short` in one call and 45.0 s for `context` in 17 calls of up to 1,024 tokens with
the cache of the implementation. It kept 26 rows of `short` (27,263,104 B) and 1,428 rows of `context`
(1,497,366,656 B), all finite. Peak resident memory 51,173,048,320 B, kernel lifetime footprint peak
51,124,933,424 B.

### Comparison

Both assets passed all twelve conditions: no non-finite value in any written or reference row, no constant written
row, every largest absolute logit difference below 2.0, and no top-1 change outside a near tie. Every one of
the twelve logits files of the asset with `main` only is byte-identical to the one of the asset with `main` and
`prefill`, also where the tokens before `--start` went through `main` instead of the `prefill` function, so the table
holds for both.

| Condition | Rows | Largest absolute logit difference | Top-1 changes in a near tie / outside | Rows of the known call shape: count, largest difference | Seconds (`main` and `prefill` / `main` only) |
|---|---:|---:|---:|---:|---:|
| `short-c1` | 26 | 0.458 | 25 / 0 | – | 15.4 / 11.1 |
| `short-c26` | 26 | 0.480 | 25 / 0 | – | 3.9 / 2.8 |
| `window-c1` | 1,300 | 0.989 | 1,266 / 0 | – | 36.6 / 33.6 |
| `window-c2` | 1,300 | 0.929 | 1,266 / 0 | – | 20.7 / 18.6 |
| `window-c4` | 1,300 | 0.855 | 1,266 / 0 | – | 12.4 / 10.2 |
| `window-c8` | 1,300 | 0.861 | 1,250 / 0 | 16, 4.266 | 8.8 / 7.7 |
| `prefix-c1` | 300 | 0.836 | 294 / 0 | – | 12.2 / 10.7 |
| `capacity-2048` | 16 | 0.437 | 15 / 0 | – | 4.7 / 4.2 |
| `capacity-4096` | 16 | 0.535 | 14 / 0 | – | 5.7 / 5.8 |
| `capacity-8192` | 16 | 0.459 | 16 / 0 | – | 7.7 / 9.9 |
| `capacity-16384` | 16 | 0.567 | 15 / 0 | – | 11.9 / 18.1 |
| `tail` | 64 | 0.466 | 61 / 0 | – | 14.6 / 20.4 |

- The rows of the known call shape are rows 96 to 111 of `window-c8`, from its calls of 8 tokens that end at 104 and
  112 positions; they are left out of the judgment. The other rows of that condition stay at 0.861.
- On the asset with `prefill`, the tokens before `--start` went through the `prefill` function (1, 1, 1, 2, 4 and 5
  calls for `prefix-c1`, the four `capacity` conditions and `tail`), and no `coreai-verify` process stopped. On the
  asset with `main` only they went through `main`. The comparison checked the function, the calls and the chunk of
  these reads against the rules of the engine.
- The first six layers saturate the logits: in the reference rows of `window-c8` and `capacity-2048`, a median of
  19,423 and 25,971 of the 262,144 logits per row are at 29.98 or above in absolute value. The top two reference
  logits of nearly every row are then within 0.01, so the top-1 criterion rarely applies; the logit difference is the
  measure here, as in #665.
- The first condition of each asset specialized the asset (10.5 s and 7.7 s); the others used the specialization
  cache. The caches took 21,660,868 KiB (`resources.bin` 22,163,497,512 B) and 16,962,060 KiB (`resources.bin`
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
| FP32 reference `record.json` | `ce9788b96d98ac3725b5f7d0efa5ab40dfccc9af9c5ede4fac2f063fd9ae35d6` |
| Report of the export with `main` and `prefill` | `eead1a1dd0daae6984e66738c40521c32fe13e654aeb3ce8c2f2dc1e2486dbf4` |
| Its `export.json` | `f03e81c94d072e601060c44a9665cd17dfaf46e685e602c252bee9b213dfc5ec` |
| Its `comparison.json` | `e6271e92dccbc63dda9e5fae1316337de2b2408f48b757dbbac2202479b05b9e` |
| Report of the export with `main` only | `82da28ff5691527dfb666375d40a87773d50669663fbc3ee7e61459db8eae4bf` |
| Its `export.json` | `1f3c7c4bc2ebe2f0471c60dd0231f29b9d489114539625264778360a25bfdb28` |
| Its `comparison.json` | `8093eaeedc8b7a2fd418fa0ba0d80b80fc5c432e054b7fe1039dc8f60ed031df` |

The summary of this section was made with the tools of commit f6a3990e; the current tools read only the records with
the outputs for speculation ([Outputs for speculation in the first six layers](#outputs-for-speculation-in-the-first-six-layers-695)).

~~~sh
python3 -m Tools.Conversion.adapters.gemma4_31b.fp16_summary --reference <FP32 reference record.json> \
  --report <report.json> --export <export.json> --comparison <comparison.json> \
  --report <report.json> --export <export.json> --comparison <comparison.json> \
  > docs/validation/gemma4-31b/fp16-first-layers-20261122.json
~~~

## Outputs for speculation in the first six layers (#695)

This section copies the numbers of the FP16 export of the first six layers with the two outputs that the drafters of
speculative decoding read (#695): `hidden_states` [1, n, 5376], the output of the final norm, from `main`, and
`tap_hidden_states` [1, n, 26880], the outputs after layers 1 to 5 joined in ascending order, from `main` and
`prefill` ([Model class](../../../Tools/Conversion/recipes/gemma4_31b/README.md#model-class)). It also copies the
comparison of these outputs of the model class with the original implementation in FP32, and the comparison of the
logits that the own engine reads from the two assets with the FP32 reference by the twelve conditions of
[FP16 export of the first six layers](#fp16-export-of-the-first-six-layers). The own engine reads the logits only; the
values of the two outputs in the assets are not compared here.
[`taps-first-layers-20261123.json`](taps-first-layers-20261123.json) holds the states, numbers, hashes, versions and
fixed names of the records, from `fp16_summary`, which made the checks of the section above, made the comparison of
the model class itself from the folders of the two runs of `layers.py run`, and checked that the two runs, the FP32
reference and each export are of the same contract and that the runs are of the layers and tap layers of each export.
It has no local paths, commands or environment variables; the records, bundles and arrays stay in the local work
directory.

### Conditions

- The Mac of the runs above (Apple M4 Max, 128 GiB, macOS 27.0 (26A428), Xcode 27.0 (27A266a), on AC power). Every
  model job held the shared machine lock, and none started while another model process ran. The export with `main`
  only waited 132 s until the lock was free and no other model process ran; the times below are records, not
  measurements.
- Commit f1390476 with no local changes for every model job. The converter and reference environments are those of
  the section above. `coreai-verify` was built in release from ddba9d51, whose Swift sources are those of f1390476.
- Runs (2026-11-23 JST, pipeline run IDs in UTC): the FP32 reference (`layers.py context`) from 05:10:42; `layers.py
  run` of the original implementation from 05:14:05 and of the model class from 05:16:47; the exports
  `20261122T202000-cacdae94` (`main` and `prefill`) from 05:20:00 and `20261122T202743-f7ff6a59` (`main` only) from
  05:27:43, each with the `prepare` step of the recipe (the `export` step ran in both, not reused); the comparisons
  from 05:32:49 and from 05:37:49.
- Tap layers `1,2,3,4,5`. Inputs: fixed-seed token IDs of the recipe's FP32 tools, not text: `short` (26 tokens) and
  `long` (1,400 tokens) for the model class, `short` and `context` (16,896 tokens) for the own engine.

### Model class and original implementation

`layers.py run` used the weights rounded to float16 and FP32 compute on the CPU. The original implementation read each
input in one call; the model class read `short` in one call and `long` in 102 calls (1,024 tokens, then calls of 1,
2, 4 and 8 tokens with its caches). The comparison passed. For `hidden_states` and each tap layer, the table gives the
largest ratio over the positions of the largest absolute difference at a position to the largest absolute value of
the original implementation at that position (the criterion: below 1e-3 at every position), and then the logit
differences, whose criteria are those of [FP32 outputs of the first
layers](../../../Tools/Conversion/recipes/gemma4_31b/README.md#fp32-outputs-of-the-first-layers).

| Input | Rows | `hidden_states` | Tap 1 | Tap 2 | Tap 3 | Tap 4 | Tap 5 | Largest absolute logit difference | Largest probability difference | Top-1 changes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `short` | 26 | 1.74e-6 | 6.06e-7 | 1.85e-6 | 1.46e-6 | 1.01e-6 | 9.12e-7 | 8.03e-4 | 7.78e-10 | 0 |
| `long` | 1,400 | 5.15e-6 | 4.45e-6 | 5.52e-6 | 4.87e-6 | 4.03e-6 | 3.63e-6 | 1.78e-3 | 1.90e-9 | 0 |

The run of the original implementation took 32.0 s (23.8 s to hash the files to load, 2.2 s to load, 0.23 s for
`short` and 4.49 s for `long`), that of the model class 53.3 s (23.1 s, 3.4 s, 0.24 s and 25.6 s). Peak resident
memory 26,334,330,880 B and 28,226,437,120 B, kernel lifetime footprint peak 24,628,283,440 B and 28,170,605,184 B.
All arrays are finite.

### Exports

| | `main` and `prefill` | `main` only |
|---|---:|---:|
| Pipeline | 201.3 s | 177.9 s |
| `prepare` step | 26.8 s | 25.9 s |
| `export` step | 71.8 s | 53.5 s |
| Export tool | 69.18 s | 51.05 s |
| Hashing the files to load | 23.25 s | 22.38 s |
| Loading the model class in FP16 | 2.88 s | 2.63 s |
| Tracing and converting the functions | 33.16 s | 16.91 s |
| Saving the asset | 5.97 s | 5.42 s |
| Hashing the bundle | 3.29 s | 3.09 s |
| Peak resident memory | 23,299,063,808 B | 23,251,877,888 B |
| Kernel lifetime footprint peak | 9,480,206,184 B | 14,486,668,640 B |
| Swap increase (limit 2,000,000,000 B) | 0 B | 0 B |
| Free disk required at the start | 81,776,897,048 B | 81,776,897,048 B |
| Lowest free disk during the run (limit 64,424,509,440 B) | 392,635,736,064 B | 374,457,257,984 B |
| `main.mlirb` | 8,676,278,120 B | 8,676,248,568 B |
| `main.hash`, the SHA-256 of `main.mlirb` | `8a242c80a12e3d4c81180de930f6bc6ea8ccc0ec60b4a0b1d356e1a0a6ee6600` | `aca31adaf9bd963754624734df916b93a7883b15d5915790d98b64d12ab3d31e` |
| Bundle | 8,708,470,336 B | 8,708,440,768 B |

Both assets passed the inspection. `main` returns `logits` float16 [1, ?, 262144], `hidden_states` float16
[1, ?, 5376] and `tap_hidden_states` float16 [1, ?, 26880] in this order, and `prefill` returns `tap_hidden_states`
float16 [1, ?, 26880]; the inputs, the states, the PyTorch model (85 parameters and buffers, 4,338,096,902 float16
elements) and the storage counts of the asset (Float16 4,338,096,906, Float32 386, Int32 148, UInt32 29 and UInt64 1)
are those of the section above. `metadata.json` also has `tap_layers` [1, 2, 3, 4, 5] and `tap_convention`
`layers-passed:before-final-norm:ascending`. Against the exports of the section above, whose `main` returned `logits`
only, `main.mlirb` is 285 B and 199 B larger.

### FP32 reference

`layers.py context` took 72.3 s (24.0 s to hash, 2.2 s to load, 0.23 s for `short` and 45.1 s for `context`). Its
`short.npy` and `context.npy` have the SHA-256 of the FP32 reference of the section above. Peak resident memory
55,599,693,824 B, kernel lifetime footprint peak 55,569,719,088 B.

### Comparison

Both assets passed all twelve conditions. Each of the 24 logits files is byte-identical to the file of the same
condition from the asset with the same functions in the section above, so the largest differences, the top-1 changes
and the rows of the known call shape are those of its table. The tokens before `--start` went through the `prefill`
function on the asset with `prefill` and through `main` on the other, with the calls of the section above.

| Condition | Seconds (`main` and `prefill` / `main` only) | In the section above |
|---|---:|---:|
| `short-c1` | 15.0 / 10.0 | 15.4 / 11.1 |
| `short-c26` | 3.6 / 2.7 | 3.9 / 2.8 |
| `window-c1` | 35.2 / 33.9 | 36.6 / 33.6 |
| `window-c2` | 20.3 / 18.8 | 20.7 / 18.6 |
| `window-c4` | 11.5 / 10.2 | 12.4 / 10.2 |
| `window-c8` | 8.5 / 7.7 | 8.8 / 7.7 |
| `prefix-c1` | 11.4 / 10.7 | 12.2 / 10.7 |
| `capacity-2048` | 4.5 / 4.2 | 4.7 / 4.2 |
| `capacity-4096` | 5.6 / 5.9 | 5.7 / 5.8 |
| `capacity-8192` | 7.5 / 9.8 | 7.7 / 9.9 |
| `capacity-16384` | 12.6 / 18.0 | 11.9 / 18.1 |
| `tail` | 14.9 / 21.3 | 14.6 / 20.4 |

The first condition of each asset specialized the asset (10.7 s and 6.7 s); the others used the specialization cache.
The caches took 21,660,884 KiB (`resources.bin` 22,163,497,512 B) and 16,962,064 KiB (`resources.bin`
17,352,106,248 B) and were removed after the comparisons.

### Limits

- Six of the 60 layers with the tap layers 1 to 5, FP16 only, one machine, one OS build and one software environment.
  The export of all 60 layers with the tap layers 1, 17, 29, 47 and 59 and the 8-bit assets are later steps.
- The own engine binds the first output of `main` as the logits and reads no other output, so the values of
  `hidden_states` and `tap_hidden_states` in the assets are not compared with FP32 here; reading them is a later step.
- The comparison of the model class is of PyTorch in FP32 on the CPU, not of the FP16 assets.

### Sources

| File | SHA-256 |
|---|---|
| `record.json` of the original implementation (`layers.py run`) | `a69c4ef758688f79bb7f66de361e255ff739fd20f3ad81c05b4395bbfc9836ae` |
| `record.json` of the model class (`layers.py run`) | `86c291908830b4f552d286ccd376a2b007c8e7a3c7005da61f6d958b1bc00aa7` |
| FP32 reference `record.json` | `43b6313b217c9c2f203d6e10285a6d1c76fa0776bb1a332bc1c07d185bea0d5c` |
| Report of the export with `main` and `prefill` | `acdada5938138cf345dac594b537ea0379db0a6e0c0faf4cad7ff2388ab0f663` |
| Its `export.json` | `e67ff01f72625b6118972c81c9889b60f3e4ee59d471cc558aa36078fb680353` |
| Its `comparison.json` | `36d35e55236551aad99fd3fb11c555fed869a1b6577320ef652788db85a6cf41` |
| Report of the export with `main` only | `3fb0c924217a1e8c1abe606be65be275f22a1f20745f862754112c5caa6095da` |
| Its `export.json` | `6298528a39ff87d1c2fc2024c2ce161d761d90d9db0950e3f7d16a5efd70a807` |
| Its `comparison.json` | `a18abf9350bd420ccc5a9ac2e7c06669531964530b7d52c07cab043d9f2e5999` |

~~~sh
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.fp16_summary \
  --reference <FP32 reference record.json> \
  --original <folder of the run of the original implementation> --model-class <folder of the run of the model class> \
  --report <report.json> --export <export.json> --comparison <comparison.json> \
  --report <report.json> --export <export.json> --comparison <comparison.json> \
  > docs/validation/gemma4-31b/taps-first-layers-20261123.json
~~~

## 8-bit export of the first six layers

This section copies the numbers of the 8-bit export of the first six layers (five sliding-window layers and one
full-attention layer) of the
[`recipe-8bit.json`](../../../Tools/Conversion/recipes/gemma4_31b/README.md#8-bit-export-of-the-first-layers) recipe to a
dynamic-shape Core AI bundle for macOS with the functions `main` and `prefill` and the outputs for speculation of the
section above (tap layers 1 to 5), of the comparison of the logits that the own engine reads from it (`coreai-verify
logits`) with an FP32 reference restored from the same quantized weights, and of the effect of the quantization: how
far that reference is from the FP32 reference with float16-rounded weights (#681). Six layers do not make a usable
model: the record shows that the quantized export, the asset and the reads of the engine are right for these layers,
not the quality of the model or of the quantization.
[`8bit-first-layers-20261123.json`](8bit-first-layers-20261123.json) holds the states, numbers, hashes, versions and
fixed names of the records, from `fp16_summary`, which checked that the pipeline report is the run whose stage manifest
recorded its `export.json`, the files of its bundle and its `quantized.safetensors`; that the comparison was made from
that `export.json` and the restored reference record and records their layers, the functions and the asset; that the
restored reference and the export are of the contract of the repository and of the same layers and that the reference
was restored from the quantized tensors of that export (the SHA-256 of `export.json`, the size and SHA-256 of
`quantized.safetensors` and the number of tensors); and that the effect was computed from that reference record, with a
rounded reference of the same layers, on the conditions of the comparison. It computed the effect itself from the two
references, and the numbers of each condition (the differences, the counts of non-finite values and of top-1 changes,
the known rows and the checks) again from the logits file of the condition and the reference, and counted the
non-finite logits of the reference again, each equal to the record. It also checked the type of each number of a run
that no array gives, such as a time, and the relations that their writers give them, such as the time of a condition
not being shorter than the steps that its verifier recorded (the list is in the
[recipe](../../../Tools/Conversion/recipes/gemma4_31b/README.md#comparison-with-the-own-engine)). The hashes of the
quantized tensors are those of the tensors that the export passed to the converter (`quantized.safetensors`), because
the asset summary does not list the tensors of the asset; the asset itself is checked through the number of its
dequantizing operations and through the logits of the engine. The summary has no local paths, commands or environment
variables; the records, the bundle, the quantized tensors and the logits stay in the local work directory.

### Conditions

- The Mac of the runs above (Apple M4 Max, 128 GiB, macOS 27.0 (26A428), Xcode 27.0 (27A266a), on AC power, battery
  100 % at the start of the export, 51 % (charging) at the start of the restored reference, the reference with
  float16-rounded weights and the comparison, which ran in one hold of the lock that the run took after another
  session's job released it). Every model job held the shared machine lock, and none started while another model
  process ran. The times below are records, not measurements.
- No local changes for any model job. The export ran at commit ca63eb81. The restored reference, the reference with
  float16-rounded weights and the comparison ran at commit da7add90, from a `git archive` of that commit, the last
  change of the sources that these jobs read: the later merge of `main` only added the files of the MTP drafter
  (`assistant.py`, `drafter.py`, `mtp.py` and the recipe `assistant.json`). The commits after ca63eb81 changed only the
  reader of the export (`native.py`), the effect (`effect.py`) and the summary (`fp16_summary.py`), no input of the
  export step, so it was not run again. Converter
  `apple/coreai-models` `c8e00c01800ba6ad324707007f7af358efc980a4` with coreai-core 1.0.0b3, coreai-models 0.1.0,
  coreai-opt 0.4.0, coreai-torch 0.4.3, torch 2.9.0, transformers 5.12.1 and safetensors 0.7.0. Reference: transformers
  5.16.1, torch 2.15.0, numpy 2.4.6, safetensors 0.8.0. `coreai-verify` is the binary built in release at 7de91afa, the
  same as from ca63eb81 (the SHA-256 in the record of the comparison, unchanged); no Swift source changed after it.
- Runs (JST, the pipeline run ID in UTC): the export `20261123T043147-264994c4` from 2026-11-23 13:31:43 with the
  `prepare` step of the recipe (the `export` step ran, not reused); the restored reference (`layers.py context
  --weights 8bit`) from 2026-11-24 07:21:12, the reference with float16-rounded weights (`--weights float16`) from
  07:22:33 and the comparison from 07:23:46, in one hold of the shared machine lock (07:21:12 to 07:25:57).
- Tap layers `1,2,3,4,5`. Inputs: `short` (26 tokens) and `context` (16,896 tokens), the fixed-seed token IDs of the
  FP16 record above.

### Export

| | `main` and `prefill` |
|---|---:|
| Pipeline | 213.5 s |
| `prepare` step | 27.0 s |
| `export` step | 79.8 s |
| Export tool | 77.06 s |
| Hashing the files to load | 23.38 s |
| Loading the model class in FP16 | 4.65 s |
| `quantize_for_export` | 3.22 s |
| Writing `quantized.safetensors` and hashing it | 3.29 s |
| Tracing and converting the functions | 33.84 s |
| Saving the asset | 5.68 s |
| Hashing the bundle | 2.31 s |
| Peak resident memory | 20,349,681,664 B |
| Kernel lifetime footprint peak | 11,581,915,624 B |
| Swap increase (limit 2,000,000,000 B) | 0 B |
| Free disk required at the start | 94,791,187,754 B |
| Lowest free disk during the run (limit 64,424,509,440 B) | 287,105,204,224 B |
| `main.mlirb` | 6,113,706,114 B |
| `main.hash`, the SHA-256 of `main.mlirb` | `4284357701f6269241428ff43323cb69bec24790ecc6e06edf43912d304dc98d` |
| Bundle | 6,145,898,332 B |
| `quantized.safetensors` | 3,294,771,912 B |
| Its SHA-256 | `5dadc554a8d37b718fe93d1449618af83c41509af58c3fe771d3c718fcfb8833` |

The asset passed the inspection. The export was given the fixed config. It quantized 41 weights, the seven Linear
weights of each sliding-window layer and the six of the full-attention layer, each as uint8 data with float16 scales and
float16 minimums in blocks of [1, 32], and no other tensor, and the converter logged no `Skipping quantization.`. The
PyTorch model then held 1,592,466,182 float16 and 2,928,672,768 uint8 elements in 208 parameters and buffers, and no
other type; `quantized.safetensors` holds the 123 data, scale and minimum tensors. The asset has 77
`blockwise_shift_scale` operations, the dequantization of a weight, which the inspection requires: one for each of the
41 quantized weights in `main`, and 36 in `prefill`, which does not compute the query, output and MLP projections of the
sixth layer because that layer is not a tap layer. It stores UInt8 2,930,049,024 elements (1,376,256 more than the
uint8 elements of the PyTorch model, the difference of the six-layer 8-bit asset of #665) and Float16 1,592,466,186 (4
more), recorded without a judgment, and the constants Float32 386, Int32 148, UInt32 29 and UInt64 1 (those of the FP16
assets), and no other type. `main` returns `logits` float16 [1, ?, 262144], `hidden_states` float16 [1, ?, 5376] and
`tap_hidden_states` float16 [1, ?, 26880] in this order, and `prefill` returns `tap_hidden_states` float16
[1, ?, 26880]; the inputs and the states are those of the FP16 assets. `main.mlirb` is 0.705 times the FP16 one with
`main` and `prefill` of the section above. The bundle holds exactly the seven files of a complete bundle, the three of
the asset, `metadata.json` and the three tokenizer files, which are the distributed files, and `metadata.json` has
`compression` `8bit`, `tap_layers` [1, 2, 3, 4, 5], `tap_convention` `layers-passed:before-final-norm:ascending`,
vocabulary 262,144, context limit 16,896, `source.hf_model_id` `google/gemma-4-31B-it` and `source.revision`
`842da3794eaa0b77d5f08bae87a17459d91ff475`. The free disk check counts seven bytes per weight element
(30,366,678,314 B) plus 60 GiB.

Three earlier 8-bit exports of the same layers wrote the same `quantized.safetensors` (the same size and SHA-256), the
same storage counts and the same 77 operations: the one from e873b4ee, before the last changes of the checks of the
export and of its reader, whose `main.mlirb` is 7 B larger and differs from this one only in its last 7,512 bytes; the
one from 18eedf75, with the same outputs, whose `main.mlirb` is 5 B larger (it differed from that of e873b4ee only in
its last 7,510 bytes); and the one from 6abb7325, before the outputs for speculation, whose `main` returned `logits`
only and whose `main.mlirb` is 283 B smaller.

### FP32 references

| | Restored from the quantized weights | float16-rounded weights |
|---|---:|---:|
| Total | 74.7 s | 72.4 s |
| Hashing the files to load | 25.01 s | 23.57 s |
| Loading the original implementation (and restoring the 41 weights) | 2.33 s | 2.26 s |
| `context` in 17 calls of up to 1,024 tokens | 46.5 s | 45.6 s |
| Peak resident memory | 51,240,747,008 B | 55,566,876,672 B |
| Kernel lifetime footprint peak | 51,208,721,272 B | 55,536,934,752 B |

Before it made its folder, the restored reference read the export with the reader that the comparison also uses (a
complete bundle under its final name with the sizes and SHA-256 of its record, the tokenizer files of the contract, the
expected `metadata.json` and `main.hash` the SHA-256 of `main.mlirb`; real folders and regular files, each read only at
its recorded size; no failure record or unfinished bundle; a passed record of these layers and tap layers, whose
inspection it ran again and whose numbers that the export derives from others follow from them; and the 41 Linear
weights in blocks of [1, 32] with `quantized.safetensors` of the recorded size and SHA-256), checked the 123 tensors
against their SHA-256 and each weight against its shape in the checkpoint, with scales and minimums of exactly [rows,
columns / 32], and then restored the 41 weights (these checks are not in the times above). Both references kept 26 rows
of `short` and 1,428 rows of `context`, all finite, and their logits files have the SHA-256 of those of the references
of the runs at 22af0632, 7de91afa, 8dd53a88, f15f6f79, ca63eb81, e873b4ee, 18eedf75 and 6abb7325.

### Comparison

The asset passed all twelve conditions against the restored reference: no non-finite value in any written or reference
row, no constant written row, every largest absolute logit difference below 2.0, and no top-1 change outside a near tie.
Each of the twelve logits files is byte-identical to the file of the same condition from the comparisons of the same
export at ca63eb81, f15f6f79, 8dd53a88, 7de91afa and 22af0632 and from the exports from e873b4ee, 18eedf75 and 6abb7325, so the
differences are those of these exports. The FP16 asset with `main` and `prefill` against
its reference of the FP16 section above is in parentheses.

| Condition | Rows | Largest absolute logit difference (FP16) | Top-1 changes in a near tie / outside | Rows of the known call shape: count, largest difference (FP16) | Seconds |
|---|---:|---:|---:|---:|---:|
| `short-c1` | 26 | 0.516 (0.458) | 25 / 0 | – | 9.2 |
| `short-c26` | 26 | 0.521 (0.480) | 25 / 0 | – | 2.1 |
| `window-c1` | 1,300 | 0.926 (0.989) | 1,266 / 0 | – | 25.8 |
| `window-c2` | 1,300 | 0.984 (0.929) | 1,266 / 0 | – | 13.9 |
| `window-c4` | 1,300 | 0.953 (0.855) | 1,266 / 0 | – | 7.7 |
| `window-c8` | 1,300 | 0.902 (0.861) | 1,250 / 0 | 16, 4.379 (4.266) | 6.6 |
| `prefix-c1` | 300 | 0.832 (0.836) | 294 / 0 | – | 8.0 |
| `capacity-2048` | 16 | 0.583 (0.437) | 15 / 0 | – | 3.1 |
| `capacity-4096` | 16 | 0.612 (0.535) | 14 / 0 | – | 4.8 |
| `capacity-8192` | 16 | 0.761 (0.459) | 16 / 0 | – | 6.4 |
| `capacity-16384` | 16 | 0.553 (0.567) | 15 / 0 | – | 11.1 |
| `tail` | 64 | 0.913 (0.466) | 61 / 0 | – | 13.1 |

- The rows of the known call shape are rows 96 to 111 of `window-c8`; they are left out of the judgment.
- The tokens before `--start` went through the `prefill` function (1, 1, 1, 2, 4 and 5 calls for `prefix-c1`, the four
  `capacity` conditions and `tail`), and no `coreai-verify` process stopped.
- The first condition specialized the asset (6.6 s); the others used the specialization cache, which took 8,739,724 KiB
  (`resources.bin` 8,932,172,280 B, the size that #665 found for the same form) and was removed after the comparison.
- The differences, 0.516 to 0.984 outside the known rows, are of the size of those of the FP16 asset against its own
  reference (0.437 to 0.989).

### Effect of the quantization

`effect` compared the restored reference with the reference of float16-rounded weights on the rows of each condition.
These differences are the effect of the quantization itself and are recorded, not judged.

| Condition | Rows | Largest absolute logit difference | Top-1 changes in a near tie of the rounded reference / outside |
|---|---:|---:|---:|
| `short-c1`, `short-c26` | 26 | 2.800 | 0 / 0 |
| `window-c1`, `window-c2`, `window-c4`, `window-c8` | 1,300 | 5.263 | 8 / 0 |
| `prefix-c1` | 300 | 5.263 | 1 / 0 |
| `capacity-2048` | 16 | 2.614 | 0 / 0 |
| `capacity-4096` | 16 | 2.060 | 0 / 0 |
| `capacity-8192` | 16 | 1.870 | 0 / 0 |
| `capacity-16384` | 16 | 2.447 | 0 / 0 |
| `tail` | 64 | 2.712 | 0 / 0 |

- The window conditions read the same 1,300 rows of the references and `prefix-c1` the last 300 of them, so the 1,454
  distinct rows (26 of `short`, 1,428 of `context`) have 8 top-1 changes, all within a near tie of the rounded reference.
- The quantization moves the logits of these six layers by up to 5.263, more than the conversion and the engine do
  (up to 0.984 in the comparison above). Saturated logits make the top-1 changes a weak signal here; the quality of the
  8-bit model is a matter for the evaluation of all 60 layers.

### Limits

- Six of the 60 layers with the tap layers 1 to 5, one function layout, one machine, one OS build and one software
  environment; the inputs are token IDs of fixed seeds, not a quality set. The export and comparison of all 60 layers
  and the quality evaluation of the 8-bit model are later steps.
- The asset summary gives totals per storage type and per operation, not per tensor: the count of the dequantizing
  operations finds a quantized weight that the asset holds in another form, leaves out or adds, not a change of its
  values, which only the logits of the engine show, and the per-tensor hashes cover the tensors that the export passed
  to the converter, not the stored asset.
- The own engine binds the first output of `main` as the logits and reads no other output, so the values of
  `hidden_states` and `tap_hidden_states` in the 8-bit asset are not compared here.
- The rows of the call shape that `Engine` avoids for this model are recorded but not judged.

### Sources

| File | SHA-256 |
|---|---|
| Restored FP32 reference `record.json` | `b57c1539d3ef062021ddc5b6d8a761b26c8e720e251eccf3f9dbb9f79356515f` |
| FP32 reference of float16-rounded weights `record.json` | `d3e52e7ae6d6c1728c03c9e104c39e0677b1725e7f24f955b863c4d31904abc1` |
| Report of the export | `0c118a098bfb814f4eb4ab145310f660aaf34865329f79f35b63b968d3faf483` |
| Its `export.json` | `ce5a3eb725214faac782150e2606476b666da7a1c7ec43cbcea889d6aec7b025` |
| Its `comparison.json` | `05b43a87268bf855d9c89f0d180ca947d6a1f1e19628b4f46464640aa40025c4` |

~~~sh
python3 -m Tools.Conversion.adapters.gemma4_31b.fp16_summary --reference <restored reference record.json> \
  --rounded <folder of the reference of float16-rounded weights> \
  --report <report.json> --export <export.json> --comparison <comparison.json> \
  > docs/validation/gemma4-31b/8bit-first-layers-20261123.json
~~~

## Outputs for speculation read by the own engine in the first six layers (#702)

This section copies the numbers of reading the two outputs for speculation from the FP16 and the 8-bit export of the
first six layers with the own engine and `coreai-verify` (#702), and of their comparison with the original
implementation in FP32. Both exports have `main` and `prefill` ([FP16 export of the first six
layers](#fp16-export-of-the-first-six-layers) and [8-bit export of the first six
layers](#8-bit-export-of-the-first-six-layers)); [the section of
#695](#outputs-for-speculation-in-the-first-six-layers-695) compared the two outputs only as the model class in FP32,
and the own engine read the logits only. `hidden_states` [1, n, 5376] and `tap_hidden_states` [1, n, 26880] (the outputs
after layers 1 to 5 joined in ascending order) are read from `main`, and `tap_hidden_states` from `prefill`.
[`speculation-outputs-first-layers-20261124.json`](speculation-outputs-first-layers-20261124.json) holds the states,
numbers, hashes and versions of the references, the comparisons, the checks of the logits files, the costs and the test.
It has no local paths, commands or environment variables; the records, bundles and arrays stay in the local work
directory.

### Conditions

- The Mac of the runs above (Apple M4 Max, 128 GiB, macOS 27.0 (26A428), Xcode 27.0 (27A266a), on AC power, battery 100
  %). Every model job held the shared machine lock and started only when no process of the model tools (`coreai-chat`,
  `coreai-verify`, `CoreAISamples` and the Python conversion tools) was running. The two references,
  the two comparisons, the checks of the command and the costs ran in one hold from 16:12:26 to 16:17:14 JST on
  2026-11-24, taken as a timing measurement with no wait for the lock; the other processes of the machine were not
  stopped. The test of the engine ran in a second hold (16:19:29 to 16:19:57), which is not a timing measurement.
- Commit 4c88220b with no local changes (the jobs ran from a `git archive` of the commit). The reference environment is
  that of the sections above (numpy 2.4.6, safetensors 0.8.0, torch 2.15.0, transformers 5.16.1). `coreai-verify` was
  built in release from d89fd838, from ac954297 and from 4c88220b (`swift build --package-path Tools/Verify -c release`),
  each time with the SHA-256 `e1754211fb67f045fcbc8beb4ac497776cc48270b3362785ccf9644c7b37337f`; their Swift sources
  are the same.
- Assets: the FP16 export `20261122T202000-cacdae94` (`main.hash` `8a242c80a12e…`) and the 8-bit export
  `20261123T043147-264994c4` (`main.hash` `4284357701f6…`), each with the tap layers 1 to 5.
- References: `layers.py run` of the original implementation with `--layers 6 --taps 1,2,3,4,5,6`, which holds the
  output of every layer. The FP16 export is compared with the weights rounded to float16 (`--weights float16`), the
  8-bit export with the quantized Linear weights restored from the tensors that the export passed to the converter,
  every other tensor rounded to float16 (`--weights 8bit --export`); both in float32 on the CPU. Inputs: the fixed-seed
  token IDs of the recipe, not text: `short` (26 tokens) and `long` (1,400 tokens).

### FP32 references

| | FP16 weights | 8-bit weights |
|---|---:|---:|
| Run | 32.2 s | 36.6 s |
| Hashing the files to load | 24.2 s | 22.7 s |
| Loading | 2.1 s | 7.8 s |
| `short` (26 rows) | 0.22 s | 0.23 s |
| `long` (1,400 rows) | 4.33 s | 4.29 s |
| Peak resident memory | 26,721,943,552 B | 29,637,787,648 B |
| Kernel lifetime footprint peak | 24,625,989,704 B | 28,403,782,488 B |

`Run` is the whole run, from the check of the files to load to the last array, and `Hashing the files to load` is that
check against the contract, which comes first. `Loading` is what lies between that check and the first input: for the
8-bit reference it includes the check of its export and the reading of its quantized tensors (which is why it is longer
than that of the FP16 reference), then, for both, the loading of the model and the application of the weights.

The 8-bit reference restored 123 tensors from the quantized file of the export (3,294,771,912 B,
`5dadc554a8d37b718fe93d1449618af83c41509af58c3fe771d3c718fcfb8833`), whose export record has the SHA-256
`ce5a3eb725214faac782150e2606476b666da7a1c7ec43cbcea889d6aec7b025`. All arrays are finite.

### Comparison

`speculation` ran `coreai-verify logits` once per condition and compared the files that it wrote. Both assets passed all
four conditions; the checks and values are those of
[`speculation`](../../../Tools/Conversion/recipes/gemma4_31b/README.md#outputs-for-speculation-read-by-the-own-engine):
for every output and every tap layer the sum of the squared differences from the reference of the same layer is at most
1/2 of the sum from every other layer (the smallest of them above 0), the cosine of every row is at least 0.99 and the
median at least 0.999, and every value is finite. For `hidden_states` the other layer is the last layer, and for each
tap layer the table gives the largest ratio to the nearest other layer (the criterion: at most 0.5) and the smallest
cosine and median cosine over the tap layers.

| Condition | Output | FP16: largest ratio | Smallest cosine | Smallest median | 8-bit: largest ratio | Smallest cosine | Smallest median |
|---|---|---:|---:|---:|---:|---:|---:|
| `short-c1` | `main.hidden_states` | 3.97e-07 | 1.000000 | 0.9999998 | 5.70e-07 | 0.999999 | 0.9999998 |
| `short-c1` | `main.tap_hidden_states` | 3.87e-05 | 1.000000 | 0.9999999 | 5.93e-05 | 1.000000 | 0.9999999 |
| `short-c8` | `main.hidden_states` | 3.85e-07 | 1.000000 | 0.9999999 | 6.41e-07 | 0.999999 | 0.9999997 |
| `short-c8` | `main.tap_hidden_states` | 4.17e-05 | 1.000000 | 0.9999999 | 5.71e-05 | 1.000000 | 0.9999999 |
| `long-main` | `main.hidden_states` | 3.75e-07 | 0.999999 | 0.9999998 | 4.91e-07 | 0.999999 | 0.9999998 |
| `long-main` | `main.tap_hidden_states` | 4.55e-05 | 1.000000 | 0.9999999 | 5.86e-05 | 0.999999 | 0.9999999 |
| `long-prefill` | `main.hidden_states` | 3.77e-07 | 1.000000 | 0.9999998 | 4.73e-07 | 1.000000 | 0.9999998 |
| `long-prefill` | `main.tap_hidden_states` | 4.66e-05 | 1.000000 | 0.9999999 | 5.90e-05 | 1.000000 | 0.9999999 |
| `long-prefill` | `prefill.tap_hidden_states` | 4.52e-05 | 1.000000 | 0.9999999 | 5.88e-05 | 0.999999 | 0.9999999 |

In every case the nearest other layer of the largest ratio is the last layer (`hidden_states`) or layer 4 of tap layer
5. `long-prefill` reads positions 0 to 1,099 with the `prefill` function (one call) and the rest with `main`, one token
at a time (300 calls); `long-main` reads the 1,400 tokens in one `main` call; `short-c1` and `short-c8` read 26 tokens
in 26 and in 4 calls.

| Condition | Seconds (FP16 / 8-bit) | Scoring seconds (FP16 / 8-bit) |
|---|---:|---:|
| `short-c1` | 20.0 / 5.7 | 0.75 / 0.56 |
| `short-c8` | 3.7 / 1.7 | 0.23 / 0.18 |
| `long-main` | 5.0 / 3.0 | 1.45 / 1.49 |
| `long-prefill` | 11.3 / 7.5 | 7.26 / 5.39 |

The first condition of each asset specialized the asset (15.1 s and 3.6 s); the others used the specialization cache.
The caches took 21,660,884 KiB and 8,739,724 KiB after the comparisons, and were removed after the comparisons and
again after the checks.

### Logits and the cost of reading the outputs

In these runs, asking for the outputs did not change the logits. Five runs of `coreai-verify logits` per asset were
made; four of them without outputs and the fifth (`short-c1`) with both outputs of `main`, and every logits file is
byte-identical (SHA-256 and size) to the file of the same condition of the comparison of the section of #695 (FP16,
`36d35e55…`) and of the comparison of the 8-bit section (`05b43a87…`).

| Run | Condition | Bytes | Same as the earlier comparison (FP16 / 8-bit) |
|---|---|---:|---|
| Without outputs | `short-c1` | 13,631,488 | yes / yes |
| Without outputs | `short-c26` | 13,631,488 | yes / yes |
| Without outputs | `window-c8` | 681,574,400 | yes / yes |
| Without outputs | `prefix-c1` | 157,286,400 | yes / yes |
| With both outputs of `main` | `short-c1` | 13,631,488 | yes / yes |

A request for the `logits` output, for a name that the function does not have, and for `hidden_states` from `prefill`
each ended with exit code 1 (`unreadableOutput`). A run of `long` from position 1,100 with both outputs of `main`
and `tap_hidden_states` of `prefill` ended with exit code 0 on both assets.

The cost of reading both outputs of `main` for the whole `long` input (one call of 1,400 tokens, specialization cache in
place; two runs of each kind, alternating, in this order of plain, with outputs, plain, with outputs):

| | Seconds of the call | Peak resident set | Peak memory footprint |
|---|---|---:|---:|
| FP16, without outputs | 0.946, 0.986 | 35,682,795,520 B | 2,287,502,040 B |
| FP16, with both outputs | 1.500, 0.963 | 35,682,779,136 B | 2,288,124,656 B |
| 8-bit, without outputs | 1.265, 1.544 | 12,255,838,208 B | 2,153,848,296 B |
| 8-bit, with both outputs | 1.486, 1.927 | 12,255,821,824 B | 2,153,979,320 B |

The logits of the four runs of each asset are identical (SHA-256). Peak values are the largest of the two runs.

### Test of the engine

`ModelIntegrationTests.outputRowsFollowTheLastCall` ran on the FP16 export (opt-in with `COREAI_OUTPUT_ROWS_BUNDLE`,
under the shared lock) and passed in 13.993 s. It reads both outputs in one call and by row, and checks that a row out
of range, an output that the last read did not ask for, `logits` or a name that the function does not have as an output
(the engine's tokens stay as they were), and `hidden_states` from `prefill` are refused, and that the outputs are
unavailable after the next read, a reset, a restore or an unload. The test requires the asset to have the `prefill`
function. The cache of the testing helper (21,660,884 KiB) was removed after the run.

### Limits

- Six of the 60 layers with the tap layers 1 to 5, one machine, one OS build and one software environment. The export of
  all 60 layers with the tap layers 1, 17, 29, 47 and 59 is a later step.
- The references are PyTorch in float32 on the CPU, not the FP16 or 8-bit assets; the comparison checks which layer an
  output is of and how close it is, not the quality of the model. The criteria are those of the recipe, not tolerances
  that a drafter needs.
- The timings are those of one hold on one machine whose other processes were not stopped; the specialization cache was
  in place for every condition after the first.
- The check of the command and the cost used no more than two runs per kind.

### Sources

| File | SHA-256 |
|---|---|
| `record.json` of the FP32 reference with the FP16 weights | `ad043097d1796a773ecee410b5690844c2e472682da4b322b4ebb0491714d540` |
| `record.json` of the FP32 reference with the 8-bit weights | `3b1edbc607604b1170e0ce253b4a34645dc04796f7781860339345520427ea0b` |
| `export.json` of the FP16 export | `e67ff01f72625b6118972c81c9889b60f3e4ee59d471cc558aa36078fb680353` |
| `export.json` of the 8-bit export | `ce5a3eb725214faac782150e2606476b666da7a1c7ec43cbcea889d6aec7b025` |
| `comparison.json` of the FP16 export | `aab3ba5ba8cc68a74ef357fe74b3e9e6a1db367dbd2764d9241a37d6dbd60dc4` |
| `comparison.json` of the 8-bit export | `553a52529e21f54756374eeb1cd4fd0b30985b86b9d4e7a769b14f3dcf566d7c` |
| `coreai-verify` (release) | `e1754211fb67f045fcbc8beb4ac497776cc48270b3362785ccf9644c7b37337f` |

~~~sh
: "${EXPORT_FOLDER:?set it to the export folder of the artifacts of the export step}" &&
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.layers run \
  --implementation original --layers 6 --taps 1,2,3,4,5,6 --weights float16 \
  --model Tools/Conversion/recipes/gemma4_31b/model.json --snapshot .local/conversion/models/google--gemma-4-31B-it \
  --output .local/conversion/gemma4_31b/speculation/reference-float16 &&
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.speculation \
  --export "$EXPORT_FOLDER" --reference .local/conversion/gemma4_31b/speculation/reference-float16 \
  --verifier Tools/Verify/.build/release/coreai-verify \
  --output .local/conversion/gemma4_31b/speculation/comparison-float16
~~~

## MTP drafter class (#704)

This section copies the numbers of the drafter class of the recipe for the MTP drafter
`google/gemma-4-31B-it-assistant` ([MTP drafter](../../../Tools/Conversion/recipes/gemma4_31b/README.md#mtp-drafter)):
its comparison with the drafter of the original implementation on the features that #691 recorded, the accepted drafts
on the teacher path, and the export of a synthetic drafter. The class is the drafter in PyTorch; no Core AI asset of the
real weights was made or run here (the assets of the real weights are in
[MTP drafter assets with the real weights](#mtp-drafter-assets-with-the-real-weights-713)).
[`mtp-drafter-comparison-20261123.json`](mtp-drafter-comparison-20261123.json) and
[`mtp-drafter-acceptance-20261123.json`](mtp-drafter-acceptance-20261123.json) are the JSON results that
`mtp.py compare` and `acceptance` print, and [`mtp-drafter-export-20261123.json`](mtp-drafter-export-20261123.json) is
the `export.json` that `mtp.py export` writes. They hold item ids, positions, numbers, hashes, versions and fixed names
(the export also the synthetic configuration), and no token sequences, text, local paths, commands or environment
variables; the run folders and arrays stay in the local work directory.

### Conditions

- Apple M4 Max, 128 GiB, macOS 27.0 (26A428), Xcode 27.0 (27A266a). The runs of both drafters, the comparison and the
  accepted drafts held the shared machine lock in one session (2026-11-23 19:24:00 to 19:34:18 JST, on AC power), the
  export in an earlier one (14:30:08 to 14:41:56 JST, with the battery at 100% at both ends). Each session took the lock
  only when it was free and no `coreai-verify`, `coreai-chat` or demo-app process was running; the times below are those
  of single runs, not timing measurements.
- Commit 977309b1 with no local changes; the export at commit 15d35fd8, whose drafter class, synthetic configuration
  and export function are those of 977309b1. The reference environment of the recipe (torch 2.15.0,
  transformers 5.16.1) ran the original drafter and the comparison; the converter environment (torch 2.9.0, transformers
  5.12.1, coreai-torch 0.4.3, coreai-core 1.0.0b3, coreai-models at `upstream_revision`) ran the class, the accepted
  drafts and the export.
- `google/gemma-4-31B-it-assistant@627c5ec1458b9086b841a91e0512fd31fd2fbbf1` (`assistant.json`) and the embedding table
  of `model-00001-of-00002.safetensors` of the pinned target, every tensor rounded to float16 and computed in float32
  on the CPU. The features are those of #691: the original implementation in BF16 on MPS under teacher forcing with the
  reference outputs of 48 items of the primary set. Seven draft steps, stop tokens 1, 106 and 50.
- The arrays of both runs and every compared value equal those of the runs made before the tools checked their inputs
  as they do now; only the times, the memory and the hashes of the records differ.

### Drafter class and original drafter

On a synthetic assistant (`OneStepTests` and `UnrolledTests` in `Tools/Conversion/tests/export_gemma4_31b_drafter.py`,
against the drafter of Transformers 5.12.1), the logits and the `post_projection` output of one step were within 1e-4 of
those of the original drafter (the bound these tests assert) at positions 1, 1,024, 1,025, 1,026, 4,000 and 16,895 and
with large values at both sides of the window edge, and the unrolled drafts of 1, 2 and 7 steps were those of the
original generator. `BuildTests` requires `assistant_config` to refuse 11 of 46 configurations at boundary values,
builds both drafters from the other 35, and requires finite outputs of one step, and non-finite ones from a non-finite
hidden state. On the real weights the comparison passed for 47 rows of 13 items, the first item of each category:

| Item | Anchors a (output index) | w | Largest absolute logit difference | Largest projection ratio |
|---|---|---|---:|---:|
| `ja-choice-0001` | 91 (0) | 91 | 1.14e-05 | 3.40e-07 |
| `ja-reading-0001` | 156 (0), 157 (1) | 156, 157 | 1.53e-05 | 3.24e-07 |
| `code-gen-he-0001` | 176 (0), 239 (63), 303 (127), 366 (190) | 176, 239, 303, 366 | 1.53e-05 | 3.82e-07 |
| `code-gen-mbpp-0001` | 202 (0), 225 (23), 248 (46), 271 (69) | 202, 225, 248, 271 | 1.53e-05 | 2.26e-07 |
| `code-fix-0001` | 242 (0), 257 (15), 272 (30), 287 (45) | 242, 257, 272, 287 | 9.54e-06 | 1.90e-07 |
| `extract-ja-0001` | 137 (0), 172 (35), 208 (71), 243 (106) | 137, 172, 208, 243 | 1.53e-05 | 2.46e-07 |
| `long-ja-4k-0001` | 3,647 (0), 3,648 (1), 3,649 (2), 3,650 (3) | 1,025 | 1.14e-05 | 2.36e-07 |
| `long-ja-16k-0001` | 14,875 (0), 14,876 (1), 14,878 (3), 14,879 (4) | 1,025 | 9.54e-06 | 3.89e-07 |
| `synth-extract-0001` | 209 (0), 294 (85), 380 (171), 465 (256) | 209, 294, 380, 465 | 1.91e-05 | 3.72e-07 |
| `synth-format-0001` | 120 (0), 129 (9), 138 (18), 147 (27) | 120, 129, 138, 147 | 6.68e-06 | 3.15e-07 |
| `free-ja-0001` | 28 (0), 113 (85), 198 (170), 283 (255) | 28, 113, 198, 283 | 1.14e-05 | 2.98e-07 |
| `synth-long-4k-0001` | 3,791 (0), 3,804 (13), 3,817 (26), 3,830 (39) | 1,025 | 1.14e-05 | 3.35e-07 |
| `synth-long-16k-0001` | 15,345 (0), 15,372 (27), 15,399 (54), 15,426 (81) | 1,025 | 1.14e-05 | 3.36e-07 |

The comparison passes when every row has a largest absolute logit difference of at most 1e-3, a projection ratio (the
largest absolute difference of the `post_projection` output over its largest absolute value in the original drafter) of
at most 1e-3, and unrolled drafts equal to one step repeated (the JSON lists the two bounds as `criteria` and the three
checks of each row in `rows[].checks`); no row has an original `post_projection` output that is all zero. `compare`
checked both records against what it built from the fixed inputs: the two contracts, the pins of the index and of the
shard with the embedding table and of the two files of the assistant, `subset.json` with the input and output records
that it pins, the features of the 13 items hashed again, the 47 rows of the fixed selection, the stop tokens, window,
layers, embedding scale, weight rounding, seven steps and the schema name; the package versions, times and memory of the
runs only had to be present. At all 47 rows the unrolled drafts of seven steps were those of one step repeated seven
times; the argmax was the same and the unrolled drafts cut after the first stop token were the drafts of the original
generator, which are records, not criteria. `ja-choice-0001` has one output token, so its four anchors are one. The run
of the original drafter took 28.7 s (7.0 s after the files to read were hashed), that of the class 39.1 s (19.4 s);
kernel lifetime footprint peak 28,484,473,640 B and 31,658,892,304 B.

### Accepted drafts on the teacher path

The class drafted seven tokens at each of the 4,462 anchors of the 48 items; a round accepts the leading drafts, up to
and including the first stop token, that match the reference output and adds one token of the target. The rounds and the
tokens of every category are those of #691 (the fixed position with seven drafts: `trajectories.json` of its records,
whose tokens per round §4.5 of its conclusion shows to two places), which the per-anchor records of #691 also gave
again:

| Category | Items | Anchors | Rounds | Tokens | Tokens per round | #691: rounds, tokens, tokens per round | Anchors with another accepted count | Largest #691 margin there |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| ja-choice | 4 | 4 | 4 | 8 | 2.0000 | 4, 8, 2.0000 | 0 | |
| ja-reading | 4 | 16 | 4 | 20 | 5.0000 | 4, 20, 5.0000 | 0 | |
| code-gen-he | 4 | 585 | 82 | 589 | 7.1829 | 82, 589, 7.1829 | 2 | 0.125 |
| code-gen-mbpp | 4 | 748 | 113 | 751 | 6.6460 | 113, 751, 6.6460 | 4 | 0.125 |
| code-fix | 4 | 259 | 39 | 263 | 6.7436 | 39, 263, 6.7436 | 0 | |
| extract-ja | 6 | 559 | 75 | 563 | 7.5067 | 75, 563, 7.5067 | 0 | |
| long-ja-4k | 2 | 6 | 2 | 8 | 4.0000 | 2, 8, 4.0000 | 0 | |
| long-ja-16k | 1 | 5 | 1 | 6 | 6.0000 | 1, 6, 6.0000 | 0 | |
| synth-extract | 4 | 732 | 96 | 736 | 7.6667 | 96, 736, 7.6667 | 0 | |
| synth-format | 6 | 176 | 58 | 181 | 3.1207 | 58, 181, 3.1207 | 0 | |
| free-ja | 4 | 1,020 | 348 | 1,023 | 2.9397 | 348, 1,023, 2.9397 | 13 | 0.125 |
| synth-long-4k | 3 | 171 | 27 | 174 | 6.4444 | 27, 174, 6.4444 | 1 | 0.0 |
| synth-long-16k | 2 | 181 | 41 | 182 | 4.4390 | 41, 182, 4.4390 | 1 | 0.0 |
| all | 48 | 4,462 | 890 | 4,504 | 5.0607 | 890, 4,504, 5.0607 | 21 | 0.125 |

At 21 anchors the number of accepted drafts differs from that of the records of #691, counted up to seven drafts; at the
first draft that differs, the top-two logit margin that #691 recorded in BF16 was at most 0.125 at each of the 21
(`synth-long-4k` and `synth-long-16k` have one differing anchor each, with the margin 0.0; the JSON holds the largest
margin of each category, not one per anchor). These are records, not criteria. The accepted counts of the records of
#691, computed again from their drafts and the reference outputs, equal the recorded ones at all 4,462 anchors, and the
tokens per round of `trajectories.json` are its tokens over its rounds rounded to four places in the 14 rows (13
categories and pooled) of its fixed-position level with seven drafts, which the tool reads. The run took 548.3 s (525 s
of drafting); kernel lifetime footprint peak 65,812,085,752 B.

### Export of a synthetic drafter

`mtp.py export --steps 7 --context 16896 --seed 11 --output <new folder>` converted the class with the random weights of
`SYNTHETIC` (vocabulary 301, hidden size 32, target width 48; the PyTorch model holds 74,372 float16 elements in 49
tensors: 74,368 in 45 parameters and 4 in the 4 buffers `layer_scalar`) in 72.4 s, and every asset passed its
inspection:

| Asset | Functions | Bytes | `main.mlirb` | Float16 | `argmax` | `gather_nd` | Conversion |
|---|---|---:|---:|---:|---:|---:|---:|
| One step | `draft_1` | 175,724 | 175,587 | 72,839 | 1 | 4 | 5.3 s |
| Unrolled | `draft_7` | 235,331 | 235,194 | 74,375 | 7 | 10 | 29.4 s |
| Both functions | `draft_1`, `draft_7` | 249,409 | 249,272 | 74,375 | 8 | 11 | 35.0 s |

The inputs of both functions are `input_ids` int32 [1, 1], `hidden_states` float16 [1, 1, 48], `position_ids` int32
[1, 1], `k_sliding` and `v_sliding` float16 [1, 1, 2, ?, 16] and `k_full` and `v_full` float16 [1, 1, 1, ?, 32]; the
output is `draft_tokens` int32 [1, 1] or [1, 7], and there are no states. `Float16` is the number of Float16 elements
that an asset stores (`storage_types` of its summary), a total over the asset and not over a function. The asset with
both functions holds the Float16 count of the unrolled asset alone. The one-step asset holds 1,536 Float16 elements
fewer, as many as the weight of `post_projection` has (48 × 32), whose output one step does not read. Each count is 3
above the elements of the model that the functions of the asset use: 74,372 for the unrolled asset and for the asset
with both functions, 74,372 − 1,536 = 72,836 for the one-step asset.

### Limits

- PyTorch in float32 on the CPU with the weights rounded to float16; no Core AI asset of the real drafter, no FP16 or
  8-bit numerics of the drafter, no timing.
- The features are those of the original implementation in BF16 on MPS, not of the Core AI target, and the accepted
  drafts are teacher forcing on the reference outputs, not generation.
- The synthetic checks compare with the drafter of the converter environment (Transformers 5.12.1); the comparison on
  the real weights is with Transformers 5.16.1.
- The export is of a synthetic drafter; the size of the asset of the real drafter is not known from it (see the next
  section).

### Sources

| File | SHA-256 |
|---|---|
| `record.json` of the run of the original drafter | `8c87bd62f3fe96c42a813b23061ca6c86233383bce15faa9f1fad89ff20d2b80` |
| `record.json` of the run of the class | `04ea4763d7c026894a0f723644b716e95a5e3342331147781309d8a0ef0503c6` |
| `subset.json` of #691 | `7ec2c60f8d453a3626f167d0a2187dcbbd4e056db70dab5ccc9adda4bfcc6dc9` |

~~~sh
.local/conversion/gemma4_31b/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.mtp compare \
  --original <run of the original drafter> --model-class <run of the class> \
  --model Tools/Conversion/recipes/gemma4_31b/model.json --assistant Tools/Conversion/recipes/gemma4_31b/assistant.json \
  --snapshot <target snapshot> --records <records of #691> --inputs <inputs.jsonl> --outputs <outputs.jsonl> --steps 7 \
  > docs/validation/gemma4-31b/mtp-drafter-comparison-20261123.json
.local/conversion/upstream-gemma4-31b/coreai-models/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.mtp \
  acceptance --model Tools/Conversion/recipes/gemma4_31b/model.json \
  --assistant Tools/Conversion/recipes/gemma4_31b/assistant.json --snapshot <target snapshot> \
  --assistant-snapshot <assistant snapshot> --records <records of #691> --inputs <inputs.jsonl> \
  --outputs <outputs.jsonl> --steps 7 > docs/validation/gemma4-31b/mtp-drafter-acceptance-20261123.json
~~~

## MTP drafter assets with the real weights (#713)

This section copies the numbers of the `export.json` that `mtp.py export-weights` wrote for the MTP drafter
`google/gemma-4-31B-it-assistant`
([Export with the real weights](../../../Tools/Conversion/recipes/gemma4_31b/README.md#export-with-the-real-weights)):
two Core AI assets of the drafter class with the real weights in float16, `verification.aimodel` (one step, with the
logits over the whole vocabulary and the output of `post_projection`) and `distribution.aimodel` (seven unrolled steps,
with the draft tokens). The assets were inspected here, not run: the next work (#714) runs them through Core AI and
compares them with the class.

### Conditions

- Apple M4 Max, 128 GiB, macOS 27.0 (26A428), Xcode 27.0 (27A266a). The tool ran in the converter environment of the
  recipe (Python 3.11, torch 2.9.0, transformers 5.12.1, coreai-torch 0.4.3, coreai-core 1.0.0b3, coreai-opt 0.4.0,
  coreai-models 0.1.0, numpy 2.3.5, safetensors 0.7.0) with `HF_HUB_OFFLINE=1`, in one run that held the shared machine lock
  from 2026-11-24 14:10:09 to 14:11:33 JST (84 s, on AC power with the battery at 100%; no `coreai-verify`, `coreai-chat` or
  demo-app process was running). The times below are those of that run, not timing measurements.
- Commit 1553f573 with no local changes. Three earlier runs, at commit 475e9134 (before the tool read the contracts once
  and read `main.hash` first), at commit 542f17c4 (before it checked the files again after the assets were made) and at
  commit ffab7889 (before an interruption while the handler of SIGTERM is set again was recorded as a failure), had the
  same descriptors of the functions (the inputs, outputs and states, and the order of the outputs), the same counts of
  stored elements by type and of dequantizations that the inspection read from each asset, the same tensors and elements
  by type in the PyTorch model, and the same pinned inputs. The values inside the assets were not compared.
- Inputs: `google/gemma-4-31B-it-assistant@627c5ec1458b9086b841a91e0512fd31fd2fbbf1` (`assistant.json`, SHA-256
  `0878b84df2892e4db2f019255f3f6f1f8bb5afcceda09c873f6262852bac160f`; `config.json` 2,316 B and `model.safetensors`
  939,042,560 B) and the embedding table of `model-00001-of-00002.safetensors` of
  `google/gemma-4-31B-it@842da3794eaa0b77d5f08bae87a17459d91ff475` (`model.json`, SHA-256
  `8729e7d4e1d4c9dcc8616bc968e9292f4fa35b952567c2a220aa3c8321ab9531`; the shard is 49,784,788,364 B, SHA-256
  `eeef8791537bc04f110967c513149e037d2a9ae97d49add7291ebfa62806bbfa`, and the index 120,246 B), every tensor rounded to
  float16. `--steps 7`, `--context 16896`.

### Assets

| Asset | Function | Outputs | Bytes | `main.mlirb` | Conversion | Save |
|---|---|---|---:|---:|---:|---:|
| `verification.aimodel` | `verify_step` | `logits` float16 [1, 1, 262144], `post_projection` float16 [1, 1, 5376] | 3,757,639,741 | 3,757,639,604 | 5.60 s | 3.20 s |
| `distribution.aimodel` | `draft_7` | `draft_tokens` int32 [1, 7] | 3,757,696,192 | 3,757,696,055 | 29.37 s | 3.23 s |

Conversion and Save are the `convert_s` and `save_s` of each asset; they and `total_s` do not include the checks of the
files before and after the assets were made.

Both functions have no state and the seven inputs `input_ids` int32 [1, 1], `hidden_states` float16 [1, 1, 5376],
`position_ids` int32 [1, 1], `k_sliding` and `v_sliding` float16 [1, 1, 16, ?, 256] and `k_full` and `v_full` float16
[1, 1, 4, ?, 512]. Each asset folder holds `main.hash` (32 B), `main.mlirb` and `metadata.json` (105 B); the SHA-256 of the
files, which `export.json` records:

| File | SHA-256 |
|---|---|
| `verification.aimodel/main.hash` | `89f7a8780095b9bece28e96b343b24d367ae4b4d56f5714e41578d4244ca7d3f` |
| `verification.aimodel/main.mlirb` | `5378f7acc8e3cb34b2d056e61de6e78cf04c88cf9a134df07f9a3fdc6033f33e` |
| `verification.aimodel/metadata.json` | `4b39bb499ac6069ac1b035520d4d54db73dc34318a1109ec30a3c19e789f6d23` |
| `distribution.aimodel/main.hash` | `4c2d0edd2e04e5d0b097764c063621ca2c315ca04d092c27f3810c45e968e161` |
| `distribution.aimodel/main.mlirb` | `f358241c28920457c377d62ea34e89bf22b889b8228c2f979674f820aa752144` |
| `distribution.aimodel/metadata.json` | `ab4b4b4283589cd20486e9f0b5a84ff483f37b6f660e4531184243adcc5840d3` |

The content of `main.hash` is the SHA-256 of `main.mlirb` in both assets (`main_hash` of the record is
`5378f7acc8e3…` and `f358241c2892…`).

### Inspection

`passed` is true and `problems` is empty for both assets and for the output folder: exactly the functions `verify_step`
and `draft_7`, the inputs and outputs above with the dynamic dimensions where `?` stands, no state, the outputs in the
order above, and exactly the three files in each asset folder, as regular files. The PyTorch model of each asset holds
1,878,804,740 elements in 49 tensors, all float16, with no quantized layout. The asset stores 1,878,804,230 (`verification`)
and 1,878,804,231 (`distribution`) Float16 elements next to 386 Float32, 113 Int32, 61 UInt32 and 1 UInt64 element.

### Resources

The tool took 84.56 s in all (`/usr/bin/time -l`). Of that, 44.81 s are `total_s` of the record, from the load of the
embedding table (`load_s` 0.30 s) to the end of the inspections and of the reading of the versions of the packages. The
other 39.75 s lie outside `total_s`; they contain the start of the process, the checks of the files before the output
folder is made, the same checks again after the assets were made (the hashing of the shard in each) and the write of
`export.json`, and the time of each of these was not measured on its own. This run and the run at commit ffab7889, which
checked the files a second time, took 84.56 s and 87.15 s; the runs at commits 542f17c4 and 475e9134, which checked them
once, took 66.67 s and 66.71 s. The time of the second check was not measured on its own, and it is not in `total_s`.
The largest resident set of the process was 19,285,934,080 B and the lifetime peak of its footprint 15,753,804,248 B; it
had no swaps. At commit ffab7889 the two values were 19,285,114,880 B and 15,753,001,456 B, and the run wrote
`main.mlirb` files of 3,757,639,617 B (`verification`) and 3,757,696,068 B (`distribution`); at commit 542f17c4 they
were 19,286,622,208 B and 15,754,115,520 B, and the run wrote 3,757,639,625 B and 3,757,696,076 B; at commit 475e9134
they were 19,287,031,808 B and 15,753,165,296 B, and the run wrote 3,757,639,619 B and 3,757,696,070 B. The four runs
differ in the size of `main.mlirb` by a few bytes and in every `main.hash`; the cause was not examined.

### Form that the verification CLI accepts

`coreai-verify function` of #710 (the build of the `Tools/Verify` tree of this commit) was given the inputs of both
assets in the shapes of `k_sliding` and `v_sliding` [1, 1, 16, 5, 256] and `k_full` and `v_full` [1, 1, 4, 9, 512], with
`v_full.bin` one byte short. For `verification.aimodel` with `verify_step` and for `distribution.aimodel` with `draft_7`
it accepted the function, the names, the types and the shapes of the seven inputs and stopped at the size of the file
(`v_full.bin has 36863 bytes, but its shape and dtype need 36864.`, exit code 64), before it specialized the asset.

### Limits

- The assets were not specialized or run; nothing here says that Core AI computes the drafter as the class does.
- The inspection compares the descriptors, the types of the weights of the PyTorch model and the files of the assets, not
  their values.
- The tool does not check that the converter environment is at the `upstream_revision` of `model.json`; the record names
  the versions of the packages only, as `run` and the export of a synthetic drafter do.
- Between its two checks of the files the tool opens the configuration and the weights of the assistant and the embedding
  table again by their paths. A file that is replaced in between is read as it was replaced, and the second check then
  finds a difference that is still there when it runs (`failure.json` stays and `export.json` is not written); a change
  that was undone before then is not found.
- An output folder that holds neither `export.json` nor `failure.json` is unfinished. Another interruption that arrives
  between the start of the handling of an error or an interruption and the start of the section that writes
  `failure.json` can leave it so; the two other cases that are not guaranteed are in
  [Export with the real weights](../../../Tools/Conversion/recipes/gemma4_31b/README.md#export-with-the-real-weights).
- One run of this commit; the four runs wrote different bytes for the same inputs.

### Sources

| File | SHA-256 |
|---|---|
| `export.json` of the run | `b707e9390d4712e4ee00cb08503867814aca190808d72ee804790be138de9690` |

~~~sh
.local/conversion/upstream-gemma4-31b/coreai-models/.venv/bin/python -m Tools.Conversion.adapters.gemma4_31b.mtp \
  export-weights --model Tools/Conversion/recipes/gemma4_31b/model.json \
  --assistant Tools/Conversion/recipes/gemma4_31b/assistant.json \
  --snapshot .local/conversion/models/google--gemma-4-31B-it \
  --assistant-snapshot .local/conversion/models/google--gemma-4-31B-it-assistant \
  --steps 7 --context 16896 --output .local/conversion/gemma4_31b/drafter-assets
~~~
