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
