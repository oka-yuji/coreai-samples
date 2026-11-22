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
