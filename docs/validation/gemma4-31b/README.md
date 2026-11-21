# Gemma 4 31B BF16 reference record

This record copies the numbers of two runs of the [`gemma4_31b` recipe](../../../Tools/Conversion/recipes/gemma4_31b/README.md)
(#660): the outputs of the original Transformers implementation of
[google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) with its distributed BF16 weights, BF16 compute
and MPS, for the twelve self-written diagnostic inputs of the recipe. Later steps (the FP32 reference and the Core AI
comparison) read the local outputs of these runs. This is not a quality evaluation, and it does not measure Core AI.

[`reference-20261122.json`](reference-20261122.json) holds the states, numbers, hashes, versions and case IDs of both
runs and of their comparison. It has no fixture text, local paths, commands or environment variables. The run outputs
(the `ready.json` and `reference.json` files, the token files and the saved logits) stay in the local work directory.

## Conditions

- Mac16,6 (Apple M4 Max, 16 cores) with 128 GiB of memory, macOS 27.0 (26A428), Xcode 27.0 (27A266a), on AC power
  at the start and end of both runs (`pmset -g batt`; the power log has no change of power source between 06:20 and
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
- Runs (2026-11-22 JST, run IDs in UTC): `20261121T214327-72fc48b5` from 06:43:23 and `20261121T215114-6e010de9`
  (`--refresh`, a new process for each step) from 06:51:13. Both ran commit 2c3781c with no local changes.
- The four runs before the code changes stay in the local work directory: `20261121T181402-3993e870` and
  `20261121T182226-ec24068e` saved only the scoring rows, and `20261121T203620-82b17299` and `20261121T204316-1480d82e`
  ran before the checks of the loaded files, the first monitor sample and the second greedy run were added. Their
  tokens and their logits (the scoring rows of the first two, every saved row of the last two) are identical to
  these runs.

## What the runs checked

- `prepare` hashed all 12 files again and found the sizes and SHA-256 of the contract and of the integrity record of
  the download. `config.json`, `generation_config.json` and the special tokens matched the contract (stop tokens
  1 `<eos>`, 106 `<turn|>` and 50 `<|tool_response>`; BOS 2), and the tokenizer that `AutoTokenizer` builds
  (`GemmaTokenizer`) equalled the distributed `tokenizer.json`.
- `reference` started only after the monitor's first sample read swap and the free percentage, and hashed the
  files it loads (both weight shards, `config.json`, `generation_config.json`, `model.safetensors.index.json` and the
  three tokenizer files) right before loading them: every size and SHA-256 equalled the contract, and the weight
  hashes in `reference.json` come from this check. It then loaded `Gemma4ForConditionalGeneration` with
  `dtype=torch.bfloat16`, `device_map="mps"` and
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
  no swap above 16,000,000,000 B rising for three samples, and every sample read swap and the free percentage.

## Results

Both runs passed every check of `reference.json`: clean loading, no non-finite value in any saved row or in the
step logits of either greedy run, each greedy token at the maximum of its step logits, the same tokens in a second
greedy run within the process, a clean monitor and recorded resources. The runs in new processes produced the same input and output tokens for every
case, and the saved logits of all 3,241 positions were byte-identical: the largest absolute difference is 0.0. The
comparison also found both `reference.json` files passed and every saved array float32 with the shape of its token
file.

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

| Run | `20261121T214327-72fc48b5` | `20261121T215114-6e010de9` |
|---|---:|---:|
| Pipeline | 469.3 s | 440.2 s |
| `prepare` step (hashing all files) | 76.1 s | 68.3 s |
| `reference` step | 389.4 s | 370.6 s |
| Hashing the loaded files before loading | 22.52 s | 21.03 s |
| Model load | 6.48 s | 5.74 s |
| Greedy, 673 tokens of 12 cases (with prefill) | 132.9 s, 5.06 tokens/s | 129.4 s, 5.20 tokens/s |
| Decode after the first token, short inputs | 5.63 to 6.48 tokens/s | 5.90 to 6.56 tokens/s |
| `ja_long`: first token, decode | 14.10 s, 4.44 tokens/s | 12.98 s, 4.85 tokens/s |
| Peak resident memory | 60,409,315,328 B | 57,864,798,208 B |
| Kernel lifetime footprint peak | 79,959,655,456 B | 79,961,949,144 B |
| MPS driver allocation, after load / peak | 62,556,012,544 B / 74,663,854,080 B | 62,556,012,544 B / 75,108,450,304 B |
| Swap increase (limit 2,000,000,000 B) | 0 B | 0 B |
| Lowest memory free percentage | 40 | 39 |
| Saved logits (12 files) | 3,398,436,352 B | 3,398,436,352 B |

- The model loaded with the weights in the file cache: `prepare` had just read every file. These are not cold loads.
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
| Report of the first run | `ee72144a77f8946155c8853b005efd9ddd64ac56f13b8693d47807feec3fb27b` |
| Report of the second run | `b9cded5c844f2c4d32cb6ffcd0943eb6479ce33078d77a25329b8ce1e5f35f2b` |
| `reference.json` of the first run | `50587bde70990bd8caa6438e367310ce77b39d53af46cbb1a8b2077b5176a7d9` |
| `reference.json` of the second run | `e42e060c50214f5d7c1430f747659c228ab4bfef9bc6067315eda6c8380824cc` |
| Comparison of the two runs | `aaf5c99d9c720255ed0f13cfc0a7b9b398af1c8ed511aa177ed2afd78a418864` |

~~~sh
python3 -m Tools.Conversion.adapters.gemma4_31b.summary \
  --report <first report.json> --reference <first reference.json> \
  --report <second report.json> --reference <second reference.json> \
  --comparison <comparison.json> > docs/validation/gemma4-31b/reference-20261122.json
~~~
