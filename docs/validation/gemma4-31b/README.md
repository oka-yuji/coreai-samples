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
  (the system power log shows no change of power source from 02:00 to 03:31). Each run held the shared machine lock
  from its input fingerprints to its last check, and no other model process ran when either run started.
- Model: revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, all 12 files (62,578,686,256 B). Weight shards
  `model-00001-of-00002.safetensors` (49,784,788,364 B, SHA-256
  `eeef8791537bc04f110967c513149e037d2a9ae97d49add7291ebfa62806bbfa`) and `model-00002-of-00002.safetensors`
  (12,761,549,884 B, `018912220f559f7025d60333e0996183cd538aa77ad6f4988a89ce47be681f10`).
- Contract `model.json` SHA-256 `8729e7d4e1d4c9dcc8616bc968e9292f4fa35b952567c2a220aa3c8321ab9531`; fixtures
  `a86c373c60bc9f86df95b2d94c9f068b34d536577109993194599c282d67e75e`.
- Environment: Python 3.11.9, torch 2.15.0, transformers 5.16.1, accelerate 1.15.0, tokenizers 0.23.3, numpy 2.4.6,
  safetensors 0.8.0, huggingface-hub 1.33.0, jinja2 3.1.6.
- Runs (2026-11-22 JST, run IDs in UTC): `20261121T181402-3993e870` from 03:13:59 and `20261121T182226-ec24068e`
  (`--refresh`, a new process for each step) from 03:22:25. Both ran commit 4863d3a with no local changes.

## What the runs checked

- `prepare` hashed all 12 files again and found the sizes and SHA-256 of the contract and of the integrity record of
  the download. `config.json`, `generation_config.json` and the special tokens matched the contract (stop tokens
  1 `<eos>`, 106 `<turn|>` and 50 `<|tool_response>`; BOS 2), and the tokenizer that `AutoTokenizer` builds
  (`GemmaTokenizer`) equalled the distributed `tokenizer.json`.
- `reference` loaded `Gemma4ForConditionalGeneration` with `dtype=torch.bfloat16`, `device_map="mps"` and
  `output_loading_info=True`. The missing, unexpected and mismatched key lists and the error list were empty; all
  31,273,086,512 parameters (the tied embedding counted once) were BF16 on `mps:0`, attention was `sdpa`, and the
  output embedding was the input embedding. Inputs were text only; the vision tower was loaded but not called.
- Conversations were rendered by the distributed template with `add_generation_prompt=True` and
  `enable_thinking=False`, which ends the prompt with an empty thought channel, and tokenized with
  `add_special_tokens=False`; the template writes the single BOS. The raw case got an explicit BOS. Greedy decoding
  used `generate` with `do_sample=False`, one beam and the contract stop tokens.

## Results

Both runs passed every check of `reference.json`: clean loading, no non-finite value, each greedy token at the maximum
of its step logits, the same tokens in a second greedy run within the process, and no monitor stop. The runs in new
processes produced the same input and output tokens for every case, and the saved logits (float32
`[scoring positions, 262144]`) were byte-identical: the largest absolute difference is 0.0.

| Case | Input tokens | Output tokens | Stop | Scoring positions | Non-finite (saved, steps) | Forced argmax = greedy |
|---|---:|---:|---|---:|---|---:|
| `ja_question` | 26 | 64 | limit | 64 | 0, 0 | 64 |
| `ja_system` | 53 | 64 | limit | 64 | 0, 0 | 64 |
| `ja_two_turns` | 64 | 4 | `<turn\|>` | 4 | 0, 0 | 4 |
| `en_question` | 26 | 51 | `<turn\|>` | 51 | 0, 0 | 50 |
| `en_system` | 44 | 58 | `<turn\|>` | 58 | 0, 0 | 58 |
| `code_python` | 51 | 128 | limit | 128 | 0, 0 | 128 |
| `code_swift` | 99 | 128 | limit | 128 | 0, 0 | 128 |
| `json_ja` | 67 | 37 | `<turn\|>` | 37 | 0, 0 | 37 |
| `json_en` | 67 | 39 | `<turn\|>` | 39 | 0, 0 | 39 |
| `ja_long` | 2,042 | 24 | `<turn\|>` | 24 | 0, 0 | 24 |
| `one_word` | 14 | 12 | `<turn\|>` | 12 | 0, 0 | 12 |
| `raw_text` | 15 | 64 | limit | 64 | 0, 0 | 64 |

The table holds the same values for both runs. The last column compares the argmax of the saved logits (one forward
pass of input and output without a cache) with the greedy tokens (`generate` with its cache). They differ at one of
673 positions: position 37 of `en_question`, where the two top candidates of the saved logits differ by 0.124 in
probability. The record keeps this as a diagnostic; it is not a criterion.

The two paths of the same implementation also differ in logit values. For the top candidate of each step, the median
absolute difference is 0 in ten of the eleven conversation cases and 0.125 in `one_word`; the maximum is 0.375 or less
except in `json_ja` (0.875) and `code_swift` (3.25). Without the template, `raw_text` has a median of 0.75 and a
maximum of 7.25; over the whole vocabulary its largest difference is 23.25. Both runs give the same values. The saved
logits are the single-pass ones; of the per-step logits of `generate`, only the top two candidates of each step are
kept in the token files.

## Resources and time

| Run | `20261121T181402-3993e870` | `20261121T182226-ec24068e` |
|---|---:|---:|
| Pipeline | 460.2 s | 420.8 s |
| `prepare` step (hashing all files) | 72.6 s | 69.5 s |
| `reference` step | 384.8 s | 350.0 s |
| Model load | 8.10 s | 6.02 s |
| Greedy, 673 tokens of 12 cases (with prefill) | 149.4 s, 4.51 tokens/s | 131.4 s, 5.12 tokens/s |
| Decode after the first token, short inputs | 4.78 to 6.39 tokens/s | 5.92 to 6.55 tokens/s |
| `ja_long`: first token, decode | 16.41 s, 2.77 tokens/s | 13.60 s, 4.74 tokens/s |
| Peak resident memory | 39,715,127,296 B | 47,944,204,288 B |
| Kernel lifetime footprint peak | 74,743,561,112 B | 74,732,895,152 B |
| MPS driver allocation, after load / peak | 62,556,012,544 B / 72,442,527,744 B | 62,556,012,544 B / 72,440,872,960 B |
| Swap increase (limit 2,000,000,000 B) | 0 B | 0 B |
| Lowest memory free percentage | 37 | 34 |

- The model loaded with the weights in the file cache: `prepare` had just read every file. These are not cold loads.
- The decode rates exclude the first token and the cases with four or fewer output tokens. Each step synchronizes MPS
  once to take its time. The first case of each process also includes the first use of the MPS kernels.
- `torch.mps.current_allocated_memory()` reported 3,840 B after loading, so it does not count the loaded weights;
  the table uses the driver allocation.
- Peak resident memory and the kernel lifetime footprint peak count different things and are not compared with each
  other; the resident peak also depends on how much of the weight files is mapped at the time.

## Limits

- Twelve self-written diagnostic inputs. Five cases stopped at their output limit (64 or 128 tokens). They are not a
  quality set, and they do not cover images, audio, tools or thinking.
- One machine, one OS build and one software environment. The same outputs in two processes do not show that another
  machine, OS or library version gives the same outputs.
- BF16 compute on MPS only. The FP32 reference, the Core AI conversion and the comparison with them are separate
  steps.

## Sources

| File | SHA-256 |
|---|---|
| Report of the first run | `e3626cbc2897b36ea4de398bedb4fc2cad4dadf3d3b78d1e41d2af7374c20c95` |
| Report of the second run | `108605bcb69076eb787be14a539e408e383887de177287cd373b2b5203b0ea0b` |
| `reference.json` of the first run | `dd94662dd1ef1ded46c6ad6a213f20bf8dff1fcbd7bbe9b6395b64e5bd2d4ff1` |
| `reference.json` of the second run | `8e195f25ad98b1b5404cab2ba93492050b925dccbf1f204872ebfa3c8f501434` |
| Comparison of the two runs | `c8a0a03615664f4db1ca0bcf35a017008eeeafcd1734c048e7d5071aa13634b2` |

~~~sh
python3 -m Tools.Conversion.adapters.gemma4_31b.summary \
  --report <first report.json> --reference <first reference.json> \
  --report <second report.json> --reference <second reference.json> \
  --comparison <comparison.json> > docs/validation/gemma4-31b/reference-20261122.json
~~~
