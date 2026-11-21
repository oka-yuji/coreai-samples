# Gemma 4 31B quality contract

| Field | Value |
|---|---|
| Version | `G31-QC-v1` |
| Frozen | 2026-11-22 |
| Issue | #666 (parent #658) |

| File | SHA-256 |
|---|---|
| `Tools/LLMEvaluation/contract.json` | `1e8143fb6dad5529432b7ee60b7422383de1e13397706d38e629dbb3f949c40f` |
| `Tools/LLMEvaluation/sets/primary.json` | `06aef4839d2f9c2df36a4597f28fb050587ea4b1ff4dd14737e01a08a8d247f4` |
| `Tools/LLMEvaluation/sets/reserve.json` | `cf38e09f6efb2d1beb8224f2211f14ef88946c1eed31a8c72f6d44f6861adbba` |
| `Tools/LLMEvaluation/prompts.json` | `8dcb70749a2cfd43d39b44849d7ea577116ae93bb2ee4b646dbcb1c4e8a81cf8` |

`python3 -m Tools.LLMEvaluation validate`, run from the repository root, rejects the contract when this table, the version or the freeze date differs from the files. `check.sh` runs the same check in isolated mode as `python3 -I Tools/LLMEvaluation/__main__.py validate`, because isolated mode does not put the repository on the import path. The [README](../../Tools/LLMEvaluation/README.md) describes both forms and the other tools. `Tools/LLMEvaluation/contract.json` holds the constants, the lists, and the verdict labels with their precedence. The conditions of the four verdicts and the prohibitions of §7.3 are only in this document. Whether the numbers in this text agree with `contract.json` is checked by a machine only through the SHA-256 table above: the table fixes the reviewed `contract.json`, and no tool compares this text with it.

This contract is fixed before any candidate is measured. It decides how quality is judged for the Core AI conversions of `google/gemma-4-31B-it` (FP16, 8-bit weights and the conditional 4-bit candidate). It does not contain the correctness rules for speculative decoding or the details of the performance measurement plan; each of those is fixed in its own version before its own measurement.

## 1. Binding conditions from the parent issue

The parent issue #658 states these conditions. They are quoted in the original and apply to every rule in this contract.

> 品質低下を許す予算は承認されていない。
>
> 数値誤差の許容を課題品質の劣化許容へ読み替えない。
>
> 日本語、コード、抽出/構造化出力、長文脈を分ける。平均値だけで相殺せず、例ごとの回帰と重大な失敗を確認する。差が判断できない候補を「劣化なし」としない。
>
> 4bitで低下が検出されなくても、その評価範囲の結果として報告する。全入力に対する品質同一保証には広げない。
>
> 予想が外れた場合も条件と結果を残す。遅い候補や不合格の記録を削除せず、結果を見て基準を緩めない。

In English: no budget for quality loss is approved; a numerical tolerance is never read as a tolerance for task-quality loss; Japanese, code, extraction or structured output and long context are evaluated separately, without offsetting by averages, with per-example regressions and critical failures checked, and a candidate whose difference cannot be judged is not called free of degradation; a 4-bit result without detected loss is reported only for this evaluation range; unexpected results, slow candidates and failures are kept, and criteria are not relaxed after seeing results.

## 2. Roles and pairs

| Role | Meaning |
|---|---|
| `reference` | Original implementation, original BF16 weights, BF16 compute |
| `anchor` | The same implementation with FP32 compute |
| `fp16` | Core AI, FP16 |
| `w8` | Core AI, 8-bit weights |
| `w4` | Core AI, 4-bit weights; conditional candidate |

Pairs are written baseline → candidate: `reference → fp16`, `fp16 → w8`, `fp16 → w4`. When `fp16` cannot be run, `reference` is the substitute baseline and the reason is recorded.

## 3. Two layers

Layer 1 checks numerical correctness of an implementation. Layer 2 judges task quality. A pass in one layer is never read as a pass in the other.

## 4. Layer 1: numerical checks

Layer 1 compares the same weights through different implementations or compute, for example `anchor` and the Core AI asset made from the same weights. A fixed token sequence is forced as input, and the probability distribution (softmax at temperature 1) is compared at every scored position.

All of these must hold:

| Check | Limit |
|---|---|
| Non-finite values | 0 |
| Maximum probability error, the largest \|p − q\| over the full vocabulary, at every position | ≤ 0.03 |
| Mean of the per-position maximum error within each input-length band | ≤ 0.01 |
| Top-1 mismatches that are not near ties | 0 |

- **Near tie, different implementations** (the rule of `Tools/Conversion/adapters/verify_llm.py`): both sides have the same top-2 token set, and the reference-probability gap between the reference winner and the candidate winner is ≤ 0.01.
- **Near tie, same assets called differently** (for example chunk length, or speculation on and off): only the top-2 tokens swapped, and on each side the logit distance between its top-2 tokens is within 8 float16 ticks, counted as the difference of ordered float16 bit patterns (`StateCheck` in `CoreAIBackend/Generation.swift`). The maximum probability error ≤ 0.03 is checked separately for these pairs.
- **Bands** by the input token count of the item: 1–512, 513–2,048, 2,049–4,096 and 4,097–16,896. A band without items is reported as not evaluated.
- **Reported only, never a pass condition:** KL divergence (mean, p99, maximum), the maximum absolute logit difference, the top-1 agreement rate and the number of near ties.
- **Items:** from the primary set, the first 8 items of each category in ascending item-id order. The long-context families take 6 items from the 4k category and 2 from the 16k category: `long-ja-4k` 6 and `long-ja-16k` 2, `synth-long-4k` 6 and `synth-long-16k` 2. `free-ja` is included. This gives 88 items. In `G31-QC-v1` they fall into the bands as 71, 1, 12 and 4 items (1–512, 513–2,048, 2,049–4,096 and 4,097–16,896). Inputs longer than the 1,024-token sliding window are checked by the 16 items of 2,049 tokens or more.
- **Forced input and scored positions:** the forced input is the item input followed by the tokens that the `reference` role generated for that item: its `output_ids`, and its stop token when it stopped on one. The scored positions run from the last input position to the position before the last token of the forced input, at most 128 positions per item (the first 128 when there are more). Row i predicts generated token i. Every pair (`fp16`, `w8` and `w4`) uses the same forced input in layer 1.
- **When a limit is exceeded:** the comparison does not pass. Investigate in this order: input ids and weights; positions, RoPE and window masks; the first diverging layer and operation. The limits are not changed.

## 5. Generation conditions and the input frame

- Greedy decoding with ties broken by the lowest token id. No temperature, top-k, top-p or penalties. Stop on `eos_token_id` [1, 106, 50] (`<eos>`, `<turn|>`, `<|tool_response>`). No thinking, no tools, no system message. Quality claims are limited to greedy decoding.
- The input is rendered once from the distributed `chat_template.jinja` and tokenizer of `google/gemma-4-31B-it` at revision `842da3794eaa0b77d5f08bae87a17459d91ff475`. The same token ids are given to the baseline and the candidate.
- The rendering condition is `enable_thinking` false, `add_generation_prompt` true, `bos_token` `<bos>`, no system message, no tools and exactly one user message. The template trims the user content; every item's content already has no leading or trailing whitespace. In this condition, confirmed against template lines 186–194 and 381–389, the rendered input is:

  | Part | Text |
  |---|---|
  | Before the content | `<bos><\|turn>user\n` |
  | After the content | `<turn\|>\n<\|turn>model\n<\|channel>thought\n<channel\|>` |

  The generation therefore starts after `<|turn>model\n` and an empty thought channel.
- `select` renders every item with jinja2 (a sandbox with `trim_blocks`, `lstrip_blocks` and `loopcontrols`), requires the result to equal the frame above, and tokenizes it with `tokenizers` without adding special tokens. The ids must start with [2, 105], and the special tokens must be exactly those of the frame, so no item text produces a control token. The resulting count is `token_count`.
- The run that produces outputs (a separate child issue) renders with transformers. It must confirm that its token sequence has the same length as `token_count` for every item that carries one, and that the rendered text equals the frame around the content. Any difference makes the comparison `incomplete`.
- Sampling with the distributed defaults (temperature 1.0, top_k 64, top_p 0.95) is not evaluated and is not claimed.

### 5.1 Output records

A run writes one record per item of the set, one JSON object per line, in any order. `contract.json` lists the same fields under `outputs`.

| Field | Value |
|---|---|
| `id` | The item id |
| `input_sha256` | SHA-256 of the input token ids written in decimal and joined by commas, as UTF-8 (the `tokens_sha256` of `coreai-verify`, `tokensDigest` in `Tools/Verify`) |
| `input_tokens` | The number of input tokens |
| `output_ids` | The generated token ids before the first stop token |
| `stop` | `stop_token` or `limit`; with `stop_token`, the field `stop_token` holds that token id |
| `top2` | For every generated token, including the stop token: `[first id, first probability, second id, second probability]` of the temperature-1 softmax over the full vocabulary at that step |
| `error` | Absent, or the runtime error that ended the item |
| `non_finite` | The number of non-finite logits seen while generating the item |

- The scored text is `output_ids` decoded with the pinned tokenizer, skipping special tokens. The scorer decodes it; the run does not supply text.
- The first id of every `top2` entry equals the token generated at that step (greedy, ties broken by the lowest id).
- Each run also writes one run record: the contract version and the SHA-256 of `contract.json`, the set name, the role, the model revision or asset hash, the implementation and its version, and the call shape (for example the prefill chunk length).
- Output positions are counted in these records as the generated token index, counting the stop token.

## 6. Layer 2 sets

### 6.1 Categories

| Category | Source | Primary / reserve | Unit | Max new tokens | Correct when |
|---|---|---|---|---|---|
| `ja-choice` | JGLUE JCommonsenseQA v1.1 validation | 200 / 100 | Question | 8 | The first digit 0–4 of the output is the answer |
| `ja-reading` | JGLUE JSQuAD v1.1 train (the same blob as v1.2; see below for v1.3) | 200 / 100 | Article, one question per article | 64 | After normalization it equals one of the answers exactly; character F1 is also recorded |
| `code-gen-he` | HumanEval, even problem numbers | 82 / 0 | Problem | 512 | The extracted code passes the problem's unit tests |
| `code-gen-mbpp` | MBPP sanitized, test split | 100 / 100 | Problem | 512 | The same, with `test_imports` and `test_list` |
| `code-fix` | HumanEvalPack HumanEvalFix, Python, odd problem numbers | 82 / 0 | Problem | 512 | The same |
| `extract-ja` | Stockmark Japanese Wikipedia NER | 200 / 100 | Sentence | 256 | Valid JSON whose string sets for the 8 keys equal the answer; micro-F1 is also recorded |
| `long-ja-4k` | JSQuAD train, articles not used by `ja-reading` | 60 / 30 | Article | 64 | As `ja-reading`, stratified by the answer paragraph position (front, middle, back) |
| `long-ja-16k` | The same | 20 / 10 | Article | 64 | The same |
| `synth-extract` | Synthetic records with nesting and empty lists → JSON | 100 / 50 | Generated template | 384 | The canonical structure equals the answer |
| `synth-long-4k` | Synthetic records; the answer combines two or more distant records | 40 / 20 | Generated template | 32 | The integer after `答え:` equals the answer |
| `synth-long-16k` | The same | 16 / 8 | Generated template | 32 | The same |
| `synth-format` | Synthetic instructions on item count, length, style and required words | 100 / 50 | Generated template | 160 | Every format rule holds; reported as format compliance, not as correctness of meaning |
| `free-ja` | Self-authored Japanese instructions, 10 kinds × 5 | 50 / 0 | Instruction | 256 | Not scored; used only to detect critical failures |

The parent's domains map to these categories: Japanese (`ja-choice`, `ja-reading`, `free-ja`), code (`code-gen-he`, `code-gen-mbpp`, `code-fix`), extraction and structured output (`extract-ja`, `synth-extract`, `synth-format`) and long context (`long-ja-4k`, `long-ja-16k`, `synth-long-4k`, `synth-long-16k`). Verdicts are counted per category, never by averages across categories.

**Why JSQuAD train.** The JSQuAD validation file (JGLUE `c96c6af`, the same blob from v1.1 to v1.3) has only 59 articles (titles), 1,145 paragraphs and 4,442 questions. One question per article cannot give 300 `ja-reading` items, nor 120 long-context question articles and their filler articles. The train file at the same revision has 710 articles, 15,652 paragraphs and 62,859 questions, and no title is shared with validation. The validation articles are not used. Train has been public as long as validation, so it may be in model training data as much as or more than validation. Absolute accuracy on these sets is therefore not comparable with published validation scores. These sets exist for the paired comparison of a baseline and a candidate on identical inputs.

The train file is the same blob (`20ce716e`) in v1.1.0 and v1.2.0. JGLUE v1.3.0 (`6f071c0`) discarded 162 low-quality training questions, keeping all 710 articles and every paragraph. Its train file is therefore a different blob (`87b42821`) and is not used. Of the 420 JSQuAD questions in these sets, only `ja-reading-0137` (question `a549351p10q1`, primary set) is among the discarded ones. The item is kept, and its result is reported with this note.

### 6.2 Sampling

- Seed `20261122`. For each category, the candidate rows are sorted by their source key and drawn with `random.Random("20261122:<category id>")` of the standard library: `sample(rows, primary + reserve)`. The first `primary` draws form the primary set and the rest the reserve set.
- Item ids are `<category id>-NNNN`, numbered in draw order: the primary items first, then the reserve items.
- Source keys: JCommonsenseQA `q_id`; JSQuAD article `aN` and question id `aNpMqK` (paragraph key `aNpM`, all taken from the question ids); HumanEval and HumanEvalPack problem number; MBPP `task_id` in 11–510, the test split of `mbpp/README.md`; NER position in `ner.json` together with its `curid`; synthetic generator index; `free-ja` instruction id.
- JSQuAD: categories draw articles in contract order (`ja-reading`, `long-ja-4k`, `long-ja-16k`), each from articles not yet drawn. The question of an article is the next `choice` of the category generator over its sorted question ids.
- Synthetic categories generate indexes 0, 1, 2, … and skip an index whose combination (see §6.8) already appeared in the category, so the primary and reserve combinations never overlap.
- `select --check` regenerates both set files and requires them to be byte-identical to the committed files. The recorded `selected_with` gives the Python, tokenizers and jinja2 versions used.

### 6.3 Overlap rules

- `code-gen-he` uses even HumanEval problem numbers and `code-fix` odd ones, so no original problem is in both.
- An article is the question article of at most one item across `ja-reading`, `long-ja-4k` and `long-ja-16k`, primary and reserve together.
- Filler paragraphs come only from articles that are not question articles of any primary or reserve item.
- Within each category, the primary and reserve sets share no source key; for synthetic categories, no combination.
- These checks guard different things:
  - `validate` checks the item ids and counts, the numbering of the primary and reserve items, the overlaps of articles and problems, the token-count ranges, the scorer kinds and hashes of the synthetic and `free-ja` items, and the version, freeze date and hash table of this document. It checks that the required constants are present and well formed, not that their values are in range; the values are fixed through the SHA-256 table. A malformed contract can stop it with an exception instead of a `FAIL` line, and it still exits with a non-zero status.
  - `build` checks the pins of the source files, and the hash and scorer kind of every item rebuilt from the pinned data.
  - Only `select --check` checks that the committed sets are exactly the sets that the sampling rules produce. For example, an item replaced by a row that the rules did not draw, together with its new hash, passes `validate` and `build`, and is found only by `select --check`.

### 6.4 Long-context assembly

- **`long-ja-*`.** The context is a list of original JSQuAD context strings (each `title [SEP] paragraph`) joined by a blank line. The answer paragraph is first (front), at position ⌊fillers / 2⌋ (middle) or last (back); the strata are assigned front, middle, back in turn by position within each set. Filler paragraphs are shuffled with `random.Random("20261122:<category id>:<item id>")` and added in that order. A paragraph that would push the rendered input above the upper bound is skipped, and assembly stops at the first count that reaches the lower bound. Fillers may recur across items but not within an item. The ordered paragraph keys are recorded in the set.
- **`synth-long-*`.** The record count is tuned by `select` until the rendered input is in range, and is recorded as `records`.
- **Ranges** of the whole rendered input: 3,600–4,096 tokens for the 4k categories and 14,800–16,384 for the 16k categories. In `G31-QC-v1`: `long-ja-4k` 3,600 / 3,650.5 / 3,788 (minimum / median / maximum over 90 items), `long-ja-16k` 14,807 / 14,855 / 15,043 (30), `synth-long-4k` 3,773 / 3,795.5 / 3,813 (60) and `synth-long-16k` 15,344 / 15,369 / 15,392 (24).

### 6.5 Items and hashes

`build` writes each item with `id`, `category`, `unit`, `messages` (one user message), `max_new_tokens`, `scorer` (kind and arguments), `answer` and, for long-context categories only, `token_count`. `item_sha256` is the SHA-256 of that object as UTF-8 JSON with sorted keys, `ensure_ascii` off and the separators `,` and `:`. The set files hold only `id`, `category`, `unit`, `source` (the source keys, or the generator arguments for synthetic items), `item_sha256` and `token_count`. Public data text is never stored in the repository.

| Kind | `answer` | `scorer` arguments |
|---|---|---|
| `choice_digit` | Answer index | – |
| `reading_exact` | List of answer strings | `stratum` for long-context items |
| `python_tests` | `null` | `setup`, `test`, `call` |
| `entity_sets` | Object with all 8 keys; sorted unique strings | – |
| `json_exact` | Canonical structure | – |
| `final_integer` | Integer | – |
| `format_rules` | `null` | `bullet`, `items`, `max_chars`, `endings`, `required` |
| `unscored` | `null` | – |

### 6.6 Prompts

All wording is in `Tools/LLMEvaluation/prompts.json`. Each instruction states only the output form, with no examples (no few-shot). `ja-choice` asks for the number only. `ja-reading` and `long-ja-*` ask for the phrase from the text only. The code categories, written in English like their sources, ask for a single ```python block. `code-gen-mbpp` shows its `test_imports` and `test_list`, and `code-fix` shows the buggy function and its tests. `extract-ja` asks for JSON with the 8 keys 人名, 法人名, 政治的組織名, その他の組織名, 地名, 施設名, 製品名 and イベント名. `synth-extract` gives the target schema. `synth-long-*` asks for a last line `答え: <整数>`.

### 6.7 Scoring rules

The scorer is a separate child issue; these rules bind it.

- **`ja-choice`:** apply NFKC and take the first character of the output that is one of `0`–`4`.
- **`ja-reading` and `long-ja-*`:** normalize both the output and each answer.
  1. Apply NFKC.
  2. Strip with Python's `str.strip()` without an argument, which removes Unicode whitespace including newlines.
  3. Remove one final `。`.
  4. Strip again with `str.strip()`.
  5. If the text starts with `「` and ends with `」`, or starts with `『` and ends with `』`, remove that first and last character, once. The marks inside are not matched.
  6. Strip again with `str.strip()`.

  The item is correct when the normalized output equals any normalized answer. Character F1 is the multiset overlap of characters after normalization, the maximum over the answers.
- **JSON (`extract-ja`, `synth-extract`):** use the first JSON object, at the first `{` from which a JSON object decodes. A duplicate key, an extra or missing key, or a type difference is incorrect.
  - `extract-ja`: values are arrays of strings, compared as sets ignoring order and duplicates, without normalization. Micro-F1 is over (item, key, string) triples.
  - `synth-extract`: compared as the canonical serialization (sorted keys), which keeps array order (the order of appearance in the records) and distinguishes `1`, `1.0`, `"1"` and `true`.
- **`synth-long-*`:** apply NFKC and take the integer (an optional minus sign and digits) after the last `答え:`. NFKC also accepts the full-width colon.
- **`synth-format`:** split the output into lines with `str.splitlines()`, strip each line with `str.strip()` and drop the empty lines. Every remaining line must start with the bullet, and their number must equal `items`. For each line, the text after the bullet, stripped, has at most `max_chars` Unicode scalar values and ends with one of `endings`. Each required word appears in at least one item.
- **Code:** take the first ```python block; without one, the first ``` block; without one, the whole output. A block runs from its opening fence line to the next fence line or the end of the output. The program is `setup`, the code, `test` and `call`, joined by newlines.
  - HumanEval: the program starts with the problem's prompt as `setup`, which defines its imports and helper functions, followed by the code, the test and `check(<entry point>)`.
  - MBPP: `test_imports` and the `test_list` asserts.
  - HumanEvalFix: its test, which already calls `check`.

  The item is correct when the program finishes normally. Each program runs under these execution rules:
  - **Isolation:** an isolated child process with no network, an empty temporary folder as its working folder, no writes outside that folder and no standard input.
  - **Time limit:** 10 seconds of wall-clock time per program. A program that times out is run once more and is incorrect only when both runs time out. The report gives the number of timeouts and the number of programs that passed on the rerun.
  - **Same run:** the baseline and candidate outputs are scored in the same scoring run, with the same Python version on the same machine. The report records that Python version.
  - The scorer (a separate child issue) implements the isolation. This contract fixes only these rules and constants.

The examples of these rules are in `Tools/LLMEvaluation/tests/test_synthetic.py` and `tests/examples.json`. In those examples, a well-formed answer with a wrong value is incorrect.

### 6.8 Synthetic tasks

The generators in `Tools/LLMEvaluation/synthetic/` build each item from the seed, the category id and the index (`random.Random("20261122:<category id>:<index>")`). They use the self-authored vocabulary in `synthetic/vocabulary.json`, which is pinned by SHA-256 in `contract.json`, and the wording in `prompts.json`. The template is chosen by index parity.

| Category | Templates | Content | Combination key |
|---|---|---|---|
| `synth-extract` | `orders`, `teams` | 2–3 records. Orders have goods with quantities; teams have members and areas, up to 2 each. Every item has at least one empty list and one non-empty list | Template and the people named |
| `synth-long-*` | `stock` | A ledger of stock moves; the answer is the start stock plus every move of one warehouse and good | Template and the target |
| `synth-long-*` | `shipping` | Shipments; the answer is the total that one person sent of one good to one place | Template and the target |
| `synth-format` | – | 3–5 items, 15–25 characters, polite `です。` or `ます。` endings or plain `である。` endings, 1–2 required words | Topic and the required words |

In `synth-long-*`, the 2–4 relevant records sit at 5%, 95% and evenly spaced points between them, and distractors never use the target. `solve.py` computes every answer, and the format rules, again from the rendered text alone, without the generator's state. The tests require both computations to agree. They also require that unrelated records and reordering leave the `synth-long-*` answer unchanged. The passing claim for these tasks is limited to the distribution of these generators. They are made from a self-authored generator and vocabulary; this contract does not claim that they are free of contamination or of licensing questions.

### 6.9 free-ja

`free-ja` has 50 self-authored Japanese instructions: 10 kinds (explanation, steps, letters, summary, rewriting, comparison, story, advice, ideas and English-to-Japanese translation) × 5. The texts and the passages used for summary and translation are in `prompts.json`. The outputs are not scored; they are checked only for critical failures.

## 7. Layer 2 verdicts

### 7.1 Critical failures

An output is the generated tokens before the first stop token.

| Failure | Judged | Condition |
|---|---|---|
| (a) | Candidate only | A runtime error, or non-finite logits |
| (b) | Relative to the baseline | The candidate output has 0 tokens and the baseline output of the same item does not |
| (c) | Relative to the baseline | The candidate output contains a control token and the baseline output of the same item contains none |
| (d) | Relative to the baseline | The candidate output contains a sequence of 8 or more tokens repeated 4 or more times in a row, and the baseline output of the same item contains no such repetition |

The control tokens are the 24 added special tokens of the tokenizer except the stop tokens [1, 106, 50]: ids 0, 2, 3, 4, 46–49, 51, 52, 98, 100, 101, 105, 255999, 256000 and 258880–258884. The report lists the baseline outputs with no tokens, control tokens or repetitions separately, as a record of baseline quality.

### 7.2 Automatic verdicts

Each pair receives exactly one of four labels, in this precedence:

| Precedence | Label | When |
|---|---|---|
| 1 | `incomplete` | Missing outputs; interruption; a version or hash mismatch; a duplicate id; a record whose id is not in the set; baseline and candidate records of the same item with different `input_sha256`; an `error` in a baseline record (an `error` in a candidate record is critical failure (a)); a record with a `top2` entry whose first id differs from the generated token; or a token count that differs from §5 |
| 2 | `critical_failure` | One or more critical failures |
| 3 | `regression_found` | One or more items where the baseline is correct and the candidate is incorrect. Not adopted automatically |
| 4 | `no_regression_found` | No critical failure, and no such item in any scored category |

### 7.3 Prohibited judgments

- Passing because a test is not significant.
- Passing because improvements offset regressions.
- Excluding near ties from regressions.

A candidate with a regression can be adopted only by the owner's explicit decision. That decision is recorded as a contract change, either a new version or a reference to the decision record. Such a candidate is never labeled `no_regression_found`. Baseline outputs for layer 2 are produced only after this contract is frozen.

### 7.4 Required report items

- The item count and the correct count per category, for the baseline and the candidate.
- The ids of every regression and every improvement.
- For each changed item, the first diverging output position and the baseline top-2 probability gap there. Positions are counted as in §5.1.
- Every critical failure, and the separate baseline counts of §7.1.
- A reference to the layer 1 result.
- When the same assets are rerun with a different call shape, the counts of correct → incorrect and incorrect → correct. This indicates the swaps caused by computation order alone; it is not used for the verdict.
- For the code categories, the number of timeouts, the number of programs that passed on the rerun and the Python version used for scoring (§6.7).
- For the categories with independent units (`ja-choice`, `ja-reading`, `code-gen-mbpp`, `extract-ja`), the one-sided 95% upper bound of the regression rate when there are zero regressions, 1 − 0.05^(1/N). For the primary set this is 1.49% at N = 200 and 2.95% at N = 100.

### 7.5 Wording

A pass is written as "no regression was detected on this evaluation set". It is not extended to all inputs, to sampling with the distributed defaults, or beyond the distribution of the synthetic tasks.

A scored category in which the baseline has fewer than 10 correct items (`layer2.min_baseline_correct`) is named in the claim as a category that could not be evaluated, and the claim does not cover it. The verdict label does not change.

## 8. Reserve set

The reserve set is disjoint from the primary set and is not used while the primary set is evaluated. If a procedure, a set or a threshold is changed after results were seen, the seen primary set becomes development data, and the change is confirmed on the reserve set. The reserve set can be used once; the reason, the date and time and the contract version are recorded. `code-gen-he`, `code-fix` and `free-ja` have no reserve items; a change affecting them is reported as unconfirmed for those categories.

## 9. Choosing a candidate

Eligible candidates are those labeled `no_regression_found` and those the owner explicitly approved. Among them, the candidate with the smallest primary metric is chosen. The primary metric is the median total completion time for a single conversation, warm, with 4,096 input tokens, 256 output tokens and greedy decoding. When the min–max ranges of the leading candidates overlap, the higher precision is preferred (FP16 over 8-bit over 4-bit), then no speculation. The measurement details (512 and 16,384 input tokens, sampling with the distributed defaults, first specialization and memory) are fixed by the performance comparison plan.

## 10. Data sources and licenses

Only ids, source keys, hashes, counts and token counts are kept in the repository. Public data text is not redistributed. The files are retrieved anonymously into `.local/conversion/datasets/<folder>/`, which Git ignores. Each folder has an `<folder>.integrity.json` beside it, recording the per-file size, SHA-256 and git blob SHA-1 against the distributor's records. Each license below was confirmed on the distributor's own notice at retrieval on 2026-11-22.

| Data | Distributor and URL | Revision | File and SHA-256 | License as displayed | Credit |
|---|---|---|---|---|---|
| JCommonsenseQA validation | [yahoojapan/JGLUE](https://github.com/yahoojapan/JGLUE/tree/c96c6af54200a404f31a3a4cd0b289786aa78809) | `c96c6af54200a404f31a3a4cd0b289786aa78809` (v1.2.0) | `datasets/jcommonsenseqa-v1.2/valid-v1.2.json`, `0d8d76f3bfa0d174866939882faccdd01fbc2bcd5a76c43748ba0c40a7b3b8d4`; the same blob (`1099679e`) as v1.1.0 and v1.3.0 | CC BY-SA 4.0 (LICENSE and README) | JGLUE, Yahoo Japan Corporation and Kawahara Lab at Waseda University (Kurihara et al., LREC 2022) |
| JSQuAD train | The same repository | The same | `datasets/jsquad-v1.2/train-v1.2.json`, `24a0233a5d2cc80571e499de487fe2188981126312f159fc90136db82f3584b5`; the same blob (`20ce716e`) as v1.1.0 | CC BY-SA 4.0 | The same |
| HumanEval | [openai/human-eval](https://github.com/openai/human-eval/tree/b9453e8f1865df0a0057a1383767caa400acfbcf) | `b9453e8f1865df0a0057a1383767caa400acfbcf` | `data/HumanEval.jsonl.gz`, `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef` | MIT (LICENSE). The README has marked the repository deprecated and unmaintained since 2026-10-16; the data file is unchanged since its first commit | HumanEval, OpenAI (Chen et al., 2021) |
| MBPP sanitized | [google-research/google-research](https://github.com/google-research/google-research/tree/f46ca8374b4cddef97ca4208ad986049d74d296a/mbpp) | `f46ca8374b4cddef97ca4208ad986049d74d296a` | `mbpp/sanitized-mbpp.json`, `ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9` | CC BY 4.0 for datasets (repository README); source files are Apache-2.0 | MBPP, Google Research (Austin et al., 2021) |
| HumanEvalPack Python | [bigcode/humanevalpack](https://huggingface.co/datasets/bigcode/humanevalpack/tree/7980d99405b56c9745d51040cabbd9e86a056397) | `7980d99405b56c9745d51040cabbd9e86a056397`, the last revision with the JSONL files before the Parquet conversion | `data/python/data/humanevalpack.jsonl`, `b1376d72be4bd016a62ad043d9de3846edb8d63edf1222da802a31330c14c3a7` | MIT (dataset card) | HumanEvalPack, BigCode (Muennighoff et al., OctoPack, 2023) |
| Japanese Wikipedia NER | [stockmark/ner-wikipedia-dataset](https://huggingface.co/datasets/stockmark/ner-wikipedia-dataset/tree/d6c4d4821885ed656b9fbc7b2283dc7d165916d1) | `d6c4d4821885ed656b9fbc7b2283dc7d165916d1` | `ner.json`, `5796effb85c473aec5b85784545349ed42e408bb0fecdb78f82bc107d53edbe3` | CC BY-SA 3.0 (dataset card and README) | Wikipediaを用いた日本語の固有表現抽出データセット, Stockmark Inc. |

- The same values, with sizes and git blob SHA-1, are pinned in `contract.json`.
- The current Hugging Face revisions of HumanEval, MBPP and HumanEvalPack hold only Parquet files, which the standard library cannot read. The JSON or JSONL originals above are used instead.
- The tokenizer and template are those of `google/gemma-4-31B-it` at revision `842da3794eaa0b77d5f08bae87a17459d91ff475`: `tokenizer.json` with SHA-256 `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f` and `chat_template.jinja` with SHA-256 `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4`, both read only.
- The synthetic vocabulary, the synthetic templates and the `free-ja` instructions are self-authored and live in the repository.

## 11. Changes and versions

Any change to a rule, a threshold, a set, a prompt or a generator makes a new version (`G31-QC-v2`, …). The new version:

- updates the table at the top and adds its own section.
- keeps the sections of the old version.
- keeps the old results with their version.

If a procedure, a set or a threshold is changed after results were seen, §8 applies. Unexpected results, slow candidates and failed checks are kept and labeled with the version they ran under; criteria are not relaxed after results are seen.
