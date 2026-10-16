# Decide: a document checklist with Clef-Flash 9B

Decide runs a converted Clef-Flash 9B model locally through Core AI. Supply a document and named questions with choices; it returns a probability for every option. The app keeps the document context while you edit the questions.

## Open the demo

Download the [Clef-Flash 9B Core AI bundle](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide) with the Hugging Face CLI:

```sh
hf download okayuji/clef-flash-9b-coreai-decide --local-dir ./models/clef-flash-9b-decide-v1
```

The model guide is available in [English](clef-flash.md) and [日本語](clef-flash.ja.md). The download needs internet access; document decisions run locally after the bundle is available.

Use macOS 27 and Xcode 27 on an Apple silicon Mac. Open `CoreAISamples.xcodeproj`, select the `CoreAISamples` scheme and **My Mac**, and run. The scheme uses Release. Select **Decide**, then **Open decision bundle…** and choose the directory containing `decision.json`. The same button is available in **Models**. Model weights are separate from this repository.

The supported bundle is `clef-flash-9b-decide-v1`: a uint4 backbone with fp32 gated-delta updates, a separate fp16 decision head, BF16 lexical embeddings and a tokenizer. Loading checks the manifest, the required files and all entries in `SHA256SUMS` before specializing the models.

## Try the checklist

1. Choose **Maintenance agreement** from **Examples**, then **Run checklist**. The bundled Japanese document and its four questions are fictional. A second example contains inspection records for the same equipment.
2. Add a choice question with the **+** button, or edit, move or remove an item. Run again to reuse the document. All questions are recomputed together; **Changed** marks an edited item or a changed candidate/review status. Expand **Compare previous result** to see the previous distribution.
3. Sort by **Lowest probability first**. **Review first** marks an unknown/not-stated candidate or a maximum probability below 0.6; for yes/no, it uses the larger of the two probabilities.
4. Search the original document to highlight matching text, then choose a value under **Confirmed by reviewer**. This choice is stored separately from **Model candidate**.
5. Press **Read again** to compare a complete read with reuse. A successful replacement result clears prior confirmations. Stop or Escape cancels the current operation; a failed or cancelled rerun preserves the last result and its confirmations.
6. Use **Export JSON** to save the request, model response, model version, timing and reviewer confirmations. Saving and cancelling use the native file dialog.

Probabilities are labeled **Model probability (uncalibrated)**. They describe this model's distribution over the supplied options. They are not a measured chance that the answer is correct. An option such as “not stated” is useful when the document may not answer a question.

Import accepts UTF-8 `.txt` and `.md` regular files up to 4 MiB. You can also paste text. The counter includes the rendered questions and options. This bundle accepts 1–16 questions, 1–8 options per question and at most 9,216 input tokens. The reusable document context reserves 1,024 tokens for questions; exceeding either budget produces an error without truncation. **Read again** uses the complete-input limit instead. PDF, batch processing and iOS are outside this demo.

## Swift API

```swift
import CoreAIBackend
import Foundation

let model = try await DecisionModel.load(bundle: URL(filePath: bundlePath))
let context = try await model.context(state: document, questionBudget: 1_024)
let questions = [
    DecisionQuestion(
        id: "renewal",
        instructions: "Does the agreement renew automatically?",
        criteria: .noul()
    )
]
let reused = try await context.decide(questions)
let fresh = try await model.decide(DecisionRequest(state: document, questions: questions))
```

A model owns one current context. Creating another context or requesting a complete read invalidates the old handle. Calls are serialized, and cancellation is checked before modifying shared state. A context can be used again after a cancelled query. Questions retain input order; option IDs retain their UTF-8 spelling. Use the array form of choice options when distinct Unicode scalar sequences must stay distinct.

## Headless checks

Set `DECIDE_APP` to the built app executable and `COREAI_DECISION_BUNDLE` to the bundle directory. On the shared development Mac, acquire the GPU lock described in `CLAUDE.md` before loading any model, including the GUI.

```sh
export COREAI_DECISION_BUNDLE=/path/to/clef-flash-9b-decide-v1
DECIDE_APP=/path/to/CoreAISamples.app/Contents/MacOS/CoreAISamples
"$DECIDE_APP" --decide-json request.json response.json
"$DECIDE_APP" --decide-json request.json reused-response.json --reuse-document
"$DECIDE_APP" --decide-e2e /tmp/decide-check
"$DECIDE_APP" --decide-json /tmp/decide-check/request.json /tmp/decide-check/cli-response.json
```

`--decision-bundle /path/to/bundle` can replace the environment variable. The JSON request uses `state` and an ordered `questions` object. Each question has a `type` of `choice`, `noul` or `score`, optional `instructions`, and `criteria` (an object for choice/yes-no, an array for score).

`--decide-e2e` uses the same Store and service as the screen: four contract questions, one added question, reuse, review ordering, human confirmation, a complete read and export. It writes `report.json`, `request.json`, `response.json`, `review.json` and `review-reused.json`. The review format is `decision-review/1`, with separate `request`, `response` and `reviewer_confirmations` fields. `response.metadata.diagnostics` holds the unrounded per-option probabilities; `calibrated` is false.

Ordinary `swift test --package-path CoreAISamplesKit` does not load a model. Model tests use the dedicated decision-bundle variable and the `DecisionInferenceTests` filter; do not set the language-model variable `COREAI_TEST_BUNDLES` at the same time.

## Verified scope

On M4 Max 128 GB, macOS 27.0 (26A428), Xcode 27 and a Release build (2026-10-07–08):

- Ten fixed documents across the 1K/2K/4K/8K document sizes matched the lab Core AI probabilities exactly. Against the same uint4 weights evaluated by PyTorch in fp32, the maximum difference was 0.016694 and the largest band mean was 0.002940. The input band at or below 512 tokens was not evaluated here.
- Reuse versus a complete read differed by at most 0.010339 on the fixed 2K/8K question sets (limit 0.011). For the fixed four-question 8K set, 20 reuse calls had a p95 of 0.847 seconds.
- Alternating one and twelve questions requires model/function reloads while retaining the document state. This kept measured footprint growth to 0.000790 GB per call over 20 calls; those calls took about 1.5–2.7 seconds. The sub-second result applies to the fixed four-question case.
- The five-question app scenario and an independent JSON command matched all 17 probabilities exactly. Native GUI checks covered question editing, reuse, source highlighting, review confirmation, a complete read, Stop, and JSON Save/Cancel.

These measurements establish integration and numerical agreement for the tested bundle and fixtures. They do not calibrate the model or establish accuracy on other documents.
