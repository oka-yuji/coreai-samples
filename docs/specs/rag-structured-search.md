# Structured BM25 retrieval

Issue #375, parent #378 plan v1.1. The primary comparison is Qwen3.8 27B 4bit in Work. Model conversion, weights, tokenizer, normal UI and conversation archive schema are unchanged.

`--rag-strategy structure-bm25` selects B for pre-answer retrieval in both Chat and Work. Unspecified strategy remains A (`flat`). Additional Work tools retain their existing behavior.

## Candidates and evidence

Candidate text for ranking combines document title, ancestor headings, section heading and original body. Long bodies use the existing 500-character chunks with 100-character overlap, contained within the section. Heading text affects ranking but is supplied as a separate original passage, never prepended to the evidence body. All passages retain their original document/version and Character offsets. The same original body cannot consume more than one rank.

Preamble text is eligible. Empty parent sections only supply ancestor headings. A heading-only leaf triggers a recorded document-level `headingOnlyLeaf` flat fallback so facts contained only in that heading remain available. Other unsupported/malformed structures fall back per document. Entirely unstructured collections and collections with no matching candidates use A with explicit reasons.

The existing BM25 and fallback ranking/tokenizer remain unchanged. As in A, unranked candidates may fill unused space. A matching candidate rejected by the page budget is distinguished from no lexical match (`matchingCandidatesExceedPageBudget`); it does not cause an A fallback. If all candidates exceed the budget, B returns zero sources.

## Budget and ownership

Web page budgets use characters, and the final source budget uses the active model's token counter. A candidate's ancestor headings and body are one indivisible group. Shared original ranges count once. Final conversation prompt packing also removes complete groups, including the question and prompt overhead in its token check. It cannot leave a heading after dropping its body.

Retrieval remains in the Runner's current run. Existing permission checks, cancellation, stale-run rejection and the conversation source ledger remain authoritative. Selected groups are converted into existing `ConversationSource`/`SourcePassage` values. The ledger assigns document-version `[S#]` IDs and retains original passages across continuation and archive restoration.

## Diagnostics and interpretation

The native evaluation result stores requested/effective strategy, candidate/body/supplied passages, section IDs, fallback reasons and retrieval/answer timing. Hit@1 and Hit@3 count ranked body passages only; final recall includes all supplied original ranges, including headings. These measures answer different questions. A heading-only gold span can therefore have zero body Hit@k but full supplied recall.

Without model weights, inspect the fixed corpus using:

```sh
CoreAISamples --rag-eval Tools/RAGEvaluation/corpus.json --validate-only \
  --rag-strategy structure-bm25 --retrieval-output /new/b.json
```

Repeat with `flat` for A. This path explicitly records `budgetUnit: characters`; it checks extraction/selection and does not represent native tokenizer budgets, generated answer quality, or Qwen runtime performance. Real-model evaluation uses the pinned manifest/bundle through the separate serial evaluation driver.
