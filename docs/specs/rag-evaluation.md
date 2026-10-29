# Reproducible RAG evaluation

Issue #373, parent #378, plan v1.1, October 29, 2026.

The target is Qwen3.8 27B 4bit in Mac Work. The existing converted bundle is reused. This change supplies the fixed-input evaluation and baseline A; structure extraction, B/C selection, and measured adoption belong to #374–377.

The acceptance conditions RG-E01–07 remain canonical in issue #373. The executable/data contract and record definitions are in [Tools/RAGEvaluation](../../Tools/RAGEvaluation/README.md).

## Ownership and compatibility

`RAGInputs` injects local documents, search results and page text at the existing retrieval boundary. The live implementation preserves the previous acquisition, per-page selection, global ranking and packing. Snapshot replay enters those same downstream functions. `ConversationHost` carries the dependency across resets; `ConversationRunner` owns its asynchronous use. The Host has no added asynchronous implementation.

The Feature/Store/Runner retain their existing request, stop, source permission and archive ownership. The diagnostic enables source precollection and sends through the existing Host in Work mode. It has an isolated archive and notes directory. It does not change the normal UI, saved conversation format, default model, or existing diagnostic meanings.

The source ledger determines whether a source was actually supplied. The diagnostic records those passages in retrieval order rather than the ledger's document-grouped order. Source identity, original text and semantic support remain separate checks.

## Failure and comparison boundaries

- Invalid datasets, unsupported strategies and unpinned models are rejected before model loading.
- Snapshot misses are errors with no network fallback, including later Work web tools.
- Deadlines request the existing stop path and wait for generation and archive work to settle. An outer process timeout is a separate driver failure, with confirmed cleanup required before the lock is released.
- Native cancellation, archive failure, missing outputs and unconfirmed child termination remain distinct states. They do not disappear from the requested denominator.
- Every completed result contains the execution outcome; quality grades remain unset until a reviewer checks the actual evidence.

The implementation does not claim any improvement over A, reproduce STAIR's published numbers, or establish iPhone support. Real-model and manual acceptance remain separate from CPU and fake-model tests.
