# EmbeddingGemma 2 native text and image runtime

The runtime accepts a local text/vision bundle, raw text, and opaque images decoded by ImageIO. It uses the same S128 / vision70 contract as the CoreLLM #27 conversion. Audio is not included. macOS 27 and iOS 27 are required; Core AI is unavailable in the iOS simulator.

## Prepare the local bundle

Use the Python environment from the conversion experiment, with NumPy installed:

```sh
python Tools/Embedding/package_bundle.py /path/to/embeddinggemma2-20261014 /path/to/new-bundle
```

The output contains the two safe FP16 assets, the original tokenizer, an exact BF16 representation of the unscaled vocabulary embeddings, the original model card, license references, and SHA256 checksums. The original assets are unchanged. Accept the source model's Gemma terms before acquiring or distributing weights.

## Runtime contract

- Queries use `task: search result | query: `; documents use `title: none | text: `.
- BOS/EOS and the prefix count toward 128 tokens. Longer text is rejected.
- Images respect EXIF orientation, use RGB and antialiased separable bicubic interpolation, then form 16×16 patches and spatial 3×3 pooling groups. Transparent and animated images are rejected. Decoded images are limited to 40 million pixels.
- Only the converted `fp16-safe-s128-v70` signature is accepted. File hashes, paths, tensor shapes, and scalar types are checked before inference.
- `EmbeddingModel` owns its models and vocabulary table in one actor. It rejects overlapping load/encode/unload operations. Cancellation does not allow replacement while an awaited runtime call is still returning.
- Embeddings have 768 finite components and must be within 0.001 of unit norm. Search uses cosine similarity, not probability.

## Validation

```sh
swift test --package-path CoreAISamplesKit
EG2_TOKENIZER=/path/to/bundle swift test --package-path CoreAISamplesKit -c release --filter Embedding
```

CPU tests do not load models. `EG2_TOKENIZER` enables the pinned twelve-text token fixture and length-limit check.

`embedding-verify BUNDLE PLAN OUTPUT --preprocess` checks token IDs and records pixel differences without loading models. Remove `--preprocess` for native text/image inference. The plan contains raw text/image paths and the original FP32 reference vectors. The low-precision gate is cosine ≥ 0.999, maximum absolute difference ≤ 0.01, finite values, and unit norm within 0.001. This fixed set is not a general retrieval-quality benchmark.

Model validation must run under the repository's shared heavy-job lock. `Tools/Embedding/run_validation.py` waits for other model processes, acquires the lock, runs the supplied validation command, and releases the lock when it exits.

At the runtime implementation checkpoint, all twelve native text token sequences match and all eight image token counts match. Pixel differences are at most 1/255 for the five recipe images and zero for the three synthetic gradients. Full native vector and application checks are recorded in the subsequent demo verification report.
