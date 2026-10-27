# Managed Irodori bundles

The managed store accepts the existing prepared Irodori FP32 folder. It retains `irodori.json`, `tokenizer.json`, and the four `artifacts/*.aimodel` directories. Use [the preparation script](runtime.md#prepare-a-local-model-folder); do not edit the manifest or add files to the prepared folder.

The local source identity uses the SHA-256 of `irodori.json` (`ce41c61dd2249cab61aa69017316627c54329177fbfb19a1d1df4d775d65dbc7`), together with the artifact ID, contract and source kind. That manifest describes the speech format rather than the asset checksums. The application separately pins all 14 paths, byte sizes and SHA-256 digests, including the tokenizer. The complete folder is 3,375,061,725 bytes. A changed inventory cannot redefine the trusted hashes.

Import copies into a new staging directory while retaining access to the selected folder. Size, checksum, manifest, exact inventory and symbolic-link checks precede promotion into the managed store. Failure or cancellation leaves the source untouched. An active lease prevents replacement or deletion. Inspection repeats verification before returning a ready resource. Existing image bundles continue to use `bundle.json` by default, and previous local-source records retain their identity.

The storage contract alone does not expose a runnable demo. The Models speech adapter is tracked by [#336](https://github.com/oka-yuji/coreai-samples/issues/336). Physical-device generation, memory, playback and listening remain pending under [#329](https://github.com/oka-yuji/coreai-samples/issues/329) and [#95](https://github.com/oka-yuji/coreai-samples/issues/95).

## Verification

`swift test --package-path CoreAISamplesKit` covers pinned definitions, missing files, manifest settings and legacy local-source decoding, alongside existing local import and Hugging Face regressions. The real-bundle test is disabled by default. Under the shared heavy-job lock:

```sh
COREAI_IRODORI_IMPORT_BUNDLE=/path/to/sampler-irodori-fp32 \
  swift test --package-path CoreAISamplesKit --filter IrodoriImportTests.knownBundle
```

This copies the known bundle into temporary managed storage, verifies it, tests lease protection, and rejects modified, extra, missing and symbolic-link files in that disposable copy. It does not specialize or execute a model, or modify the original bundle. A host import pass is not an iPhone inference result.
