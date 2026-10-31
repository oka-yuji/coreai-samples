# Managed Irodori bundles

The normal iOS Models screen downloads the fixed reference v2 release from Hugging Face. Its 22-file definition includes the model inventory and LICENSE/NOTICE. The store accepts only the known repository, commit, root and file inventory, verifies every size/SHA, and validates the manifest before making the model ready. The [iPhone guide](iphone.md) describes this flow.

## Legacy text-only import

The managed store accepts the existing prepared Irodori FP32 folder. It retains `irodori.json`, `tokenizer.json`, and the four `artifacts/*.aimodel` directories. Use [the preparation script](runtime.md#prepare-a-local-model-folder); do not edit the manifest or add files to the prepared folder.

The local source identity uses the SHA-256 of `irodori.json` (`ce41c61dd2249cab61aa69017316627c54329177fbfb19a1d1df4d775d65dbc7`), together with the artifact ID, contract and source kind. That manifest describes the speech format rather than the asset checksums. The application separately pins all 14 paths, byte sizes and SHA-256 digests, including the tokenizer. The complete folder is 3,375,061,725 bytes. A changed inventory cannot redefine the trusted hashes.

Import copies into a new staging directory while retaining access to the selected folder. Size, checksum, manifest, exact inventory and symbolic-link checks precede promotion into the managed store. Failure or cancellation leaves the source untouched. An active lease prevents replacement or deletion. Inspection repeats verification before returning a ready resource. Existing image bundles continue to use `bundle.json` by default, and previous local-source records retain their identity.

The normal iOS Models screen registers the pinned artifact and connects it to the managed speech adapter. See [the iPhone guide](iphone.md#use-irodori-from-models) for acquisition, generation, playback and WAV export. Display and import completion do not start inference. Models retains the resource lease until speech shutdown joins generation, playback and audio-session deactivation. The [reference diagnostic](reference-input-validation.md) records later device generation. Microphone capture, audible playback and subjective quality were not evaluated by that diagnostic.

## Verification

`swift test --package-path CoreAISamplesKit` covers pinned definitions, missing files, manifest settings and legacy local-source decoding, alongside existing local import and Hugging Face regressions. The real-bundle test is disabled by default. Under the shared heavy-job lock:

```sh
COREAI_IRODORI_IMPORT_BUNDLE=/path/to/sampler-irodori-fp32 \
  swift test --package-path CoreAISamplesKit --filter IrodoriImportTests.knownBundle
```

This copies the known bundle into temporary managed storage, verifies it, tests lease protection, and rejects modified, extra, missing and symbolic-link files in that disposable copy. It does not specialize or execute a model, or modify the original bundle. A host import pass is not an iPhone inference result.
