# Runtime implementation review

Issue #91 is based on 12a5955. The fixed input implementation is adapted from 4b187d2e and the FP32 pipeline from the validated native CLI. UI integration is the dependent #92; the parent tracking issue is #93.

R1: Rust builds from Cargo.lock, Swift uses language mode 6 with the two selected concurrency features, and upstream licenses are retained. The ordinary check.sh entry point now prepares the Rust library before Swift build/test.

R2/R3: Debug and Release each passed 11 CPU tests, including the frozen 283 normalization cases, all 49 noise conditions, input layout/count bounds, bundle format and hash checks, frame policy, and Float32 WAV bytes. The app rejects more than 255 body tokens without starting a pipeline.

R4: Controlled fake pipelines test repeated calls, shape/periodic recycling, context-before-model release, overlap refusal, cancellation with an uncooperative operation, and recovery. An independent reviewer added three CPU scenarios for cancellation during pipeline creation, prepare failure, and nonfinite PCM; all 14 checks passed. GPU operations were never executed.

R5: Three small bundle preparation tests verify complete output, existing/racing output preservation, and symlink materialization. The directory is published using renamex_np with RENAME_EXCL. Build/lint/format and the manual preparation instructions are recorded separately from GPU acceptance.

Independent review found two P2 defects: the existing check.sh lacked Rust preparation, and directory copying retained source symlinks that runtime validation rejected. Both were reproduced, fixed, and independently rechecked. There are no unresolved confirmed or plausible findings. Vendor math, FFI, dependency lock and licenses match their source implementation, apart from documented representation/metadata changes.

This historical review did not measure actual model inference, memory behavior, app interaction or listening. Later native generation checks are in [release asset validation](release-asset-validation.json); those checks do not establish original-PyTorch waveform parity or subjective quality.

The upstream PYTORCH_LICENSE is retained byte for byte, including its original trailing whitespace. Whitespace validation excludes that license and raw logs only. The source/import count is 1,031 lines before the small build/manifest configuration files; this is within the original 850–1,200-line runtime estimate, with most input code reused.
