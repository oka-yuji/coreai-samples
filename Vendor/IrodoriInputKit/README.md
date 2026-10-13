# Irodori input dependency

Local Swift/Rust input adapter for Irodori-TTS-v4.1-Small-MF. It pins tokenizers 0.22.2 and Unicode 15.0.0 normalization, preserves embedded NUL bytes, and reproduces the pinned PyTorch CPU noise algorithm on Apple silicon.

Build the static library with `Tools/Irodori/build-inputs.sh` from the repository root before running package tests. The Mac application build also runs that script. Rust/Cargo and the Xcode command line tools are required.

This code is adapted from the validated local input implementation at revision 4b187d2e. The normalization order, token IDs, mask, and noise arithmetic are retained. Package-only test/CLI targets are omitted here; public source comments are removed to match the repository convention, and fixed Unicode mapping symbols use escaped scalar values. Additional token-count information is used by the app to reject truncated input.

The full upstream notices are in THIRD_PARTY_NOTICES.md and the accompanying license files. The tokenizer/model files are not bundled with the source.
