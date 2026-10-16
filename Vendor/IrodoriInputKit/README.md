# Irodori input dependency

Local Swift/Rust input adapter for Irodori-TTS-v4.1-Small-MF. It pins tokenizers 0.22.2 and Unicode 15.0.0 normalization, preserves embedded NUL bytes, and reproduces the pinned PyTorch CPU noise algorithm on Apple silicon.

Build the static library with `Tools/Irodori/build-inputs.sh` from the repository root before running package tests. The Mac application build also runs that script. Rust/Cargo and the Xcode command line tools are required.

Release build dependencies keep their symbols intact with
`profile.release.build-override.strip = "none"`. On macOS 27, Rust 1.96 can
produce a misaligned Mach-O string table when stripping a procedural macro.
This appears as `castaway` failing to find `rustversion`, even though the
dependency exists. See [Rust issue #157750](https://github.com/rust-lang/rust/issues/157750).
The override applies to build scripts, procedural macros and their dependencies;
the application's deployment target and runtime optimization settings are retained.

For a fresh-cache check with the deployment target exported by Xcode, run
`MACOSX_DEPLOYMENT_TARGET=27.0 cargo build --manifest-path Vendor/IrodoriInputKit/rust/Cargo.toml --release --locked -p castaway --target-dir /tmp/irodori-macro-check`
from the repository root, using an empty output directory.

This code is adapted from the validated local input implementation at revision 4b187d2e. The normalization order, token IDs, mask, and noise arithmetic are retained. Package-only test/CLI targets are omitted here; public source comments are removed to match the repository convention, and fixed Unicode mapping symbols use escaped scalar values. Additional token-count information is used by the app to reject truncated input.

The full upstream notices are in THIRD_PARTY_NOTICES.md and the accompanying license files. The tokenizer/model files are not bundled with the source.
