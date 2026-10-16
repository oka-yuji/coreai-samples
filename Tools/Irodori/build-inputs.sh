#!/bin/bash
set -euo pipefail
speech_root="$(cd "$(dirname "$0")/../.." && pwd)"
speech_cargo="$(command -v cargo || true)"
if [[ -z "$speech_cargo" && -x "$HOME/.cargo/bin/cargo" ]]; then
  speech_cargo="$HOME/.cargo/bin/cargo"
fi
if [[ -z "$speech_cargo" ]]; then
  echo "Rust/Cargo is required to build the Irodori tokenizer." >&2
  exit 1
fi
"$speech_cargo" build --manifest-path "$speech_root/Vendor/IrodoriInputKit/rust/Cargo.toml" --release --locked
