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
speech_platform="${1:-${PLATFORM_NAME:-macosx}}"
case "$speech_platform" in
  macosx)
    "$speech_cargo" build --manifest-path "$speech_root/Vendor/IrodoriInputKit/rust/Cargo.toml" --release --locked
    ;;
  iphoneos)
    speech_sdk="$(xcrun --sdk iphoneos --show-sdk-path)"
    SDKROOT="$speech_sdk" IPHONEOS_DEPLOYMENT_TARGET=27.0 "$speech_cargo" build --manifest-path "$speech_root/Vendor/IrodoriInputKit/rust/Cargo.toml" --target aarch64-apple-ios --release --locked
    ;;
  *)
    echo "Unsupported Irodori platform: $speech_platform. Use macosx or iphoneos." >&2
    exit 1
    ;;
esac
