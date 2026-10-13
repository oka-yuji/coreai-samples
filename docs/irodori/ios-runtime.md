# Irodori runtime on iOS 27

This is the native runtime part of issue #94, based on Mac Speech commit `043cf1f`. The iPhone app and actual model/device acceptance are tracked separately by #95 and parent #96. No Mac GPU inference or changes to another worktree are part of this unit.

## Build targets

The pinned Rust input adapter has separate archives for the host Mac and the physical arm64 iPhone. Install the official iOS Rust standard library once, then build the required slice:

```sh
rustup target add aarch64-apple-ios
./Tools/Irodori/build-inputs.sh macosx
./Tools/Irodori/build-inputs.sh iphoneos
```

The iOS build uses the iPhoneOS SDK and a minimum OS of 27.0. SwiftPM selects the matching archive using platform conditions. `iphoneos` builds do not overwrite the host archive. The simulator is unsupported: the script refuses that platform, and the iOS Simulator SDK lacks Core AI. Do not treat the `.iOS` package condition as simulator support.

`SpeechRuntime` is available on macOS and iOS. On iOS, the fixed-shape text/duration stages disable frequent-reshape specialization; the dynamic-frame DiT/codec stages enable it. The Mac keeps its previously verified GPU-preferred/EFR settings. Normalization, tokenizer SHA, seed/noise, four-step integration, frame bounds and WAV encoding are unchanged.

## Audio lifetime

`AudioPlayback` owns one player, its continuation and its generation token. The platform session is passive on Mac. On iOS, `ManagedPlaybackSession` owns asynchronous activation/deactivation through `PhonePlaybackSessionDevice`:

- Register inactive/route/reset notifications before activation, and play only after successful activation.
- A stop during activation waits for its callback and, if activation succeeded, for asynchronous deactivation.
- Player stop precedes session deactivation. The play continuation and stop waiters resume only after cleanup completes. A newer play is refused while that cleanup is pending.
- A cancellation flag prevents an activation callback from starting audio after cancellation was requested. Generation tokens discard old callbacks.
- The Runner discards results from cancelled Tasks. An external playback interruption in a Task that is still active sends a session-tagged `playbackInterrupted` action so the screen returns to idle while preserving its WAV.

SDK evidence: iPhoneOS27.0 AVAudioSession.h lines 247–290 describes blocking `setActive` and the asynchronous activate/deactivate APIs. AVAudioSessionTypes.h lines 307–321 declares the iOS27 inactive notification that replaces the old interruption notification. MainActor isolation is retained for all player/session state.

## Validation boundary

CPU tests cover input compatibility, model lifetime, playback/session errors, immediate and late interruptions, cancellation during activation and waiting for deactivation. The physical-device SDK link probe checks the real Swift API and arm64 Rust archive without executing a model. Native archive inspection distinguishes Mach-O platform 1 (Mac) from platform 2 (iOS); the resulting framework requires iOS 27.0.

The iPhone model's specialization, output fidelity, latency, footprint and audible playback require the subsequent device run. A successful cross-build does not establish those results. The actual connected device's OS must be checked before installation; the development tools initially reported 26.6.1 while the user reported updating to 27.


## Acceptance evidence

P1–P4 of #94 are covered by the [validation artifacts](ios-runtime-validation/manifest.json): 34 CPU tests, 16 Thread Sanitizer tests, iPhoneOS arm64 framework link, Mac Release app build, strict lint and format. The native-build review found no unresolved issues. The independent audio review found R94-1 (synchronous MainActor activation), which was corrected using the SDK27 asynchronous APIs; its final independent CPU harness passed 41 tests. There are no unresolved confirmed or plausible findings. Real audio-session notifications, playback and Irodori inference remain #95 device checks, not results established by these fake-based tests.

The retained original estimate was 240–410 source/configuration lines and 5–8 provisional SP. The implementation reuses the existing player and runtime while adding an explicit asynchronous session owner. Review caused the synchronous activation draft to be replaced; the final start/cancel/cleanup contract is reflected above. The next integration step is #95, with the real-device OS/connectivity checked separately from successful cross-compilation.
