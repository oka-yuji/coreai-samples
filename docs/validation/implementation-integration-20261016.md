# Existing implementation integration — 2026-10-16

This record covers the integration of PRs #133–#138 into the current sample app.
The combined code candidate is `ec75ddf503595ce1f6bf0c52ec7d760ee877f6f0`;
the subsequent record-only commit adds this document and the probe results.

| PR | Unit | Integration checks |
| --- | --- | --- |
| #133 | Conversation logic | Full CPU suite, strict lint, format, Mac Release; export regression |
| #134 | Chat and Work screen | Full CPU suite, strict lint, format, Mac Release, online/offline app self-tests, localization |
| #135 | Mac speech runtime | Rust Release, full CPU suite, lint, format, Mac Release, three bundle-preparation tests |
| #136 | Mac Speech screen | Full CPU suite, lint, format, Mac Release, navigation localization |
| #137 | iOS speech runtime | Full CPU suite, lint, format, Mac Release, iPhoneOS Rust build and final dynamic-framework link |
| #138 | iPhone Speech app | Full CPU suite, lint, format, Mac Release, signed SpeechPhone and existing iPhone Release builds |

The final combined tree passed the full default SwiftPM test command, strict
SwiftLint with zero violations, the format checker, Mac Release build,
`--agent-selftest` both with and without `--offline`, and `--check-localization` with 72 runtime keys and
zero missing Japanese translations. The catalog has 624 keys; its existing 622
entries retain their meanings and the Speech navigation adds two.

The temporary checkout exposed a SwiftLint path mismatch between `/private/tmp`
and `/tmp`. The temporary baseline normalized only file paths. Rule identifiers,
source texts and permitted occurrence counts were preserved. Source lint rules
were not relaxed. Third-party license text and retained raw validation logs keep
their original whitespace.

## Review fixes

Three defects were reproduced and corrected:

- Markdown export used a fixed four-backtick delimiter. An embedded delimiter
  could end the block early. The exporter now uses a delimiter longer than every
  run in the text. The new reasoning/tool-output regression failed before the
  fix and passed afterward.
- A tool request could set the step limit before transcript observation
  recorded earlier completed steps. The request now records progress first.
  Both observation-first and request-first probes record steps 1 and 2.
- Two conversations created within one second could share their Markdown path.
  Conversation IDs now include a UUID suffix. Same-time export checks preserve
  both files.

The [probe results](integration-probes-20261016.json) retain before/after evidence.
The step-order probe uses the real session implementation and Foundation Models
transcript types with stubbed model/log boundaries. It does not measure inference
frequency or token-accounting accuracy under a real model.

Independent branch reviews and a separate integration audit found no remaining
confirmed issues. The integration retained the current model-switch guards,
citation-guard default off, existing demos, Speech navigation, and both iPhone
targets. Conversation code and tests match the reviewed `ad05f36` snapshot;
Speech, iPhone and native-input code match the reviewed `a5313c3` main snapshot.

## Device and model boundary

The signed SpeechPhone product passed signature verification. Its actual
Info.plist contains both file-sharing keys and its signature contains the
increased-memory-limit entitlement. A fresh device-info read reported the
requested iPhone 17 Pro on iOS 27.0.1, with a connected tunnel and Developer Mode
enabled. No app installation, model transfer, device generation, playback or
device WAV export was performed during this integration.

A SwiftPM automatic-library scheme built an intermediate iPhoneOS object. An
additional dynamic framework consuming SpeechRuntime was linked to verify the
native archive: all seven `iro_*` functions were defined, none remained
unresolved, and the product targeted iOS 27.0. The signed SpeechPhone app also
completed its final link.

Real-model tests were not rerun. Earlier 0.8B evidence remains historical
evidence for the original conversation implementation, including the known
Agent GUI result of 7/8. Physical interaction and listening remain unchecked.
Issues #78, #92 and #95 retain those checks.

Conversation JSON persistence/restoration (#79), the saved-conversation sidebar
(#80), continuation/split-note work (#81), and RAG conversation continuity (#82)
are separate unimplemented work. This integration does not complete them.
