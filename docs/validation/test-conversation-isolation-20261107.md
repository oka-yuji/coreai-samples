# Diagnostic conversation isolation

Issue #421. Base `a749575`, Swift 6.4, Xcode 27.0 (27A266a), macOS 27.2 (26B123).

The default archive is selected once per process. The three diagnostic entry points use a UUID-named temporary directory; ordinary launches retain the existing directory. GUI transcript exports use the same archive as the diagnostic host. Existing histories and preferences are not migrated or removed.

## Verification

Both builds use a separate validation bundle identifier. The original application's conversation directory is compared before and after each run. No existing user conversations are used as test fixtures.

| Check | Before | After |
|---|---|---|
| `--selftest`, Qwen3.5 0.8B, one fixed prompt | PASS; ordinary validation archive changed | PASS; archive unchanged |
| `--agent-e2e`, same model, fixed tool-free prompt | PASS; ordinary validation archive changed | PASS; archive unchanged |
| `--gui-e2e`, same model, default seven steps | 7/7; ordinary validation archive changed | 7/7; archive unchanged |
| Original application's conversations | Unchanged | Unchanged |
| GUI `--agent` scenario | External 420-second deadline at the tool-list step | Pending |

The GUI Agent scenario's transcript export and complete TCH-03/TCH-04 evidence remain pending. The baseline timeout is retained and is not counted as a pass. The ordinary GUI scenario does not produce the Agent transcript export.

- The offline Agent self-test passes, including four diagnostic argument forms, distinct launch directories, normal-directory compatibility, and JSON/selection/Markdown round trips.
- The default package suite reports 1,277 tests across 43 runs, with no failures. Model-dependent and other opt-in checks remain skipped.
- Rust input build, strict SwiftLint, changed-file swift-format, Mac Release build with the original bundle settings, package pins and local-path checks pass.
- Independent review covers all changed files and their save/restore/reset/export callers. A test-cleanup candidate was corrected: deletion is registered only after rejecting the ordinary archive. No confirmed findings remain.

The change remains one implementation unit: selecting a lifetime-stable archive and exporting from it are the same persistence contract. Production delta is 17 added/deleted lines; the regression checks are separate. The existing 2 SP estimate is retained. No unrelated history cleanup or preference-policy change is included.
