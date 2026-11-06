# Agent self-test note isolation

Issue #423, based on #421. Swift 6.4, Xcode 27.0, macOS 27.2 (26B123). The notes/write and notes/writeSuffixesOnCollision checks now use a fresh temporary directory for each execution. The default note location is checked without writing there. The note naming and production write behavior are unchanged.

## Results

- NSF-01: With a disposable validation bundle identifier, the old self-test changed its ordinary notes directory's modification time. The patched executable kept the ordinary notes directory absent across all runs. The original application's note directory, entries, file hashes and modification times remained unchanged.
- NSF-02: Ten pairs of simultaneous `--agent-selftest --offline` processes completed. All 20 `notes/write` checks and all 20 `notes/writeSuffixesOnCollision` checks passed.
- Nineteen complete suites passed. Run 12 failed only at `conversation/budget-toolInFinal`, reporting `finish=stopped final=2 tools=["web_search"] error=none requests=5`. This is the already tracked #425 scenario, not a note collision, and is retained as a failure for that investigation.
- NSF-03: Strict SwiftLint, changed-file swift-format, Rust input build, privacy and package-pin checks passed. The default package suite passed 1,277 tests across 43 runs. Release compilation with both the disposable and original bundle identifiers passed.

The change is a single small test-fixture isolation fix. The existing S classification and omitted SP estimate are retained. No other self-tests or production note behavior are changed. The new folder is owned by the test's do scope and removed on success or failure.
