# Terminal budget refusal

Issue #522, following #425. Swift 6.4, Xcode 27.0 (27A266a), macOS 27.2 (26B123). Native Foundation Models uses a synthetic executor and fake tools; no trained model or GPU inference is involved.

## Behavior and ownership

Feature selects terminateTool only at the active turn's budget boundary. Runner delivers the call-specific terminal decision. ToolRunner records the refusal, then the native AgentTool callback throws CancellationError instead of returning a result that can start another model request. Ordinary permission and operational errors remain strings. Store remains the task owner.

Session's existing limitReached flag is set before the tool-request action. Thus an early native loop exit still reaches budgetStopped and the bounded final-answer transition. After stream join and while the native session is no longer responding, incomplete calls are closed before the final transcript update and persistence. The final marker is retained and no tool is replayed.

## Evidence

| Candidate and check | Result |
|---|---|
| Original runtime with final stop delivery held | 3/3 fail: final requests 2; order final-1, stop-wait, final-2, stop-deliver, final-return |
| Initial terminal-refusal candidate, ordinary/delayed cases | All 23 checks pass across three solo runs and ten concurrent pairs |
| Initial candidate, same-session next-turn check | Fails: final requests 1 and next answer eos, but one pending call remains |
| Final candidate with join-boundary closure and next-turn check | All 23 complete offline suites pass across three solo runs and ten concurrent pairs |

BTF-01: ConversationFeatureTests and ConversationGuardTests distinguish terminal cutoff from normal refusals; the 76-test Feature run has no failures. `tools/failuresAreText` also passes.

BTF-02: The final delayed-stop case retains an expected final-request count of one and the original failure/no-answer requirement. Its AsyncStream gate releases on a second final request or generation return, without a sleep controlling order.

BTF-03: The final case preserves the first saved failure/no-answer result, then sends an ordinary follow-up through the same live session. It checks two saved turns, an eos answer, zero pending calls in the actual next model request, and no additional tool execution. The full offline suites also retain normal answers, empty/error finals, Stop, drafts and restore checks.

The initial candidate's concurrent run 17 separately failed `fetch/http/webView/redirect` with WKErrorDomain code 6 during content-rule compilation. Its budget checks passed. The unchanged candidate then passed one solo run and two pairs. The original failure remains recorded under #524; no WebKit policy or code was changed here. The final candidate's 23 runs have no failure lines.

BTF-04: After the last source change, the default package suite reports 1,280 tests across 43 runs with no failures; opt-in checks remain skipped. Rust input build, strict SwiftLint, changed-file swift-format, Mac Release with the original bundle settings, package pins and both working-tree/commit privacy checks pass. Independent review found and verified the next-turn pending-call issue; its correction and the preserved first-turn failure assertion were re-read. There are no unresolved confirmed findings.

The 3 SP estimate remains provisional. The Feature decision, Runner mapping, native exception and transcript closure form one termination contract; no partial version is enabled separately. Real-model output quality and live-model frequency of malformed final tool calls remain unmeasured.
