# Conversation step budgets

Issue #145, specification v1, October 16, 2026. This implements the user's request
to fix early step-limit termination and merge the current app changes. The
[original diagnosis](../research/chat-step-limit-20261016.md) concerns the old
fixed two-step Chat policy; the [validation record](../validation/conversation-step-budget-20261016.md)
distinguishes automated coverage from live-model acceptance.

## Behavior

- A step is one model generation. Chat defaults to 16 steps per reply and Work
  defaults to eight. Each mode has its own persisted setting, adjustable from
  1 to 64 in the conversation settings. New ordinary messages reset the budget.
- The last research step asks for an answer. A further tool call is refused
  before execution. Once the native stream stops, the app gives the same
  logical turn one extra answer generation using the existing session and
  collected information. Tool calling is disallowed and reasoning is off.
  The configured output-token limit still applies.
- A budget stop is distinct from a user stop. Explicit Stop, termination, or
  a stale run notification cannot start a final answer. Stop during the final
  answer ends it normally. There is at most one automatic final attempt.
- Failure, empty output, context exhaustion and token exhaustion retain the
  history and report the actual outcome. There is no automatic retry loop.
  An explicit continuation of a truncated final answer keeps tools disabled
  and reasoning off; a subsequent ordinary question gets a fresh research budget.
- The transcript preserves completed tool results. Pending calls receive
  interrupted outputs before the final prompt, without replaying any tool.
  Work keeps its fetched-page ledger and draft notes across the handoff.
  Cancellation settles citation repair and releases the busy state before
  acknowledging completion; incomplete drafts remain held.
- The internal finalization prompt is absent from the visible conversation
  and Markdown export. A research-limit note explains the bounded answer.
  Citation checks inspect the same concatenated answer shown after a continuation.

## Ownership and ordering

`ConversationFeature` owns the budget, run identity, cancellation and handoff.
The Store owns effect tasks. `ConversationRunner` waits for the previous
generation to finish before requesting the answer. `ConversationSession`
preserves the native Foundation Models session and changes generation options,
not the registered tool set. The runtime rejects tool execution while stopping;
the final generation also uses `toolCallingMode: .disallowed`.

`settleBudgetStop` is a separate effect for a handoff that is no longer eligible.
It settles the citation guard, retains or reports drafts, releases the app's
busy state and then sends the stop acknowledgement. Run identity prevents an
older stream from overwriting a newer result.

The native tool-mode declaration and mutable transcript are present in the
installed macOS 27 SDK. Apple's [tool-calling documentation](https://developer.apple.com/documentation/foundationmodels/expanding-generation-with-tool-calling)
describes using a session's accumulated context with tool calling disabled.
CPU integration checks exercise the native API; they do not establish model
answer quality or latency.

## Storage compatibility

Archive format 4 adds optional `chatMaxSteps`, per-turn `stepLimit`,
`finalAnswerStart`, and `toolsDisabled`. Formats 1–4 are accepted; unknown future
formats retain the existing read-only behavior. Old turns lacking `stepLimit`
keep Chat's actual historical limit of two or the archived Work limit. New
turns freeze the normalized setting, so later preference changes cannot rewrite
old status. Missing legacy Chat preferences preserve the current default or
chosen value.

The final marker must lie inside its owning turn at a prompt or its current end.
Restore remaps it when native transcript repair inserts missing tool outputs.
The private final prompt remains in canonical JSON for session reconstruction,
but rendering and Markdown skip it. No private conversation text is included
in the fixtures or this documentation.

## Acceptance and scope

STEP-01 through STEP-08 retain the original wording in Issue #145. Their
tests and command evidence are mapped in the validation record. The dependent
merge order is #132 (Rust build), #144 (approved conversation UI), then #146
(this budget fix). Integration is checked by this change's owner. The user
explicitly requested merging this stack after the fix; live-model acceptance
remains a separate, unchecked result.

Unlimited execution, model conversion and automatic conversation summarization
are outside this change. The provisional estimate remains 5 SP. Keeping the
budget, cancellation and final-answer transition together avoids an intermediate
implementation that still stops without an answer.
