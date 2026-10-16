# Chat step limit investigation

The user reported a maximum-step stop after two exchanges on 2026-10-16.
This investigation reads existing device logs and the saved conversation; it
does not rerun a model or change the execution policy. Private prompts, answers,
URLs, paths and transcript contents are not copied into this report.

## Observed sequence

The metrics record at `2026-10-16T13:11:54Z` (22:11:54 JST) has
`stopReason=stepLimit`, `outputTokens=165` and model
`qwen3_8_27b_qwen3_5_cfg_gdnfp32_dynamic`. The saved transcript corroborates the
following sequence:

| Turn | Observed behavior |
| --- | --- |
| Initial cancelled input | No completed generation step; not one of the two substantive user requests. |
| First substantive request | Step 1 calls `web_search`; step 2 produces an answer. |
| Follow-up request | Step 1 calls `web_search`; step 2 requests `fetch_page`, which is rejected before execution. No final answer is produced. |

The refused tool output is `error: the turn ended before this tool ran`.
The final generation records 13,515 reserved context tokens, and only 165 output
tokens were generated across its two steps. The stored settings have
`maxSteps=8`, but the turn mode is Chat. The saved instructions version is
`cite-fetched-v3`. Each normal turn has `offset=0` and its
own transcript start index. This is an internal per-answer step cap, not a
two-exchange conversation limit or a context/token exhaustion.

## Cause in the current implementation

At `d153eef`, `TurnPolicy.init` in `ConversationState.swift` fixes Chat to
`maxSteps=2`. Only Work uses `settings.maxSteps`; the settings control accurately
labels that value as Work-only. `ConversationFeature.decide` rejects a tool call
when the current step reaches the limit and then cuts off the turn. The final
step is expected to be an answer, not another tool call.

This contract cannot complete a sequential search, page fetch and answer: they
need at least three model steps. The first search output already receives the
final-answer instruction. The follow-up model requests a page anyway and is
stopped by the application. No page fetch/network failure occurred in that step.

There is also a source-quality consequence: the first answer cites URLs from
search results without fetching their pages. The saved instructions ask for
page verification, while the two-step budget leaves no sequential step for it.
Chat also disables the citation guard. This report does not validate the facts
or prices in those responses.

## Improvements

1. Make the Chat budget explicit and sufficient for search, fetch and answer.
   A minimal three-step path is necessary; a default of four to six should be
   evaluated with a retry and more than one source. Changing the Work slider or
   output-token limit does not change today's fixed Chat budget.
2. Separate the tool allowance from the final-answer allowance. When tool work
   reaches its budget, retain the retrieved content and run a bounded final
   response with no additional tool execution. Do not rely only on text asking
   the model to stop using tools, and keep explicit cancellation protection.
3. Keep the existing distinct notices for step limits, token limits, full
   context and unfinished operations. Explain the fixed Chat budget and add a
   continuation path for a step-limit stop, preserving prior tool results.
   The current Continue in Work button covers a refused note-save request, not
   this page-fetch cutoff.
4. Validate a follow-up using search and fetch, a retry, repeated tools, user
   cancellation and per-turn counter reset. Check that cited pages were actually
   fetched and that a bounded answer is produced when the tool budget ends.

As an immediate workaround, select Work in the upper mode control and send a
message asking it to continue the previous request. Work uses the configured
budget (eight in this saved conversation) in the same conversation. That does
not guarantee every request finishes, but it removes this fixed two-step Chat
cap. The proposed budget values and final-answer path still need implementation
and evaluation, including token/time bounds, cancellation and a no-answer fallback.

Execution-policy changes need their own implementation and acceptance checks;
they are not included in the presentation-only PR 144. The associated message
alignment request is implemented there as Issue 143 UI-09.

## Implemented follow-up

Issue #145 / PR #146 implements an independently configurable Chat budget
(default 16, range 1–64), a fresh budget for each ordinary question, and one
extra answer-only generation after a research cutoff. It preserves the same
session, forbids tools in the extra generation, and separates explicit Stop
from a budget stop. The diagnosis above remains the record of the old behavior.
See the [current contract](../specs/conversation-step-budget.md) and
[validation coverage](../validation/conversation-step-budget-20261016.md).
