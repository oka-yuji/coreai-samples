# Search and source access

Issue #175, specification v1, October 18, 2026. The user approved replacing the
ambiguous RAG toggle with a control whose off state performs no new retrieval,
while keeping Chat and Work independent. PR #176 implements this contract.

## User behavior

| Control | Behavior |
|---|---|
| Search and sources off | Answer from model knowledge and existing conversation. Do not search the web, fetch pages or read the document folder. Work may still save notes. |
| On, Web | Retrieve web sources before answering. Work may perform additional web calls within the existing step budget. |
| On, Docs | Retrieve documents from `~/Documents/rag/`; prohibit web tools. |
| On, Web and Docs | Retrieve both selected source types. |
| On, neither selected | Show the source-selection explanation and disable sending and continuation. Never silently treat this as off. |

The source budget limits retrieved input, not answer length. Existing supported
document formats, selection/ranking, token budgets and network address checks
are unchanged. Empty or failed retrieval ends with its actual result and no
model-only fallback. Stop cancels retrieval and leaves the conversation usable.

Chat and Work stay selected across source collection. Chat's source reply does
not call tools. Work retains its note tools and task budget. Its source prompt
allows note saving rather than prescribing a three-sentence, tool-free reply.
The switch starts off at app launch. Turning it off does not erase earlier
messages or retrieved material; the model can still refer to that history.

## Execution and storage contract

The existing Feature/Store/Runner ownership remains in place. Host captures
the selected access in each send. The Feature decides tool permission; Session
sets native tool calling to disallowed when no tool is permitted. ToolRunner
rechecks cancellation and the active allowed tool set after its asynchronous
token snapshot and immediately before tool execution. The per-turn prompt also
states that disabled web tools must not be used or claimed as used.

`webSearchEnabled` is optional in archived settings. Missing legacy values mean
off. New turns store an explicit choice. Format 5 reads formats 1–5; older apps
use their existing future-format read-only path. Continue and citation repair
intersect the original permission with the current setting, so a later switch
cannot broaden an existing turn's permission. Settings cannot change while a
turn is executing. Existing run IDs reject stale results.

Ordinary Host sends, including Continue in Work, collect selected sources when
enabled. The existing `--agent-e2e` diagnostic explicitly enables Web and uses
`collectSources: false` to preserve its model-driven tool benchmark; this only
skips precollection and does not bypass the tool permission checks.

At a research cutoff, the final answer keeps source IDs, native history,
the fetched-page ledger and draft notes. Its effective policy drops the
retrieval request as well as all tool permissions. The historical turn still
records its original retrieval request for archive reconstruction.

Work's automatic citation requirement accepts valid, supplied source IDs as
well as its existing URL path. A valid `[S1]` reference must not trigger a
redundant repair merely because no URL was repeated in the answer. Unknown URLs,
unknown or unsupplied IDs, and a missing answer citation still fail validation.
An explicit `sourcesRequired: true` retains the diagnostic requirement for a
fetched URL. Source identity checks do not prove factual support for a claim.

## Acceptance and verification

ACCESS-01–07 retain their wording in Issue #175. The regression first reproduced
both unauthorized web execution with the toggle off and Work being forced into
Chat, using the pre-change Feature sources. Pure tests cover permissions,
continuation, repair, archive migration and finalization. Native Foundation
Models CPU fixtures cover actual runner/tool gates, selected retrieval flags,
Work notes, empty/failed/cancelled retrieval and citation checks.

The [validation record](../validation/conversation-search-access-20261018.md)
records commands, outcomes, review corrections and manual limitations. The
existing model-first UI work and unrelated catalog changes in the primary
checkout are outside this PR. The provisional 5 SP remains unchanged; this is
one implementation unit because the label and its execution guarantee must
arrive together.
