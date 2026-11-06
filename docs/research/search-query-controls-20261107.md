# Search query controls

Issue #427, source snapshot `fcedb7f`. This is a source-based assessment and implementation proposal. No production search behavior has changed. The owner's WSQ-03 choice is pending.

## WSQ-01: Outbound paths

| Input and boundary | Source | Current control and remaining scope |
|---|---|---|
| The model supplies `web_search.query`; AgentTools trims it and sends it as DuckDuckGo's `q` parameter | `CoreAISamples/Demos/Agent/AgentTools.swift:76`, `:167`, `:193` | ToolRunner checks the turn's allowed tools, but does not ask for approval of the actual query. The query can be derived from any material visible to the model. |
| RAG searches the submitted question before generation | `CoreAISamples/Demos/Chat/ChatRAG.swift:156`, `CoreAISamples/Demos/Chat/RAGInputs.swift:15` | The Web source setting enables the request. It sends the question even without a model tool call, so controlling only ToolRunner leaves this path unchanged. |
| `fetch_page` and RAG page retrieval send destination URLs to remote servers | `CoreAISamples/Demos/Conversation/ToolRunner.swift:263`, `CoreAISamples/Demos/Agent/AgentTools.swift:210`, `CoreAISamples/Demos/Chat/RAGInputs.swift:16` | ToolRunner restricts fetch origins to the user's URLs and discovered URLs; HTTPS/address/redirect checks remain independent. The fetch tool's `query` is used locally to select page text, not appended to the destination request. |
| Pure cassette replay and fixed RAG evaluation read recorded responses | `CoreAISamples/Demos/Agent/AgentTools.swift:727`, `CoreAISamples/Demos/Chat/RAGEvaluationReplay.swift:29` | Replay misses do not fall back to the network. Record and replay-or-record can invoke the live request and must follow the live policy. |

The active search provider is DuckDuckGo; the older Issue text mentioning Brave predates #431/PR #432. This assessment covers the conversation's production search/page routes, not all networking or webpage subresources in the application. It does not demonstrate an actual disclosure of private user material.

## WSQ-02: Options

| Option | Effect | Remaining limitation | Interaction and evaluation impact |
|---|---|---|---|
| Confirm the exact query immediately before each live search | The request cannot send an unapproved query; covers both model queries and initial RAG search when placed at both boundaries | A person can still approve sensitive text; this is not an automatic confidentiality classifier | Adds a visible decision per search. Pure replay remains unchanged; live headless evaluation needs an explicit noninteractive policy. |
| Search only the current user-written question and reject model-generated follow-up queries | Prevents earlier conversation or fetched material from being copied into an added model query | The current question itself may contain sensitive material; repeated searches lose useful reformulation | Changes multi-step research and cassette keys if queries are substituted. Rejecting rather than silently rewriting preserves the meaning of an existing key. |
| Disable follow-up search after reading external material | Restricts some subsequent query paths | The first query and pre-existing private conversation remain exposed; RAG already fetches before model generation | Requires tracking retrieval state and reduces iterative research. |
| Show queries without waiting for approval | Makes outgoing content visible | Does not prevent transmission | Small UI change but cannot meet a pre-send approval contract. |
| Limit query length or filter particular strings | Reduces some accidental copying | Cannot establish that an allowed short query contains no private information | Can reject legitimate searches and alter frozen evaluation requests. |
| Keep Web off by default | Prevents live search while off | Already the default; enabling Web still permits arbitrary generated queries | Retains current behavior but does not address the enabled case. |

The recommended option is confirmation of the actual query for every live search, including the initial RAG request. The final outgoing normalized query and destination must match what the person approved. An approval is specific to the active conversation/run/request and cannot be reused after cancellation, a new turn, restoration or a query change.

## Proposed implementation contract

This is a concrete proposal for WSQ-03, not an adopted product contract.

- Pending query, destination and request identity belong to the conversation Feature state. The Store owns the asynchronous operation and its cancellation. A service owns the suspended request and resolves it exactly once.
- The UI shows the full outgoing query and destination with Send and Decline. Until Send, the live network request has not started. Decline returns an explicit tool/source refusal; it does not silently substitute another query.
- Stop, closing, switching conversations and stale completion discard the pending approval and end the wait. An old approval cannot release a newer request.
- Both ToolRunner live search and RAGInputs.live.search use the same approval boundary. The existing Web-off and fetch-origin checks remain in force.
- Pure cassette and fixed-corpus replay do not send data and do not prompt. A replay miss stays a miss. Recording or replay-or-record cannot bypass approval when they reach the network.
- A live headless run without an explicit policy refuses the request. The name and scope of any explicit noninteractive opt-in must be included in the adopted contract. Existing offline fixture keys and result schemas remain unchanged.
- No model, GPU, external API or new classifier is needed to verify the permission state and fake transport boundaries. Actual UI acceptance remains an owner check.

## Follow-up implementation Issue draft

Title: `feat: ChatとWorkで外部検索の前に実際の検索語を確認する`

Purpose: Let the user decide whether a particular query may be sent to the search provider, while keeping local answers and recorded evaluation available.

Acceptance criteria proposed for approval:

- [test] A live model search and an initial RAG search make zero transport calls before approval.
- [test] Approval sends the displayed query to the displayed destination exactly once.
- [test] Decline, Stop, close and conversation switching finish the pending request without transmitting it.
- [test] A stale approval, changed query or restored conversation cannot approve another request.
- [test] Web-off remains enforced, pure replay performs no live request, and replay misses remain errors.
- [cmd] Fixed offline evaluation retains its cassette keys and output schema, and live headless behavior follows the explicit adopted policy.
- [manual] The owner can inspect, approve and decline both kinds of query in the Mac UI and continue or stop the conversation.

The implementation estimate and final split remain pending the selected behavior and headless contract. No approval is inferred from elapsed time. WSQ-01 and WSQ-02 have evidence above; WSQ-03 remains pending. The independent #429 and #425 work can proceed on top of this documentation-only branch.
