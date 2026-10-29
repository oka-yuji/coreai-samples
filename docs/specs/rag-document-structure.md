# Original document structure

Issue #374; parent #378 plan v1.1. This adds document structure acquisition, not strategy B or C.

`RAGDocumentSnapshot` retains the original document identity, display URL, final URL, format, raw and normalized text, and the normalized UTF-8 SHA-256. The stored version is checked on decoding. Its source fields are immutable; adding structure does not rewrite the document. Existing source IDs continue to use the original document identity, so recording a redirect destination does not renumber citations.

`RAGDocumentStructure` contains the parser version, content and preamble ranges, and ordered sections. Each section has a unique document/version/position ID, its original heading and direct-body ranges, a semantic title, and parent/ancestor IDs. Body ranges stop before the next heading; child text is not duplicated into parents. The title is search metadata and is not an invented original-text quotation. All persisted ranges use Swift Character offsets.

## Parsers and mapping

- HTML: SwiftSoup 2.14.3, revision `e7f85f4d3dd91ca9a1e94d3328aa1ed5120a8c0f`. Select an eligible single article, single main, or body. Check the candidate and its ancestors for non-body/hidden contexts. Unknown or mixed content roots fall back. Parse-only operations do not execute scripts or fetch resources.
- Markdown: swift-markdown 0.9.0, revision `25cb61d3482054b09ae76ca4f281b1bfe7fe5a43`, with swift-cmark 0.9.0. Use top-level AST headings; code blocks, fences, raw HTML and nested quotation/list headings are not promoted into the document outline.
- Markdown locations are one-based UTF-8 byte columns. Map them only at original grapheme boundaries. Confirm an AST heading's actual ATX line or Setext underline before using its end; the pinned parser can report a Setext range extending into the following paragraph.
- HTML heading markers are projected through the same `RAGHTMLText` normalizer used by the existing app. Accept structure only when the full projected text and each heading match the original UTF-8 bytes. No fuzzy occurrence matching is used for repeated headings.

The existing HTML-to-text implementation moved without changing its function bodies or entity table. `AgentTools` delegates to it, keeping A's text extraction and source hashes unchanged.

Limits are one MiB of raw and normalized text, 256 headings, HTML depth 128 and 16,384 nodes. A limit, unsupported format, ambiguous root or failed mapping produces a recorded `unavailableReason` and preserves readable text. This is static markup structure, not a browser layout/CSS visibility analysis.

## Acquisition and cancellation

`RAGInputs` carries complete documents/pages through the existing live or replay acquisition boundary. Static HTML and Markdown can receive structure. Text, PDF-extracted text and rendered text remain readable but structurally unavailable. The existing allowed destinations, redirects, size checks and render-fallback refusal are preserved.

Acquisition and parsing belong to the same retrieval request. Cancellation is checked before accepting the page and after parsing; cancelled retrieval returns no sources or document metadata. The existing Runner also checks cancellation/run identity before source registration.

Normal A does not parse structure or retain full document diagnostics. `--rag-structure` enables capture for inspection. Later strategies opt into it explicitly. Diagnostic recording is enabled by the evaluation command; it is separate from ordinary UI execution.

## Inspect without model weights

```sh
CoreAISamples --rag-eval Tools/RAGEvaluation/corpus.json --validate-only \
  --structure-output /new/structure.json
```

This passes fixed HTML and Markdown through the existing retrieval boundary and verifies that the resulting original text and hash match the manifest. The output includes explicit versions and structure. It does not load the Qwen model, enable B/C, prove model quality, or change the normal UI.
