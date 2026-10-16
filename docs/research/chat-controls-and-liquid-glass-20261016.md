# Chat controls and Liquid Glass

Research date: 2026-10-16. Scope: the follow-up to Issue 143 and PR 144.
The user requested a left-aligned Chat / Work switcher, model selection at the
right edge of the same row, and a native Liquid Glass appearance.

## Product references

| Product | Placement described in the official documentation | Design implication |
| --- | --- | --- |
| ChatGPT desktop | Chat / Work is selected above the composer at the top of a new chat. Model and reasoning selection is a separate control beneath the composer in Work / Codex. | Treat task mode and model selection as distinct choices. |
| Claude | The model name beside the send button opens the model selector. | Model choice is available near the current interaction, without becoming a separate page. |
| Gemini | The instructions place model selection at the bottom of the chat box. | A model is a compact per-conversation control. |

Sources: [ChatGPT quickstart](https://learn.chatgpt.com/docs/quickstart),
[ChatGPT model selection](https://learn.chatgpt.com/docs/models?surface=app),
[Claude model-switch instructions](https://support.claude.com/en/articles/8664678-change-the-model-effort-and-thinking-settings),
[Gemini model selection](https://support.google.com/gemini/answer/18648722?hl=en).

These are documentation checks, not direct inspections of the signed-in apps.
The exact centered position in ChatGPT is the user's observation; the official
text establishes the upper location, not exact pixel alignment. Placement can
vary by client and version. There is no single position shared by every product.
For this demo, the explicit left-mode/right-model request takes precedence over
copying a particular product's composer controls.

## Apple material and layout references

[Adopting Liquid Glass](https://developer.apple.com/documentation/technologyoverviews/adopting-liquid-glass)
recommends standard bars and controls, reducing backgrounds that obscure system
effects, and reserving custom glass for important controls. It also recommends
testing reduced transparency, contrast and motion settings.

[Materials](https://developer.apple.com/design/human-interface-guidelines/materials)
distinguishes the floating control/navigation layer from the content layer.
Regular glass supports text legibility; clear glass is intended for richer media
backgrounds. Conversation text should remain ordinary readable content.

[Applying Liquid Glass to custom views](https://developer.apple.com/documentation/swiftui/applying-liquid-glass-to-custom-views)
provides real `glassEffect` and `GlassEffectContainer` APIs. `safeAreaBar` registers
custom edge controls with the scrolling area instead of requiring a hard divider.
The installed SDK 27 declares `Glass`, `GlassEffectContainer`, `glassEffect`,
`GlassButtonStyle` and `safeAreaBar` available from macOS 26, below this app's
macOS 27 deployment target.

## Adopted design

- One upper row: the standard Chat / Work segmented picker on the leading edge,
  with the model menu on the trailing edge. The mode explanation becomes a help
  tooltip and accessibility hint. A long model name truncates before displacing
  the mode control; its full value remains in the accessibility value.
- The model menu uses the standard glass button style. The mode picker remains
  a native segmented control, retaining keyboard and selection semantics.
  No extra glass layer is placed behind that native control.
- Top and bottom `safeAreaBar` regions replace the fixed dividers. New, copy and
  settings keep their existing native toolbar positions.
- One regular-glass surface contains the message field and send/stop controls.
  The previous opaque rectangle and manual border are removed. Its child buttons
  do not add nested glass layers. The transcript, notices and saved content are
  not decorated with glass.
- There is no custom blur, simulated gloss, forced color scheme or custom motion.
  System materials retain their accessibility adaptation; actual appearance
  under different settings remains a manual acceptance check.

This changes presentation only. Model loading, history protection, mode guards,
generation and persistence keep their existing owners and behavior. Verification
is recorded in [the layout evidence](../validation/conversation-layout-20261016.md).
