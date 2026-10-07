import AppKit
import SwiftUI

/// A plain-text editor that reports the committed text, and when an input method starts and stops converting a
/// word, so that a word still being converted never starts an extraction. `TextEditor` does not tell whether text is
/// marked, and `NSTextView` sends no `textDidChange` while it is.
struct NoteEditor: NSViewRepresentable {
    let text: String
    let onChange: (String) -> Void
    let onComposing: (Bool) -> Void

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = ComposingTextView.scrollableTextView()
        guard let textView = scrollView.documentView as? ComposingTextView else { return scrollView }
        textView.delegate = context.coordinator
        textView.onComposing = { [weak coordinator = context.coordinator] in coordinator?.onComposing($0) }
        textView.isRichText = false
        textView.allowsUndo = true
        textView.font = .systemFont(ofSize: NSFont.systemFontSize + 1)
        textView.textContainerInset = NSSize(width: 4, height: 6)
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.string = text
        textView.setAccessibilityLabel("Note")
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        context.coordinator.onChange = onChange
        context.coordinator.onComposing = onComposing
        guard let textView = scrollView.documentView as? NSTextView, !textView.hasMarkedText(), textView.string != text
        else { return }
        textView.string = text
    }

    func makeCoordinator() -> Coordinator { Coordinator(onChange: onChange, onComposing: onComposing) }

    @MainActor
    final class Coordinator: NSObject, NSTextViewDelegate {
        var onChange: (String) -> Void
        var onComposing: (Bool) -> Void

        init(onChange: @escaping (String) -> Void, onComposing: @escaping (Bool) -> Void) {
            self.onChange = onChange
            self.onComposing = onComposing
        }

        /// Sent for committed text only: a commit replaces the marked text, so the conversion has ended.
        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView, !textView.hasMarkedText() else { return }
            onComposing(false)
            onChange(textView.string)
        }
    }
}

/// Reports whether text is marked after every change of the marked text, including a conversion cancelled with no
/// change to the text.
final class ComposingTextView: NSTextView {
    var onComposing: (Bool) -> Void = { _ in }

    override func setMarkedText(_ string: Any, selectedRange: NSRange, replacementRange: NSRange) {
        super.setMarkedText(string, selectedRange: selectedRange, replacementRange: replacementRange)
        onComposing(hasMarkedText())
    }

    override func unmarkText() {
        super.unmarkText()
        onComposing(hasMarkedText())
    }
}
