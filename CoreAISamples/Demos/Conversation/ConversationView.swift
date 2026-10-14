import AppKit
import ConversationFeature
import RAGContext
import SwiftUI

struct ConversationView: View {
    @Environment(ChatViewModel.self) private var chat
    @Environment(ConversationHost.self) private var host
    @Environment(DemoNavigator.self) private var navigator
    @Environment(\.locale) private var locale
    @State private var showSettings = false
    private static let timelineEnd = "timeline-end"

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            switch chat.phase {
            case .ready:
                timeline
                Divider()
                if showSettings {
                    ConversationSettingsView()
                    Divider()
                }
                statusBar
                ComposerView()
            case .loading(let preparing):
                LoadingView(name: chat.modelName, preparing: preparing, since: chat.loadStarted)
            case .failed(let message):
                ContentUnavailableView {
                    Label("Could not open the model", systemImage: "exclamationmark.triangle")
                } description: {
                    RuntimeMessageText(message)
                } actions: {
                    Button("Open Models") { navigator.selection = .models }
                }
            case .empty:
                ContentUnavailableView {
                    Label("No model loaded", systemImage: "square.and.arrow.down")
                } description: {
                    Text("Choose a model from the menu above, or open a bundle folder in Models.")
                } actions: {
                    Button("Open Models") { navigator.selection = .models }
                }
            }
        }
        .demoNavigationTitle("Chat")
    }

    private var header: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                ModelMenu()
                if let url = chat.bundleURL {
                    Text(ModelStorage.key(url))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            Spacer()
            Button("New", systemImage: "square.and.pencil", action: host.startNew)
                .disabled(chat.phase != .ready || host.state.isRunning || chat.isResponding)
                .help("Start a new conversation")
            Button("Copy", systemImage: "doc.on.doc", action: copy)
                .disabled(host.state.entries.isEmpty)
                .help("Copy the conversation as Markdown")
            Toggle(isOn: $showSettings) {
                Label("Conversation settings", systemImage: "gearshape")
            }
            .toggleStyle(.button)
            .labelStyle(.iconOnly)
            .help("Show or hide the conversation settings")
            .disabled(chat.phase != .ready)
        }
        .padding(10)
    }

    @ViewBuilder
    private var timeline: some View {
        if host.state.entries.isEmpty && chat.ragTurns.isEmpty {
            ContentUnavailableView(
                "New conversation", systemImage: "bubble.left.and.bubble.right",
                description: Text(
                    "Chat answers and searches the web when it needs facts. Work runs multi-step research and saves notes."
                )
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 0) {
                        VStack(alignment: .leading, spacing: 10) {
                            ForEach(TimelineItem.merge(host.state.entries, chat.ragTurns)) { item in
                                switch item {
                                case .entry(let entry):
                                    ConversationEntryRow(entry: entry)
                                case .rag(let turn):
                                    TurnView(turn: turn, live: chat.isResponding && turn.id == chat.ragTurns.last?.id)
                                }
                            }
                        }
                        .padding(12)
                        Color.clear.frame(height: 1).id(Self.timelineEnd)
                    }
                }
                .defaultScrollAnchor(.bottom)
                .onScrollGeometryChange(for: TimelineFit.self) { geometry in
                    TimelineFit(
                        height: geometry.containerSize.height,
                        atBottom: geometry.visibleRect.maxY >= geometry.contentSize.height - 16)
                } action: { old, new in
                    if new.height < old.height, old.atBottom {
                        proxy.scrollTo(Self.timelineEnd, anchor: .bottom)
                    }
                }
            }
        }
    }

    private var statusBar: some View {
        HStack(spacing: 8) {
            if host.state.isRunning || chat.isResponding {
                ProgressView().controlSize(.small)
            }
            statusView
                .font(.caption.monospacedDigit())
                .foregroundStyle(chat.ragIsOn && chat.last?.error != nil ? Color.red : Color.secondary)
                .lineLimit(2)
                .truncationMode(.tail)
                .textSelection(.enabled)
            Spacer()
            if chat.ragIsOn {
                Label {
                    Text(verbatim: ragLabel)
                } icon: {
                    Image(systemName: "books.vertical")
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    private var ragLabel: String {
        let strings = DemoStrings(locale: locale)
        return (["RAG"] + (chat.ragWeb ? [strings.text("Web")] : []) + (chat.ragLocal ? [strings.text("Docs")] : []))
            .joined(separator: " · ")
    }

    @ViewBuilder
    private var statusView: some View {
        if chat.isResponding {
            Text(chat.retrieving ? "Collecting sources…" : "Generating…")
        } else if chat.ragIsOn, chat.last != nil {
            RuntimeMessageText(chat.statusLine)
        } else if let status = host.state.status {
            switch status {
            case .generating(let step, let limit): Text("Step \(step) of \(limit): generating…")
            case .running(let step, let tools): Text("Step \(step): running \(tools.joined(separator: " + "))…")
            case .stopping: Text("Stopping…")
            case .cutOff(let limit): Text("Stopped at the \(limit)-step limit.")
            case .stopped(let steps):
                if host.state.turns.last?.mode == .chat, let stopped = host.runner.session.lastGeneration {
                    RuntimeMessageText(stopped.statusLine)
                } else {
                    Text("Stopped after \(steps) step(s).")
                }
            case .failed(let step): Text("Failed at step \(step).")
            case .overflow(let step): Text("Context overflow at step \(step).")
            case .finished(let stats): RuntimeMessageText(stats.line)
            }
        } else {
            RuntimeMessageText(chat.loadNote)
        }
    }

    private func copy() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(ConversationSnapshot(host.state).markdown(), forType: .string)
    }
}

private struct TimelineFit: Equatable {
    var height: CGFloat
    var atBottom: Bool
}

enum TimelineItem: Identifiable {
    case entry(ConversationEntry)
    case rag(ChatViewModel.Turn)

    var id: String {
        switch self {
        case .entry(let entry): entry.id
        case .rag(let turn): "rag-" + turn.id
        }
    }

    static func merge(_ entries: [ConversationEntry], _ rag: [ChatViewModel.Turn]) -> [TimelineItem] {
        var items: [TimelineItem] = []
        var next = 0
        for (index, entry) in entries.enumerated() {
            while next < rag.count, rag[next].anchor <= index {
                items.append(.rag(rag[next]))
                next += 1
            }
            items.append(.entry(entry))
        }
        return items + rag[next...].map(TimelineItem.rag)
    }
}

struct ConversationEntryRow: View {
    let entry: ConversationEntry
    @Environment(\.locale) private var locale
    @State private var expanded = false

    var body: some View {
        if entry.collapsed {
            DisclosureGroup(isExpanded: $expanded) {
                Text(entry.text)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                statsCaption
            } label: {
                HStack(spacing: 4) {
                    title
                    Text("(\(entry.text.count) chars)").foregroundStyle(.secondary)
                    if entry.live { ProgressView().controlSize(.mini) }
                }
                .font(.caption)
            }
        } else {
            row
        }
    }

    @ViewBuilder
    private var row: some View {
        switch entry.kind {
        case .user:
            VStack(alignment: .leading, spacing: 2) {
                label("You", "person")
                Text(entry.text).textSelection(.enabled)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        case .answer:
            VStack(alignment: .leading, spacing: 2) {
                label("Assistant", "sparkles")
                Group {
                    if entry.detail == .noAnswer { Text("(no answer text)") } else { Text(entry.text) }
                }
                .textSelection(.enabled)
                statsCaption
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        case .failure:
            VStack(alignment: .leading, spacing: 2) {
                Label {
                    failureTitle
                } icon: {
                    Image(systemName: "exclamationmark.triangle")
                }
                .font(.caption)
                .foregroundStyle(.orange)
                failureBody
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        case .note:
            Label(
                "Note (original): \(entry.title): \(entry.text)",
                systemImage: entry.warning ? "exclamationmark.triangle" : "checkmark.seal"
            )
            .font(.caption)
            .foregroundStyle(entry.warning ? Color.orange : Color.secondary)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
        case .toolCall:
            VStack(alignment: .leading, spacing: 4) {
                label("Calling \(entry.title)", "wrench.and.screwdriver")
                Text(entry.text)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                statsCaption
            }
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.background, in: .rect(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(.separator))
        case .thinking, .toolResult:
            collapsible
        }
    }

    @ViewBuilder
    private var failureTitle: some View {
        if case .tokenLimit(let limit)? = entry.detail {
            Text("Answer cut at the token limit (\(limit))")
        } else {
            RuntimeMessageText(entry.title)
        }
    }

    @ViewBuilder
    private var failureBody: some View {
        switch entry.detail {
        case .stepLimit(let steps, let fetched, let saved, let open)?:
            VStack(alignment: .leading, spacing: 2) {
                Text("The turn reached its \(steps)-step limit before a final answer.")
                Text("Fetched: \(list(fetched))")
                Text("Saved notes: \(list(saved))")
                Text("Still open: \(list(open))")
            }
        case .stopped(partial: false)?:
            Text("(stopped before any answer)")
        case .tokenLimit?:
            Text(
                "The answer above stops where generation ran out of tokens. Raise Max tokens or ask for something shorter."
            )
        case .failure?:
            failureMessage
        default:
            Text(entry.text)
        }
    }

    @ViewBuilder
    private var failureMessage: some View {
        let prefix = ConversationFeature.stopReasonPrefix
        if let numbers = ConversationFeature.contextFullNumbers(entry.text) {
            Text(
                "The context window is full: the conversation needs \(numbers.needed) tokens and the limit is \(numbers.limit). Start a new conversation, lower the page budget or ask for something narrower."
            )
        } else if entry.text == ConversationFeature.endedWithoutStopReason {
            Text("Generation ended without a stop reason")
        } else if entry.text.hasPrefix(prefix) {
            Text("Generation ended with stop reason: \(String(entry.text.dropFirst(prefix.count)))")
        } else {
            RuntimeMessageText(entry.text)
        }
    }

    private func list(_ items: [String]) -> String {
        items.isEmpty ? DemoStrings(locale: locale).text("none") : items.joined(separator: ", ")
    }

    @ViewBuilder
    private var statsCaption: some View {
        if let stats = entry.stats {
            RuntimeMessageText(stats)
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.tertiary)
        }
    }

    private var collapsible: some View {
        VStack(alignment: .leading, spacing: 4) {
            Button {
                expanded.toggle()
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: expanded ? "chevron.down" : "chevron.right")
                    title
                    Text("(\(entry.text.count) chars)").foregroundStyle(.secondary)
                    if entry.live { ProgressView().controlSize(.mini) }
                }
                .font(.caption)
            }
            .buttonStyle(.plain)
            .accessibilityValue(expanded ? Text("Expanded") : Text("Collapsed"))
            Text(expanded ? entry.text : preview)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
            statsCaption
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private var title: some View {
        switch entry.kind {
        case .toolCall: Text("Calling \(entry.title)")
        case .thinking: Text(LocalizedStringKey(entry.live ? "Thinking" : "Thoughts"))
        default: Text("Tool result (original): \(entry.title)")
        }
    }

    private var preview: String {
        let flat = entry.text.replacingOccurrences(of: "\n", with: " ")
        return flat.count > 300 ? String(flat.prefix(300)) + "…" : flat
    }

    private func label(_ text: LocalizedStringKey, _ systemImage: String) -> some View {
        Label(text, systemImage: systemImage)
            .font(.caption)
            .foregroundStyle(.secondary)
    }
}

struct TurnView: View {
    let turn: ChatViewModel.Turn
    let live: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("You")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(turn.prompt)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(10)
                .background(Color.accentColor.opacity(0.12), in: .rect(cornerRadius: 10))
            Text("Assistant")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .padding(.top, 4)
            if !turn.reasoning.isEmpty {
                ThinkingView(text: turn.reasoning, live: live && turn.response.isEmpty)
            }
            if !turn.response.isEmpty || (live && turn.reasoning.isEmpty) {
                Group {
                    if turn.response == RAGContext.noSourcesReply {
                        Text("Could not retrieve any sources.")
                    } else {
                        Text(turn.response.isEmpty ? "…" : turn.response)
                    }
                }
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(10)
                .background(.quaternary.opacity(0.5), in: .rect(cornerRadius: 10))
            }
            if !turn.notes.isEmpty || !turn.sources.isEmpty {
                RAGSourceFooter(notes: turn.notes, sources: turn.sources)
                    .padding(.horizontal, 10)
            }
        }
    }
}

struct ThinkingView: View {
    let text: String
    let live: Bool
    @State private var expanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            Text(text)
                .font(.callout)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 4)
        } label: {
            HStack(spacing: 6) {
                if live {
                    ProgressView().controlSize(.small)
                }
                Text(LocalizedStringKey(live ? "Thinking" : "Thoughts"))
                    .fontWeight(.medium)
                if !expanded {
                    Text(text.split(whereSeparator: \.isNewline).last.map(String.init) ?? "")
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.head)
                }
            }
            .font(.callout)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.quaternary.opacity(0.3), in: .rect(cornerRadius: 10))
    }
}

struct LoadingView: View {
    let name: String
    let preparing: Bool
    let since: Date
    @Environment(\.locale) private var locale

    var body: some View {
        VStack(spacing: 12) {
            ProgressView()
            TimelineView(.periodic(from: since, by: 1)) { context in
                Text(
                    "\(DemoStrings(locale: locale).text(preparing ? "Preparing" : "Loading")) \(name)… \(Int(context.date.timeIntervalSince(since))) s"
                )
                .font(.headline)
                .monospacedDigit()
            }
            Text(
                LocalizedStringKey(
                    preparing
                        ? "The first load of a model on this Mac prepares it for the GPU. This happens once per model; later loads reuse the prepared copy."
                        : "Reading the model from disk.")
            )
            .font(.callout)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
            .frame(maxWidth: 440)
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
