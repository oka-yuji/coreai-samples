import AppKit
import ConversationFeature
import SwiftUI

struct ConversationView: View {
    @Environment(ChatViewModel.self) private var chat
    @Environment(ConversationHost.self) private var host
    @Environment(DemoNavigator.self) private var navigator
    @State private var showSettings = false

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
                    Text(message)
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
        .navigationTitle("Chat")
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
            Button {
                showSettings.toggle()
            } label: {
                Image(systemName: "gearshape")
            }
            .accessibilityLabel("Conversation settings")
        }
        .padding(10)
    }

    private var timeline: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 10) {
                ForEach(host.state.entries) { entry in
                    ConversationEntryRow(entry: entry)
                }
                ForEach(chat.ragTurns.map(\.turn)) { turn in
                    TurnView(turn: turn, live: chat.isResponding && turn.id == chat.ragTurns.last?.turn.id)
                }
            }
            .padding(12)
        }
        .defaultScrollAnchor(.bottom)
        .overlay {
            if host.state.entries.isEmpty && chat.ragTurns.isEmpty {
                ContentUnavailableView(
                    "New conversation", systemImage: "bubble.left.and.bubble.right",
                    description: Text(
                        "Chat answers and searches the web when it needs facts. Work runs multi-step research and saves notes."
                    ))
            }
        }
    }

    private var statusBar: some View {
        HStack(spacing: 8) {
            if host.state.isRunning || chat.isResponding {
                ProgressView().controlSize(.small)
            }
            Text(statusText)
                .font(.caption.monospacedDigit())
                .foregroundStyle(chat.ragIsOn && chat.last?.error != nil ? Color.red : Color.secondary)
                .lineLimit(2)
                .truncationMode(.tail)
                .textSelection(.enabled)
            Spacer()
            if chat.ragIsOn {
                Label(
                    "RAG" + (chat.ragWeb ? " · Web" : "") + (chat.ragLocal ? " · Docs" : ""),
                    systemImage: "books.vertical"
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    private var statusText: String {
        if chat.isResponding { return chat.retrieving ? "Collecting sources…" : "Generating…" }
        if chat.ragIsOn, chat.last != nil { return chat.statusLine }
        let line = host.state.statusLine
        return line.isEmpty ? chat.loadNote : line
    }

    private func copy() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(ConversationSnapshot(host.state).markdown(), forType: .string)
    }
}

extension ConversationState {
    var statusLine: String {
        guard let turn = turns.last else { return "" }
        let limit = turn.policy.maxSteps
        if isRunning {
            if stopRequested { return "Stopping…" }
            if case .toolCalls(_, let calls, let metadata)? = transcript.entries.last, metadata != nil,
                transcript.entries.count > turn.start + 1
            {
                return "Step \(step): running " + calls.map(\.name).joined(separator: " + ") + "…"
            }
            return "Step \(step) of \(limit): generating…"
        }
        if turn.finalization == .cutOff { return "Stopped at the \(limit)-step limit." }
        if turn.finish == .stopped { return "Stopped after \(step) step(s)." }
        if let failure = turn.failure {
            return failure.hasPrefix(ConversationSession.contextFullPrefix)
                ? "Context overflow at step \(step)." : "Failed at step \(step)."
        }
        return turn.steps.last?.line ?? ""
    }

    var contextEndNotice: String? {
        guard !isRunning, let turn = turns.last, turn.finish == .cap else { return nil }
        let reserved = transcript.entries.reversed().lazy.compactMap { entry -> Int? in
            if case .response(_, _, let metadata) = entry { metadata?.reservedTokens } else { nil }
        }.first
        guard let reserved, reserved < turn.policy.maxTokens else { return nil }
        return
            "The reply stopped at the end of the context window (\(reserved) tokens were left for it). Start a new conversation if the next message does not fit."
    }
}

struct ConversationEntryRow: View {
    let entry: ConversationEntry
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
                    Text(entry.kind == .toolCall ? "Calling \(entry.title)" : entry.title)
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
                Text(entry.text).textSelection(.enabled)
                statsCaption
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        case .failure:
            VStack(alignment: .leading, spacing: 2) {
                label(entry.title, "exclamationmark.triangle")
                    .foregroundStyle(.orange)
                Text(entry.text)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        case .note:
            Label(
                "\(entry.title): \(entry.text)",
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
    private var statsCaption: some View {
        if let stats = entry.stats {
            Text(stats)
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
                    Text(entry.title)
                    Text("(\(entry.text.count) chars)").foregroundStyle(.secondary)
                    if entry.live { ProgressView().controlSize(.mini) }
                }
                .font(.caption)
            }
            .buttonStyle(.plain)
            Text(expanded ? entry.text : preview)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
            statsCaption
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var preview: String {
        let flat = entry.text.replacingOccurrences(of: "\n", with: " ")
        return flat.count > 300 ? String(flat.prefix(300)) + "…" : flat
    }

    private func label(_ text: String, _ systemImage: String) -> some View {
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
                Text(turn.response.isEmpty ? "…" : turn.response)
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
                Text(live ? "Thinking" : "Thoughts")
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

    var body: some View {
        VStack(spacing: 12) {
            ProgressView()
            TimelineView(.periodic(from: since, by: 1)) { context in
                Text("\(preparing ? "Preparing" : "Loading") \(name)… \(Int(context.date.timeIntervalSince(since))) s")
                    .font(.headline)
                    .monospacedDigit()
            }
            Text(
                preparing
                    ? "The first load of a model on this Mac prepares it for the GPU. This happens once per model; later loads reuse the prepared copy."
                    : "Reading the model from disk."
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
