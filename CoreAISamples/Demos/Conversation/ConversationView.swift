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
    @State private var timelinePosition = ScrollPosition(edge: .bottom)

    var body: some View {
        VStack(spacing: 0) {
            archiveStatus
            if let markdown = host.state.readOnlyMarkdown {
                ScrollView {
                    Text(markdown).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading).padding()
                }
            } else {
                if chat.phase != .ready || !host.matchesConversationModel {
                    modelStatus
                        .padding(20)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                if chat.phase == .ready || !host.state.entries.isEmpty {
                    timeline
                } else {
                    Spacer(minLength: 0)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .safeAreaBar(edge: .top, spacing: 0) {
            header
        }
        .safeAreaBar(edge: .bottom, spacing: 0) {
            if host.state.readOnlyMarkdown == nil && chat.phase == .ready && host.matchesConversationModel {
                VStack(spacing: 0) {
                    statusBar
                    ComposerView()
                }
            }
        }
        .demoNavigationTitle(host.state.mode == .chat ? "Chat" : "Work")
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button("Copy conversation", systemImage: "doc.on.doc", action: copy)
                    .disabled(host.state.entries.isEmpty && host.state.readOnlyMarkdown == nil)
                    .help("Copy the conversation as Markdown")
                Button("Conversation settings", systemImage: "gearshape") { showSettings.toggle() }
                    .help("Conversation settings")
                    .disabled(chat.phase != .ready || host.state.readOnlyMarkdown != nil || host.state.isRestoring)
                    .popover(isPresented: $showSettings, arrowEdge: .bottom) {
                        VStack(alignment: .leading, spacing: 0) {
                            HStack {
                                Text("Conversation settings").font(.headline)
                                Spacer()
                                Button("Done") { showSettings = false }
                            }
                            .padding(16)
                            Divider()
                            ScrollView { ConversationSettingsView() }
                                .frame(maxHeight: 520)
                        }
                        .frame(width: 560)
                    }
            }
        }
        .onChange(of: host.state.id) { showSettings = false }
    }

    @ViewBuilder
    private var archiveStatus: some View {
        if host.state.isRestoring {
            HStack {
                ProgressView().controlSize(.small)
                Text("Opening conversation…")
            }.padding(8)
        } else if case .failed(let message) = host.state.persistence {
            HStack {
                Image(systemName: "exclamationmark.triangle")
                VStack(alignment: .leading) {
                    Text(
                        LocalizedStringKey(
                            host.state.readOnlyMarkdown == nil
                                ? "Could not save the conversation" : "Read-only conversation"))
                    Text(message).font(.caption).textSelection(.enabled)
                }
                Spacer()
                if host.state.readOnlyMarkdown == nil, !host.state.turns.isEmpty {
                    Button("Retry save") { host.store.send(.retrySave) }
                    Button("Save a copy", action: host.saveCopy)
                }
            }
            .padding(8)
            .background(Color.orange.opacity(0.1))
        } else if host.state.isSaving {
            Text("Saving conversation…").font(.caption).padding(4)
        }
    }

    @ViewBuilder
    private var modelStatus: some View {
        switch chat.phase {
        case .ready:
            VStack(alignment: .leading, spacing: 10) {
                Label("Choose the conversation model", systemImage: "arrow.triangle.2.circlepath")
                    .font(.headline)
                Text("Choose \(host.state.modelID) to continue this conversation.")
                    .foregroundStyle(.secondary)
                Button("Open Models") { navigator.selection = .models }
            }
        case .loading(let preparing):
            LoadingView(name: chat.modelName, preparing: preparing, since: chat.loadStarted)
        case .failed(let message):
            VStack(alignment: .leading, spacing: 10) {
                Label("Could not open the model", systemImage: "exclamationmark.triangle")
                    .font(.headline)
                RuntimeMessageText(message).foregroundStyle(.secondary).textSelection(.enabled)
                Button("Open Models") { navigator.selection = .models }
            }
        case .empty:
            VStack(alignment: .leading, spacing: 10) {
                Label("No model loaded", systemImage: "square.and.arrow.down")
                    .font(.headline)
                Group {
                    if !host.state.modelID.isEmpty {
                        Text("Choose \(host.state.modelID) to continue this conversation.")
                    } else {
                        Text("Choose a model from the menu above, or open a bundle folder in Models.")
                    }
                }
                .foregroundStyle(.secondary)
                Button("Open Models") { navigator.selection = .models }
            }
        }
    }

    private var header: some View {
        GlassEffectContainer(spacing: 16) {
            HStack(spacing: 12) {
                modePicker
                    .layoutPriority(1)
                Spacer(minLength: 0)
                ModelMenu()
                    .buttonStyle(.glass)
                    .frame(maxWidth: 360, alignment: .trailing)
            }
            .controlSize(.large)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
    }

    private var modePicker: some View {
        Picker(
            "Mode",
            selection: Binding(get: { host.state.mode }, set: { host.store.send(.modeChanged($0)) })
        ) {
            Text("Chat").tag(ConversationMode.chat)
            Text("Work").tag(ConversationMode.work)
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .frame(width: 180)
        .help(modeDescription)
        .accessibilityHint(modeDescription)
        .disabled(
            host.state.isRunning || host.state.isRestoring || host.state.isTerminating
                || chat.isResponding || chat.ragIsOn || host.state.readOnlyMarkdown != nil)
    }

    private var modeDescription: Text {
        Text(
            chat.ragIsOn
                ? "Answers from the collected sources in one reply."
                : host.state.mode == .chat
                    ? "Conversation. Searches the web when it needs facts." : "Multi-step research and notes."
        )
    }

    @ViewBuilder
    private var timeline: some View {
        if host.state.entries.isEmpty {
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    Text("New conversation").font(.title2.weight(.semibold))
                    Text(
                        "Chat answers and searches the web when it needs facts. Work runs multi-step research and saves notes."
                    )
                    .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(20)
            }
            .frame(maxHeight: .infinity, alignment: .top)
        } else {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    ForEach(host.state.entries) { entry in
                        ConversationEntryRow(entry: entry)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(20)
            }
            .scrollPosition($timelinePosition)
            .defaultScrollAnchor(.bottom)
            .defaultScrollAnchor(.top, for: .alignment)
            .onScrollGeometryChange(for: TimelineFit.self) { geometry in
                TimelineFit(
                    height: max(
                        0, geometry.containerSize.height - geometry.contentInsets.top - geometry.contentInsets.bottom),
                    atBottom: geometry.contentOffset.y + geometry.containerSize.height - geometry.contentInsets.bottom
                        >= geometry.contentSize.height - 16)
            } action: { old, new in
                if new.height < old.height, old.atBottom { timelinePosition.scrollTo(edge: .bottom) }
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
                .foregroundStyle(Color.secondary)
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
        .padding(.horizontal, 20)
        .padding(.vertical, 6)
    }

    private var ragLabel: String {
        let strings = DemoStrings(locale: locale)
        return (["RAG"] + (chat.ragWeb ? [strings.text("Web")] : []) + (chat.ragLocal ? [strings.text("Docs")] : []))
            .joined(separator: " · ")
    }

    @ViewBuilder
    private var statusView: some View {
        if host.state.stopRequested {
            Text("Stopping…")
        } else if let preparation = host.state.preparation {
            Text("Preparing conversation… \(preparation.processed) / \(preparation.total) tokens")
        } else if host.state.isRetrieving {
            Text("Collecting sources…")
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
        NSPasteboard.general.setString(
            host.state.readOnlyMarkdown ?? ConversationSnapshot(host.state).markdown(), forType: .string)
    }
}

private struct TimelineFit: Equatable {
    var height: CGFloat
    var atBottom: Bool
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
            if entry.title == "Sources" || entry.title == "Source citations" {
                VStack(alignment: .leading, spacing: 3) {
                    Label(
                        LocalizedStringKey(entry.title),
                        systemImage: entry.warning ? "exclamationmark.triangle" : "books.vertical")
                    ForEach(Array(entry.text.components(separatedBy: "\n").enumerated()), id: \.offset) { _, line in
                        if line.hasPrefix("Unavailable source IDs: ") {
                            Text("Unavailable source IDs: \(String(line.dropFirst("Unavailable source IDs: ".count)))")
                        } else if entry.title == "Source citations" {
                            RuntimeMessageText(line)
                        } else {
                            Text(verbatim: line)
                        }
                    }
                }
                .font(.caption)
                .foregroundStyle(entry.warning ? Color.orange : Color.secondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                Label(
                    "Note (original): \(entry.title): \(entry.text)",
                    systemImage: entry.warning ? "exclamationmark.triangle" : "checkmark.seal"
                )
                .font(.caption)
                .foregroundStyle(entry.warning ? Color.orange : Color.secondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
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
        if case .cutToolCall(_, let note)? = entry.detail {
            Text(LocalizedStringKey(note ? "Note was not saved" : "Tool call was not executed"))
        } else if case .tokenLimit(let limit)? = entry.detail {
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
        case .cutToolCall(let name, let note)?:
            if note, name == "write_note" {
                Text("The write_note call was cut at the token limit; the note was not saved.")
            } else if note {
                Text(
                    "The append_note call was cut at the token limit; the new part was not saved. Earlier parts are preserved."
                )
            } else {
                Text("The tool call was cut at the token limit and was not executed.")
            }
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

struct LoadingView: View {
    let name: String
    let preparing: Bool
    let since: Date
    @Environment(\.locale) private var locale

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                ProgressView().controlSize(.small)
                TimelineView(.periodic(from: since, by: 1)) { context in
                    Text(
                        "\(DemoStrings(locale: locale).text(preparing ? "Preparing" : "Loading")) \(name)… \(Int(context.date.timeIntervalSince(since))) s"
                    )
                    .font(.headline)
                    .monospacedDigit()
                }
            }
            Text(
                LocalizedStringKey(
                    preparing
                        ? "The first load of a model on this Mac prepares it for the GPU. This happens once per model; later loads reuse the prepared copy."
                        : "Reading the model from disk.")
            )
            .font(.callout)
            .foregroundStyle(.secondary)
            .frame(maxWidth: 560, alignment: .leading)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
