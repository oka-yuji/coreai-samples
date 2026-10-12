import AppKit
import SwiftUI

struct AgentView: View {
    @Environment(ChatViewModel.self) private var chat
    @Environment(DemoNavigator.self) private var navigator
    @Environment(AgentViewModel.self) private var agent
    @State private var showSettings = false
    @FocusState private var inputFocused: Bool

    static let examples = [
        "Search for the latest Core ML news this year and summarise it in three lines.",
        "Read https://developer.apple.com/documentation/coreml and give me three key points.",
        "Look up tomorrow's weather in Tokyo, then save a summary as a note.",
    ]

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            switch chat.phase {
            case .ready:
                transcript
                Divider()
                if showSettings {
                    settings
                    Divider()
                }
                status
                composer
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
                    Text(
                        "The agent uses the model that Chat has loaded. Choose a model from the menu above, or open a bundle folder in Models."
                    )
                } actions: {
                    Button("Open Models") { navigator.selection = .models }
                }
            }
        }
        .navigationTitle("Agent")
    }

    private var header: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                ModelMenu(agent: agent)
                Text(
                    "web_search · fetch_page · write_note" + (agent.allowFileOperations ? " · move_note" : "")
                        + " — search via \(agent.searchProvider)"
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
            }
            Spacer()
            Button("Copy") { agent.copyTranscript() }
                .disabled(agent.entries.isEmpty)
                .help("Copy the whole transcript as Markdown")
            Button("New Task") { agent.reset() }
                .disabled(!agent.canReset)
            Button {
                showSettings.toggle()
            } label: {
                Image(systemName: "gearshape")
            }
            .accessibilityLabel("Agent settings")
        }
        .padding(10)
    }

    private var transcript: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 10) {
                if agent.entries.isEmpty { placeholder }
                ForEach(agent.entries) { entry in
                    AgentEntryRow(entry: entry)
                }
            }
            .padding(12)
        }
        .defaultScrollAnchor(.bottom)
    }

    private var placeholder: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Ask for something that needs the web:")
                .font(.callout)
                .foregroundStyle(.secondary)
            ForEach(Self.examples, id: \.self) { example in
                Button(example) { agent.input = example }
                    .buttonStyle(.plain)
                    .font(.callout)
                    .foregroundStyle(Color.accentColor)
                    .multilineTextAlignment(.leading)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var settings: some View {
        @Bindable var agent = agent
        return VStack(alignment: .leading, spacing: 8) {
            Picker("Reasoning effort", selection: $agent.effort) {
                ForEach(AgentReasoningEffort.allCases) { Text($0.label).tag($0) }
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 320)
            HStack(spacing: 8) {
                Text("Max steps: \(agent.maxIterations)")
                    .monospacedDigit()
                    .frame(width: 190, alignment: .leading)
                Slider(value: sliderBinding($agent.maxIterations), in: 1...16, step: 1)
                    .frame(maxWidth: 320)
            }
            HStack(spacing: 8) {
                Text("Page budget: \(agent.pageCharacterBudget) chars")
                    .monospacedDigit()
                    .frame(width: 190, alignment: .leading)
                Slider(value: sliderBinding($agent.pageCharacterBudget), in: 500...20_000, step: 500)
                    .frame(maxWidth: 320)
            }
            Picker("Max tokens", selection: $agent.maxNewTokens) {
                ForEach(AgentViewModel.maxTokenChoices, id: \.self) { Text("\($0)").tag($0) }
            }
            .fixedSize()
            HStack(spacing: 6) {
                Text("Brave Search API key")
                SecureField("optional — empty uses DuckDuckGo", text: $agent.braveAPIKey)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 280)
            }
            Toggle("Allow file operations (move_note, from the next task)", isOn: $agent.allowFileOperations)
            Toggle("Citation guard", isOn: $agent.citationGuard)
                .help(
                    "Check the answer's and the notes' URLs against the pages fetched in this task, ask the model to fix them once, and keep notes as drafts until the check passes"
                )
            HStack(spacing: 6) {
                Text("Notes folder")
                Text(agent.notesDirectory.path(percentEncoded: false))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.head)
                Button("Choose…") { chooseNotesFolder() }
                Button("Default") { agent.notesFolderPath = "" }
                    .disabled(agent.notesFolderPath.isEmpty)
            }
        }
        .controlSize(.small)
        .font(.caption)
        .disabled(agent.running)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func sliderBinding(_ value: Binding<Int>) -> Binding<Double> {
        Binding(get: { Double(value.wrappedValue) }, set: { value.wrappedValue = Int($0.rounded()) })
    }

    private func chooseNotesFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = "Choose"
        panel.message = "Where the agent saves its notes"
        panel.directoryURL = agent.notesDirectory
        guard panel.runModal() == .OK, let url = panel.url else { return }
        agent.notesFolderPath = url.path(percentEncoded: false)
    }

    private var status: some View {
        HStack(spacing: 8) {
            if agent.running { ProgressView().controlSize(.small) }
            Text(agent.statusLine.isEmpty ? "Ready." : agent.statusLine)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
                .textSelection(.enabled)
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 6) {
            if !agent.running && chat.isResponding {
                Text("Chat is replying on this model. The agent can start when it finishes.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            input
        }
        .padding(10)
    }

    private var input: some View {
        @Bindable var agent = agent
        return HStack(alignment: .bottom, spacing: 8) {
            TextField("Task", text: $agent.input, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...6)
                .padding(8)
                .background(.background, in: .rect(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(.separator))
                .focused($inputFocused)
                .disabled(agent.running)
                .onSubmit(sendIfPossible)
            if agent.running {
                Button("Stop", systemImage: "stop.circle.fill", action: agent.stop)
                    .labelStyle(.iconOnly)
                    .font(.system(size: 26))
                    .buttonStyle(.plain)
                    .foregroundStyle(.tint)
                    .keyboardShortcut(".", modifiers: .command)
            } else {
                Button("Send", systemImage: "arrow.up.circle.fill", action: sendIfPossible)
                    .labelStyle(.iconOnly)
                    .font(.system(size: 26))
                    .buttonStyle(.plain)
                    .foregroundStyle(agent.canSend(chat: chat) ? Color.accentColor : Color.secondary)
                    .disabled(!agent.canSend(chat: chat))
            }
        }
    }

    private func sendIfPossible() {
        guard agent.canSend(chat: chat) else { return }
        inputFocused = false
        agent.send(chat: chat)
    }
}

struct AgentEntryRow: View {
    let entry: AgentViewModel.Entry
    @State private var expanded = false

    var body: some View {
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
