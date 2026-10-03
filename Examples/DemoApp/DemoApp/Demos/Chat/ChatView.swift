import SwiftUI

struct ChatView: View {
    @Environment(ChatViewModel.self) private var vm
    @Environment(DemoNavigator.self) private var navigator

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            switch vm.phase {
            case .ready:
                transcript
                Divider()
                statusBar
                composer
            case .loading(let preparing):
                LoadingView(name: vm.modelName, preparing: preparing, since: vm.loadStarted)
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
                    Text("Open a model bundle folder in Models, then load it.")
                } actions: {
                    Button("Open Models") { navigator.selection = .models }
                }
            }
        }
        .navigationTitle("Chat")
    }

    private var header: some View {
        @Bindable var vm = vm
        return VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(vm.modelName.isEmpty ? "No model loaded" : vm.modelName)
                        .font(.headline)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if let url = vm.bundleURL {
                        Text(ModelStorage.key(url))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
                Spacer()
                Button("New Chat", systemImage: "square.and.pencil", action: vm.newChat)
                    .disabled(vm.phase != .ready || vm.isResponding)
            }
            HStack(spacing: 20) {
                Picker("Thinking", selection: $vm.reasoning) {
                    ForEach(ChatViewModel.Reasoning.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .fixedSize()
                Picker("Max tokens", selection: $vm.maxTokens) {
                    ForEach(ChatViewModel.maxTokenChoices, id: \.self) { Text("\($0)").tag($0) }
                }
                .fixedSize()
                Spacer()
            }
            .controlSize(.small)
        }
        .padding(10)
    }

    private var transcript: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 16) {
                ForEach(vm.turns) { turn in
                    TurnView(turn: turn, live: vm.isResponding && turn.id == vm.turns.last?.id)
                }
            }
            .padding(14)
        }
        .defaultScrollAnchor(.bottom)
        .overlay {
            if vm.turns.isEmpty {
                ContentUnavailableView("New chat", systemImage: "bubble.left.and.bubble.right", description: Text("Ask anything. The reply streams in as the model writes it."))
            }
        }
    }

    private var statusBar: some View {
        HStack(spacing: 8) {
            if vm.isResponding {
                ProgressView().controlSize(.small)
            }
            Text(vm.isResponding ? (vm.retrieving ? "Collecting sources…" : "Generating…") : vm.statusLine)
                .font(.caption.monospacedDigit())
                .foregroundStyle(vm.last?.error == nil ? Color.secondary : Color.red)
                .lineLimit(2)
                .truncationMode(.tail)
                .textSelection(.enabled)
            Spacer()
            RAGControls()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    private var composer: some View {
        @Bindable var vm = vm
        return VStack(alignment: .leading, spacing: 8) {
            if let notice = vm.contextNotice {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.circle")
                    Text(notice)
                    Spacer(minLength: 0)
                    if vm.contextFull {
                        Button("New Chat", action: vm.newChat)
                    }
                }
                .font(.callout)
                .foregroundStyle(.orange)
            }
            HStack(alignment: .bottom, spacing: 8) {
                TextField("Message", text: $vm.input, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...6)
                    .padding(8)
                    .background(.background, in: .rect(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(.separator))
                    .disabled(vm.contextFull)
                    .onSubmit { vm.send() }
                if vm.isResponding {
                    Button("Stop", systemImage: "stop.circle.fill", action: vm.stop)
                        .labelStyle(.iconOnly)
                        .font(.system(size: 26))
                        .buttonStyle(.plain)
                        .foregroundStyle(.tint)
                        .keyboardShortcut(".", modifiers: .command)
                } else {
                    Button("Send", systemImage: "arrow.up.circle.fill") { vm.send() }
                        .labelStyle(.iconOnly)
                        .font(.system(size: 26))
                        .buttonStyle(.plain)
                        .foregroundStyle(vm.canSend ? Color.accentColor : Color.secondary)
                        .disabled(!vm.canSend)
                }
            }
        }
        .padding(10)
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
            Text(preparing
                ? "The first load of a model on this Mac prepares it for the GPU. This happens once per model; later loads reuse the prepared copy."
                : "Reading the model from disk.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 440)
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
