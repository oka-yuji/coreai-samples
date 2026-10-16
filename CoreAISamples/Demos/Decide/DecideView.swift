import DecideFeature
import DecideRuntime
import FeatureStore
import SwiftUI
import UniformTypeIdentifiers

@MainActor
enum DecideAssembly {
    static func makeStore() -> Store<DecideFeature> {
        let service = LiveDecideService(
            examplesURL: Bundle.main.url(forResource: "decide-examples", withExtension: "json"))
        return Store(initialState: DecideState(), run: DecideRunner(service: service).run)
    }
}

struct DecideView: View {
    @Bindable var store: Store<DecideFeature>
    @State private var importingDocument = false
    @State private var editing: DecideItem?
    @State private var search = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            HSplitView {
                documentPanel.frame(minWidth: 250, idealWidth: 340)
                VStack(alignment: .leading, spacing: 12) {
                    checklist
                    Divider()
                    results
                }
                .frame(minWidth: 350, maxWidth: .infinity, maxHeight: .infinity)
                .padding(.leading, 12)
            }
            .frame(maxHeight: .infinity)
            status
        }
        .padding(20)
        .demoNavigationTitle("Decide")
        .onAppear { store.send(.appeared) }
        .onDisappear { store.send(.ended) }
        .fileImporter(isPresented: $importingDocument, allowedContentTypes: [.plainText, .text]) { result in
            switch result {
            case .success(let url): store.send(.readDocument(url))
            case .failure(let error):
                if (error as? CocoaError)?.code != .userCancelled {
                    store.send(.uiError("The selected document could not be opened."))
                }
            }
        }
        .sheet(item: $editing) { item in
            DecideItemEditor(item: item) { updated in
                store.send(.updateItem(updated))
                editing = nil
            }
        }
        .background {
            if let ticket = store.state.exportTicket, let data = store.state.exportData {
                DecideExportPresenter(ticket: ticket, data: data, send: store.send).id(ticket)
            }
        }
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Document checklist").font(.title2.bold())
                Group {
                    if let name = store.state.model?.name {
                        Text(verbatim: name)
                    } else {
                        Text("Open a Clef-Flash decision bundle to begin.")
                    }
                }
                .font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
            }
            Spacer()
            DecideModelButton(store: store)
        }
    }

    private var documentPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Document").font(.headline)
                Spacer()
                Menu("Examples") {
                    ForEach(store.state.examples) { example in
                        Button(LocalizedStringKey(example.title)) {
                            search = ""
                            store.send(.replaceInput(example.input))
                        }
                    }
                }
                .disabled(store.state.busy || store.state.examples.isEmpty)
                Button("Import…", systemImage: "doc") { importingDocument = true }
                    .disabled(store.state.busy)
            }
            HStack {
                TextField("Find in document", text: $search)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityIdentifier("decide.search")
                if !search.isEmpty {
                    Button("Clear", systemImage: "xmark.circle.fill") { search = "" }
                        .labelStyle(.iconOnly).buttonStyle(.plain)
                }
            }
            Group {
                if search.isEmpty {
                    TextEditor(
                        text: Binding(
                            get: { store.state.input.document },
                            set: { store.send(.setDocument($0)) })
                    )
                    .font(.system(.body, design: .monospaced))
                    .autocorrectionDisabled()
                    .disabled(store.state.busy)
                    .accessibilityLabel("Document text")
                    .accessibilityIdentifier("decide.document")
                } else {
                    ScrollView {
                        Text(highlightedDocument)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled).padding(8)
                    }
                    .accessibilityLabel("Document search results")
                }
            }
            .background(.background)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
            tokenLabel.font(.caption).foregroundStyle(.secondary)
            if let problem = store.state.countError ?? store.state.input.problem, !store.state.input.document.isEmpty {
                RuntimeMessageText(problem).font(.caption).foregroundStyle(.orange)
            }
        }
        .padding(.trailing, 12)
    }

    private var checklist: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Checklist").font(.headline)
                Text("\(store.state.input.items.count) / 16").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Add item", systemImage: "plus") { store.send(.addItem) }
                    .accessibilityIdentifier("decide.add")
                    .disabled(store.state.busy || store.state.input.items.count >= 16)
            }
            ScrollView {
                VStack(spacing: 4) {
                    ForEach(store.state.input.items) { item in
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(item.key).lineLimit(1)
                                Text(LocalizedStringKey(item.kind.title)).font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button("Move up", systemImage: "chevron.up") { store.send(.moveItem(item.id, -1)) }
                                .disabled(store.state.input.items.first?.id == item.id)
                            Button("Move down", systemImage: "chevron.down") { store.send(.moveItem(item.id, 1)) }
                                .disabled(store.state.input.items.last?.id == item.id)
                            Button("Edit", systemImage: "pencil") { editing = item }
                                .accessibilityLabel(Text("Edit \(item.key)"))
                                .accessibilityIdentifier("decide.edit.\(item.id)")
                            Button("Remove", systemImage: "minus.circle") { store.send(.removeItem(item.id)) }
                        }
                        .buttonStyle(.borderless)
                        .labelStyle(.iconOnly)
                        .padding(.horizontal, 8).padding(.vertical, 5)
                        .background(.quinary, in: RoundedRectangle(cornerRadius: 6))
                    }
                }
            }
            .frame(minHeight: 80, maxHeight: 195)
            .disabled(store.state.busy)
            HStack {
                Button("Run checklist", systemImage: "play.fill") { store.send(.run(reuse: true)) }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(!store.state.canRun)
                    .accessibilityIdentifier("decide.run")
                Button("Read again") { store.send(.run(reuse: false)) }
                    .disabled(!store.state.canRun)
                    .help("Read the complete document again to compare with reuse.")
                if store.state.busy {
                    Button("Stop") { store.send(.stop) }
                        .keyboardShortcut(.cancelAction)
                        .disabled(store.state.phase == .stopping)
                }
            }
        }
    }

    private var results: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Results").font(.headline)
                Spacer()
                Toggle(
                    "Lowest probability first",
                    isOn: Binding(
                        get: { store.state.lowestFirst }, set: { store.send(.setLowestFirst($0)) })
                )
                .toggleStyle(.checkbox).font(.caption)
                Button("Export JSON", systemImage: "square.and.arrow.up") { store.send(.prepareExport) }
                    .disabled(!store.state.currentResult || store.state.busy)
            }
            Text("Model probability (uncalibrated)").font(.caption).foregroundStyle(.secondary)
            if store.state.output != nil && !store.state.currentResult {
                Text("The checklist changed. Run it again to update these results.")
                    .font(.caption).foregroundStyle(.orange)
            }
            if store.state.output?.recomputedAll == true {
                Text("Recomputed all items").font(.caption).foregroundStyle(.secondary)
            }
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    ForEach(store.state.rows) { row in
                        DecideResultRow(
                            row: row, previous: store.state.previous?.rows.first { $0.id == row.id },
                            changed: store.state.changedItems.contains(row.id),
                            confirmation: store.state.confirmations[row.id],
                            enabled: store.state.currentResult && !store.state.busy,
                            confirm: { store.send(.confirm(row.id, $0)) })
                    }
                    if store.state.output == nil {
                        ContentUnavailableView(
                            "No decisions yet", systemImage: "checklist",
                            description: Text("Run the checklist to compare model candidates with the source document.")
                        )
                    }
                }
            }
        }
        .frame(maxHeight: .infinity)
    }

    private var status: some View {
        HStack {
            if store.state.busy || store.state.loadingExamples {
                ProgressView().controlSize(.small)
                Text(LocalizedStringKey(phaseLabel)).font(.callout)
            } else if let message = store.state.message {
                RuntimeMessageText(message).font(.callout).textSelection(.enabled)
            }
            Spacer()
            if let initial = store.state.initialSeconds {
                Text("Initial read \(initial, format: .number.precision(.fractionLength(2))) s")
            }
            if let reused = store.state.reuseSeconds {
                Text("Reused \(reused, format: .number.precision(.fractionLength(2))) s")
            }
            if let output = store.state.output, output.reusedDocument {
                Text("\(output.reusedTokens) tokens reused")
            }
        }
        .font(.caption).foregroundStyle(.secondary)
    }

    @ViewBuilder
    private var tokenLabel: some View {
        if let tokens = store.state.inputTokens {
            Text("\(tokens) / \(store.state.model?.tokenLimit ?? 9_216) input tokens")
        } else if store.state.model == nil {
            Text("Token count is available after opening a model.")
        } else {
            Text("Input token count pending")
        }
    }

    private var phaseLabel: String {
        switch store.state.phase {
        case .loading: "Opening and preparing the decision model…"
        case .running: "Evaluating the checklist…"
        case .stopping: "Stopping the current operation…"
        case .reading: "Reading the document…"
        case .idle, .closed: "Loading example documents…"
        }
    }

    private var highlightedDocument: AttributedString {
        var text = AttributedString()
        var remaining = store.state.input.document[...]
        while !remaining.isEmpty, let range = remaining.range(of: search, options: .caseInsensitive), !range.isEmpty {
            text.append(AttributedString(remaining[..<range.lowerBound]))
            var match = AttributedString(remaining[range])
            match.backgroundColor = .yellow.opacity(0.35)
            text.append(match)
            remaining = remaining[range.upperBound...]
        }
        text.append(AttributedString(remaining))
        return text
    }
}

struct DecideModelButton: View {
    let store: Store<DecideFeature>
    @Environment(DemoNavigator.self) private var navigator
    @State private var importing = false

    var body: some View {
        Button("Open decision bundle…", systemImage: "folder") { importing = true }
            .disabled(store.state.busy)
            .fileImporter(isPresented: $importing, allowedContentTypes: [.folder]) { result in
                switch result {
                case .success(let url):
                    store.send(.appeared)
                    store.send(.openBundle(url.path))
                    navigator.selection = .decide
                case .failure(let error):
                    if (error as? CocoaError)?.code != .userCancelled {
                        store.send(.appeared)
                        store.send(.uiError("The selected decision folder could not be opened."))
                        navigator.selection = .decide
                    }
                }
            }
    }
}

struct DecisionJSONDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.json] }
    var data: Data

    init(data: Data) { self.data = data }

    init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents else { throw CocoaError(.fileReadCorruptFile) }
        self.data = data
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}

private struct DecideExportPresenter: View {
    let ticket: DecideExportTicket
    let data: Data
    let send: (DecideAction) -> Void
    @State private var presented = false

    var body: some View {
        Color.clear.frame(width: 0, height: 0)
            .fileExporter(
                isPresented: $presented, document: DecisionJSONDocument(data: data),
                contentType: .json, defaultFilename: "decision-review"
            ) { result in
                switch result {
                case .success: send(.exportFinished(ticket, nil))
                case .failure(let error):
                    if (error as? CocoaError)?.code == .userCancelled {
                        send(.dismissExport(ticket))
                    } else {
                        send(.exportFinished(ticket, error.localizedDescription))
                    }
                }
            }
            .onAppear { presented = true }
            .onChange(of: presented) { _, value in
                if !value { send(.dismissExport(ticket)) }
            }
    }
}
