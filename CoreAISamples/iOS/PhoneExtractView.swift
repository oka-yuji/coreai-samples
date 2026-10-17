#if os(iOS)
    import ExtractFeature
    import FeatureStore
    import GLiNERCore
    import SwiftUI

    struct PhoneExtractView: View {
        let store: Store<ExtractFeature>
        let assetFolders: [URL]
        let computeUnits: ComputeUnits
        @State private var text = ExtractResources.examples.notes.first ?? ""

        private var state: ExtractState { store.state }

        private var isOpening: Bool {
            if case .opening = state.asset { true } else { false }
        }

        var body: some View {
            NavigationStack {
                Form {
                    Section {
                        Label(ExtractRunner.notice, systemImage: "info.circle")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Section {
                        ForEach(assetFolders, id: \.self) { folder in
                            Button(folder.lastPathComponent) {
                                store.send(.assetFolderChosen(path: folder.path(percentEncoded: false)))
                            }
                        }
                        .disabled(isOpening)
                        assetStatus
                    } header: {
                        Text("Asset folder in Documents")
                    } footer: {
                        Text("Compute units: \(computeUnits.rawValue)")
                    }
                    Section("Note") {
                        TextField("Type or paste a note", text: $text, axis: .vertical)
                            .lineLimit(3...8)
                        Button("Extract") {
                            store.send(.textChanged(text))
                            store.send(.extractRequested)
                        }
                        .disabled(text.allSatisfy(\.isWhitespace))
                        runStatus
                    }
                    if let mapping = state.result?.mapping {
                        Section("Labels") {
                            ForEach(mapping.fields, id: \.field.key, content: labelRow)
                        }
                    }
                }
                .navigationTitle("GLiNER2.5-multi")
            }
        }

        @ViewBuilder private var assetStatus: some View {
            switch state.asset {
            case .none:
                Text(assetFolders.isEmpty ? "No asset folder in Documents." : "Choose an asset folder.")
                    .foregroundStyle(.secondary)
            case .opening(let path):
                HStack(spacing: 8) {
                    ProgressView()
                    Text("Opening \(URL(filePath: path).lastPathComponent)…")
                }
            case .ready(let info):
                let seconds = info.loadSeconds.formatted(.number.precision(.fractionLength(1)))
                let revision = info.revision.map { " · revision \($0.prefix(7))" } ?? ""
                Text(verbatim: "\(URL(filePath: info.path).lastPathComponent) · loaded in \(seconds) s\(revision)")
            case .failed(_, let message):
                Text("Could not open the asset: \(message)").foregroundStyle(.red)
            }
        }

        @ViewBuilder private var runStatus: some View {
            switch state.phase {
            case .extracting:
                Text("Extracting…").foregroundStyle(.secondary)
            case .failed(let message):
                Text(message).foregroundStyle(.red)
            case .ready, .idle:
                if let timing = state.result?.timing {
                    Text(ExtractRunner.describe(timing)).font(.caption).monospacedDigit().foregroundStyle(.secondary)
                }
            }
        }

        private func labelRow(_ label: FieldMapping) -> some View {
            VStack(alignment: .leading, spacing: 4) {
                Text(label.field.name).font(.headline)
                if label.candidates.isEmpty {
                    Text("Not detected").foregroundStyle(.secondary)
                }
                ForEach(Array(label.candidates.enumerated()), id: \.offset) { _, candidate in
                    Text(
                        "\u{201C}\(candidate.span.text)\u{201D} span "
                            + candidate.span.probability.formatted(.number.precision(.fractionLength(2)))
                    )
                }
            }
        }
    }
#endif
