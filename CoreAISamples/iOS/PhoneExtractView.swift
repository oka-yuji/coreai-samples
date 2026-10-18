#if os(iOS)
    import ExtractFeature
    import FeatureStore
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
                    Section {
                        PhoneExtractContent(state: state, text: $text, send: store.send)
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
    }
#endif
