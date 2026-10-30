#if os(macOS)
    import AppKit
    import FeatureStore
    import OCRFeature
    import SwiftUI
    import UniformTypeIdentifiers

    public struct OCRPageView: View {
        private let store: Store<OCRFeature>
        @State private var choosingImage = false

        public init(store: Store<OCRFeature>) { self.store = store }

        public var body: some View {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Read a document", bundle: .module).font(.title2.bold())
                        Text("Choose an image to read its text and tables.", bundle: .module).foregroundStyle(
                            .secondary)
                    }
                    Spacer()
                    if let document = store.state.document { exports(document) }
                }
                HSplitView {
                    input.frame(minWidth: 240, idealWidth: 300, maxWidth: 380)
                    result.frame(minWidth: 300, maxWidth: .infinity, maxHeight: .infinity)
                }
                if let message = store.state.message {
                    Text(LocalizedStringKey(message), bundle: .module)
                        .font(.callout).foregroundStyle(.red).textSelection(.enabled)
                }
            }
            .padding(20)
            .fileImporter(isPresented: $choosingImage, allowedContentTypes: [.image]) { result in
                switch result {
                case .success(let url): store.send(.selectImage(url))
                case .failure(let error):
                    if (error as? CocoaError)?.code != .userCancelled {
                        store.send(.uiError(error.localizedDescription))
                    }
                }
            }
            .background {
                if let ticket = store.state.export {
                    OCRExportPresenter(ticket: ticket, send: store.send).id(ticket.id)
                }
            }
        }

        private var input: some View {
            VStack(spacing: 14) {
                ZStack {
                    RoundedRectangle(cornerRadius: 8).fill(.quaternary.opacity(0.4))
                    if let data = store.state.input?.preview, let preview = NSImage(data: data) {
                        Image(nsImage: preview).resizable().scaledToFit().padding(10)
                            .accessibilityLabel(Text("Selected document", bundle: .module))
                    } else {
                        VStack(spacing: 10) {
                            Image(systemName: "doc.viewfinder").font(.largeTitle).foregroundStyle(.secondary)
                            Text("Drop an image file here", bundle: .module).foregroundStyle(.secondary)
                        }.padding()
                    }
                }
                .frame(minHeight: 220, maxHeight: .infinity)
                .dropDestination(for: URL.self) { urls, _ in
                    guard store.state.editable, urls.count == 1, let url = urls.first, url.isFileURL else {
                        return false
                    }
                    store.send(.selectImage(url))
                    return true
                }
                if let input = store.state.input {
                    Text(verbatim: input.name).lineLimit(1).truncationMode(.middle)
                    Text("\(input.image.width) × \(input.image.height) px", bundle: .module)
                        .font(.caption).foregroundStyle(.secondary)
                }
                Button {
                    choosingImage = true
                } label: {
                    Label {
                        Text("Choose Image…", bundle: .module)
                    } icon: {
                        Image(systemName: "photo")
                    }
                }.disabled(!store.state.editable)
                HStack {
                    Button {
                        store.send(.read)
                    } label: {
                        Text("Read", bundle: .module)
                    }
                    .buttonStyle(.borderedProminent).disabled(!store.state.canRead)
                    Button {
                        store.send(.stop)
                    } label: {
                        Text("Stop", bundle: .module)
                    }
                    .disabled(!store.state.busy || store.state.phase == .stopping)
                }
                status
            }.padding(.trailing, 14)
        }

        @ViewBuilder private var status: some View {
            switch store.state.phase {
            case .selecting: ProgressView { Text("Opening image…", bundle: .module) }
            case .loading:
                VStack(spacing: 6) {
                    ProgressView { Text("Preparing model…", bundle: .module) }
                    Text("The first preparation may take longer.", bundle: .module).font(.caption).foregroundStyle(
                        .secondary)
                }
            case .reading: ProgressView { Text("Reading document…", bundle: .module) }
            case .stopping: ProgressView { Text("Stopping…", bundle: .module) }
            case .cancelled: Text("Stopped. Choose an image or read again.", bundle: .module).font(.callout)
            default: EmptyView()
            }
        }

        @ViewBuilder private var result: some View {
            if let document = store.state.document {
                OCRResultView(document: document)
            } else {
                ContentUnavailableView {
                    Label {
                        Text("Document results", bundle: .module)
                    } icon: {
                        Image(systemName: "doc.text")
                    }
                } description: {
                    Text("Text and tables will appear here after reading.", bundle: .module)
                }
            }
        }

        private func exports(_ document: OCRDocument) -> some View {
            HStack {
                Menu {
                    Button {
                        store.send(.exportRequested(.markdown))
                    } label: {
                        Text("Save Markdown…", bundle: .module)
                    }
                    Button {
                        store.send(.exportRequested(.json))
                    } label: {
                        Text("Save JSON…", bundle: .module)
                    }
                } label: {
                    Text("Save", bundle: .module)
                }
                Menu {
                    ShareLink(item: document.markdown) { Text("Share Markdown", bundle: .module) }
                    if let data = try? document.json(), let text = String(data: data, encoding: .utf8) {
                        ShareLink(item: text) { Text("Share JSON", bundle: .module) }
                    } else {
                        Button {
                            store.send(.exportRequested(.json))
                        } label: {
                            Text("Share JSON", bundle: .module)
                        }
                    }
                } label: {
                    Text("Share", bundle: .module)
                }
            }.disabled(!store.state.editable)
        }
    }
#endif
