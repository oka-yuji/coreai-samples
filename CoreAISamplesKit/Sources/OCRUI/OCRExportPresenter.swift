#if os(macOS)
    import Foundation
    import OCRFeature
    import SwiftUI
    import UniformTypeIdentifiers

    nonisolated struct OCRFileDocument: FileDocument {
        static var markdownType: UTType { UTType(filenameExtension: "md") ?? .plainText }
        static var readableContentTypes: [UTType] { [markdownType, .plainText, .json] }
        var data: Data

        init(data: Data) { self.data = data }
        init(configuration: ReadConfiguration) throws {
            guard let bytes = configuration.file.regularFileContents else { throw CocoaError(.fileReadCorruptFile) }
            data = bytes
        }
        func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
            FileWrapper(regularFileWithContents: data)
        }
    }

    struct OCRExportPresenter: View {
        let ticket: OCRExport
        let send: (OCRAction) -> Void
        @State private var presented = false

        var body: some View {
            Color.clear.frame(width: 0, height: 0)
                .fileExporter(
                    isPresented: $presented, document: OCRFileDocument(data: ticket.data),
                    contentType: ticket.format == .json ? .json : OCRFileDocument.markdownType,
                    defaultFilename: ticket.name + (ticket.format == .json ? ".json" : ".md")
                ) { result in
                    switch result {
                    case .success:
                        send(.exportFinished(id: ticket.id, session: ticket.session, error: nil))
                    case .failure(let error):
                        if (error as? CocoaError)?.code == .userCancelled {
                            send(.exportDismissed(id: ticket.id, session: ticket.session))
                        } else {
                            send(
                                .exportFinished(
                                    id: ticket.id, session: ticket.session, error: error.localizedDescription))
                        }
                    }
                }
                .onAppear { presented = true }
                .onChange(of: presented) { _, value in
                    if !value { send(.exportDismissed(id: ticket.id, session: ticket.session)) }
                }
        }
    }
#endif
