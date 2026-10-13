import Foundation
import SpeechFeature
import SwiftUI
import UniformTypeIdentifiers

struct SpeechWAVDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.wav] }
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

struct SpeechExportPresenter: View {
    let ticket: SpeechExportTicket
    let data: Data
    let send: (SpeechAction) -> Void
    @State private var presented = false

    var body: some View {
        Color.clear.frame(width: 0, height: 0)
            .fileExporter(
                isPresented: $presented, document: SpeechWAVDocument(data: data),
                contentType: .wav, defaultFilename: "irodori-speech.wav"
            ) { result in
                switch result {
                case .success: send(.exportFinished(ticket, nil))
                case .failure(let error):
                    if (error as? CocoaError)?.code == .userCancelled {
                        send(.exportDismissed(ticket))
                    } else {
                        send(.exportFinished(ticket, error.localizedDescription))
                    }
                }
            }
            .onAppear { presented = true }
            .onChange(of: presented) { _, value in
                if !value { send(.exportDismissed(ticket)) }
            }
    }
}
