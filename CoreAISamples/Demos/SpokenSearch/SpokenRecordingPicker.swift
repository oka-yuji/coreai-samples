import SwiftUI
import UniformTypeIdentifiers

struct SpokenRecordingPicker: View {
    let complete: (Result<[URL], any Error>) -> Void
    @State private var choosing = true

    var body: some View {
        Color.clear.frame(width: 0, height: 0)
            .fileImporter(isPresented: $choosing, allowedContentTypes: [.audio], allowsMultipleSelection: false) {
                complete($0)
            } onCancellation: {
                complete(.success([]))
            }
    }
}
