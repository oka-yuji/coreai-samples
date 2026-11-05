#if os(macOS)
    import Foundation
    import ReceiptFeature
    import SwiftUI
    import UniformTypeIdentifiers

    struct ReceiptExportPicker: View {
        let selection: ReceiptExportSelection
        let send: (ReceiptAction) -> Void
        @State private var choosing = true

        var body: some View {
            Color.clear.frame(width: 0, height: 0)
                .fileImporter(isPresented: $choosing, allowedContentTypes: [.folder], allowsMultipleSelection: false) {
                    result in
                    switch result {
                    case .success(let directories): send(.exportSelected(selection.id, directories.first))
                    case .failure(let error):
                        send(.exportSelected(selection.id, nil))
                        if (error as? CocoaError)?.code != .userCancelled { send(.uiError(error.localizedDescription)) }
                    }
                } onCancellation: {
                    send(.exportSelected(selection.id, nil))
                }
        }
    }
#endif
