import CoreTransferable
import Foundation
import ImageGenerationFeature
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

struct ImagePNGView: View {
    let data: Data

    var body: some View {
        Self.image(data).resizable().scaledToFit().background {
            Canvas { context, size in
                let tile: CGFloat = 12
                for row in 0...Int(size.height / tile) {
                    for column in 0...Int(size.width / tile) {
                        let color: Color = (row + column).isMultiple(of: 2) ? .gray.opacity(0.14) : .gray.opacity(0.04)
                        context.fill(
                            Path(CGRect(x: CGFloat(column) * tile, y: CGFloat(row) * tile, width: tile, height: tile)),
                            with: .color(color))
                    }
                }
            }
        }
        .clipShape(.rect(cornerRadius: 8))
        .accessibilityElement(children: .ignore)
        .accessibilityAddTraits(.isImage)
    }

    static func image(_ data: Data) -> Image {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            return Image(systemName: "photo")
        }
        return Image(decorative: image, scale: 1)
    }
}

struct ImageReferenceConfirmation: View {
    let preview: ImageReferencePreview
    let send: (ImageGenerationAction) -> Void

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    ImagePNGView(data: preview.thumbnail).frame(maxHeight: 160)
                        .accessibilityLabel(Text("Original image", bundle: .module))
                } header: {
                    Text("Original image", bundle: .module)
                }
                Section {
                    ImagePNGView(data: preview.cropped).frame(maxHeight: 350)
                        .accessibilityLabel(Text("Centered square crop", bundle: .module))
                    Text("Confirm this square image to use it for editing.", bundle: .module)
                } header: {
                    Text("Centered square crop", bundle: .module)
                }
            }
            .formStyle(.grouped)
            .navigationTitle(Text("Confirm reference", bundle: .module))
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button {
                        send(.dismissReference(preview.id))
                    } label: {
                        Text("Cancel", bundle: .module)
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button {
                        send(.confirmReference(preview.id))
                    } label: {
                        Text("Use this crop", bundle: .module)
                    }.keyboardShortcut(.defaultAction)
                }
            }
        }
    }
}

nonisolated struct ImagePNGTransfer: Transferable {
    let data: Data
    let filename: String

    static var transferRepresentation: some TransferRepresentation {
        DataRepresentation(exportedContentType: .png) { $0.data }.suggestedFileName { $0.filename }
    }
}

struct ImagePNGExportPresenter: View {
    let ticket: ImageExportTicket
    let result: ImageGeneratedAsset
    let send: (ImageGenerationAction) -> Void
    @State private var presented = false

    var body: some View {
        Color.clear.frame(width: 0, height: 0)
            .fileExporter(
                isPresented: $presented, item: ImagePNGTransfer(data: result.png, filename: result.filename),
                contentTypes: [.png], defaultFilename: result.filename,
                onCompletion: { outcome in
                    switch outcome {
                    case .success: send(.exportFinished(ticket, true))
                    case .failure: send(.exportFinished(ticket, false))
                    }
                },
                onCancellation: { send(.exportCancelled(ticket)) }
            )
            .onAppear { presented = true }
            .onChange(of: presented) { _, value in if !value { send(.exportDismissed(ticket)) } }
    }
}

enum ImageGenerationCopy {
    static func failure(_ value: ImageGenerationFailure) -> LocalizedStringKey {
        switch value {
        case .unavailable: "Generation is unavailable. Return to Models and reopen this model."
        case .invalidImage: "The image could not be opened. Choose another image."
        case .invalidInput: "Check the prompt and seed, then try again."
        case .insufficientMemory: "Not enough memory is available for this model."
        case .invalidModel: "Model files could not be verified. Reimport the model from Models."
        case .generationFailed: "The image could not be generated. Try again."
        case .cleanupFailed: "Stopping has not completed. Retry stopping before starting again."
        case .exportFailed: "The PNG could not be saved. Choose a location and try again."
        case .promptTooLong: "The prompt is too long. Shorten it and try again."
        }
    }
}
