import Foundation

public struct ReceiptSample: Identifiable, Equatable, Sendable {
    public let id: String
    public let title: String
    public let url: URL
    public let purpose: String
}

extension ReceiptPipeline {
    @concurrent public static func samples() async throws -> [ReceiptSample] {
        guard
            let manifest = Bundle.module.url(
                forResource: "receipts", withExtension: "json", subdirectory: "Resources/Examples")
        else { throw ReceiptRuntimeError.invalidSource }
        let descriptors = try JSONDecoder().decode([Descriptor].self, from: Data(contentsOf: manifest))
        return try descriptors.map { value in
            guard value.file == URL(filePath: value.file).lastPathComponent, value.file.hasSuffix(".png") else {
                throw ReceiptRuntimeError.invalidSource
            }
            return ReceiptSample(
                id: value.file, title: value.title,
                url: manifest.deletingLastPathComponent().appending(path: value.file),
                purpose: value.purpose)
        }
    }

    private struct Descriptor: Decodable {
        let file: String
        let title: String
        let purpose: String
    }
}
