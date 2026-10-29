import CoreGraphics
import CryptoKit
import Foundation
import ImageIO
import OCRCore
import OCRFeature
import OCRRuntime
import UniformTypeIdentifiers

protocol OCRPageExecuting: Sendable {
    func read(_ input: OCRInput) async throws -> OCRDocument
}

actor OCRPipeline {
    typealias Loader = @Sendable (URL) async throws -> any OCRPageExecuting
    private let loader: Loader
    private let onReleaseWaiting: @Sendable () -> Void
    private var page: (any OCRPageExecuting)?
    private var bundle: URL?
    private var busy = false
    private var releasing = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    init(loader: @escaping Loader = CoreAIOCRPage.load, onReleaseWaiting: @escaping @Sendable () -> Void = {}) {
        self.loader = loader
        self.onReleaseWaiting = onReleaseWaiting
    }

    func image(at url: URL) throws -> OCRInput {
        try begin()
        defer { finish() }
        guard url.isFileURL else { throw OCRRuntimeError.invalidRequest("Choose an image file.") }
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        let data = try Data(contentsOf: url)
        try Task.checkCancellation()
        let image = try OCRImage(data: data)
        let preview = try Self.preview(image)
        try Task.checkCancellation()
        return OCRInput(
            name: url.lastPathComponent, image: image, preview: preview,
            sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
    }

    func load(bundle url: URL) async throws {
        try begin()
        defer { finish() }
        let url = url.standardizedFileURL
        if page != nil {
            guard bundle == url else { throw OCRRuntimeError.invalidRequest("Stop before changing the model.") }
            return
        }
        let loaded = try await loader(url)
        try Task.checkCancellation()
        page = loaded
        bundle = url
    }

    func read(_ input: OCRInput) async throws -> OCRDocument {
        try begin()
        defer { finish() }
        guard let page else { throw OCRRuntimeError.invalidRequest("Load the model before reading an image.") }
        let document = try await page.read(input)
        try Task.checkCancellation()
        return document
    }

    func release() async {
        if busy {
            releasing = true
            onReleaseWaiting()
            await withCheckedContinuation { waiters.append($0) }
        } else {
            page = nil
            bundle = nil
        }
    }

    private func begin() throws {
        try Task.checkCancellation()
        guard !busy && !releasing else { throw OCRRuntimeError.busy }
        busy = true
    }

    private func finish() {
        busy = false
        if releasing {
            page = nil
            bundle = nil
            releasing = false
            let pending = waiters
            waiters.removeAll()
            for continuation in pending { continuation.resume() }
        }
    }

    private static func preview(_ image: OCRImage) throws -> Data {
        let scale = min(1, 800 / Double(max(image.width, image.height)))
        let thumbnail = try image.resized(
            width: Int(ceil(Double(image.width) * scale)), height: Int(ceil(Double(image.height) * scale)),
            resampling: .pillow)
        guard let provider = CGDataProvider(data: Data(thumbnail.rgb) as CFData),
            let raster = CGImage(
                width: thumbnail.width, height: thumbnail.height, bitsPerComponent: 8, bitsPerPixel: 24,
                bytesPerRow: thumbnail.width * 3, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue), provider: provider,
                decode: nil, shouldInterpolate: true, intent: .defaultIntent)
        else { throw OCRError.invalidImage }
        let encoded = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(encoded, UTType.png.identifier as CFString, 1, nil)
        else {
            throw OCRError.invalidImage
        }
        CGImageDestinationAddImage(destination, raster, nil)
        guard CGImageDestinationFinalize(destination) else { throw OCRError.invalidImage }
        return encoded as Data
    }
}

private struct CoreAIOCRPage: OCRPageExecuting {
    let runtime: OCRRuntime
    let bundle: String

    @concurrent static func load(_ bundle: URL) async throws -> any OCRPageExecuting {
        CoreAIOCRPage(runtime: try await OCRRuntime.load(bundle: bundle), bundle: bundle.lastPathComponent)
    }

    func read(_ input: OCRInput) async throws -> OCRDocument {
        let page = try await runtime.read(input.image)
        let layout = OCRDocument.Layout(
            parsed: page.layoutParsed, text: page.layout.text, skippedBoxes: page.skippedBoxes,
            seconds: Self.seconds(page.layout), tokens: Self.tokens(page.layout), finishedBy: Self.finish(page.layout))
        let blocks = page.blocks.enumerated().map {
            OCRDocument.Block(
                index: $0.offset, layout: $0.element.block.layout, box: $0.element.block.box,
                text: $0.element.recognition.text, seconds: Self.seconds($0.element.recognition),
                tokens: Self.tokens($0.element.recognition), finishedBy: Self.finish($0.element.recognition))
        }
        return OCRDocument(
            bundle: bundle, input: input, layout: layout, blocks: blocks, generateS: page.totalSeconds,
            loadS: runtime.loadStats.totalSeconds)
    }

    private static func seconds(_ result: OCRRecognition) -> OCRDocument.Seconds {
        OCRDocument.Seconds(
            vision: result.visionSeconds, firstToken: result.firstTokenSeconds, total: result.totalSeconds)
    }

    private static func tokens(_ result: OCRRecognition) -> OCRDocument.Tokens {
        OCRDocument.Tokens(prompt: result.promptTokens, image: result.imageTokens, output: result.tokens.count)
    }

    private static func finish(_ result: OCRRecognition) -> String {
        switch result.finish {
        case .stopToken: "eos"
        case .maxNewTokens: "max_new_tokens"
        }
    }
}
