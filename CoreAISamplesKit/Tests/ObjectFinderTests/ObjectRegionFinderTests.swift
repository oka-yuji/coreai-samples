import CoreGraphics
import EmbeddingCore
import EmbeddingRuntime
import Foundation
import ImageIO
import ObjectFinderCore
import Testing
import UniformTypeIdentifiers

@testable import ObjectFinderRuntime

struct ObjectRegionFinderTests {
    private func image() throws -> CGImage {
        let pixels: [UInt8] = [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255]
        let provider = try #require(CGDataProvider(data: Data(pixels) as CFData))
        return try #require(
            CGImage(
                width: 2, height: 2, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 8,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    }

    @Test("In-memory image input preserves the existing file-based RGB and orientation contract")
    func imageInput() throws {
        let image = try image()
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appending(path: "image.png")
        let destination = try #require(
            CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        let memory = try EmbeddingImageReader.read(image)
        let file = try EmbeddingImageReader.read(url)
        #expect(memory.width == file.width && memory.height == file.height && memory.pixels == file.pixels)
        #expect(memory.pixels == [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
        let top = try #require(ObjectRegionProposer.crop(image, to: .init(x: 0, y: 0, width: 1, height: 0.5)))
        #expect(try EmbeddingImageReader.read(top).pixels == [255, 0, 0, 0, 255, 0])
    }

    @Test("The finder rejects missing preparation and returns only proposed regions")
    func preparationAndRegions() async throws {
        let vector = try EmbeddingVector([1] + Array(repeating: Float(0), count: 767))
        let region = ObjectRegion(x: 0, y: 0, width: 1, height: 0.5)
        let finder = ObjectRegionFinder(
            embedding: .init(load: { _ in }, text: { _ in vector }, image: { _ in vector }, close: {}),
            propose: { _ in [region] })
        let frame = try image()
        await #expect(throws: ObjectFinderError.notPrepared) { try await finder.candidates(in: frame) }
        await #expect(throws: ObjectFinderError.emptyQuery) {
            try await finder.prepare(folder: URL(filePath: "/unused"), query: "  ")
        }
        try await finder.prepare(folder: URL(filePath: "/unused"), query: "glasses")
        #expect(try await finder.candidates(in: frame) == [ObjectCandidate(id: 0, region: region, similarity: 1)])
        try await finder.close()
        await #expect(throws: ObjectFinderError.notPrepared) { try await finder.candidates(in: frame) }
    }

    @Test("A failed query clears preparation and releases the model")
    func preparationFailure() async throws {
        let fake = EncoderFake()
        let finder = ObjectRegionFinder(
            embedding: .init(
                load: { _ in }, text: { _ in throw ObjectFinderError.emptyQuery },
                image: { _ in throw ObjectFinderError.invalidImage }, close: { await fake.close() }),
            propose: { _ in [] })
        await #expect(throws: ObjectFinderError.emptyQuery) {
            try await finder.prepare(folder: URL(filePath: "/unused"), query: "glasses")
        }
        #expect(await fake.closed == 1)
    }

    @Test("Cancellation keeps the finder busy until an in-flight encoder returns", .timeLimit(.minutes(1)))
    func cancellationGate() async throws {
        let vector = try EmbeddingVector([1] + Array(repeating: Float(0), count: 767))
        let entered = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let gate = EncoderGate()
        let finder = ObjectRegionFinder(
            embedding: .init(
                load: { _ in }, text: { _ in vector },
                image: { _ in
                    await gate.wait(entered.continuation)
                    return vector
                }, close: {}),
            propose: { _ in [ObjectRegion(x: 0, y: 0, width: 1, height: 1)] })
        try await finder.prepare(folder: URL(filePath: "/unused"), query: "glasses")
        let frame = try image()
        let operation = Task { try await finder.candidates(in: frame) }
        var signal = entered.stream.makeAsyncIterator()
        _ = await signal.next()
        operation.cancel()
        await #expect(throws: ObjectFinderError.busy) { try await finder.close() }
        await #expect(throws: ObjectFinderError.busy) {
            try await finder.prepare(folder: URL(filePath: "/unused"), query: "keys")
        }
        await gate.open()
        await #expect(throws: CancellationError.self) { try await operation.value }
        try await finder.close()
    }
}

private actor EncoderFake {
    var closed = 0
    func close() { closed += 1 }
}

private actor EncoderGate {
    private var continuation: CheckedContinuation<Void, Never>?

    func wait(_ entered: AsyncStream<Void>.Continuation) async {
        await withCheckedContinuation {
            continuation = $0
            entered.yield(())
            entered.finish()
        }
    }

    func open() {
        continuation?.resume()
        continuation = nil
    }
}
