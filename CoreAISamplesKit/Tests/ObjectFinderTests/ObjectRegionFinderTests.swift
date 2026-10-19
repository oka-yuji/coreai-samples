import CoreGraphics
import EmbeddingCore
import EmbeddingRuntime
import Foundation
import ImageIO
import ObjectFinderCore
import Synchronization
import Testing
import UniformTypeIdentifiers

@testable import ObjectFinderRuntime

private final class Locked<Value: Sendable>: Sendable {
    let value: Mutex<Value>

    init(_ value: Value) { self.value = Mutex(value) }
}

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

    private func patch() throws -> CGImage {
        var pixels = [UInt8](repeating: 0, count: 10 * 10 * 4)
        for y in 0..<10 {
            for x in 0..<10 {
                pixels[(y * 10 + x) * 4 + 3] = 255
                if x >= 6 && y >= 6 { pixels[(y * 10 + x) * 4] = 255 }
            }
        }
        let provider = try #require(CGDataProvider(data: Data(pixels) as CFData))
        return try #require(
            CGImage(
                width: 10, height: 10, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 40,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    }

    private static func redness(_ image: CGImage) throws -> EmbeddingVector {
        let pixels = try EmbeddingImageReader.read(image).pixels
        let count = pixels.count / 3
        let red = (0..<count).filter { pixels[$0 * 3] > 128 }.count
        let share = Float(red) / Float(count)
        let length = (share * share + (1 - share) * (1 - share)).squareRoot()
        return try EmbeddingVector([share / length, (1 - share) / length] + Array(repeating: Float(0), count: 766))
    }

    @Test("TWO-01 The standard search scores only the first three objectness regions")
    func standardSearch() async throws {
        let query = try EmbeddingVector([1] + Array(repeating: Float(0), count: 767))
        let regions = (0..<4).map { ObjectRegion(x: Double($0) * 0.2, y: 0, width: 0.2, height: 0.2) }
        let instanceCalls = Locked(0)
        let finder = ObjectRegionFinder(
            embedding: .init(load: { _ in }, text: { _ in query }, image: { _ in query }, close: {}),
            propose: { _ in regions },
            instances: { _ in
                instanceCalls.value.withLock { $0 += 1 }
                return [.frame]
            })
        try await finder.prepare(folder: URL(filePath: "/unused"), query: "glasses")
        let found = try await finder.candidates(in: try patch(), mode: .standard)
        #expect(found.map(\.region) == Array(regions.prefix(3)))
        #expect(instanceCalls.value.withLock { $0 } == 0)
    }

    @Test(
        "TWO-02 The two-step search scores five windows, proposes again only inside the best partial window, places those regions back in the frame and never returns a window"
    )
    func twoStepSearch() async throws {
        let query = try EmbeddingVector([1] + Array(repeating: Float(0), count: 767))
        let base = ObjectRegion(x: 0, y: 0, width: 0.3, height: 0.3)
        let baseInstance = ObjectRegion(x: 0.1, y: 0.5, width: 0.2, height: 0.2)
        let local = [
            ObjectRegion(x: 0.5, y: 0.5, width: 0.4, height: 0.4), ObjectRegion(x: 0, y: 0, width: 0.3, height: 0.3),
            ObjectRegion(x: 0.6, y: 0, width: 0.3, height: 0.3), ObjectRegion(x: 0, y: 0.6, width: 0.3, height: 0.3),
        ]
        let proposedWidths = Locked<[Int]>([])
        let embeddings = Locked(0)
        let finder = ObjectRegionFinder(
            embedding: .init(
                load: { _ in }, text: { _ in query },
                image: {
                    embeddings.value.withLock { $0 += 1 }
                    return try Self.redness($0)
                }, close: {}),
            propose: { image in
                proposedWidths.value.withLock { $0.append(image.width) }
                return image.width == 10 ? [base] : [local[0], local[1]]
            },
            instances: { image in image.width == 10 ? [base, baseInstance] : [local[2], local[3]] })
        try await finder.prepare(folder: URL(filePath: "/unused"), query: "glasses")
        let found = try await finder.candidates(in: try patch(), mode: .twoStep)
        let window = ObjectRegion.searchWindows[4]
        let placed = local.prefix(3).map { $0.placed(in: window) }
        #expect(found.map(\.region) == [base, baseInstance] + placed)
        #expect(abs(placed[0].x - 0.7) < 1e-9 && abs(placed[0].width - 0.24) < 1e-9)
        #expect(found[2].similarity == 1)
        #expect(proposedWidths.value.withLock { $0 } == [10, 6])
        #expect(embeddings.value.withLock { $0 } == ObjectRegion.searchWindows.count + found.count)
        #expect(!found.contains { ObjectRegion.searchWindows.contains($0.region) })

        let everywhere = ObjectRegionFinder(
            embedding: .init(load: { _ in }, text: { _ in query }, image: { _ in query }, close: {}),
            propose: { image in
                proposedWidths.value.withLock { $0.append(image.width) }
                return [base]
            })
        proposedWidths.value.withLock { $0 = [] }
        try await everywhere.prepare(folder: URL(filePath: "/unused"), query: "glasses")
        #expect(try await everywhere.candidates(in: try patch(), mode: .twoStep).map(\.region) == [base])
        #expect(proposedWidths.value.withLock { $0 } == [10])
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

    @Test(
        "Changing a query in the same prepared folder reuses one model load and produces the new query result, while close or a different folder requires loading again"
    )
    func modelReuse() async throws {
        let fake = try ReusedEncoder()
        let region = ObjectRegion(x: 0, y: 0, width: 1, height: 1)
        let finder = ObjectRegionFinder(
            embedding: .init(
                load: { await fake.load($0) }, text: { await fake.text($0) },
                image: { _ in fake.keys }, close: { await fake.close() }), propose: { _ in [region] })
        let frame = try image()
        let folder = URL(filePath: "/first")
        try await finder.prepare(folder: folder, query: "glasses")
        #expect(try await finder.candidates(in: frame).first?.similarity == 0)
        try await finder.prepare(folder: folder, query: "keys")
        #expect(try await finder.candidates(in: frame).first?.similarity == 1)
        #expect(await fake.loads == [folder])
        #expect(await fake.queries == ["glasses", "keys"])
        try await finder.close()
        try await finder.prepare(folder: folder, query: "keys")
        let second = URL(filePath: "/second")
        try await finder.prepare(folder: second, query: "glasses")
        #expect(await fake.loads == [folder, folder, second])
        try await finder.close()
        #expect(await fake.closeCancellation == [false, false])
    }

    @Test("An empty replacement query releases a prepared model and requires a fresh load")
    func emptyReplacementQuery() async throws {
        let fake = try ReusedEncoder()
        let finder = ObjectRegionFinder(
            embedding: .init(
                load: { await fake.load($0) }, text: { await fake.text($0) },
                image: { _ in fake.keys }, close: { await fake.close() }), propose: { _ in [] })
        let folder = URL(filePath: "/first")
        try await finder.prepare(folder: folder, query: "glasses")
        await #expect(throws: ObjectFinderError.emptyQuery) {
            try await finder.prepare(folder: folder, query: "  \n")
        }
        #expect(await fake.closeCancellation == [false])
        #expect(await fake.queries == ["glasses"])
        let frame = try image()
        await #expect(throws: ObjectFinderError.notPrepared) { try await finder.candidates(in: frame) }
        try await finder.prepare(folder: folder, query: "keys")
        #expect(await fake.loads == [folder, folder])
        try await finder.close()
    }

    @Test("Cancelling a reused text encoder clears the cache and shields model cleanup", .timeLimit(.minutes(1)))
    func cancelReusedQuery() async throws {
        let fake = try ReusedEncoder()
        let entered = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let gate = EncoderGate()
        let finder = ObjectRegionFinder(
            embedding: .init(
                load: { await fake.load($0) },
                text: { text in
                    if text == "keys" { await gate.wait(entered.continuation) }
                    return await fake.text(text)
                }, image: { _ in fake.keys }, close: { await fake.close() }), propose: { _ in [] })
        let folder = URL(filePath: "/first")
        try await finder.prepare(folder: folder, query: "glasses")
        let operation = Task { try await finder.prepare(folder: folder, query: "keys") }
        var started = entered.stream.makeAsyncIterator()
        _ = await started.next()
        operation.cancel()
        await #expect(throws: ObjectFinderError.busy) { try await finder.close() }
        await gate.open()
        await #expect(throws: CancellationError.self) { try await operation.value }
        #expect(await fake.closeCancellation == [false])
        try await finder.prepare(folder: folder, query: "glasses")
        #expect(await fake.loads == [folder, folder])
        try await finder.close()
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

actor ReusedEncoder {
    let keys: EmbeddingVector
    private let glasses: EmbeddingVector
    var loads: [URL] = []
    var queries: [String] = []
    var closeCancellation: [Bool] = []

    init() throws {
        keys = try EmbeddingVector([0, 1] + Array(repeating: Float(0), count: 766))
        glasses = try EmbeddingVector([1] + Array(repeating: Float(0), count: 767))
    }

    func load(_ folder: URL) { loads.append(folder) }
    func text(_ text: String) -> EmbeddingVector {
        queries.append(text)
        return text == "keys" ? keys : glasses
    }
    func close() { closeCancellation.append(Task.isCancelled) }
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
