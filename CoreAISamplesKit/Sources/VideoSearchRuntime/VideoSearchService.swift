import CoreGraphics
import EmbeddingCore
import EmbeddingRuntime
import Foundation
import ImageIO
import UniformTypeIdentifiers
import VideoSearchFeature

@MainActor
public protocol VideoSearchService {
    func load(_ folder: URL) async throws
    func index(_ video: URL, progress: (Int, Int) -> Void) async throws -> VideoIndex
    func search(_ query: String) async throws -> [VideoSearchMatch]
    func export(_ plan: VideoClipPlan, of video: URL, to destination: URL, progress: (Double) -> Void) async throws
    func waitForIdle() async
    func close() async throws
}

@MainActor
public final class LiveVideoSearchService: VideoSearchService {
    private let model = EmbeddingModel()
    private var indexed: [(moment: VideoMoment, vector: EmbeddingVector)] = []
    private var busy = false
    private var waiting: [CheckedContinuation<Void, Never>] = []

    public init() {}

    public func load(_ folder: URL) async throws {
        try begin()
        defer { finish() }
        indexed = []
        try await model.load(directory: folder)
    }

    public func index(_ video: URL, progress: (Int, Int) -> Void) async throws -> VideoIndex {
        try begin()
        defer { finish() }
        indexed = []
        let duration = try await VideoFrameSource.duration(of: video)
        let times = VideoSampling.times(duration: duration)
        let frames = VideoFrameSource.frames(of: video, at: times, maximumSize: CGSize(width: 640, height: 640))
        var prepared: [(moment: VideoMoment, vector: EmbeddingVector)] = []
        var completed = 0
        for try await frame in frames {
            try Task.checkCancellation()
            completed += 1
            if !prepared.contains(where: { $0.moment.time == frame.time }) {
                let vector = try await model.encode(image: frame.image)
                let moment = VideoMoment(time: frame.time, thumbnail: try Self.thumbnail(frame.image))
                try Task.checkCancellation()
                prepared.append((moment, vector))
            }
            progress(completed, times.count)
        }
        guard !prepared.isEmpty else { throw VideoSearchError.invalidVideo }
        indexed = prepared.sorted { $0.moment.time < $1.moment.time }
        return VideoIndex(duration: duration, moments: indexed.map(\.moment))
    }

    public func search(_ query: String) async throws -> [VideoSearchMatch] {
        try begin()
        defer { finish() }
        guard !indexed.isEmpty else { throw EmbeddingError.notLoaded }
        let vector = try await model.encode(query)
        try Task.checkCancellation()
        return indexed.map { VideoSearchMatch(moment: $0.moment, similarity: vector.similarity(to: $0.vector)) }
            .sorted {
                $0.similarity == $1.similarity ? $0.moment.time < $1.moment.time : $0.similarity > $1.similarity
            }
    }

    public func export(
        _ plan: VideoClipPlan, of video: URL, to destination: URL, progress: (Double) -> Void
    ) async throws {
        try begin()
        defer { finish() }
        let (values, continuation) = AsyncStream.makeStream(of: Double.self)
        try await withThrowingTaskGroup { group in
            group.addTask {
                defer { continuation.finish() }
                try await VideoClipExporter.export(plan, of: video, to: destination) { continuation.yield($0) }
            }
            for await value in values { progress(value) }
            try await group.waitForAll()
        }
    }

    public func waitForIdle() async {
        while busy { await withCheckedContinuation { waiting.append($0) } }
    }

    public func close() async throws {
        await waitForIdle()
        busy = true
        defer { finish() }
        indexed = []
        try await model.unload()
    }

    private func begin() throws {
        try Task.checkCancellation()
        guard !busy else { throw EmbeddingError.busy }
        busy = true
    }

    private func finish() {
        busy = false
        let continuations = waiting
        waiting = []
        for continuation in continuations { continuation.resume() }
    }

    private static func thumbnail(_ image: CGImage) throws -> Data {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil)
        else { throw VideoSearchError.invalidVideo }
        CGImageDestinationAddImage(destination, image, [kCGImageDestinationImageMaxPixelSize: 320] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { throw VideoSearchError.invalidVideo }
        return data as Data
    }
}
