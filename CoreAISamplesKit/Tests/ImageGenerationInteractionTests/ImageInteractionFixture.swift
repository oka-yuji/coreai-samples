import Foundation
import ImageGenerationCore
import ImageGenerationRuntime
import Testing

@MainActor
final class ImageInteractionService: ImageGenerationService {
    let events: AsyncStream<String>
    private let sink: AsyncStream<String>.Continuation
    var gate: AsyncStream<Void>?
    var failure = false
    var stopFailure = false
    var releases = 0
    var requests: [ImageGenerationRequest] = []
    private var running = false
    private var stopping = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    init() { (events, sink) = AsyncStream.makeStream() }

    func generate(
        _ request: ImageGenerationRequest, trace: (any ImageTraceSink)?,
        progress: @MainActor @Sendable (ImageExecutionProgress) -> Void
    ) async throws -> ImageGenerationResult {
        running = true
        stopping = false
        requests.append(request)
        defer {
            running = false
            let pending = waiters
            waiters.removeAll()
            for waiter in pending { waiter.resume() }
        }
        progress(ImageExecutionProgress(requestID: request.id, phase: .loading(.text)))
        sink.yield("generate")
        if let gate {
            await withTaskCancellationShield { for await _ in gate { break } }
        }
        progress(ImageExecutionProgress(requestID: request.id, phase: .running(.denoiser, step: 3)))
        if stopping { throw CancellationError() }
        if failure { throw ImageExecutionError.invalidResource }
        return try imageInteractionResult(request)
    }

    func stop() async throws {
        stopping = true
        sink.yield("stop")
        if stopFailure { throw ImageExecutionError.releaseUnconfirmed("fixture") }
        if running { await withCheckedContinuation { waiters.append($0) } }
    }

    func release() async throws {
        releases += 1
        try await stop()
    }

    func latestReport() async -> ImageExecutionReport? { nil }
}

func imageInteractionResult(_ request: ImageGenerationRequest) throws -> ImageGenerationResult {
    let pixels = try ImageGenerationPixels(
        width: 512, height: 512, rgba: (0..<(512 * 512 * 4)).map { UInt8(truncatingIfNeeded: $0 * 37) })
    var report = ImageExecutionReport(
        requestID: request.id, profileID: "fixture", manifestSHA256: String(repeating: "a", count: 64),
        noiseSource: ImageGenerationNoise.version, diagnosticsEnabled: false)
    report.status = .completed
    return ImageGenerationResult(requestID: request.id, pixels: pixels, report: report)
}

@MainActor
func imageInteractionEvent(_ wanted: String, from iterator: inout AsyncStream<String>.Iterator) async throws {
    while let event = await iterator.next(isolation: #isolation) {
        if event == wanted { return }
    }
    throw ImageExecutionError.invalidResource
}
