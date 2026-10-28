import Darwin
import Foundation
import ImageDiagnosticFeature
import ImageGenerationCore
import ModelLibraryFeature

@MainActor
public final class ImageDiagnosticRunner {
    typealias Factory =
        @MainActor (ModelDemoSession, ImageMemoryBudget, URL, @escaping @MainActor @Sendable () -> Bool) ->
        any ImageGenerationService
    private let files: ImageDiagnosticFiles
    private let makeService: Factory
    private let environment: () throws -> (hardware: String, build: String)
    private var service: (any ImageGenerationService)?
    private var active = false
    private var revoked = false
    private var outcome: ImageDiagnosticOutcome?
    private var started: ContinuousClock.Instant?
    private var runtimeStarted: ContinuousClock.Instant?
    private var latestPhase: ImageExecutionPhase?
    private var stopRecord: ImageDiagnosticStopRecord?
    private var cleanupAttempts = 0
    private var waiters: [CheckedContinuation<Void, Never>] = []

    public convenience init() {
        self.init(files: ImageDiagnosticFiles(), environment: Self.identity) { session, budget, directory, permission in
            LiveImageGenerationService(
                session: session, budget: budget, diagnosticDirectory: directory, isPermitted: permission)
        }
    }

    init(
        files: ImageDiagnosticFiles, environment: @escaping () throws -> (hardware: String, build: String),
        makeService: @escaping Factory
    ) {
        self.files = files
        self.environment = environment
        self.makeService = makeService
    }

    public func run(_ effect: ImageDiagnosticEffect, send: (ImageDiagnosticAction) -> Void) async {
        switch effect {
        case .execute(let generation, let configuration):
            do {
                let result = try await execute(configuration) { plan in
                    send(.configured(generation, plan))
                } progress: {
                    phase in
                    if !Task.isCancelled { send(.progressed(generation, phase)) }
                }
                if !Task.isCancelled { send(.completed(generation, result)) }
            } catch {
                if !Task.isCancelled {
                    if let outcome {
                        send(.completed(generation, outcome))
                    } else {
                        send(.failed(generation, String(describing: error)))
                    }
                }
            }
        case .delayStop(let generation, let milliseconds):
            do {
                try await Task.sleep(for: .milliseconds(milliseconds))
                try Task.checkCancellation()
                send(.stopElapsed(generation))
            } catch {}
        case .stop(let generation):
            revoked = true
            if stopRecord == nil {
                stopRecord = ImageDiagnosticStopRecord(
                    phase: latestPhase.map { String(describing: $0) },
                    secondsSinceDiagnosticStart: started.map { (ContinuousClock.now - $0) / .seconds(1) },
                    secondsSinceRuntimeStart: runtimeStarted.map { (ContinuousClock.now - $0) / .seconds(1) })
            }
            var failure: (any Error)?
            do { try await service?.stop() } catch { failure = error }
            if active { await withCheckedContinuation { waiters.append($0) } }
            if let failure {
                send(.stopFailed(generation, String(describing: failure)))
            } else if outcome?.status == .releaseUnconfirmed {
                do {
                    guard let service, let previous = outcome else {
                        throw ImageExecutionError.releaseUnconfirmed("Cleanup owner is unavailable")
                    }
                    try await service.release()
                    self.service = nil
                    cleanupAttempts += 1
                    var message = "Cleanup completed after: " + previous.message
                    do {
                        try await files.recoveredCleanup(directory: previous.directory, attempt: cleanupAttempts)
                    } catch { message += "; cleanup record failed: " + String(describing: error) }
                    let recovered = ImageDiagnosticOutcome(
                        status: .failed, directory: previous.directory, message: message)
                    outcome = recovered
                    send(.stopped(generation, recovered))
                } catch { send(.stopFailed(generation, String(describing: error))) }
            } else {
                send(.stopped(generation, outcome))
            }
        }
    }

    private func execute(
        _ configuration: URL, configured: (ImageDiagnosticStop?) -> Void,
        progress: @MainActor @Sendable (ImageExecutionPhase) -> Void
    ) async throws -> ImageDiagnosticOutcome {
        guard !active, service == nil, !revoked else { throw ImageExecutionError.busy }
        active = true
        started = ContinuousClock.now
        defer {
            active = false
            let pending = waiters
            waiters.removeAll()
            for waiter in pending { waiter.resume() }
        }
        let context: ImageDiagnosticContext
        do {
            let identity = try environment()
            context = try await files.prepare(
                configuration, requestID: UUID(), hardware: identity.hardware, osBuild: identity.build)
        } catch {
            let failed = ImageDiagnosticOutcome(
                status: error is CancellationError ? .cancelled : .failed,
                directory: "", message: String(describing: error))
            outcome = failed
            return failed
        }
        configured(context.configuration.stop)
        let session = ModelDemoSession(
            id: LibraryTicket(owner: UUID(), number: 1),
            selection: DemoSelection(modelID: context.resource.modelID, demoID: "image-generation"),
            resource: context.resource)
        var result: ImageGenerationResult?
        var primary: (any Error)?
        var cleanup: (any Error)?
        do {
            try Task.checkCancellation()
            guard !revoked else { throw CancellationError() }
            let service = makeService(session, context.configuration.budget, context.directory) { [weak self] in
                self?.active == true && self?.revoked == false
            }
            self.service = service
            let trace =
                context.configuration.trace
                ? ImageDirectoryTrace(directory: context.directory.appending(path: "trace")) : nil
            runtimeStarted = ContinuousClock.now
            result = try await service.generate(context.request, trace: trace) {
                self.latestPhase = $0.phase
                progress($0.phase)
            }
        } catch { primary = error }
        do { try await service?.release() } catch { cleanup = error }
        let report = await service?.latestReport()
        let completed: ImageDiagnosticOutcome
        do {
            completed = try await files.finish(
                context, result: result, report: report,
                error: primary.map { String(describing: $0) },
                cleanupError: cleanup.map { String(describing: $0) },
                cancelled: primary is CancellationError, stop: stopRecord)
        } catch {
            let failure = ImageDiagnosticRecordingFailure(
                primary: primary.map { String(describing: $0) },
                cleanup: cleanup.map { String(describing: $0) },
                recording: String(describing: error))
            outcome = ImageDiagnosticOutcome(
                status: cleanup == nil ? .failed : .releaseUnconfirmed,
                directory: context.directory.path, message: String(describing: failure))
            if cleanup == nil { service = nil }
            throw failure
        }
        outcome = completed
        if cleanup == nil { service = nil }
        return completed
    }

    private static func identity() throws -> (hardware: String, build: String) {
        #if os(iOS)
            let hardware = systemValue("hw.machine")
        #else
            let hardware = systemValue("hw.model")
        #endif
        guard let hardware, let build = systemValue("kern.osversion") else { throw ImageExecutionError.invalidResource }
        return (hardware, build)
    }

    private static func systemValue(_ name: String) -> String? {
        var count = 0
        guard sysctlbyname(name, nil, &count, nil, 0) == 0, (1...4_096).contains(count) else { return nil }
        var bytes = [UInt8](repeating: 0, count: count)
        guard bytes.withUnsafeMutableBytes({ sysctlbyname(name, $0.baseAddress, &count, nil, 0) }) == 0 else {
            return nil
        }
        return String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
    }
}

struct ImageDiagnosticRecordingFailure: Error {
    let primary: String?
    let cleanup: String?
    let recording: String
}
