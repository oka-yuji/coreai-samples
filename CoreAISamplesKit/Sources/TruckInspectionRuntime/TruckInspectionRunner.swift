import Foundation
import TruckInspectionCore
import TruckInspectionFeature

@MainActor
public protocol TruckInspectionModelService {
    func prepare(folder: URL) async throws -> InspectionModelInfo
    func classify(text: String, item: InspectionItem, catalog: InspectionCatalog) async throws -> InspectionCandidate
    func waitForIdle() async
    func unload() throws
}

public struct InspectionReleaseError: LocalizedError {
    public let reason: String

    public var errorDescription: String? {
        "The inspection demo could not stop its audio or release its model (\(reason)). Try stopping again."
    }
}

enum InspectionGateError: Error {
    case notPermitted
}

@MainActor
public final class TruckInspectionRunner {
    private let model: any TruckInspectionModelService
    private let speech: (any TruckInspectionSpeechService)?
    private let permits: @MainActor () -> Bool
    private let clock = ContinuousClock()
    private let started = ContinuousClock.now
    private var operations = 0
    private var waiting: [CheckedContinuation<Void, Never>] = []

    public init(
        model: any TruckInspectionModelService, speech: (any TruckInspectionSpeechService)? = nil,
        permits: @escaping @MainActor () -> Bool = { true }
    ) {
        self.model = model
        self.speech = speech
        self.permits = permits
    }

    public func release() async throws {
        do {
            try await drain()
            try model.unload()
        } catch {
            throw InspectionReleaseError(reason: String(describing: error))
        }
    }

    public func run(_ effect: InspectionEffect, send: (InspectionAction) -> Void) async {
        if case .stop(let id, let unload) = effect { return await stop(id, unload: unload, send: send) }
        operations += 1
        defer { finish() }
        let id: Int
        switch effect {
        case .prepare(let value, _), .classify(let value, _, _, _), .stop(let value, _), .voice(let value, _),
            .measure(let value, _, _, _):
            id = value
        }
        stamp(id, send: send)
        do {
            switch effect {
            case .prepare(let id, let folder):
                send(.diagnostic(id, "environment: \(InspectionMemory.environment())"))
                send(.diagnostic(id, "memory-before-load: \(InspectionMemory.snapshot())"))
                try admit()
                let info = try await model.prepare(folder: folder)
                try Task.checkCancellation()
                send(.diagnostic(id, "memory-after-load: \(InspectionMemory.snapshot())"))
                send(.modelLoaded(id, info))
            case .classify(let id, let text, let item, let catalog):
                try admit()
                let start = clock.now
                let candidate = try await model.classify(text: text, item: item, catalog: catalog)
                try Task.checkCancellation()
                stamp(id, send: send)
                send(.diagnostic(id, "memory-after-classify: \(InspectionMemory.snapshot())"))
                send(.classified(id, candidate, (clock.now - start) / .milliseconds(1)))
            case .measure(let id, let text, let item, let catalog):
                try await measure(text: text, item: item, catalog: catalog, id: id, send: send)
            case .voice(let id, let voice):
                try await runVoice(voice, id: id, send: send)
            case .stop: break
            }
        } catch {
            guard !Task.isCancelled, !(error is CancellationError) else { return }
            if case .voice = effect {
                send(.voice(.failed(id, String(describing: error))))
            } else {
                send(.failed(id, String(describing: error)))
            }
        }
    }

    private func runVoice(_ effect: InspectionVoiceEffect, id: Int, send: (InspectionAction) -> Void) async throws {
        guard let speech else { throw InspectionSpeechError.unavailable }
        try admit()
        switch effect {
        case .prepare:
            let diagnostics: String
            do {
                diagnostics = try await speech.prepare()
            } catch let error as InspectionSpeechError {
                guard let limit = error.limit else { throw error }
                try Task.checkCancellation()
                send(.voice(.limited(id, limit)))
                return
            }
            try Task.checkCancellation()
            send(.voice(.prepared(id, diagnostics)))
        case .speak(let text):
            let start = clock.now
            do {
                try await speech.speak(text)
                try Task.checkCancellation()
                stamp(id, send: send)
                send(.diagnostic(id, "timing: speak=\((clock.now - start) / .seconds(1))s"))
                send(.voice(.spoken(id, timedOut: false)))
            } catch InspectionSpeechError.playbackTimedOut {
                try Task.checkCancellation()
                send(.diagnostic(id, "tts: deadline exceeded"))
                send(.voice(.spoken(id, timedOut: true)))
            }
        case .listen(let hints):
            let start = clock.now
            let text = try await listen(speech, hints: hints, id: id, send: send)
            try Task.checkCancellation()
            stamp(id, send: send)
            send(.diagnostic(id, "timing: listen=\((clock.now - start) / .seconds(1))s"))
            send(.voice(.heard(id, text)))
        }
    }

    private func listen(
        _ speech: any TruckInspectionSpeechService, hints: [String], id: Int, send: (InspectionAction) -> Void
    ) async throws -> String {
        let events = try await speech.startListening(hints: hints)
        var window = InspectionListeningWindow()
        var resolved: String?
        var levels: [Double] = []
        var quietSince: Double?
        do {
            listening: for await event in events {
                try Task.checkCancellation()
                switch event {
                case .partial(let text, let elapsed):
                    window.partial(text, elapsed: elapsed)
                    send(.voice(.partial(id, window.lastPartial)))
                case .route(let text): send(.diagnostic(id, "route-listen: \(text)"))
                case .final(let text):
                    send(.diagnostic(id, "raw-final: \(text)"))
                    resolved = InspectionRules.resolveFinal(text, lastPartial: window.lastPartial)
                    break listening
                case .tick(let elapsed):
                    switch window.tick(elapsed: elapsed) {
                    case .keepListening: break
                    case .endAudio: await speech.endAudio()
                    case .resolve:
                        resolved = window.lastPartial
                        send(.diagnostic(id, "stt: final deadline; using last partial"))
                        break listening
                    }
                case .level(let level, let elapsed):
                    levels.append(level)
                    if levels.count > 32 { levels.removeFirst() }
                    send(.diagnostic(id, "rms: \(level)dBFS"))
                    if level < -70 {
                        if quietSince == nil { quietSince = elapsed }
                        if let quietSince, elapsed - quietSince >= 2 {
                            send(.diagnostic(id, "audio: very low input level; check microphone route"))
                        }
                    } else {
                        quietSince = nil
                    }
                case .failure(let error): throw InspectionSpeechError.recognition(error)
                }
            }
        } catch {
            await speech.finishListening()
            throw error
        }
        await speech.finishListening()
        try Task.checkCancellation()
        if let lower = levels.min(), let upper = levels.max() {
            send(.diagnostic(id, "rms-range: \(lower)...\(upper)dBFS"))
        }
        return resolved ?? window.lastPartial
    }

    private func measure(
        text: String, item: InspectionItem, catalog: InspectionCatalog, id: Int, send: (InspectionAction) -> Void
    ) async throws {
        send(.diagnostic(id, "benchmark-environment: \(InspectionMemory.environment()) item=\(item.id) text=\(text)"))
        var times: [Double] = []
        for index in 0...20 {
            try Task.checkCancellation()
            try admit()
            let start = clock.now
            let candidate = try await model.classify(text: text, item: item, catalog: catalog)
            try Task.checkCancellation()
            let elapsed = (clock.now - start) / .milliseconds(1)
            times.append(elapsed)
            send(
                .diagnostic(
                    id,
                    "benchmark-sample: index=\(index) ms=\(elapsed) result=\(candidate.status.rawValue) \(InspectionMemory.snapshot())"
                ))
        }
        guard let first = times.first,
            let result = InspectionMeasurement(warmupMilliseconds: first, samples: Array(times.dropFirst()))
        else {
            throw InspectionError.invalidClassification
        }
        send(.measured(id, result))
    }

    private func stop(_ id: Int, unload: Bool, send: (InspectionAction) -> Void) async {
        do {
            if unload { try await release() } else { try await drain() }
        } catch {
            guard !Task.isCancelled else { return }
            send(.stopFailed(id, String(describing: error)))
            return
        }
        guard !Task.isCancelled else { return }
        send(.stopped(id, unloaded: unload))
    }

    private func drain() async throws {
        var failure: (any Error)?
        do { try await speech?.stop() } catch { failure = error }
        await model.waitForIdle()
        if operations > 0 { await withCheckedContinuation { waiting.append($0) } }
        if let failure { throw failure }
    }

    private func admit() throws {
        guard permits() else { throw InspectionGateError.notPermitted }
    }

    private func stamp(_ id: Int, send: (InspectionAction) -> Void) {
        send(.voice(.clock(id, (clock.now - started) / .seconds(1))))
    }

    private func finish() {
        operations -= 1
        guard operations == 0 else { return }
        let continuations = waiting
        waiting.removeAll()
        for continuation in continuations { continuation.resume() }
    }
}
