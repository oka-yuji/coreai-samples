import Foundation
import SpeechCore

@MainActor
final class ManagedSpeechService: SpeechService {
    private let service: any SpeechService
    private let directory: URL
    private let isPermitted: @MainActor () -> Bool
    private var active = true
    private var opened = false
    private var stopping = false
    private var operations = 0
    private var operationWaiters: [CheckedContinuation<Void, Never>] = []
    private var stopWaiters: [CheckedContinuation<Void, Never>] = []

    init(service: any SpeechService, directory: URL, isPermitted: @escaping @MainActor () -> Bool) {
        self.service = service
        self.directory = directory
        self.isPermitted = isPermitted
    }

    func deactivate() { active = false }

    func open(path: String) async throws -> SpeechModel {
        try await perform {
            guard path == directory.path else { throw SpeechError.noModel }
            let model = try await service.open(path: path)
            try checkPermission()
            opened = true
            return model
        }
    }

    func generate(text: String, seed: UInt64) async throws -> SpeechAudio {
        try await generate(text: text, seed: seed, reference: nil)
    }

    func generate(text: String, seed: UInt64, reference: SpeechReferenceAudio?) async throws -> SpeechAudio {
        try await perform {
            if !opened {
                _ = try await service.open(path: directory.path)
                try checkPermission()
                opened = true
            }
            return try await service.generate(text: text, seed: seed, reference: reference)
        }
    }

    func play(_ data: Data) async throws {
        try await perform { try await service.play(data) }
    }

    func stop() async {
        if stopping {
            await withCheckedContinuation { stopWaiters.append($0) }
            return
        }
        stopping = true
        let hadOperations = operations > 0
        await withTaskCancellationShield {
            await service.stop()
            if operations > 0 { await withCheckedContinuation { operationWaiters.append($0) } }
            if hadOperations { await service.stop() }
        }
        opened = false
        stopping = false
        let pending = stopWaiters
        stopWaiters.removeAll()
        for waiter in pending { waiter.resume() }
    }

    private func perform<Value>(_ body: () async throws -> Value) async throws -> Value {
        try checkPermission()
        operations += 1
        defer {
            operations -= 1
            if operations == 0 {
                let pending = operationWaiters
                operationWaiters.removeAll()
                for waiter in pending { waiter.resume() }
            }
        }
        do {
            let value = try await body()
            try checkPermission()
            return value
        } catch {
            try checkPermission()
            throw error
        }
    }

    private func checkPermission() throws {
        try Task.checkCancellation()
        guard active, !stopping, isPermitted() else { throw CancellationError() }
    }
}
