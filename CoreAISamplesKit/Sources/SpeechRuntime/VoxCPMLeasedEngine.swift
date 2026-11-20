#if os(macOS) || os(iOS)
    import Foundation
    import SpeechCore

    actor VoxCPMLeasedEngine: SpeechGenerating {
        private struct Opened {
            let directory: URL
            let registrationGeneration: Int
        }

        private let engine: any SpeechGenerating
        private let acquire:
            @Sendable (URL, Int?) async throws -> (
                registrationGeneration: Int, release: @Sendable () async -> Void
            )
        private let didRequestRelease: @Sendable () -> Void
        private var opened: Opened?
        private var held: (@Sendable () async -> Void)?
        private var busy = false
        private var closing = false
        private var operationWaiters: [CheckedContinuation<Void, Never>] = []
        private var releaseWaiters: [CheckedContinuation<Void, Never>] = []

        init(
            engine: any SpeechGenerating,
            acquire:
                @escaping @Sendable (URL, Int?) async throws -> (
                    registrationGeneration: Int, release: @Sendable () async -> Void
                ),
            didRequestRelease: @escaping @Sendable () -> Void = {}
        ) {
            self.engine = engine
            self.acquire = acquire
            self.didRequestRelease = didRequestRelease
        }

        func open(_ directory: URL) async throws -> SpeechModel {
            try begin()
            defer { finish() }
            opened = nil
            if let previous = held {
                held = nil
                await engine.release()
                await previous()
            }
            let lease = try await obtain(directory, expecting: nil)
            held = lease.release
            do {
                let model = try await engine.open(directory)
                try checkCurrent()
                opened = Opened(directory: directory, registrationGeneration: lease.registrationGeneration)
                return model
            } catch {
                if !closing, let release = held {
                    held = nil
                    await engine.release()
                    await release()
                }
                throw error
            }
        }

        func generate(text: String, seed: UInt64) async throws -> SpeechAudio {
            try await leased { try await $0.generate(text: text, seed: seed) }
        }

        func generate(text: String, seed: UInt64, reference: SpeechReferenceAudio?) async throws -> SpeechAudio {
            try await leased { try await $0.generate(text: text, seed: seed, reference: reference) }
        }

        func release() async {
            didRequestRelease()
            if closing {
                await withCheckedContinuation { releaseWaiters.append($0) }
                return
            }
            closing = true
            while busy { await withCheckedContinuation { operationWaiters.append($0) } }
            let release = held
            held = nil
            await engine.release()
            await release?()
            closing = false
            let waiting = releaseWaiters
            releaseWaiters.removeAll()
            for waiter in waiting { waiter.resume() }
        }

        private func leased(_ body: (any SpeechGenerating) async throws -> SpeechAudio) async throws -> SpeechAudio {
            try begin()
            defer { finish() }
            guard let opened else { throw SpeechError.modelNotOpened("VoxCPM2") }
            if held == nil {
                held = try await obtain(opened.directory, expecting: opened.registrationGeneration).release
            }
            return try await body(engine)
        }

        private func obtain(_ directory: URL, expecting expected: Int?) async throws -> (
            registrationGeneration: Int, release: @Sendable () async -> Void
        ) {
            let lease = try await acquire(directory, expected)
            guard !closing, !Task.isCancelled, expected == nil || lease.registrationGeneration == expected else {
                await lease.release()
                try checkCurrent()
                throw SpeechError.modelNotOpened("VoxCPM2")
            }
            return lease
        }

        private func checkCurrent() throws {
            try Task.checkCancellation()
            guard !closing else { throw SpeechError.modelNotOpened("VoxCPM2") }
        }

        private func begin() throws {
            guard !busy, !closing else { throw SpeechError.busy }
            try Task.checkCancellation()
            busy = true
        }

        private func finish() {
            busy = false
            let waiting = operationWaiters
            operationWaiters.removeAll()
            for waiter in waiting { waiter.resume() }
        }
    }
#endif
