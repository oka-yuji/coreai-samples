import EmbeddingCore
import EmbeddingRuntime
import Foundation
import SpokenSearchFeature

actor SpokenTopicIndex {
    private let folder: URL
    private let makeModel: @Sendable () -> any SpokenTextModel
    private var cached: (generation: Int, entries: [SpokenIndexedPassage])?
    private var minimumGeneration = 0
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    init(folder: URL, makeModel: @escaping @Sendable () -> any SpokenTextModel = { EmbeddingModel() }) {
        self.folder = folder
        self.makeModel = makeModel
    }

    func invalidate(before generation: Int) {
        minimumGeneration = max(minimumGeneration, generation)
        if let cached, cached.generation < minimumGeneration { self.cached = nil }
    }

    func search(_ query: String, transcript: SpokenTranscript, generation: Int) async throws -> [SpokenHit] {
        while busy {
            try check(generation)
            await withCheckedContinuation { waiters.append($0) }
        }
        try check(generation)
        busy = true
        defer {
            busy = false
            let pending = waiters
            waiters.removeAll()
            for waiter in pending { waiter.resume() }
        }
        let model = makeModel()
        try await model.load(directory: folder)
        let hits: [SpokenHit]
        do {
            try check(generation)
            let vector = try await model.encode(query, role: .query)
            try check(generation)
            let entries = try await indexed(transcript, model: model, generation: generation)
            try check(generation)
            hits = entries.map { ($0.hit, $0.vector.similarity(to: vector)) }
                .sorted { left, right in
                    if left.1 != right.1 { return left.1 > right.1 }
                    if left.0.start != right.0.start { return left.0.start < right.0.start }
                    return left.0.id < right.0.id
                }.map(\.0)
        } catch {
            try? await model.unload()
            throw error
        }
        try await model.unload()
        try check(generation)
        return hits
    }

    private func indexed(_ transcript: SpokenTranscript, model: any SpokenTextModel, generation: Int) async throws
        -> [SpokenIndexedPassage]
    {
        if let cached, cached.generation == generation { return cached.entries }
        var entries: [SpokenIndexedPassage] = []
        var remaining = Array(SpokenPassage.read(transcript).reversed())
        while let passage = remaining.popLast() {
            try check(generation)
            guard let hit = passage.hit else { continue }
            do {
                let vector = try await model.encode(hit.text, role: .document)
                entries.append(SpokenIndexedPassage(hit: hit, vector: vector))
            } catch EmbeddingError.tooManyTokens {
                guard let (left, right) = passage.halves() else { throw EmbeddingError.invalidVector }
                remaining.append(right)
                remaining.append(left)
            }
        }
        try check(generation)
        cached = (generation, entries)
        return entries
    }

    private func check(_ generation: Int) throws {
        try Task.checkCancellation()
        if generation < minimumGeneration { throw CancellationError() }
    }
}
