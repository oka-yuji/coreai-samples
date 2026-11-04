import CoreAIBackend
import Foundation

@MainActor
final class ChatModelLoader {
    private let managed: Bool
    private var generation = 0
    private var loading = false
    private var releasing = 0
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private var relinquish: (@MainActor () async -> Void)?

    init(managed: Bool) { self.managed = managed }

    func load(_ url: URL, contextLimit: Int) async throws -> LocalLanguageModel {
        guard !loading, releasing == 0 else { throw CancellationError() }
        loading = true
        let ticket = generation
        defer {
            loading = false
            let pending = waiters
            waiters = []
            for waiter in pending { waiter.resume() }
        }
        let previous = relinquish
        relinquish = nil
        await previous?()
        try Task.checkCancellation()
        _ = try BundleSupport.supportedBundle(at: url)
        guard ticket == generation else { throw CancellationError() }
        let release: @MainActor () async -> Void
        if managed {
            release = try await MacModelLibraryServices.acquire(url.path, modelID: "qwen3.8-27b")
        } else {
            release = {}
        }
        do {
            try Task.checkCancellation()
            guard ticket == generation else { throw CancellationError() }
            let model = try await LocalLanguageModel(
                bundle: url, contextLimit: contextLimit, overflow: .dropOldToolOutputs)
            try Task.checkCancellation()
            guard ticket == generation else { throw CancellationError() }
            relinquish = release
            return model
        } catch {
            await release()
            throw error
        }
    }

    func release() async {
        generation += 1
        releasing += 1
        defer { releasing -= 1 }
        if loading { await withCheckedContinuation { waiters.append($0) } }
        let previous = relinquish
        relinquish = nil
        await previous?()
    }
}
