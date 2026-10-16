import CoreAIBackend
import Foundation
import Observation
import RAGContext

struct GenerationStats: Encodable, Sendable {
    var turn = 0
    var answer = ""
    var inputTokens: Int?
    var cachedTokens: Int?
    var outputTokens = 0
    var ttftSeconds: Double?
    var tokensPerSecond: Double?
    var stopReason = ""
    var error: String?
    var sources: Int?
    var sourceTokens: Int?
    var ragPromptTokens: Int?
    var retrievalSeconds: Double?

    var statusLine: String {
        if let error { return "Error: \(error)" }
        var parts: [String] = []
        if let ttftSeconds { parts.append(String(format: "TTFT %.2f s", ttftSeconds)) }
        if let tokensPerSecond { parts.append(String(format: "%.1f tok/s", tokensPerSecond)) }
        if let inputTokens { parts.append("input \(inputTokens)") }
        if let cachedTokens { parts.append("reused \(cachedTokens)") }
        parts.append("output \(outputTokens)")
        parts.append(stopReason == "cancelled" ? "stopped" : stopReason)
        if let sources { parts.append("sources \(sources)") }
        if let sourceTokens { parts.append("source tok \(sourceTokens)") }
        if let ragPromptTokens { parts.append("prompt(rag) \(ragPromptTokens) tok") }
        if let retrievalSeconds { parts.append(String(format: "retrieval %.1fs", retrievalSeconds)) }
        return parts.joined(separator: "  |  ")
    }
}

@MainActor
@Observable
final class ChatViewModel {
    enum Phase: Equatable {
        case empty
        case loading(preparing: Bool)
        case ready
        case failed(String)
    }

    static let defaultContextLimit = 16_384

    private(set) var phase = Phase.empty
    private(set) var bundleURL: URL?
    private(set) var modelName = ""
    private(set) var loadNote = ""
    private(set) var loadStarted = Date.now
    private(set) var loadSeconds = 0.0
    var isResponding: Bool { externallyBusy }
    var input = ""
    var externallyBusy = false
    var ragEnabled = false
    var ragWeb = true
    var ragLocal = true
    var ragBudget = ChatViewModel.launchRAGBudget ?? RAGContext.sourceTokenBudget
    let contextLimit = LaunchArguments.value("--context-limit").flatMap(Int.init) ?? ChatViewModel.defaultContextLimit

    @ObservationIgnored private(set) var model: LocalLanguageModel?
    nonisolated static var launchRAGBudget: Int? {
        LaunchArguments.value("--rag-budget").flatMap(Int.init).flatMap {
            RAGSettings.budgetChoices.contains($0) ? $0 : nil
        }
    }

    var isLoading: Bool { if case .loading = phase { true } else { false } }
    var trimmedInput: String { input.trimmingCharacters(in: .whitespacesAndNewlines) }
    var canSend: Bool { phase == .ready && !isResponding && !externallyBusy && !trimmedInput.isEmpty }
    var loadStats: Engine.LoadStats? { model?.loadStats }
    var ragIsOn: Bool { ragEnabled && (ragWeb || ragLocal) }

    func isUsing(_ url: URL) -> Bool {
        phase != .empty && bundleURL.map(ModelStorage.key) == ModelStorage.key(url)
    }

    func state(of url: URL) -> String? {
        guard isUsing(url) else { return nil }
        switch phase {
        case .ready: return "Loaded"
        case .loading(let preparing): return preparing ? "Preparing…" : "Loading…"
        default: return nil
        }
    }

    func load(_ url: URL) async {
        guard !isLoading, !isResponding, !externallyBusy else { return }
        let key = ModelStorage.key(url)
        model = nil
        bundleURL = url
        modelName = (try? ModelBundle(url: url).name) ?? url.lastPathComponent
        loadStarted = .now
        phase = .loading(preparing: !ModelStorage.prepared.contains(key))
        let clock = ContinuousClock()
        let start = clock.now
        do {
            _ = try BundleSupport.supportedBundle(at: url)
            let model = try await LocalLanguageModel(
                bundle: url, contextLimit: contextLimit, overflow: .dropOldToolOutputs)
            loadSeconds = (clock.now - start) / .seconds(1)
            self.model = model
            ModelStorage.prepared.insert(key)
            loadNote =
                model.loadStats.cached
                ? String(format: "Loaded in %.1f s", loadSeconds)
                : String(
                    format: "Prepared for this Mac in %.1f s, loaded in %.1f s", model.loadStats.specializeSeconds,
                    loadSeconds)
            phase = .ready
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

}
