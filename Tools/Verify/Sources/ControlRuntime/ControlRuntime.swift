import CoreAILanguageModels
import Foundation

public enum Engine: String, CaseIterable, Sendable {
    case `default`
    case sequential
}

public enum ControlRuntime {
    public static func makeModel(bundle: URL, engine: Engine) async throws -> CoreAILanguageModel {
        try await CoreAILanguageModel(
            resourcesAt: bundle,
            variant: engine == .sequential ? "coreai-sequential" : nil
        )
    }
}
