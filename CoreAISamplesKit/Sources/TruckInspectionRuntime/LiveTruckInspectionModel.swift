import CoreAI
import Foundation
import GLiNERRuntime
import TruckInspectionCore
import TruckInspectionFeature

@MainActor
public final class LiveTruckInspectionModel: TruckInspectionModelService {
    private var extractor: GLiNERExtractor?
    private var busy = false
    private var waiting: [CheckedContinuation<Void, Never>] = []
    private let automaticPlacement: Bool

    public init(automaticPlacement: Bool = false) { self.automaticPlacement = automaticPlacement }

    public func prepare(folder: URL) async throws -> InspectionModelInfo {
        try begin()
        defer { finish() }
        extractor = nil
        let clock = ContinuousClock()
        let started = clock.now
        let options: SpecializationOptions = automaticPlacement ? .default : .init(preferredComputeUnitKind: .gpu)
        let (loaded, revision) = try await Self.load(folder, options: options)
        try Task.checkCancellation()
        extractor = loaded
        return InspectionModelInfo(
            folder: folder, revision: revision, milliseconds: (clock.now - started) / .milliseconds(1),
            computePreference: automaticPlacement ? "default" : "gpu",
            floatType: String(describing: loaded.model.floatType))
    }

    public func classify(
        text: String, item: InspectionItem, catalog: InspectionCatalog
    ) async throws -> InspectionCandidate {
        try begin()
        defer { finish() }
        guard let extractor else { throw InspectionError.noModel }
        let result = try await extractor.classify(
            text: text, task: item.modelTask, labels: catalog.choices.map(\.modelLabel))
        try Task.checkCancellation()
        return try InspectionRules.classify(result.probabilities, choices: catalog.choices)
    }

    public func waitForIdle() async {
        if busy { await withCheckedContinuation { waiting.append($0) } }
    }

    public func unload() throws {
        guard !busy else { throw InspectionError.busy }
        extractor = nil
    }

    private func begin() throws {
        try Task.checkCancellation()
        guard !busy else { throw InspectionError.busy }
        busy = true
    }

    private func finish() {
        busy = false
        let continuations = waiting
        waiting.removeAll()
        for continuation in continuations { continuation.resume() }
    }

    @concurrent
    private static func load(_ folder: URL, options: SpecializationOptions) async throws -> (GLiNERExtractor, String) {
        let asset = try GLiNERAsset(directory: folder)
        let candidates = [folder.appending(path: "tokenizer", directoryHint: .isDirectory), folder]
        guard
            let tokenizerFolder = candidates.first(where: {
                FileManager.default.fileExists(atPath: $0.appending(path: "tokenizer.json").path)
            })
        else { throw InspectionError.noModel }
        let tokenizer = try GLiNERTokenizer(directory: tokenizerFolder)
        let model = try await GLiNERModel(asset: asset, options: options)
        struct Revision: Decodable { let revision: String? }
        let contract = try JSONDecoder().decode(
            Revision.self, from: Data(contentsOf: folder.appending(path: "contract.json")))
        return (
            GLiNERExtractor(preprocessor: GLiNERPreprocessor(tokenizer: tokenizer), model: model),
            contract.revision ?? "unknown"
        )
    }
}
