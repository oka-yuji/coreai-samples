import CoreAI
import ExtractFeature
import ExtractRuntime
import Foundation
import GLiNERCore
import GLiNERRuntime

enum ComputeUnits: String {
    case gpu
    case automatic = "default"

    static var launchValue: ComputeUnits? {
        guard let value = LaunchArguments.value("--compute-units") else { return .gpu }
        return ComputeUnits(rawValue: value)
    }

    var options: SpecializationOptions {
        switch self {
        case .gpu: SpecializationOptions(preferredComputeUnitKind: .gpu)
        case .automatic: .default
        }
    }
}

enum ExtractServiceError: LocalizedError {
    case noAsset
    case noTokenizer(String)

    var errorDescription: String? {
        switch self {
        case .noAsset: "Open an asset folder first."
        case .noTokenizer(let path): "No tokenizer.json in \(path) or in its tokenizer folder."
        }
    }
}

@MainActor
final class GLiNERExtractService: ExtractServicing {
    private let correctionWords: [String]
    private let computeUnits: ComputeUnits
    private var extractor: GLiNERExtractor?

    init(correctionWords: [String], computeUnits: ComputeUnits = .gpu) {
        self.correctionWords = correctionWords
        self.computeUnits = computeUnits
    }

    func open(folder: URL) async throws -> AssetInfo {
        let clock = ContinuousClock()
        let started = clock.now
        let (extractor, revision) = try await Self.load(folder, options: computeUnits.options)

        try Task.checkCancellation()
        self.extractor = extractor
        return AssetInfo(
            path: folder.path(percentEncoded: false), revision: revision,
            loadSeconds: (clock.now - started) / .seconds(1),
            floatType: extractor.model.floatType == .float16 ? "float16" : "float32")
    }

    func extract(text: String, schema: FormSchema, filter: CandidateFilter) async throws -> (
        mapping: FormMapping, timing: ExtractTiming
    ) {
        guard let extractor else { throw ExtractServiceError.noAsset }
        let clock = ContinuousClock()
        let started = clock.now
        let labels: [EntityLabel] = if case .entities(let labels) = schema.extractionSchema() { labels } else { [] }
        let (extraction, times) = try await extractor.timedExtract(text: text, labels: labels)
        let mapper = FormMapper(schema: schema, correctionWords: correctionWords, candidateFilter: filter)
        let requests = mapper.classificationRequests(text: text, extraction: extraction)
        let classifyStarted = clock.now
        var results: [ClassificationRequest: ClassificationResult] = [:]
        for request in requests {
            try Task.checkCancellation()
            guard case .classification(let task, let labels) = request.schema else { continue }
            results[request] = try await extractor.classify(text: request.clause, task: task, labels: labels)
        }
        let mapStarted = clock.now
        let mapping = mapper.map(text: text, extraction: extraction) { clause, schema in
            results[ClassificationRequest(clause: clause, schema: schema)] ?? ClassificationResult(probabilities: [])
        }
        let ended = clock.now
        func milliseconds(_ duration: Duration) -> Double { duration / .milliseconds(1) }
        let timing = ExtractTiming(
            preprocess: milliseconds(times.preprocess), encode: milliseconds(times.encode),
            pair: milliseconds(times.pair), classify: milliseconds(mapStarted - classifyStarted),
            classifications: requests.count, map: milliseconds(ended - mapStarted), total: milliseconds(ended - started)
        )
        return (mapping, timing)
    }

    func release() {
        extractor = nil
    }

    @concurrent
    private static func load(_ folder: URL, options: SpecializationOptions) async throws -> (GLiNERExtractor, String?) {
        let asset = try GLiNERAsset(directory: folder)
        let nested = folder.appending(path: "tokenizer", directoryHint: .isDirectory)
        let tokenizerFolder = [nested, folder].first {
            FileManager.default.fileExists(atPath: $0.appending(path: "tokenizer.json").path(percentEncoded: false))
        }
        guard let tokenizerFolder else { throw ExtractServiceError.noTokenizer(folder.path(percentEncoded: false)) }
        let tokenizer = try GLiNERTokenizer(directory: tokenizerFolder)
        let model = try await GLiNERModel(asset: asset, options: options)
        let contract = try? JSONDecoder().decode(
            ContractRevision.self, from: Data(contentsOf: folder.appending(path: "contract.json")))
        return (
            GLiNERExtractor(preprocessor: GLiNERPreprocessor(tokenizer: tokenizer), model: model), contract?.revision
        )
    }

    private struct ContractRevision: Decodable {
        let revision: String?
    }
}
