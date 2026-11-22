import ArgumentParser
@_spi(Verification) import CoreAIBackend
import CryptoKit
import Foundation

struct Evaluate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "evaluate",
        abstract:
            "Generate greedily from the token sequences of input records with the own engine, and write the output records and the run record of the quality contract."
    )

    @Option(help: "Model bundle folder.")
    var model: String

    @Option(help: "Input records, one JSON object per line with id, input_ids, input_sha256 and max_new_tokens.")
    var inputs: String

    @Option(help: "The contract.json of the quality contract.")
    var contract: String

    @Option(help: "Name of the evaluation set, such as primary.")
    var set: String

    @Option(help: "Role of this run, one of the keys of roles in the contract.")
    var role: String

    @Option(help: "Repository commit written to run.json. The command does not run git.")
    var commit: String

    @Option(help: "Run folder: a new folder inside an existing folder, or the folder of the same run to resume.")
    var output: String

    @Option(help: "After this many minutes, end once the item in progress is written.")
    var maxMinutes: Double?

    @Option(help: "Run only the first N input records.")
    var limit: Int?

    func validate() throws {
        if let limit, limit <= 0 { throw ValidationError("--limit takes a positive value.") }
        if let maxMinutes, !(maxMinutes > 0 && maxMinutes.isFinite) {
            throw ValidationError("--max-minutes takes a positive number of minutes.")
        }
    }

    func run() async throws {
        let started = ContinuousClock.now
        let contractData = try Data(contentsOf: URL(filePath: contract))
        let terms = try JSONDecoder.snake.decode(EvaluationContract.self, from: contractData)
        guard terms.roles[role] != nil else {
            throw ValidationError("--role must be one of \(terms.roles.keys.sorted().joined(separator: ", ")).")
        }
        let inputData = try Data(contentsOf: URL(filePath: inputs))
        let bundle = try ModelBundle(url: URL(filePath: model))
        guard let mainHash = bundle.mainHash else {
            throw ValidationError("The bundle has no main.hash, so the run record could not name the asset.")
        }
        guard bundle.vocabSize >= 2 else { throw ValidationError("language.vocab_size must be at least 2.") }
        let items = try EvaluationInput.read(inputData, vocabulary: bundle.vocabSize)
        let folder = URL(filePath: output)
        _ = try Evaluation.stored(in: folder)
        let engine = try await CoreAIBackend.Engine(bundle: bundle)
        let stats = engine.loadStats
        let record = EvaluationRun(
            contract: .init(version: terms.version, sha256: hex(Data(SHA256.hash(data: contractData)))), set: set,
            role: role, asset: .init(name: bundle.name, mainHash: mainHash, sourceModelId: bundle.sourceModelID),
            implementation: .init(commit: commit),
            callShape: .init(
                prefillChunk: engine.prefillChunkSize, prefillFunction: engine.hasPrefillFunction,
                expectFrequentReshapes: stats.expectsFrequentReshapes, requestedCompute: stats.requestedComputeUnit,
                contextLimit: engine.contextLimit),
            inputs: .init(sha256: hex(Data(SHA256.hash(data: inputData))), count: items.count), items: 0,
            started: .now,
            load: .init(cached: stats.cached, specializeS: stats.specializeSeconds, loadS: stats.totalSeconds))
        _ = try await Evaluation.run(
            items, limit: limit, in: folder, record: record, source: engine,
            stopTokens: Set(terms.generation.eosTokenId), started: started, maxMinutes: maxMinutes)
    }
}
