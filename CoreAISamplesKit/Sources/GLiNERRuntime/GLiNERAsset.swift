import Foundation
import GLiNERCore

public enum GLiNERAssetError: Error, Equatable {

    case modelCount(Int)

    case contractMismatch(String)
}

public struct GLiNERAsset: Sendable {
    public let modelURL: URL

    public init(directory: URL) throws {

        let directory = directory.resolvingSymlinksInPath()
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let contract = try decoder.decode(
            Contract.self, from: Data(contentsOf: directory.appending(path: "contract.json")))
        let expected = Contract.implemented
        guard contract.functions == expected.functions else { throw GLiNERAssetError.contractMismatch("functions") }
        guard contract.caps == expected.caps else { throw GLiNERAssetError.contractMismatch("caps") }
        guard contract.host == expected.host else { throw GLiNERAssetError.contractMismatch("host") }
        let models = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "aimodel" }
        guard models.count == 1, let modelURL = models.first else {
            throw GLiNERAssetError.modelCount(models.count)
        }
        self.modelURL = modelURL
    }
}

struct Contract: Decodable, Equatable {
    struct Function: Decodable, Equatable {
        let inputs: [String]
        let outputs: [String]
    }

    struct Caps: Decodable, Equatable {
        let tokens: [Int]
        let queries: Int
        let classLabels: Int
        let candidates: Int
        let boundaries: String
        let words: String
    }

    struct Host: Decodable, Equatable {
        let poolBoundaryTopK: Int
        let poolSize: Int
        let minPoolPerQuery: Int
        let maskLogit: Float
        let quotaScoreBase: Float
        let pairTemperature: Float
        let classificationTemperature: Float
        let abstentionThreshold: Float
        let overlapPolicy: String
        let defaultThreshold: Float
        let tokenPooling: String
        let appendPeriodUnlessTextEndsWith: [String]
    }

    static let encodeInputs = [
        "input_ids", "attention_mask", "word_idx", "word_mask", "query_idx", "query_mask", "cls_idx", "cls_mask",
    ]

    static let pairEncodeInputs = [
        "scorer_start", "scorer_end", "content_prefix", "query_proj", "start_logits", "end_logits", "inside_prefix",
        "inside_mean",
    ]
    static let encodeOutputs = [
        "start_logits", "end_logits", "inside_prefix", "inside_mean", "null_logits", "cls_logits", "pool_start",
        "pool_end", "scorer_start", "scorer_end", "content_prefix", "query_proj",
    ]
    static let pairInputs =
        pairEncodeInputs + ["word_mask", "query_mask", "cand_start", "cand_end", "cand_mask", "cand_prior"]
    static let pairOutputs = ["pair_logits"]

    static let implemented = Contract(
        functions: Dictionary(
            uniqueKeysWithValues: PreparedInput.tokenCapacities.flatMap { capacity in
                [
                    ("encode_t\(capacity)", Function(inputs: encodeInputs, outputs: encodeOutputs)),
                    ("pair_t\(capacity)", Function(inputs: pairInputs, outputs: pairOutputs)),
                ]
            }),
        caps: Caps(
            tokens: PreparedInput.tokenCapacities, queries: PreparedInput.labelCapacity,
            classLabels: PreparedInput.labelCapacity, candidates: CandidatePool.size, boundaries: "T + 1", words: "T"),
        host: Host(
            poolBoundaryTopK: CandidatePool.boundaryLimit, poolSize: CandidatePool.size,
            minPoolPerQuery: CandidatePool.pairsPerLabel, maskLogit: CandidatePool.maskLogit,
            quotaScoreBase: CandidatePool.reservedScore, pairTemperature: 1, classificationTemperature: 1,
            abstentionThreshold: ExtractionResult.abstentionThreshold, overlapPolicy: "flat",
            defaultThreshold: ExtractionResult.threshold, tokenPooling: "first",
            appendPeriodUnlessTextEndsWith: [".", "!", "?"]))

    let functions: [String: Function]
    let caps: Caps
    let host: Host
}
